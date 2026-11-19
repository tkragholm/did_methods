use epistat_inference::Clusters;
use epistat_inference::simultaneous::multiplier_bands;
use epistat_rng::{Stream, replicate_seed};

use crate::inference::{
    quantile_unsorted, standard_error_from_influence, validate_confidence_level,
};
use crate::types::{AttGtBandConfig, AttGtBandError, AttGtBandEstimate, AttGtEstimate};

/// Compute simultaneous confidence bands for ATT(g,t) estimates from aligned
/// influence functions via multiplier bootstrap.
///
/// `influence_functions[k]` must correspond to `estimates[k]`. All vectors must
/// have identical length and represent contributions on the same sample index.
///
/// # Errors
///
/// Returns [`AttGtBandError`] when inputs/configuration are invalid or influence
/// vectors are malformed/degenerate for bootstrap inference.
pub fn att_gt_simultaneous_bands_with_influence(
    estimates: &[AttGtEstimate],
    influence_functions: &[Vec<f64>],
    config: AttGtBandConfig,
) -> Result<Vec<AttGtBandEstimate>, AttGtBandError> {
    validate_band_inputs(estimates, config)?;
    if influence_functions.len() != estimates.len() {
        return Err(AttGtBandError::InfluenceCountMismatch);
    }
    if influence_functions.is_empty() {
        return Err(AttGtBandError::EmptyInput);
    }
    let n = influence_functions[0].len();
    if n == 0 {
        return Err(AttGtBandError::EmptyInfluence);
    }
    for influence in influence_functions {
        if influence.len() != n {
            return Err(AttGtBandError::InconsistentInfluenceLength);
        }
        if influence.iter().any(|value| !value.is_finite()) {
            return Err(AttGtBandError::InvalidInfluence);
        }
    }

    let se_if = influence_functions
        .iter()
        .map(|influence| {
            let se = standard_error_from_influence(influence);
            if !se.is_finite() || se <= 0.0 {
                return Err(AttGtBandError::DegenerateInfluence);
            }
            Ok(se)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut maxima = Vec::with_capacity(config.reps);
    let mut signs = vec![1.0; n];
    for replicate in 0..config.reps as u64 {
        let mut stream = Stream::seeded(replicate_seed(config.seed, replicate));
        stream.fill_signs(&mut signs);
        let mut max_abs = 0.0_f64;
        for influence in influence_functions {
            let numerator = influence
                .iter()
                .zip(signs.iter())
                .map(|(value, sign)| value * sign)
                .sum::<f64>();
            let denominator = influence
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if denominator <= 0.0 || !denominator.is_finite() {
                return Err(AttGtBandError::DegenerateInfluence);
            }
            let t_stat = numerator / denominator;
            max_abs = max_abs.max(t_stat.abs());
        }
        maxima.push(max_abs);
    }

    let alpha = (1.0 - config.confidence_level.confidence_level).clamp(0.0, 1.0);
    let critical = quantile_unsorted(&mut maxima, 1.0 - alpha);

    Ok(estimates
        .iter()
        .zip(se_if.iter())
        .map(|(estimate, se_if)| {
            let margin = critical * *se_if;
            AttGtBandEstimate {
                group: estimate.group,
                time: estimate.time,
                event_time: estimate.event_time,
                att: estimate.att,
                se: *se_if,
                band_low: estimate.att - margin,
                band_high: estimate.att + margin,
            }
        })
        .collect())
}

/// Compute simultaneous confidence bands for ATT(g,t) estimates.
///
/// This is a Gaussian-max approximation fallback when aligned influence
/// functions are unavailable.
///
/// # Errors
///
/// Returns [`AttGtBandError`] when inputs/configuration are invalid.
pub fn att_gt_simultaneous_bands(
    estimates: &[AttGtEstimate],
    config: AttGtBandConfig,
) -> Result<Vec<AttGtBandEstimate>, AttGtBandError> {
    validate_band_inputs(estimates, config)?;

    let mut rng = Stream::seeded(config.seed);
    let mut maxima = Vec::with_capacity(config.reps);
    let k = estimates.len();
    for _ in 0..config.reps {
        let mut max_abs = 0.0_f64;
        for _ in 0..k {
            max_abs = max_abs.max(rng.normal().abs());
        }
        maxima.push(max_abs);
    }
    let alpha = (1.0 - config.confidence_level.confidence_level).clamp(0.0, 1.0);
    let critical = quantile_unsorted(&mut maxima, 1.0 - alpha);

    Ok(estimates
        .iter()
        .map(|estimate| {
            let margin = critical * estimate.se;
            AttGtBandEstimate {
                group: estimate.group,
                time: estimate.time,
                event_time: estimate.event_time,
                att: estimate.att,
                se: estimate.se,
                band_low: estimate.att - margin,
                band_high: estimate.att + margin,
            }
        })
        .collect())
}

fn validate_band_inputs(
    estimates: &[AttGtEstimate],
    config: AttGtBandConfig,
) -> Result<(), AttGtBandError> {
    if estimates.is_empty() {
        return Err(AttGtBandError::EmptyInput);
    }
    if !validate_confidence_level(config.confidence_level.confidence_level) {
        return Err(AttGtBandError::InvalidConfidenceLevel);
    }
    if config.reps == 0 {
        return Err(AttGtBandError::InvalidReps);
    }
    for estimate in estimates {
        if !estimate.att.is_finite() {
            return Err(AttGtBandError::InvalidEstimate);
        }
        if !estimate.se.is_finite() || estimate.se < 0.0 {
            return Err(AttGtBandError::InvalidSe);
        }
    }
    Ok(())
}

/// Simultaneous bands by the multiplier bootstrap, as `did:::mboot` does it.
///
/// # How this differs from [`att_gt_simultaneous_bands_with_influence`]
///
/// That function draws Rademacher signs and scales by the analytic influence
/// standard error. `did` does neither:
///
/// * The scale of each estimate is the interquartile range of its bootstrap
///   draws divided by `qnorm(0.75) - qnorm(0.25)`, so a few extreme draws do not
///   move it.
/// * The critical value is the type 1 quantile, the inverse empirical CDF, of the
///   maximum absolute t statistic over all cells.
/// * One weight is drawn per cluster, or per unit without clustering, and applied
///   to the cluster's summed influence. Clusters are numbered in increasing label
///   order, as `did` numbers them.
///
/// The bands are `att +- crit * se`, with `se = sigma * sqrt(G) / n` for `G`
/// clusters and `n` units. An estimate whose draws are degenerate takes no part
/// in the maximum, as in `did`, and has `NaN` for its standard error and band.
///
/// # Errors
///
/// Returns [`AttGtBandError`] when inputs are malformed or every influence vector
/// is degenerate.
pub fn att_gt_mboot_bands(
    estimates: &[AttGtEstimate],
    influence_functions: &[Vec<f64>],
    clusters: Option<&[i64]>,
    config: AttGtBandConfig,
) -> Result<Vec<AttGtBandEstimate>, AttGtBandError> {
    validate_band_inputs(estimates, config)?;
    if influence_functions.len() != estimates.len() {
        return Err(AttGtBandError::InfluenceCountMismatch);
    }
    if influence_functions.is_empty() {
        return Err(AttGtBandError::EmptyInput);
    }
    let n = influence_functions[0].len();
    if n == 0 {
        return Err(AttGtBandError::EmptyInfluence);
    }
    for influence in influence_functions {
        if influence.len() != n {
            return Err(AttGtBandError::InconsistentInfluenceLength);
        }
        if influence.iter().any(|value| !value.is_finite()) {
            return Err(AttGtBandError::InvalidInfluence);
        }
    }
    let cluster_numbers = match clusters {
        None => None,
        Some(labels) if labels.len() == n => Some(Clusters::from_sorted_labels(labels)),
        Some(_) => return Err(AttGtBandError::InconsistentInfluenceLength),
    };

    // The weights are Rademacher, as in BMisc::multiplier_bootstrap, which is the
    // kernel did calls, and not the Mammen weights its documentation names.
    let alpha = (1.0 - config.confidence_level.confidence_level).clamp(0.0, 1.0);
    let columns: Vec<&[f64]> = influence_functions.iter().map(Vec::as_slice).collect();
    let bands = multiplier_bands(
        &columns,
        cluster_numbers.as_ref(),
        alpha,
        config.reps,
        config.seed,
    )
    .map_err(|_| AttGtBandError::DegenerateInfluence)?;
    debug_assert_eq!(bands.se.len(), estimates.len());

    Ok(estimates
        .iter()
        .zip(&bands.se)
        .map(|(estimate, se)| {
            let se = se.unwrap_or(f64::NAN);
            let margin = bands.critical * se;
            AttGtBandEstimate {
                group: estimate.group,
                time: estimate.time,
                event_time: estimate.event_time,
                att: estimate.att,
                se,
                band_low: estimate.att - margin,
                band_high: estimate.att + margin,
            }
        })
        .collect())
}
