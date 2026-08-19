use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::inference::{
    quantile_unsorted, standard_error_from_influence, validate_confidence_level,
};
use crate::types::{AttGtBandConfig, AttGtBandError, AttGtBandEstimate, AttGtEstimate};
use crate::util::usize_to_f64;

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

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut maxima = Vec::with_capacity(config.reps);
    let mut signs = vec![1.0; n];
    for _ in 0..config.reps {
        for sign in &mut signs {
            *sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        }
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

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut maxima = Vec::with_capacity(config.reps);
    let k = estimates.len();
    for _ in 0..config.reps {
        let mut max_abs = 0.0_f64;
        for _ in 0..k {
            // Box-Muller normal draw.
            let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
            let u2 = rng.random::<f64>();
            let z = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
            max_abs = max_abs.max(z.abs());
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
/// standard error. `did` does neither, and the differences are not cosmetic:
///
/// * **An interquartile-range scale** estimated from the bootstrap draws
///   themselves, rather than the analytic standard error. This is the deliberate
///   robustness choice in `did`: a few extreme draws move a sample standard
///   deviation and do not move an IQR.
/// * **Type-1 quantiles**, the inverse empirical CDF, with no interpolation.
///
/// Use this when the number has to be the same one R would report. The bands are
/// `att +- crit * sigma / sqrt(n)`, and `crit` is the `1 - alpha` quantile of the
/// bootstrap maximum absolute t statistic across all cells at once, which is what
/// makes them simultaneous rather than pointwise.
///
/// # Errors
///
/// Returns [`AttGtBandError`] when inputs are malformed or an influence vector is
/// degenerate.
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

    // RADEMACHER weights, +-1 with equal probability, and this is worth stating
    // because the literature and did's own documentation both say Mammen.
    // BMisc::multiplier_bootstrap, which is the C++ kernel did actually calls,
    // does not use Mammen. Probed directly with a 1x1 influence matrix of ones
    // over 200,000 draws, it returns exactly two values, -1 and 1, at 0.4994 and
    // 0.5006. Implementing the documented Mammen weights instead puts the
    // simultaneous critical value at 2.72 where did reports 2.62, which is
    // twenty Monte Carlo standard deviations away and would look like a bug in
    // whichever side was checked second.
    // With clustering, the bootstrap resamples CLUSTERS, not units: one weight is
    // drawn per cluster and applied to that cluster's summed influence. Drawing
    // per unit would treat two parents of the same child as independent draws and
    // shrink the standard error by roughly the square root of the cluster size.
    //
    // `did` does the same thing, and its normalisation is worth stating because
    // the two denominators differ: the bootstrap is over `n_clusters`, so the
    // draws carry sqrt(n_clusters), but the reported standard error is
    // `bSigma * sqrt(n_clusters) / n` with `n` the UNIT count. Both appear below.
    let (assignments, n_clusters) = match clusters {
        None => ((0..n).collect::<Vec<usize>>(), n),
        Some(labels) => {
            if labels.len() != n {
                return Err(AttGtBandError::InconsistentInfluenceLength);
            }
            let mut distinct = labels.to_vec();
            distinct.sort_unstable();
            distinct.dedup();
            let position = distinct
                .iter()
                .enumerate()
                .map(|(index, label)| (*label, index))
                .collect::<std::collections::BTreeMap<i64, usize>>();
            let assignments = labels
                .iter()
                .map(|label| position[label])
                .collect::<Vec<usize>>();
            (assignments, distinct.len())
        }
    };

    // Influence summed within cluster, did's `cluster_sum_if`. With no clustering
    // this is the influence matrix itself.
    let clustered = influence_functions
        .iter()
        .map(|influence| {
            let mut sums = vec![0.0_f64; n_clusters];
            for (value, &cluster) in influence.iter().zip(&assignments) {
                sums[cluster] += value;
            }
            sums
        })
        .collect::<Vec<Vec<f64>>>();

    let n_f = usize_to_f64(n);
    let clusters_f = usize_to_f64(n_clusters);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut draws = vec![vec![0.0_f64; config.reps]; influence_functions.len()];
    let mut weights = vec![0.0_f64; n_clusters];

    // `rep` indexes a column of `draws`, which is statistic-major so that the
    // per-statistic quantiles below are over contiguous memory. The lint reads
    // this as a needless range loop; the loop variable is a write target, not an
    // iteration over `draws`.
    #[expect(clippy::needless_range_loop, reason = "rep indexes a second structure")]
    for rep in 0..config.reps {
        for weight in &mut weights {
            *weight = if rng.random_bool(0.5) { -1.0 } else { 1.0 };
        }
        for (statistic, influence) in clustered.iter().enumerate() {
            // sqrt(n_clusters) * mean over clusters of (V * psi), did's `Ub`.
            let mean = influence
                .iter()
                .zip(&weights)
                .map(|(value, weight)| value * weight)
                .sum::<f64>()
                / clusters_f;
            draws[statistic][rep] = clusters_f.sqrt() * mean;
        }
    }

    // The robust scale, one per statistic, from the bootstrap draws.
    //
    // On a COPY of each column. `quantile_type1` sorts in place, and sorting the
    // stored draws would silently destroy the pairing across statistics within a
    // bootstrap replication: every column would become monotone, the maximum
    // below would then be taken over co-monotone values, and the critical value
    // would collapse toward the pointwise one. Measured, that bug produced 2.02
    // where did reports 2.62.
    let sigma = draws
        .iter()
        .map(|column| {
            let mut sorted = column.clone();
            let spread = quantile_type1(&mut sorted, 0.75) - quantile_type1(&mut sorted, 0.25);
            let scale = spread / NORMAL_IQR;
            if !scale.is_finite() || scale <= 0.0 {
                return Err(AttGtBandError::DegenerateInfluence);
            }
            Ok(scale)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut maxima = (0..config.reps)
        .map(|rep| {
            draws
                .iter()
                .zip(&sigma)
                .map(|(column, scale)| (column[rep] / scale).abs())
                .fold(0.0_f64, f64::max)
        })
        .collect::<Vec<_>>();

    let alpha = (1.0 - config.confidence_level.confidence_level).clamp(0.0, 1.0);
    let critical = quantile_type1(&mut maxima, 1.0 - alpha);

    Ok(estimates
        .iter()
        .zip(&sigma)
        .map(|(estimate, scale)| {
            // bSigma * sqrt(n_clusters) / n, which is bSigma / sqrt(n) when every
            // unit is its own cluster.
            let se = scale * clusters_f.sqrt() / n_f;
            let margin = critical * se;
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

/// `qnorm(0.75) - qnorm(0.25)`, the interquartile range of a standard normal.
///
/// Written as a literal rather than computed so the value cannot drift with
/// whichever inverse-normal routine happens to be linked.
const NORMAL_IQR: f64 = 1.348_979_500_392_162_4;

/// R's `quantile(x, p, type = 1)`: the inverse empirical CDF, no interpolation.
///
/// `did` asks for `type = 1` explicitly and the choice is visible in the result,
/// so the default type 7 used elsewhere in this crate is not interchangeable
/// here.
fn quantile_type1(values: &mut [f64], probability: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let n = usize_to_f64(values.len());
    let position = n * probability;
    let floor = position.floor();
    let index = if (position - floor) > 0.0 {
        floor as usize
    } else {
        (floor as usize).saturating_sub(1)
    };
    values[index.min(values.len() - 1)]
}
