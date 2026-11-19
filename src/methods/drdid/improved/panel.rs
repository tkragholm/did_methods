//! Improved panel `DR-DiD`, following `DRDID::drdid_imp_panel`.

use crate::inference::{standard_error_from_influence, validate_confidence_level};
use crate::methods::drdid::design::{DEPENDENCE_TOLERANCE, prune_design};
use crate::methods::drdid::moments::normalize_weights_to_n;
use crate::types::{DrDidConfig, DrDidError, DrDidEstimate, DrDidObservation};
use crate::util::usize_to_f64;

use super::pscore::{calibrated_propensity_scores, linear_index, weighted_least_squares};

const TRIM_LEVEL: f64 = 0.995;
const SCORE_CAP: f64 = 1.0 - 1e-6;

/// Estimate the improved / locally efficient panel `DR-DiD` ATT.
///
/// The propensity score is the calibrated score of Sant'Anna and Zhao (2020).
/// The control outcome regression of the outcome change is weighted by
/// `w * ps / (1 - ps)`. Comparators with a score of 0.995 or more are trimmed.
/// `DrDidObservation::covariates` is the finished design, intercept first.
///
/// # Errors
///
/// Returns [`DrDidError`] when inputs or nuisance fits are invalid.
pub fn estimate_drdid_improved_panel(
    observations: &[DrDidObservation],
    config: DrDidConfig,
) -> Result<DrDidEstimate, DrDidError> {
    validate_config(config)?;
    if observations.is_empty() {
        return Err(DrDidError::EmptyInput);
    }
    let covariate_count = observations[0].covariates.len();
    let features = covariate_count.max(1);
    let n = observations.len();

    let mut treated = Vec::with_capacity(n);
    let mut delta = Vec::with_capacity(n);
    let mut sampling = Vec::with_capacity(n);
    let mut design = Vec::with_capacity(n * features);
    let mut total_weight = 0.0;
    for row in observations {
        if row.covariates.len() != covariate_count {
            return Err(DrDidError::InconsistentCovariateCount {
                expected: covariate_count,
                actual: row.covariates.len(),
            });
        }
        if !row.delta_outcome.is_finite() {
            return Err(DrDidError::InvalidOutcome {
                value: row.delta_outcome,
            });
        }
        if !row.weight.is_finite() || row.weight <= 0.0 {
            return Err(DrDidError::InvalidWeight { value: row.weight });
        }
        if let Some(bad) = row.covariates.iter().find(|c| !c.is_finite()) {
            return Err(DrDidError::InvalidCovariate { value: *bad });
        }
        treated.push(if row.treated { 1.0 } else { 0.0 });
        delta.push(row.delta_outcome);
        sampling.push(row.weight);
        total_weight += row.weight;
        if covariate_count == 0 {
            design.push(1.0);
        } else {
            design.extend_from_slice(&row.covariates);
        }
    }
    let treated_n = observations.iter().filter(|o| o.treated).count();
    let control_n = n - treated_n;
    if treated_n == 0 {
        return Err(DrDidError::NoTreated);
    }
    if control_n == 0 {
        return Err(DrDidError::NoControl);
    }

    let arms: Vec<usize> = treated.iter().map(|&t| usize::from(t > 0.5)).collect();
    let pruned = prune_design(&design, features, &arms, &sampling, DEPENDENCE_TOLERANCE)?;
    let p = pruned.feature_count;
    let design = pruned.design_flat;

    let weights = normalize_weights_to_n(&sampling)
        .map_err(|_| DrDidError::InvalidConfig("sum of weights must be > 0".to_string()))?;
    let scores: Vec<f64> = calibrated_propensity_scores(&design, p, &treated, &weights)?
        .scores
        .into_iter()
        .map(|s| s.min(SCORE_CAP))
        .collect();

    let odds_weights: Vec<f64> = weights
        .iter()
        .zip(&scores)
        .map(|(w, s)| w * s / (1.0 - s))
        .collect();
    let beta = weighted_least_squares(&design, p, &delta, &odds_weights, |i| treated[i] < 0.5)?;
    let fitted = linear_index(&design, p, &beta);

    let n_f = usize_to_f64(n);
    let mean_treated_weight = treated
        .iter()
        .zip(&weights)
        .map(|(d, w)| d * w)
        .sum::<f64>()
        / n_f;
    let trim: Vec<f64> = (0..n)
        .map(|i| {
            if treated[i] < 0.5 && scores[i] >= TRIM_LEVEL {
                0.0
            } else {
                1.0
            }
        })
        .collect();
    let summand: Vec<f64> = (0..n)
        .map(|i| trim[i] * (1.0 - (1.0 - treated[i]) / (1.0 - scores[i])) * (delta[i] - fitted[i]))
        .collect();
    let att = summand
        .iter()
        .zip(&weights)
        .map(|(s, w)| w * s)
        .sum::<f64>()
        / n_f
        / mean_treated_weight;
    let influence_function: Vec<f64> = (0..n)
        .map(|i| weights[i] * trim[i] * (summand[i] - treated[i] * att) / mean_treated_weight)
        .collect();

    let se = standard_error_from_influence(&influence_function);
    let (ci_low, ci_high) = (att - 1.96 * se, att + 1.96 * se);
    Ok(DrDidEstimate {
        att,
        se,
        ci_low,
        ci_high,
        treated_n,
        control_n,
        total_weight,
        influence_function,
        design_columns_dropped: pruned.dropped,
    })
}

fn validate_config(config: DrDidConfig) -> Result<(), DrDidError> {
    if !validate_confidence_level(config.confidence_level) {
        return Err(DrDidError::InvalidConfig(
            "confidence_level must be finite and in (0, 1)".to_string(),
        ));
    }
    if !config.propensity_clip.is_finite()
        || config.propensity_clip <= 0.0
        || config.propensity_clip >= 0.5
    {
        return Err(DrDidError::InvalidConfig(
            "propensity_clip must be finite and in (0, 0.5)".to_string(),
        ));
    }
    if !config.ridge.is_finite() || config.ridge < 0.0 {
        return Err(DrDidError::InvalidConfig(
            "ridge must be finite and non-negative".to_string(),
        ));
    }
    if config.max_iter == 0 {
        return Err(DrDidError::InvalidConfig(
            "max_iter must be > 0".to_string(),
        ));
    }
    if !config.tol.is_finite() || config.tol <= 0.0 {
        return Err(DrDidError::InvalidConfig(
            "tol must be finite and > 0".to_string(),
        ));
    }
    if config.bootstrap_reps == 0 {
        return Err(DrDidError::InvalidConfig(
            "bootstrap_reps must be > 0".to_string(),
        ));
    }
    Ok(())
}
