//! Test-oriented numerical helpers shared across crate tests.
//!
//! These are compiled only under `cfg(test)` via `common::mod`.

use crate::util::usize_to_f64;
use faer::Mat;
use tracing::warn;

/// Compute `X' W X` given a precomputed `X' W`.
#[must_use]
pub fn xt_w_x(x_t_w: &Mat<f64>, x: &Mat<f64>) -> Mat<f64> {
    x_t_w * x
}

/// Compute `X' W b` given a precomputed `X' W`.
#[must_use]
pub fn xt_w_b(x_t_w: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    x_t_w * b
}

/// Clamp propensity scores into `[epsilon, 1 - epsilon]`.
#[must_use]
pub fn trim(propensity_scores: &[f64], epsilon: f64) -> Vec<f64> {
    propensity_scores
        .iter()
        .map(|&score| score.clamp(epsilon, 1.0 - epsilon))
        .collect()
}

/// Rescale weights to unit mean when possible.
///
/// If the mean is zero or non-finite, returns the original weights and emits a
/// warning.
#[must_use]
pub fn stabilize(weights: &[f64]) -> Vec<f64> {
    if weights.is_empty() {
        return Vec::new();
    }
    let mean_weight = weights.iter().sum::<f64>() / usize_to_f64(weights.len());
    if !mean_weight.is_finite() || mean_weight.abs() < f64::EPSILON {
        warn!("stabilize: mean weight is non-finite or zero; returning unscaled weights");
        return weights.to_vec();
    }
    weights.iter().map(|weight| weight / mean_weight).collect()
}

/// Compute the Horvitz-Thompson style contrast using binary treatment weights.
///
/// # Returns
/// Returns a finite estimate on valid finite inputs.
///
/// Propensity scores are internally clamped to `[1e-6, 1 - 1e-6]`.
///
/// # Errors
/// Returns an error on length mismatch, empty inputs, or non-finite values.
pub fn horvitz_thompson(
    outcome: &[f64],
    treated: &[f64],
    propensity_scores: &[f64],
) -> Result<f64, &'static str> {
    let observation_count = outcome.len();
    if observation_count == 0
        || treated.len() != observation_count
        || propensity_scores.len() != observation_count
    {
        warn!(
            "horvitz_thompson: length mismatch or empty inputs (outcome={}, treated={}, propensity_scores={})",
            outcome.len(),
            treated.len(),
            propensity_scores.len()
        );
        return Err("length mismatch or empty inputs");
    }

    let mut sum = 0.0;
    let epsilon = 1e-6;
    for row_index in 0..observation_count {
        let outcome_value = outcome[row_index];
        let treated_value = treated[row_index];
        let mut propensity_score = propensity_scores[row_index];
        if !outcome_value.is_finite() || !treated_value.is_finite() || !propensity_score.is_finite()
        {
            warn!("horvitz_thompson: non-finite input at row {row_index}; returning error");
            return Err("non-finite input");
        }

        propensity_score = propensity_score.clamp(epsilon, 1.0 - epsilon);
        sum += treated_value * outcome_value / propensity_score
            - (1.0 - treated_value) * outcome_value / (1.0 - propensity_score);
    }

    Ok(sum / usize_to_f64(observation_count))
}

/// Normalize a one-column weight matrix to have mean one.
///
/// For zero rows, returns an empty vector. If the column sum is effectively
/// zero, returns all-ones weights.
#[must_use]
pub fn normalize_weights(weights: &Mat<f64>) -> Vec<f64> {
    let observation_count = weights.nrows();
    if observation_count == 0 {
        return Vec::new();
    }

    let total_weight = weights.col_as_slice(0).iter().sum::<f64>();
    if total_weight.abs() < f64::EPSILON {
        return vec![1.0; observation_count];
    }

    let mean_weight = total_weight / usize_to_f64(observation_count);
    weights
        .col_as_slice(0)
        .iter()
        .map(|weight| weight / mean_weight)
        .collect()
}
