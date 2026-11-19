//! Nuisance estimation for improved repeated-cross-section `DR-DiD`.
//!
//! The improved repeated estimator requires two nuisance components:
//!
//! 1. a calibrated propensity score for treatment assignment, and
//! 2. group-period outcome regressions for the four `(G, T)` cells.
//!
//! The implementation here follows the structure of `DRDID::drdid_imp_rc`:
//!
//! - start from a weighted logit fit for the propensity score,
//! - refine it with the calibrated score equations used by the improved
//!   estimator,
//! - perform the inverse-probability tilting refinement,
//! - fit weighted linear outcome regressions for treated/control and pre/post
//!   cells,
//! - trim control observations with extreme fitted scores.
//!
//! The resulting nuisance estimates are then fed into the final ATT assembly in
//! [`super::estimate`].
//!
//! References:
//! - Sant'Anna, P. H. C. and Zhao, J. (2020). "Doubly Robust Difference-in-
//!   Differences Estimators". *Journal of Econometrics*.
//! - `DRDID` R package, `drdid_imp_rc` implementation.

use crate::types::{DidCell, DrDidConfig, DrDidError, TimePeriod, TreatmentGroup};

use super::super::super::moments::normalize_weights_to_n as normalize_weights_to_n_shared;
use super::super::pscore::{calibrated_propensity_scores, linear_index, weighted_least_squares};
use super::data::RepeatedPreparedData;

pub(super) const IMPROVED_TRIM_LEVEL: f64 = 0.995;
const SCORE_CAP: f64 = 1.0 - 1e-6;

pub(super) struct ImprovedRepeatedNuisanceFits {
    pub(super) normalized_weights: Vec<f64>,
    pub(super) propensity_scores: Vec<f64>,
    pub(super) trim_indicator: Vec<f64>,
    pub(super) out_y_cont_pre: Vec<f64>,
    pub(super) out_y_cont_post: Vec<f64>,
    pub(super) out_y_treat_pre: Vec<f64>,
    pub(super) out_y_treat_post: Vec<f64>,
}

pub(super) fn fit_improved_repeated_nuisance_models(
    prepared: &RepeatedPreparedData,
    _config: DrDidConfig,
) -> Result<ImprovedRepeatedNuisanceFits, DrDidError> {
    let normalized_weights = normalize_weights_to_n(&prepared.sampling_weights)?;
    let propensity_scores = calibrated_propensity_scores(
        &prepared.design_matrix_flat,
        prepared.feature_count,
        &prepared.treated_indicator,
        &normalized_weights,
    )?
    .scores
    .into_iter()
    .map(|score| score.min(SCORE_CAP))
    .collect::<Vec<_>>();

    let trim_indicator = prepared
        .treated_indicator
        .iter()
        .zip(propensity_scores.iter())
        .map(|(treated, propensity_score)| {
            if *treated < 0.5 && *propensity_score >= IMPROVED_TRIM_LEVEL {
                0.0
            } else {
                1.0
            }
        })
        .collect::<Vec<_>>();

    let out_y_cont_pre = fit_group_period_outcome(
        prepared,
        &propensity_scores,
        &normalized_weights,
        false,
        false,
        0.0,
    )?;
    let out_y_cont_post = fit_group_period_outcome(
        prepared,
        &propensity_scores,
        &normalized_weights,
        false,
        true,
        0.0,
    )?;
    let out_y_treat_pre = fit_group_period_outcome(
        prepared,
        &propensity_scores,
        &normalized_weights,
        true,
        false,
        0.0,
    )?;
    let out_y_treat_post = fit_group_period_outcome(
        prepared,
        &propensity_scores,
        &normalized_weights,
        true,
        true,
        0.0,
    )?;

    Ok(ImprovedRepeatedNuisanceFits {
        normalized_weights,
        propensity_scores,
        trim_indicator,
        out_y_cont_pre,
        out_y_cont_post,
        out_y_treat_pre,
        out_y_treat_post,
    })
}

pub(super) fn normalize_weights_to_n(weights: &[f64]) -> Result<Vec<f64>, DrDidError> {
    normalize_weights_to_n_shared(weights)
        .map_err(|_| DrDidError::InvalidConfig("sum of weights must be > 0".to_string()))
}

impl ImprovedRepeatedNuisanceFits {
    pub(super) fn post_indicator_mix(&self, post_indicator: &[f64]) -> Vec<f64> {
        post_indicator
            .iter()
            .enumerate()
            .map(|(row_index, post)| {
                if *post > 0.5 {
                    self.out_y_cont_post[row_index]
                } else {
                    self.out_y_cont_pre[row_index]
                }
            })
            .collect()
    }
}

fn fit_group_period_outcome(
    prepared: &RepeatedPreparedData,
    propensity_scores: &[f64],
    normalized_weights: &[f64],
    treated_group: bool,
    post_period: bool,
    _ridge: f64,
) -> Result<Vec<f64>, DrDidError> {
    let treated_value = if treated_group { 1.0 } else { 0.0 };
    let post_value = if post_period { 1.0 } else { 0.0 };
    let row_indices = prepared
        .treated_indicator
        .iter()
        .zip(prepared.post_indicator.iter())
        .enumerate()
        .filter_map(|(row_index, (treated, post))| {
            if (*treated - treated_value).abs() < f64::EPSILON
                && (*post - post_value).abs() < f64::EPSILON
            {
                Some(row_index)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    if row_indices.is_empty() {
        return Err(DrDidError::MissingCell {
            cell: DidCell::from_parts(
                TreatmentGroup::from_bool(treated_group),
                TimePeriod::from_bool(post_period),
            ),
        });
    }

    let weights = (0..prepared.treated_indicator.len())
        .map(|row_index| {
            if treated_group {
                normalized_weights[row_index]
            } else {
                normalized_weights[row_index] * propensity_scores[row_index]
                    / (1.0 - propensity_scores[row_index])
            }
        })
        .collect::<Vec<_>>();
    let mut in_cell = vec![false; prepared.treated_indicator.len()];
    for row_index in &row_indices {
        in_cell[*row_index] = true;
    }
    let beta = weighted_least_squares(
        &prepared.design_matrix_flat,
        prepared.feature_count,
        &prepared.outcome,
        &weights,
        |row_index| in_cell[row_index],
    )?;
    Ok(linear_index(
        &prepared.design_matrix_flat,
        prepared.feature_count,
        &beta,
    ))
}
