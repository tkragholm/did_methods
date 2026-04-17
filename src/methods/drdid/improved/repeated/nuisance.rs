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

use faer::Mat;

use crate::estimators::common::linalg::solve_dense_system;
use crate::estimators::outcome::linear::LinearOutcome;
use crate::estimators::outcome::model::OutcomeModel;
use crate::estimators::propensity::common::logistic_scores;
use crate::estimators::propensity::logistic::LogisticPS;
use crate::estimators::propensity::types::{Config as PropensityConfig, PropensityEstimator};
use crate::types::{DidCell, DrDidConfig, DrDidError, TimePeriod, TreatmentGroup};

use super::super::super::moments::normalize_weights_to_n as normalize_weights_to_n_shared;
use super::data::RepeatedPreparedData;

pub(super) const IMPROVED_TRIM_LEVEL: f64 = 0.995;

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
    config: DrDidConfig,
) -> Result<ImprovedRepeatedNuisanceFits, DrDidError> {
    let normalized_weights = normalize_weights_to_n(&prepared.sampling_weights)?;
    let propensity_scores = fit_weighted_calibrated_propensity_scores(
        &prepared.design_matrix_flat,
        prepared.feature_count,
        &prepared.treated_indicator,
        &normalized_weights,
        config,
    )?;

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
    ridge: f64,
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

    let weights = row_indices
        .iter()
        .map(|row_index| {
            if treated_group {
                normalized_weights[*row_index]
            } else {
                normalized_weights[*row_index] * propensity_scores[*row_index]
                    / (1.0 - propensity_scores[*row_index])
            }
        })
        .collect::<Vec<_>>();

    let design = Mat::from_fn(row_indices.len(), prepared.feature_count, |row, col| {
        prepared.design_matrix_flat[row_indices[row] * prepared.feature_count + col]
    });
    let outcome = Mat::from_fn(row_indices.len(), 1, |row, _| {
        prepared.outcome[row_indices[row]]
    });
    let model = LinearOutcome { ridge };
    let beta = model.fit(design.as_ref(), outcome.as_ref(), Some(&weights));
    let full_design = Mat::from_fn(
        prepared.treated_indicator.len(),
        prepared.feature_count,
        |row, col| prepared.design_matrix_flat[row * prepared.feature_count + col],
    );
    Ok(model.predict(full_design.as_ref(), beta.as_ref()))
}

fn fit_weighted_calibrated_propensity_scores(
    design_matrix_flat: &[f64],
    feature_count: usize,
    treated_indicator: &[f64],
    normalized_weights: &[f64],
    config: DrDidConfig,
) -> Result<Vec<f64>, DrDidError> {
    if weights_are_uniform(normalized_weights) {
        let observation_count = treated_indicator.len();
        let design_matrix = Mat::from_fn(observation_count, feature_count, |row, col| {
            design_matrix_flat[row * feature_count + col]
        });
        let treated_target = Mat::from_fn(observation_count, 1, |row, _| treated_indicator[row]);
        let propensity_cfg = PropensityConfig {
            max_iter: u64::try_from(config.max_iter)
                .map_err(|_| DrDidError::InvalidConfig("max_iter exceeds u64".to_string()))?,
            tol: config.tol,
            min_weight: config.propensity_clip,
            vstar: 700.0,
        };
        let propensity = LogisticPS::new(propensity_cfg);
        let params = propensity
            .fit(design_matrix.as_ref(), treated_target.as_ref())
            .map_err(|err| DrDidError::InvalidConfig(err.to_string()))?;
        return Ok(
            logistic_scores(design_matrix.as_ref(), params.beta.as_ref())
                .into_iter()
                .map(|score| score.clamp(config.propensity_clip, 1.0 - config.propensity_clip))
                .collect(),
        );
    }

    let coefficients = fit_weighted_calibrated_propensity(
        design_matrix_flat,
        feature_count,
        treated_indicator,
        normalized_weights,
        config,
    )?;
    Ok(predict_logistic_scores(
        design_matrix_flat,
        feature_count,
        &coefficients,
        config.propensity_clip,
    ))
}

fn weights_are_uniform(normalized_weights: &[f64]) -> bool {
    let Some((&first, rest)) = normalized_weights.split_first() else {
        return true;
    };
    rest.iter().all(|weight| (*weight - first).abs() <= 1e-12)
}

fn fit_weighted_calibrated_propensity(
    design_matrix_flat: &[f64],
    feature_count: usize,
    treated_indicator: &[f64],
    normalized_weights: &[f64],
    config: DrDidConfig,
) -> Result<Vec<f64>, DrDidError> {
    let observation_count = treated_indicator.len();
    let design_matrix = Mat::from_fn(observation_count, feature_count, |row, col| {
        design_matrix_flat[row * feature_count + col]
    });
    let mut coefficients = fit_weighted_logit_initial(
        design_matrix_flat,
        feature_count,
        treated_indicator,
        normalized_weights,
        config.propensity_clip,
        u64::try_from(config.max_iter.max(100)).unwrap_or(u64::MAX),
        config.tol,
    );

    let mut trust_candidate = vec![0.0; feature_count];
    let mut trust_values = vec![0.0; observation_count];
    let current_loss = weighted_calibrated_ps_loss(
        &coefficients,
        design_matrix_flat,
        feature_count,
        treated_indicator,
        normalized_weights,
        &mut trust_values,
    );

    let calibration_problem = WeightedCalibrationProblem {
        design_matrix: &design_matrix,
        design_matrix_flat,
        feature_count,
        treated_indicator,
        normalized_weights,
        config,
    };

    coefficients = run_weighted_calibration_newton(
        coefficients,
        &mut trust_candidate,
        &mut trust_values,
        current_loss,
        &calibration_problem,
    )?;

    run_weighted_ipt_refinement(
        coefficients,
        &design_matrix,
        design_matrix_flat,
        feature_count,
        treated_indicator,
        normalized_weights,
        config,
    )
}

struct WeightedCalibrationProblem<'a> {
    design_matrix: &'a Mat<f64>,
    design_matrix_flat: &'a [f64],
    feature_count: usize,
    treated_indicator: &'a [f64],
    normalized_weights: &'a [f64],
    config: DrDidConfig,
}

fn stepped_candidate(current: &[f64], step: &[f64], step_scale: f64, out: &mut [f64]) {
    for ((candidate, current_value), step_value) in
        out.iter_mut().zip(current.iter()).zip(step.iter())
    {
        *candidate = current_value - step_scale * step_value;
    }
}

fn run_weighted_calibration_newton(
    mut coefficients: Vec<f64>,
    trust_candidate: &mut [f64],
    trust_values: &mut [f64],
    mut current_loss: f64,
    problem: &WeightedCalibrationProblem<'_>,
) -> Result<Vec<f64>, DrDidError> {
    for _ in 0..problem.config.max_iter.max(1_000) {
        let (gradient, hessian) = weighted_calibrated_ps_grad_hess(
            problem.design_matrix,
            problem.feature_count,
            problem.treated_indicator,
            problem.normalized_weights,
            trust_values,
        );
        let step =
            solve_dense_system(&hessian, &gradient).map_err(|_| DrDidError::SingularSystem)?;
        let max_step = step.iter().map(|value| value.abs()).fold(0.0, f64::max);
        if max_step <= problem.config.tol.min(1e-12) {
            return Ok(coefficients);
        }

        let mut step_scale = 1.0;
        let mut accepted = false;
        for _ in 0..11 {
            stepped_candidate(&coefficients, &step, step_scale, trust_candidate);
            let candidate_loss = weighted_calibrated_ps_loss(
                trust_candidate,
                problem.design_matrix_flat,
                problem.feature_count,
                problem.treated_indicator,
                problem.normalized_weights,
                trust_values,
            );
            if candidate_loss <= current_loss {
                coefficients.clone_from_slice(trust_candidate);
                current_loss = candidate_loss;
                accepted = true;
                break;
            }
            step_scale *= 0.5;
            if step_scale < 1.0 / 1024.0 {
                break;
            }
        }

        if !accepted {
            break;
        }
    }

    Ok(coefficients)
}

fn run_weighted_ipt_refinement(
    mut coefficients: Vec<f64>,
    design_matrix: &Mat<f64>,
    design_matrix_flat: &[f64],
    feature_count: usize,
    treated_indicator: &[f64],
    normalized_weights: &[f64],
    config: DrDidConfig,
) -> Result<Vec<f64>, DrDidError> {
    let observation_count = treated_indicator.len();
    let mut ipt_candidate = vec![0.0; feature_count];
    let mut current_objective = weighted_ipt_objective(
        &coefficients,
        design_matrix_flat,
        feature_count,
        treated_indicator,
        normalized_weights,
    );
    let vstar = (crate::util::usize_to_f64(observation_count) - 1.0)
        .ln()
        .max(1.0);
    for _ in 0..config.max_iter.max(1_000) {
        let (gradient, hessian) = weighted_ipt_grad_hess(
            &coefficients,
            design_matrix,
            design_matrix_flat,
            feature_count,
            treated_indicator,
            normalized_weights,
            vstar,
        );
        let step =
            solve_dense_system(&hessian, &gradient).map_err(|_| DrDidError::SingularSystem)?;
        let max_step = step.iter().map(|value| value.abs()).fold(0.0, f64::max);
        if max_step <= config.tol.min(1e-12) {
            return Ok(coefficients);
        }

        let mut step_scale = 1.0;
        let mut accepted = false;
        for _ in 0..11 {
            stepped_candidate(&coefficients, &step, step_scale, &mut ipt_candidate);
            let candidate_objective = weighted_ipt_objective(
                &ipt_candidate,
                design_matrix_flat,
                feature_count,
                treated_indicator,
                normalized_weights,
            );
            if candidate_objective <= current_objective {
                coefficients.clone_from_slice(&ipt_candidate);
                current_objective = candidate_objective;
                accepted = true;
                break;
            }
            step_scale *= 0.5;
            if step_scale < 1.0 / 1024.0 {
                break;
            }
        }

        if !accepted {
            break;
        }
    }

    Ok(coefficients)
}

fn fit_weighted_logit_initial(
    design_matrix_flat: &[f64],
    feature_count: usize,
    treated_indicator: &[f64],
    normalized_weights: &[f64],
    clip: f64,
    max_iter: u64,
    tol: f64,
) -> Vec<f64> {
    let mut coefficients = vec![0.0; feature_count];
    coefficients[0] = weighted_empirical_logit(treated_indicator, normalized_weights, clip);

    let observation_count = treated_indicator.len();
    let design = Mat::from_fn(observation_count, feature_count, |row, col| {
        design_matrix_flat[row * feature_count + col]
    });
    let mut linear_index = vec![0.0; observation_count];
    let mut probabilities = vec![0.0; observation_count];
    let mut working_response = vec![0.0; observation_count];
    let mut row_weights = vec![0.0; observation_count];

    for _ in 0..max_iter {
        for row_index in 0..observation_count {
            let row =
                &design_matrix_flat[row_index * feature_count..(row_index + 1) * feature_count];
            let index = dot(row, &coefficients).clamp(-35.0, 35.0);
            linear_index[row_index] = index;
            let probability = 1.0 / (1.0 + (-index).exp());
            probabilities[row_index] = probability.clamp(clip, 1.0 - clip);
        }

        for row_index in 0..observation_count {
            let probability = probabilities[row_index];
            let variance = (probability * (1.0 - probability)).max(1e-8);
            let weight = normalized_weights[row_index] * variance;
            row_weights[row_index] = weight;
            working_response[row_index] =
                linear_index[row_index] + (treated_indicator[row_index] - probability) / variance;
        }

        let weighted_design = Mat::from_fn(observation_count, feature_count, |row, col| {
            design[(row, col)] * row_weights[row].sqrt()
        });
        let weighted_working_response = Mat::from_fn(observation_count, 1, |row, _| {
            row_weights[row] * working_response[row]
        });
        let weighted_crossprod_mat = weighted_design.transpose() * &weighted_design;
        let weighted_response_mat = design.transpose() * weighted_working_response;

        let mut weighted_crossprod = vec![0.0; feature_count * feature_count];
        for row in 0..feature_count {
            for col in 0..feature_count {
                weighted_crossprod[row * feature_count + col] = weighted_crossprod_mat[(row, col)];
            }
        }
        let mut weighted_response = vec![0.0; feature_count];
        for row in 0..feature_count {
            weighted_response[row] = weighted_response_mat[(row, 0)];
        }
        for diagonal_index in 0..feature_count {
            weighted_crossprod[diagonal_index * feature_count + diagonal_index] += 1e-8;
        }

        let Ok(next) = solve_dense_system(&weighted_crossprod, &weighted_response) else {
            break;
        };
        let max_change = next
            .iter()
            .zip(coefficients.iter())
            .map(|(new_value, old_value)| (new_value - old_value).abs())
            .fold(0.0, f64::max);
        coefficients = next;
        if max_change <= tol.max(1e-10) {
            break;
        }
    }

    coefficients
}

fn weighted_empirical_logit(
    treated_indicator: &[f64],
    normalized_weights: &[f64],
    clip: f64,
) -> f64 {
    let treated_weight = treated_indicator
        .iter()
        .zip(normalized_weights.iter())
        .map(|(treated, weight)| treated * weight)
        .sum::<f64>();
    let total_weight = normalized_weights.iter().sum::<f64>();
    let share = (treated_weight / total_weight).clamp(clip, 1.0 - clip);
    (share / (1.0 - share)).ln()
}

fn weighted_calibrated_ps_loss(
    coefficients: &[f64],
    design_matrix_flat: &[f64],
    feature_count: usize,
    treated_indicator: &[f64],
    normalized_weights: &[f64],
    linear_index: &mut [f64],
) -> f64 {
    let n = crate::util::usize_to_f64(treated_indicator.len());
    let mut loss = 0.0;
    for row_index in 0..treated_indicator.len() {
        let row = &design_matrix_flat[row_index * feature_count..(row_index + 1) * feature_count];
        let index = dot(row, coefficients);
        linear_index[row_index] = index;
        let exp_index = index.exp();
        let term = if treated_indicator[row_index] > 0.5 {
            index
        } else {
            -exp_index
        };
        loss -= normalized_weights[row_index] * term / n;
    }
    loss
}

fn weighted_calibrated_ps_grad_hess(
    design_matrix: &Mat<f64>,
    feature_count: usize,
    treated_indicator: &[f64],
    normalized_weights: &[f64],
    linear_index: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = crate::util::usize_to_f64(treated_indicator.len());
    let mut gradient_scalars = vec![0.0; treated_indicator.len()];
    let mut hessian_scalars = vec![0.0; treated_indicator.len()];
    for row_index in 0..treated_indicator.len() {
        let exp_index = linear_index[row_index].exp();
        gradient_scalars[row_index] = if treated_indicator[row_index] > 0.5 {
            -normalized_weights[row_index] / n
        } else {
            normalized_weights[row_index] * exp_index / n
        };
        hessian_scalars[row_index] = if treated_indicator[row_index] > 0.5 {
            0.0
        } else {
            normalized_weights[row_index] * exp_index / n
        };
    }
    weighted_grad_hess_from_scalars(
        design_matrix,
        feature_count,
        &gradient_scalars,
        &hessian_scalars,
    )
}

fn weighted_ipt_objective(
    coefficients: &[f64],
    design_matrix_flat: &[f64],
    feature_count: usize,
    treated_indicator: &[f64],
    normalized_weights: &[f64],
) -> f64 {
    let n = crate::util::usize_to_f64(treated_indicator.len());
    let vstar = (n - 1.0).ln().max(1.0);
    let cn = -(n - 1.0);
    let bn = (n - 1.0).mul_add((n - 1.0).ln(), -n);
    let an = -(n - 1.0) * 0.5f64.mul_add((n - 1.0).ln().powi(2), 1.0 - (n - 1.0).ln());

    let mut objective = 0.0;
    for row_index in 0..treated_indicator.len() {
        let row = &design_matrix_flat[row_index * feature_count..(row_index + 1) * feature_count];
        let value = dot(row, coefficients);
        let phi = if value < vstar {
            -value - value.exp()
        } else {
            (0.5 * cn * value).mul_add(value, an + bn * value)
        };
        objective -= (normalized_weights[row_index] * (1.0 - treated_indicator[row_index]))
            .mul_add(phi, value);
    }
    objective
}

fn weighted_ipt_grad_hess(
    coefficients: &[f64],
    design_matrix: &Mat<f64>,
    design_matrix_flat: &[f64],
    feature_count: usize,
    treated_indicator: &[f64],
    normalized_weights: &[f64],
    vstar: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = crate::util::usize_to_f64(treated_indicator.len());
    let cn = -(n - 1.0);
    let bn = (n - 1.0).mul_add((n - 1.0).ln(), -n);

    let mut gradient_scalars = vec![0.0; treated_indicator.len()];
    let mut hessian_scalars = vec![0.0; treated_indicator.len()];
    for row_index in 0..treated_indicator.len() {
        let row = &design_matrix_flat[row_index * feature_count..(row_index + 1) * feature_count];
        let value = dot(row, coefficients);
        let (phi1, phi2) = if value < vstar {
            (-1.0 - value.exp(), -value.exp())
        } else {
            (bn + cn * value, cn)
        };
        gradient_scalars[row_index] = -normalized_weights[row_index]
            * (1.0 - treated_indicator[row_index]).mul_add(phi1, 1.0);
        hessian_scalars[row_index] =
            -normalized_weights[row_index] * (1.0 - treated_indicator[row_index]) * phi2;
    }
    weighted_grad_hess_from_scalars(
        design_matrix,
        feature_count,
        &gradient_scalars,
        &hessian_scalars,
    )
}

fn weighted_grad_hess_from_scalars(
    design_matrix: &Mat<f64>,
    feature_count: usize,
    gradient_scalars: &[f64],
    hessian_scalars: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let observation_count = gradient_scalars.len();
    let gradient_rhs = Mat::from_fn(observation_count, 1, |row, _| gradient_scalars[row]);
    let gradient_mat = design_matrix.transpose() * gradient_rhs;
    let weighted_design = Mat::from_fn(observation_count, feature_count, |row, col| {
        design_matrix[(row, col)] * hessian_scalars[row].sqrt()
    });
    let hessian_mat = weighted_design.transpose() * &weighted_design;

    let mut gradient = vec![0.0; feature_count];
    for row in 0..feature_count {
        gradient[row] = gradient_mat[(row, 0)];
    }
    let mut hessian = vec![0.0; feature_count * feature_count];
    for row in 0..feature_count {
        for col in 0..feature_count {
            hessian[row * feature_count + col] = hessian_mat[(row, col)];
        }
    }
    (gradient, hessian)
}

fn predict_logistic_scores(x: &[f64], p: usize, beta: &[f64], clip: f64) -> Vec<f64> {
    let n = x.len() / p;
    let mut out = Vec::with_capacity(n);
    for row_idx in 0..n {
        let x_row = &x[row_idx * p..(row_idx + 1) * p];
        let value = dot(x_row, beta);
        let clipped = value.clamp(-35.0, 35.0);
        let score: f64 = 1.0 / (1.0 + (-clipped).exp());
        out.push(score.clamp(clip, 1.0 - clip));
    }
    out
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}
