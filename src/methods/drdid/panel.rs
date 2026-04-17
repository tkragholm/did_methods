use faer::Mat;
use faer::prelude::SolveLstsq;
use itertools::izip;

use crate::estimators::common::linalg::{
    SpdCholeskyScratch, solve_spd_system, solve_spd_system_multi_rhs,
};
use crate::inference::{
    multiplier_bootstrap_ci, standard_error_from_influence, validate_confidence_level,
};
use crate::methods::drdid::moments::normalize_weights_to_n as normalize_weights_to_n_shared;
use crate::types::{DrDidConfig, DrDidError, DrDidEstimate, DrDidObservation};
use crate::util::usize_to_f64;

/// Estimate the improved / locally efficient panel `DR-DiD` ATT from outcome deltas.
///
/// This is the panel estimator in Sant'Anna and Zhao (2020), corresponding to
/// the locally efficient / improved specification used by
/// `DRDID::drdid_panel`. Let `ΔY_i = Y_{i1} - Y_{i0}` and let `D_i` denote
/// treatment. The estimator combines:
///
/// - a propensity score model `π(X_i) = P(D_i = 1 | X_i)`,
/// - a control-group outcome regression `m_0(X_i) = E[ΔY_i | D_i = 0, X_i]`,
/// - and the augmented inverse-probability weighted score
///
/// ```text
/// ATT = E[w_1(D_i) {ΔY_i - m_0(X_i)}] - E[w_0(D_i, X_i) {ΔY_i - m_0(X_i)}]
/// ```
///
/// where the weights follow the improved `DRDID` normalization.
///
/// References:
/// - Sant'Anna, P. H. C. and Zhao, J. (2020). "Doubly Robust Difference-in-
///   Differences Estimators". *Journal of Econometrics*.
/// - `DRDID::drdid_panel` (R), used as the numerical parity reference.
///
/// This function is retained under the historical `estimate_drdid_panel` name for
/// compatibility, but the implemented estimator is the improved / locally efficient
/// panel estimator corresponding to `DRDID::drdid_panel`.
///
/// # Errors
///
/// Returns [`DrDidError`] when inputs/configuration are invalid or nuisance models cannot be fit.
pub fn estimate_drdid_panel(
    observations: &[DrDidObservation],
    config: DrDidConfig,
) -> Result<DrDidEstimate, DrDidError> {
    validate_drdid_config(config)?;
    if observations.is_empty() {
        return Err(DrDidError::EmptyInput);
    }

    let prepared = prepare_panel_data(observations)?;
    let nuisance = fit_panel_nuisance_models(&prepared, config)?;
    let panel_att = estimate_panel_att_and_influence(
        &prepared,
        &nuisance.normalized_weights,
        &nuisance.propensity_scores,
        &nuisance.outcome_predictions,
    )?;

    let se = standard_error_from_influence(&panel_att.influence_function);
    let (ci_low, ci_high) = multiplier_bootstrap_ci(
        panel_att.att,
        &panel_att.influence_function,
        config.inference(),
        config.bootstrap(),
    );

    Ok(DrDidEstimate {
        att: panel_att.att,
        se,
        ci_low,
        ci_high,
        treated_n: prepared.treated_n,
        control_n: prepared.control_n,
        total_weight: prepared.total_weight,
        influence_function: panel_att.influence_function,
    })
}

struct PanelPreparedData {
    feature_count: usize,
    treated_n: usize,
    control_n: usize,
    total_weight: f64,
    treated_indicator: Vec<f64>,
    outcome_delta: Vec<f64>,
    sampling_weights: Vec<f64>,
    design_matrix_flat: Vec<f64>,
}

struct PanelNuisanceFits {
    normalized_weights: Vec<f64>,
    propensity_scores: Vec<f64>,
    outcome_predictions: Vec<f64>,
}

struct PanelAttEstimate {
    att: f64,
    influence_function: Vec<f64>,
}

const PANEL_TRIM_LEVEL: f64 = 0.995;

fn prepare_panel_data(observations: &[DrDidObservation]) -> Result<PanelPreparedData, DrDidError> {
    let covariate_count = observations.first().map_or(0, |row| row.covariates.len());
    let feature_count = covariate_count.max(1);
    let observation_count = observations.len();

    let mut treated_n = 0_usize;
    let mut control_n = 0_usize;
    let mut total_weight = 0.0_f64;

    let mut treated_indicator = Vec::with_capacity(observation_count);
    let mut outcome_delta = Vec::with_capacity(observation_count);
    let mut sampling_weights = Vec::with_capacity(observation_count);
    let mut design_matrix_flat = Vec::with_capacity(observation_count * feature_count);

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
        for covariate in &row.covariates {
            if !covariate.is_finite() {
                return Err(DrDidError::InvalidCovariate { value: *covariate });
            }
        }

        if row.treated {
            treated_n += 1;
            treated_indicator.push(1.0);
        } else {
            control_n += 1;
            treated_indicator.push(0.0);
        }
        outcome_delta.push(row.delta_outcome);
        total_weight += row.weight;
        sampling_weights.push(row.weight);
        if covariate_count == 0 {
            design_matrix_flat.push(1.0);
        } else {
            design_matrix_flat.extend_from_slice(&row.covariates);
        }
    }

    if treated_n == 0 {
        return Err(DrDidError::NoTreated);
    }
    if control_n == 0 {
        return Err(DrDidError::NoControl);
    }

    Ok(PanelPreparedData {
        feature_count,
        treated_n,
        control_n,
        total_weight,
        treated_indicator,
        outcome_delta,
        sampling_weights,
        design_matrix_flat,
    })
}

fn fit_panel_nuisance_models(
    prepared: &PanelPreparedData,
    config: DrDidConfig,
) -> Result<PanelNuisanceFits, DrDidError> {
    let normalized_weights = normalize_weights_to_n(&prepared.sampling_weights)?;
    let observation_count = prepared.treated_indicator.len();
    let beta_ps = fit_logistic_irls(
        LogisticIrlsInputs {
            design_matrix_flat: &prepared.design_matrix_flat,
            feature_count: prepared.feature_count,
            treated_indicator: &prepared.treated_indicator,
            normalized_weights: &normalized_weights,
        },
        LogisticIrlsSettings {
            ridge: config.ridge,
            max_iter: config.max_iter,
            tol: config.tol,
            propensity_clip: config.propensity_clip,
        },
    )?;
    let propensity_scores = predict_logistic_scores(
        &prepared.design_matrix_flat,
        prepared.feature_count,
        &beta_ps,
        config.propensity_clip,
    );

    let mut control_design = Vec::with_capacity(prepared.control_n * prepared.feature_count);
    let mut control_outcome = Vec::with_capacity(prepared.control_n);
    let mut control_weights = Vec::with_capacity(prepared.control_n);
    for (row_index, normalized_weight) in normalized_weights
        .iter()
        .enumerate()
        .take(observation_count)
    {
        if prepared.treated_indicator[row_index] < 0.5 {
            let row_start = row_index * prepared.feature_count;
            let row_end = row_start + prepared.feature_count;
            control_design.extend_from_slice(&prepared.design_matrix_flat[row_start..row_end]);
            control_outcome.push(prepared.outcome_delta[row_index]);
            control_weights.push(*normalized_weight);
        }
    }

    let beta_outcome = fit_weighted_linear(
        &control_design,
        prepared.feature_count,
        &control_outcome,
        &control_weights,
        config.ridge,
    )?;
    let outcome_predictions = predict_linear(
        &prepared.design_matrix_flat,
        prepared.feature_count,
        &beta_outcome,
    );

    Ok(PanelNuisanceFits {
        normalized_weights,
        propensity_scores,
        outcome_predictions,
    })
}

fn estimate_panel_att_and_influence(
    prepared: &PanelPreparedData,
    normalized_weights: &[f64],
    propensity_scores: &[f64],
    outcome_predictions: &[f64],
) -> Result<PanelAttEstimate, DrDidError> {
    let observation_count = prepared.treated_indicator.len();
    let observation_count_f64 = usize_to_f64(observation_count);

    let mut trim_indicator = vec![1.0; observation_count];
    let mut treated_weights = Vec::with_capacity(observation_count);
    let mut control_ipw_weights = Vec::with_capacity(observation_count);
    let mut treated_dr_contributions = Vec::with_capacity(observation_count);
    let mut control_dr_contributions = Vec::with_capacity(observation_count);
    let mut hessian_weights = Vec::with_capacity(observation_count);

    for (
        row_index,
        (treated, outcome_delta, propensity_score, normalized_weight, outcome_prediction),
    ) in izip!(
        prepared.treated_indicator.iter(),
        prepared.outcome_delta.iter(),
        propensity_scores.iter(),
        normalized_weights.iter(),
        outcome_predictions.iter()
    )
    .enumerate()
    {
        let residual = outcome_delta - outcome_prediction;
        if *treated < 0.5 && *propensity_score >= PANEL_TRIM_LEVEL {
            trim_indicator[row_index] = 0.0;
        }
        let treated_weight = normalized_weight * treated;
        let control_weight =
            trim_indicator[row_index] * normalized_weight * propensity_score * (1.0 - treated)
                / (1.0 - propensity_score);
        let trimmed_treated_weight = trim_indicator[row_index] * treated_weight;
        treated_weights.push(trimmed_treated_weight);
        control_ipw_weights.push(control_weight);
        treated_dr_contributions.push(trimmed_treated_weight * residual);
        control_dr_contributions.push(control_weight * residual);
        hessian_weights.push(propensity_score * (1.0 - propensity_score) * normalized_weight);
    }

    let mean_treated_weight = treated_weights.iter().sum::<f64>() / observation_count_f64;
    let mean_control_weight = control_ipw_weights.iter().sum::<f64>() / observation_count_f64;
    if mean_treated_weight <= 0.0 {
        return Err(DrDidError::NoTreated);
    }
    if mean_control_weight <= 0.0 {
        return Err(DrDidError::NoControl);
    }

    let treated_eta = (treated_dr_contributions.iter().sum::<f64>() / observation_count_f64)
        / mean_treated_weight;
    let control_eta = (control_dr_contributions.iter().sum::<f64>() / observation_count_f64)
        / mean_control_weight;
    let att = treated_eta - control_eta;

    let wols_influence =
        weighted_outcome_influence(prepared, normalized_weights, outcome_predictions)?;
    let propensity_influence = propensity_score_influence(
        prepared,
        normalized_weights,
        &hessian_weights,
        propensity_scores,
    )?;

    let treated_moment = column_means(prepared, &treated_weights, &vec![1.0; observation_count]);
    let control_ps_moment = column_means(
        prepared,
        &control_ipw_weights,
        &prepared
            .outcome_delta
            .iter()
            .zip(outcome_predictions.iter())
            .map(|(outcome_delta, prediction)| outcome_delta - prediction - control_eta)
            .collect::<Vec<_>>(),
    );
    let control_reg_moment = column_means(
        prepared,
        &control_ipw_weights,
        &vec![1.0; observation_count],
    );

    let mut influence_function = Vec::with_capacity(observation_count);
    for row_index in 0..observation_count {
        let treated_main =
            treated_weights[row_index].mul_add(-treated_eta, treated_dr_contributions[row_index]);
        let treated_nuisance = dot(&wols_influence[row_index], &treated_moment);
        let treated_component = (treated_main - treated_nuisance) / mean_treated_weight;

        let control_main = control_ipw_weights[row_index]
            .mul_add(-control_eta, control_dr_contributions[row_index]);
        let control_ps_nuisance = dot(&propensity_influence[row_index], &control_ps_moment);
        let control_reg_nuisance = dot(&wols_influence[row_index], &control_reg_moment);
        let control_component =
            (control_main + control_ps_nuisance - control_reg_nuisance) / mean_control_weight;

        influence_function.push(treated_component - control_component);
    }

    Ok(PanelAttEstimate {
        att,
        influence_function,
    })
}

fn weighted_outcome_influence(
    prepared: &PanelPreparedData,
    normalized_weights: &[f64],
    outcome_predictions: &[f64],
) -> Result<Vec<Vec<f64>>, DrDidError> {
    let observation_count = prepared.treated_indicator.len();
    let feature_count = prepared.feature_count;
    let n_f = observation_count_f64(observation_count);
    let design = Mat::from_fn(observation_count, feature_count, |row, col| {
        prepared.design_matrix_flat[row * feature_count + col]
    });
    let weighted_residual_features = Mat::from_fn(observation_count, feature_count, |row, col| {
        let weight = normalized_weights[row] * (1.0 - prepared.treated_indicator[row]);
        let residual = prepared.outcome_delta[row] - outcome_predictions[row];
        weight * residual * design[(row, col)]
    });
    let weighted_design = Mat::from_fn(observation_count, feature_count, |row, col| {
        let weight = normalized_weights[row] * (1.0 - prepared.treated_indicator[row]);
        design[(row, col)] * (weight / n_f).sqrt()
    });
    let xp_x_mat = weighted_design.transpose() * &weighted_design;
    let mut xp_x = vec![0.0; feature_count * feature_count];
    for row in 0..feature_count {
        for col in 0..feature_count {
            xp_x[row * feature_count + col] = xp_x_mat[(row, col)];
        }
    }

    let weighted_residual_features_t = weighted_residual_features.transpose().to_owned();
    let solved = solve_spd_system_multi_rhs(&xp_x, &weighted_residual_features_t)
        .map_err(|_| DrDidError::SingularSystem)?;
    let influence = solved.transpose().to_owned();
    Ok(mat_rows_to_nested(&influence))
}

fn propensity_score_influence(
    prepared: &PanelPreparedData,
    normalized_weights: &[f64],
    hessian_weights: &[f64],
    propensity_scores: &[f64],
) -> Result<Vec<Vec<f64>>, DrDidError> {
    let observation_count = prepared.treated_indicator.len();
    let feature_count = prepared.feature_count;
    let design = Mat::from_fn(observation_count, feature_count, |row, col| {
        prepared.design_matrix_flat[row * feature_count + col]
    });
    let score_rows = Mat::from_fn(observation_count, feature_count, |row, col| {
        let score_scalar =
            normalized_weights[row] * (prepared.treated_indicator[row] - propensity_scores[row]);
        score_scalar * design[(row, col)]
    });
    let weighted_design = Mat::from_fn(observation_count, feature_count, |row, col| {
        design[(row, col)] * hessian_weights[row].sqrt()
    });
    let hessian_mat = weighted_design.transpose() * &weighted_design;
    let mut hessian = vec![0.0; feature_count * feature_count];
    for row in 0..feature_count {
        for col in 0..feature_count {
            hessian[row * feature_count + col] = hessian_mat[(row, col)];
        }
    }

    let scale = usize_to_f64(observation_count);
    let score_rows_t = score_rows.transpose().to_owned();
    let solved = solve_spd_system_multi_rhs(&hessian, &score_rows_t)
        .map_err(|_| DrDidError::SingularSystem)?;
    let mut influence = solved.transpose().to_owned();
    for row in 0..influence.nrows() {
        for col in 0..influence.ncols() {
            influence[(row, col)] *= scale;
        }
    }
    Ok(mat_rows_to_nested(&influence))
}

fn mat_rows_to_nested(matrix: &Mat<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|col| matrix[(row, col)])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

fn column_means(prepared: &PanelPreparedData, row_weights: &[f64], row_values: &[f64]) -> Vec<f64> {
    let observation_count = prepared.treated_indicator.len();
    let feature_count = prepared.feature_count;
    let mut means = vec![0.0; feature_count];
    for row_index in 0..observation_count {
        let row_start = row_index * feature_count;
        let row = &prepared.design_matrix_flat[row_start..row_start + feature_count];
        let scalar = row_weights[row_index] * row_values[row_index]
            / observation_count_f64(observation_count);
        for feature_index in 0..feature_count {
            means[feature_index] = scalar.mul_add(row[feature_index], means[feature_index]);
        }
    }
    means
}

fn observation_count_f64(observation_count: usize) -> f64 {
    usize_to_f64(observation_count)
}

fn validate_drdid_config(config: DrDidConfig) -> Result<(), DrDidError> {
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

fn normalize_weights_to_n(weights: &[f64]) -> Result<Vec<f64>, DrDidError> {
    normalize_weights_to_n_shared(weights)
        .map_err(|_| DrDidError::InvalidConfig("sum of weights must be > 0".to_string()))
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
}

#[derive(Clone, Copy)]
struct LogisticIrlsInputs<'a> {
    design_matrix_flat: &'a [f64],
    feature_count: usize,
    treated_indicator: &'a [f64],
    normalized_weights: &'a [f64],
}

#[derive(Clone, Copy)]
struct LogisticIrlsSettings {
    ridge: f64,
    max_iter: usize,
    tol: f64,
    propensity_clip: f64,
}

fn fit_weighted_linear(
    design_matrix_flat: &[f64],
    feature_count: usize,
    outcome: &[f64],
    normalized_weights: &[f64],
    ridge: f64,
) -> Result<Vec<f64>, DrDidError> {
    let observation_count = outcome.len();
    let design = Mat::from_fn(observation_count, feature_count, |row, col| {
        design_matrix_flat[row * feature_count + col]
    });
    let weighted_design = Mat::from_fn(observation_count, feature_count, |row, col| {
        design[(row, col)] * normalized_weights[row].sqrt()
    });
    let weighted_outcome = Mat::from_fn(observation_count, 1, |row, _| {
        normalized_weights[row] * outcome[row]
    });
    let weighted_outcome_lstsq = Mat::from_fn(observation_count, 1, |row, _| {
        normalized_weights[row].sqrt() * outcome[row]
    });
    let normal_matrix_mat = weighted_design.transpose() * &weighted_design;
    let rhs_mat = design.transpose() * weighted_outcome;

    let mut normal_matrix = vec![0.0; feature_count * feature_count];
    for row in 0..feature_count {
        for col in 0..feature_count {
            normal_matrix[row * feature_count + col] = normal_matrix_mat[(row, col)];
        }
    }
    for feature_index in 0..feature_count {
        normal_matrix[feature_index * feature_count + feature_index] += ridge;
    }
    let mut rhs_vector = vec![0.0; feature_count];
    for feature_index in 0..feature_count {
        rhs_vector[feature_index] = rhs_mat[(feature_index, 0)];
    }

    match solve_spd_system(&normal_matrix, &rhs_vector) {
        Ok(solution) => Ok(solution),
        Err(_) if ridge > 0.0 && ridge <= 1e-6 => {
            let solution = weighted_design
                .col_piv_qr()
                .solve_lstsq(&weighted_outcome_lstsq);
            let mut out = vec![0.0; feature_count];
            for feature_index in 0..feature_count {
                let value = solution[(feature_index, 0)];
                if !value.is_finite() {
                    return Err(DrDidError::SingularSystem);
                }
                out[feature_index] = value;
            }
            Ok(out)
        }
        Err(_) => Err(DrDidError::SingularSystem),
    }
}

fn fit_logistic_irls(
    inputs: LogisticIrlsInputs<'_>,
    settings: LogisticIrlsSettings,
) -> Result<Vec<f64>, DrDidError> {
    let observation_count = inputs.treated_indicator.len();
    let feature_count = inputs.feature_count;
    let convergence_tol = settings.tol.min(1e-12);
    let max_iterations = settings.max_iter.max(1_000);
    let mut coefficients = vec![0.0; feature_count];
    let treated_share =
        inputs.treated_indicator.iter().sum::<f64>() / usize_to_f64(observation_count);
    coefficients[0] = logit_clamped(treated_share, settings.propensity_clip);

    let mut hessian = vec![0.0; feature_count * feature_count];
    let mut gradient = vec![0.0; feature_count];
    let mut candidate = vec![0.0; feature_count];
    let mut probabilities = vec![0.0; observation_count];
    let mut step = vec![0.0; feature_count];
    let mut scratch_solver = SpdCholeskyScratch::new(feature_count);
    let mut current_loglik = weighted_loglik(&coefficients, inputs, &mut probabilities);

    for _ in 0..max_iterations {
        hessian.fill(0.0);
        gradient.fill(0.0);

        for feature_index in 0..feature_count {
            hessian[feature_index * feature_count + feature_index] = settings.ridge;
        }

        for (row_index, probability) in probabilities.iter().copied().enumerate() {
            let feature_row = &inputs.design_matrix_flat
                [row_index * feature_count..(row_index + 1) * feature_count];
            let response = inputs.treated_indicator[row_index];
            let weight = inputs.normalized_weights[row_index];
            let residual = weight * (response - probability);
            let variance =
                (weight * probability * (1.0 - probability)).max(settings.propensity_clip);

            for left_feature_index in 0..feature_count {
                gradient[left_feature_index] = residual.mul_add(
                    feature_row[left_feature_index],
                    gradient[left_feature_index],
                );
                for right_feature_index in left_feature_index..feature_count {
                    let weighted_outer = variance
                        * feature_row[left_feature_index]
                        * feature_row[right_feature_index];
                    hessian[left_feature_index * feature_count + right_feature_index] +=
                        weighted_outer;
                    if left_feature_index != right_feature_index {
                        hessian[right_feature_index * feature_count + left_feature_index] +=
                            weighted_outer;
                    }
                }
            }
        }

        if scratch_solver
            .solve_single_rhs_into(&hessian, &gradient, &mut step)
            .is_err()
        {
            step = solve_spd_system(&hessian, &gradient).map_err(|_| DrDidError::SingularSystem)?;
        }
        let max_step = step.iter().map(|value| value.abs()).fold(0.0, f64::max);
        if max_step <= convergence_tol {
            return Ok(coefficients);
        }

        let mut step_scale = 1.0;
        let mut accepted = false;
        for _ in 0..11 {
            for feature_index in 0..feature_count {
                candidate[feature_index] =
                    coefficients[feature_index] + step_scale * step[feature_index];
            }
            let candidate_loglik = weighted_loglik(&candidate, inputs, &mut probabilities);
            if candidate_loglik >= current_loglik {
                coefficients.clone_from_slice(&candidate);
                current_loglik = candidate_loglik;
                accepted = true;
                break;
            }
            step_scale *= 0.5;
            if step_scale < 1.0 / 1024.0 {
                break;
            }
        }

        if !accepted {
            return Ok(coefficients);
        }
    }
    Ok(coefficients)
}

fn predict_linear(x: &[f64], p: usize, beta: &[f64]) -> Vec<f64> {
    let n = x.len() / p;
    let mut out = Vec::with_capacity(n);
    for row_idx in 0..n {
        let x_row = &x[row_idx * p..(row_idx + 1) * p];
        out.push(dot(x_row, beta));
    }
    out
}

fn predict_logistic_scores(x: &[f64], p: usize, beta: &[f64], clip: f64) -> Vec<f64> {
    predict_linear(x, p, beta)
        .into_iter()
        .map(|value| sigmoid(value).min(1.0 - clip))
        .collect()
}

fn sigmoid(value: f64) -> f64 {
    let clipped = value.clamp(-35.0, 35.0);
    1.0 / (1.0 + (-clipped).exp())
}

fn logit_clamped(probability: f64, clip: f64) -> f64 {
    let p = probability.clamp(clip, 1.0 - clip);
    (p / (1.0 - p)).ln()
}

fn weighted_loglik(
    coefficients: &[f64],
    inputs: LogisticIrlsInputs<'_>,
    probabilities: &mut [f64],
) -> f64 {
    let mut loglik = 0.0;
    for (prob, feature_row, &response, &weight) in izip!(
        probabilities.iter_mut(),
        inputs.design_matrix_flat.chunks_exact(inputs.feature_count),
        inputs.treated_indicator.iter(),
        inputs.normalized_weights.iter()
    ) {
        let probability = sigmoid(dot(feature_row, coefficients)).clamp(1e-12, 1.0 - 1e-12);
        *prob = probability;
        loglik = weight.mul_add(
            response.mul_add(
                probability.ln(),
                (1.0 - response) * (1.0 - probability).ln(),
            ),
            loglik,
        );
    }
    loglik
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_standard_error_from_influence_function() {
        let influence = [2.0, -1.0, 0.0, 1.0];
        let se = standard_error_from_influence(&influence);
        assert!((se - 0.559_016_994_374_947_5).abs() < 1e-12);
    }

    #[test]
    fn validate_config_rejects_invalid_values() {
        assert!(matches!(
            validate_drdid_config(DrDidConfig::builder().confidence_level(1.0).build()),
            Err(DrDidError::InvalidConfig(msg)) if msg.contains("confidence_level")
        ));
        assert!(matches!(
            validate_drdid_config(DrDidConfig::builder().propensity_clip(0.5).build()),
            Err(DrDidError::InvalidConfig(msg)) if msg.contains("propensity_clip")
        ));
        assert!(matches!(
            validate_drdid_config(DrDidConfig::builder().ridge(-1.0).build()),
            Err(DrDidError::InvalidConfig(msg)) if msg.contains("ridge")
        ));
        assert!(matches!(
            validate_drdid_config(DrDidConfig::builder().max_iter(0).build()),
            Err(DrDidError::InvalidConfig(msg)) if msg.contains("max_iter")
        ));
        assert!(matches!(
            validate_drdid_config(DrDidConfig::builder().tol(0.0).build()),
            Err(DrDidError::InvalidConfig(msg)) if msg.contains("tol")
        ));
        assert!(matches!(
            validate_drdid_config(DrDidConfig::builder().bootstrap_reps(0).build()),
            Err(DrDidError::InvalidConfig(msg)) if msg.contains("bootstrap_reps")
        ));
    }

    #[test]
    fn normalize_weights_to_n_scales_and_rejects_invalid_totals() {
        let out = normalize_weights_to_n(&[2.0, 2.0, 4.0]).expect("normalized");
        assert!((out.iter().sum::<f64>() - 3.0).abs() < 1e-12);

        let err = normalize_weights_to_n(&[0.0, 0.0]).expect_err("must fail");
        assert!(matches!(
            err,
            DrDidError::InvalidConfig(msg) if msg.contains("sum of weights")
        ));

        let err = normalize_weights_to_n(&[f64::INFINITY, 1.0]).expect_err("must fail");
        assert!(matches!(
            err,
            DrDidError::InvalidConfig(msg) if msg.contains("sum of weights")
        ));
    }

    #[test]
    fn shared_dense_solver_handles_regular_and_singular_cases() {
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![1.0, 2.0];
        let sol = crate::estimators::common::linalg::solve_dense_system(&a, &b).expect("solve");
        assert!((sol[0] - 0.2).abs() < 1e-12);
        assert!((sol[1] - 0.6).abs() < 1e-12);

        let singular = vec![1.0, 2.0, 2.0, 4.0];
        let err = crate::estimators::common::linalg::solve_dense_system(&singular, &b)
            .expect_err("must fail");
        assert!(err.contains("non-finite"));
    }

    #[test]
    fn fit_weighted_linear_and_predict_helpers_work() {
        let x = vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0];
        let y = vec![1.0, 3.0, 5.0];
        let w = vec![1.0, 2.0, 1.0];
        let beta = fit_weighted_linear(&x, 2, &y, &w, 1e-8).expect("fit");
        assert!((beta[0] - 1.0).abs() < 1e-6);
        assert!((beta[1] - 2.0).abs() < 1e-6);

        let pred = predict_linear(&x, 2, &beta);
        assert_eq!(pred.len(), 3);
        assert!((pred[0] - 1.0).abs() < 1e-6);
        assert!((pred[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn fit_weighted_linear_rejects_singular_system_without_ridge() {
        let x = vec![1.0, 0.0, 1.0, 0.0];
        let y = vec![1.0, 2.0];
        let w = vec![1.0, 1.0];
        let err = fit_weighted_linear(&x, 2, &y, &w, 0.0).expect_err("must fail");
        assert_eq!(err, DrDidError::SingularSystem);
    }

    #[test]
    fn fit_logistic_and_prediction_helpers_are_finite_and_clipped() {
        let x = vec![1.0, -2.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0];
        let d = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let w = vec![1.0; 5];
        let beta = fit_logistic_irls(
            LogisticIrlsInputs {
                design_matrix_flat: &x,
                feature_count: 2,
                treated_indicator: &d,
                normalized_weights: &w,
            },
            LogisticIrlsSettings {
                ridge: 1e-8,
                max_iter: 200,
                tol: 1e-8,
                propensity_clip: 1e-6,
            },
        )
        .expect("irls");
        assert_eq!(beta.len(), 2);
        assert!(beta.iter().all(|value| value.is_finite()));

        let ps = predict_logistic_scores(&x, 2, &beta, 0.1);
        assert_eq!(ps.len(), 5);
        assert!(ps.iter().all(|value| *value > 0.0 && *value <= 0.9));
    }

    #[test]
    fn fit_logistic_irls_rejects_singular_system_without_ridge() {
        let x = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        let w = vec![1.0; 3];
        let err = fit_logistic_irls(
            LogisticIrlsInputs {
                design_matrix_flat: &x,
                feature_count: 2,
                treated_indicator: &d,
                normalized_weights: &w,
            },
            LogisticIrlsSettings {
                ridge: 0.0,
                max_iter: 1,
                tol: 1e-12,
                propensity_clip: 1e-6,
            },
        )
        .expect_err("must fail");
        assert_eq!(err, DrDidError::SingularSystem);
    }

    #[test]
    fn bootstrap_and_quantile_helpers_are_stable() {
        let influence = [1.0, -1.0, 2.0, -2.0];
        let config = DrDidConfig::builder()
            .bootstrap_reps(200)
            .bootstrap_seed(42)
            .build();
        let ci_a = multiplier_bootstrap_ci(1.0, &influence, config.inference(), config.bootstrap());
        let ci_b = multiplier_bootstrap_ci(1.0, &influence, config.inference(), config.bootstrap());
        assert_eq!(ci_a, ci_b);
        assert!(ci_a.0 <= ci_a.1);
    }

    #[test]
    fn small_numeric_helpers_cover_edge_branches() {
        assert!((dot(&[1.0, 2.0], &[3.0, 4.0]) - 11.0).abs() < 1e-12);
        assert!(sigmoid(-1_000.0) > 0.0);
        assert!(sigmoid(1_000.0) < 1.0);
        assert!(logit_clamped(0.0, 1e-6).is_finite());
        assert!(logit_clamped(1.0, 1e-6).is_finite());
    }
}
