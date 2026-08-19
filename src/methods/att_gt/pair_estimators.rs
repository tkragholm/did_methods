use std::collections::BTreeSet;

use faer::Mat;
use itertools::izip;

use crate::estimators::outcome::linear::LinearOutcome;
use crate::estimators::outcome::model::OutcomeModel;
use crate::estimators::propensity::common::logistic_scores;
use crate::estimators::propensity::logistic::LogisticPS;
use crate::estimators::propensity::types::{Config as PropensityConfig, PropensityEstimator};
use crate::inference::{multiplier_bootstrap_ci, standard_error_from_influence};
use crate::methods::drdid::moments::{
    RepeatedMomentInputs, normalize_weights_to_n, repeated_att_moments,
};
use crate::methods::drdid::repeated::estimate_drdid_repeated_cross_section;
use crate::types::{
    AttGtConfig, AttGtDrConfig, AttGtDrObservation, AttGtError, AttGtEstimate,
    AttGtInfluenceOutput, BasePeriod, DidCell, DrDidConfig, DrDidRepeatedObservation, TimePeriod,
    TreatmentGroup,
};
use crate::util::usize_to_f64;

#[derive(Debug, Clone)]
pub struct PairEstimateWithInfluence {
    pub estimate: AttGtEstimate,
    pub influence_function: Vec<f64>,
}

pub fn estimate_att_gt_dr_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    estimate_att_gt_pair_with_influence(
        observations,
        config,
        "dr",
        |pair_rows, group, time, method_cfg| {
            let dr = estimate_drdid_repeated_cross_section(pair_rows, method_cfg.drdid)
                .map_err(|_| ())?;
            Ok(PairEstimateWithInfluence {
                estimate: AttGtEstimate {
                    group,
                    time,
                    event_time: time - group,
                    att: dr.att,
                    se: dr.se,
                    ci_low: dr.ci_low,
                    ci_high: dr.ci_high,
                    treated_n: dr.treated_n,
                    control_n: dr.control_n,
                    total_weight: dr.total_weight,
                },
                influence_function: dr.influence_function,
            })
        },
    )
}

pub fn estimate_att_gt_dr(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<Vec<AttGtEstimate>, AttGtError> {
    estimate_att_gt_pair(
        observations,
        config,
        "dr",
        |pair_rows, group, time, method_cfg| {
            let dr = estimate_drdid_repeated_cross_section(pair_rows, method_cfg.drdid)
                .map_err(|_| ())?;
            Ok(AttGtEstimate {
                group,
                time,
                event_time: time - group,
                att: dr.att,
                se: dr.se,
                ci_low: dr.ci_low,
                ci_high: dr.ci_high,
                treated_n: dr.treated_n,
                control_n: dr.control_n,
                total_weight: dr.total_weight,
            })
        },
    )
}

pub fn estimate_att_gt_or_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    estimate_att_gt_pair_with_influence(
        observations,
        config,
        "or",
        |pair_rows, group, time, method_cfg| {
            estimate_pair_or(pair_rows, group, time, method_cfg.att_gt, method_cfg.drdid)
        },
    )
}

pub fn estimate_att_gt_ipw_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    estimate_att_gt_pair_with_influence(
        observations,
        config,
        "ipw",
        |pair_rows, group, time, method_cfg| {
            estimate_pair_ipw(pair_rows, group, time, method_cfg.att_gt, method_cfg.drdid)
        },
    )
}

fn estimate_att_gt_pair_with_influence<F>(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
    method: &'static str,
    mut pair_estimator: F,
) -> Result<AttGtInfluenceOutput, AttGtError>
where
    F: FnMut(
        &[DrDidRepeatedObservation],
        i32,
        i32,
        AttGtDrConfig,
    ) -> Result<PairEstimateWithInfluence, ()>,
{
    let (all_times, treated_groups) = prepare_att_gt_dr_inputs(observations, config)?;
    let mut estimates = Vec::new();
    let mut influence_functions = Vec::new();
    let full_n = observations.len();
    let mut pair_rows = Vec::new();
    let mut pair_indices = Vec::new();

    for group in treated_groups {
        let universal_baseline_time = group - config.att_gt.anticipation_periods - 1;
        for &time in &all_times {
            let baseline_time =
                baseline_time_for_pair(time, group, universal_baseline_time, config.att_gt);
            if time == baseline_time {
                continue;
            }

            build_pair_rows_into(
                observations,
                group,
                time,
                baseline_time,
                config.att_gt,
                &mut pair_rows,
                &mut pair_indices,
            );
            if let Some(cell) = missing_pair_cell(&pair_rows) {
                if config.att_gt.skip_incomplete_pairs {
                    continue;
                }
                return Err(AttGtError::MissingCell {
                    group,
                    time,
                    baseline_time,
                    cell,
                });
            }

            let pair = pair_estimator(&pair_rows, group, time, config).map_err(|()| {
                AttGtError::PairEstimationFailure {
                    method,
                    group,
                    time,
                }
            })?;
            if pair.influence_function.len() != pair_rows.len() {
                return Err(AttGtError::InfluenceLengthMismatch {
                    method,
                    group,
                    time,
                    expected: pair_rows.len(),
                    actual: pair.influence_function.len(),
                });
            }

            // Rescale from the cell's own sample to the full one, for the same
            // reason as `panel_pairs::estimate_panel_cell`: psi is normalised so
            // that sqrt(sum(psi^2)) / n_cell is the cell's standard error, and
            // padding to full_n without rescaling shrinks each cell by a
            // different factor. Any consumer that recomputes a standard error
            // from these vectors, which is what a correlation-aware aggregation
            // must do, would otherwise get a number that depends on how large the
            // cell happened to be.
            let scale = usize_to_f64(full_n) / usize_to_f64(pair_rows.len());
            let mut aligned = vec![0.0; full_n];
            for (local_idx, global_idx) in pair_indices.iter().enumerate() {
                aligned[*global_idx] = pair.influence_function[local_idx] * scale;
            }
            estimates.push(pair.estimate);
            influence_functions.push(aligned);
        }
    }

    if estimates.is_empty() {
        return Err(AttGtError::NoEstimablePairs);
    }
    Ok(AttGtInfluenceOutput {
        estimates,
        influence_functions,
    })
}

fn estimate_att_gt_pair<F>(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
    method: &'static str,
    mut pair_estimator: F,
) -> Result<Vec<AttGtEstimate>, AttGtError>
where
    F: FnMut(&[DrDidRepeatedObservation], i32, i32, AttGtDrConfig) -> Result<AttGtEstimate, ()>,
{
    let (all_times, treated_groups) = prepare_att_gt_dr_inputs(observations, config)?;
    let mut estimates = Vec::new();
    let mut pair_rows = Vec::new();
    let mut pair_indices = Vec::new();

    for group in treated_groups {
        let universal_baseline_time = group - config.att_gt.anticipation_periods - 1;
        for &time in &all_times {
            let baseline_time =
                baseline_time_for_pair(time, group, universal_baseline_time, config.att_gt);
            if time == baseline_time {
                continue;
            }

            build_pair_rows_into(
                observations,
                group,
                time,
                baseline_time,
                config.att_gt,
                &mut pair_rows,
                &mut pair_indices,
            );
            if let Some(cell) = missing_pair_cell(&pair_rows) {
                if config.att_gt.skip_incomplete_pairs {
                    continue;
                }
                return Err(AttGtError::MissingCell {
                    group,
                    time,
                    baseline_time,
                    cell,
                });
            }

            let pair = pair_estimator(&pair_rows, group, time, config).map_err(|()| {
                AttGtError::PairEstimationFailure {
                    method,
                    group,
                    time,
                }
            })?;
            estimates.push(pair);
        }
    }

    if estimates.is_empty() {
        return Err(AttGtError::NoEstimablePairs);
    }
    Ok(estimates)
}

pub fn prepare_att_gt_dr_inputs(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<(Vec<i32>, Vec<i32>), AttGtError> {
    super::validate_config(config.att_gt)?;
    if observations.is_empty() {
        return Err(AttGtError::EmptyInput);
    }

    let covariate_count = observations.first().map_or(0, |row| row.covariates.len());
    let mut times = BTreeSet::new();
    let mut treated_groups = BTreeSet::new();
    let mut has_never_treated = false;

    for row in observations {
        if row.covariates.len() != covariate_count {
            return Err(AttGtError::InconsistentCovariateCount {
                expected: covariate_count,
                actual: row.covariates.len(),
            });
        }
        if !row.weight.is_finite() || row.weight <= 0.0 {
            return Err(AttGtError::InvalidWeight { value: row.weight });
        }
        if !row.outcome.is_finite() {
            return Err(AttGtError::InvalidOutcome { value: row.outcome });
        }
        for covariate in &row.covariates {
            if !covariate.is_finite() {
                return Err(AttGtError::InvalidCovariate { value: *covariate });
            }
        }
        times.insert(row.time);
        if let Some(group) = row.first_treated_time {
            treated_groups.insert(group);
        } else {
            has_never_treated = true;
        }
    }

    if config.att_gt.comparison_group == crate::types::ComparisonGroup::NeverTreated
        && !has_never_treated
    {
        return Err(AttGtError::MissingNeverTreatedGroup);
    }

    Ok((
        times.into_iter().collect::<Vec<_>>(),
        treated_groups.into_iter().collect::<Vec<_>>(),
    ))
}

pub(super) const fn baseline_time_for_pair(
    time: i32,
    group: i32,
    universal_baseline_time: i32,
    config: AttGtConfig,
) -> i32 {
    if time < group && matches!(config.base_period, BasePeriod::Varying) {
        time - 1
    } else {
        universal_baseline_time
    }
}

pub fn missing_pair_cell(rows: &[DrDidRepeatedObservation]) -> Option<&'static str> {
    let (treated_pre, treated_post, control_pre, control_post) = cell_counts(rows);
    [
        (treated_pre == 0, "treated_pre"),
        (treated_post == 0, "treated_post"),
        (control_pre == 0, "control_pre"),
        (control_post == 0, "control_post"),
    ]
    .into_iter()
    .find_map(|(is_missing, label)| if is_missing { Some(label) } else { None })
}

pub fn build_pair_rows(
    observations: &[AttGtDrObservation],
    group: i32,
    time: i32,
    baseline_time: i32,
    config: AttGtConfig,
) -> (Vec<DrDidRepeatedObservation>, Vec<usize>) {
    let mut rows = Vec::with_capacity(observations.len().min(4096));
    let mut indices = Vec::with_capacity(observations.len().min(4096));
    build_pair_rows_into(
        observations,
        group,
        time,
        baseline_time,
        config,
        &mut rows,
        &mut indices,
    );
    (rows, indices)
}

pub fn build_pair_rows_into(
    observations: &[AttGtDrObservation],
    group: i32,
    time: i32,
    baseline_time: i32,
    config: AttGtConfig,
    rows: &mut Vec<DrDidRepeatedObservation>,
    indices: &mut Vec<usize>,
) {
    rows.clear();
    indices.clear();
    for (idx, row) in observations.iter().enumerate() {
        if row.time != baseline_time && row.time != time {
            continue;
        }
        let entry = if row.first_treated_time == Some(group) {
            Some(DrDidRepeatedObservation {
                weight: row.weight,
                covariates: row.covariates.clone(),
                ..DrDidRepeatedObservation::new(
                    DidCell::from_parts(
                        TreatmentGroup::Treated,
                        TimePeriod::from_bool(row.time == time),
                    ),
                    row.outcome,
                )
            })
        } else if super::is_control_for_pair(
            row.first_treated_time,
            // The LATER of the two periods being compared, not `time`. Under a
            // universal base period a pre-treatment cell has baseline > time, and
            // a unit treated in between is already treated when the baseline is
            // read. See the note in `panel_pairs::collect_cell_units` for the
            // measured difference against did 2.5.1.
            time.max(baseline_time),
            config.comparison_group,
            config.anticipation_periods,
        ) {
            Some(DrDidRepeatedObservation {
                weight: row.weight,
                covariates: row.covariates.clone(),
                ..DrDidRepeatedObservation::new(
                    DidCell::from_parts(
                        TreatmentGroup::Control,
                        TimePeriod::from_bool(row.time == time),
                    ),
                    row.outcome,
                )
            })
        } else {
            None
        };

        if let Some(entry) = entry {
            rows.push(entry);
            indices.push(idx);
        }
    }
}

fn cell_counts(rows: &[DrDidRepeatedObservation]) -> (usize, usize, usize, usize) {
    let mut treated_pre = 0_usize;
    let mut treated_post = 0_usize;
    let mut control_pre = 0_usize;
    let mut control_post = 0_usize;

    for row in rows {
        match (row.treated, row.post_period) {
            (true, false) => treated_pre += 1,
            (true, true) => treated_post += 1,
            (false, false) => control_pre += 1,
            (false, true) => control_post += 1,
        }
    }

    (treated_pre, treated_post, control_pre, control_post)
}

pub fn estimate_pair_or(
    rows: &[DrDidRepeatedObservation],
    group: i32,
    time: i32,
    config: AttGtConfig,
    drdid_cfg: DrDidConfig,
) -> Result<PairEstimateWithInfluence, ()> {
    let feature_count = rows.first().map_or(1, |row| row.covariates.len() + 1);
    let observation_count = rows.len();
    let design_matrix = Mat::from_fn(
        observation_count,
        feature_count,
        |row_index, column_index| {
            if column_index == 0 {
                1.0
            } else {
                rows[row_index].covariates[column_index - 1]
            }
        },
    );
    let outcomes = rows.iter().map(|row| row.outcome).collect::<Vec<_>>();

    let (control_pre_idx, control_post_idx, treated_post_n) = split_or_cells(rows);
    let nuisance_predictions = fit_or_nuisance_predictions(
        rows,
        &design_matrix,
        &outcomes,
        &control_pre_idx,
        &control_post_idx,
        drdid_cfg.ridge,
    );

    let sample_weights = rows.iter().map(|row| row.weight).collect::<Vec<_>>();
    let normalized_weights = normalize_weights_to_n(&sample_weights).map_err(|_| ())?;
    let (att, influence_function) = compute_or_att_and_influence(
        rows,
        &outcomes,
        &nuisance_predictions.predicted_pre_outcomes,
        &nuisance_predictions.predicted_post_outcomes,
        &normalized_weights,
    )?;

    let se = standard_error_from_influence(&influence_function);
    let (ci_low, ci_high) = multiplier_bootstrap_ci(
        att,
        &influence_function,
        config.confidence_level,
        drdid_cfg.bootstrap(),
    );

    Ok(PairEstimateWithInfluence {
        estimate: AttGtEstimate {
            group,
            time,
            event_time: time - group,
            att,
            se,
            ci_low,
            ci_high,
            treated_n: treated_post_n,
            control_n: control_post_idx.len(),
            total_weight: rows.iter().map(|row| row.weight).sum::<f64>(),
        },
        influence_function,
    })
}

struct OrNuisancePredictions {
    predicted_pre_outcomes: Vec<f64>,
    predicted_post_outcomes: Vec<f64>,
}

fn split_or_cells(rows: &[DrDidRepeatedObservation]) -> (Vec<usize>, Vec<usize>, usize) {
    let control_pre_idx = rows
        .iter()
        .enumerate()
        .filter_map(|(idx, row)| (!row.treated && !row.post_period).then_some(idx))
        .collect::<Vec<_>>();
    let control_post_idx = rows
        .iter()
        .enumerate()
        .filter_map(|(idx, row)| (!row.treated && row.post_period).then_some(idx))
        .collect::<Vec<_>>();
    let treated_post_n = rows
        .iter()
        .filter(|row| row.treated && row.post_period)
        .count();
    (control_pre_idx, control_post_idx, treated_post_n)
}

fn fit_or_nuisance_predictions(
    rows: &[DrDidRepeatedObservation],
    design_matrix: &Mat<f64>,
    outcomes: &[f64],
    control_pre_idx: &[usize],
    control_post_idx: &[usize],
    ridge: f64,
) -> OrNuisancePredictions {
    let (control_pre_design, control_pre_outcome, control_pre_weights) =
        design_outcome_weights_subset(rows, design_matrix, outcomes, control_pre_idx);
    let (control_post_design, control_post_outcome, control_post_weights) =
        design_outcome_weights_subset(rows, design_matrix, outcomes, control_post_idx);

    let model = LinearOutcome { ridge };
    let control_pre_coefficients = model.fit(
        control_pre_design.as_ref(),
        control_pre_outcome.as_ref(),
        Some(&control_pre_weights),
    );
    let control_post_coefficients = model.fit(
        control_post_design.as_ref(),
        control_post_outcome.as_ref(),
        Some(&control_post_weights),
    );

    OrNuisancePredictions {
        predicted_pre_outcomes: model
            .predict(design_matrix.as_ref(), control_pre_coefficients.as_ref()),
        predicted_post_outcomes: model
            .predict(design_matrix.as_ref(), control_post_coefficients.as_ref()),
    }
}

fn design_outcome_weights_subset(
    rows: &[DrDidRepeatedObservation],
    design_matrix: &Mat<f64>,
    outcomes: &[f64],
    indices: &[usize],
) -> (Mat<f64>, Mat<f64>, Vec<f64>) {
    let design_submatrix = Mat::from_fn(indices.len(), design_matrix.ncols(), |row, col| {
        *design_matrix.get(indices[row], col)
    });
    let outcome_subvector = Mat::from_fn(indices.len(), 1, |row, _| outcomes[indices[row]]);
    let sample_weights = indices
        .iter()
        .map(|idx| rows[*idx].weight)
        .collect::<Vec<_>>();
    (design_submatrix, outcome_subvector, sample_weights)
}

fn compute_or_att_and_influence(
    rows: &[DrDidRepeatedObservation],
    outcomes: &[f64],
    predicted_pre_outcomes: &[f64],
    predicted_post_outcomes: &[f64],
    normalized_weights: &[f64],
) -> Result<(f64, Vec<f64>), ()> {
    let observation_count = rows.len();
    let observation_count_f64 = usize_to_f64(observation_count);

    let mut treated_post_weights = vec![0.0; observation_count];
    let mut treated_pre_weights = vec![0.0; observation_count];
    let mut treated_post_contributions = vec![0.0; observation_count];
    let mut treated_pre_contributions = vec![0.0; observation_count];

    for (
        row,
        outcome,
        predicted_pre_outcome,
        predicted_post_outcome,
        normalized_weight,
        treated_post_weight,
        treated_pre_weight,
        treated_post_contribution,
        treated_pre_contribution,
    ) in izip!(
        rows.iter(),
        outcomes.iter(),
        predicted_pre_outcomes.iter(),
        predicted_post_outcomes.iter(),
        normalized_weights.iter(),
        treated_post_weights.iter_mut(),
        treated_pre_weights.iter_mut(),
        treated_post_contributions.iter_mut(),
        treated_pre_contributions.iter_mut()
    ) {
        if row.treated && row.post_period {
            let residual = outcome - predicted_post_outcome;
            *treated_post_weight = *normalized_weight;
            *treated_post_contribution = *normalized_weight * residual;
        } else if row.treated && !row.post_period {
            let residual = outcome - predicted_pre_outcome;
            *treated_pre_weight = *normalized_weight;
            *treated_pre_contribution = *normalized_weight * residual;
        }
    }

    let mean_treated_post_weight = treated_post_weights.iter().sum::<f64>() / observation_count_f64;
    let mean_treated_pre_weight = treated_pre_weights.iter().sum::<f64>() / observation_count_f64;
    if mean_treated_post_weight <= 0.0 || mean_treated_pre_weight <= 0.0 {
        return Err(());
    }

    let treated_post_mean = (treated_post_contributions.iter().sum::<f64>()
        / observation_count_f64)
        / mean_treated_post_weight;
    let treated_pre_mean = (treated_pre_contributions.iter().sum::<f64>() / observation_count_f64)
        / mean_treated_pre_weight;
    let att = treated_post_mean - treated_pre_mean;

    let influence_function = izip!(
        treated_post_weights.iter(),
        treated_post_contributions.iter(),
        treated_pre_weights.iter(),
        treated_pre_contributions.iter()
    )
    .map(
        |(
            treated_post_weight,
            treated_post_contribution,
            treated_pre_weight,
            treated_pre_contribution,
        )| {
            let treated_post_term = treated_post_mean
                .mul_add(-treated_post_weight, *treated_post_contribution)
                / mean_treated_post_weight;
            let treated_pre_term = treated_pre_mean
                .mul_add(-treated_pre_weight, *treated_pre_contribution)
                / mean_treated_pre_weight;
            treated_post_term - treated_pre_term
        },
    )
    .collect::<Vec<_>>();

    Ok((att, influence_function))
}

pub fn estimate_pair_ipw(
    rows: &[DrDidRepeatedObservation],
    group: i32,
    time: i32,
    config: AttGtConfig,
    drdid_cfg: DrDidConfig,
) -> Result<PairEstimateWithInfluence, ()> {
    let observation_count = rows.len();
    let feature_count = rows.first().map_or(1, |row| row.covariates.len() + 1);

    let mut design_matrix_flat = Vec::with_capacity(observation_count * feature_count);
    let mut treated_indicator = Vec::with_capacity(observation_count);
    let mut post_indicator = Vec::with_capacity(observation_count);
    let mut outcomes = Vec::with_capacity(observation_count);
    let mut sample_weights = Vec::with_capacity(observation_count);

    for row in rows {
        treated_indicator.push(if row.treated { 1.0 } else { 0.0 });
        post_indicator.push(if row.post_period { 1.0 } else { 0.0 });
        outcomes.push(row.outcome);
        sample_weights.push(row.weight);
        design_matrix_flat.push(1.0);
        design_matrix_flat.extend_from_slice(&row.covariates);
    }

    let normalized_weights = normalize_weights_to_n(&sample_weights).map_err(|_| ())?;
    let design_matrix = Mat::from_fn(
        observation_count,
        feature_count,
        |row_index, column_index| design_matrix_flat[row_index * feature_count + column_index],
    );
    let treated_target = Mat::from_fn(observation_count, 1, |row_index, _| {
        treated_indicator[row_index]
    });

    let ps_cfg = PropensityConfig {
        max_iter: u64::try_from(drdid_cfg.max_iter).map_err(|_| ())?,
        tol: drdid_cfg.tol,
        min_weight: drdid_cfg.propensity_clip,
        vstar: 700.0,
    };
    let est = LogisticPS::new(ps_cfg);
    let params = est
        .fit(design_matrix.as_ref(), treated_target.as_ref())
        .map_err(|_| ())?;
    let propensity_scores = logistic_scores(design_matrix.as_ref(), params.beta.as_ref())
        .into_iter()
        .map(|value| value.clamp(drdid_cfg.propensity_clip, 1.0 - drdid_cfg.propensity_clip))
        .collect::<Vec<_>>();

    let moments = repeated_att_moments(RepeatedMomentInputs {
        normalized_weights: &normalized_weights,
        treated: &treated_indicator,
        post_period: &post_indicator,
        propensity: &propensity_scores,
        signal: &outcomes,
    })
    .map_err(|_| ())?;

    let se = standard_error_from_influence(&moments.influence_function);
    let (ci_low, ci_high) = multiplier_bootstrap_ci(
        moments.att,
        &moments.influence_function,
        config.confidence_level,
        drdid_cfg.bootstrap(),
    );

    let treated_n = rows
        .iter()
        .filter(|row| row.treated && row.post_period)
        .count();
    let control_n = rows
        .iter()
        .filter(|row| !row.treated && row.post_period)
        .count();

    Ok(PairEstimateWithInfluence {
        estimate: AttGtEstimate {
            group,
            time,
            event_time: time - group,
            att: moments.att,
            se,
            ci_low,
            ci_high,
            treated_n,
            control_n,
            total_weight: rows.iter().map(|row| row.weight).sum::<f64>(),
        },
        influence_function: moments.influence_function,
    })
}
