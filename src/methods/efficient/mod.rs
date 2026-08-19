//! Baseline no-covariate `Efficient_DiD` estimators.
//!
//! This module implements the first production slice of Chen, Sant'Anna, and
//! Xie (2025): panel-data `ATT(g,t)` estimation under unconditional `PT-All`
//! with never-treated controls. Efficiency is achieved by combining all
//! baseline-specific panel `DiD` estimators for a fixed `(g, t)` using the
//! inverse covariance of their influence functions.
//!
//! Concretely, for each pre-treatment baseline `b < g`, define the baseline-
//! specific panel estimator
//!
//! ```text
//! θ_b(g, t) = E[Y_t - Y_b | G = g] - E[Y_t - Y_b | G = ∞].
//! ```
//!
//! Let `IF_b(g, t)` denote its influence function. The implemented efficient
//! estimator uses the generalized least-squares combination
//!
//! ```text
//! w(g, t) ∝ Ω(g, t)^{-1} 1,
//! ATT_eff(g, t) = Σ_b w_b(g, t) θ_b(g, t),
//! ```
//!
//! where `Ω(g, t) = Cov(IF(g, t))` is the covariance matrix over baseline-
//! specific influence functions.

use std::collections::{BTreeMap, BTreeSet};

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};

use crate::estimators::common::linalg::solve_spd_system;
use crate::inference::{validate_confidence_level, z_score_for_confidence};
use crate::methods::att_gt::pair_estimators::{
    build_pair_rows, missing_pair_cell, prepare_att_gt_dr_inputs,
};
use crate::methods::drdid::repeated::estimate_drdid_repeated_cross_section;
use crate::types::{
    AttGtAggregatedEstimate, AttGtAggregationConfig, AttGtAggregationError,
    AttGtAggregationWeighting, AttGtCalendarEstimate, AttGtCohortEstimate, AttGtDrConfig,
    AttGtDrObservation, AttGtError, AttGtObservation, BasePeriod, EfficientBaselineWeight,
    EfficientDidConfig, EfficientDidDiagnostics, EfficientDidError, EfficientDidEstimate,
    EfficientDidEventTimeEstimate, EfficientDidEventTimeInfluenceOutput,
};
use crate::util::usize_to_f64;

#[derive(Debug, Clone)]
struct UnitSeries {
    first_treated_time: Option<i32>,
    outcomes_by_time: BTreeMap<i32, f64>,
}

#[derive(Debug, Clone)]
struct BaselineSample {
    baseline_time: i32,
    att: f64,
    treated_n: usize,
    control_n: usize,
}

#[derive(Debug, Clone)]
struct BaselineSamples {
    samples: Vec<BaselineSample>,
    influence_matrix_flat: Vec<f64>,
    influence_len: usize,
}

#[derive(Debug, Clone)]
struct EfficientWeightSolution {
    normalized_weights: Vec<f64>,
    raw_precision_solution: Vec<f64>,
    ridge_penalty: Option<f64>,
}

type AggregatedEfficientInfluenceOutput = (BTreeMap<i32, AttGtAggregatedEstimate>, Vec<Vec<f64>>);

/// Estimate efficient baseline-weighted `ATT(g,t)` effects for unconditional
/// panel-data `PT-All` designs with never-treated controls.
///
/// # Errors
/// Returns [`EfficientDidError`] if inputs are invalid, no never-treated group
/// exists, panel identifiers are missing, or the baseline-weighting covariance
/// system is singular.
pub fn estimate_att_gt_efficient(
    observations: &[AttGtObservation],
    config: EfficientDidConfig,
) -> Result<Vec<EfficientDidEstimate>, EfficientDidError> {
    if observations.is_empty() {
        return Err(EfficientDidError::EmptyInput);
    }
    if !validate_confidence_level(config.confidence_level.confidence_level) {
        return Err(EfficientDidError::InvalidConfidenceLevel);
    }

    let panel = build_panel(observations)?;
    let treated_groups = collect_treated_groups(&panel);
    let never_treated_ids = collect_never_treated_ids(&panel);
    if never_treated_ids.is_empty() {
        return Err(EfficientDidError::MissingNeverTreatedGroup);
    }

    let mut estimates = Vec::new();
    for group in treated_groups {
        let pre_times = collect_pre_times(&panel, group);
        if pre_times.is_empty() {
            return Err(EfficientDidError::NoPrePeriods { group });
        }
        let post_times = collect_post_times(&panel, group);
        if post_times.is_empty() {
            return Err(EfficientDidError::NoPostPeriods { group });
        }

        let treated_ids = collect_group_ids(&panel, Some(group));
        for time in post_times {
            estimates.push(estimate_group_time(
                &panel,
                &treated_ids,
                &never_treated_ids,
                group,
                time,
                &pre_times,
                config,
            )?);
        }
    }

    Ok(estimates)
}

/// Estimate covariate-adjusted efficient baseline-weighted `ATT(g,t)` effects
/// by combining baseline-specific DR repeated-cross-section `ATT(g,t)` pairs.
///
/// This extends the baseline GLS weighting logic to the existing
/// covariate-aware `att_gt` row type and DR nuisance-estimation path. It keeps
/// the `att_gt` validation and pair-construction rules intact instead of
/// introducing a second conditional-data pipeline.
///
/// # Errors
/// Returns [`AttGtError`] if `att_gt` inputs/configuration are invalid, if
/// required pair cells are missing and `skip_incomplete_pairs` is disabled, or
/// if no estimable efficient pairs can be formed.
pub fn estimate_att_gt_efficient_dr(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<Vec<EfficientDidEstimate>, AttGtError> {
    let (all_times, treated_groups) = prepare_att_gt_dr_inputs(observations, config)?;
    let mut estimates = Vec::new();

    for group in treated_groups {
        let all_pre_treatment_times = all_times
            .iter()
            .copied()
            .filter(|time| *time < group)
            .collect::<Vec<_>>();
        for &time in &all_times {
            let baselines = baseline_times_for_pair_efficient_dr(
                time,
                group,
                &all_pre_treatment_times,
                config.att_gt,
            );
            if baselines.is_empty() || (baselines.len() == 1 && baselines[0] == time) {
                continue;
            }
            if let Some(estimate) =
                estimate_group_time_efficient_dr(observations, group, time, &baselines, config)?
            {
                estimates.push(estimate);
            }
        }
    }

    if estimates.is_empty() {
        return Err(AttGtError::NoEstimablePairs);
    }
    Ok(estimates)
}

fn baseline_times_for_pair_efficient_dr(
    time: i32,
    group: i32,
    all_pre_treatment_times: &[i32],
    config: crate::types::AttGtConfig,
) -> Vec<i32> {
    if time < group && matches!(config.base_period, BasePeriod::Varying) {
        vec![time - 1]
    } else {
        all_pre_treatment_times.to_vec()
    }
}

fn estimate_group_time_efficient_dr(
    observations: &[AttGtDrObservation],
    group: i32,
    time: i32,
    baselines: &[i32],
    config: AttGtDrConfig,
) -> Result<Option<EfficientDidEstimate>, AttGtError> {
    let baseline_samples =
        collect_dr_baseline_samples(observations, group, time, baselines, config)?;
    if baseline_samples.samples.is_empty() {
        return Ok(None);
    }

    let full_n = observations.len();
    let covariance = influence_covariance_matrix(&baseline_samples);
    let weight_solution =
        efficient_weights(&covariance).map_err(|_| AttGtError::PairEstimationFailure {
            method: "efficient_gls",
            group,
            time,
        })?;
    let weights = &weight_solution.normalized_weights;
    let att = baseline_samples
        .samples
        .iter()
        .zip(weights.iter())
        .map(|(sample, weight)| sample.att * weight)
        .sum::<f64>();
    let mut influence = vec![0.0; full_n];
    combine_influences_into(&baseline_samples, weights, &mut influence);
    let variance = influence.iter().map(|value| value * value).sum::<f64>() / usize_to_f64(full_n);
    let se = (variance / usize_to_f64(full_n)).sqrt();
    let z = z_score_for_confidence(config.att_gt.confidence_level.confidence_level);
    let margin = z * se;
    let reference_counts = &baseline_samples.samples[0];

    Ok(Some(EfficientDidEstimate {
        group,
        time,
        event_time: time - group,
        att,
        se,
        ci_low: att - margin,
        ci_high: att + margin,
        treated_n: reference_counts.treated_n,
        control_n: reference_counts.control_n,
        baseline_weights: baseline_samples
            .samples
            .iter()
            .zip(weights.iter())
            .map(|(sample, weight)| EfficientBaselineWeight {
                baseline_time: sample.baseline_time,
                att: sample.att,
                weight: *weight,
                treated_n: sample.treated_n,
                control_n: sample.control_n,
            })
            .collect(),
        influence_function: influence,
        diagnostics: EfficientDidDiagnostics {
            baseline_covariance: covariance_matrix_rows(
                &covariance,
                baseline_samples.samples.len(),
            ),
            raw_precision_solution: weight_solution.raw_precision_solution,
            ridge_penalty: weight_solution.ridge_penalty,
        },
    }))
}

fn collect_dr_baseline_samples(
    observations: &[AttGtDrObservation],
    group: i32,
    time: i32,
    baselines: &[i32],
    config: AttGtDrConfig,
) -> Result<BaselineSamples, AttGtError> {
    let full_n = observations.len();
    let mut baseline_samples = Vec::new();
    let mut influence_matrix_flat = Vec::with_capacity(baselines.len() * full_n);
    for baseline_time in baselines {
        let (pair_rows, pair_indices) =
            build_pair_rows(observations, group, time, *baseline_time, config.att_gt);
        if let Some(cell) = missing_pair_cell(&pair_rows) {
            if config.att_gt.skip_incomplete_pairs {
                continue;
            }
            return Err(AttGtError::MissingCell {
                group,
                time,
                baseline_time: *baseline_time,
                cell,
            });
        }

        let estimate =
            estimate_drdid_repeated_cross_section(&pair_rows, config.drdid).map_err(|_| {
                AttGtError::PairEstimationFailure {
                    method: "dr",
                    group,
                    time,
                }
            })?;
        if estimate.influence_function.len() != pair_rows.len() {
            return Err(AttGtError::InfluenceLengthMismatch {
                method: "dr",
                group,
                time,
                expected: pair_rows.len(),
                actual: estimate.influence_function.len(),
            });
        }

        let row_start = influence_matrix_flat.len();
        influence_matrix_flat.resize(row_start + full_n, 0.0);
        let aligned = &mut influence_matrix_flat[row_start..row_start + full_n];
        for (local_idx, global_idx) in pair_indices.iter().enumerate() {
            aligned[*global_idx] = estimate.influence_function[local_idx];
        }
        baseline_samples.push(BaselineSample {
            baseline_time: *baseline_time,
            att: estimate.att,
            treated_n: estimate.treated_n,
            control_n: estimate.control_n,
        });
    }
    Ok(BaselineSamples {
        samples: baseline_samples,
        influence_matrix_flat,
        influence_len: full_n,
    })
}

fn build_panel(
    observations: &[AttGtObservation],
) -> Result<BTreeMap<i64, UnitSeries>, EfficientDidError> {
    let mut panel = BTreeMap::<i64, UnitSeries>::new();
    for row in observations {
        if !row.weight.is_finite() || row.weight <= 0.0 {
            return Err(EfficientDidError::InvalidWeight { value: row.weight });
        }
        if !row.outcome.is_finite() {
            return Err(EfficientDidError::InvalidOutcome { value: row.outcome });
        }
        let unit_id = row.unit_id.ok_or(EfficientDidError::MissingUnitId)?;
        let entry = panel.entry(unit_id).or_insert_with(|| UnitSeries {
            first_treated_time: row.first_treated_time,
            outcomes_by_time: BTreeMap::new(),
        });
        entry.first_treated_time = row.first_treated_time;
        entry.outcomes_by_time.insert(row.time, row.outcome);
    }
    Ok(panel)
}

fn collect_treated_groups(panel: &BTreeMap<i64, UnitSeries>) -> Vec<i32> {
    panel
        .values()
        .filter_map(|series| series.first_treated_time)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn collect_never_treated_ids(panel: &BTreeMap<i64, UnitSeries>) -> Vec<i64> {
    collect_group_ids(panel, None)
}

fn collect_group_ids(panel: &BTreeMap<i64, UnitSeries>, group: Option<i32>) -> Vec<i64> {
    panel
        .iter()
        .filter_map(|(unit_id, series)| (series.first_treated_time == group).then_some(*unit_id))
        .collect()
}

fn collect_pre_times(panel: &BTreeMap<i64, UnitSeries>, group: i32) -> Vec<i32> {
    panel
        .values()
        .filter(|series| series.first_treated_time == Some(group))
        .flat_map(|series| series.outcomes_by_time.keys().copied())
        .filter(|time| *time < group)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn collect_post_times(panel: &BTreeMap<i64, UnitSeries>, group: i32) -> Vec<i32> {
    panel
        .values()
        .filter(|series| series.first_treated_time == Some(group))
        .flat_map(|series| series.outcomes_by_time.keys().copied())
        .filter(|time| *time >= group)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn estimate_group_time(
    panel: &BTreeMap<i64, UnitSeries>,
    treated_ids: &[i64],
    control_ids: &[i64],
    group: i32,
    time: i32,
    baselines: &[i32],
    config: EfficientDidConfig,
) -> Result<EfficientDidEstimate, EfficientDidError> {
    let treated_panel = filter_complete_ids(panel, treated_ids, time, baselines);
    if treated_panel.is_empty() {
        return Err(EfficientDidError::MissingTreatedPanel { group, time });
    }
    let control_panel = filter_complete_ids(panel, control_ids, time, baselines);
    if control_panel.is_empty() {
        return Err(EfficientDidError::MissingControlPanel { group, time });
    }

    let total_n = treated_panel.len() + control_panel.len();
    let mut baseline_rows = Vec::with_capacity(baselines.len());
    let mut influence_matrix_flat = vec![0.0; baselines.len() * total_n];
    for (baseline_idx, baseline_time) in baselines.iter().copied().enumerate() {
        let row_start = baseline_idx * total_n;
        let influence_row = &mut influence_matrix_flat[row_start..row_start + total_n];
        baseline_rows.push(build_baseline_sample(
            panel,
            &treated_panel,
            &control_panel,
            time,
            baseline_time,
            influence_row,
        ));
    }
    let baseline_samples = BaselineSamples {
        samples: baseline_rows,
        influence_matrix_flat,
        influence_len: total_n,
    };

    let covariance = influence_covariance_matrix(&baseline_samples);
    let weight_solution = efficient_weights(&covariance)?;
    let weights = &weight_solution.normalized_weights;
    let att = baseline_samples
        .samples
        .iter()
        .zip(weights.iter())
        .map(|(sample, weight)| sample.att * weight)
        .sum::<f64>();
    let n_total = treated_panel.len() + control_panel.len();
    let mut influence = vec![0.0; n_total];
    combine_influences_into(&baseline_samples, weights, &mut influence);
    let variance = influence.iter().map(|value| value * value).sum::<f64>() / usize_to_f64(n_total);
    let se = (variance / usize_to_f64(n_total)).sqrt();
    let z = z_score_for_confidence(config.confidence_level.confidence_level);
    let margin = z * se;

    Ok(EfficientDidEstimate {
        group,
        time,
        event_time: time - group,
        att,
        se,
        ci_low: att - margin,
        ci_high: att + margin,
        treated_n: treated_panel.len(),
        control_n: control_panel.len(),
        baseline_weights: baseline_samples
            .samples
            .iter()
            .zip(weights.iter())
            .map(|(sample, weight)| EfficientBaselineWeight {
                baseline_time: sample.baseline_time,
                att: sample.att,
                weight: *weight,
                treated_n: sample.treated_n,
                control_n: sample.control_n,
            })
            .collect(),
        influence_function: influence,
        diagnostics: EfficientDidDiagnostics {
            baseline_covariance: covariance_matrix_rows(
                &covariance,
                baseline_samples.samples.len(),
            ),
            raw_precision_solution: weight_solution.raw_precision_solution,
            ridge_penalty: weight_solution.ridge_penalty,
        },
    })
}

/// Aggregate efficient `ATT(g,t)` estimates by event time.
///
/// # Errors
/// Returns [`AttGtAggregationError`] if estimates/configuration are invalid.
pub fn aggregate_efficient_att_gt_event_time(
    estimates: &[EfficientDidEstimate],
    config: AttGtAggregationConfig,
) -> Result<Vec<EfficientDidEventTimeEstimate>, AttGtAggregationError> {
    aggregate_efficient_att_gt_event_time_with_influence(estimates, config)
        .map(|output| output.estimates)
}

/// Aggregate efficient `ATT(g,t)` estimates by event time while preserving
/// aligned influence vectors.
///
/// # Errors
/// Returns [`AttGtAggregationError`] if estimates/configuration are invalid or
/// influence vectors are inconsistent.
pub fn aggregate_efficient_att_gt_event_time_with_influence(
    estimates: &[EfficientDidEstimate],
    config: AttGtAggregationConfig,
) -> Result<EfficientDidEventTimeInfluenceOutput, AttGtAggregationError> {
    aggregate_efficient_with_influence(estimates, config, |estimate| estimate.event_time).map(
        |(grouped, influence_functions)| EfficientDidEventTimeInfluenceOutput {
            estimates: grouped
                .into_iter()
                .map(|(event_time, summary)| EfficientDidEventTimeEstimate {
                    event_time,
                    summary,
                })
                .collect(),
            influence_functions,
        },
    )
}

/// Aggregate efficient `ATT(g,t)` estimates by treatment cohort.
///
/// # Errors
/// Returns [`AttGtAggregationError`] if estimates/configuration are invalid.
pub fn aggregate_efficient_att_gt_by_cohort(
    estimates: &[EfficientDidEstimate],
    config: AttGtAggregationConfig,
) -> Result<Vec<AttGtCohortEstimate>, AttGtAggregationError> {
    aggregate_efficient_by_key(estimates, config, |estimate| estimate.group).map(|grouped| {
        grouped
            .into_iter()
            .map(|(group, summary)| AttGtCohortEstimate { group, summary })
            .collect()
    })
}

/// Aggregate efficient `ATT(g,t)` estimates by calendar time.
///
/// # Errors
/// Returns [`AttGtAggregationError`] if estimates/configuration are invalid.
pub fn aggregate_efficient_att_gt_by_calendar_time(
    estimates: &[EfficientDidEstimate],
    config: AttGtAggregationConfig,
) -> Result<Vec<AttGtCalendarEstimate>, AttGtAggregationError> {
    aggregate_efficient_by_key(estimates, config, |estimate| estimate.time).map(|grouped| {
        grouped
            .into_iter()
            .map(|(time, summary)| AttGtCalendarEstimate { time, summary })
            .collect()
    })
}

/// Aggregate efficient `ATT(g,t)` estimates into one overall summary.
///
/// # Errors
/// Returns [`AttGtAggregationError`] if estimates/configuration are invalid.
pub fn aggregate_efficient_att_gt_overall(
    estimates: &[EfficientDidEstimate],
    config: AttGtAggregationConfig,
) -> Result<AttGtAggregatedEstimate, AttGtAggregationError> {
    aggregate_efficient_by_key(estimates, config, |_| 0)
        .map(|grouped| grouped.into_values().next())
        .and_then(|summary| summary.ok_or(AttGtAggregationError::EmptyInput))
}

fn aggregate_efficient_by_key<F>(
    estimates: &[EfficientDidEstimate],
    config: AttGtAggregationConfig,
    key_fn: F,
) -> Result<BTreeMap<i32, AttGtAggregatedEstimate>, AttGtAggregationError>
where
    F: Fn(&EfficientDidEstimate) -> i32,
{
    aggregate_efficient_with_influence(estimates, config, key_fn).map(|(grouped, _)| grouped)
}

fn aggregate_efficient_with_influence<F>(
    estimates: &[EfficientDidEstimate],
    config: AttGtAggregationConfig,
    key_fn: F,
) -> Result<AggregatedEfficientInfluenceOutput, AttGtAggregationError>
where
    F: Fn(&EfficientDidEstimate) -> i32,
{
    validate_efficient_aggregation_inputs(estimates, config)?;
    let influence_len = estimates
        .first()
        .map_or(0, |estimate| estimate.influence_function.len());

    let mut grouped = BTreeMap::<i32, Vec<&EfficientDidEstimate>>::new();
    for estimate in estimates {
        grouped.entry(key_fn(estimate)).or_default().push(estimate);
    }

    let z = z_score_for_confidence(config.confidence_level.confidence_level);
    let mut grouped_estimates = BTreeMap::new();
    let mut grouped_influence = Vec::with_capacity(grouped.len());
    let mut influence_scratch = vec![0.0; influence_len];
    for (key, bucket) in grouped {
        let component_count = bucket.len();
        let raw_total = bucket
            .iter()
            .map(|estimate| aggregation_weight(estimate, config.weighting))
            .sum::<f64>();
        let mut estimate = 0.0;
        let mut variance = 0.0;
        influence_scratch.fill(0.0);
        let mut total_weight = 0.0;
        for component in bucket {
            let normalized_weight = aggregation_weight(component, config.weighting) / raw_total;
            estimate = normalized_weight.mul_add(component.att, estimate);
            variance = (normalized_weight * normalized_weight * component.se)
                .mul_add(component.se, variance);
            total_weight += aggregation_weight(component, AttGtAggregationWeighting::ByTotalWeight);
            for (destination, source) in influence_scratch
                .iter_mut()
                .zip(&component.influence_function)
            {
                *destination = normalized_weight.mul_add(*source, *destination);
            }
        }
        let se = variance.sqrt();
        let margin = z * se;
        grouped_estimates.insert(
            key,
            AttGtAggregatedEstimate {
                estimate,
                se,
                ci_low: estimate - margin,
                ci_high: estimate + margin,
                components: component_count,
                total_weight,
            },
        );
        grouped_influence.push(influence_scratch.clone());
    }
    Ok((grouped_estimates, grouped_influence))
}

fn filter_complete_ids(
    panel: &BTreeMap<i64, UnitSeries>,
    unit_ids: &[i64],
    time: i32,
    baselines: &[i32],
) -> Vec<i64> {
    unit_ids
        .iter()
        .copied()
        .filter(|unit_id| {
            panel[unit_id].outcomes_by_time.contains_key(&time)
                && baselines
                    .iter()
                    .all(|baseline| panel[unit_id].outcomes_by_time.contains_key(baseline))
        })
        .collect()
}

fn build_baseline_sample(
    panel: &BTreeMap<i64, UnitSeries>,
    treated_ids: &[i64],
    control_ids: &[i64],
    time: i32,
    baseline_time: i32,
    influence_out: &mut [f64],
) -> BaselineSample {
    debug_assert_eq!(influence_out.len(), treated_ids.len() + control_ids.len());
    let mut treated_sum = 0.0;
    let mut control_sum = 0.0;
    for (idx, unit_id) in treated_ids.iter().enumerate() {
        let series = &panel[unit_id];
        let delta = series.outcomes_by_time[&time] - series.outcomes_by_time[&baseline_time];
        influence_out[idx] = delta;
        treated_sum += delta;
    }
    for (idx, unit_id) in control_ids.iter().enumerate() {
        let series = &panel[unit_id];
        let delta = series.outcomes_by_time[&time] - series.outcomes_by_time[&baseline_time];
        influence_out[treated_ids.len() + idx] = delta;
        control_sum += delta;
    }
    let treated_mean = treated_sum / usize_to_f64(treated_ids.len());
    let control_mean = control_sum / usize_to_f64(control_ids.len());
    let total_n = treated_ids.len() + control_ids.len();
    let treated_share = usize_to_f64(treated_ids.len()) / usize_to_f64(total_n);
    let control_share = usize_to_f64(control_ids.len()) / usize_to_f64(total_n);
    for value in &mut influence_out[..treated_ids.len()] {
        *value = (*value - treated_mean) / treated_share;
    }
    for value in &mut influence_out[treated_ids.len()..] {
        *value = -(*value - control_mean) / control_share;
    }

    BaselineSample {
        baseline_time,
        att: treated_mean - control_mean,
        treated_n: treated_ids.len(),
        control_n: control_ids.len(),
    }
}

fn influence_covariance_matrix(samples: &BaselineSamples) -> Vec<f64> {
    let dimension = samples.samples.len();
    let sample_len = samples.influence_len;
    let if_matrix =
        MatRef::from_row_major_slice(&samples.influence_matrix_flat, dimension, sample_len);
    let mut covariance = Mat::<f64>::zeros(dimension, dimension);
    matmul(
        covariance.as_mut(),
        Accum::Replace,
        if_matrix,
        if_matrix.transpose(),
        1.0 / usize_to_f64(sample_len),
        Par::Seq,
    );
    let mut out = vec![0.0; dimension * dimension];
    for row in 0..dimension {
        for col in 0..dimension {
            out[row * dimension + col] = covariance[(row, col)];
        }
    }
    out
}

fn efficient_weights(covariance: &[f64]) -> Result<EfficientWeightSolution, EfficientDidError> {
    let dimension = checked_square_dimension(covariance.len()).expect("square covariance");
    let covariance_scale = covariance.iter().map(|value| value.abs()).sum::<f64>();
    if covariance_scale <= f64::EPSILON {
        let equal_weight = 1.0 / usize_to_f64(dimension);
        return Ok(EfficientWeightSolution {
            normalized_weights: vec![equal_weight; dimension],
            raw_precision_solution: vec![1.0; dimension],
            ridge_penalty: None,
        });
    }
    let ones = vec![1.0; dimension];
    let (mut weights, ridge_penalty) = solve_with_ridge_fallback(covariance, &ones)
        .ok_or(EfficientDidError::SingularWeightingSystem)?;
    let raw_precision_solution = weights.clone();
    let normalizer = weights.iter().sum::<f64>();
    if !normalizer.is_finite() || normalizer.abs() < f64::EPSILON {
        return Err(EfficientDidError::SingularWeightingSystem);
    }
    for weight in &mut weights {
        *weight /= normalizer;
    }
    Ok(EfficientWeightSolution {
        normalized_weights: weights,
        raw_precision_solution,
        ridge_penalty,
    })
}

fn solve_with_ridge_fallback(covariance: &[f64], rhs: &[f64]) -> Option<(Vec<f64>, Option<f64>)> {
    if let Ok(solution) = solve_spd_system(covariance, rhs) {
        return Some((solution, None));
    }

    let dimension = rhs.len();
    let mean_diagonal = (0..dimension)
        .map(|index| covariance[index * dimension + index].abs())
        .sum::<f64>()
        / usize_to_f64(dimension).max(1.0);

    let mut ridge_scale = 1e-8 * mean_diagonal;
    let mut regularized = covariance.to_vec();
    for _ in 0..6 {
        regularized.copy_from_slice(covariance);
        for index in 0..dimension {
            regularized[index * dimension + index] += ridge_scale;
        }
        if let Ok(solution) = solve_spd_system(&regularized, rhs) {
            return Some((solution, Some(ridge_scale)));
        }
        ridge_scale *= 100.0;
    }
    None
}

fn covariance_matrix_rows(covariance: &[f64], dimension: usize) -> Vec<Vec<f64>> {
    covariance.chunks(dimension).map(<[f64]>::to_vec).collect()
}

fn validate_efficient_aggregation_inputs(
    estimates: &[EfficientDidEstimate],
    config: AttGtAggregationConfig,
) -> Result<(), AttGtAggregationError> {
    if estimates.is_empty() {
        return Err(AttGtAggregationError::EmptyInput);
    }
    if !validate_confidence_level(config.confidence_level.confidence_level) {
        return Err(AttGtAggregationError::InvalidConfidenceLevel);
    }
    let influence_len = estimates
        .first()
        .map_or(0, |estimate| estimate.influence_function.len());
    if influence_len == 0 {
        return Err(AttGtAggregationError::EmptyInput);
    }
    for estimate in estimates {
        if !estimate.att.is_finite() {
            return Err(AttGtAggregationError::InvalidEstimate);
        }
        if !estimate.se.is_finite() || estimate.se < 0.0 {
            return Err(AttGtAggregationError::InvalidSe);
        }
        if estimate.influence_function.len() != influence_len {
            return Err(AttGtAggregationError::EmptyInput);
        }
        if estimate
            .influence_function
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(AttGtAggregationError::EmptyInput);
        }
    }
    Ok(())
}

fn aggregation_weight(
    estimate: &EfficientDidEstimate,
    weighting: AttGtAggregationWeighting,
) -> f64 {
    match weighting {
        AttGtAggregationWeighting::Equal => 1.0,
        AttGtAggregationWeighting::ByTreatedCount => usize_to_f64(estimate.treated_n),
        AttGtAggregationWeighting::ByTotalWeight => {
            usize_to_f64(estimate.treated_n + estimate.control_n)
        }
    }
}

fn combine_influences_into(samples: &BaselineSamples, weights: &[f64], out: &mut [f64]) {
    debug_assert_eq!(samples.samples.len(), weights.len());
    debug_assert_eq!(samples.influence_len, out.len());
    out.fill(0.0);
    for (baseline_idx, weight) in weights.iter().copied().enumerate() {
        let row_start = baseline_idx * samples.influence_len;
        let row = &samples.influence_matrix_flat[row_start..row_start + samples.influence_len];
        for (combined_value, influence_value) in out.iter_mut().zip(row.iter()) {
            *combined_value = weight.mul_add(*influence_value, *combined_value);
        }
    }
}

fn checked_square_dimension(value: usize) -> Option<usize> {
    let mut candidate = 0usize;
    while candidate.saturating_mul(candidate) < value {
        candidate = candidate.saturating_add(1);
    }
    (candidate.saturating_mul(candidate) == value).then_some(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct SyntheticEstimateSpec {
        group: i32,
        time: i32,
        event_time: i32,
        att: f64,
        se: f64,
        treated_n: usize,
        control_n: usize,
        influence_function: Vec<f64>,
    }

    fn panel_row(unit_id: i64, group: Option<i32>, time: i32, outcome: f64) -> AttGtObservation {
        AttGtObservation {
            unit_id: Some(unit_id),
            first_treated_time: group,
            time,
            outcome,
            weight: 1.0,
        }
    }

    #[test]
    fn efficient_att_gt_recovers_constant_effect_and_weights_sum_to_one() {
        let mut rows = Vec::new();
        for unit_id in 0_i64..40 {
            let group = if unit_id < 20 { Some(3) } else { None };
            for time in 1..=4 {
                let base = usize_to_f64(usize::try_from(unit_id + i64::from(time)).expect("fits"));
                let effect = if unit_id < 20 && time >= 3 { 2.0 } else { 0.0 };
                rows.push(panel_row(unit_id, group, time, base + effect));
            }
        }

        let estimates = estimate_att_gt_efficient(&rows, EfficientDidConfig::default()).unwrap();
        let target = estimates
            .iter()
            .find(|estimate| estimate.group == 3 && estimate.time == 4)
            .unwrap();
        assert!((target.att - 2.0).abs() < 1e-10);
        let weight_sum = target
            .baseline_weights
            .iter()
            .map(|weight| weight.weight)
            .sum::<f64>();
        assert!((weight_sum - 1.0).abs() < 1e-10);
        assert_eq!(target.baseline_weights.len(), 2);
        assert_eq!(target.baseline_weights[0].treated_n, 20);
        assert_eq!(target.baseline_weights[0].control_n, 20);
        assert_eq!(target.diagnostics.baseline_covariance.len(), 2);
        assert_eq!(target.diagnostics.raw_precision_solution.len(), 2);
    }

    #[test]
    fn efficient_att_gt_uses_equal_weights_when_covariance_is_degenerate() {
        let mut rows = Vec::new();
        for unit_id in 0_i64..12 {
            let group = if unit_id < 6 { Some(3) } else { None };
            for time in 1..=4 {
                let baseline = if unit_id < 6 { 10.0 } else { 20.0 };
                let effect = if unit_id < 6 && time >= 3 { 3.0 } else { 0.0 };
                rows.push(panel_row(unit_id, group, time, baseline + effect));
            }
        }

        let estimates = estimate_att_gt_efficient(&rows, EfficientDidConfig::default()).unwrap();
        let target = estimates
            .iter()
            .find(|estimate| estimate.group == 3 && estimate.time == 4)
            .unwrap();
        assert_eq!(target.diagnostics.ridge_penalty, None);
        for baseline in &target.baseline_weights {
            assert!((baseline.weight - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn efficient_event_time_aggregation_preserves_influence_alignment() {
        let mut rows = Vec::new();
        for unit_id in 0_i64..60 {
            let group = match unit_id {
                0..=19 => Some(3),
                20..=39 => Some(4),
                _ => None,
            };
            for time in 1..=5 {
                let baseline = usize_to_f64(usize::try_from(unit_id).expect("fits")) * 0.1;
                let effect = match group {
                    Some(3) if time >= 3 => 1.0,
                    Some(4) if time >= 4 => 2.0,
                    _ => 0.0,
                };
                rows.push(panel_row(
                    unit_id,
                    group,
                    time,
                    baseline + usize_to_f64(usize::try_from(time).expect("positive time")) + effect,
                ));
            }
        }

        let efficient = estimate_att_gt_efficient(&rows, EfficientDidConfig::default()).unwrap();
        let aggregated = aggregate_efficient_att_gt_event_time_with_influence(
            &efficient,
            AttGtAggregationConfig::default(),
        )
        .unwrap();

        let event_zero = aggregated
            .estimates
            .iter()
            .find(|estimate| estimate.event_time == 0)
            .unwrap();
        assert_eq!(event_zero.summary.components, 2);
        assert_eq!(
            aggregated.influence_functions.len(),
            aggregated.estimates.len()
        );
        assert_eq!(
            aggregated.influence_functions[0].len(),
            efficient[0].influence_function.len()
        );
    }

    fn synthetic_estimate(spec: SyntheticEstimateSpec) -> EfficientDidEstimate {
        EfficientDidEstimate {
            group: spec.group,
            time: spec.time,
            event_time: spec.event_time,
            att: spec.att,
            se: spec.se,
            ci_low: 1.96f64.mul_add(-spec.se, spec.att),
            ci_high: 1.96f64.mul_add(spec.se, spec.att),
            treated_n: spec.treated_n,
            control_n: spec.control_n,
            baseline_weights: Vec::new(),
            influence_function: spec.influence_function,
            diagnostics: EfficientDidDiagnostics {
                baseline_covariance: Vec::new(),
                raw_precision_solution: Vec::new(),
                ridge_penalty: None,
            },
        }
    }

    #[test]
    fn efficient_cohort_aggregation_groups_by_treatment_time() {
        let estimates = vec![
            synthetic_estimate(SyntheticEstimateSpec {
                group: 3,
                time: 3,
                event_time: 0,
                att: 1.0,
                se: 0.1,
                treated_n: 10,
                control_n: 20,
                influence_function: vec![0.1, 0.2, 0.3],
            }),
            synthetic_estimate(SyntheticEstimateSpec {
                group: 3,
                time: 4,
                event_time: 1,
                att: 3.0,
                se: 0.2,
                treated_n: 30,
                control_n: 40,
                influence_function: vec![0.3, 0.4, 0.5],
            }),
            synthetic_estimate(SyntheticEstimateSpec {
                group: 4,
                time: 4,
                event_time: 0,
                att: 2.0,
                se: 0.15,
                treated_n: 20,
                control_n: 50,
                influence_function: vec![0.2, 0.3, 0.4],
            }),
        ];

        let aggregated =
            aggregate_efficient_att_gt_by_cohort(&estimates, AttGtAggregationConfig::default())
                .unwrap();

        assert_eq!(aggregated.len(), 2);
        assert_eq!(aggregated[0].group, 3);
        assert_eq!(aggregated[0].summary.components, 2);
        assert!((aggregated[0].summary.estimate - 2.5).abs() < 1e-10);
        assert!((aggregated[0].summary.total_weight - 100.0).abs() < 1e-10);
        assert_eq!(aggregated[1].group, 4);
        assert!((aggregated[1].summary.estimate - 2.0).abs() < 1e-10);
    }

    #[test]
    fn efficient_calendar_aggregation_groups_by_time() {
        let estimates = vec![
            synthetic_estimate(SyntheticEstimateSpec {
                group: 3,
                time: 4,
                event_time: 1,
                att: 1.5,
                se: 0.1,
                treated_n: 20,
                control_n: 30,
                influence_function: vec![0.1, 0.0],
            }),
            synthetic_estimate(SyntheticEstimateSpec {
                group: 4,
                time: 4,
                event_time: 0,
                att: 2.5,
                se: 0.2,
                treated_n: 40,
                control_n: 50,
                influence_function: vec![0.0, 0.1],
            }),
        ];

        let aggregated = aggregate_efficient_att_gt_by_calendar_time(
            &estimates,
            AttGtAggregationConfig::default(),
        )
        .unwrap();

        assert_eq!(aggregated.len(), 1);
        assert_eq!(aggregated[0].time, 4);
        assert_eq!(aggregated[0].summary.components, 2);
        assert!((aggregated[0].summary.estimate - (13.0 / 6.0)).abs() < 1e-10);
        assert!((aggregated[0].summary.total_weight - 140.0).abs() < 1e-10);
    }

    #[test]
    fn efficient_overall_aggregation_combines_all_components() {
        let estimates = vec![
            synthetic_estimate(SyntheticEstimateSpec {
                group: 3,
                time: 3,
                event_time: 0,
                att: 1.0,
                se: 0.1,
                treated_n: 10,
                control_n: 20,
                influence_function: vec![0.1, 0.2],
            }),
            synthetic_estimate(SyntheticEstimateSpec {
                group: 4,
                time: 5,
                event_time: 1,
                att: 3.0,
                se: 0.2,
                treated_n: 30,
                control_n: 40,
                influence_function: vec![0.3, 0.4],
            }),
        ];

        let overall =
            aggregate_efficient_att_gt_overall(&estimates, AttGtAggregationConfig::default())
                .unwrap();

        assert_eq!(overall.components, 2);
        assert!((overall.estimate - 2.5).abs() < 1e-10);
        assert!((overall.total_weight - 100.0).abs() < 1e-10);
    }

    #[test]
    fn efficient_dr_recovers_constant_effect_with_covariates() {
        let mut rows = Vec::new();
        for time in 1..=4 {
            for replicate in 0_i32..20 {
                let x = if replicate % 2 == 0 { 0.0 } else { 1.0 };
                let control_outcome = x + f64::from(time);
                rows.push(AttGtDrObservation {
                    // Each replicate is a unit followed across all four periods;
                    // controls and treated are numbered in disjoint ranges.
                    unit_id: Some(i64::from(replicate)),
                    first_treated_time: None,
                    time,
                    outcome: control_outcome,
                    weight: 1.0,
                    covariates: vec![x],
                });

                let treated_outcome = control_outcome + if time >= 3 { 2.0 } else { 0.0 };
                rows.push(AttGtDrObservation {
                    unit_id: Some(i64::from(replicate) + 1_000),
                    first_treated_time: Some(3),
                    time,
                    outcome: treated_outcome,
                    weight: 1.0,
                    covariates: vec![x],
                });
            }
        }

        let estimates = estimate_att_gt_efficient_dr(&rows, AttGtDrConfig::default()).unwrap();
        let target = estimates
            .iter()
            .find(|estimate| estimate.group == 3 && estimate.time == 4)
            .unwrap();
        let placebo = estimates
            .iter()
            .find(|estimate| estimate.group == 3 && estimate.time == 2)
            .unwrap();

        assert!((target.att - 2.0).abs() < 1e-8);
        assert!(placebo.event_time < 0);
        assert!(placebo.att.abs() < 1e-8);
        assert_eq!(placebo.baseline_weights.len(), 1);
        assert_eq!(target.baseline_weights.len(), 2);
        let weight_sum = target
            .baseline_weights
            .iter()
            .map(|weight| weight.weight)
            .sum::<f64>();
        assert!((weight_sum - 1.0).abs() < 1e-10);
        assert_eq!(target.influence_function.len(), rows.len());
    }
}
