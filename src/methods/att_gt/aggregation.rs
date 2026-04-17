use std::collections::BTreeMap;

use crate::inference::{validate_confidence_level, z_score_for_confidence};
use crate::types::{
    AttGtAggregatedEstimate, AttGtAggregationConfig, AttGtAggregationError,
    AttGtAggregationWeighting, AttGtCalendarEstimate, AttGtCohortEstimate, AttGtEstimate,
    AttGtEventTimeEstimate, AttGtEventTimeInfluenceOutput,
};
use crate::util::usize_to_f64;

/// Aggregate ATT(g,t) estimates by event-time.
///
/// # Errors
///
/// Returns [`AttGtAggregationError`] when estimates/configuration are invalid.
pub fn aggregate_att_gt_event_time(
    estimates: &[AttGtEstimate],
    config: AttGtAggregationConfig,
) -> Result<Vec<AttGtEventTimeEstimate>, AttGtAggregationError> {
    aggregate_by_key(estimates, config, |est| est.event_time).map(|grouped| {
        grouped
            .into_iter()
            .map(|(event_time, summary)| AttGtEventTimeEstimate {
                event_time,
                summary,
            })
            .collect()
    })
}

/// Aggregate ATT(g,t) estimates by event-time while preserving aligned
/// influence-function vectors for joint-process inference.
///
/// # Errors
///
/// Returns [`AttGtAggregationError`] when estimates/configuration are invalid or
/// when the influence-function matrix is inconsistent with the estimate vector.
pub fn aggregate_att_gt_event_time_with_influence(
    estimates: &[AttGtEstimate],
    influence_functions: &[Vec<f64>],
    config: AttGtAggregationConfig,
) -> Result<AttGtEventTimeInfluenceOutput, AttGtAggregationError> {
    validate_aggregation_inputs(estimates, config)?;
    if influence_functions.len() != estimates.len() {
        return Err(AttGtAggregationError::EmptyInput);
    }
    let n = influence_functions.first().map_or(0, Vec::len);
    if n == 0 {
        return Err(AttGtAggregationError::EmptyInput);
    }
    if influence_functions
        .iter()
        .any(|influence| influence.len() != n)
    {
        return Err(AttGtAggregationError::EmptyInput);
    }

    let mut grouped = BTreeMap::<i32, Vec<usize>>::new();
    for (idx, estimate) in estimates.iter().enumerate() {
        grouped.entry(estimate.event_time).or_default().push(idx);
    }

    let z = z_score_for_confidence(config.confidence_level.confidence_level);
    let mut aggregated_estimates = Vec::with_capacity(grouped.len());
    let mut aggregated_influence = Vec::with_capacity(grouped.len());
    for (event_time, bucket) in grouped {
        let component_count = bucket.len();
        let raw_total = bucket
            .iter()
            .map(|idx| raw_weight(&estimates[*idx], config.weighting))
            .sum::<f64>();
        let mut estimate = 0.0;
        let mut variance = 0.0;
        let mut total_influence = vec![0.0; n];
        let mut total_weight = 0.0;
        for idx in bucket {
            let component = &estimates[idx];
            let normalized_weight = raw_weight(component, config.weighting) / raw_total;
            estimate = normalized_weight.mul_add(component.att, estimate);
            variance = (normalized_weight * normalized_weight * component.se)
                .mul_add(component.se, variance);
            total_weight += component.total_weight;
            for (dst, src) in total_influence.iter_mut().zip(&influence_functions[idx]) {
                *dst = normalized_weight.mul_add(*src, *dst);
            }
        }
        let se = variance.sqrt();
        let margin = z * se;
        aggregated_estimates.push(AttGtEventTimeEstimate {
            event_time,
            summary: AttGtAggregatedEstimate {
                estimate,
                se,
                ci_low: estimate - margin,
                ci_high: estimate + margin,
                components: component_count,
                total_weight,
            },
        });
        aggregated_influence.push(total_influence);
    }

    Ok(AttGtEventTimeInfluenceOutput {
        estimates: aggregated_estimates,
        influence_functions: aggregated_influence,
    })
}

/// Aggregate ATT(g,t) estimates by treatment cohort.
///
/// # Errors
///
/// Returns [`AttGtAggregationError`] when estimates/configuration are invalid.
pub fn aggregate_att_gt_by_cohort(
    estimates: &[AttGtEstimate],
    config: AttGtAggregationConfig,
) -> Result<Vec<AttGtCohortEstimate>, AttGtAggregationError> {
    aggregate_by_key(estimates, config, |est| est.group).map(|grouped| {
        grouped
            .into_iter()
            .map(|(group, summary)| AttGtCohortEstimate { group, summary })
            .collect()
    })
}

/// Aggregate ATT(g,t) estimates by calendar time.
///
/// # Errors
///
/// Returns [`AttGtAggregationError`] when estimates/configuration are invalid.
pub fn aggregate_att_gt_by_calendar_time(
    estimates: &[AttGtEstimate],
    config: AttGtAggregationConfig,
) -> Result<Vec<AttGtCalendarEstimate>, AttGtAggregationError> {
    aggregate_by_key(estimates, config, |est| est.time).map(|grouped| {
        grouped
            .into_iter()
            .map(|(time, summary)| AttGtCalendarEstimate { time, summary })
            .collect()
    })
}

/// Aggregate ATT(g,t) estimates into a single overall summary.
///
/// # Errors
///
/// Returns [`AttGtAggregationError`] when estimates/configuration are invalid.
pub fn aggregate_att_gt_overall(
    estimates: &[AttGtEstimate],
    config: AttGtAggregationConfig,
) -> Result<AttGtAggregatedEstimate, AttGtAggregationError> {
    aggregate_by_key(estimates, config, |_| 0)
        .map(|grouped| grouped.into_values().next())
        .and_then(|opt| opt.ok_or(AttGtAggregationError::EmptyInput))
}

fn aggregate_by_key<F>(
    estimates: &[AttGtEstimate],
    config: AttGtAggregationConfig,
    key_fn: F,
) -> Result<BTreeMap<i32, AttGtAggregatedEstimate>, AttGtAggregationError>
where
    F: Fn(&AttGtEstimate) -> i32,
{
    validate_aggregation_inputs(estimates, config)?;

    let mut grouped = BTreeMap::<i32, Vec<&AttGtEstimate>>::new();
    for estimate in estimates {
        grouped.entry(key_fn(estimate)).or_default().push(estimate);
    }

    let z = z_score_for_confidence(config.confidence_level.confidence_level);
    Ok(grouped
        .into_iter()
        .map(|(key, bucket)| {
            let raw_total = bucket
                .iter()
                .map(|estimate| raw_weight(estimate, config.weighting))
                .sum::<f64>();
            let estimate = bucket
                .iter()
                .map(|entry| {
                    let w = raw_weight(entry, config.weighting) / raw_total;
                    w * entry.att
                })
                .sum::<f64>();
            let variance = bucket
                .iter()
                .map(|entry| {
                    let w = raw_weight(entry, config.weighting) / raw_total;
                    w * w * entry.se * entry.se
                })
                .sum::<f64>();
            let se = variance.sqrt();
            let margin = z * se;
            (
                key,
                AttGtAggregatedEstimate {
                    estimate,
                    se,
                    ci_low: estimate - margin,
                    ci_high: estimate + margin,
                    components: bucket.len(),
                    total_weight: raw_total,
                },
            )
        })
        .collect())
}

fn validate_aggregation_inputs(
    estimates: &[AttGtEstimate],
    config: AttGtAggregationConfig,
) -> Result<(), AttGtAggregationError> {
    if estimates.is_empty() {
        return Err(AttGtAggregationError::EmptyInput);
    }
    if !validate_confidence_level(config.confidence_level.confidence_level) {
        return Err(AttGtAggregationError::InvalidConfidenceLevel);
    }
    for estimate in estimates {
        if !estimate.att.is_finite() {
            return Err(AttGtAggregationError::InvalidEstimate);
        }
        if !estimate.se.is_finite() || estimate.se < 0.0 {
            return Err(AttGtAggregationError::InvalidSe);
        }
    }
    Ok(())
}

fn raw_weight(estimate: &AttGtEstimate, weighting: AttGtAggregationWeighting) -> f64 {
    match weighting {
        AttGtAggregationWeighting::Equal => 1.0,
        AttGtAggregationWeighting::ByTreatedCount => usize_to_f64(estimate.treated_n),
        AttGtAggregationWeighting::ByTotalWeight => estimate.total_weight,
    }
}
