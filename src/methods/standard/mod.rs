use std::collections::BTreeMap;

use crate::inference::{validate_confidence_level, z_score_for_confidence};
use crate::types::{
    CellSummary, DidCell, DidConfig, DidError, DidEstimate, DidInputSummary, EventTimeError,
    EventTimeEstimate, EventTimePoint, EventTimeWeighting, PanelObservation, TimePeriod,
    TreatmentGroup,
};

/// Aggregate event-time estimate points by event-time index.
///
/// # Errors
///
/// Returns [`EventTimeError`] when confidence level or point values are invalid.
pub fn aggregate_event_time(
    points: &[EventTimePoint],
    confidence_level: f64,
    weighting: EventTimeWeighting,
) -> Result<Vec<EventTimeEstimate>, EventTimeError> {
    if !validate_confidence_level(confidence_level) {
        return Err(EventTimeError::InvalidConfidenceLevel);
    }

    let mut grouped = BTreeMap::<i32, Vec<EventTimePoint>>::new();
    for point in points {
        if !point.estimate.is_finite() {
            return Err(EventTimeError::InvalidPointEstimate);
        }
        if !point.se.is_finite() || point.se < 0.0 {
            return Err(EventTimeError::InvalidPointSe);
        }
        if !point.weight.is_finite() || point.weight <= 0.0 {
            return Err(EventTimeError::InvalidPointWeight);
        }
        grouped.entry(point.event_time).or_default().push(*point);
    }

    let z = z_score_for_confidence(confidence_level);
    let mut output = Vec::with_capacity(grouped.len());
    for (event_time, bucket) in grouped {
        let total_weight = bucket
            .iter()
            .map(|point| match weighting {
                EventTimeWeighting::Equal => 1.0,
                EventTimeWeighting::ByWeight => point.weight,
            })
            .sum::<f64>();
        let estimate = bucket
            .iter()
            .map(|point| {
                let raw_weight = match weighting {
                    EventTimeWeighting::Equal => 1.0,
                    EventTimeWeighting::ByWeight => point.weight,
                };
                let normalized_weight = raw_weight / total_weight;
                normalized_weight * point.estimate
            })
            .sum::<f64>();
        let variance = bucket
            .iter()
            .map(|point| {
                let raw_weight = match weighting {
                    EventTimeWeighting::Equal => 1.0,
                    EventTimeWeighting::ByWeight => point.weight,
                };
                let normalized_weight = raw_weight / total_weight;
                normalized_weight * normalized_weight * point.se * point.se
            })
            .sum::<f64>();
        let se = variance.sqrt();
        let margin = z * se;

        output.push(EventTimeEstimate {
            event_time,
            estimate,
            se,
            ci_low: estimate - margin,
            ci_high: estimate + margin,
            points: bucket.len(),
            total_weight,
        });
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy, Default)]
struct CellStats {
    observations: usize,
    weight: f64,
    weight_sq: f64,
    outcome_w: f64,
    outcome_sq_w: f64,
}

impl CellStats {
    fn add(&mut self, weight: f64, outcome: f64) {
        self.observations += 1;
        self.weight += weight;
        self.weight_sq = weight.mul_add(weight, self.weight_sq);
        self.outcome_w = weight.mul_add(outcome, self.outcome_w);
        self.outcome_sq_w = (weight * outcome).mul_add(outcome, self.outcome_sq_w);
    }

    fn validate_non_empty(self, cell: DidCell) -> Result<Self, DidError> {
        if self.weight > 0.0 {
            Ok(self)
        } else {
            Err(DidError::EmptyCell { cell })
        }
    }

    fn into_summary(self, cell: DidCell) -> CellSummary {
        CellSummary {
            cell,
            observations: self.observations,
            weight_sum: self.weight,
            effective_n: self.effective_n(),
            mean_outcome: self.mean(),
            variance: self.variance(),
        }
    }

    fn mean(self) -> f64 {
        self.outcome_w / self.weight
    }

    fn effective_n(self) -> f64 {
        self.weight * self.weight / self.weight_sq
    }

    fn variance(self) -> f64 {
        let mean = self.mean();
        mean.mul_add(-mean, self.outcome_sq_w / self.weight)
            .max(0.0)
    }
}

/// Summarize panel input into canonical 2x2 diagnostics.
///
/// # Errors
///
/// Returns [`DidError`] when any panel cell is empty or invalid weights are supplied.
pub fn summarize_two_by_two(
    observations: &[PanelObservation],
) -> Result<DidInputSummary, DidError> {
    let mut treated_pre = CellStats::default();
    let mut treated_post = CellStats::default();
    let mut control_pre = CellStats::default();
    let mut control_post = CellStats::default();

    for row in observations {
        if !row.weight.is_finite() || row.weight <= 0.0 {
            return Err(DidError::InvalidWeight { value: row.weight });
        }
        if !row.outcome.is_finite() {
            return Err(DidError::InvalidOutcome { value: row.outcome });
        }
        match DidCell::from_parts(
            TreatmentGroup::from_bool(row.treated),
            TimePeriod::from_bool(row.post_period),
        ) {
            DidCell::TreatedPre => treated_pre.add(row.weight, row.outcome),
            DidCell::TreatedPost => treated_post.add(row.weight, row.outcome),
            DidCell::ControlPre => control_pre.add(row.weight, row.outcome),
            DidCell::ControlPost => control_post.add(row.weight, row.outcome),
        }
    }

    let treated_pre = treated_pre.validate_non_empty(DidCell::TreatedPre)?;
    let treated_post = treated_post.validate_non_empty(DidCell::TreatedPost)?;
    let control_pre = control_pre.validate_non_empty(DidCell::ControlPre)?;
    let control_post = control_post.validate_non_empty(DidCell::ControlPost)?;

    Ok(DidInputSummary {
        treated_pre: treated_pre.into_summary(DidCell::TreatedPre),
        treated_post: treated_post.into_summary(DidCell::TreatedPost),
        control_pre: control_pre.into_summary(DidCell::ControlPre),
        control_post: control_post.into_summary(DidCell::ControlPost),
    })
}

/// Estimate ATT for a canonical 2x2 `DiD` setup.
///
/// # Errors
///
/// Returns [`DidError`] when any panel cell is empty or when invalid weights/config are supplied.
pub fn estimate_att_two_by_two(
    observations: &[PanelObservation],
    config: DidConfig,
) -> Result<DidEstimate, DidError> {
    let summary = summarize_two_by_two(observations)?;
    estimate_att_from_summary(summary, config)
}

/// Estimate ATT from pre-computed 2x2 input diagnostics.
///
/// # Errors
///
/// Returns [`DidError`] when confidence-level configuration is invalid.
pub fn estimate_att_from_summary(
    summary: DidInputSummary,
    config: DidConfig,
) -> Result<DidEstimate, DidError> {
    if !validate_confidence_level(config.confidence_level) {
        return Err(DidError::InvalidConfidenceLevel {
            value: config.confidence_level,
        });
    }

    let att = (summary.treated_post.mean_outcome - summary.treated_pre.mean_outcome)
        - (summary.control_post.mean_outcome - summary.control_pre.mean_outcome);
    let se = ((summary.treated_pre.variance / summary.treated_pre.effective_n)
        + (summary.treated_post.variance / summary.treated_post.effective_n)
        + (summary.control_pre.variance / summary.control_pre.effective_n)
        + (summary.control_post.variance / summary.control_post.effective_n))
        .sqrt();

    let z = z_score_for_confidence(config.confidence_level);
    let margin = z * se;
    Ok(DidEstimate {
        att,
        se,
        ci_low: att - margin,
        ci_high: att + margin,
    })
}
