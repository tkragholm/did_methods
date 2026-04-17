use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::inference::z_score_for_confidence;
use crate::types::{
    AttGtConfig, AttGtError, AttGtEstimate, AttGtObservation, BasePeriod, ComparisonGroup,
    InferenceConfig,
};

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

fn diff_in_diff(
    treated_pre: CellStats,
    treated_post: CellStats,
    control_pre: CellStats,
    control_post: CellStats,
    confidence: InferenceConfig,
) -> (f64, f64, f64, f64) {
    let att =
        (treated_post.mean() - treated_pre.mean()) - (control_post.mean() - control_pre.mean());
    let se = ((treated_pre.variance() / treated_pre.effective_n())
        + (treated_post.variance() / treated_post.effective_n())
        + (control_pre.variance() / control_pre.effective_n())
        + (control_post.variance() / control_post.effective_n()))
    .sqrt();
    let z = z_score_for_confidence(confidence.confidence_level);
    let margin = z * se;
    (att, se, att - margin, att + margin)
}

#[derive(Debug, Clone, Copy, Default)]
struct UnitPeriodCell {
    pre_weight: f64,
    pre_outcome: f64,
    post_weight: f64,
    post_outcome: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct DeltaStats {
    observations: usize,
    weight: f64,
    weight_sq: f64,
    delta_w: f64,
    delta_sq_w: f64,
}

impl DeltaStats {
    fn add(&mut self, weight: f64, delta: f64) {
        self.observations += 1;
        self.weight += weight;
        self.weight_sq = weight.mul_add(weight, self.weight_sq);
        self.delta_w = weight.mul_add(delta, self.delta_w);
        self.delta_sq_w = (weight * delta).mul_add(delta, self.delta_sq_w);
    }

    fn mean(self) -> f64 {
        self.delta_w / self.weight
    }

    fn effective_n(self) -> f64 {
        self.weight * self.weight / self.weight_sq
    }

    fn variance(self) -> f64 {
        let mean = self.mean();
        mean.mul_add(-mean, self.delta_sq_w / self.weight).max(0.0)
    }
}

fn weighted_diff_in_diff_panel(
    treated: DeltaStats,
    control: DeltaStats,
    confidence: InferenceConfig,
) -> (f64, f64, f64, f64, usize, usize, f64) {
    let att = treated.mean() - control.mean();
    let se = ((treated.variance() / treated.effective_n())
        + (control.variance() / control.effective_n()))
    .sqrt();
    let z = z_score_for_confidence(confidence.confidence_level);
    let margin = z * se;
    (
        att,
        se,
        att - margin,
        att + margin,
        treated.observations,
        control.observations,
        treated.weight + control.weight,
    )
}

/// Estimate staggered-adoption `ATT(g,t)` effects.
///
/// # Errors
///
/// Returns [`AttGtError`] when configuration/inputs are invalid or no estimable
/// group-time pairs can be formed.
pub fn estimate_att_gt(
    observations: &[AttGtObservation],
    config: AttGtConfig,
) -> Result<Vec<AttGtEstimate>, AttGtError> {
    super::validate_config(config)?;
    if observations.is_empty() {
        return Err(AttGtError::EmptyInput);
    }

    let scan = validate_observations_and_collect_sets(observations)?;
    if config.comparison_group == ComparisonGroup::NeverTreated && !scan.has_never_treated {
        return Err(AttGtError::MissingNeverTreatedGroup);
    }

    let all_times = scan.times.into_iter().collect::<Vec<_>>();
    let mut out = Vec::new();

    for group in scan.treated_groups {
        let universal_baseline_time = group - config.anticipation_periods - 1;
        for &time in &all_times {
            let baseline_time =
                baseline_time_for_pair(time, group, universal_baseline_time, config);
            if time == baseline_time {
                continue;
            }
            if let Some(estimate) = estimate_pair(observations, group, time, baseline_time, config)?
            {
                out.push(estimate);
            }
        }
    }

    if out.is_empty() {
        return Err(AttGtError::NoEstimablePairs);
    }
    Ok(out)
}

const fn baseline_time_for_pair(
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

struct ObservationScan {
    times: BTreeSet<i32>,
    treated_groups: BTreeSet<i32>,
    has_never_treated: bool,
}

fn validate_observations_and_collect_sets(
    observations: &[AttGtObservation],
) -> Result<ObservationScan, AttGtError> {
    let mut times = BTreeSet::new();
    let mut treated_groups = BTreeSet::new();
    let mut has_never_treated = false;

    for row in observations {
        if !row.weight.is_finite() || row.weight <= 0.0 {
            return Err(AttGtError::InvalidWeight { value: row.weight });
        }
        if !row.outcome.is_finite() {
            return Err(AttGtError::InvalidOutcome { value: row.outcome });
        }
        times.insert(row.time);
        if let Some(group) = row.first_treated_time {
            treated_groups.insert(group);
        } else {
            has_never_treated = true;
        }
    }

    Ok(ObservationScan {
        times,
        treated_groups,
        has_never_treated,
    })
}

struct PairCells {
    treated_pre: CellStats,
    treated_post: CellStats,
    control_pre: CellStats,
    control_post: CellStats,
}

impl PairCells {
    fn missing_cell_label(&self) -> Option<&'static str> {
        [
            (self.treated_pre.weight <= 0.0, "treated_pre"),
            (self.treated_post.weight <= 0.0, "treated_post"),
            (self.control_pre.weight <= 0.0, "control_pre"),
            (self.control_post.weight <= 0.0, "control_post"),
        ]
        .into_iter()
        .find_map(|(is_missing, label)| if is_missing { Some(label) } else { None })
    }

    fn total_weight(&self) -> f64 {
        self.treated_pre.weight
            + self.treated_post.weight
            + self.control_pre.weight
            + self.control_post.weight
    }
}

fn estimate_pair(
    observations: &[AttGtObservation],
    group: i32,
    time: i32,
    baseline_time: i32,
    config: AttGtConfig,
) -> Result<Option<AttGtEstimate>, AttGtError> {
    if let Some(estimate) = estimate_pair_panel(observations, group, time, baseline_time, config)? {
        return Ok(Some(estimate));
    }

    let cells = compute_pair_cells(
        observations,
        group,
        time,
        baseline_time,
        config.comparison_group,
        config.anticipation_periods,
    );

    if let Some(cell) = cells.missing_cell_label() {
        if config.skip_incomplete_pairs {
            return Ok(None);
        }
        return Err(AttGtError::MissingCell {
            group,
            time,
            baseline_time,
            cell,
        });
    }

    let (att, se, ci_low, ci_high) = diff_in_diff(
        cells.treated_pre,
        cells.treated_post,
        cells.control_pre,
        cells.control_post,
        config.confidence_level,
    );

    Ok(Some(AttGtEstimate {
        group,
        time,
        event_time: time - group,
        att,
        se,
        ci_low,
        ci_high,
        treated_n: cells.treated_post.observations,
        control_n: cells.control_post.observations,
        total_weight: cells.total_weight(),
    }))
}

fn estimate_pair_panel(
    observations: &[AttGtObservation],
    group: i32,
    time: i32,
    baseline_time: i32,
    config: AttGtConfig,
) -> Result<Option<AttGtEstimate>, AttGtError> {
    let mut units = BTreeMap::<(i64, bool), UnitPeriodCell>::new();
    let mut saw_panel_rows = false;

    for row in observations {
        if row.time != baseline_time && row.time != time {
            continue;
        }

        let Some(unit_id) = row.unit_id else {
            continue;
        };

        let treated = if row.first_treated_time == Some(group) {
            true
        } else if super::is_control_for_pair(
            row.first_treated_time,
            time,
            config.comparison_group,
            config.anticipation_periods,
        ) {
            false
        } else {
            continue;
        };

        saw_panel_rows = true;
        let slot = units.entry((unit_id, treated)).or_default();
        if row.time == baseline_time {
            slot.pre_weight = row.weight;
            slot.pre_outcome = row.outcome;
        } else {
            slot.post_weight = row.weight;
            slot.post_outcome = row.outcome;
        }
    }

    if !saw_panel_rows {
        return Ok(None);
    }

    let mut treated = DeltaStats::default();
    let mut control = DeltaStats::default();

    for ((_, is_treated), cell) in units {
        if cell.pre_weight <= 0.0 || cell.post_weight <= 0.0 {
            continue;
        }
        let delta = cell.post_outcome - cell.pre_outcome;
        let weight = 0.5 * (cell.pre_weight + cell.post_weight);
        if is_treated {
            treated.add(weight, delta);
        } else {
            control.add(weight, delta);
        }
    }

    let missing_cell = [
        (treated.weight <= 0.0, "treated_panel"),
        (control.weight <= 0.0, "control_panel"),
    ]
    .into_iter()
    .find_map(|(missing, label)| if missing { Some(label) } else { None });

    if let Some(cell) = missing_cell {
        if config.skip_incomplete_pairs {
            return Ok(None);
        }
        return Err(AttGtError::MissingCell {
            group,
            time,
            baseline_time,
            cell,
        });
    }

    let (att, se, ci_low, ci_high, treated_n, control_n, total_weight) =
        weighted_diff_in_diff_panel(treated, control, config.confidence_level);

    Ok(Some(AttGtEstimate {
        group,
        time,
        event_time: time - group,
        att,
        se,
        ci_low,
        ci_high,
        treated_n,
        control_n,
        total_weight,
    }))
}

fn compute_pair_cells(
    observations: &[AttGtObservation],
    group: i32,
    time: i32,
    baseline_time: i32,
    comparison_group: ComparisonGroup,
    anticipation_periods: i32,
) -> PairCells {
    let mut treated_pre = CellStats::default();
    let mut treated_post = CellStats::default();
    let mut control_pre = CellStats::default();
    let mut control_post = CellStats::default();

    for row in observations {
        if row.time == baseline_time && row.first_treated_time == Some(group) {
            treated_pre.add(row.weight, row.outcome);
        }
        if row.time == time && row.first_treated_time == Some(group) {
            treated_post.add(row.weight, row.outcome);
        }
        if row.time == baseline_time
            && super::is_control_for_pair(
                row.first_treated_time,
                time,
                comparison_group,
                anticipation_periods,
            )
        {
            control_pre.add(row.weight, row.outcome);
        }
        if row.time == time
            && super::is_control_for_pair(
                row.first_treated_time,
                time,
                comparison_group,
                anticipation_periods,
            )
        {
            control_post.add(row.weight, row.outcome);
        }
    }

    PairCells {
        treated_pre,
        treated_post,
        control_pre,
        control_post,
    }
}
