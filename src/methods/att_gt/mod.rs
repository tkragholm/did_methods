#[cfg(feature = "honest")]
use crate::inference::sensitivity::{
    HonestAssessment, HonestEventStudyInput, HonestSensitivity, HonestWorkflowConfig,
    assess_honest_event_study_functional_with_config,
};
use crate::inference::{
    ClusterIndex, clustered_standard_error_from_index, validate_confidence_level,
    z_score_for_confidence,
};
use crate::types::{
    AttGtAggregationError, AttGtConfig, AttGtDrConfig, AttGtDrObservation, AttGtError,
    AttGtEstimate, AttGtEventTimeEstimate, AttGtInfluenceOutput, AttGtObservation, ComparisonGroup,
    EventStudyConfig, ReportingSummary,
};

mod aggregation;
pub mod aggte;
mod bands;
mod core;
pub mod pair_estimators;
pub mod panel_pairs;

pub use aggregation::{
    aggregate_att_gt_by_calendar_time, aggregate_att_gt_by_cohort, aggregate_att_gt_event_time,
    aggregate_att_gt_event_time_with_influence, aggregate_att_gt_overall,
};
pub use aggte::{
    AggteConfig, AggteError, AggteEstimate, AggteResult, AggteType, UnitPanel, aggregate_att_gt,
    att_gt_clustered_standard_errors, row_panel,
};
pub use bands::{
    att_gt_mboot_bands, att_gt_simultaneous_bands, att_gt_simultaneous_bands_with_influence,
};
pub use pair_estimators::estimate_att_gt_dr_efficient_with_influence;
pub use panel_pairs::{
    estimate_att_gt_dr_panel, estimate_att_gt_dr_panel_with_influence, unit_panel,
};

/// Consolidated result of an event-study estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct EventStudyResult {
    pub estimates: Vec<AttGtEstimate>,
    pub influence_functions: Vec<Vec<f64>>,
    pub cluster_index: Option<ClusterIndex>,
    pub config: EventStudyConfig,
    pub outcome_name: Option<String>,
}

impl EventStudyResult {
    /// Aggregate estimates by event-time.
    ///
    /// # Errors
    ///
    /// Returns [`AttGtAggregationError`] when event-time aggregation fails.
    pub fn aggregate_event_time(self) -> Result<EventTimeResult, AttGtAggregationError> {
        let aggregated = aggregate_att_gt_event_time_with_influence(
            &self.estimates,
            &self.influence_functions,
            self.config.aggregation,
        )?;

        let mut res = EventTimeResult {
            estimates: aggregated.estimates,
            influence_functions: aggregated.influence_functions,
            cluster_index: self.cluster_index,
            config: self.config,
            outcome_name: self.outcome_name,
        };

        res.update_clustered_inference();
        Ok(res)
    }
}

/// Robust assessment result for a post-treatment window.
#[cfg(feature = "honest")]
#[derive(Debug, Clone, PartialEq)]
pub struct WindowAssessment {
    pub assessment: HonestAssessment,
    pub matched_periods: Vec<i32>,
    pub requested_range: (i32, i32),
}

#[cfg(feature = "honest")]
impl WindowAssessment {
    /// Require that all requested periods in the window are present in the data.
    ///
    /// # Errors
    /// Returns an error if any integer in `[start, end]` is missing from the period grid.
    pub fn require_full_window(self) -> Result<HonestAssessment, String> {
        let (start, end) = self.requested_range;
        for p in start..=end {
            if !self.matched_periods.contains(&p) {
                return Err(format!(
                    "Partial coverage: window [{start}, {end}] is missing period {p}"
                ));
            }
        }
        Ok(self.assessment)
    }
}

/// Consolidated result of an event-time aggregated event-study.
#[derive(Debug, Clone, PartialEq)]
pub struct EventTimeResult {
    pub estimates: Vec<AttGtEventTimeEstimate>,
    pub influence_functions: Vec<Vec<f64>>,
    pub cluster_index: Option<ClusterIndex>,
    pub config: EventStudyConfig,
    pub outcome_name: Option<String>,
}

impl EventTimeResult {
    fn update_clustered_inference(&mut self) {
        if let Some(ref ci) = self.cluster_index {
            let z = z_score_for_confidence(self.config.confidence_level);
            for (idx, est) in self.estimates.iter_mut().enumerate() {
                if let Ok(se) =
                    clustered_standard_error_from_index(&self.influence_functions[idx], ci)
                {
                    est.summary.se = se;
                    let margin = z * se;
                    est.summary.ci_low = est.summary.estimate - margin;
                    est.summary.ci_high = est.summary.estimate + margin;
                }
            }
        }
    }

    /// Access the standard error for a specific event-time estimate by index.
    #[must_use]
    pub fn se(&self, index: usize) -> f64 {
        self.estimates[index].summary.se
    }

    /// Access the standard error for a specific event-time estimate.
    ///
    /// Returns 0.0 if the event time is not present in the results.
    #[must_use]
    pub fn se_at(&self, event_time: i32) -> f64 {
        self.estimates
            .iter()
            .find(|est| est.event_time == event_time)
            .map_or(0.0, |est| est.summary.se)
    }

    /// Access the confidence interval for a specific event-time estimate by index.
    #[must_use]
    pub fn ci(&self, index: usize) -> (f64, f64) {
        let est = &self.estimates[index].summary;
        (est.ci_low, est.ci_high)
    }

    /// Access the confidence interval for a specific event-time estimate.
    ///
    /// Returns (0.0, 0.0) if the event time is not present in the results.
    #[must_use]
    pub fn ci_at(&self, event_time: i32) -> (f64, f64) {
        self.estimates
            .iter()
            .find(|est| est.event_time == event_time)
            .map_or((0.0, 0.0), |est| (est.summary.ci_low, est.summary.ci_high))
    }

    /// Convert to `HonestEventStudyInput` for sensitivity analysis.
    ///
    /// # Errors
    ///
    /// Returns an error when there are no pre-treatment periods or covariance
    /// construction fails.
    #[cfg(feature = "honest")]
    pub fn to_honest_input(&self) -> Result<HonestEventStudyInput, String> {
        let mut pre_periods = Vec::new();
        let mut post_periods = Vec::new();
        let mut betahat = Vec::new();
        let mut if_matrix = Vec::new();

        // Sort estimates by event time to ensure consistent ordering
        let mut sorted_estimates = self.estimates.iter().enumerate().collect::<Vec<_>>();
        sorted_estimates.sort_by_key(|(_, est)| est.event_time);

        for (idx, est) in sorted_estimates {
            if est.event_time < 0 {
                pre_periods.push(est.event_time);
            } else {
                post_periods.push(est.event_time);
            }
            betahat.push(est.summary.estimate);
            if_matrix.push(self.influence_functions[idx].clone());
        }

        if pre_periods.is_empty() {
            return Err("Honest DiD analysis requires at least one pre-treatment event-time point to bound bias (none found in estimates)".to_string());
        }

        // Compute covariance matrix from influence functions
        // If clustering is present, use clustered covariance
        let covariance = if let Some(ref ci) = self.cluster_index {
            crate::inference::vcov::clustered_covariance_matrix_from_influence_matrix_index(
                &if_matrix, ci,
            )?
        } else {
            crate::inference::vcov::covariance_matrix_from_influence_matrix(&if_matrix)?
        };

        Ok(HonestEventStudyInput {
            betahat,
            covariance,
            pre_periods,
            post_periods,
        })
    }

    /// Assess sensitivity for an average window of post-treatment periods.
    ///
    /// # Errors
    /// Returns an error if the honest input construction or assessment fails.
    #[cfg(feature = "honest")]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "forward-compatible API accepts owned workflow config"
    )]
    pub fn assess_window(
        &self,
        start: i32,
        end: i32,
        sensitivity: HonestSensitivity,
        config: HonestWorkflowConfig,
    ) -> Result<WindowAssessment, String> {
        let input = self.to_honest_input()?;
        let mut post_weights = vec![0.0; input.post_periods.len()];
        let mut matched_periods = Vec::new();

        for (idx, &p) in input.post_periods.iter().enumerate() {
            if p >= start && p <= end {
                post_weights[idx] = 1.0;
                matched_periods.push(p);
            }
        }

        if matched_periods.is_empty() {
            return Err(format!(
                "Window [{start}, {end}] covers no post-treatment periods"
            ));
        }

        let total_weight: f64 = post_weights.iter().sum();
        for val in &mut post_weights {
            *val /= total_weight;
        }

        let assessment = assess_honest_event_study_functional_with_config(
            &input,
            &post_weights,
            sensitivity,
            self.config.inference(),
            0.0,
            config.relative_magnitude,
        )?;

        Ok(WindowAssessment {
            assessment,
            matched_periods,
            requested_range: (start, end),
        })
    }

    /// Export a specific event-time estimate to a reporting summary.
    #[must_use]
    pub fn to_summary(
        &self,
        outcome: &str,
        subgroup: &str,
        event_time: i32,
    ) -> Option<ReportingSummary> {
        self.estimates
            .iter()
            .find(|est| est.event_time == event_time)
            .map(|est| ReportingSummary {
                schema_version: 1,
                crate_version: env!("CARGO_PKG_VERSION").to_string(),
                config_hash: None,
                estimate: est.summary.estimate,
                std_error: est.summary.se,
                ci_lower: est.summary.ci_low,
                ci_upper: est.summary.ci_high,
                p_value: None, // Could be computed if needed
                outcome: outcome.to_string(),
                subgroup: subgroup.to_string(),
                estimator: "DR-DiD".to_string(),
                n_obs: est.summary.components, // This is component count, maybe we want unit count?
                n_clusters: self.cluster_index.as_ref().map(|ci| ci.n_groups),
            })
    }

    /// Export all event-time estimates to reporting summaries.
    #[must_use]
    pub fn to_summaries(&self, subgroup: &str) -> Vec<ReportingSummary> {
        let outcome = self
            .outcome_name
            .as_deref()
            .unwrap_or("unnamed_outcome")
            .to_string();
        self.estimates
            .iter()
            .map(|est| ReportingSummary {
                schema_version: 1,
                crate_version: env!("CARGO_PKG_VERSION").to_string(),
                config_hash: None,
                estimate: est.summary.estimate,
                std_error: est.summary.se,
                ci_lower: est.summary.ci_low,
                ci_upper: est.summary.ci_high,
                p_value: None,
                outcome: outcome.clone(),
                subgroup: subgroup.to_string(),
                estimator: format!("DR-DiD (t={})", est.event_time),
                n_obs: est.summary.components,
                n_clusters: self.cluster_index.as_ref().map(|ci| ci.n_groups),
            })
            .collect()
    }
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
    core::estimate_att_gt(observations, config)
}

/// Estimate staggered-adoption `ATT(g,t)` effects using DR repeated cross-section
/// `DiD` estimators for each `(g, t)` pair.
///
/// # Errors
///
/// Returns [`AttGtError`] when configuration/inputs are invalid, when required
/// pair cells are missing, or when DR nuisance estimation fails.
pub fn estimate_att_gt_dr(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<Vec<AttGtEstimate>, AttGtError> {
    pair_estimators::estimate_att_gt_dr(observations, config)
}

/// Estimate staggered-adoption `ATT(g,t)` via DR repeated cross-section `DiD`,
/// and return aligned influence vectors for joint-process inference.
///
/// # Errors
///
/// Returns [`AttGtError`] for invalid inputs/configuration, missing required
/// cells, failed nuisance estimation, or incompatible influence-vector shapes.
pub fn estimate_att_gt_dr_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    pair_estimators::estimate_att_gt_dr_with_influence(observations, config)
}

/// Estimate staggered-adoption ATT(g,t) using an outcome-regression (OR) path.
///
/// # Errors
///
/// Returns [`AttGtError`] for invalid inputs/configuration, missing required
/// cells, or failed pair estimation.
pub fn estimate_att_gt_or(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<Vec<AttGtEstimate>, AttGtError> {
    estimate_att_gt_or_with_influence(observations, config).map(|out| out.estimates)
}

/// Estimate staggered-adoption ATT(g,t) using an outcome-regression (OR) path,
/// and return aligned influence vectors for joint-process inference.
///
/// # Errors
///
/// Returns [`AttGtError`] for invalid inputs/configuration, missing required
/// cells, failed pair estimation, or incompatible influence-vector shapes.
pub fn estimate_att_gt_or_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    pair_estimators::estimate_att_gt_or_with_influence(observations, config)
}

/// Estimate staggered-adoption ATT(g,t) using an inverse-probability-weighted (IPW) path.
///
/// # Errors
///
/// Returns [`AttGtError`] for invalid inputs/configuration, missing required
/// cells, or failed pair estimation.
pub fn estimate_att_gt_ipw(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<Vec<AttGtEstimate>, AttGtError> {
    estimate_att_gt_ipw_with_influence(observations, config).map(|out| out.estimates)
}

/// Estimate staggered-adoption ATT(g,t) using an inverse-probability-weighted
/// (IPW) path, and return aligned influence vectors for joint-process inference.
///
/// # Errors
///
/// Returns [`AttGtError`] for invalid inputs/configuration, missing required
/// cells, failed pair estimation, or incompatible influence-vector shapes.
pub fn estimate_att_gt_ipw_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    pair_estimators::estimate_att_gt_ipw_with_influence(observations, config)
}

fn is_control_for_pair(
    first_treated_time: Option<i32>,
    pair_time: i32,
    mode: ComparisonGroup,
    anticipation_periods: i32,
) -> bool {
    match mode {
        ComparisonGroup::NeverTreated => first_treated_time.is_none(),
        ComparisonGroup::NotYetTreated => {
            first_treated_time.is_none_or(|g| g > pair_time + anticipation_periods)
        }
    }
}

fn validate_config(config: AttGtConfig) -> Result<(), AttGtError> {
    if !validate_confidence_level(config.confidence_level.confidence_level) {
        return Err(AttGtError::InvalidConfidenceLevel);
    }
    if config.anticipation_periods < 0 {
        return Err(AttGtError::InvalidAnticipationPeriods);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
