//! Unified data abstraction for `DiD` inference.

use crate::error::DatasetError;
use crate::inference::{ClusterIndex, ClusterSpec};
use std::collections::HashMap;
use std::hash::Hash;

/// Trait for panel data records that can be consumed by `InferenceDataset`.
pub trait PanelRecord {
    type UnitId: Eq + Hash + Clone + ToString;

    fn unit_id(&self) -> &Self::UnitId;
    fn time_period(&self) -> i32;
    fn treatment_status(&self) -> bool;
    fn first_treated_time(&self) -> Option<i32>;
    fn outcome(&self) -> f64;
    fn weight(&self) -> f64 {
        1.0
    }
    fn covariates(&self) -> &[f64] {
        &[]
    }
}

/// Diagnostic severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "Error"),
            Self::Warning => write!(f, "Warning"),
        }
    }
}

/// A single diagnostic finding during dataset validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub category: String,
    pub message: String,
    pub affected_units: Vec<String>,
}

/// A comprehensive validation report for a dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    pub diagnostics: Vec<Diagnostic>,
    pub row_count: usize,
    pub unit_count: usize,
    pub period_count: usize,
}

impl ValidationReport {
    /// Returns true if the report contains no errors.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self
            .diagnostics
            .iter()
            .any(|d| matches!(d.severity, Severity::Error))
    }
}

/// An unvalidated collection of observations for inference.
pub struct InferenceDataset {
    pub(crate) unit_ids: Vec<String>,
    pub(crate) time_periods: Vec<i32>,
    pub(crate) first_treated_times: Vec<Option<i32>>,
    pub(crate) outcomes: Vec<f64>,
    pub(crate) weights: Vec<f64>,
    pub(crate) covariates: Vec<Vec<f64>>,
    pub(crate) cluster_index: Option<ClusterIndex>,
    pub(crate) allow_unbalanced: bool,
    pub(crate) outcome_name: Option<String>,
}

/// A verified collection of observations that has passed all integrity checks.
///
/// This type can only be obtained via [`InferenceDataset::validate`].
pub struct ValidatedInferenceDataset {
    inner: InferenceDataset,
}

impl InferenceDataset {
    /// Construct from column vectors.
    ///
    /// # Panics
    /// Panics if input vectors have inconsistent lengths.
    #[must_use]
    pub fn from_columns<L: Eq + Hash + Clone + ToString>(
        unit_ids: Vec<L>,
        time_periods: Vec<i32>,
        first_treated_times: Vec<Option<i32>>,
        outcomes: Vec<f64>,
    ) -> Self {
        let n = unit_ids.len();
        assert_eq!(time_periods.len(), n);
        assert_eq!(first_treated_times.len(), n);
        assert_eq!(outcomes.len(), n);

        Self {
            unit_ids: unit_ids.into_iter().map(|id| id.to_string()).collect(),
            time_periods,
            first_treated_times,
            outcomes,
            weights: vec![1.0; n],
            covariates: vec![Vec::new(); n],
            cluster_index: None,
            allow_unbalanced: false,
            outcome_name: None,
        }
    }

    /// Construct from an iterator of panel records.
    pub fn from_panel<R: PanelRecord>(records: impl IntoIterator<Item = R>) -> Self {
        let mut unit_ids = Vec::new();
        let mut time_periods = Vec::new();
        let mut first_treated_times = Vec::new();
        let mut outcomes = Vec::new();
        let mut weights = Vec::new();
        let mut covariates = Vec::new();

        for r in records {
            unit_ids.push(r.unit_id().to_string());
            time_periods.push(r.time_period());
            first_treated_times.push(r.first_treated_time());
            outcomes.push(r.outcome());
            weights.push(r.weight());
            covariates.push(r.covariates().to_vec());
        }

        Self {
            unit_ids,
            time_periods,
            first_treated_times,
            outcomes,
            weights,
            covariates,
            cluster_index: None,
            allow_unbalanced: false,
            outcome_name: None,
        }
    }

    /// Set the outcome name for reporting.
    #[must_use]
    pub fn with_outcome_name(mut self, name: impl Into<String>) -> Self {
        self.outcome_name = Some(name.into());
        self
    }

    /// Add cluster labels to the dataset.
    #[must_use]
    pub fn with_clusters<L: Eq + Hash + Clone>(mut self, spec: ClusterSpec<L>) -> Self {
        let ClusterSpec::OneWay(labels) = spec;

        let mut label_map = HashMap::new();
        let mut assignments = Vec::with_capacity(labels.len());
        let mut next_idx = 0;

        for label in labels {
            let idx = *label_map.entry(label).or_insert_with(|| {
                let current = next_idx;
                next_idx += 1;
                current
            });
            assignments.push(idx);
        }

        self.cluster_index = Some(ClusterIndex {
            group_assignments: assignments,
            n_groups: next_idx,
        });
        self
    }

    /// Opt-in to unbalanced panel support.
    #[must_use]
    pub const fn allow_unbalanced(mut self) -> Self {
        self.allow_unbalanced = true;
        self
    }

    /// Validate the dataset and ensure internal consistency.
    ///
    /// # Errors
    /// Returns [`DatasetError`] if the data fails integrity checks.
    pub fn validate(self) -> Result<ValidatedInferenceDataset, DatasetError> {
        let n_rows = self.unit_ids.len();

        // 1. Cluster length check
        if let Some(ref ci) = self.cluster_index
            && ci.group_assignments.len() != n_rows
        {
            return Err(DatasetError::ClusterMismatch {
                expected: n_rows,
                actual: ci.group_assignments.len(),
            });
        }

        // 2. Finite values check
        for i in 0..n_rows {
            if !self.outcomes[i].is_finite() {
                return Err(DatasetError::MissingOutcome {
                    unit: self.unit_ids[i].clone(),
                    period: self.time_periods[i],
                });
            }
            if !self.weights[i].is_finite() || self.weights[i] <= 0.0 {
                return Err(DatasetError::InvalidField(
                    "weight".to_string(),
                    format!("must be finite and positive, got {}", self.weights[i]),
                ));
            }
        }

        // 3. Treatment timing consistency
        let mut unit_first_treat = HashMap::new();
        for i in 0..n_rows {
            let id = &self.unit_ids[i];
            let ft = self.first_treated_times[i];

            if let Some(&existing_ft) = unit_first_treat.get(id) {
                if existing_ft != ft {
                    return Err(DatasetError::InvalidTreatmentTiming {
                        unit: id.clone(),
                        reason: format!(
                            "inconsistent first_treated_time: expected {existing_ft:?}, got {ft:?}"
                        ),
                    });
                }
            } else {
                unit_first_treat.insert(id.clone(), ft);
            }
        }

        // 4. Panel balance check (if not allowed)
        if !self.allow_unbalanced {
            let mut periods_per_unit = HashMap::new();
            let all_periods: Vec<i32> = {
                let mut p = self.time_periods.clone();
                p.sort_unstable();
                p.dedup();
                p
            };

            for (i, id) in self.unit_ids.iter().enumerate() {
                periods_per_unit
                    .entry(id.clone())
                    .or_insert_with(Vec::new)
                    .push(self.time_periods[i]);
            }

            for (_, periods) in periods_per_unit {
                if periods.len() != all_periods.len() {
                    return Err(DatasetError::UnbalancedPanel {
                        missing: all_periods,
                    });
                }
            }
        }

        Ok(ValidatedInferenceDataset { inner: self })
    }

    /// Diagnostic report for a dataset.
    #[allow(clippy::too_many_lines)]
    pub fn validate_collecting(&self) -> ValidationReport {
        let n_rows = self.unit_ids.len();
        let mut diagnostics = Vec::new();

        // 1. Cluster length check
        if let Some(ref ci) = self.cluster_index
            && ci.group_assignments.len() != n_rows
        {
            diagnostics.push(Diagnostic {
                severity: Severity::Error,
                category: "cluster".to_string(),
                message: format!(
                    "cluster label mismatch: expected {n_rows} labels, got {}",
                    ci.group_assignments.len()
                ),
                affected_units: Vec::new(),
            });
        }

        // 2. Finite values check
        let mut missing_outcome_units = Vec::new();
        let mut invalid_weight_units = Vec::new();
        for i in 0..n_rows {
            if !self.outcomes[i].is_finite() {
                missing_outcome_units.push(self.unit_ids[i].clone());
            }
            if !self.weights[i].is_finite() || self.weights[i] <= 0.0 {
                invalid_weight_units.push(self.unit_ids[i].clone());
            }
        }
        if !missing_outcome_units.is_empty() {
            diagnostics.push(Diagnostic {
                severity: Severity::Error,
                category: "outcome".to_string(),
                message: format!("missing outcome for {} rows", missing_outcome_units.len()),
                affected_units: missing_outcome_units,
            });
        }
        if !invalid_weight_units.is_empty() {
            diagnostics.push(Diagnostic {
                severity: Severity::Error,
                category: "weight".to_string(),
                message: format!("invalid weight for {} rows", invalid_weight_units.len()),
                affected_units: invalid_weight_units,
            });
        }

        // 3. Treatment timing consistency
        let mut unit_first_treat = HashMap::new();
        let mut inconsistent_timing_units = Vec::new();
        for i in 0..n_rows {
            let id = &self.unit_ids[i];
            let ft = self.first_treated_times[i];

            if let Some(&existing_ft) = unit_first_treat.get(id) {
                if existing_ft != ft {
                    inconsistent_timing_units.push(id.clone());
                }
            } else {
                unit_first_treat.insert(id.clone(), ft);
            }
        }
        if !inconsistent_timing_units.is_empty() {
            diagnostics.push(Diagnostic {
                severity: Severity::Error,
                category: "treatment_timing".to_string(),
                message: format!(
                    "inconsistent first_treated_time for {} units",
                    inconsistent_timing_units.len()
                ),
                affected_units: inconsistent_timing_units,
            });
        }

        let all_periods: Vec<i32> = {
            let mut p = self.time_periods.clone();
            p.sort_unstable();
            p.dedup();
            p
        };

        // 4. Panel balance
        let mut periods_per_unit = HashMap::new();
        for (i, id) in self.unit_ids.iter().enumerate() {
            periods_per_unit
                .entry(id.clone())
                .or_insert_with(Vec::new)
                .push(self.time_periods[i]);
        }

        let mut unbalanced_units = Vec::new();
        for (id, periods) in &periods_per_unit {
            if periods.len() != all_periods.len() {
                unbalanced_units.push(id.clone());
            }
        }

        if !unbalanced_units.is_empty() && !self.allow_unbalanced {
            diagnostics.push(Diagnostic {
                severity: Severity::Error,
                category: "balance".to_string(),
                message: format!(
                    "unbalanced panel: {} units affected",
                    unbalanced_units.len()
                ),
                affected_units: unbalanced_units,
            });
        } else if !unbalanced_units.is_empty() {
            diagnostics.push(Diagnostic {
                severity: Severity::Warning,
                category: "balance".to_string(),
                message: format!(
                    "unbalanced panel: {} units affected (opted-in)",
                    unbalanced_units.len()
                ),
                affected_units: unbalanced_units,
            });
        }

        // 5. Pre-treatment coverage check
        let mut units_without_pre = Vec::new();
        for (id, periods) in &periods_per_unit {
            let ft = unit_first_treat.get(id).copied().flatten();
            if let Some(g) = ft {
                let has_pre = periods.iter().any(|&t| t < g);
                if !has_pre {
                    units_without_pre.push(id.clone());
                }
            }
        }
        if !units_without_pre.is_empty() {
            diagnostics.push(Diagnostic {
                severity: Severity::Warning,
                category: "coverage".to_string(),
                message: format!(
                    "no pre-treatment periods found for {} units; Honest DiD sensitivity analysis will fail",
                    units_without_pre.len()
                ),
                affected_units: units_without_pre,
            });
        }

        ValidationReport {
            diagnostics,
            row_count: n_rows,
            unit_count: periods_per_unit.len(),
            period_count: all_periods.len(),
        }
    }
}

impl ValidatedInferenceDataset {
    /// Estimate ATT(g,t) using doubly robust repeated cross-section estimators.
    ///
    /// # Errors
    /// Returns an error if the underlying estimation fails.
    pub fn estimate_att_gt_dr(
        &self,
        config: &crate::types::EventStudyConfig,
    ) -> Result<crate::methods::att_gt::EventStudyResult, crate::types::AttGtError> {
        self.estimate_with_method(
            config,
            crate::methods::att_gt::estimate_att_gt_dr_with_influence,
        )
    }

    /// Estimate ATT(g,t) using inverse-probability-weighted (IPW) repeated cross-section estimators.
    ///
    /// # Errors
    /// Returns an error if the underlying estimation fails.
    pub fn estimate_att_gt_ipw(
        &self,
        config: &crate::types::EventStudyConfig,
    ) -> Result<crate::methods::att_gt::EventStudyResult, crate::types::AttGtError> {
        self.estimate_with_method(
            config,
            crate::methods::att_gt::estimate_att_gt_ipw_with_influence,
        )
    }

    /// Estimate ATT(g,t) using outcome-regression (OR) repeated cross-section estimators.
    ///
    /// # Errors
    /// Returns an error if the underlying estimation fails.
    pub fn estimate_att_gt_or(
        &self,
        config: &crate::types::EventStudyConfig,
    ) -> Result<crate::methods::att_gt::EventStudyResult, crate::types::AttGtError> {
        self.estimate_with_method(
            config,
            crate::methods::att_gt::estimate_att_gt_or_with_influence,
        )
    }

    fn estimate_with_method<F>(
        &self,
        config: &crate::types::EventStudyConfig,
        method: F,
    ) -> Result<crate::methods::att_gt::EventStudyResult, crate::types::AttGtError>
    where
        F: Fn(
            &[crate::types::AttGtDrObservation],
            crate::types::AttGtDrConfig,
        ) -> Result<crate::types::AttGtInfluenceOutput, crate::types::AttGtError>,
    {
        let ds = &self.inner;
        // Unit ids arrive as strings. The panel route needs a stable integer per
        // unit, so distinct ids are numbered in sorted order. Sorted rather than
        // first-seen so that the numbering does not depend on row order, which
        // would make an influence vector depend on how the frame was assembled.
        let unit_codes: std::collections::BTreeMap<&str, i64> = {
            let mut distinct: Vec<&str> = ds.unit_ids.iter().map(String::as_str).collect();
            distinct.sort_unstable();
            distinct.dedup();
            distinct
                .into_iter()
                .enumerate()
                .map(|(position, id)| (id, crate::util::usize_to_i64(position)))
                .collect()
        };
        let observations: Vec<crate::types::AttGtDrObservation> = ds
            .unit_ids
            .iter()
            .enumerate()
            .map(|(i, id)| crate::types::AttGtDrObservation {
                unit_id: unit_codes.get(id.as_str()).copied(),
                first_treated_time: ds.first_treated_times[i],
                time: ds.time_periods[i],
                outcome: ds.outcomes[i],
                weight: ds.weights[i],
                covariates: ds.covariates[i].clone(),
            })
            .collect();

        let out = method(
            &observations,
            crate::types::AttGtDrConfig {
                att_gt: config.att_gt,
                drdid: config.drdid,
            },
        )?;

        let mut res = crate::methods::att_gt::EventStudyResult {
            estimates: out.estimates,
            influence_functions: out.influence_functions,
            cluster_index: ds.cluster_index.clone(),
            config: config.clone(),
            outcome_name: ds.outcome_name.clone(),
        };

        // Update clustered SEs immediately if clustering is requested
        if let Some(ref ci) = res.cluster_index {
            let z = crate::inference::z_score_for_confidence(config.confidence_level);
            for (idx, est) in res.estimates.iter_mut().enumerate() {
                if let Ok(clustered_se) = crate::inference::clustered_standard_error_from_index(
                    &res.influence_functions[idx],
                    ci,
                ) {
                    est.se = clustered_se;
                    let margin = z * est.se;
                    est.ci_low = est.att - margin;
                    est.ci_high = est.att + margin;
                }
            }
        }

        Ok(res)
    }
}
