//! Aggregation of `ATT(g,t)` into the four summaries `did::aggte` produces.
//!
//! # Why this is not [`super::aggregation`]
//!
//! That module groups estimates by a key and takes a weighted average with one
//! of three fixed weighting rules. It is useful, and it is not `aggte`. Two
//! differences matter:
//!
//! * **It averages pre-treatment cells in.** `aggte` uses only cells with
//!   `t >= g` for the group, calendar and simple summaries; a placebo cell is
//!   not part of an average treatment effect.
//! * **It treats the cells as independent.** Its variance is
//!   `sum(w^2 * se^2)`, which ignores every off-diagonal term. The cells share
//!   units, so those terms are not zero.
//!
//! # The weights are estimated, and that shows up in the standard error
//!
//! Three of the four summaries weight cells by cohort share `pg = P(G = g)`,
//! which is estimated from the same sample. Treating it as fixed understates the
//! variance: measured against `did` 2.5.1 on `mpdta` at event time 0, the
//! fixed-weight standard error is 0.0114607 and the correct one is 0.0114942.
//! [`weight_influence`] is the correction, ported from `did:::wif`.
//!
//! The group summary is the exception: within a cohort the cells are averaged
//! with equal weights, which are not estimated, so no correction applies.

use std::collections::{BTreeMap, BTreeSet};

use crate::inference::z_score_for_confidence;
use crate::types::{AttGtEstimate, InferenceConfig};
use crate::util::usize_to_f64;

/// Which summary to produce, matching `aggte(type = ...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggteType {
    /// One number: every post-treatment cell, weighted by cohort share.
    Simple,
    /// By event time, then averaged over non-negative event times.
    Dynamic,
    /// By cohort, then averaged over cohorts by share.
    Group,
    /// By calendar period, then averaged over periods.
    Calendar,
}

/// Configuration for [`aggregate_att_gt`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AggteConfig {
    pub aggregation: AggteType,
    /// Restrict the dynamic path to a FIXED cohort composition: keep only
    /// cohorts observed for at least this many post-treatment periods, and only
    /// event times up to it. `did::aggte(balance_e = )`.
    ///
    /// This is the answer to "did the effect change with duration, or is the
    /// long-horizon estimate simply made of different cohorts?" Without it the
    /// two are not separable. Ignored for the other three types, as in R.
    pub balance_e: Option<i32>,
    /// Drop event times below this, inclusive bound. `did::aggte(min_e = )`.
    ///
    /// Applied AFTER `balance_e`, and it narrows the overall number as well as
    /// the path: the dynamic overall is the mean over retained non-negative event
    /// times, so trimming the path moves it. Dynamic only, as in R.
    pub min_e: Option<i32>,
    /// Drop event times above this, inclusive bound. `did::aggte(max_e = )`.
    pub max_e: Option<i32>,
    pub confidence_level: InferenceConfig,
}

impl Default for AggteConfig {
    fn default() -> Self {
        Self {
            aggregation: AggteType::Dynamic,
            balance_e: None,
            min_e: None,
            max_e: None,
            confidence_level: InferenceConfig::default(),
        }
    }
}

/// One aggregated point, plus the influence function behind it.
#[derive(Debug, Clone, PartialEq)]
pub struct AggteEstimate {
    /// Event time, cohort or calendar period, depending on the type.
    pub key: i32,
    pub att: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub influence: Vec<f64>,
}

/// The result of an aggregation: the headline number and, where the type has
/// one, the path behind it.
#[derive(Debug, Clone, PartialEq)]
pub struct AggteResult {
    pub overall_att: f64,
    pub overall_se: f64,
    pub overall_ci_low: f64,
    pub overall_ci_high: f64,
    pub overall_influence: Vec<f64>,
    /// Empty for [`AggteType::Simple`], which has no path.
    pub by_key: Vec<AggteEstimate>,
}

/// Errors from aggregation.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum AggteError {
    #[error("aggregation requires at least one ATT(g,t) estimate")]
    EmptyInput,
    #[error("influence matrix has {influence} vectors for {estimates} estimates")]
    ShapeMismatch { influence: usize, estimates: usize },
    #[error("influence vectors must all have the unit count; got {actual}, expected {expected}")]
    RaggedInfluence { expected: usize, actual: usize },
    #[error("unit_groups has {actual} entries but the influence vectors have {expected}")]
    UnitGroupMismatch { expected: usize, actual: usize },
    #[error("no post-treatment cells survive the requested aggregation")]
    NoPostTreatmentCells,
}

/// The per-unit inputs the weights are estimated from.
#[derive(Debug, Clone)]
pub struct UnitPanel {
    /// Each unit's cohort, `None` for never-treated. Indexed like the influence
    /// vectors: position in the sorted distinct unit ids.
    pub groups: Vec<Option<i32>>,
    /// Sampling weights, one per unit. `did` calls these `weights.ind` and
    /// defaults them to 1.
    pub weights: Vec<f64>,
    /// Cluster label per unit, `None` to treat each unit as its own cluster.
    ///
    /// Set this when units are not independent. In a matched family design the
    /// two parents of one child are two units sharing every shock to that
    /// household, so the family is the cluster and the unit is not.
    pub clusters: Option<Vec<i64>>,
}

impl UnitPanel {
    /// Unweighted, unclustered panel.
    #[must_use]
    pub fn unweighted(groups: Vec<Option<i32>>) -> Self {
        let weights = vec![1.0; groups.len()];
        Self {
            groups,
            weights,
            clusters: None,
        }
    }

    /// The same panel with a cluster label attached to each unit.
    #[must_use]
    pub fn clustered_by(mut self, clusters: Vec<i64>) -> Self {
        self.clusters = Some(clusters);
        self
    }

    /// Compact cluster assignments and the cluster count.
    ///
    /// `None` clusters yields one cluster per unit, which is the identity case:
    /// [`standard_error`] then reduces exactly to the unclustered formula, and
    /// `clustering_by_unit_reproduces_the_unclustered_standard_error` pins that.
    fn cluster_index(&self) -> (Vec<usize>, usize) {
        match self.clusters {
            None => ((0..self.groups.len()).collect(), self.groups.len()),
            Some(ref labels) => {
                let mut distinct = labels.clone();
                distinct.sort_unstable();
                distinct.dedup();
                let position = distinct
                    .iter()
                    .enumerate()
                    .map(|(index, label)| (*label, index))
                    .collect::<BTreeMap<i64, usize>>();
                let assignments = labels
                    .iter()
                    .map(|label| position[label])
                    .collect::<Vec<usize>>();
                (assignments, distinct.len())
            }
        }
    }
}

/// Standard error of an aggregate from its influence function, clustered.
///
/// Units are centred first and then summed within cluster, rather than the other
/// way round. That ordering is what makes the singleton-cluster case reduce
/// exactly to [`standard_error_from_influence`], so switching clustering on for a
/// design that has none cannot silently move a number.
///
/// No `G / (G - 1)` finite-cluster correction, which is the choice `did` makes
/// and which [`crate::inference::clustered_variance_from_index`] makes
/// differently. The two are not interchangeable; this one is here because the
/// aggregation has to agree with `did`.
fn standard_error(influence: &[f64], assignments: &[usize], n_clusters: usize) -> f64 {
    let n_f = usize_to_f64(influence.len());
    let mean = influence.iter().sum::<f64>() / n_f;
    let mut sums = vec![0.0; n_clusters];
    for (value, &cluster) in influence.iter().zip(assignments) {
        sums[cluster] += value - mean;
    }
    sums.iter().map(|value| value * value).sum::<f64>().sqrt() / n_f
}

/// Cohort share `pg = sum_i w_i * 1(G_i = g) / n`, as `did` computes it.
///
/// Note the denominator is the FULL unit count, never-treated included. It
/// cancels out of the normalised weights, but not out of [`weight_influence`],
/// so it has to be the same `n` in both.
fn cohort_share(panel: &UnitPanel, group: i32) -> f64 {
    let n = usize_to_f64(panel.groups.len());
    let total: f64 = panel
        .groups
        .iter()
        .zip(&panel.weights)
        .filter(|(unit_group, _)| **unit_group == Some(group))
        .map(|(_, weight)| *weight)
        .sum();
    total / n
}

/// The influence function of the estimated weights, ported from `did:::wif`.
///
/// Returns a `n_units x n_cells` matrix in row-major order. Without this term
/// the standard error of any cohort-share-weighted summary is too small,
/// because it treats a quantity estimated from the sample as if it were known.
fn weight_influence(panel: &UnitPanel, groups: &[i32], shares: &[f64]) -> Vec<Vec<f64>> {
    let share_total: f64 = shares.iter().sum();
    let n_units = panel.groups.len();

    // `if2` needs the row sum of the centred indicators, so it is built once.
    let row_sums = (0..n_units)
        .map(|unit| {
            groups
                .iter()
                .zip(shares)
                .map(|(group, share)| indicator(panel, unit, *group) - share)
                .sum::<f64>()
        })
        .collect::<Vec<f64>>();

    (0..n_units)
        .map(|unit| {
            groups
                .iter()
                .zip(shares)
                .map(|(group, share)| {
                    let if1 = (indicator(panel, unit, *group) - share) / share_total;
                    let if2 = row_sums[unit] * share / (share_total * share_total);
                    if1 - if2
                })
                .collect()
        })
        .collect()
}

fn indicator(panel: &UnitPanel, unit: usize, group: i32) -> f64 {
    if panel.groups[unit] == Some(group) {
        panel.weights[unit]
    } else {
        0.0
    }
}

/// Combine selected cells with cohort-share weights, correcting for the fact
/// that those weights were estimated.
fn combine_by_share(
    cells: &[usize],
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    panel: &UnitPanel,
) -> (f64, Vec<f64>) {
    let groups = cells
        .iter()
        .map(|&c| estimates[c].group)
        .collect::<Vec<_>>();
    let shares = groups
        .iter()
        .map(|&g| cohort_share(panel, g))
        .collect::<Vec<_>>();
    let share_total: f64 = shares.iter().sum();
    let weights = shares.iter().map(|s| s / share_total).collect::<Vec<_>>();

    let att = cells
        .iter()
        .zip(&weights)
        .map(|(&c, w)| w * estimates[c].att)
        .sum::<f64>();

    let wif = weight_influence(panel, &groups, &shares);
    let combined = (0..panel.groups.len())
        .map(|unit| {
            let weighted = cells
                .iter()
                .zip(&weights)
                .map(|(&c, w)| w * influence[c][unit])
                .sum::<f64>();
            let correction = wif[unit]
                .iter()
                .zip(cells)
                .map(|(w, &c)| w * estimates[c].att)
                .sum::<f64>();
            weighted + correction
        })
        .collect();

    (att, combined)
}

/// Combine selected cells with equal weights. No correction: `1/k` is not
/// estimated from the sample.
fn combine_equally(
    cells: &[usize],
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    n_units: usize,
) -> (f64, Vec<f64>) {
    let weight = 1.0 / usize_to_f64(cells.len());
    let att = cells.iter().map(|&c| weight * estimates[c].att).sum();
    let combined = (0..n_units)
        .map(|unit| cells.iter().map(|&c| weight * influence[c][unit]).sum())
        .collect();
    (att, combined)
}

/// Average already-aggregated points with equal weights.
fn average_points(points: &[&AggteEstimate], n_units: usize) -> (f64, Vec<f64>) {
    let weight = 1.0 / usize_to_f64(points.len());
    let att = points.iter().map(|p| weight * p.att).sum();
    let influence = (0..n_units)
        .map(|unit| points.iter().map(|p| weight * p.influence[unit]).sum())
        .collect();
    (att, influence)
}

fn finish(
    key: i32,
    att: f64,
    influence: Vec<f64>,
    z: f64,
    assignments: &[usize],
    n_clusters: usize,
) -> AggteEstimate {
    let se = standard_error(&influence, assignments, n_clusters);
    AggteEstimate {
        key,
        att,
        se,
        ci_low: att - z * se,
        ci_high: att + z * se,
        influence,
    }
}

/// Aggregate `ATT(g,t)` the way `did::aggte` does.
///
/// `influence` must be unit-indexed and on the full-sample scale, which is what
/// [`super::panel_pairs::estimate_att_gt_dr_panel_with_influence`] returns.
///
/// # Errors
/// See [`AggteError`].
#[expect(
    clippy::too_many_lines,
    reason = "the four aggregation types differ in which cells they select and \
              how they weight them; splitting them apart hides the comparison"
)]
pub fn aggregate_att_gt(
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    panel: &UnitPanel,
    config: AggteConfig,
) -> Result<AggteResult, AggteError> {
    if estimates.is_empty() {
        return Err(AggteError::EmptyInput);
    }
    if influence.len() != estimates.len() {
        return Err(AggteError::ShapeMismatch {
            influence: influence.len(),
            estimates: estimates.len(),
        });
    }
    let n_units = panel.groups.len();
    if panel.weights.len() != n_units {
        return Err(AggteError::UnitGroupMismatch {
            expected: n_units,
            actual: panel.weights.len(),
        });
    }
    if let Some(bad) = influence.iter().find(|vector| vector.len() != n_units) {
        return Err(AggteError::RaggedInfluence {
            expected: n_units,
            actual: bad.len(),
        });
    }

    let z = z_score_for_confidence(config.confidence_level.confidence_level);
    let (assignments, n_clusters) = panel.cluster_index();
    if assignments.len() != n_units {
        return Err(AggteError::UnitGroupMismatch {
            expected: n_units,
            actual: assignments.len(),
        });
    }
    let post = (0..estimates.len())
        .filter(|&i| estimates[i].time >= estimates[i].group)
        .collect::<Vec<_>>();

    match config.aggregation {
        AggteType::Simple => {
            if post.is_empty() {
                return Err(AggteError::NoPostTreatmentCells);
            }
            let (att, influence_vector) = combine_by_share(&post, estimates, influence, panel);
            let summary = finish(0, att, influence_vector, z, &assignments, n_clusters);
            Ok(AggteResult {
                overall_att: summary.att,
                overall_se: summary.se,
                overall_ci_low: summary.ci_low,
                overall_ci_high: summary.ci_high,
                overall_influence: summary.influence,
                by_key: Vec::new(),
            })
        }
        AggteType::Dynamic => {
            // balance_e keeps only cohorts with at least `b` post-treatment
            // periods observed, and then shows event times up to `b`. Both halves
            // are needed: without the first the composition still moves, without
            // the second the path runs past where every kept cohort is present.
            let max_time = estimates.iter().map(|e| e.time).max().unwrap_or_default();
            let kept_groups = config.balance_e.map(|b| {
                estimates
                    .iter()
                    .map(|e| e.group)
                    .filter(|g| max_time - g >= b)
                    .collect::<BTreeSet<i32>>()
            });

            let mut buckets = BTreeMap::<i32, Vec<usize>>::new();
            for (index, estimate) in estimates.iter().enumerate() {
                if kept_groups
                    .as_ref()
                    .is_some_and(|keep| !keep.contains(&estimate.group))
                {
                    continue;
                }
                if config.balance_e.is_some_and(|b| estimate.event_time > b) {
                    continue;
                }
                if config.min_e.is_some_and(|e| estimate.event_time < e)
                    || config.max_e.is_some_and(|e| estimate.event_time > e)
                {
                    continue;
                }
                buckets.entry(estimate.event_time).or_default().push(index);
            }

            let by_key = buckets
                .into_iter()
                .map(|(event_time, cells)| {
                    let (att, vector) = combine_by_share(&cells, estimates, influence, panel);
                    finish(event_time, att, vector, z, &assignments, n_clusters)
                })
                .collect::<Vec<_>>();

            let positive = by_key.iter().filter(|p| p.key >= 0).collect::<Vec<_>>();
            if positive.is_empty() {
                return Err(AggteError::NoPostTreatmentCells);
            }
            let (overall_att, overall_influence) = average_points(&positive, n_units);
            let overall = finish(
                0,
                overall_att,
                overall_influence,
                z,
                &assignments,
                n_clusters,
            );
            Ok(AggteResult {
                overall_att: overall.att,
                overall_se: overall.se,
                overall_ci_low: overall.ci_low,
                overall_ci_high: overall.ci_high,
                overall_influence: overall.influence,
                by_key,
            })
        }
        AggteType::Group => {
            if post.is_empty() {
                return Err(AggteError::NoPostTreatmentCells);
            }
            let mut buckets = BTreeMap::<i32, Vec<usize>>::new();
            for &index in &post {
                buckets
                    .entry(estimates[index].group)
                    .or_default()
                    .push(index);
            }
            // Equal weights WITHIN a cohort: an ATT(g,t) for one cohort is
            // averaged over its own post periods, and 1/k is not estimated.
            let by_key = buckets
                .into_iter()
                .map(|(group, cells)| {
                    let (att, vector) = combine_equally(&cells, estimates, influence, n_units);
                    finish(group, att, vector, z, &assignments, n_clusters)
                })
                .collect::<Vec<_>>();

            // Cohort shares ACROSS cohorts, so the correction applies here.
            let shares = by_key
                .iter()
                .map(|point| cohort_share(panel, point.key))
                .collect::<Vec<_>>();
            let share_total: f64 = shares.iter().sum();
            let weights = shares.iter().map(|s| s / share_total).collect::<Vec<_>>();
            let groups = by_key.iter().map(|point| point.key).collect::<Vec<_>>();
            let atts = by_key.iter().map(|point| point.att).collect::<Vec<_>>();
            let wif = weight_influence(panel, &groups, &shares);

            let overall_att = by_key
                .iter()
                .zip(&weights)
                .map(|(point, w)| w * point.att)
                .sum::<f64>();
            let overall_influence = (0..n_units)
                .map(|unit| {
                    let weighted = by_key
                        .iter()
                        .zip(&weights)
                        .map(|(point, w)| w * point.influence[unit])
                        .sum::<f64>();
                    let correction = wif[unit]
                        .iter()
                        .zip(&atts)
                        .map(|(w, att)| w * att)
                        .sum::<f64>();
                    weighted + correction
                })
                .collect();
            let overall = finish(
                0,
                overall_att,
                overall_influence,
                z,
                &assignments,
                n_clusters,
            );
            Ok(AggteResult {
                overall_att: overall.att,
                overall_se: overall.se,
                overall_ci_low: overall.ci_low,
                overall_ci_high: overall.ci_high,
                overall_influence: overall.influence,
                by_key,
            })
        }
        AggteType::Calendar => {
            if post.is_empty() {
                return Err(AggteError::NoPostTreatmentCells);
            }
            let mut buckets = BTreeMap::<i32, Vec<usize>>::new();
            for &index in &post {
                buckets
                    .entry(estimates[index].time)
                    .or_default()
                    .push(index);
            }
            let by_key = buckets
                .into_iter()
                .map(|(time, cells)| {
                    let (att, vector) = combine_by_share(&cells, estimates, influence, panel);
                    finish(time, att, vector, z, &assignments, n_clusters)
                })
                .collect::<Vec<_>>();

            let all = by_key.iter().collect::<Vec<_>>();
            let (overall_att, overall_influence) = average_points(&all, n_units);
            let overall = finish(
                0,
                overall_att,
                overall_influence,
                z,
                &assignments,
                n_clusters,
            );
            Ok(AggteResult {
                overall_att: overall.att,
                overall_se: overall.se,
                overall_ci_low: overall.ci_low,
                overall_ci_high: overall.ci_high,
                overall_influence: overall.influence,
                by_key,
            })
        }
    }
}

/// Per-cell standard errors and intervals for `ATT(g,t)`, clustered.
///
/// # Why this exists
///
/// The `se` on an [`AttGtEstimate`] comes from the pair estimator, which sees
/// one `(g, t)` cell and has no idea that two of its units are the two parents of
/// one child. It is therefore an unclustered standard error, and in a design
/// where units share a household it is the wrong one.
///
/// `did` reports the clustered version whenever `clustervars` is set, under both
/// `bstrap = TRUE` and `bstrap = FALSE`. Under `FALSE` it is analytic, and this
/// is that: measured against `did` 2.5.1 on mpdta with counties clustered into
/// states, `sqrt(sum_c s_c^2) / n` where `s_c` is the cluster's summed influence.
/// No `G / (G - 1)` correction, which is the same convention as the aggregation
/// above and NOT the one
/// [`crate::inference::clustered_variance_from_index`] uses.
///
/// The effect is not marginal. On that data the first cell goes from 0.0221
/// unclustered to 0.0102 clustered, and another from 0.0312 to 0.0489. It moves
/// in both directions, so there is no safe direction to round in.
///
/// Pass a [`UnitPanel`] whose `clusters` are set. With `clusters: None` every
/// unit is its own cluster and the result reproduces the input `se`, which
/// `clustered_per_cell_reduces_to_the_pair_estimator` pins.
///
/// # Errors
/// [`AggteError::ShapeMismatch`] if the influence matrix does not match the
/// estimates, [`AggteError::RaggedInfluence`] if the vectors are not all the unit
/// count, [`AggteError::UnitGroupMismatch`] if the panel is a different size.
pub fn att_gt_clustered_standard_errors(
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    panel: &UnitPanel,
    inference: InferenceConfig,
) -> Result<Vec<AttGtEstimate>, AggteError> {
    if estimates.is_empty() {
        return Err(AggteError::EmptyInput);
    }
    if influence.len() != estimates.len() {
        return Err(AggteError::ShapeMismatch {
            influence: influence.len(),
            estimates: estimates.len(),
        });
    }
    let n_units = panel.groups.len();
    if let Some(bad) = influence.iter().find(|vector| vector.len() != n_units) {
        return Err(AggteError::RaggedInfluence {
            expected: n_units,
            actual: bad.len(),
        });
    }
    let (assignments, n_clusters) = panel.cluster_index();
    if assignments.len() != n_units {
        return Err(AggteError::UnitGroupMismatch {
            expected: n_units,
            actual: assignments.len(),
        });
    }

    let z = z_score_for_confidence(inference.confidence_level);
    Ok(estimates
        .iter()
        .zip(influence)
        .map(|(estimate, vector)| {
            let se = standard_error(vector, &assignments, n_clusters);
            let margin = z * se;
            AttGtEstimate {
                se,
                ci_low: estimate.att - margin,
                ci_high: estimate.att + margin,
                ..*estimate
            }
        })
        .collect())
}

/// A [`UnitPanel`] for the repeated-cross-section route, indexed by input ROW.
///
/// [`super::panel_pairs::unit_panel`] is indexed by unit, because the panel
/// route's influence vectors are. The RC route's are indexed by input row, one
/// entry per observation, so its panel has to be too. Passing the wrong one is
/// caught by the length check in [`aggregate_att_gt`] rather than being absorbed
/// silently, but the two are easy to confuse, hence two named constructors
/// instead of one that guesses.
///
/// Cohort shares come out the same either way on a balanced panel, since every
/// unit contributes the same number of rows. On an unbalanced one they do not,
/// and the row-level share is the right one here: it is what `did` computes with
/// `panel = FALSE`, where a row IS the observation.
#[must_use]
pub fn row_panel(observations: &[crate::types::AttGtDrObservation]) -> UnitPanel {
    UnitPanel {
        groups: observations
            .iter()
            .map(|row| row.first_treated_time)
            .collect(),
        weights: observations.iter().map(|row| row.weight).collect(),
        clusters: None,
    }
}
