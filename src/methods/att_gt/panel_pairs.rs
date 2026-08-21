//! Panel `ATT(g,t)` with covariate adjustment, matching `did::att_gt(panel = TRUE)`.
//!
//! # Why this exists separately from [`super::pair_estimators`]
//!
//! That module routes every covariate-adjusted cell through
//! [`estimate_drdid_repeated_cross_section`](crate::methods::drdid::repeated::estimate_drdid_repeated_cross_section),
//! which treats the baseline and follow-up periods as two independent samples.
//! On panel data they are the same units observed twice, and ignoring the
//! pairing inflates the variance. R's `did` defaults to `panel = TRUE` and uses
//! `DRDID::drdid_panel`; this module is that route.
//!
//! # The alignment convention, which is the whole correctness argument
//!
//! The repeated-cross-section path aligns each cell's influence function to the
//! **input row** index and pads with zeros. That is right for it: one row is one
//! observation. For panel data one *unit* is one observation, and the influence
//! function `DRDID::drdid_panel` returns has one entry per unit, not per row.
//!
//! This matters numerically rather than cosmetically. Measured against
//! `did` 2.5.1 on `mpdta` (`tests/cs_panel_dr_universal_ref.json`), R's reported
//! standard error is exactly
//!
//! ```text
//! se = sqrt(sum(psi^2)) / n_units
//! ```
//!
//! and [`standard_error_from_influence`](crate::inference::standard_error_from_influence)
//! computes the same quantity from a vector of length `n`. So a unit-indexed
//! vector reproduces R and a row-indexed one is wrong by a factor of the number
//! of periods: on `mpdta`, five. Every aggregation, every simultaneous band and
//! the whole `HonestDiD` surface is a function of this matrix, so the factor
//! would propagate everywhere and fail silently.
//!
//! Influence vectors from this module are therefore indexed by **position in the
//! sorted list of distinct `unit_id` values across the whole input**, and units
//! absent from a cell carry zero.

use std::collections::BTreeMap;

use crate::methods::drdid::panel::{PanelFlatInput, estimate_drdid_panel_flat};
use crate::types::{
    AttGtDrConfig, AttGtDrObservation, AttGtError, AttGtEstimate, AttGtInfluenceOutput,
};

/// One unit's contribution to a single `(g, t)` cell.
struct PanelUnit<'a> {
    /// Index into the sorted distinct-unit list, for influence alignment.
    unit_index: usize,
    treated: bool,
    baseline_outcome: Option<f64>,
    follow_up_outcome: Option<f64>,
    weight: f64,
    /// Taken from the BASELINE row, which is what `did` does: covariates are
    /// measured before treatment so that treatment cannot have moved them.
    ///
    /// BORROWED from that row rather than cloned. This was a `Vec<f64>` filled
    /// by `clone_from` once per unit per cell, and the cell loop runs hundreds
    /// of times per slice: on Study I's shape that is millions of small
    /// allocations whose contents never change.
    covariates: &'a [f64],
}

/// Buffers the cell loop reuses, so the per-cell cost is the cell's own size.
///
/// `collect_cell_units` built a `BTreeMap<i64, PanelUnit>` per cell: a node
/// allocation and an O(log n) descent for every row of every cell, to rebuild a
/// mapping `unit_universe` had already computed. `slots` is that mapping made
/// dense, indexed by the unit position the influence vectors are aligned on
/// anyway.
///
/// `touched` is what keeps the reuse honest. Clearing `slots` wholesale would
/// cost the WHOLE panel per cell, which is worse than the map for a small cell;
/// clearing only the entries a cell wrote costs the cell.
struct CellScratch<'a> {
    slots: Vec<Option<PanelUnit<'a>>>,
    touched: Vec<usize>,
    units: Vec<PanelUnit<'a>>,
    treated: Vec<bool>,
    delta: Vec<f64>,
    weights: Vec<f64>,
    design: Vec<f64>,
}

impl<'a> CellScratch<'a> {
    fn new(unit_count: usize) -> Self {
        let mut slots = Vec::new();
        slots.resize_with(unit_count, || None);
        Self {
            slots,
            touched: Vec::new(),
            units: Vec::new(),
            treated: Vec::new(),
            delta: Vec::new(),
            weights: Vec::new(),
            design: Vec::new(),
        }
    }

    fn reset(&mut self) {
        for &index in &self.touched {
            self.slots[index] = None;
        }
        self.touched.clear();
        self.units.clear();
        self.treated.clear();
        self.delta.clear();
        self.weights.clear();
        self.design.clear();
    }
}

/// The sorted distinct `unit_id` values, and a lookup from id to position.
///
/// # Errors
/// Returns [`AttGtError::MissingUnitId`] if any row lacks a unit id. A panel
/// estimator cannot difference a unit it cannot name, and guessing (say, by row
/// order) would produce a plausible number from mismatched units.
fn unit_universe(observations: &[AttGtDrObservation]) -> Result<BTreeMap<i64, usize>, AttGtError> {
    let mut ids = observations
        .iter()
        .map(|row| row.unit_id.ok_or(AttGtError::MissingUnitId))
        .collect::<Result<Vec<i64>, AttGtError>>()?;
    ids.sort_unstable();
    ids.dedup();
    Ok(ids
        .into_iter()
        .enumerate()
        .map(|(position, id)| (id, position))
        .collect())
}

/// Collect the units contributing to one `(group, time)` cell against `baseline_time`.
///
/// Only units observed at BOTH periods survive. A unit seen at one period is not
/// a panel observation for this cell, and `did` drops it the same way.
fn collect_cell_units<'a>(
    observations: &'a [AttGtDrObservation],
    positions: &BTreeMap<i64, usize>,
    group: i32,
    time: i32,
    baseline_time: i32,
    config: AttGtDrConfig,
    scratch: &mut CellScratch<'a>,
) -> Result<(), AttGtError> {
    scratch.reset();
    let mut duplicate: Option<i64> = None;

    for row in observations {
        if row.time != baseline_time && row.time != time {
            continue;
        }
        let treated = row.first_treated_time == Some(group);
        // Not-yet-treated eligibility is judged at the LATER of the two periods
        // the cell compares, not at `time`. Under a universal base period a
        // pre-treatment cell has base > time, and a unit treated between the two
        // is already treated when the baseline is read; admitting it as a control
        // contaminates exactly the placebo cells the pre-trend analysis reads.
        // Verified against did 2.5.1 on mpdta cell (g=2006, t=2003, base=2005):
        // `G > t` gives 0.008901391, `G > max(t, base)` gives 0.012018613, and
        // att_gt reports the latter.
        if !treated
            && !super::is_control_for_pair(
                row.first_treated_time,
                time.max(baseline_time),
                config.att_gt.comparison_group,
                config.att_gt.anticipation_periods,
            )
        {
            continue;
        }
        let Some(id) = row.unit_id else { continue };
        let Some(&unit_index) = positions.get(&id) else {
            continue;
        };

        let slot = &mut scratch.slots[unit_index];
        if slot.is_none() {
            scratch.touched.push(unit_index);
            *slot = Some(PanelUnit {
                unit_index,
                treated,
                baseline_outcome: None,
                follow_up_outcome: None,
                weight: row.weight,
                covariates: &[],
            });
        }
        // `is_none` above: the slot is Some here.
        let Some(entry) = slot.as_mut() else { continue };

        // `baseline_time == time` cannot reach here: the caller skips that cell.
        //
        // A second row for the same unit at the same period is a duplicate, and
        // it is reported rather than absorbed. Last-write-wins would silently
        // change the estimate: a caller expressing "this unit counts four times"
        // by repeating its rows gets one unit and no error, and the weight they
        // thought they had applied simply vanishes. Use `weight` for that.
        let outcome_slot = if row.time == baseline_time {
            &mut entry.baseline_outcome
        } else {
            &mut entry.follow_up_outcome
        };
        if outcome_slot.is_some() {
            duplicate = Some(id);
        }
        *outcome_slot = Some(row.outcome);
        if row.time == baseline_time {
            entry.weight = row.weight;
            entry.covariates = row.covariates.as_slice();
        }
    }

    if let Some(unit_id) = duplicate {
        return Err(AttGtError::DuplicatePanelRow {
            unit_id,
            group,
            time,
        });
    }

    // Ascending unit position, which is ascending unit id, which is the order
    // the `BTreeMap` produced. Kept exactly: the design matrix is summed in this
    // order, so a different one would move the last bits of every estimate.
    scratch.touched.sort_unstable();
    for index in 0..scratch.touched.len() {
        let slot = scratch.touched[index];
        if let Some(unit) = scratch.slots[slot].take()
            && unit.baseline_outcome.is_some()
            && unit.follow_up_outcome.is_some()
        {
            scratch.units.push(unit);
        }
    }
    // `take` above emptied every slot this cell wrote, so the next `reset` has
    // nothing left to clear but the bookkeeping.
    Ok(())
}

/// Estimate one `(g, t)` cell and return the estimate plus a unit-aligned
/// influence vector of length `unit_count`.
fn estimate_panel_cell(
    scratch: &mut CellScratch<'_>,
    unit_count: usize,
    group: i32,
    time: i32,
    config: AttGtDrConfig,
) -> Result<(AttGtEstimate, Vec<f64>), &'static str> {
    let treated_count = scratch.units.iter().filter(|unit| unit.treated).count();
    if treated_count == 0 {
        return Err("treated_panel");
    }
    if treated_count == scratch.units.len() {
        return Err("control_panel");
    }

    // The design built FLAT, straight into a buffer the cell loop reuses.
    //
    // This was a `Vec<DrDidObservation>`, each owning a fresh `Vec<f64>` of the
    // intercept and the unit's covariates, which `prepare_panel_data` then
    // copied into a flat buffer of its own. Two allocations and two copies per
    // unit per cell, to move numbers that were already sitting in the caller's
    // rows. `estimate_drdid_panel_flat` takes the flat form directly.
    //
    // The intercept is prepended HERE rather than left to the pair estimator,
    // and the asymmetry is deliberate. The two DR routes in this crate disagree
    // about whose job it is: `estimate_drdid_repeated_cross_section` always
    // prepends one (`repeated.rs`, feature_count = covariate_count + 1), while
    // the panel route treats its covariates as a finished design matrix and adds
    // a constant column only when there are none at all. Left alone, the same
    // `AttGtDrObservation` would be adjusted for `1 + X` through the RC route
    // and for `X` alone through the panel route, which is a difference in the
    // fitted model that produces a plausible number and fails nothing. R's
    // `xformla = ~ x` means intercept plus x, so prepending here makes both
    // routes agree with each other and with `did`.
    let covariate_count = scratch
        .units
        .first()
        .map_or(0, |unit| unit.covariates.len());
    let feature_count = covariate_count + 1;
    scratch.treated.reserve(scratch.units.len());
    scratch.delta.reserve(scratch.units.len());
    scratch.weights.reserve(scratch.units.len());
    scratch.design.reserve(scratch.units.len() * feature_count);
    for unit in &scratch.units {
        if unit.covariates.len() != covariate_count {
            return Err("covariate_count");
        }
        scratch.treated.push(unit.treated);
        // Both are Some by construction: collect_cell_units filters on it.
        scratch.delta.push(
            unit.follow_up_outcome.unwrap_or_default() - unit.baseline_outcome.unwrap_or_default(),
        );
        scratch.weights.push(unit.weight);
        scratch.design.push(1.0);
        scratch.design.extend_from_slice(unit.covariates);
    }

    let fit = estimate_drdid_panel_flat(
        PanelFlatInput {
            treated: &scratch.treated,
            delta_outcome: &scratch.delta,
            weight: &scratch.weights,
            design_matrix_flat: &scratch.design,
            feature_count,
        },
        config.drdid,
    )
    .map_err(|_| "panel_fit")?;
    if fit.influence_function.len() != scratch.units.len() {
        return Err("influence_length");
    }

    // Rescale from the cell's own sample to the full one. `estimate_drdid_panel`
    // returns psi normalised so that sqrt(sum(psi^2)) / n_cell is the standard
    // error; padding with zeros and dividing by n_total instead would shrink
    // every SE by n_cell / n_total, and by a DIFFERENT factor per cell, since
    // cells differ in size. That would leave a covariance matrix whose blocks are
    // on incompatible scales, which no single rescale downstream could repair.
    // `did` embeds its influence functions in the full sample the same way:
    // measured on mpdta cell (2004, 2005), R's psi is ours times 500/329.
    let scale =
        crate::util::usize_to_f64(unit_count) / crate::util::usize_to_f64(scratch.units.len());
    let mut aligned = vec![0.0; unit_count];
    for (unit, value) in scratch.units.iter().zip(&fit.influence_function) {
        aligned[unit.unit_index] = value * scale;
    }

    Ok((
        AttGtEstimate {
            group,
            time,
            event_time: time - group,
            att: fit.att,
            se: fit.se,
            ci_low: fit.ci_low,
            ci_high: fit.ci_high,
            treated_n: fit.treated_n,
            control_n: fit.control_n,
            total_weight: fit.total_weight,
        },
        aligned,
    ))
}

/// Panel `ATT(g,t)` with influence functions, aligned to the unit index.
///
/// The covariate-adjusted counterpart of `did::att_gt(..., panel = TRUE,
/// est_method = "dr")`. Set
/// [`BasePeriod::Universal`](crate::types::BasePeriod::Universal) to anchor every
/// cell at `g - 1 - anticipation`, which is what a fixed-baseline event study
/// means; R's default is `"varying"` and produces different coefficients.
///
/// # Errors
/// - [`AttGtError::MissingUnitId`] if any row lacks a unit id.
/// - [`AttGtError::MissingCell`] if a cell has no treated or no control units
///   and `skip_incomplete_pairs` is false.
/// - [`AttGtError::NoEstimablePairs`] if no cell survives.
pub fn estimate_att_gt_dr_panel_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    let (all_times, treated_groups) =
        super::pair_estimators::prepare_att_gt_dr_inputs(observations, config)?;
    let positions = unit_universe(observations)?;
    let unit_count = positions.len();

    let mut estimates = Vec::new();
    let mut influence_functions = Vec::new();
    // Allocated once for the whole grid, not once per cell. See `CellScratch`.
    let mut scratch = CellScratch::new(unit_count);

    for group in treated_groups {
        let universal_baseline_time = group - config.att_gt.anticipation_periods - 1;
        for &time in &all_times {
            let baseline_time = super::pair_estimators::baseline_time_for_pair(
                time,
                group,
                universal_baseline_time,
                config.att_gt,
            );
            if time == baseline_time {
                continue;
            }

            collect_cell_units(
                observations,
                &positions,
                group,
                time,
                baseline_time,
                config,
                &mut scratch,
            )?;

            match estimate_panel_cell(&mut scratch, unit_count, group, time, config) {
                Ok((estimate, influence)) => {
                    estimates.push(estimate);
                    influence_functions.push(influence);
                }
                Err(cell) => {
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
            }
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

/// Panel `ATT(g,t)` without influence functions.
///
/// # Errors
/// As [`estimate_att_gt_dr_panel_with_influence`].
pub fn estimate_att_gt_dr_panel(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<Vec<AttGtEstimate>, AttGtError> {
    estimate_att_gt_dr_panel_with_influence(observations, config).map(|out| out.estimates)
}

/// Each unit's cohort and sampling weight, indexed the same way as the influence
/// vectors this module returns.
///
/// Built here rather than by the caller because the ordering has to be the same
/// `unit_universe` uses, and reproducing it outside would be one more place for
/// the two to drift apart.
///
/// # Errors
/// [`AttGtError::MissingUnitId`] if any row lacks a unit id.
pub fn unit_panel(
    observations: &[AttGtDrObservation],
) -> Result<super::aggte::UnitPanel, AttGtError> {
    let positions = unit_universe(observations)?;
    let mut groups = vec![None; positions.len()];
    let mut weights = vec![1.0; positions.len()];
    for row in observations {
        let Some(id) = row.unit_id else { continue };
        let Some(&position) = positions.get(&id) else {
            continue;
        };
        // A unit's cohort is time-invariant, so the last write is the same as
        // the first. The weight is taken the same way for the same reason.
        groups[position] = row.first_treated_time;
        weights[position] = row.weight;
    }
    Ok(super::aggte::UnitPanel {
        groups,
        weights,
        // Unclustered by default. A caller who knows units share a household,
        // a family or a matched set attaches the labels with
        // `UnitPanel::clustered_by`; nothing in an ATT(g,t) input frame reveals
        // that structure, so it cannot be inferred here.
        clusters: None,
    })
}
