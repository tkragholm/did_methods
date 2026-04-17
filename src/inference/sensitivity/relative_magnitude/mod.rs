//! Relative-magnitude (`DeltaRM`) sensitivity analysis for event-study `ATT(g,t)`.
//!
//! This module implements the `\Delta^{RM}(\bar M)` restrictions from
//! Rambachan & Roth (2023), where post-period violations are bounded by a
//! multiple of the largest observed pre-period deviation from linear trend.
//!
//! The implementation is split into:
//! - public identified-set and conditional-CS entrypoints in this file
//! - branch geometry / constraint preparation in [`geometry`]
//! - Clarabel-backed confidence-set solver machinery in [`confidence_set`]
//!
//! References:
//! - Rambachan, A. and Roth, J. (2023), "A More Credible Approach to Parallel Trends"
//! - `HonestDiD` R package, `DeltaRM` path

pub(in crate::inference::sensitivity) mod conditional_confidence_set;
pub(in crate::inference::sensitivity) mod geometry;
pub mod least_favorable_intervals;

use rayon::prelude::*;

use crate::inference::z_score_for_confidence;
use crate::types::InferenceConfig;

use super::adaptive_grid::{AcceptedGridRange, nearest_grid_index};
use super::{
    HonestBiasDirection, HonestConditionalConfidenceSet, HonestEventStudyInput,
    HonestIdentifiedSet, HonestMonotonicityDirection, HonestOriginalConfidenceSet,
    RelativeMagnitudeConfidenceSetConfig,
};
use crate::inference::sensitivity::linear_algebra::linear_grid;
use crate::inference::sensitivity::smoothness;
use conditional_confidence_set::{
    RelativeMagnitudeConditionalBranch,
    compute_relative_magnitude_branch_accepted_range_for_matrix,
    solve_relative_magnitude_branch_with_matrix,
};
use geometry::prepare_relative_magnitude_functional_transform;

pub(in crate::inference::sensitivity) fn grid_bounds_around_identified_set(
    original: &HonestOriginalConfidenceSet,
    identified: &HonestIdentifiedSet,
) -> (f64, f64) {
    let padding = (20.0 * original.se).max(1e-8);
    let lower = identified.lb.min(original.estimate) - padding;
    let upper = identified.ub.max(original.estimate) + padding;
    if upper > lower {
        (lower, upper)
    } else {
        (lower - padding, upper + padding)
    }
}

#[derive(Debug, Clone, Copy)]
enum RelativeMagnitudeFamily {
    Base,
    SignedBase(HonestBiasDirection),
    MonotoneBase(HonestMonotonicityDirection),
    Linear,
    SignedLinear(HonestBiasDirection),
    MonotoneLinear(HonestMonotonicityDirection),
}

impl RelativeMagnitudeFamily {
    const fn description(self) -> &'static str {
        match self {
            Self::Base => "relative-magnitude sensitivity",
            Self::SignedBase(_) => "sign-restricted relative-magnitude sensitivity",
            Self::MonotoneBase(_) => "monotone relative-magnitude sensitivity",
            Self::Linear => "linear-trend relative-magnitude sensitivity",
            Self::SignedLinear(_) => "sign-restricted linear-trend relative-magnitude sensitivity",
            Self::MonotoneLinear(_) => "monotone linear-trend relative-magnitude sensitivity",
        }
    }

    const fn requires_two_pre_periods(self) -> bool {
        matches!(
            self,
            Self::Linear | Self::SignedLinear(_) | Self::MonotoneLinear(_)
        )
    }

    fn min_s(self, num_pre: usize) -> Result<isize, String> {
        let num_pre = isize::try_from(num_pre).map_err(|_| "too many pre-periods".to_string())?;
        Ok(match self {
            Self::Base | Self::SignedBase(_) | Self::MonotoneBase(_) => -(num_pre - 1),
            Self::Linear | Self::SignedLinear(_) | Self::MonotoneLinear(_) => -(num_pre - 2),
        })
    }

    const fn empty_message(self) -> &'static str {
        match self {
            Self::Base => "relative-magnitude conditional confidence set accepted no grid points",
            Self::SignedBase(_) => {
                "sign-restricted relative-magnitude conditional confidence set accepted no grid points"
            }
            Self::MonotoneBase(_) => {
                "monotone relative-magnitude conditional confidence set accepted no grid points"
            }
            Self::Linear => {
                "linear-trend relative-magnitude conditional confidence set accepted no grid points"
            }
            Self::SignedLinear(_) => {
                "sign-restricted linear-trend relative-magnitude conditional confidence set accepted no grid points"
            }
            Self::MonotoneLinear(_) => {
                "monotone linear-trend relative-magnitude conditional confidence set accepted no grid points"
            }
        }
    }

    fn build_constraint_matrix(
        self,
        num_pre: usize,
        num_post: usize,
        mbar: f64,
        s: isize,
        max_positive: bool,
    ) -> Result<Vec<Vec<f64>>, String> {
        match self {
            Self::Base => geometry::build_relative_magnitude_constraint_matrix(
                num_pre,
                num_post,
                mbar,
                s,
                max_positive,
            ),
            Self::SignedBase(direction) => build_signed_parallel_trend_constraint_matrix(
                num_pre,
                num_post,
                mbar,
                s,
                max_positive,
                direction,
            ),
            Self::MonotoneBase(direction) => build_monotone_parallel_trend_constraint_matrix(
                num_pre,
                num_post,
                mbar,
                s,
                max_positive,
                direction,
            ),
            Self::Linear => build_linear_trend_relative_magnitude_constraint_matrix(
                num_pre,
                num_post,
                mbar,
                s,
                max_positive,
            ),
            Self::SignedLinear(direction) => build_signed_linear_trend_constraint_matrix(
                num_pre,
                num_post,
                mbar,
                s,
                max_positive,
                direction,
            ),
            Self::MonotoneLinear(direction) => build_monotone_linear_trend_constraint_matrix(
                num_pre,
                num_post,
                mbar,
                s,
                max_positive,
                direction,
            ),
        }
    }
}

fn validate_post_weights(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    require_nonzero: bool,
) -> Result<(), String> {
    if post_weights.len() != input.num_post_periods() {
        return Err(format!(
            "post_weights length {} does not match number of post periods {}",
            post_weights.len(),
            input.num_post_periods()
        ));
    }
    if post_weights.iter().any(|weight| !weight.is_finite()) {
        return Err("post_weights weights must be finite".to_string());
    }
    if require_nonzero && post_weights.iter().all(|weight| weight.abs() <= 1e-12) {
        return Err("post_weights must contain at least one non-zero weight".to_string());
    }
    Ok(())
}

fn validate_relative_magnitude_family_inputs(
    family: RelativeMagnitudeFamily,
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
) -> Result<(), String> {
    input.validate()?;
    if !mbar.is_finite() || mbar < 0.0 {
        return Err(format!(
            "{} requires finite non-negative Mbar, got {mbar}",
            family.description()
        ));
    }
    if family.requires_two_pre_periods() && input.num_pre_periods() <= 1 {
        return Err(format!(
            "{} requires at least two pre-periods",
            family.description()
        ));
    }
    validate_post_weights(input, post_weights, true)
}

fn build_signed_parallel_trend_constraint_matrix(
    num_pre: usize,
    num_post: usize,
    mbar: f64,
    s: isize,
    max_positive: bool,
    bias_direction: HonestBiasDirection,
) -> Result<Vec<Vec<f64>>, String> {
    let mut a_matrix = geometry::build_relative_magnitude_constraint_matrix(
        num_pre,
        num_post,
        mbar,
        s,
        max_positive,
    )?;
    a_matrix.extend(smoothness::build_bias_sign_restriction_matrix(
        num_pre,
        num_post,
        bias_direction,
    ));
    Ok(a_matrix)
}

fn build_monotone_parallel_trend_constraint_matrix(
    num_pre: usize,
    num_post: usize,
    mbar: f64,
    s: isize,
    max_positive: bool,
    monotonicity_direction: HonestMonotonicityDirection,
) -> Result<Vec<Vec<f64>>, String> {
    let mut a_matrix = geometry::build_relative_magnitude_constraint_matrix(
        num_pre,
        num_post,
        mbar,
        s,
        max_positive,
    )?;
    a_matrix.extend(smoothness::build_monotonicity_restriction_matrix(
        num_pre,
        num_post,
        monotonicity_direction,
        false,
    ));
    Ok(a_matrix)
}

fn build_linear_trend_relative_magnitude_constraint_matrix(
    num_pre: usize,
    num_post: usize,
    mbar: f64,
    s: isize,
    max_positive: bool,
) -> Result<Vec<Vec<f64>>, String> {
    let total_diffs = num_pre + num_post - 1;
    let full_cols = num_pre + num_post + 1;
    let mut a_tilde = vec![vec![0.0; full_cols]; total_diffs];
    for (row_idx, row) in a_tilde.iter_mut().enumerate() {
        row[row_idx] = 1.0;
        row[row_idx + 1] = -2.0;
        row[row_idx + 2] = 1.0;
    }

    let start = isize::try_from(num_pre).map_err(|_| "too many pre-periods".to_string())? + s - 2;
    if start < 0
        || start + 2 >= isize::try_from(full_cols).map_err(|_| "too many periods".to_string())?
    {
        return Err(format!(
            "invalid linear-trend relative-magnitude branch index s={s} for num_pre={num_pre}"
        ));
    }
    let start = usize::try_from(start)
        .map_err(|_| "invalid linear-trend relative-magnitude branch index".to_string())?;
    let mut v_max = vec![0.0; full_cols];
    v_max[start] = 1.0;
    v_max[start + 1] = -2.0;
    v_max[start + 2] = 1.0;
    if !max_positive {
        for value in &mut v_max {
            *value = -*value;
        }
    }

    let mut a_ub = Vec::with_capacity(total_diffs);
    for _ in 0..num_pre.saturating_sub(1) {
        a_ub.push(v_max.clone());
    }
    for _ in 0..num_post {
        a_ub.push(v_max.iter().map(|value| mbar * value).collect());
    }

    let mut constraints = Vec::with_capacity(total_diffs * 2);
    for (tilde_row, ub_row) in a_tilde.iter().zip(&a_ub) {
        constraints.push(
            tilde_row
                .iter()
                .zip(ub_row)
                .map(|(left, right)| left - right)
                .collect::<Vec<_>>(),
        );
        constraints.push(
            tilde_row
                .iter()
                .zip(ub_row)
                .map(|(left, right)| -left - right)
                .collect::<Vec<_>>(),
        );
    }

    let zero_col = num_pre;
    let mut dropped = Vec::new();
    for row in constraints {
        let mut compact = Vec::with_capacity(num_pre + num_post);
        for (col_idx, value) in row.into_iter().enumerate() {
            if col_idx != zero_col {
                compact.push(value);
            }
        }
        if compact.iter().any(|value| value.abs() > 1e-10) {
            dropped.push(compact);
        }
    }
    Ok(dropped)
}

fn build_signed_linear_trend_constraint_matrix(
    num_pre: usize,
    num_post: usize,
    mbar: f64,
    s: isize,
    max_positive: bool,
    bias_direction: HonestBiasDirection,
) -> Result<Vec<Vec<f64>>, String> {
    let mut a_matrix = build_linear_trend_relative_magnitude_constraint_matrix(
        num_pre,
        num_post,
        mbar,
        s,
        max_positive,
    )?;
    a_matrix.extend(smoothness::build_bias_sign_restriction_matrix(
        num_pre,
        num_post,
        bias_direction,
    ));
    Ok(a_matrix)
}

fn build_monotone_linear_trend_constraint_matrix(
    num_pre: usize,
    num_post: usize,
    mbar: f64,
    s: isize,
    max_positive: bool,
    monotonicity_direction: HonestMonotonicityDirection,
) -> Result<Vec<Vec<f64>>, String> {
    let mut a_matrix = build_linear_trend_relative_magnitude_constraint_matrix(
        num_pre,
        num_post,
        mbar,
        s,
        max_positive,
    )?;
    a_matrix.extend(smoothness::build_monotonicity_restriction_matrix(
        num_pre,
        num_post,
        monotonicity_direction,
        false,
    ));
    Ok(a_matrix)
}

fn compute_relative_magnitude_identified_set_with_builder<F>(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    min_s: isize,
    build_matrix: F,
) -> Result<HonestIdentifiedSet, String>
where
    F: Fn(isize, bool) -> Result<Vec<Vec<f64>>, String> + Sync,
{
    let num_pre = input.num_pre_periods();
    let num_post = input.num_post_periods();
    if num_pre == 0 {
        return Err("relative-magnitude sensitivity requires at least one pre-period".to_string());
    }
    let target_estimate = post_weights
        .iter()
        .zip(&input.betahat[num_pre..])
        .map(|(weight, beta)| weight * beta)
        .sum::<f64>();
    let branches: Vec<(isize, bool)> = (min_s..=0)
        .flat_map(|s| {
            [true, false]
                .into_iter()
                .map(move |max_positive| (s, max_positive))
        })
        .collect();
    let branch_results: Result<Vec<Option<HonestIdentifiedSet>>, String> = branches
        .into_par_iter()
        .map(|(s, max_positive)| {
            let inequality_matrix = build_matrix(s, max_positive)?;
            solve_relative_magnitude_branch_with_matrix(
                num_pre,
                num_post,
                &input.betahat,
                post_weights,
                &inequality_matrix,
            )
        })
        .collect();
    let branch_results = branch_results?;

    let mut global_lower = f64::INFINITY;
    let mut global_upper = f64::NEG_INFINITY;
    let mut solved_any_branch = false;
    for branch in branch_results.into_iter().flatten() {
        solved_any_branch = true;
        global_lower = global_lower.min(branch.lb);
        global_upper = global_upper.max(branch.ub);
    }
    if solved_any_branch {
        Ok(HonestIdentifiedSet {
            lb: global_lower,
            ub: global_upper,
        })
    } else {
        Ok(HonestIdentifiedSet {
            lb: target_estimate,
            ub: target_estimate,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_relative_magnitude_confidence_set_with_builder<F>(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
    original: &HonestOriginalConfidenceSet,
    identified: &HonestIdentifiedSet,
    min_s: isize,
    build_matrix: F,
    empty_message: &'static str,
) -> Result<HonestConditionalConfidenceSet, String>
where
    F: Fn(isize, bool) -> Result<Vec<Vec<f64>>, String> + Sync,
{
    let num_pre = input.num_pre_periods();
    let (grid_lower, grid_upper) = grid_bounds_around_identified_set(original, identified);
    let grid_points = 1_000usize;
    let alpha = 1.0 - inference.confidence_level;
    let prepared_transform = prepare_relative_magnitude_functional_transform(post_weights)?;
    let grid = linear_grid(grid_lower, grid_upper, grid_points);
    let anchor_idx = nearest_grid_index(&grid, original.estimate);

    let branches: Vec<(isize, bool)> = (min_s..=0)
        .flat_map(|s| {
            [true, false]
                .into_iter()
                .map(move |max_positive| (s, max_positive))
        })
        .collect();
    let branch_acceptance: Result<Vec<Option<AcceptedGridRange>>, String> = branches
        .into_par_iter()
        .map(|(s, max_positive)| {
            let inequality_matrix = build_matrix(s, max_positive)?;
            let branch_problem = RelativeMagnitudeConditionalBranch {
                input,
                num_pre,
                prepared_transform: &prepared_transform,
                alpha,
                hybrid: config.hybrid,
                hybrid_kappa: config.hybrid_kappa,
                grid: &grid,
                anchor_idx,
            };
            compute_relative_magnitude_branch_accepted_range_for_matrix(
                &branch_problem,
                &inequality_matrix,
            )
        })
        .collect();
    let branch_acceptance = branch_acceptance?;
    let accepted_range = branch_acceptance.into_iter().flatten().fold(
        None,
        |acc: Option<AcceptedGridRange>, range| {
            Some(acc.map_or(range, |existing| AcceptedGridRange {
                lower_idx: existing.lower_idx.min(range.lower_idx),
                upper_idx: existing.upper_idx.max(range.upper_idx),
            }))
        },
    );
    let Some(accepted_range) = accepted_range else {
        return Err(empty_message.to_string());
    };
    Ok(HonestConditionalConfidenceSet {
        lb: grid[accepted_range.lower_idx],
        ub: grid[accepted_range.upper_idx],
    })
}

/// Construct the original parallel-trends confidence set for a post-period
/// linear functional.
///
/// This matches `HonestDiD::constructOriginalCS` for a user-supplied
/// post-treatment weight vector `post_weights`.
///
/// # Errors
/// Returns an error if `input` is invalid or `post_weights` does not match the number
/// of post-treatment periods.
pub fn compute_original_confidence_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    inference: InferenceConfig,
) -> Result<HonestOriginalConfidenceSet, String> {
    input.validate()?;
    validate_post_weights(input, post_weights, false)?;
    let post_offset = input.num_pre_periods();
    let post_betahat = &input.betahat[post_offset..];
    let estimate = post_weights
        .iter()
        .zip(post_betahat)
        .map(|(weight, beta)| weight * beta)
        .sum::<f64>();
    let mut variance = 0.0;
    for (i, left_weight) in post_weights.iter().enumerate() {
        for (j, right_weight) in post_weights.iter().enumerate() {
            variance = (left_weight * right_weight)
                .mul_add(input.covariance[post_offset + i][post_offset + j], variance);
        }
    }
    let se = variance.max(0.0).sqrt();
    let z = z_score_for_confidence(inference.confidence_level);
    let margin = z * se;
    Ok(HonestOriginalConfidenceSet {
        estimate,
        se,
        ci: (estimate - margin, estimate + margin),
    })
}

fn compute_relative_magnitude_family_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    family: RelativeMagnitudeFamily,
) -> Result<HonestIdentifiedSet, String> {
    validate_relative_magnitude_family_inputs(family, input, post_weights, mbar)?;
    let num_pre = input.num_pre_periods();
    let num_post = input.num_post_periods();
    let min_s = family.min_s(num_pre)?;
    compute_relative_magnitude_identified_set_with_builder(
        input,
        post_weights,
        min_s,
        move |s, max_positive| {
            family.build_constraint_matrix(num_pre, num_post, mbar, s, max_positive)
        },
    )
}

fn compute_relative_magnitude_family_conditional_cs_with_config(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
    family: RelativeMagnitudeFamily,
) -> Result<HonestConditionalConfidenceSet, String> {
    validate_relative_magnitude_family_inputs(family, input, post_weights, mbar)?;
    config.validate(inference)?;
    let original = compute_original_confidence_set(input, post_weights, inference)?;
    let identified =
        compute_relative_magnitude_family_identified_set(input, post_weights, mbar, family)?;
    compute_relative_magnitude_family_conditional_cs_with_precomputed_sets(
        input,
        post_weights,
        mbar,
        inference,
        config,
        &original,
        &identified,
        family,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_relative_magnitude_family_conditional_cs_with_precomputed_sets(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
    original: &HonestOriginalConfidenceSet,
    identified: &HonestIdentifiedSet,
    family: RelativeMagnitudeFamily,
) -> Result<HonestConditionalConfidenceSet, String> {
    let num_pre = input.num_pre_periods();
    let num_post = input.num_post_periods();
    compute_relative_magnitude_confidence_set_with_builder(
        input,
        post_weights,
        inference,
        config,
        original,
        identified,
        family.min_s(num_pre)?,
        move |s, max_positive| {
            family.build_constraint_matrix(num_pre, num_post, mbar, s, max_positive)
        },
        family.empty_message(),
    )
}

/// Compute the exact `\Delta^{RM}(\bar M)` identified set for a post-period
/// linear functional.
///
/// This matches the LP construction used by `HonestDiD` and aggregates across
/// the branch decomposition of the relative-magnitude restriction.
///
/// # Errors
/// Returns an error if dimensions are inconsistent or the LP solver fails on
/// every branch.
pub fn compute_relative_magnitude_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
) -> Result<HonestIdentifiedSet, String> {
    compute_relative_magnitude_family_identified_set(
        input,
        post_weights,
        mbar,
        RelativeMagnitudeFamily::Base,
    )
}

/// Compute a `\Delta^{RM}` conditional confidence set using the default
/// least-favorable hybrid path.
///
/// The current implementation follows the scalar conditional-CS path used for a
/// fixed post-period functional `post_weights' \tau_{post}`.
///
/// # Errors
/// Returns an error if `input` is inconsistent, `post_weights` is invalid, or the
/// underlying LP path fails.
pub fn compute_relative_magnitude_confidence_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_relative_magnitude_confidence_set_with_config(
        input,
        post_weights,
        mbar,
        inference,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
    )
}

/// Compute a `\Delta^{RM}` conditional confidence set with an explicit hybrid
/// configuration.
///
/// This mirrors the `HonestDiD::computeConditionalCS_DeltaRM` interface more
/// closely than [`compute_relative_magnitude_confidence_set`], which uses the
/// least-favorable default. In particular:
///
/// - [`RelativeMagnitudeHybrid::LeastFavorable`] corresponds to `hybrid_flag = "LF"`
/// - [`RelativeMagnitudeHybrid::ArpOnly`] corresponds to `hybrid_flag = "ARP"`
///
/// # Errors
/// Returns an error if `input` is inconsistent, `post_weights` is invalid, or the
/// underlying LP path fails.
pub fn compute_relative_magnitude_confidence_set_with_config(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_relative_magnitude_family_conditional_cs_with_config(
        input,
        post_weights,
        mbar,
        inference,
        config,
        RelativeMagnitudeFamily::Base,
    )
}

pub(in crate::inference::sensitivity) fn compute_relative_magnitude_confidence_set_with_precomputed_sets(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
    original: &HonestOriginalConfidenceSet,
    identified: &HonestIdentifiedSet,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_relative_magnitude_family_conditional_cs_with_precomputed_sets(
        input,
        post_weights,
        mbar,
        inference,
        config,
        original,
        identified,
        RelativeMagnitudeFamily::Base,
    )
}

/// Compute the exact identified set under a sign-restricted relative-magnitude
/// bias class.
///
/// # Errors
/// Returns an error if the event-study input, functional weights, or bound are
/// invalid, or if the branch LP setup fails.
pub fn compute_signed_relative_magnitude_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    bias_direction: HonestBiasDirection,
) -> Result<HonestIdentifiedSet, String> {
    compute_relative_magnitude_family_identified_set(
        input,
        post_weights,
        mbar,
        RelativeMagnitudeFamily::SignedBase(bias_direction),
    )
}

/// Compute the conditional confidence set under a sign-restricted
/// relative-magnitude bias class using default hybrid settings.
///
/// # Errors
/// Returns an error if validation fails or the conditional confidence-set
/// solver cannot produce an interval.
pub fn compute_signed_relative_magnitude_confidence_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    bias_direction: HonestBiasDirection,
    inference: InferenceConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_signed_relative_magnitude_confidence_set_with_config(
        input,
        post_weights,
        mbar,
        bias_direction,
        inference,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
    )
}

/// Compute the conditional confidence set under a sign-restricted
/// relative-magnitude bias class with explicit hybrid settings.
///
/// # Errors
/// Returns an error if validation fails or the configured solver cannot
/// produce an interval.
pub fn compute_signed_relative_magnitude_confidence_set_with_config(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    bias_direction: HonestBiasDirection,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_relative_magnitude_family_conditional_cs_with_config(
        input,
        post_weights,
        mbar,
        inference,
        config,
        RelativeMagnitudeFamily::SignedBase(bias_direction),
    )
}

/// Compute the exact identified set under a monotone relative-magnitude bias
/// class.
///
/// # Errors
/// Returns an error if the event-study input, functional weights, or bound are
/// invalid, or if the branch LP setup fails.
pub fn compute_monotone_relative_magnitude_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    monotonicity_direction: HonestMonotonicityDirection,
) -> Result<HonestIdentifiedSet, String> {
    compute_relative_magnitude_family_identified_set(
        input,
        post_weights,
        mbar,
        RelativeMagnitudeFamily::MonotoneBase(monotonicity_direction),
    )
}

/// Compute the conditional confidence set under a monotone
/// relative-magnitude bias class using default hybrid settings.
///
/// # Errors
/// Returns an error if validation fails or the conditional confidence-set
/// solver cannot produce an interval.
pub fn compute_monotone_relative_magnitude_confidence_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    monotonicity_direction: HonestMonotonicityDirection,
    inference: InferenceConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_monotone_relative_magnitude_confidence_set_with_config(
        input,
        post_weights,
        mbar,
        monotonicity_direction,
        inference,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
    )
}

/// Compute the conditional confidence set under a monotone
/// relative-magnitude bias class with explicit hybrid settings.
///
/// # Errors
/// Returns an error if validation fails or the configured solver cannot
/// produce an interval.
pub fn compute_monotone_relative_magnitude_confidence_set_with_config(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    monotonicity_direction: HonestMonotonicityDirection,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_relative_magnitude_family_conditional_cs_with_config(
        input,
        post_weights,
        mbar,
        inference,
        config,
        RelativeMagnitudeFamily::MonotoneBase(monotonicity_direction),
    )
}

/// Compute the exact identified set under the linear-trend-deviation relative
/// magnitude class.
///
/// # Errors
/// Returns an error if the event-study input, functional weights, or bound are
/// invalid, or if the branch LP setup fails.
pub fn compute_linear_trend_relative_magnitude_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
) -> Result<HonestIdentifiedSet, String> {
    compute_relative_magnitude_family_identified_set(
        input,
        post_weights,
        mbar,
        RelativeMagnitudeFamily::Linear,
    )
}

/// Compute the conditional confidence set under the linear-trend-deviation
/// relative-magnitude class using default hybrid settings.
///
/// # Errors
/// Returns an error if validation fails or the conditional confidence-set
/// solver cannot produce an interval.
pub fn compute_linear_trend_relative_magnitude_confidence_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_linear_trend_relative_magnitude_confidence_set_with_config(
        input,
        post_weights,
        mbar,
        inference,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
    )
}

/// Compute the conditional confidence set under the linear-trend-deviation
/// relative-magnitude class with explicit hybrid settings.
///
/// # Errors
/// Returns an error if validation fails or the configured solver cannot
/// produce an interval.
pub fn compute_linear_trend_relative_magnitude_confidence_set_with_config(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_relative_magnitude_family_conditional_cs_with_config(
        input,
        post_weights,
        mbar,
        inference,
        config,
        RelativeMagnitudeFamily::Linear,
    )
}

/// Compute the exact identified set under a sign-restricted linear-trend
/// relative-magnitude class.
///
/// # Errors
/// Returns an error if the event-study input, functional weights, or bound are
/// invalid, or if the branch LP setup fails.
pub fn compute_signed_linear_trend_relative_magnitude_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    bias_direction: HonestBiasDirection,
) -> Result<HonestIdentifiedSet, String> {
    compute_relative_magnitude_family_identified_set(
        input,
        post_weights,
        mbar,
        RelativeMagnitudeFamily::SignedLinear(bias_direction),
    )
}

/// Compute the conditional confidence set under a sign-restricted
/// linear-trend relative-magnitude class using default hybrid settings.
///
/// # Errors
/// Returns an error if validation fails or the conditional confidence-set
/// solver cannot produce an interval.
pub fn compute_signed_linear_trend_relative_magnitude_confidence_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    bias_direction: HonestBiasDirection,
    inference: InferenceConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_signed_linear_trend_relative_magnitude_confidence_set_with_config(
        input,
        post_weights,
        mbar,
        bias_direction,
        inference,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
    )
}

/// Compute the conditional confidence set under a sign-restricted
/// linear-trend relative-magnitude class with explicit hybrid settings.
///
/// # Errors
/// Returns an error if validation fails or the configured solver cannot
/// produce an interval.
pub fn compute_signed_linear_trend_relative_magnitude_confidence_set_with_config(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    bias_direction: HonestBiasDirection,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_relative_magnitude_family_conditional_cs_with_config(
        input,
        post_weights,
        mbar,
        inference,
        config,
        RelativeMagnitudeFamily::SignedLinear(bias_direction),
    )
}

/// Compute the exact identified set under a monotone linear-trend
/// relative-magnitude class.
///
/// # Errors
/// Returns an error if the event-study input, functional weights, or bound are
/// invalid, or if the branch LP setup fails.
pub fn compute_monotone_linear_trend_relative_magnitude_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    monotonicity_direction: HonestMonotonicityDirection,
) -> Result<HonestIdentifiedSet, String> {
    compute_relative_magnitude_family_identified_set(
        input,
        post_weights,
        mbar,
        RelativeMagnitudeFamily::MonotoneLinear(monotonicity_direction),
    )
}

/// Compute the conditional confidence set under a monotone linear-trend
/// relative-magnitude class using default hybrid settings.
///
/// # Errors
/// Returns an error if validation fails or the conditional confidence-set
/// solver cannot produce an interval.
pub fn compute_monotone_linear_trend_relative_magnitude_confidence_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    monotonicity_direction: HonestMonotonicityDirection,
    inference: InferenceConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_monotone_linear_trend_relative_magnitude_confidence_set_with_config(
        input,
        post_weights,
        mbar,
        monotonicity_direction,
        inference,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
    )
}

/// Compute the conditional confidence set under a monotone linear-trend
/// relative-magnitude class with explicit hybrid settings.
///
/// # Errors
/// Returns an error if validation fails or the configured solver cannot
/// produce an interval.
pub fn compute_monotone_linear_trend_relative_magnitude_confidence_set_with_config(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    monotonicity_direction: HonestMonotonicityDirection,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_relative_magnitude_family_conditional_cs_with_config(
        input,
        post_weights,
        mbar,
        inference,
        config,
        RelativeMagnitudeFamily::MonotoneLinear(monotonicity_direction),
    )
}
