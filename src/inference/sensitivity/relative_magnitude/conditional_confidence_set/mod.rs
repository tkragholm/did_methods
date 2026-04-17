//! Optimization-backed `DeltaRM` sensitivity calculations.
//!
//! This module implements the `HonestDiD` relative-magnitude (`DeltaRM`) path for
//! event-study linear functionals. The core objects are:
//!
//! - the identified set, obtained from branch-specific linear programs
//! - the conditional confidence set, obtained from ARP test inversion using the
//!   least-favorable critical value and dual geometry described by
//!   Rambachan and Roth (2023)
//!
//! References:
//! - Rambachan, A. and Roth, J. (2023), "A More Credible Approach to Parallel Trends"
//! - `HonestDiD` R package, `DeltaRM` path

mod identified_set_lp_workspace;

use clarabel::algebra::CscMatrix;
use once_map::OnceMap;
use rayon::prelude::*;
use std::time::Instant;
use tracing::{debug, warn};

#[cfg(test)]
use super::super::adaptive_grid::compute_accepted_grid_range_full_grid;
use super::super::adaptive_grid::{
    AcceptedGridRange, AdaptiveGridDiagnostics,
    compute_accepted_grid_range_adaptive_with_diagnostics, nearest_grid_index,
};
use super::super::conditional_confidence_sets::{
    ConditionalMomentLpWorkspace, DualMaxLpWorkspace, build_v_b_row_major_into,
    compute_least_favorable_cv, compute_least_favorable_cv_uncached, dual_conditional_test,
    recover_dual_vertex_from_binding, row_nonbinding_coeff_row_major_into,
};
use super::super::linear_algebra::{
    bilinear_form_into, build_clarabel_matrix, diag_sqrt, dot, linear_grid, mat_vec_mul_into,
    sandwich_covariance, try_invert_square_matrix_row_major_into,
};
use super::super::{
    HonestConditionalConfidenceSet, HonestEventStudyInput, HonestIdentifiedSet,
    HonestOriginalConfidenceSet, RelativeMagnitudeConfidenceSetConfig, RelativeMagnitudeHybrid,
};
use super::geometry::{
    RelativeMagnitudePreparedBranch, RelativeMagnitudePreparedFunctionalTransform,
    build_selected_target_and_design_from_transform, create_pre_period_equality_matrix,
    find_post_period_constraint_rows, prepare_relative_magnitude_functional_transform,
    relative_magnitude_objective,
};
use crate::types::InferenceConfig;
use clarabel::solver::SupportedConeT;
use identified_set_lp_workspace::RelativeMagnitudeIdentifiedSetWorkspace;

/// Branch-specific conditional-CS request for `\Delta^{RM}`.
///
/// This packages the scalar ARP inversion settings together with the branch
/// geometry so the solver entrypoint matches the underlying statistical object
/// rather than a long list of positional arguments.
pub(in crate::inference::sensitivity) struct RelativeMagnitudeConditionalBranch<'a> {
    pub(in crate::inference::sensitivity) input: &'a HonestEventStudyInput,
    pub(in crate::inference::sensitivity) num_pre: usize,
    pub(in crate::inference::sensitivity) prepared_transform:
        &'a RelativeMagnitudePreparedFunctionalTransform,
    pub(in crate::inference::sensitivity) alpha: f64,
    pub(in crate::inference::sensitivity) hybrid: RelativeMagnitudeHybrid,
    pub(in crate::inference::sensitivity) hybrid_kappa: f64,
    pub(in crate::inference::sensitivity) grid: &'a [f64],
    pub(in crate::inference::sensitivity) anchor_idx: usize,
}

pub(in crate::inference::sensitivity) struct RelativeMagnitudePreparedInputBranch {
    pub(in crate::inference::sensitivity) y_arp_base: Vec<f64>,
    pub(in crate::inference::sensitivity) sigma_arp: Vec<Vec<f64>>,
    pub(in crate::inference::sensitivity) sd_arp: Vec<f64>,
}

pub(in crate::inference::sensitivity) struct RelativeMagnitudePreparedFunctionalBranch {
    pub(in crate::inference::sensitivity) a_target_arp: Vec<f64>,
    pub(in crate::inference::sensitivity) x_arp: Vec<Vec<f64>>,
    pub(in crate::inference::sensitivity) w_t: Vec<Vec<f64>>,
    least_favorable_cv_cache: OnceMap<u64, Result<f64, String>>,
}

impl RelativeMagnitudePreparedFunctionalBranch {
    fn least_favorable_cv(&self, sigma_arp: &[Vec<f64>], hybrid_kappa: f64) -> Result<f64, String> {
        let cache_key = normalized_f64_bits(hybrid_kappa);
        self.least_favorable_cv_cache.insert_cloned(cache_key, |_| {
            compute_least_favorable_cv_uncached(&self.x_arp, sigma_arp, hybrid_kappa, 1_000, 0)
        })
    }
}

struct RelativeMagnitudeThetaEvaluator<'a> {
    y_arp_base: &'a [f64],
    a_target_arp: &'a [f64],
    x_arp: &'a [Vec<f64>],
    sigma_arp: &'a [Vec<f64>],
    sd_arp: &'a [f64],
    alpha: f64,
    hybrid_kappa: f64,
    lf_cv: f64,
    workspace: ConditionalMomentLpWorkspace,
    dual_workspace: DualMaxLpWorkspace,
    shifted_y_arp: Vec<f64>,
    binding: Vec<usize>,
    gamma_row: Vec<f64>,
    coeff: Vec<f64>,
    sigma_gamma_scratch: Vec<f64>,
    s_t_scratch: Vec<f64>,
    m_scratch: Vec<f64>,
    binding_mask_scratch: Vec<bool>,
    v_b_scratch: Vec<f64>,
    inv_m_scratch: Vec<f64>,
    rho_tmp_scratch: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ThetaSearchSummary {
    feasible_branches: usize,
    skipped_infeasible_branches: usize,
    branches_with_accepted_points: usize,
    full_grid_branches: usize,
    fallback_branches: usize,
    nearest_anchor_branches: usize,
    anchor_accepted_branches: usize,
    unique_evaluations_total: usize,
    cache_hits_total: usize,
    max_unique_evaluations: usize,
    early_full_grid_exit: bool,
}

impl ThetaSearchSummary {
    fn record(
        &mut self,
        range: Option<AcceptedGridRange>,
        diagnostics: AdaptiveGridDiagnostics,
        grid_len: usize,
    ) {
        self.feasible_branches = self.feasible_branches.saturating_add(1);
        if range.is_some() {
            self.branches_with_accepted_points =
                self.branches_with_accepted_points.saturating_add(1);
        }
        if diagnostics.anchor_accepted {
            self.anchor_accepted_branches = self.anchor_accepted_branches.saturating_add(1);
        }
        if diagnostics.used_nearest_accepted_anchor {
            self.nearest_anchor_branches = self.nearest_anchor_branches.saturating_add(1);
        }
        if diagnostics.used_full_grid_fallback {
            self.fallback_branches = self.fallback_branches.saturating_add(1);
        }
        self.unique_evaluations_total = self
            .unique_evaluations_total
            .saturating_add(diagnostics.unique_evaluations);
        self.cache_hits_total = self.cache_hits_total.saturating_add(diagnostics.cache_hits);
        self.max_unique_evaluations = self
            .max_unique_evaluations
            .max(diagnostics.unique_evaluations);
        if let Some(accepted) = range
            && accepted.lower_idx == 0
            && accepted.upper_idx + 1 == grid_len
        {
            self.full_grid_branches = self.full_grid_branches.saturating_add(1);
        }
    }
}

fn log_theta_search_summary(scope: &'static str, grid_len: usize, summary: ThetaSearchSummary) {
    tracing::trace!(
        target: "did_methods::theta_search",
        scope,
        grid_len,
        feasible_branches = summary.feasible_branches,
        skipped_infeasible_branches = summary.skipped_infeasible_branches,
        branches_with_accepted_points = summary.branches_with_accepted_points,
        full_grid_branches = summary.full_grid_branches,
        fallback_branches = summary.fallback_branches,
        nearest_anchor_branches = summary.nearest_anchor_branches,
        anchor_accepted_branches = summary.anchor_accepted_branches,
        unique_evaluations_total = summary.unique_evaluations_total,
        cache_hits_total = summary.cache_hits_total,
        max_unique_evaluations = summary.max_unique_evaluations,
        early_full_grid_exit = summary.early_full_grid_exit,
        "relative-magnitude theta search summary"
    );
}

fn merge_accepted_range(global_range: &mut Option<AcceptedGridRange>, range: AcceptedGridRange) {
    *global_range = Some(global_range.map_or(range, |existing| AcceptedGridRange {
        lower_idx: existing.lower_idx.min(range.lower_idx),
        upper_idx: existing.upper_idx.max(range.upper_idx),
    }));
}

const fn is_full_grid_range(range: AcceptedGridRange, grid_len: usize) -> bool {
    range.lower_idx == 0 && range.upper_idx + 1 == grid_len
}

impl<'a> RelativeMagnitudeThetaEvaluator<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        y_arp_base: &'a [f64],
        a_target_arp: &'a [f64],
        x_arp: &'a [Vec<f64>],
        sigma_arp: &'a [Vec<f64>],
        sd_arp: &'a [f64],
        w_t: &'a [Vec<f64>],
        alpha: f64,
        hybrid_kappa: f64,
        lf_cv: f64,
    ) -> Result<Self, String> {
        let k = x_arp.first().map_or(0, Vec::len);
        Ok(Self {
            y_arp_base,
            a_target_arp,
            x_arp,
            sigma_arp,
            sd_arp,
            alpha,
            hybrid_kappa,
            lf_cv,
            workspace: ConditionalMomentLpWorkspace::new(x_arp, sigma_arp)?,
            dual_workspace: DualMaxLpWorkspace::new(w_t)?,
            shifted_y_arp: vec![0.0; y_arp_base.len()],
            binding: Vec::with_capacity(y_arp_base.len()),
            gamma_row: vec![0.0; y_arp_base.len()],
            coeff: Vec::with_capacity(k + 1),
            sigma_gamma_scratch: Vec::with_capacity(y_arp_base.len()),
            s_t_scratch: Vec::with_capacity(y_arp_base.len()),
            m_scratch: Vec::with_capacity((k + 1).pow(2)),
            binding_mask_scratch: vec![false; y_arp_base.len()],
            v_b_scratch: vec![0.0; y_arp_base.len()],
            inv_m_scratch: Vec::with_capacity((k + 1).pow(2)),
            rho_tmp_scratch: Vec::with_capacity(y_arp_base.len()),
        })
    }

    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn accepts(&mut self, theta: f64) -> Result<bool, String> {
        for (out, (y, a)) in self
            .shifted_y_arp
            .iter_mut()
            .zip(self.y_arp_base.iter().zip(self.a_target_arp.iter()))
        {
            *out = y - a * theta;
        }
        relative_magnitude_lp_conditional_test_prepared(
            &self.shifted_y_arp,
            self.x_arp,
            self.sigma_arp,
            self.sd_arp,
            self.alpha,
            self.hybrid_kappa,
            self.lf_cv,
            &mut self.workspace,
            &mut self.dual_workspace,
            &mut self.binding,
            &mut self.gamma_row,
            &mut self.coeff,
            &mut self.sigma_gamma_scratch,
            &mut self.s_t_scratch,
            &mut self.m_scratch,
            &mut self.binding_mask_scratch,
            &mut self.v_b_scratch,
            &mut self.inv_m_scratch,
            &mut self.rho_tmp_scratch,
        )
        .map(|rejected| !rejected)
    }
}

pub(in crate::inference::sensitivity) fn prepare_relative_magnitude_input_branches(
    input: &HonestEventStudyInput,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
) -> Vec<RelativeMagnitudePreparedInputBranch> {
    prepared_branches
        .par_iter()
        .map(|branch| {
            let mut y_vec = Vec::with_capacity(branch.constraint_rows.len());
            mat_vec_mul_into(&branch.constraint_rows, &input.betahat, &mut y_vec);
            let sigma_y = sandwich_covariance(&branch.constraint_rows, &input.covariance);
            let y_arp_base = branch
                .rows_for_arp
                .iter()
                .map(|row_idx| y_vec[*row_idx])
                .collect();
            let sigma_arp =
                super::super::linear_algebra::subset_square_matrix(&sigma_y, &branch.rows_for_arp);
            let sd_arp = diag_sqrt(&sigma_arp);
            RelativeMagnitudePreparedInputBranch {
                y_arp_base,
                sigma_arp,
                sd_arp,
            }
        })
        .collect()
}

pub(in crate::inference::sensitivity) fn prepare_relative_magnitude_functional_branches(
    post_weights: &[f64],
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
) -> Result<Vec<RelativeMagnitudePreparedFunctionalBranch>, String> {
    if prepared_branches.len() != prepared_input_branches.len() {
        return Err(format!(
            "prepared relative-magnitude branch count {} does not match prepared input branch count {}",
            prepared_branches.len(),
            prepared_input_branches.len()
        ));
    }
    let prepared_transform = prepare_relative_magnitude_functional_transform(post_weights)?;
    prepared_branches
        .iter()
        .zip(prepared_input_branches.iter())
        .map(|(branch, prepared_input)| {
            let (a_target_arp, x_arp) = build_selected_target_and_design_from_transform(
                &branch.a_post,
                &branch.rows_for_arp,
                &prepared_transform,
            );
            let x_arp = super::super::linear_algebra::drop_zero_columns(&x_arp, 1e-12);
            let w_t = build_w_t(&x_arp, &prepared_input.sd_arp);
            Ok(RelativeMagnitudePreparedFunctionalBranch {
                a_target_arp,
                x_arp,
                w_t,
                least_favorable_cv_cache: OnceMap::new(),
            })
        })
        .collect()
}

pub(in crate::inference::sensitivity) fn compute_relative_magnitude_identified_set_with_prepared_branches(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    prepared_branches: &[RelativeMagnitudePreparedBranch],
) -> Result<HonestIdentifiedSet, String> {
    let branch_sets = compute_relative_magnitude_branch_identified_sets_with_prepared_branches(
        input,
        post_weights,
        prepared_branches,
    )?;
    global_identified_set_from_branch_identified_sets(input, post_weights, &branch_sets)
}

pub(in crate::inference::sensitivity) fn compute_relative_magnitude_branch_identified_sets_with_prepared_branches(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    prepared_branches: &[RelativeMagnitudePreparedBranch],
) -> Result<Vec<Option<HonestIdentifiedSet>>, String> {
    let num_pre = input.num_pre_periods();
    let num_post = input.num_post_periods();
    let target_estimate = post_weights
        .iter()
        .zip(&input.betahat[num_pre..])
        .map(|(weight, beta)| weight * beta)
        .sum::<f64>();
    let objective = relative_magnitude_objective(num_pre, num_post, post_weights);
    let max_q: Vec<f64> = objective.iter().map(|value| -*value).collect();
    let quadratic = CscMatrix::<f64>::zeros((num_pre + num_post, num_pre + num_post));
    prepared_branches
        .iter()
        .map(|branch| {
            let mut rhs = vec![0.0; branch.inequality_len];
            rhs.extend_from_slice(&input.betahat[..num_pre]);
            let mut workspace = RelativeMagnitudeIdentifiedSetWorkspace::new(
                &quadratic,
                &branch.solver_matrix,
                &rhs,
                &branch.cones,
                &objective,
            )?;
            let max_solution = workspace.solve_with_q(&max_q)?;
            let min_solution = workspace.solve_with_q(&objective)?;
            Ok(match (max_solution, min_solution) {
                (Some(maximum), Some(minimum)) => Some(HonestIdentifiedSet {
                    lb: target_estimate - maximum,
                    ub: target_estimate - minimum,
                }),
                _ => None,
            })
        })
        .collect()
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "keeps branch/global solver signatures aligned for the surrounding pipeline"
)]
pub(in crate::inference::sensitivity) fn global_identified_set_from_branch_identified_sets(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    branch_sets: &[Option<HonestIdentifiedSet>],
) -> Result<HonestIdentifiedSet, String> {
    let num_pre = input.num_pre_periods();
    let mut global_lower = f64::INFINITY;
    let mut global_upper = f64::NEG_INFINITY;
    let mut solved_any_branch = false;
    for branch in branch_sets.iter().flatten() {
        solved_any_branch = true;
        global_lower = global_lower.min(branch.lb);
        global_upper = global_upper.max(branch.ub);
    }
    let target_estimate = post_weights
        .iter()
        .zip(&input.betahat[num_pre..])
        .map(|(weight, beta)| weight * beta)
        .sum::<f64>();
    Ok(if solved_any_branch {
        HonestIdentifiedSet {
            lb: global_lower,
            ub: global_upper,
        }
    } else {
        HonestIdentifiedSet {
            lb: target_estimate,
            ub: target_estimate,
        }
    })
}

fn branch_anchor_idx(
    grid: &[f64],
    original_estimate: f64,
    identified: &HonestIdentifiedSet,
) -> usize {
    let anchor_value = if (identified.lb..=identified.ub).contains(&original_estimate) {
        original_estimate
    } else {
        0.5 * (identified.lb + identified.ub)
    };
    nearest_grid_index(grid, anchor_value)
}

pub(in crate::inference::sensitivity) fn compute_relative_magnitude_identified_set_for_basis_period(
    input: &HonestEventStudyInput,
    post_idx: usize,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
) -> Result<HonestIdentifiedSet, String> {
    let post_weights = super::geometry::basis_post_weights(input.num_post_periods(), post_idx);
    compute_relative_magnitude_identified_set_with_prepared_branches(
        input,
        &post_weights,
        prepared_branches,
    )
}

#[allow(clippy::too_many_lines)]
pub(in crate::inference::sensitivity) fn compute_relative_magnitude_confidence_set_for_basis_period_with_prepared_input(
    input: &HonestEventStudyInput,
    post_idx: usize,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
) -> Result<HonestConditionalConfidenceSet, String> {
    if prepared_branches.len() != prepared_input_branches.len() {
        return Err(format!(
            "prepared relative-magnitude branch count {} does not match prepared input branch count {}",
            prepared_branches.len(),
            prepared_input_branches.len()
        ));
    }
    let post_weights = super::geometry::basis_post_weights(input.num_post_periods(), post_idx);
    let original = super::compute_original_confidence_set(input, &post_weights, inference)?;
    let identified = compute_relative_magnitude_identified_set_for_basis_period(
        input,
        post_idx,
        prepared_branches,
    )?;
    let (grid_lower, grid_upper) = super::grid_bounds_around_identified_set(&original, &identified);
    let grid_points = 1_000usize;
    let alpha = 1.0 - inference.confidence_level;
    let grid = linear_grid(grid_lower, grid_upper, grid_points);
    let branch_identified_sets =
        compute_relative_magnitude_branch_identified_sets_with_prepared_branches(
            input,
            &post_weights,
            prepared_branches,
        )?;
    let mut global_range: Option<AcceptedGridRange> = None;
    let mut theta_summary = ThetaSearchSummary::default();
    let prepared_functional_branches = prepared_branches
        .iter()
        .zip(prepared_input_branches.iter())
        .map(|(branch, prepared_input)| {
            let (a_target_arp, x_arp) = build_selected_target_and_design_from_transform(
                &branch.a_post,
                &branch.rows_for_arp,
                &RelativeMagnitudePreparedFunctionalTransform::Basis {
                    target_idx: post_idx,
                },
            );
            let x_arp = super::super::linear_algebra::drop_zero_columns(&x_arp, 1e-12);
            let w_t = build_w_t(&x_arp, &prepared_input.sd_arp);
            let prepared_functional = RelativeMagnitudePreparedFunctionalBranch {
                a_target_arp,
                x_arp,
                w_t,
                least_favorable_cv_cache: OnceMap::new(),
            };
            let lf_cv = match config.hybrid {
                RelativeMagnitudeHybrid::LeastFavorable => prepared_functional
                    .least_favorable_cv(&prepared_input.sigma_arp, config.hybrid_kappa)?,
                RelativeMagnitudeHybrid::ArpOnly => f64::INFINITY,
            };
            Ok((prepared_functional, lf_cv))
        })
        .collect::<Result<Vec<_>, String>>()?;
    for ((prepared_input, prepared_functional), branch_identified) in prepared_input_branches
        .iter()
        .zip(prepared_functional_branches.iter())
        .zip(branch_identified_sets.iter())
    {
        let Some(branch_identified) = branch_identified else {
            theta_summary.skipped_infeasible_branches =
                theta_summary.skipped_infeasible_branches.saturating_add(1);
            continue;
        };
        let (prepared_functional, lf_cv) = prepared_functional;
        let (range, diagnostics) = compute_relative_magnitude_branch_accepted_range_for_branch(
            &grid,
            branch_anchor_idx(&grid, original.estimate, branch_identified),
            &prepared_input.y_arp_base,
            &prepared_functional.a_target_arp,
            &prepared_functional.x_arp,
            &prepared_input.sigma_arp,
            &prepared_input.sd_arp,
            &prepared_functional.w_t,
            alpha,
            config.hybrid_kappa,
            *lf_cv,
        )?;
        theta_summary.record(range, diagnostics, grid.len());
        if let Some(range) = range {
            if is_full_grid_range(range, grid.len()) {
                theta_summary.early_full_grid_exit = true;
                log_theta_search_summary("basis_period", grid.len(), theta_summary);
                return Ok(HonestConditionalConfidenceSet {
                    lb: grid[0],
                    ub: grid[grid.len() - 1],
                });
            }
            merge_accepted_range(&mut global_range, range);
        }
    }
    let Some(accepted_range) = global_range else {
        log_theta_search_summary("basis_period", grid.len(), theta_summary);
        warn!(
            grid_lower,
            grid_upper,
            grid_points,
            prepared_branch_count = prepared_branches.len(),
            branches_with_accepted_points = theta_summary.branches_with_accepted_points,
            "relative_magnitude_confidence_set_empty_grid"
        );
        return Err(
            "relative-magnitude conditional confidence set accepted no grid points".to_string(),
        );
    };
    log_theta_search_summary("basis_period", grid.len(), theta_summary);
    Ok(HonestConditionalConfidenceSet {
        lb: grid[accepted_range.lower_idx],
        ub: grid[accepted_range.upper_idx],
    })
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
#[cfg(test)]
pub(in crate::inference::sensitivity) fn compute_relative_magnitude_confidence_set_with_prepared_branches(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
    original: &HonestOriginalConfidenceSet,
    identified: &HonestIdentifiedSet,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
) -> Result<HonestConditionalConfidenceSet, String> {
    let prepared_functional_branches = prepare_relative_magnitude_functional_branches(
        post_weights,
        prepared_branches,
        prepared_input_branches,
    )?;
    compute_relative_magnitude_confidence_set_with_prepared_functional_branches(
        input,
        post_weights,
        inference,
        config,
        original,
        identified,
        prepared_branches,
        prepared_input_branches,
        &prepared_functional_branches,
    )
}

#[allow(clippy::too_many_lines)]
#[cfg_attr(feature = "hotpath", hotpath::measure)]
#[allow(
    clippy::too_many_arguments,
    reason = "prepared branch evaluation carries distinct calibrated inputs"
)]
pub(in crate::inference::sensitivity) fn compute_relative_magnitude_confidence_set_with_prepared_functional_branches(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
    original: &HonestOriginalConfidenceSet,
    identified: &HonestIdentifiedSet,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
    prepared_functional_branches: &[RelativeMagnitudePreparedFunctionalBranch],
) -> Result<HonestConditionalConfidenceSet, String> {
    let branch_identified_sets =
        compute_relative_magnitude_branch_identified_sets_with_prepared_branches(
            input,
            post_weights,
            prepared_branches,
        )?;
    compute_relative_magnitude_confidence_set_with_prepared_functional_branches_and_identified_sets(
        input,
        post_weights,
        inference,
        config,
        original,
        identified,
        prepared_branches,
        prepared_input_branches,
        prepared_functional_branches,
        &branch_identified_sets,
    )
}

#[allow(clippy::too_many_lines)]
#[allow(
    clippy::too_many_arguments,
    reason = "identified-set variant threads calibrated branch state explicitly"
)]
pub(in crate::inference::sensitivity) fn compute_relative_magnitude_confidence_set_with_prepared_functional_branches_and_identified_sets(
    _input: &HonestEventStudyInput,
    _post_weights: &[f64],
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
    original: &HonestOriginalConfidenceSet,
    identified: &HonestIdentifiedSet,
    _prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
    prepared_functional_branches: &[RelativeMagnitudePreparedFunctionalBranch],
    branch_identified_sets: &[Option<HonestIdentifiedSet>],
) -> Result<HonestConditionalConfidenceSet, String> {
    if prepared_functional_branches.len() != prepared_input_branches.len() {
        return Err(format!(
            "prepared functional branch count {} does not match prepared input branch count {}",
            prepared_functional_branches.len(),
            prepared_input_branches.len()
        ));
    }
    if branch_identified_sets.len() != prepared_input_branches.len() {
        return Err(format!(
            "branch identified set count {} does not match prepared input branch count {}",
            branch_identified_sets.len(),
            prepared_input_branches.len()
        ));
    }
    let (grid_lower, grid_upper) = super::grid_bounds_around_identified_set(original, identified);
    let grid_points = 1_000usize;
    let alpha = 1.0 - inference.confidence_level;
    let grid = linear_grid(grid_lower, grid_upper, grid_points);
    let mut global_range: Option<AcceptedGridRange> = None;
    let mut theta_summary = ThetaSearchSummary::default();
    for ((prepared_functional, prepared_input), branch_identified) in prepared_functional_branches
        .iter()
        .zip(prepared_input_branches.iter())
        .zip(branch_identified_sets.iter())
    {
        let Some(branch_identified) = branch_identified else {
            theta_summary.skipped_infeasible_branches =
                theta_summary.skipped_infeasible_branches.saturating_add(1);
            continue;
        };
        let lf_cv = match config.hybrid {
            RelativeMagnitudeHybrid::LeastFavorable => prepared_functional
                .least_favorable_cv(&prepared_input.sigma_arp, config.hybrid_kappa)?,
            RelativeMagnitudeHybrid::ArpOnly => f64::INFINITY,
        };
        let (range, diagnostics) = compute_relative_magnitude_branch_accepted_range_for_branch(
            &grid,
            branch_anchor_idx(&grid, original.estimate, branch_identified),
            &prepared_input.y_arp_base,
            &prepared_functional.a_target_arp,
            &prepared_functional.x_arp,
            &prepared_input.sigma_arp,
            &prepared_input.sd_arp,
            &prepared_functional.w_t,
            alpha,
            config.hybrid_kappa,
            lf_cv,
        )?;
        theta_summary.record(range, diagnostics, grid.len());
        if let Some(range) = range {
            if is_full_grid_range(range, grid.len()) {
                theta_summary.early_full_grid_exit = true;
                log_theta_search_summary("functional", grid.len(), theta_summary);
                return Ok(HonestConditionalConfidenceSet {
                    lb: grid[0],
                    ub: grid[grid.len() - 1],
                });
            }
            merge_accepted_range(&mut global_range, range);
        }
    }
    let Some(accepted_range) = global_range else {
        log_theta_search_summary("functional", grid.len(), theta_summary);
        return Err(
            "relative-magnitude conditional confidence set accepted no grid points".to_string(),
        );
    };
    log_theta_search_summary("functional", grid.len(), theta_summary);
    Ok(HonestConditionalConfidenceSet {
        lb: grid[accepted_range.lower_idx],
        ub: grid[accepted_range.upper_idx],
    })
}

#[cfg(test)]
#[allow(clippy::too_many_lines)]
pub(in crate::inference::sensitivity) fn compute_relative_magnitude_confidence_set_with_prepared_functional_branches_full_grid(
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
    original: &HonestOriginalConfidenceSet,
    identified: &HonestIdentifiedSet,
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
    prepared_functional_branches: &[RelativeMagnitudePreparedFunctionalBranch],
) -> Result<HonestConditionalConfidenceSet, String> {
    if prepared_functional_branches.len() != prepared_input_branches.len() {
        return Err(format!(
            "prepared functional branch count {} does not match prepared input branch count {}",
            prepared_functional_branches.len(),
            prepared_input_branches.len()
        ));
    }
    let (grid_lower, grid_upper) = super::grid_bounds_around_identified_set(original, identified);
    let grid_points = 1_000usize;
    let alpha = 1.0 - inference.confidence_level;
    let grid = linear_grid(grid_lower, grid_upper, grid_points);
    let mut global_range: Option<AcceptedGridRange> = None;
    for (prepared_functional, prepared_input) in prepared_functional_branches
        .iter()
        .zip(prepared_input_branches.iter())
    {
        let lf_cv = match config.hybrid {
            RelativeMagnitudeHybrid::LeastFavorable => prepared_functional
                .least_favorable_cv(&prepared_input.sigma_arp, config.hybrid_kappa)?,
            RelativeMagnitudeHybrid::ArpOnly => f64::INFINITY,
        };
        let mut evaluator = RelativeMagnitudeThetaEvaluator::new(
            &prepared_input.y_arp_base,
            &prepared_functional.a_target_arp,
            &prepared_functional.x_arp,
            &prepared_input.sigma_arp,
            &prepared_input.sd_arp,
            &prepared_functional.w_t,
            alpha,
            config.hybrid_kappa,
            lf_cv,
        )?;
        if let Some(range) =
            compute_relative_magnitude_branch_accepted_range_full_grid(&grid, &mut evaluator)?
        {
            global_range = Some(global_range.map_or(range, |existing| AcceptedGridRange {
                lower_idx: existing.lower_idx.min(range.lower_idx),
                upper_idx: existing.upper_idx.max(range.upper_idx),
            }));
        }
    }
    let Some(accepted_range) = global_range else {
        return Err(
            "relative-magnitude conditional confidence set accepted no grid points".to_string(),
        );
    };
    Ok(HonestConditionalConfidenceSet {
        lb: grid[accepted_range.lower_idx],
        ub: grid[accepted_range.upper_idx],
    })
}

#[allow(clippy::too_many_lines)]
pub(in crate::inference::sensitivity) fn compute_relative_magnitude_branch_accepted_range_for_matrix(
    branch_problem: &RelativeMagnitudeConditionalBranch<'_>,
    a_matrix: &[Vec<f64>],
) -> Result<Option<AcceptedGridRange>, String> {
    let rows_for_arp = find_post_period_constraint_rows(a_matrix, branch_problem.num_pre);
    let a_post: Vec<Vec<f64>> = a_matrix
        .iter()
        .map(|row| row[branch_problem.num_pre..].to_vec())
        .collect();
    let (a_target_arp, x_arp) = build_selected_target_and_design_from_transform(
        &a_post,
        &rows_for_arp,
        branch_problem.prepared_transform,
    );
    let x_arp = super::super::linear_algebra::drop_zero_columns(&x_arp, 1e-12);
    let mut y_vec = Vec::with_capacity(a_matrix.len());
    mat_vec_mul_into(a_matrix, &branch_problem.input.betahat, &mut y_vec);
    let sigma_y = super::super::linear_algebra::sandwich_covariance(
        a_matrix,
        &branch_problem.input.covariance,
    );
    let y_arp_base: Vec<f64> = rows_for_arp.iter().map(|row_idx| y_vec[*row_idx]).collect();
    let sigma_arp = super::super::linear_algebra::subset_square_matrix(&sigma_y, &rows_for_arp);
    let sd_arp = diag_sqrt(&sigma_arp);
    let w_t = build_w_t(&x_arp, &sd_arp);
    let lf_cv = match branch_problem.hybrid {
        RelativeMagnitudeHybrid::LeastFavorable => {
            compute_least_favorable_cv(&x_arp, &sigma_arp, branch_problem.hybrid_kappa, 1_000, 0)?
        }
        RelativeMagnitudeHybrid::ArpOnly => f64::INFINITY,
    };
    compute_relative_magnitude_branch_accepted_range_for_branch(
        branch_problem.grid,
        branch_problem.anchor_idx,
        &y_arp_base,
        &a_target_arp,
        &x_arp,
        &sigma_arp,
        &sd_arp,
        &w_t,
        branch_problem.alpha,
        branch_problem.hybrid_kappa,
        lf_cv,
    )
    .map(|(range, _)| range)
}

fn normalized_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_relative_magnitude_branch_accepted_range_for_branch(
    grid: &[f64],
    anchor_idx: usize,
    y_arp_base: &[f64],
    a_target_arp: &[f64],
    x_arp: &[Vec<f64>],
    sigma_arp: &[Vec<f64>],
    sd_arp: &[f64],
    w_t: &[Vec<f64>],
    alpha: f64,
    hybrid_kappa: f64,
    lf_cv: f64,
) -> Result<(Option<AcceptedGridRange>, AdaptiveGridDiagnostics), String> {
    let mut evaluator = RelativeMagnitudeThetaEvaluator::new(
        y_arp_base,
        a_target_arp,
        x_arp,
        sigma_arp,
        sd_arp,
        w_t,
        alpha,
        hybrid_kappa,
        lf_cv,
    )?;
    compute_relative_magnitude_branch_accepted_range_adaptive(grid, anchor_idx, &mut evaluator)
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn compute_relative_magnitude_branch_accepted_range_adaptive(
    grid: &[f64],
    anchor_idx: usize,
    evaluator: &mut RelativeMagnitudeThetaEvaluator<'_>,
) -> Result<(Option<AcceptedGridRange>, AdaptiveGridDiagnostics), String> {
    compute_accepted_grid_range_adaptive_with_diagnostics(grid, anchor_idx, |theta| {
        evaluator.accepts(theta)
    })
}

#[cfg(test)]
fn compute_relative_magnitude_branch_accepted_range_full_grid(
    grid: &[f64],
    evaluator: &mut RelativeMagnitudeThetaEvaluator<'_>,
) -> Result<Option<AcceptedGridRange>, String> {
    compute_accepted_grid_range_full_grid(grid, |theta| evaluator.accepts(theta))
}

pub(in crate::inference::sensitivity) fn solve_relative_magnitude_branch_with_matrix(
    num_pre: usize,
    num_post: usize,
    true_beta: &[f64],
    post_weights: &[f64],
    inequality_matrix: &[Vec<f64>],
) -> Result<Option<HonestIdentifiedSet>, String> {
    let target_estimate = post_weights
        .iter()
        .zip(&true_beta[num_pre..])
        .map(|(weight, beta)| weight * beta)
        .sum::<f64>();
    let objective = relative_magnitude_objective(num_pre, num_post, post_weights);
    let equality_matrix = create_pre_period_equality_matrix(num_pre, num_post);
    let mut rhs = vec![0.0; inequality_matrix.len()];
    rhs.extend_from_slice(&true_beta[..num_pre]);

    let solver_matrix = build_clarabel_matrix(inequality_matrix, &equality_matrix);
    let cones = vec![
        SupportedConeT::NonnegativeConeT(inequality_matrix.len()),
        SupportedConeT::ZeroConeT(num_pre),
    ];
    let quadratic = CscMatrix::<f64>::zeros((num_pre + num_post, num_pre + num_post));
    let mut workspace = RelativeMagnitudeIdentifiedSetWorkspace::new(
        &quadratic,
        &solver_matrix,
        &rhs,
        &cones,
        &objective,
    )?;
    let max_q: Vec<f64> = objective.iter().map(|value| -*value).collect();
    let max_solution = workspace.solve_with_q(&max_q)?;
    let min_solution = workspace.solve_with_q(&objective)?;

    match (max_solution, min_solution) {
        (Some(maximum), Some(minimum)) => Ok(Some(HonestIdentifiedSet {
            lb: target_estimate - maximum,
            ub: target_estimate - minimum,
        })),
        _ => Ok(None),
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn relative_magnitude_lp_conditional_test_prepared(
    y_arp: &[f64],
    x_arp: &[Vec<f64>],
    sigma_arp: &[Vec<f64>],
    sd_vec: &[f64],
    alpha: f64,
    hybrid_kappa: f64,
    lf_cv: f64,
    workspace: &mut ConditionalMomentLpWorkspace,
    dual_workspace: &mut DualMaxLpWorkspace,
    binding_scratch: &mut Vec<usize>,
    gamma_row_scratch: &mut Vec<f64>,
    coeff_scratch: &mut Vec<f64>,
    sigma_gamma_scratch: &mut Vec<f64>,
    s_t_scratch: &mut Vec<f64>,
    m_scratch: &mut Vec<f64>,
    binding_mask_scratch: &mut [bool],
    v_b_scratch: &mut Vec<f64>,
    inv_m_scratch: &mut Vec<f64>,
    rho_tmp_scratch: &mut Vec<f64>,
) -> Result<bool, String> {
    let started = Instant::now();
    let lp_started = Instant::now();
    if workspace.solve_in_place(y_arp).is_err() {
        // Clarabel returned NumericalError (typically a near-degenerate KKT factorisation
        // for high-variance outcomes such as income in large monetary units).  Including
        // this grid point in the confidence set is conservative: the resulting CI can only
        // be wider than the true conditional CI.
        return Ok(true);
    }
    let eta_delta_lp_ms = lp_started.elapsed().as_millis();
    let mod_size = (alpha - hybrid_kappa) / (1.0 - hybrid_kappa);
    if workspace.eta_star() > lf_cv {
        return Ok(true);
    }
    if workspace.lambda().len() == y_arp.len() {
        let dual_started = Instant::now();
        let accepted = dual_conditional_test(
            y_arp,
            sigma_arp,
            workspace.eta_star(),
            workspace.lambda(),
            mod_size,
            lf_cv,
            dual_workspace,
            sigma_gamma_scratch,
            s_t_scratch,
        )?;
        let dual_ms = dual_started.elapsed().as_millis();
        let total_ms = started.elapsed().as_millis();
        if total_ms >= 10 {
            debug!(
                target: "did_methods::theta_eval",
                mode = "explicit_lambda",
                eta_delta_lp_ms,
                dual_ms,
                total_ms,
                rows = y_arp.len(),
                cols = x_arp.first().map_or(0, Vec::len),
                "relative-magnitude theta evaluation"
            );
        }
        return Ok(accepted);
    }

    let k = x_arp.first().map_or(0, Vec::len);
    let dim = k + 1;
    binding_scratch.clear();
    for (idx, y_value) in y_arp.iter().enumerate() {
        let fitted = dot(&x_arp[idx], workspace.delta_star());
        let slack = workspace
            .eta_star()
            .mul_add(sd_vec[idx], -(y_value - fitted));
        if slack.abs() <= 1e-4 {
            binding_scratch.push(idx);
        }
    }
    let degenerate = binding_scratch.len() != dim;
    let full_rank = if degenerate {
        false
    } else {
        m_scratch.clear();
        m_scratch.resize(dim * dim, 0.0);
        for (binding_pos, &row_idx) in binding_scratch.iter().enumerate() {
            let start = binding_pos * dim;
            m_scratch[start] = sd_vec[row_idx];
            m_scratch[(start + 1)..=(start + k)].copy_from_slice(&x_arp[row_idx][..k]);
        }
        try_invert_square_matrix_row_major_into(m_scratch, dim, inv_m_scratch)
    };
    if degenerate || !full_rank {
        if binding_scratch.is_empty() {
            return Err(format!(
                "relative-magnitude primal binding set is empty (eta={}, rows={}, cols={})",
                workspace.eta_star(),
                y_arp.len(),
                k
            ));
        }
        let recovery_started = Instant::now();
        let gamma_tilde = recover_dual_vertex_from_binding(
            binding_scratch,
            sd_vec,
            x_arp,
            y_arp,
            workspace.eta_star(),
        )?;
        let recovery_ms = recovery_started.elapsed().as_millis();
        let dual_started = Instant::now();
        let accepted = dual_conditional_test(
            y_arp,
            sigma_arp,
            workspace.eta_star(),
            &gamma_tilde,
            mod_size,
            lf_cv,
            dual_workspace,
            sigma_gamma_scratch,
            s_t_scratch,
        )?;
        let dual_ms = dual_started.elapsed().as_millis();
        let total_ms = started.elapsed().as_millis();
        if total_ms >= 10 {
            debug!(
                target: "did_methods::theta_eval",
                mode = "recovered_dual",
                eta_delta_lp_ms,
                recovery_ms,
                dual_ms,
                total_ms,
                rows = y_arp.len(),
                cols = k,
                binding = binding_scratch.len(),
                "relative-magnitude theta evaluation"
            );
        }
        return Ok(accepted);
    }
    let inv_m_flat = inv_m_scratch.as_slice();
    binding_mask_scratch.fill(false);
    for &binding_idx in binding_scratch.iter() {
        binding_mask_scratch[binding_idx] = true;
    }
    v_b_scratch.resize(y_arp.len(), 0.0);
    build_v_b_row_major_into(binding_scratch, inv_m_flat, dim, v_b_scratch.as_mut_slice());
    let v_b = v_b_scratch.as_slice();
    let sigma_b2 = bilinear_form_into(v_b, sigma_arp, v_b, rho_tmp_scratch).max(0.0);
    if sigma_b2 <= f64::EPSILON {
        return Ok(workspace.eta_star() > 0.0);
    }
    let sigma_b = sigma_b2.sqrt();
    let mut vlo = f64::NEG_INFINITY;
    let mut vup = f64::INFINITY;
    let vb_y = dot(v_b, y_arp);
    gamma_row_scratch.resize(y_arp.len(), 0.0);
    for row_idx in 0..y_arp.len() {
        if binding_mask_scratch[row_idx] {
            continue;
        }
        row_nonbinding_coeff_row_major_into(
            sd_vec[row_idx],
            &x_arp[row_idx],
            inv_m_flat,
            dim,
            coeff_scratch,
        );
        gamma_row_scratch.fill(0.0);
        for (pos, &binding_idx) in binding_scratch.iter().enumerate() {
            gamma_row_scratch[binding_idx] = coeff_scratch[pos];
        }
        gamma_row_scratch[row_idx] -= 1.0;
        let rho = bilinear_form_into(gamma_row_scratch, sigma_arp, v_b, rho_tmp_scratch);
        if rho.abs() <= 1e-12 {
            continue;
        }
        let maximand = (-dot(gamma_row_scratch, y_arp) / rho) + vb_y;
        if rho > 0.0 {
            vlo = vlo.max(maximand);
        } else {
            vup = vup.min(maximand);
        }
    }
    let zlo = vlo / sigma_b;
    let zup = vup.min(lf_cv) / sigma_b;
    let maxstat = workspace.eta_star() / sigma_b;
    if !(zlo <= maxstat && maxstat <= zup) {
        return Ok(false);
    }
    let cval = super::super::linear_algebra::truncated_normal_quantile(1.0 - mod_size, zlo, zup)?;
    let accepted = maxstat > cval.max(0.0);
    let total_ms = started.elapsed().as_millis();
    if total_ms >= 10 {
        debug!(
                target: "did_methods::theta_eval",
                mode = "binding_closed_form",
            eta_delta_lp_ms,
            total_ms,
            rows = y_arp.len(),
            cols = k,
            binding = binding_scratch.len(),
            "relative-magnitude theta evaluation"
        );
    }
    Ok(accepted)
}

fn build_w_t(x_arp: &[Vec<f64>], sd_vec: &[f64]) -> Vec<Vec<f64>> {
    x_arp
        .iter()
        .enumerate()
        .map(|(idx, x_row)| {
            let mut row = Vec::with_capacity(x_row.len() + 1);
            row.push(sd_vec[idx]);
            row.extend_from_slice(x_row);
            row
        })
        .collect()
}
