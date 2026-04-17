//! Smoothness-based `HonestDiD` sensitivity analysis for event-study designs.
//!
//! The smoothness class `\Delta^{SD}(M)` bounds adjacent changes in slope of
//! the latent parallel-trends violation. Equivalently, if `\delta_t` denotes
//! the violation path over event time, then the class constrains discrete
//! second differences:
//!
//! ```text
//! |\delta_t - 2 \delta_{t-1} + \delta_{t-2}| \le M.
//! ```
//!
//! This module exposes both identified sets and conditional confidence sets
//! for linear functionals of post-treatment coefficients, together with the
//! auxiliary moment-system builders needed for ARP, least-favorable, and
//! conditional-FLCI hybrids.
//!
//! References:
//! - Rambachan, A. and Roth, J. (2023). "A More Credible Approach to Parallel
//!   Trends". *Review of Economic Studies* 90(5), 2555-2591.
//! - Andrews, I., Roth, J., and Pakes, A. (2022). "Inference for Linear
//!   Conditional Moment Inequalities". *Econometrica* 90(5), 2345-2377.

pub mod least_favorable_intervals;

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
};

use self::least_favorable_intervals::{
    SmoothnessFlciConfig, build_smoothness_flci_problem, compute_smoothness_flci_with_config,
};
#[cfg(test)]
use super::adaptive_grid::compute_accepted_grid_range_full_grid;
use super::adaptive_grid::{
    AcceptedGridRange, compute_accepted_grid_range_adaptive, nearest_grid_index,
};
use super::conditional_confidence_sets::{
    ConditionalMomentLpWorkspace, DualMaxLpWorkspace, compute_least_favorable_cv,
    dual_acceptance_region,
};
use super::linear_algebra::{
    build_clarabel_matrix, diag_sqrt, dot, linear_grid, mat_vec_mul_into, sandwich_covariance,
    solve_square_linear_system, truncated_normal_quantile,
};
use super::relative_magnitude::geometry::{build_target_and_design, prepare_arp_views};
use super::relative_magnitude::{
    compute_original_confidence_set, grid_bounds_around_identified_set,
};
use super::{
    HonestBiasDirection, HonestConditionalConfidenceSet, HonestEventStudyInput,
    HonestIdentifiedSet, HonestMonotonicityDirection, SmoothnessConfidenceSetConfig,
    SmoothnessHybrid,
};
use crate::types::InferenceConfig;

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothnessMomentSystem {
    pub(crate) constraint_matrix: Vec<Vec<f64>>,
    pub(crate) constraint_bounds: Vec<f64>,
    pub(crate) rows_for_arp: Vec<usize>,
}

#[derive(Debug, Clone)]
struct SmoothnessFlciHybridData {
    vbar: Vec<f64>,
    optimal_half_length: f64,
    vbar_dot_d: f64,
    vbar_dot_a_target: f64,
}

struct SmoothnessThetaEvaluator<'a> {
    y_arp_base: &'a [f64],
    y_full_base: &'a [f64],
    a_target_arp: &'a [f64],
    a_target_full: &'a [f64],
    sigma_arp: &'a [Vec<f64>],
    sigma_full: &'a [Vec<f64>],
    rows_for_arp: &'a [usize],
    alpha: f64,
    config: SmoothnessConfidenceSetConfig,
    lf_cv: f64,
    flci_hybrid: Option<&'a SmoothnessFlciHybridData>,
    workspace: ConditionalMomentLpWorkspace,
    dual_workspace: DualMaxLpWorkspace,
    shifted_y_arp: Vec<f64>,
    shifted_y_full: Vec<f64>,
    gamma_full_scratch: Vec<f64>,
    sigma_gamma_full_scratch: Vec<f64>,
    s_full_scratch: Vec<f64>,
    sigma_gamma_scratch: Vec<f64>,
    s_t_scratch: Vec<f64>,
}

impl<'a> SmoothnessThetaEvaluator<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        y_arp_base: &'a [f64],
        y_full_base: &'a [f64],
        a_target_arp: &'a [f64],
        a_target_full: &'a [f64],
        x_arp: &'a [Vec<f64>],
        sigma_arp: &'a [Vec<f64>],
        sigma_full: &'a [Vec<f64>],
        rows_for_arp: &'a [usize],
        w_t: &'a [Vec<f64>],
        alpha: f64,
        config: SmoothnessConfidenceSetConfig,
        lf_cv: f64,
        flci_hybrid: Option<&'a SmoothnessFlciHybridData>,
    ) -> Result<Self, String> {
        Ok(Self {
            y_arp_base,
            y_full_base,
            a_target_arp,
            a_target_full,
            sigma_arp,
            sigma_full,
            rows_for_arp,
            alpha,
            config,
            lf_cv,
            flci_hybrid,
            workspace: ConditionalMomentLpWorkspace::new(x_arp, sigma_arp)?,
            dual_workspace: DualMaxLpWorkspace::new(w_t)?,
            shifted_y_arp: vec![0.0; y_arp_base.len()],
            shifted_y_full: vec![0.0; y_full_base.len()],
            gamma_full_scratch: Vec::with_capacity(y_full_base.len()),
            sigma_gamma_full_scratch: Vec::with_capacity(y_full_base.len()),
            s_full_scratch: Vec::with_capacity(y_full_base.len()),
            sigma_gamma_scratch: Vec::with_capacity(y_arp_base.len()),
            s_t_scratch: Vec::with_capacity(y_arp_base.len()),
        })
    }

    fn accepts(&mut self, theta: f64) -> Result<bool, String> {
        for (out, (y, a)) in self
            .shifted_y_arp
            .iter_mut()
            .zip(self.y_arp_base.iter().zip(self.a_target_arp.iter()))
        {
            *out = y - a * theta;
        }
        for (out, (y, a)) in self
            .shifted_y_full
            .iter_mut()
            .zip(self.y_full_base.iter().zip(self.a_target_full.iter()))
        {
            *out = y - a * theta;
        }
        smoothness_dual_conditional_test(
            &self.shifted_y_arp,
            &self.shifted_y_full,
            self.sigma_arp,
            self.sigma_full,
            self.rows_for_arp,
            theta,
            self.alpha,
            self.config,
            self.lf_cv,
            self.flci_hybrid,
            &mut self.workspace,
            &mut self.dual_workspace,
            &mut self.gamma_full_scratch,
            &mut self.sigma_gamma_full_scratch,
            &mut self.s_full_scratch,
            &mut self.sigma_gamma_scratch,
            &mut self.s_t_scratch,
        )
        .map(|rejected| !rejected)
    }
}

#[must_use]
pub fn max_adjacent_pre_period_change(pre_period_estimates: &[f64]) -> f64 {
    pre_period_estimates
        .windows(2)
        .map(|window| (window[1] - window[0]).abs())
        .fold(0.0, f64::max)
}

#[must_use]
pub fn pre_periods_satisfy_smoothness_bound(pre_period_estimates: &[f64], m: f64) -> bool {
    max_adjacent_pre_period_change(pre_period_estimates) <= m + 1e-8
}

pub fn build_smoothness_constraint_matrix(
    num_pre_periods: usize,
    num_post_periods: usize,
    post_period_moments_only: bool,
) -> Vec<Vec<f64>> {
    let rows = num_pre_periods + num_post_periods - 1;
    let cols = num_pre_periods + num_post_periods + 1;
    let mut second_difference_rows = vec![vec![0.0; cols]; rows];
    for (row_idx, row) in second_difference_rows.iter_mut().enumerate() {
        row[row_idx] = 1.0;
        row[row_idx + 1] = -2.0;
        row[row_idx + 2] = 1.0;
    }
    for row in &mut second_difference_rows {
        row.remove(num_pre_periods);
    }
    if post_period_moments_only {
        let post_start = num_pre_periods;
        second_difference_rows.retain(|row| {
            row[post_start..]
                .iter()
                .any(|value| *value > 1e-12 || *value < -1e-12)
        });
    }
    let mut constraint_matrix = second_difference_rows.clone();
    constraint_matrix.extend(
        second_difference_rows
            .into_iter()
            .map(|row| row.into_iter().map(|value| -value).collect()),
    );
    constraint_matrix
}

pub fn build_smoothness_constraint_bounds(
    num_pre_periods: usize,
    num_post_periods: usize,
    m: f64,
    post_period_moments_only: bool,
) -> Vec<f64> {
    vec![
        m;
        build_smoothness_constraint_matrix(
            num_pre_periods,
            num_post_periods,
            post_period_moments_only
        )
        .len()
    ]
}

pub fn build_bias_sign_restriction_matrix(
    num_pre_periods: usize,
    num_post_periods: usize,
    bias_direction: HonestBiasDirection,
) -> Vec<Vec<f64>> {
    let total = num_pre_periods + num_post_periods;
    let mut rows = Vec::with_capacity(num_post_periods);
    for post_idx in 0..num_post_periods {
        let mut row = vec![0.0; total];
        row[num_pre_periods + post_idx] = match bias_direction {
            HonestBiasDirection::Positive => -1.0,
            HonestBiasDirection::Negative => 1.0,
        };
        rows.push(row);
    }
    rows
}

pub fn build_monotonicity_restriction_matrix(
    num_pre_periods: usize,
    num_post_periods: usize,
    monotonicity_direction: HonestMonotonicityDirection,
    post_period_moments_only: bool,
) -> Vec<Vec<f64>> {
    let total = num_pre_periods + num_post_periods;
    let mut monotonicity_rows = vec![vec![0.0; total]; total];
    for (row_idx, row) in monotonicity_rows
        .iter_mut()
        .enumerate()
        .take(num_pre_periods.saturating_sub(1))
    {
        row[row_idx] = 1.0;
        row[row_idx + 1] = -1.0;
    }
    if num_pre_periods > 0 {
        monotonicity_rows[num_pre_periods - 1][num_pre_periods - 1] = 1.0;
    }
    if num_post_periods > 0 {
        monotonicity_rows[num_pre_periods][num_pre_periods] = -1.0;
        for row_idx in (num_pre_periods + 1)..total {
            monotonicity_rows[row_idx][row_idx - 1] = 1.0;
            monotonicity_rows[row_idx][row_idx] = -1.0;
        }
    }
    if post_period_moments_only {
        monotonicity_rows.retain(|row| {
            row[num_pre_periods..]
                .iter()
                .any(|value| f64::abs(*value) > 1e-12)
        });
    }
    if matches!(
        monotonicity_direction,
        HonestMonotonicityDirection::Decreasing
    ) {
        for row in &mut monotonicity_rows {
            for value in row {
                *value = -*value;
            }
        }
    }
    monotonicity_rows
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn find_post_period_constraint_rows(
    a_matrix: &[Vec<f64>],
    num_pre_periods: usize,
) -> Vec<usize> {
    a_matrix
        .iter()
        .enumerate()
        .filter_map(|(row_idx, row)| {
            row[num_pre_periods..]
                .iter()
                .any(|value| value.abs() > 1e-12)
                .then_some(row_idx)
        })
        .collect()
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn build_smoothness_moment_system(
    num_pre_periods: usize,
    num_post_periods: usize,
    m: f64,
    post_period_moments_only: bool,
) -> SmoothnessMomentSystem {
    let constraint_matrix =
        build_smoothness_constraint_matrix(num_pre_periods, num_post_periods, false);
    let constraint_bounds =
        build_smoothness_constraint_bounds(num_pre_periods, num_post_periods, m, false);
    let rows_for_arp = if post_period_moments_only && num_post_periods > 1 {
        find_post_period_constraint_rows(&constraint_matrix, num_pre_periods)
    } else {
        (0..constraint_matrix.len()).collect()
    };
    SmoothnessMomentSystem {
        constraint_matrix,
        constraint_bounds,
        rows_for_arp,
    }
}

pub fn build_signed_smoothness_moment_system(
    num_pre_periods: usize,
    num_post_periods: usize,
    m: f64,
    bias_direction: HonestBiasDirection,
    post_period_moments_only: bool,
) -> SmoothnessMomentSystem {
    let constraint_matrix =
        build_smoothness_constraint_matrix(num_pre_periods, num_post_periods, false);
    let constraint_bounds =
        build_smoothness_constraint_bounds(num_pre_periods, num_post_periods, m, false);
    let mut full_constraint_matrix = constraint_matrix;
    full_constraint_matrix.extend(build_bias_sign_restriction_matrix(
        num_pre_periods,
        num_post_periods,
        bias_direction,
    ));
    let mut full_constraint_bounds = constraint_bounds;
    full_constraint_bounds.extend(std::iter::repeat_n(0.0, num_post_periods));
    let rows_for_arp = if post_period_moments_only && num_post_periods > 1 {
        find_post_period_constraint_rows(&full_constraint_matrix, num_pre_periods)
    } else {
        (0..full_constraint_matrix.len()).collect()
    };
    SmoothnessMomentSystem {
        constraint_matrix: full_constraint_matrix,
        constraint_bounds: full_constraint_bounds,
        rows_for_arp,
    }
}

pub fn build_monotone_smoothness_moment_system(
    num_pre_periods: usize,
    num_post_periods: usize,
    m: f64,
    monotonicity_direction: HonestMonotonicityDirection,
    post_period_moments_only: bool,
) -> SmoothnessMomentSystem {
    let constraint_matrix =
        build_smoothness_constraint_matrix(num_pre_periods, num_post_periods, false);
    let constraint_bounds =
        build_smoothness_constraint_bounds(num_pre_periods, num_post_periods, m, false);
    let monotonicity_rows = build_monotonicity_restriction_matrix(
        num_pre_periods,
        num_post_periods,
        monotonicity_direction,
        false,
    );
    let mut full_constraint_matrix = constraint_matrix;
    full_constraint_matrix.extend(monotonicity_rows);
    let mut full_constraint_bounds = constraint_bounds;
    full_constraint_bounds.extend(std::iter::repeat_n(0.0, num_pre_periods + num_post_periods));
    let rows_for_arp = if post_period_moments_only && num_post_periods > 1 {
        find_post_period_constraint_rows(&full_constraint_matrix, num_pre_periods)
    } else {
        (0..full_constraint_matrix.len()).collect()
    };
    SmoothnessMomentSystem {
        constraint_matrix: full_constraint_matrix,
        constraint_bounds: full_constraint_bounds,
        rows_for_arp,
    }
}

fn create_pre_period_equality_matrix(
    num_pre_periods: usize,
    num_post_periods: usize,
) -> Vec<Vec<f64>> {
    let cols = num_pre_periods + num_post_periods;
    (0..num_pre_periods)
        .map(|row_idx| {
            let mut row = vec![0.0; cols];
            row[row_idx] = 1.0;
            row
        })
        .collect()
}

struct SmoothnessLpWorkspace {
    solver: DefaultSolver<f64>,
    objective: Vec<f64>,
    current_q: Vec<f64>,
}

impl SmoothnessLpWorkspace {
    fn new(
        quadratic: &CscMatrix<f64>,
        inequalities: &[Vec<f64>],
        equalities: &[Vec<f64>],
        rhs: &[f64],
        objective: &[f64],
    ) -> Result<Self, String> {
        let constraint_matrix = build_clarabel_matrix(inequalities, equalities);
        let cones = vec![
            SupportedConeT::NonnegativeConeT(inequalities.len()),
            SupportedConeT::ZeroConeT(equalities.len()),
        ];
        let settings = DefaultSettingsBuilder::<f64>::default()
            .verbose(false)
            .presolve_enable(false)
            .input_sparse_dropzeros(false)
            .build()
            .map_err(|err| format!("failed to build Clarabel settings: {err}"))?;
        let solver = DefaultSolver::new(
            quadratic,
            objective,
            &constraint_matrix,
            rhs,
            &cones,
            settings,
        )
        .map_err(|err| format!("failed to initialize smoothness LP solver: {err}"))?;
        Ok(Self {
            solver,
            objective: objective.to_vec(),
            current_q: objective.to_vec(),
        })
    }

    fn solve_with_q(&mut self, q: &[f64]) -> Result<Option<f64>, String> {
        self.current_q.clear();
        self.current_q.extend_from_slice(q);
        self.solver
            .update_q(&self.current_q)
            .map_err(|err| format!("failed to update smoothness LP objective: {err}"))?;
        self.solver.solve();
        match self.solver.solution.status {
            SolverStatus::Solved | SolverStatus::AlmostSolved => Ok(Some(
                self.objective
                    .iter()
                    .zip(self.solver.solution.x.iter())
                    .map(|(left, right)| left * right)
                    .sum::<f64>(),
            )),
            SolverStatus::PrimalInfeasible
            | SolverStatus::DualInfeasible
            | SolverStatus::AlmostPrimalInfeasible
            | SolverStatus::AlmostDualInfeasible => Ok(None),
            status => Err(format!(
                "Clarabel failed to solve smoothness LP: {status:?}"
            )),
        }
    }
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

fn solve_vbar_projection(
    constraint_matrix: &[Vec<f64>],
    optimal_vec: &[f64],
) -> Result<Vec<f64>, String> {
    let rows = constraint_matrix.len();
    let mut normal = vec![vec![0.0; rows]; rows];
    let mut rhs = vec![0.0; rows];
    for left in 0..rows {
        rhs[left] = dot(&constraint_matrix[left], optimal_vec);
        for right in 0..rows {
            normal[left][right] = dot(&constraint_matrix[left], &constraint_matrix[right]);
        }
        normal[left][left] += 1e-8;
    }
    let out = solve_square_linear_system(&normal, &rhs)?;
    if out.iter().any(|value| !value.is_finite()) {
        return Err(
            "failed to solve smoothness fixed-length interval projection for vbar".to_string(),
        );
    }
    Ok(out)
}

fn flci_dbar(data: &SmoothnessFlciHybridData, theta: f64) -> (f64, f64) {
    let theta_scale = (1.0 - data.vbar_dot_a_target) * theta;
    (
        data.optimal_half_length - data.vbar_dot_d + theta_scale,
        data.optimal_half_length + data.vbar_dot_d - theta_scale,
    )
}

fn flci_hybrid_rejects(y_full: &[f64], data: &SmoothnessFlciHybridData, theta: f64) -> bool {
    let dbar = flci_dbar(data, theta);
    let vbar_y = dot(&data.vbar, y_full);
    vbar_y > dbar.0 || -vbar_y > dbar.1
}

fn flci_vlo_vup(
    data: &SmoothnessFlciHybridData,
    theta: f64,
    s_full: &[f64],
    sigma_gamma_full: &[f64],
) -> (f64, f64) {
    let dbar = flci_dbar(data, theta);
    let vbar_s = dot(&data.vbar, s_full);
    let vbar_c = dot(&data.vbar, sigma_gamma_full);
    let mut vlo = f64::NEG_INFINITY;
    let mut vup = f64::INFINITY;

    if vbar_c < -1e-12 {
        vlo = vlo.max((dbar.0 - vbar_s) / vbar_c);
    } else if vbar_c > 1e-12 {
        vup = vup.min((dbar.0 - vbar_s) / vbar_c);
    }

    let neg_vbar_c = -vbar_c;
    let neg_vbar_s = -vbar_s;
    if neg_vbar_c < -1e-12 {
        vlo = vlo.max((dbar.1 - neg_vbar_s) / neg_vbar_c);
    } else if neg_vbar_c > 1e-12 {
        vup = vup.min((dbar.1 - neg_vbar_s) / neg_vbar_c);
    }

    (vlo, vup)
}

fn fill_gamma_full(rows_for_arp: &[usize], gamma_arp: &[f64], full_len: usize, out: &mut Vec<f64>) {
    out.clear();
    out.resize(full_len, 0.0);
    for (&row_idx, &value) in rows_for_arp.iter().zip(gamma_arp.iter()) {
        out[row_idx] = value;
    }
}

fn fill_sigma_projection(
    sigma_full: &[Vec<f64>],
    gamma_full: &[f64],
    sigma_b2: f64,
    out: &mut Vec<f64>,
) {
    out.clear();
    out.resize(sigma_full.len(), 0.0);
    for (row_idx, sigma_row) in sigma_full.iter().enumerate() {
        out[row_idx] = dot(sigma_row, gamma_full) / sigma_b2;
    }
}

fn build_smoothness_flci_hybrid_data(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
    hybrid_kappa: f64,
    constraint_matrix: &[Vec<f64>],
    constraint_bounds: &[f64],
    a_target: &[f64],
) -> Result<(SmoothnessFlciHybridData, (f64, f64)), String> {
    let inference = InferenceConfig::new(1.0 - hybrid_kappa);
    let problem = build_smoothness_flci_problem(input, post_weights, m, inference)?;
    let flci = compute_smoothness_flci_with_config(
        &problem,
        SmoothnessFlciConfig::default_for_production(),
    )?;
    let vbar = solve_vbar_projection(constraint_matrix, &flci.optimal_vec)?;
    Ok((
        SmoothnessFlciHybridData {
            vbar_dot_d: dot(&vbar, constraint_bounds),
            vbar_dot_a_target: dot(&vbar, a_target),
            vbar,
            optimal_half_length: flci.optimal_half_length,
        },
        flci.flci,
    ))
}

#[allow(clippy::too_many_arguments)]
fn smoothness_dual_conditional_test(
    y_arp: &[f64],
    y_full: &[f64],
    sigma_arp: &[Vec<f64>],
    sigma_full: &[Vec<f64>],
    rows_for_arp: &[usize],
    theta: f64,
    alpha: f64,
    config: SmoothnessConfidenceSetConfig,
    lf_cv: f64,
    flci_hybrid: Option<&SmoothnessFlciHybridData>,
    workspace: &mut ConditionalMomentLpWorkspace,
    dual_workspace: &mut DualMaxLpWorkspace,
    gamma_full_scratch: &mut Vec<f64>,
    sigma_gamma_full_scratch: &mut Vec<f64>,
    s_full_scratch: &mut Vec<f64>,
    sigma_gamma_scratch: &mut Vec<f64>,
    s_t_scratch: &mut Vec<f64>,
) -> Result<bool, String> {
    if workspace.solve_in_place(y_arp).is_err() {
        return Ok(true);
    }
    let mod_size = match config.hybrid {
        SmoothnessHybrid::ArpOnly => alpha,
        SmoothnessHybrid::LeastFavorable | SmoothnessHybrid::Flci => {
            (alpha - config.hybrid_kappa) / (1.0 - config.hybrid_kappa)
        }
    };
    match config.hybrid {
        SmoothnessHybrid::LeastFavorable if workspace.eta_star() > lf_cv => return Ok(true),
        SmoothnessHybrid::Flci => {
            let flci_hybrid =
                flci_hybrid.ok_or("missing smoothness fixed-length interval hybrid data")?;
            if flci_hybrid_rejects(y_full, flci_hybrid, theta) {
                return Ok(true);
            }
        }
        SmoothnessHybrid::LeastFavorable | SmoothnessHybrid::ArpOnly => {}
    }

    let gamma_arp = workspace.lambda();
    let Some(region) = dual_acceptance_region(
        y_arp,
        sigma_arp,
        workspace.eta_star(),
        gamma_arp,
        dual_workspace,
        sigma_gamma_scratch,
        s_t_scratch,
    )?
    else {
        return Ok(workspace.eta_star() > 0.0);
    };

    let (zlo, zup) = match config.hybrid {
        SmoothnessHybrid::ArpOnly => (region.vlo / region.sigma_b, region.vup / region.sigma_b),
        SmoothnessHybrid::LeastFavorable => (
            region.vlo / region.sigma_b,
            region.vup.min(lf_cv) / region.sigma_b,
        ),
        SmoothnessHybrid::Flci => {
            let flci_hybrid =
                flci_hybrid.ok_or("missing smoothness fixed-length interval hybrid data")?;
            fill_gamma_full(rows_for_arp, gamma_arp, y_full.len(), gamma_full_scratch);
            let sigma_b2 = region.sigma_b * region.sigma_b;
            fill_sigma_projection(
                sigma_full,
                gamma_full_scratch,
                sigma_b2,
                sigma_gamma_full_scratch,
            );
            s_full_scratch.clear();
            s_full_scratch.resize(y_full.len(), 0.0);
            let gamma_y = dot(gamma_full_scratch, y_full);
            for (idx, out) in s_full_scratch.iter_mut().enumerate() {
                *out = sigma_gamma_full_scratch[idx].mul_add(-gamma_y, y_full[idx]);
            }
            let (flci_vlo, flci_vup) =
                flci_vlo_vup(flci_hybrid, theta, s_full_scratch, sigma_gamma_full_scratch);
            (
                region.vlo.max(flci_vlo) / region.sigma_b,
                region.vup.min(flci_vup) / region.sigma_b,
            )
        }
    };
    if !(zlo <= region.maxstat && region.maxstat <= zup) {
        return Ok(false);
    }
    let cval = truncated_normal_quantile(1.0 - mod_size, zlo, zup)?;
    Ok(region.maxstat > cval.max(0.0))
}

fn compute_honest_linear_sensitivity_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    inequalities: &[Vec<f64>],
    d_vec: &[f64],
) -> Result<HonestIdentifiedSet, String> {
    let num_pre = input.num_pre_periods();
    let num_post = input.num_post_periods();
    let objective = {
        let mut out = vec![0.0; num_pre + num_post];
        out[num_pre..].copy_from_slice(post_weights);
        out
    };
    let equalities = create_pre_period_equality_matrix(num_pre, num_post);
    let mut rhs = d_vec.to_vec();
    rhs.extend_from_slice(&input.betahat[..num_pre]);
    let quadratic = CscMatrix::<f64>::zeros((num_pre + num_post, num_pre + num_post));
    let mut workspace =
        SmoothnessLpWorkspace::new(&quadratic, inequalities, &equalities, &rhs, &objective)?;
    let max_q: Vec<f64> = objective.iter().map(|value| -*value).collect();
    let max_delta = workspace.solve_with_q(&max_q)?;
    let min_delta = workspace.solve_with_q(&objective)?;
    let estimate = dot(post_weights, &input.betahat[num_pre..]);
    match (max_delta, min_delta) {
        (Some(maximum), Some(minimum)) => Ok(HonestIdentifiedSet {
            lb: estimate - maximum,
            ub: estimate - minimum,
        }),
        _ => Ok(HonestIdentifiedSet {
            lb: f64::NEG_INFINITY,
            ub: f64::INFINITY,
        }),
    }
}

#[allow(clippy::too_many_lines)]
fn compute_honest_linear_sensitivity_conditional_cs(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
    inference: InferenceConfig,
    config: SmoothnessConfidenceSetConfig,
    moments: &SmoothnessMomentSystem,
    identified: &HonestIdentifiedSet,
) -> Result<HonestConditionalConfidenceSet, String> {
    let original = compute_original_confidence_set(input, post_weights, inference)?;
    let num_pre = input.num_pre_periods();
    let a_post: Vec<Vec<f64>> = moments
        .constraint_matrix
        .iter()
        .map(|row| row[num_pre..].to_vec())
        .collect();
    let (a_target_full, x_matrix) = build_target_and_design(post_weights, &a_post)?;
    let mut y_full_base = Vec::with_capacity(moments.constraint_matrix.len());
    mat_vec_mul_into(&moments.constraint_matrix, &input.betahat, &mut y_full_base);
    for (y_value, d_value) in y_full_base.iter_mut().zip(moments.constraint_bounds.iter()) {
        *y_value -= d_value;
    }
    let sigma_full = sandwich_covariance(&moments.constraint_matrix, &input.covariance);
    let (x_arp, y_arp_base, a_target_arp, sigma_arp) = prepare_arp_views(
        &x_matrix,
        &y_full_base,
        &a_target_full,
        &sigma_full,
        &moments.rows_for_arp,
    );
    let sd_arp = diag_sqrt(&sigma_arp);
    let w_t = build_w_t(&x_arp, &sd_arp);
    let lf_cv = match config.hybrid {
        SmoothnessHybrid::LeastFavorable => {
            compute_least_favorable_cv(&x_arp, &sigma_arp, config.hybrid_kappa, 1_000, 0)?
        }
        SmoothnessHybrid::Flci | SmoothnessHybrid::ArpOnly => f64::INFINITY,
    };
    let (flci_hybrid, grid_lower, grid_upper) = match config.hybrid {
        SmoothnessHybrid::Flci => {
            let (data, interval) = build_smoothness_flci_hybrid_data(
                input,
                post_weights,
                m,
                config.hybrid_kappa,
                &moments.constraint_matrix,
                &moments.constraint_bounds,
                &a_target_full,
            )?;
            (Some(data), interval.0, interval.1)
        }
        SmoothnessHybrid::LeastFavorable | SmoothnessHybrid::ArpOnly => {
            let (lower, upper) = grid_bounds_around_identified_set(&original, identified);
            (None, lower, upper)
        }
    };
    let grid = linear_grid(grid_lower, grid_upper, config.grid_points);
    let alpha = 1.0 - inference.confidence_level;
    let anchor_idx = nearest_grid_index(&grid, original.estimate);
    let mut evaluator = SmoothnessThetaEvaluator::new(
        &y_arp_base,
        &y_full_base,
        &a_target_arp,
        &a_target_full,
        &x_arp,
        &sigma_arp,
        &sigma_full,
        &moments.rows_for_arp,
        &w_t,
        alpha,
        config,
        lf_cv,
        flci_hybrid.as_ref(),
    )?;
    let Some(accepted_range) =
        compute_smoothness_accepted_range_adaptive(&grid, anchor_idx, &mut evaluator)?
    else {
        return Err("linear sensitivity conditional CS accepted no grid points".to_string());
    };
    Ok(HonestConditionalConfidenceSet {
        lb: grid[accepted_range.lower_idx],
        ub: grid[accepted_range.upper_idx],
    })
}

#[cfg(test)]
#[allow(clippy::too_many_lines)]
pub(in crate::inference::sensitivity) fn compute_honest_linear_sensitivity_conditional_cs_full_grid(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
    inference: InferenceConfig,
    config: SmoothnessConfidenceSetConfig,
    moments: &SmoothnessMomentSystem,
    identified: &HonestIdentifiedSet,
) -> Result<HonestConditionalConfidenceSet, String> {
    let original = compute_original_confidence_set(input, post_weights, inference)?;
    let num_pre = input.num_pre_periods();
    let a_post: Vec<Vec<f64>> = moments
        .constraint_matrix
        .iter()
        .map(|row| row[num_pre..].to_vec())
        .collect();
    let (a_target_full, x_matrix) = build_target_and_design(post_weights, &a_post)?;
    let mut y_full_base = Vec::with_capacity(moments.constraint_matrix.len());
    mat_vec_mul_into(&moments.constraint_matrix, &input.betahat, &mut y_full_base);
    for (y_value, d_value) in y_full_base.iter_mut().zip(moments.constraint_bounds.iter()) {
        *y_value -= d_value;
    }
    let sigma_full = sandwich_covariance(&moments.constraint_matrix, &input.covariance);
    let (x_arp, y_arp_base, a_target_arp, sigma_arp) = prepare_arp_views(
        &x_matrix,
        &y_full_base,
        &a_target_full,
        &sigma_full,
        &moments.rows_for_arp,
    );
    let sd_arp = diag_sqrt(&sigma_arp);
    let w_t = build_w_t(&x_arp, &sd_arp);
    let lf_cv = match config.hybrid {
        SmoothnessHybrid::LeastFavorable => {
            compute_least_favorable_cv(&x_arp, &sigma_arp, config.hybrid_kappa, 1_000, 0)?
        }
        SmoothnessHybrid::Flci | SmoothnessHybrid::ArpOnly => f64::INFINITY,
    };
    let (flci_hybrid, grid_lower, grid_upper) = match config.hybrid {
        SmoothnessHybrid::Flci => {
            let (data, interval) = build_smoothness_flci_hybrid_data(
                input,
                post_weights,
                m,
                config.hybrid_kappa,
                &moments.constraint_matrix,
                &moments.constraint_bounds,
                &a_target_full,
            )?;
            (Some(data), interval.0, interval.1)
        }
        SmoothnessHybrid::LeastFavorable | SmoothnessHybrid::ArpOnly => {
            let (lower, upper) = grid_bounds_around_identified_set(&original, identified);
            (None, lower, upper)
        }
    };
    let grid = linear_grid(grid_lower, grid_upper, config.grid_points);
    let alpha = 1.0 - inference.confidence_level;
    let mut evaluator = SmoothnessThetaEvaluator::new(
        &y_arp_base,
        &y_full_base,
        &a_target_arp,
        &a_target_full,
        &x_arp,
        &sigma_arp,
        &sigma_full,
        &moments.rows_for_arp,
        &w_t,
        alpha,
        config,
        lf_cv,
        flci_hybrid.as_ref(),
    )?;
    let Some(accepted_range) = compute_smoothness_accepted_range_full_grid(&grid, &mut evaluator)?
    else {
        return Err("linear sensitivity conditional CS accepted no grid points".to_string());
    };
    Ok(HonestConditionalConfidenceSet {
        lb: grid[accepted_range.lower_idx],
        ub: grid[accepted_range.upper_idx],
    })
}

fn compute_smoothness_accepted_range_adaptive(
    grid: &[f64],
    anchor_idx: usize,
    evaluator: &mut SmoothnessThetaEvaluator<'_>,
) -> Result<Option<AcceptedGridRange>, String> {
    compute_accepted_grid_range_adaptive(grid, anchor_idx, |theta| evaluator.accepts(theta))
}

#[cfg(test)]
fn compute_smoothness_accepted_range_full_grid(
    grid: &[f64],
    evaluator: &mut SmoothnessThetaEvaluator<'_>,
) -> Result<Option<AcceptedGridRange>, String> {
    compute_accepted_grid_range_full_grid(grid, |theta| evaluator.accepts(theta))
}

/// Compute the exact identified set under the smoothness restriction
/// `\Delta^{SD}(M)`.
///
/// # Errors
/// Returns an error if the event-study input or post weights are invalid, or
/// if the underlying linear identified-set solve fails.
pub fn compute_smoothness_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
) -> Result<HonestIdentifiedSet, String> {
    input.validate()?;
    if post_weights.len() != input.num_post_periods() {
        return Err(format!(
            "post_weights length {} does not match number of post periods {}",
            post_weights.len(),
            input.num_post_periods()
        ));
    }
    let num_pre = input.num_pre_periods();
    if !pre_periods_satisfy_smoothness_bound(&input.betahat[..num_pre], m) {
        return Ok(HonestIdentifiedSet {
            lb: f64::NEG_INFINITY,
            ub: f64::INFINITY,
        });
    }
    compute_honest_linear_sensitivity_identified_set(
        input,
        post_weights,
        &build_smoothness_constraint_matrix(num_pre, input.num_post_periods(), false),
        &build_smoothness_constraint_bounds(num_pre, input.num_post_periods(), m, false),
    )
}

/// Compute a `ΔSD(M)` conditional confidence set using the default
/// conditional-FLCI hybrid path.
///
/// # Errors
/// Returns an error if `input` is inconsistent, `post_weights` is invalid, or the
/// underlying conditional solve fails.
pub fn compute_smoothness_confidence_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
    inference: InferenceConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_smoothness_confidence_set_with_config(
        input,
        post_weights,
        m,
        inference,
        SmoothnessConfidenceSetConfig::from_inference(inference),
    )
}

/// Compute a `ΔSD(M)` conditional confidence set with an explicit hybrid
/// configuration.
///
/// # Errors
/// Returns an error if `input` is inconsistent, `post_weights` is invalid, or the
/// underlying conditional solve fails.
pub fn compute_smoothness_confidence_set_with_config(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
    inference: InferenceConfig,
    config: SmoothnessConfidenceSetConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    input.validate()?;
    if !m.is_finite() || m < 0.0 {
        return Err(format!(
            "smoothness sensitivity requires finite non-negative M, got {m}"
        ));
    }
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
    if post_weights.iter().all(|weight| weight.abs() <= 1e-12) {
        return Err("post_weights must contain at least one non-zero weight".to_string());
    }
    config.validate(inference)?;

    let identified = compute_smoothness_identified_set(input, post_weights, m)?;
    if !identified.lb.is_finite() || !identified.ub.is_finite() {
        return Ok(HonestConditionalConfidenceSet {
            lb: identified.lb,
            ub: identified.ub,
        });
    }

    let num_pre = input.num_pre_periods();
    let moments = build_smoothness_moment_system(
        num_pre,
        input.num_post_periods(),
        m,
        config.post_period_moments_only,
    );
    compute_honest_linear_sensitivity_conditional_cs(
        input,
        post_weights,
        m,
        inference,
        config,
        &moments,
        &identified,
    )
}

/// Compute the exact identified set under the sign-restricted smoothness class.
///
/// # Errors
/// Returns an error if the event-study input or post weights are invalid, or
/// if the underlying linear identified-set solve fails.
pub fn compute_signed_smoothness_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
    bias_direction: HonestBiasDirection,
) -> Result<HonestIdentifiedSet, String> {
    input.validate()?;
    if !m.is_finite() || m < 0.0 {
        return Err(format!(
            "sign-restricted smoothness sensitivity requires finite non-negative M, got {m}"
        ));
    }
    if post_weights.len() != input.num_post_periods() {
        return Err(format!(
            "post_weights length {} does not match number of post periods {}",
            post_weights.len(),
            input.num_post_periods()
        ));
    }
    let num_pre = input.num_pre_periods();
    if !pre_periods_satisfy_smoothness_bound(&input.betahat[..num_pre], m) {
        return Ok(HonestIdentifiedSet {
            lb: f64::NEG_INFINITY,
            ub: f64::INFINITY,
        });
    }
    let moments = build_signed_smoothness_moment_system(
        num_pre,
        input.num_post_periods(),
        m,
        bias_direction,
        false,
    );
    compute_honest_linear_sensitivity_identified_set(
        input,
        post_weights,
        &moments.constraint_matrix,
        &moments.constraint_bounds,
    )
}

/// Compute the sign-restricted smoothness confidence set using default hybrid
/// settings.
///
/// # Errors
/// Returns an error if validation fails or the configured conditional solver
/// cannot produce an interval.
pub fn compute_signed_smoothness_confidence_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
    bias_direction: HonestBiasDirection,
    inference: InferenceConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_signed_smoothness_confidence_set_with_config(
        input,
        post_weights,
        m,
        bias_direction,
        inference,
        SmoothnessConfidenceSetConfig::from_inference(inference),
    )
}

/// Compute the sign-restricted smoothness confidence set with explicit hybrid
/// settings.
///
/// # Errors
/// Returns an error if validation fails or the configured conditional solver
/// cannot produce an interval.
pub fn compute_signed_smoothness_confidence_set_with_config(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
    bias_direction: HonestBiasDirection,
    inference: InferenceConfig,
    config: SmoothnessConfidenceSetConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    input.validate()?;
    if !m.is_finite() || m < 0.0 {
        return Err(format!(
            "sign-restricted smoothness sensitivity requires finite non-negative M, got {m}"
        ));
    }
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
    if post_weights.iter().all(|weight| weight.abs() <= 1e-12) {
        return Err("post_weights must contain at least one non-zero weight".to_string());
    }
    config.validate(inference)?;
    let identified =
        compute_signed_smoothness_identified_set(input, post_weights, m, bias_direction)?;
    if !identified.lb.is_finite() || !identified.ub.is_finite() {
        return Ok(HonestConditionalConfidenceSet {
            lb: identified.lb,
            ub: identified.ub,
        });
    }
    let moments = build_signed_smoothness_moment_system(
        input.num_pre_periods(),
        input.num_post_periods(),
        m,
        bias_direction,
        config.post_period_moments_only,
    );
    compute_honest_linear_sensitivity_conditional_cs(
        input,
        post_weights,
        m,
        inference,
        config,
        &moments,
        &identified,
    )
}

/// Compute the exact identified set under the monotone smoothness class.
///
/// # Errors
/// Returns an error if the event-study input or post weights are invalid, or
/// if the underlying linear identified-set solve fails.
pub fn compute_monotone_smoothness_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
    monotonicity_direction: HonestMonotonicityDirection,
) -> Result<HonestIdentifiedSet, String> {
    input.validate()?;
    if !m.is_finite() || m < 0.0 {
        return Err(format!(
            "monotone smoothness sensitivity requires finite non-negative M, got {m}"
        ));
    }
    if post_weights.len() != input.num_post_periods() {
        return Err(format!(
            "post_weights length {} does not match number of post periods {}",
            post_weights.len(),
            input.num_post_periods()
        ));
    }
    let num_pre = input.num_pre_periods();
    if !pre_periods_satisfy_smoothness_bound(&input.betahat[..num_pre], m) {
        return Ok(HonestIdentifiedSet {
            lb: f64::NEG_INFINITY,
            ub: f64::INFINITY,
        });
    }
    let moments = build_monotone_smoothness_moment_system(
        num_pre,
        input.num_post_periods(),
        m,
        monotonicity_direction,
        false,
    );
    compute_honest_linear_sensitivity_identified_set(
        input,
        post_weights,
        &moments.constraint_matrix,
        &moments.constraint_bounds,
    )
}

/// Compute the monotone smoothness confidence set using default hybrid
/// settings.
///
/// # Errors
/// Returns an error if validation fails or the configured conditional solver
/// cannot produce an interval.
pub fn compute_monotone_smoothness_confidence_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
    monotonicity_direction: HonestMonotonicityDirection,
    inference: InferenceConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_monotone_smoothness_confidence_set_with_config(
        input,
        post_weights,
        m,
        monotonicity_direction,
        inference,
        SmoothnessConfidenceSetConfig::from_inference(inference),
    )
}

/// Compute the monotone smoothness confidence set with explicit hybrid
/// settings.
///
/// # Errors
/// Returns an error if validation fails or the configured conditional solver
/// cannot produce an interval.
pub fn compute_monotone_smoothness_confidence_set_with_config(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
    monotonicity_direction: HonestMonotonicityDirection,
    inference: InferenceConfig,
    config: SmoothnessConfidenceSetConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    input.validate()?;
    if !m.is_finite() || m < 0.0 {
        return Err(format!(
            "monotone smoothness sensitivity requires finite non-negative M, got {m}"
        ));
    }
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
    if post_weights.iter().all(|weight| weight.abs() <= 1e-12) {
        return Err("post_weights must contain at least one non-zero weight".to_string());
    }
    config.validate(inference)?;
    let identified =
        compute_monotone_smoothness_identified_set(input, post_weights, m, monotonicity_direction)?;
    if !identified.lb.is_finite() || !identified.ub.is_finite() {
        return Ok(HonestConditionalConfidenceSet {
            lb: identified.lb,
            ub: identified.ub,
        });
    }
    let moments = build_monotone_smoothness_moment_system(
        input.num_pre_periods(),
        input.num_post_periods(),
        m,
        monotonicity_direction,
        config.post_period_moments_only,
    );
    compute_honest_linear_sensitivity_conditional_cs(
        input,
        post_weights,
        m,
        inference,
        config,
        &moments,
        &identified,
    )
}
