//! Least-favorable confidence intervals for the smoothness restriction
//! `\Delta^{SD}(M)`.
//!
//! For a post-treatment linear functional `\theta = l' \tau_{post}`, this
//! module computes fixed-length confidence intervals by solving the smoothness
//! worst-case-bias problem over the convex class induced by bounded second
//! differences of the latent violation path. The implementation follows the
//! `HonestDiD` smoothness workflow and exposes both scalar and simultaneous
//! multi-functional results.
//!
//! References:
//! - Rambachan, A. and Roth, J. (2023). "A More Credible Approach to Parallel
//!   Trends". *Review of Economic Studies* 90(5), 2555-2591.
//! - `HonestDiD` package source files `flci.R` and `deltasd.R`, which this
//!   implementation mirrors for the smoothness class.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
};
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

use super::compute_smoothness_identified_set;
use crate::inference::validate_confidence_level;
use crate::types::InferenceConfig;

use super::super::linear_algebra::{
    cholesky_lower, critical_value_from_pointwise_confidence, dot,
    pointwise_confidence_level_from_critical, post_covariance_block,
    simulated_lower_cholesky_maxima_batched, simulation_rank,
};
use super::super::{
    HonestEventStudyInput, HonestIdentifiedSet, HonestJointPathConfig, HonestJointPathMethod,
    HonestOriginalConfidenceSet, SmoothnessMultiFlciResult,
};
use super::super::{SmoothnessMultiFlciPoint, compute_original_confidence_set};
use crate::util::usize_to_f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmoothnessFlciConfig {
    pub num_grid_points: usize,
}

impl SmoothnessFlciConfig {
    #[must_use]
    pub const fn default_for_production() -> Self {
        Self {
            num_grid_points: 48,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SmoothnessFlciProblem {
    pub input: HonestEventStudyInput,
    pub post_weights: Vec<f64>,
    pub m: f64,
    pub inference: InferenceConfig,
    pub original: HonestOriginalConfidenceSet,
    pub identified_set: HonestIdentifiedSet,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SmoothnessFlciResult {
    pub flci: (f64, f64),
    pub original_ci: (f64, f64),
    pub identified_set: (f64, f64),
    pub optimal_vec: Vec<f64>,
    pub optimal_half_length: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SmoothnessMultiFlciProblem {
    pub input: HonestEventStudyInput,
    pub post_weight_sets: Vec<Vec<f64>>,
    pub m: f64,
    pub inference: InferenceConfig,
    pub originals: Vec<HonestOriginalConfidenceSet>,
    pub identified_sets: Vec<HonestIdentifiedSet>,
}

#[derive(Debug, Clone)]
struct SmoothnessBiasAtH {
    value_at_m_one: f64,
    optimal_l_pre: Vec<f64>,
}

#[derive(Debug)]
struct SmoothnessSocpSolution {
    solution: Vec<f64>,
}

fn weighted_post_sum(post_weights: &[f64]) -> f64 {
    post_weights
        .iter()
        .enumerate()
        .map(|(idx, weight)| usize_to_f64(idx + 1) * weight)
        .sum()
}

fn w_to_l_pre(w: &[f64]) -> Vec<f64> {
    if w.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0; w.len()];
    out[0] = w[0];
    for idx in 1..w.len() {
        out[idx] = w[idx] - w[idx - 1];
    }
    out
}

fn full_estimator_vec(num_pre: usize, post_weights: &[f64], w: &[f64]) -> Vec<f64> {
    let mut out = w_to_l_pre(w);
    debug_assert_eq!(out.len(), num_pre);
    out.extend_from_slice(post_weights);
    out
}

fn bias_objective_constant(post_weights: &[f64]) -> f64 {
    let weighted_sum = weighted_post_sum(post_weights);
    let mut tail_sum = 0.0;
    let sum_abs_tail = post_weights
        .iter()
        .rev()
        .map(|weight| {
            tail_sum += *weight;
            tail_sum.abs()
        })
        .sum::<f64>();
    sum_abs_tail - weighted_sum
}

fn lower_triangular_ones(num_pre: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; num_pre]; num_pre];
    for (row_idx, row) in out.iter_mut().enumerate() {
        for value in &mut row[..=row_idx] {
            *value = 1.0;
        }
    }
    out
}

fn build_variance_soc_terms(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
    let num_pre = input.num_pre_periods();
    let total_periods = input.betahat.len();
    let lower = cholesky_lower(&input.covariance)?;
    let mut d = vec![0.0; total_periods];
    for row in 0..total_periods {
        let mut acc = 0.0;
        for (post_idx, post_weight) in post_weights.iter().copied().enumerate() {
            let idx = num_pre + post_idx;
            acc = lower[(idx, row)].mul_add(post_weight, acc);
        }
        d[row] = acc;
    }

    let mut transform = vec![vec![0.0; num_pre]; total_periods];
    for row in 0..total_periods {
        for col in 0..num_pre {
            let mut acc = 0.0;
            acc += lower[(col, row)];
            if col + 1 < num_pre {
                acc -= lower[(col + 1, row)];
            }
            transform[row][col] = acc;
        }
    }
    Ok((transform, d))
}

/// Split the variance rows into the ones `w` can move and the norm it cannot.
///
/// `transform` has one row per period but only `num_pre` columns, and its
/// entries come from a lower-triangular Cholesky factor, so every row at or
/// after the first post period is **identically zero**: that part of the
/// estimator's variance belongs to the post-period coefficients and no choice of
/// pre-period weights touches it. Its `d` entries are still real numbers, so in
/// the cone they appear as all-zero rows with non-zero right-hand sides.
///
/// Those rows are what made Clarabel stall with `InsufficientProgress`. They fix
/// a floor under the norm -- on the case that failed, 0.2828 of an available
/// 0.2898 -- so the cone is a sliver 0.06 wide whose analytic centre sits almost
/// on the boundary, and an interior-point method has nowhere to move. The
/// problem is feasible and well-posed; it is the FORM that is degenerate.
///
/// Returned separately, the caller can put the constant where it belongs: as one
/// scalar rather than as a block of structurally rank-deficient rows.
fn split_constant_variance(
    transform: Vec<Vec<f64>>,
    d: Vec<f64>,
) -> (Vec<Vec<f64>>, Vec<f64>, f64) {
    let mut active_rows = Vec::with_capacity(transform.len());
    let mut active_d = Vec::with_capacity(d.len());
    let mut constant_sq = 0.0;
    for (row, value) in transform.into_iter().zip(d) {
        if row.iter().any(|coefficient| *coefficient != 0.0) {
            active_rows.push(row);
            active_d.push(value);
        } else {
            constant_sq += value * value;
        }
    }
    (active_rows, active_d, constant_sq.max(0.0).sqrt())
}

fn variance_for_w(input: &HonestEventStudyInput, post_weights: &[f64], w: &[f64]) -> f64 {
    let estimator = full_estimator_vec(input.num_pre_periods(), post_weights, w);
    estimator
        .iter()
        .enumerate()
        .map(|(i, left)| {
            estimator
                .iter()
                .enumerate()
                .map(|(j, right)| left * right * input.covariance[i][j])
                .sum::<f64>()
        })
        .sum::<f64>()
        .max(0.0)
}

fn solve_socp(
    objective: &[f64],
    inequalities: &[Vec<f64>],
    equalities: &[Vec<f64>],
    soc_rows: &[Vec<f64>],
    rhs: &[f64],
) -> Result<Option<SmoothnessSocpSolution>, String> {
    let mut equality_and_soc = Vec::with_capacity(equalities.len() + soc_rows.len());
    equality_and_soc.extend_from_slice(equalities);
    equality_and_soc.extend_from_slice(soc_rows);
    let constraint_matrix =
        super::super::linear_algebra::build_clarabel_matrix(inequalities, &equality_and_soc);
    let cones = vec![
        SupportedConeT::NonnegativeConeT(inequalities.len()),
        SupportedConeT::ZeroConeT(equalities.len()),
        SupportedConeT::SecondOrderConeT(soc_rows.len()),
    ];
    let quadratic = CscMatrix::<f64>::zeros((objective.len(), objective.len()));
    let settings = DefaultSettingsBuilder::<f64>::default()
        .verbose(false)
        .presolve_enable(false)
        .input_sparse_dropzeros(false)
        .build()
        .map_err(|err| format!("failed to build Clarabel settings: {err}"))?;
    let mut solver = DefaultSolver::new(
        &quadratic,
        objective,
        &constraint_matrix,
        rhs,
        &cones,
        settings,
    )
    .map_err(|err| format!("failed to initialize smoothness SOCP solver: {err}"))?;
    solver.solve();
    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => Ok(Some(SmoothnessSocpSolution {
            solution: solver.solution.x,
        })),
        SolverStatus::PrimalInfeasible
        | SolverStatus::DualInfeasible
        | SolverStatus::AlmostPrimalInfeasible
        | SolverStatus::AlmostDualInfeasible => Ok(None),
        status => Err(format!(
            "Clarabel failed to solve smoothness SOCP: {status:?}"
        )),
    }
}

fn find_lowest_h(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
) -> Result<Option<(f64, Vec<f64>)>, String> {
    let num_pre = input.num_pre_periods();
    let weighted_sum = weighted_post_sum(post_weights);
    let (transform, d) = build_variance_soc_terms(input, post_weights)?;
    // `h` is a variable here, so the immovable part cannot be folded into the
    // bound the way it is in `find_worst_case_bias_given_h`. It can still be
    // carried as ONE row holding its norm instead of a block of zero rows
    // holding its components: same cone, same optimum, one rank-deficient row
    // rather than several.
    let (transform, d, constant) = split_constant_variance(transform, d);
    let num_vars = num_pre + 1;
    let objective = {
        let mut out = vec![0.0; num_vars];
        out[num_pre] = 1.0;
        out
    };
    let equalities = vec![{
        let mut row = vec![0.0; num_vars];
        for value in &mut row[..num_pre] {
            *value = 1.0;
        }
        row
    }];
    let soc_rows = {
        let mut rows = Vec::with_capacity(d.len() + 2);
        let mut first = vec![0.0; num_vars];
        first[num_pre] = -1.0;
        rows.push(first);
        rows.extend(transform.into_iter().map(|coeffs| {
            let mut row = vec![0.0; num_vars];
            row[..num_pre].copy_from_slice(&coeffs);
            row
        }));
        rows.push(vec![0.0; num_vars]);
        rows
    };
    let mut rhs = vec![weighted_sum];
    rhs.push(0.0);
    rhs.extend(d);
    rhs.push(constant);
    let Some(solution) = solve_socp(&objective, &[], &equalities, &soc_rows, &rhs)? else {
        return Ok(None);
    };
    let w = solution.solution[..num_pre].to_vec();
    Ok(Some((solution.solution[num_pre], w)))
}

fn find_h_for_minimum_bias(input: &HonestEventStudyInput, post_weights: &[f64]) -> (f64, Vec<f64>) {
    let num_pre = input.num_pre_periods();
    let mut w = vec![0.0; num_pre];
    if let Some(last) = w.last_mut() {
        *last = weighted_post_sum(post_weights);
    }
    (variance_for_w(input, post_weights, &w).sqrt(), w)
}

fn find_worst_case_bias_given_h(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    h: f64,
) -> Result<Option<SmoothnessBiasAtH>, String> {
    let num_pre = input.num_pre_periods();
    let weighted_sum = weighted_post_sum(post_weights);
    let constant = bias_objective_constant(post_weights);
    let lower_tri = lower_triangular_ones(num_pre);
    let (transform, d) = build_variance_soc_terms(input, post_weights)?;
    // `h` is FIXED here, so the immovable variance folds into the bound:
    // ||d - Tw|| <= h with a constant block c is exactly
    // ||d_active - T_active w|| <= sqrt(h^2 - c^2), and if h < c there is no w
    // at all. Solving it in that form is what stops Clarabel stalling: the cone
    // that failed was 0.2898 wide with 0.2828 of it already spent on rows no
    // variable appears in, and the same problem restated is 0.0637 wide with
    // nothing fixed inside it.
    // NOT `constant`: that name is already taken in this function by
    // `bias_objective_constant`, and shadowing it here would have silently
    // changed the returned bias rather than the cone.
    let (transform, d, immovable) = split_constant_variance(transform, d);
    let radius_sq = h.mul_add(h, -(immovable * immovable));
    if radius_sq < 0.0 {
        if std::env::var("SOCP_DEBUG").is_ok() {
            eprintln!("INFEASIBLE h={h} immovable={immovable}");
        }
        // Infeasible rather than unsolvable: no weighting achieves a standard
        // deviation below the part of it that does not depend on the weights.
        // The caller's `None` arm already means exactly this.
        return Ok(None);
    }
    let radius = radius_sq.sqrt();
    let num_vars = num_pre * 2;
    let objective = {
        let mut out = vec![0.0; num_vars];
        for value in &mut out[..num_pre] {
            *value = 1.0;
        }
        out
    };
    let mut inequalities = Vec::with_capacity(num_pre * 2);
    for row_idx in 0..num_pre {
        let mut row = vec![0.0; num_vars];
        row[row_idx] = -1.0;
        row[num_pre..].copy_from_slice(&lower_tri[row_idx]);
        inequalities.push(row);
    }
    for row_idx in 0..num_pre {
        let mut row = vec![0.0; num_vars];
        row[row_idx] = -1.0;
        for (dst, src) in row[num_pre..].iter_mut().zip(lower_tri[row_idx].iter()) {
            *dst = -*src;
        }
        inequalities.push(row);
    }
    let equalities = vec![{
        let mut row = vec![0.0; num_vars];
        for value in &mut row[num_pre..] {
            *value = 1.0;
        }
        row
    }];
    let soc_rows = {
        let mut rows = Vec::with_capacity(d.len() + 1);
        rows.push(vec![0.0; num_vars]);
        rows.extend(transform.into_iter().map(|coeffs| {
            let mut row = vec![0.0; num_vars];
            row[num_pre..].copy_from_slice(&coeffs);
            row
        }));
        rows
    };
    let mut rhs = vec![0.0; inequalities.len()];
    rhs.push(weighted_sum);
    rhs.push(radius);
    rhs.extend(d);
    let Some(solution) = solve_socp(&objective, &inequalities, &equalities, &soc_rows, &rhs)?
    else {
        return Ok(None);
    };
    let w = solution.solution[num_pre..].to_vec();
    Ok(Some(SmoothnessBiasAtH {
        value_at_m_one: constant + solution.solution[..num_pre].iter().sum::<f64>(),
        optimal_l_pre: w_to_l_pre(&w),
    }))
}

fn folded_normal_quantile(p: f64, mu: f64) -> Result<f64, String> {
    if !(0.0..1.0).contains(&p) {
        return Err(format!(
            "folded-normal quantile probability must lie in (0,1), got {p}"
        ));
    }
    let normal = Normal::new(0.0, 1.0)
        .map_err(|err| format!("failed to create normal distribution: {err}"))?;
    let mu_abs = mu.abs();
    let cdf = |q: f64| normal.cdf(q - mu_abs) + normal.cdf(q + mu_abs) - 1.0;
    let mut lower = 0.0;
    let mut upper = mu_abs + 8.0;
    loop {
        if cdf(upper) >= p {
            break;
        }
        upper *= 2.0;
        if upper > 1e6 {
            break;
        }
    }
    for _ in 0..80 {
        let mid = 0.5 * (lower + upper);
        if cdf(mid) >= p {
            upper = mid;
        } else {
            lower = mid;
        }
    }
    Ok(upper)
}

fn half_length_for_bias(alpha: f64, max_bias: f64, h: f64) -> Result<f64, String> {
    if h <= 1e-12 {
        return Ok(max_bias.abs());
    }
    Ok(folded_normal_quantile(1.0 - alpha, max_bias / h)? * h)
}

fn find_optimal_flci(
    problem: &SmoothnessFlciProblem,
    config: SmoothnessFlciConfig,
) -> Result<Option<(Vec<f64>, f64)>, String> {
    let Some((h_min, _)) = find_lowest_h(&problem.input, &problem.post_weights)? else {
        return Ok(None);
    };
    let (h_zero_bias, fallback_w) = find_h_for_minimum_bias(&problem.input, &problem.post_weights);
    let alpha = 1.0 - problem.inference.confidence_level;
    let lower = h_min.max(0.0);
    let upper = h_zero_bias.max(lower);
    let grid_points = config.num_grid_points.max(2);
    let step = if grid_points == 1 {
        0.0
    } else {
        (upper - lower) / usize_to_f64(grid_points - 1)
    };
    let mut best: Option<(Vec<f64>, f64, f64)> = None;
    for idx in 0..grid_points {
        let h = if idx + 1 == grid_points {
            upper
        } else {
            step.mul_add(usize_to_f64(idx), lower)
        };
        let Some(bias) = find_worst_case_bias_given_h(&problem.input, &problem.post_weights, h)?
        else {
            continue;
        };
        let max_bias = problem.m * bias.value_at_m_one;
        let half_length = half_length_for_bias(alpha, max_bias, h)?;
        let candidate = (
            {
                let mut vec = bias.optimal_l_pre.clone();
                vec.extend_from_slice(&problem.post_weights);
                vec
            },
            half_length,
            h,
        );
        match &best {
            Some((_, best_half_length, _)) if *best_half_length <= half_length => {}
            _ => best = Some(candidate),
        }
    }
    if let Some((optimal_vec, half_length, _)) = best {
        return Ok(Some((optimal_vec, half_length)));
    }

    let estimator = full_estimator_vec(
        problem.input.num_pre_periods(),
        &problem.post_weights,
        &fallback_w,
    );
    let half_length = half_length_for_bias(alpha, 0.0, h_zero_bias)?;
    Ok(Some((estimator, half_length)))
}

/// Build the typed scalar smoothness-FLCI problem for one post-treatment
/// functional.
///
/// # Errors
/// Returns an error if the input, post weights, or smoothness bound are
/// invalid, or if the original / identified-set builders fail.
pub fn build_smoothness_flci_problem(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
    inference: InferenceConfig,
) -> Result<SmoothnessFlciProblem, String> {
    let original = compute_original_confidence_set(input, post_weights, inference)?;
    let identified_set = compute_smoothness_identified_set(input, post_weights, m)?;
    Ok(SmoothnessFlciProblem {
        input: input.clone(),
        post_weights: post_weights.to_vec(),
        m,
        inference,
        original,
        identified_set,
    })
}

/// Compute the scalar smoothness FLCI using production defaults.
///
/// # Errors
/// Returns an error if the typed problem is invalid or the smoothness
/// least-favorable optimization fails.
pub fn compute_smoothness_flci(
    problem: &SmoothnessFlciProblem,
) -> Result<SmoothnessFlciResult, String> {
    compute_smoothness_flci_with_config(problem, SmoothnessFlciConfig::default_for_production())
}

/// Compute the scalar smoothness FLCI with explicit grid settings.
///
/// # Errors
/// Returns an error if the typed problem is invalid or the smoothness
/// least-favorable optimization fails.
pub fn compute_smoothness_flci_with_config(
    problem: &SmoothnessFlciProblem,
    config: SmoothnessFlciConfig,
) -> Result<SmoothnessFlciResult, String> {
    if !validate_confidence_level(problem.inference.confidence_level) {
        return Err(format!(
            "invalid confidence level {}",
            problem.inference.confidence_level
        ));
    }
    if !problem.identified_set.lb.is_finite() || !problem.identified_set.ub.is_finite() {
        return Ok(SmoothnessFlciResult {
            flci: (f64::NEG_INFINITY, f64::INFINITY),
            original_ci: problem.original.ci,
            identified_set: (problem.identified_set.lb, problem.identified_set.ub),
            optimal_vec: {
                let mut vec = vec![0.0; problem.input.num_pre_periods()];
                vec.extend_from_slice(&problem.post_weights);
                vec
            },
            optimal_half_length: f64::INFINITY,
        });
    }
    let Some((optimal_vec, optimal_half_length)) = find_optimal_flci(problem, config)? else {
        return Err("failed to compute smoothness fixed-length confidence interval".to_string());
    };
    let center = dot(&optimal_vec, &problem.input.betahat);
    Ok(SmoothnessFlciResult {
        flci: (center - optimal_half_length, center + optimal_half_length),
        original_ci: problem.original.ci,
        identified_set: (problem.identified_set.lb, problem.identified_set.ub),
        optimal_vec,
        optimal_half_length,
    })
}

/// Build the typed simultaneous smoothness-FLCI problem for a set of
/// post-treatment functionals.
///
/// # Errors
/// Returns an error if the functional list is empty or if any original /
/// identified-set calculation fails.
pub fn build_smoothness_multi_flci_problem(
    input: &HonestEventStudyInput,
    post_weight_sets: &[Vec<f64>],
    m: f64,
    inference: InferenceConfig,
) -> Result<SmoothnessMultiFlciProblem, String> {
    if post_weight_sets.is_empty() {
        return Err("multi FLCI requires at least one functional".to_string());
    }
    let mut originals = Vec::with_capacity(post_weight_sets.len());
    let mut identified_sets = Vec::with_capacity(post_weight_sets.len());
    for post_weights in post_weight_sets {
        originals.push(compute_original_confidence_set(
            input,
            post_weights,
            inference,
        )?);
        identified_sets.push(compute_smoothness_identified_set(input, post_weights, m)?);
    }
    Ok(SmoothnessMultiFlciProblem {
        input: input.clone(),
        post_weight_sets: post_weight_sets.to_vec(),
        m,
        inference,
        originals,
        identified_sets,
    })
}

fn functional_correlation_matrix(
    input: &HonestEventStudyInput,
    post_weight_sets: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    let sigma_post = post_covariance_block(
        &input.covariance,
        input.num_pre_periods(),
        input.num_post_periods(),
    );
    let projected = post_weight_sets
        .iter()
        .map(|post_weights| {
            sigma_post
                .iter()
                .map(|row| dot(post_weights, row))
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<_>>();
    let variances = projected
        .iter()
        .zip(post_weight_sets.iter())
        .map(|(sigma_l, post_weights)| dot(post_weights, sigma_l).max(0.0))
        .collect::<Vec<_>>();
    if variances.iter().any(|variance| *variance <= 1e-14) {
        return Err(
            "one or more functionals have near-zero variance under post covariance".to_string(),
        );
    }
    let stddevs = variances
        .iter()
        .map(|value| value.sqrt())
        .collect::<Vec<_>>();
    let n = post_weight_sets.len();
    let mut corr = vec![vec![0.0; n]; n];
    for i in 0..n {
        corr[i][i] = 1.0;
        for j in (i + 1)..n {
            let cov_ij = dot(&post_weight_sets[i], &projected[j]);
            let c = (cov_ij / (stddevs[i] * stddevs[j])).clamp(-1.0, 1.0);
            corr[i][j] = c;
            corr[j][i] = c;
        }
    }
    Ok(corr)
}

fn simulated_joint_pointwise_confidence_level(
    input: &HonestEventStudyInput,
    confidence_level: f64,
    post_weight_sets: &[Vec<f64>],
    simulation_draws: usize,
    simulation_seed: u64,
) -> Result<f64, String> {
    let corr = functional_correlation_matrix(input, post_weight_sets)?;
    let chol = cholesky_lower(&corr)?;
    let mut maxima =
        simulated_lower_cholesky_maxima_batched(&chol, simulation_draws, simulation_seed);
    let n = maxima.len();
    let rank = simulation_rank(n, confidence_level);
    maxima.select_nth_unstable_by(rank.min(n - 1), f64::total_cmp);
    Ok(maxima[rank.min(n - 1)])
}

fn joint_pointwise_confidence_level(
    problem: &SmoothnessMultiFlciProblem,
    joint_config: HonestJointPathConfig,
) -> Result<(f64, f64), String> {
    let n = problem.post_weight_sets.len();
    if n == 1 {
        return Ok((
            problem.inference.confidence_level,
            critical_value_from_pointwise_confidence(problem.inference.confidence_level)?,
        ));
    }
    let alpha = 1.0 - problem.inference.confidence_level;
    match joint_config.method {
        HonestJointPathMethod::Bonferroni => {
            let pointwise = 1.0 - alpha / usize_to_f64(n);
            Ok((
                pointwise,
                critical_value_from_pointwise_confidence(pointwise)?,
            ))
        }
        HonestJointPathMethod::GaussianSimulated => {
            let critical = simulated_joint_pointwise_confidence_level(
                &problem.input,
                problem.inference.confidence_level,
                &problem.post_weight_sets,
                joint_config.simulation_draws,
                joint_config.simulation_seed,
            )?;
            Ok((
                pointwise_confidence_level_from_critical(critical)?,
                critical,
            ))
        }
    }
}

/// Compute simultaneous smoothness FLCIs using production defaults.
///
/// # Errors
/// Returns an error if the typed problem is invalid or the joint calibration /
/// scalar smoothness solves fail.
pub fn compute_smoothness_multi_flci(
    problem: &SmoothnessMultiFlciProblem,
) -> Result<SmoothnessMultiFlciResult, String> {
    compute_smoothness_multi_flci_with_config(
        problem,
        SmoothnessFlciConfig::default_for_production(),
        HonestJointPathConfig::default_for_production(),
    )
}

/// Compute simultaneous smoothness FLCIs with explicit scalar and joint
/// calibration settings.
///
/// # Errors
/// Returns an error if the typed problem is invalid or the joint calibration /
/// scalar smoothness solves fail.
pub fn compute_smoothness_multi_flci_with_config(
    problem: &SmoothnessMultiFlciProblem,
    config: SmoothnessFlciConfig,
    joint_config: HonestJointPathConfig,
) -> Result<SmoothnessMultiFlciResult, String> {
    let (pointwise_confidence_level, calibrated_max_t_critical_value) =
        joint_pointwise_confidence_level(problem, joint_config)?;
    let pointwise_inference = InferenceConfig::new(pointwise_confidence_level);
    let points = problem
        .post_weight_sets
        .par_iter()
        .zip(problem.originals.par_iter())
        .zip(problem.identified_sets.par_iter())
        .map(|((post_weights, original), identified_set)| {
            let scalar_problem = SmoothnessFlciProblem {
                input: problem.input.clone(),
                post_weights: post_weights.clone(),
                m: problem.m,
                inference: pointwise_inference,
                original: original.clone(),
                identified_set: identified_set.clone(),
            };
            let flci = compute_smoothness_flci_with_config(&scalar_problem, config)?;
            Ok(SmoothnessMultiFlciPoint {
                post_weights: post_weights.clone(),
                flci: flci.flci,
                original_ci: original.ci,
                identified_set: (identified_set.lb, identified_set.ub),
                null_value: 0.0,
                robustly_significant: flci.flci.0 > 0.0 || flci.flci.1 < 0.0,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(SmoothnessMultiFlciResult {
        confidence_level: problem.inference.confidence_level,
        pointwise_confidence_level,
        calibrated_max_t_critical_value,
        method: joint_config.method,
        points,
    })
}
