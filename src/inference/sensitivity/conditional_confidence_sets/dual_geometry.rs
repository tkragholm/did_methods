//! Dual geometry for `HonestDiD` conditional confidence sets.
//!
//! This module implements the branch used after the ARP primal program has been
//! solved. When the nonnegative dual multipliers are not already explicit, we
//! recover a feasible dual vertex from the binding set and then compute the
//! truncated-normal acceptance region used by the conditional `HonestDiD` test.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
};

use super::super::linear_algebra::{
    build_clarabel_matrix, dot, matrix_rank, solve_square_linear_system_transposed,
    truncated_normal_quantile,
};

pub(in crate::inference::sensitivity) use super::dense_workspaces::DualMaxLpWorkspace;

const DUAL_FEASIBILITY_TOL: f64 = 1e-6;
const DUAL_BOUNDARY_TOL: f64 = 1e-4;

pub(in crate::inference::sensitivity) struct DualAcceptanceRegion {
    pub(in crate::inference::sensitivity) sigma_b: f64,
    pub(in crate::inference::sensitivity) maxstat: f64,
    pub(in crate::inference::sensitivity) vlo: f64,
    pub(in crate::inference::sensitivity) vup: f64,
}

pub(in crate::inference::sensitivity) fn dual_acceptance_region(
    y_arp: &[f64],
    sigma_arp: &[Vec<f64>],
    eta_star: f64,
    gamma_tilde: &[f64],
    dual_workspace: &mut DualMaxLpWorkspace,
    sigma_gamma_scratch: &mut Vec<f64>,
    s_t_scratch: &mut Vec<f64>,
) -> Result<Option<DualAcceptanceRegion>, String> {
    sigma_gamma_scratch.resize(y_arp.len(), 0.0);
    for (row_idx, sigma_row) in sigma_arp.iter().enumerate() {
        sigma_gamma_scratch[row_idx] = dot(sigma_row, gamma_tilde);
    }
    let sigma_b2 = dot(gamma_tilde, sigma_gamma_scratch);
    if sigma_b2.abs() < f64::EPSILON {
        return Ok(None);
    }
    if sigma_b2 < 0.0 {
        return Err("HonestDiD dual branch produced negative variance".to_string());
    }
    let sigma_b = sigma_b2.sqrt();
    let maxstat = eta_star / sigma_b;
    s_t_scratch.resize(y_arp.len(), 0.0);
    let gamma_y = dot(gamma_tilde, y_arp);
    for (idx, out) in s_t_scratch.iter_mut().enumerate() {
        *out = y_arp[idx] - sigma_gamma_scratch[idx] * gamma_y / sigma_b2;
    }
    let (vlo, vup) = compute_dual_vlo_vup(
        eta_star,
        s_t_scratch,
        sigma_gamma_scratch,
        sigma_b2,
        dual_workspace,
    )?;
    Ok(Some(DualAcceptanceRegion {
        sigma_b,
        maxstat,
        vlo,
        vup,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::inference::sensitivity) fn dual_conditional_test(
    y_arp: &[f64],
    sigma_arp: &[Vec<f64>],
    eta_star: f64,
    gamma_tilde: &[f64],
    mod_size: f64,
    lf_cv: f64,
    dual_workspace: &mut DualMaxLpWorkspace,
    sigma_gamma_scratch: &mut Vec<f64>,
    s_t_scratch: &mut Vec<f64>,
) -> Result<bool, String> {
    let Some(region) = dual_acceptance_region(
        y_arp,
        sigma_arp,
        eta_star,
        gamma_tilde,
        dual_workspace,
        sigma_gamma_scratch,
        s_t_scratch,
    )?
    else {
        return Ok(eta_star > 0.0);
    };
    let zlo = region.vlo / region.sigma_b;
    let zup = region.vup.min(lf_cv) / region.sigma_b;
    if !(zlo <= region.maxstat && region.maxstat <= zup) {
        return Ok(false);
    }
    let cval = truncated_normal_quantile(1.0 - mod_size, zlo, zup)?;
    Ok(region.maxstat > cval.max(0.0))
}

pub(in crate::inference::sensitivity) fn build_v_b_row_major_into(
    binding: &[usize],
    inv_m_flat: &[f64],
    dim: usize,
    out: &mut [f64],
) {
    out.fill(0.0);
    for (binding_pos, &row_idx) in binding.iter().enumerate() {
        out[row_idx] = inv_m_flat[binding_pos];
    }
    debug_assert_eq!(inv_m_flat.len(), dim * dim);
}

pub(in crate::inference::sensitivity) fn row_nonbinding_coeff_row_major_into(
    sd: f64,
    x_row: &[f64],
    inv_m_flat: &[f64],
    dim: usize,
    out: &mut Vec<f64>,
) {
    out.clear();
    out.resize(dim, 0.0);
    for (out_idx, out_value) in out.iter_mut().enumerate().take(dim) {
        let row_start = out_idx * dim;
        let inv_row = &inv_m_flat[row_start..row_start + dim];
        *out_value = sd.mul_add(
            inv_row[0],
            x_row
                .iter()
                .enumerate()
                .map(|(idx, value)| value * inv_row[idx + 1])
                .sum::<f64>(),
        );
    }
}

pub(in crate::inference::sensitivity) fn recover_dual_vertex_from_binding(
    binding: &[usize],
    sd_vec: &[f64],
    x_arp: &[Vec<f64>],
    y_arp: &[f64],
    eta_star: f64,
) -> Result<Vec<f64>, String> {
    let k = x_arp.first().map_or(0, Vec::len);
    let target_size = k + 1;
    let candidate_rows: Vec<usize> = if binding.len() < target_size {
        (0..y_arp.len()).collect()
    } else {
        binding.to_vec()
    };
    let mut current = Vec::with_capacity(target_size);
    let mut result = None;
    let search = BindingVertexSearch {
        binding: &candidate_rows,
        target_size,
        sd_vec,
        x_arp,
        y_arp,
        eta_star,
    };
    search_binding_vertex(&search, 0, &mut current, &mut result)?;
    result.ok_or_else(|| {
        format!(
            "HonestDiD dual recovery could not find a feasible vertex (binding={}, rows={}, target={})",
            binding.len(),
            y_arp.len(),
            target_size
        )
    })
}

fn search_binding_vertex(
    context: &BindingVertexSearch<'_>,
    start: usize,
    current: &mut Vec<usize>,
    result: &mut Option<Vec<f64>>,
) -> Result<(), String> {
    if result.is_some() {
        return Ok(());
    }
    if current.len() == context.target_size {
        if let Some(gamma) = try_binding_vertex(
            current,
            context.sd_vec,
            context.x_arp,
            context.y_arp,
            context.eta_star,
        ) {
            *result = Some(gamma);
        }
        return Ok(());
    }
    for idx in start..context.binding.len() {
        current.push(context.binding[idx]);
        search_binding_vertex(context, idx + 1, current, result)?;
        current.pop();
        if result.is_some() {
            break;
        }
    }
    Ok(())
}

struct BindingVertexSearch<'a> {
    binding: &'a [usize],
    target_size: usize,
    sd_vec: &'a [f64],
    x_arp: &'a [Vec<f64>],
    y_arp: &'a [f64],
    eta_star: f64,
}

fn try_binding_vertex(
    binding_subset: &[usize],
    sd_vec: &[f64],
    x_arp: &[Vec<f64>],
    y_arp: &[f64],
    eta_star: f64,
) -> Option<Vec<f64>> {
    let mut m = Vec::with_capacity(binding_subset.len());
    for &row_idx in binding_subset {
        let mut row = Vec::with_capacity(1 + x_arp[row_idx].len());
        row.push(sd_vec[row_idx]);
        row.extend_from_slice(&x_arp[row_idx]);
        m.push(row);
    }
    if matrix_rank(&m, 1e-8) != m.len() {
        return None;
    }
    let mut first_basis = vec![0.0; m.len()];
    first_basis[0] = 1.0;
    let gamma_subset = solve_square_linear_system_transposed(&m, &first_basis).ok()?;
    if gamma_subset.iter().any(|value| *value < -1e-8) {
        return None;
    }
    let mut gamma = vec![0.0; y_arp.len()];
    for (value, &row_idx) in gamma_subset.iter().zip(binding_subset.iter()) {
        gamma[row_idx] = value.max(0.0);
    }
    let mut w_gamma = vec![0.0; 1 + x_arp.first().map_or(0, Vec::len)];
    for (row_idx, gamma_value) in gamma.iter().enumerate() {
        w_gamma[0] += gamma_value * sd_vec[row_idx];
        for (col_idx, x_value) in x_arp[row_idx].iter().enumerate() {
            w_gamma[col_idx + 1] += gamma_value * x_value;
        }
    }
    if (w_gamma[0] - 1.0).abs() > 1e-4 || w_gamma[1..].iter().any(|value| value.abs() > 1e-4) {
        return None;
    }
    if (dot(&gamma, y_arp) - eta_star).abs() > 1e-3 {
        return None;
    }
    Some(gamma)
}

fn compute_dual_vlo_vup(
    eta: f64,
    s_t: &[f64],
    sigma_gamma: &[f64],
    sigma_b2: f64,
    workspace: &mut DualMaxLpWorkspace,
) -> Result<(f64, f64), String> {
    let sigma_b = sigma_b2.max(0.0).sqrt();
    let low_initial = (-100.0_f64).min(20.0f64.mul_add(-sigma_b, eta));
    let high_initial = 100.0f64.max(20.0f64.mul_add(sigma_b, eta));
    if !check_dual_solution(
        eta,
        DUAL_FEASIBILITY_TOL,
        s_t,
        sigma_gamma,
        sigma_b2,
        workspace,
    )? {
        return Ok((eta, f64::INFINITY));
    }

    let vup = if check_dual_solution(
        high_initial,
        DUAL_FEASIBILITY_TOL,
        s_t,
        sigma_gamma,
        sigma_b2,
        workspace,
    )? {
        f64::INFINITY
    } else {
        bisect_dual_boundary(
            high_initial,
            eta,
            true,
            s_t,
            sigma_gamma,
            sigma_b2,
            workspace,
        )?
    };

    let vlo = if check_dual_solution(
        low_initial,
        DUAL_FEASIBILITY_TOL,
        s_t,
        sigma_gamma,
        sigma_b2,
        workspace,
    )? {
        f64::NEG_INFINITY
    } else {
        bisect_dual_boundary(
            eta,
            low_initial,
            false,
            s_t,
            sigma_gamma,
            sigma_b2,
            workspace,
        )?
    };
    Ok((vlo, vup))
}

fn bisect_dual_boundary(
    mut high: f64,
    mut low: f64,
    low_is_solution: bool,
    s_t: &[f64],
    sigma_gamma: &[f64],
    sigma_b2: f64,
    workspace: &mut DualMaxLpWorkspace,
) -> Result<f64, String> {
    for _ in 0..10_000 {
        if (high - low).abs() <= DUAL_BOUNDARY_TOL {
            return Ok((high + low) * 0.5);
        }
        let mid = (high + low) * 0.5;
        let honest = check_dual_solution(
            mid,
            DUAL_FEASIBILITY_TOL,
            s_t,
            sigma_gamma,
            sigma_b2,
            workspace,
        )?;
        if low_is_solution {
            if honest {
                low = mid;
            } else {
                high = mid;
            }
        } else if honest {
            high = mid;
        } else {
            low = mid;
        }
    }
    Ok((high + low) * 0.5)
}

fn check_dual_solution(
    c: f64,
    tol: f64,
    s_t: &[f64],
    sigma_gamma: &[f64],
    sigma_b2: f64,
    workspace: &mut DualMaxLpWorkspace,
) -> Result<bool, String> {
    let optimum = workspace.solve_for_c(s_t, sigma_gamma, sigma_b2, c)?;
    Ok((c - optimum).abs() <= tol)
}

pub(super) fn solve_dual_max_with_clarabel_fallback(
    w_t: &[Vec<f64>],
    f: &[f64],
) -> Result<f64, String> {
    let dim = w_t.len();
    let width = w_t.first().map_or(0, Vec::len);
    let inequalities: Vec<Vec<f64>> = (0..dim)
        .map(|idx| {
            let mut row = vec![0.0; dim];
            row[idx] = -1.0;
            row
        })
        .collect();
    let mut equalities = Vec::with_capacity(width);
    for col_idx in 0..width {
        equalities.push(w_t.iter().map(|row| row[col_idx]).collect());
    }
    let constraint_matrix = build_clarabel_matrix(&inequalities, &equalities);
    let mut rhs = vec![0.0; dim];
    rhs.push(1.0);
    rhs.extend(std::iter::repeat_n(0.0, width.saturating_sub(1)));
    let cones = vec![
        SupportedConeT::NonnegativeConeT(dim),
        SupportedConeT::ZeroConeT(width),
    ];
    let quadratic = CscMatrix::<f64>::zeros((dim, dim));
    let q = f.iter().map(|value| -*value).collect::<Vec<_>>();
    let settings = DefaultSettingsBuilder::<f64>::default()
        .verbose(false)
        .build()
        .map_err(|err| format!("failed to build Clarabel settings: {err}"))?;
    let mut solver = DefaultSolver::new(&quadratic, &q, &constraint_matrix, &rhs, &cones, settings)
        .map_err(|err| format!("failed to initialize Clarabel dual max LP: {err}"))?;
    solver.solve();
    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => Ok(dot(&solver.solution.x, f)),
        SolverStatus::PrimalInfeasible
        | SolverStatus::DualInfeasible
        | SolverStatus::AlmostPrimalInfeasible
        | SolverStatus::AlmostDualInfeasible => {
            Err("Clarabel dual max program is infeasible".to_string())
        }
        status => Err(format!(
            "Clarabel failed to solve HonestDiD dual max LP: {status:?}"
        )),
    }
}
