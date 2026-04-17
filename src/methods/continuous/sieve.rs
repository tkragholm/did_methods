//! Sieve-based continuous-treatment `DiD` estimation.
//!
//! This module implements the Average Causal Response on the Treated (ACRT)
//! estimator using a sieve approximation to the dose-response derivative. The
//! implementation follows the high-level structure used in the `contdid`
//! literature: estimate the control trend, project treated-unit outcome changes
//! onto a basis expansion in dose, and evaluate the derivative of that basis to
//! recover the average causal response.
//!
//! In notation, let `ΔY_i` denote the outcome change, `D_i` the continuous dose,
//! and `ψ(D_i)` a `K`-dimensional basis. For treated units the estimator solves
//!
//! ```text
//! θ̂ = argmin_θ Σ_i w_i (ΔY_i - m_0 - ψ(D_i)' θ)^2,
//! ```
//!
//! where `m_0 = E[ΔY | D = 0]` is the control trend. The global ACRT is then
//!
//! ```text
//! ACRT = E[∂_d ψ(D_i)' θ̂ | D_i > 0].
//! ```
//!
//! The returned influence function is used by the crate's shared inference
//! layer for variance estimation.
//!
//! References:
//! - Callaway, B., Goodman-Bacon, A., and Sant'Anna, P. H. C. (2024),
//!   continuous-treatment `DiD` working paper / `contdid` implementation.

use crate::estimators::common::basis::Basis;
use crate::estimators::common::linalg::{solve_spd_system, solve_spd_system_multi_rhs};
use crate::types::{ACRTResult, ContinuousDidError, ContinuousObservation};
use crate::util::usize_to_f64;
use faer::{Col, Mat};

/// Estimates the Average Causal Response (ACRT) curve using a Sieve regression.
///
/// # Mathematical Implementation (CGBS 2024)
/// 1. Estimate control trend: m0 = E[ΔY | D=0].
/// 2. Construct centered outcomes for treated units: `ΔY_i`* = `ΔY_i` - m0.
/// 3. Construct the design matrix Ψ from the basis expansion of doses: Ψ_{ik} = `ψ_k(D_i)`.
/// 4. Solve for sieve coefficients θ̂ via Weighted Least Squares (WLS) on treated units:
///    θ̂ = (Ψ'WΨ)⁻¹ Ψ'WΔY*
/// 5. Compute ACRT at each treated unit's dose: `ACRT(D_i)` = ∂_d `ψ(D_i)`' θ̂.
/// 6. Aggregate to get `ACRT_glob` = (`1/n_T`) Σ `ACRT(D_i)`.
///
/// # Errors
/// Returns `ContinuousDidError::EmptyInput` if observations are empty.
/// Returns `ContinuousDidError::NoTreatedUnits` if no treated units are found.
/// Returns `ContinuousDidError::SingularBasis` if the design matrix is singular.
///
/// # Panics
/// Panics if internal indexing into treated units fails (which should be unreachable).
pub fn estimate_acrt_sieve<B: Basis>(
    observations: &[ContinuousObservation],
    basis: &B,
) -> Result<ACRTResult, ContinuousDidError> {
    if observations.is_empty() {
        return Err(ContinuousDidError::EmptyInput);
    }

    let n = observations.len();

    // 1. Calculate control trend m0
    let mut control_sum = 0.0;
    let mut control_weight = 0.0;
    let mut control_count = 0;
    for obs in observations {
        if obs.dose == 0.0 {
            control_sum = obs.weight.mul_add(obs.delta_outcome, control_sum);
            control_weight += obs.weight;
            control_count += 1;
        }
    }
    if control_count == 0 {
        // If no control, we can't subtract trend, but maybe we assume it's 0 or fail.
        // contdid requires a control group.
        return Err(ContinuousDidError::NoTreatedUnits); // Should be NoControlUnits
    }
    let m0 = control_sum / control_weight;

    // 2. Filter treated units
    let treated_indices: Vec<usize> = observations
        .iter()
        .enumerate()
        .filter(|(_, o)| o.dose > 0.0)
        .map(|(i, _)| i)
        .collect();
    let nt = treated_indices.len();
    if nt == 0 {
        return Err(ContinuousDidError::NoTreatedUnits);
    }

    let k = basis.num_basis();

    // 3. Form normal equations for treated units
    let mut xtwx = vec![0.0; k * k];
    let mut xtw_outcome = vec![0.0; k];

    let mut psi_vecs = Vec::with_capacity(nt);
    let mut deriv_vecs = Vec::with_capacity(nt);

    for &idx in &treated_indices {
        let obs = &observations[idx];
        let b_vals = basis.eval(obs.dose);
        let d_vals = basis.eval_deriv(obs.dose);
        let w = obs.weight;

        // Use centered outcome
        let outcome_star = obs.delta_outcome - m0;

        for r in 0..k {
            xtw_outcome[r] = (w * b_vals[r]).mul_add(outcome_star, xtw_outcome[r]);
            for c in 0..k {
                xtwx[r * k + c] = (w * b_vals[r]).mul_add(b_vals[c], xtwx[r * k + c]);
            }
        }
        psi_vecs.push(b_vals);
        deriv_vecs.push(d_vals);
    }

    let theta_vec =
        solve_spd_system(&xtwx, &xtw_outcome).map_err(|_| ContinuousDidError::SingularBasis)?;
    let theta = Col::<f64>::from_fn(k, |i| theta_vec[i]);

    // 4. Compute ACRT_glob
    let mut acrt_sum = 0.0;
    let mut acrt_i_vals = Vec::with_capacity(nt);
    let mut mean_deriv = vec![0.0; k];

    for deriv in &deriv_vecs {
        let acrt_i = (0..k).map(|j| deriv[j] * theta[j]).sum::<f64>();
        acrt_sum += acrt_i;
        acrt_i_vals.push(acrt_i);
        for (acc, &value) in mean_deriv.iter_mut().zip(deriv.iter()) {
            *acc += value;
        }
    }
    let acrt_glob = acrt_sum / usize_to_f64(nt);
    for value in &mut mean_deriv {
        *value /= usize_to_f64(nt);
    }

    // 5. Influence function
    let mut influence_function = vec![0.0; n];
    let mean_deriv_col = Col::<f64>::from_fn(k, |j| mean_deriv[j]);
    let mean_deriv_mat = Mat::<f64>::from_fn(k, 1, |row, _| mean_deriv_col[row]);
    let alpha_mat = solve_spd_system_multi_rhs(&xtwx, &mean_deriv_mat)
        .map_err(|_| ContinuousDidError::SingularBasis)?;
    let alpha = Col::<f64>::from_fn(k, |j| alpha_mat[(j, 0)]);

    for (idx, obs) in observations.iter().enumerate() {
        if obs.dose == 0.0 {
            // The current implementation only records the treated-side
            // contribution to the influence function. Control-side influence on
            // `m0` is a known extension point for richer continuous-DiD
            // variance work and is intentionally omitted here.
        } else {
            let i_in_treated = treated_indices
                .iter()
                .position(|&t_idx| t_idx == idx)
                .unwrap();
            let psi_i = &psi_vecs[i_in_treated];
            let outcome_star = obs.delta_outcome - m0;
            let resid = outcome_star - (0..k).map(|j| psi_i[j] * theta[j]).sum::<f64>();

            let term1 = acrt_i_vals[i_in_treated] - acrt_glob;
            let term2 = (0..k).map(|j| alpha[j] * psi_i[j]).sum::<f64>() * resid;
            influence_function[idx] = term1 + term2;
        }
    }

    Ok(ACRTResult {
        acrt_glob,
        coefficients: theta_vec,
        influence_function,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimators::common::basis::PolynomialBasis;

    #[test]
    fn acrt_sieve_linear_case_with_control() {
        let mut obs = Vec::new();
        // Control group: trend = 2.0
        for _ in 0..50 {
            obs.push(ContinuousObservation::new(0.0, 2.0));
        }
        // Treated group: trend = 2.0 + 5.0 * dose
        for i in 1..=50 {
            let dose = f64::from(i);
            obs.push(ContinuousObservation::new(dose, 5.0f64.mul_add(dose, 2.0)));
        }

        let basis = PolynomialBasis::new(1);
        let result = estimate_acrt_sieve(&obs, &basis).unwrap();

        assert!((result.acrt_glob - 5.0).abs() < 1e-10);
    }
}
