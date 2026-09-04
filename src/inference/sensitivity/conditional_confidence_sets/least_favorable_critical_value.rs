//! Least-favorable critical values for `HonestDiD` conditional confidence sets.
//!
//! For ARP test inversion, the conditional critical value is approximated by
//! Monte Carlo draws from the least-favorable Gaussian process. This helper is
//! family-agnostic: it only depends on the ARP design matrix and covariance
//! surface, not on whether those came from `DeltaRM`, `DeltaSD`, or another
//! restriction class.
//!
//! The conditioning and least-favorable calibration logic follows the
//! conditional moment-inequality approach used by Rambachan and Roth's
//! `HonestDiD` implementation, which itself builds on the ARP framework:
//!
//! - Rambachan, A. and Roth, J. (2023). "A More Credible Approach to Parallel
//!   Trends". *Review of Economic Studies* 90(5), 2555-2591.
//! - Andrews, I., Roth, J., and Pakes, A. (2022). "Inference for Linear
//!   Conditional Moment Inequalities". *Econometrica* 90(5), 2345-2377.

use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use super::super::linear_algebra::{
    cholesky_lower, draw_standard_normal_vec_into, lower_mat_vec_mul_into, simulation_draw_seed,
};
use super::ConditionalMomentLpWorkspace;
use crate::util::usize_to_f64;

/// Draw count above which the simulation splits across rayon.
///
/// Every call site asks for 1,000 draws, so this used to be unreachable: the
/// gate was 2,048 and the parallel arm was dead code. It is 512 now, which the
/// production call does cross. Nested inside an outer parallel loop over
/// functionals this buys little — rayon's pool is already saturated and the
/// work is merely re-divided — but a caller assessing ONE functional had every
/// core but one idle for the whole simulation, and that is the shape a test and
/// the tail of a fan-out both have.
const LEAST_FAVORABLE_CV_PARALLEL_MIN_DRAWS: usize = 512;
const LEAST_FAVORABLE_CV_PARALLEL_MIN_DIM: usize = 16;
/// Fewest draws a rayon job takes. Without a floor, nested parallelism split a
/// thousand draws into about 128 jobs, each paying for its own workspace.
const LEAST_FAVORABLE_CV_MIN_DRAWS_PER_JOB: usize = 32;

/// Compute the least-favorable critical value for an ARP design.
///
/// There is no memo. A process-wide `OnceMap` keyed on the bit patterns of
/// `x_matrix` and `sigma` sat here until 4 September 2026 and nothing evicted
/// from it. It could not: every analysis slice carries its own covariance, so a
/// key is only ever reachable again inside the slice that made it. Measured on
/// Study I's shape it answered 140 of 1,440 calls — 9.7% — and retained 15.3 MB
/// per slice-anchor for good. Resident memory grew 16 MB per slice and stayed
/// grown (157 MB after eight, against a flat 30 MB with the memo off), and the
/// production stage runs 108 slices at two anchors in one process.
///
/// The 9.7% is real and is now paid. It is the cheaper side of the trade: the
/// key was two boxed bit-vectors built and hashed on every call including the
/// 90% that missed, and the free-column rewrite in
/// [`ConditionalMomentLpWorkspace`] took 2.5x off this simulation, which is
/// more than the memo ever returned.
///
/// # Errors
/// Returns an error if the covariance is not positive definite or too many
/// draws fail to solve.
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub(in crate::inference::sensitivity) fn compute_least_favorable_cv(
    x_matrix: &[Vec<f64>],
    sigma: &[Vec<f64>],
    hybrid_kappa: f64,
    sims: usize,
    seed: u64,
) -> Result<f64, String> {
    compute_least_favorable_cv_uncached(x_matrix, sigma, hybrid_kappa, sims, seed)
}

pub(in crate::inference::sensitivity) fn compute_least_favorable_cv_uncached(
    x_matrix: &[Vec<f64>],
    sigma: &[Vec<f64>],
    hybrid_kappa: f64,
    sims: usize,
    seed: u64,
) -> Result<f64, String> {
    let draws = simulation_draws(sigma, sims, seed)?;
    compute_least_favorable_cv_from_draws(x_matrix, sigma, hybrid_kappa, &draws)
}

/// Whether the draws split across rayon. The two arms seed their generators
/// differently, so this is part of what the draws are, not only how fast they
/// come.
const fn parallel_arm(sims: usize, dim: usize) -> bool {
    sims >= LEAST_FAVORABLE_CV_PARALLEL_MIN_DRAWS && dim >= LEAST_FAVORABLE_CV_PARALLEL_MIN_DIM
}

/// The `sims` draws of `-ξ`, `ξ ~ N(0, sigma)`, that the least-favorable
/// simulation solves against, laid out draw after draw.
///
/// They depend on the covariance alone, and twenty functionals of one branch
/// share the covariance, so a caller that computes them once and hands them to
/// [`compute_least_favorable_cv_from_draws`] for each functional gets the same
/// critical values for a twentieth of the generation. The parallel arm seeds a
/// generator per draw; the serial arm draws sequentially from one, as it always
/// has.
///
/// # Errors
/// Returns an error if the covariance is not positive definite.
pub(in crate::inference::sensitivity) fn simulation_draws(
    sigma: &[Vec<f64>],
    sims: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    let chol = cholesky_lower(sigma)?;
    let dim = sigma.len();
    let mut draws = vec![0.0; sims * dim];
    if parallel_arm(sims, dim) {
        // Each draw seeds its own generator, so the draws can be made in any
        // order on any thread and come out the same.
        draws
            .par_chunks_exact_mut(dim)
            .with_min_len(LEAST_FAVORABLE_CV_MIN_DRAWS_PER_JOB)
            .enumerate()
            .for_each_init(
                || (vec![0.0; dim], vec![0.0; dim]),
                |(z, xi), (draw_idx, y)| {
                    let mut rng = StdRng::seed_from_u64(simulation_draw_seed(seed, draw_idx));
                    draw_standard_normal_vec_into(&mut rng, z);
                    lower_mat_vec_mul_into(&chol, z, xi);
                    y.iter_mut().zip(xi.iter()).for_each(|(y_value, xi_value)| {
                        *y_value = -*xi_value;
                    });
                },
            );
    } else {
        let mut z = vec![0.0; dim];
        let mut xi = vec![0.0; dim];
        let mut rng = StdRng::seed_from_u64(seed);
        for y in draws.chunks_exact_mut(dim) {
            draw_standard_normal_vec_into(&mut rng, &mut z);
            lower_mat_vec_mul_into(&chol, &z, &mut xi);
            y.iter_mut().zip(xi.iter()).for_each(|(y_value, xi_value)| {
                *y_value = -*xi_value;
            });
        }
    }
    Ok(draws)
}

/// The least-favorable critical value from draws made by [`simulation_draws`].
///
/// # Errors
/// Returns an error if the draws do not tile the covariance's dimension, the
/// workspace cannot be built, or too many draws fail to solve.
pub(in crate::inference::sensitivity) fn compute_least_favorable_cv_from_draws(
    x_matrix: &[Vec<f64>],
    sigma: &[Vec<f64>],
    hybrid_kappa: f64,
    draws: &[f64],
) -> Result<f64, String> {
    let dim = sigma.len();
    if dim == 0 || !draws.len().is_multiple_of(dim) {
        return Err(format!(
            "least-favorable draws of length {} do not tile a dimension of {dim}",
            draws.len()
        ));
    }
    let sims = draws.len() / dim;
    let draw_results = if parallel_arm(sims, dim) {
        // One workspace per rayon job, cloned from a template whose phase one
        // is already done. Rayon splits a thousand draws into well over a
        // hundred jobs under nested parallelism, so the job count is capped
        // too.
        let template = {
            let mut template = ConditionalMomentLpWorkspace::new(x_matrix, sigma);
            if let Ok(workspace) = template.as_mut() {
                workspace.prepare();
            }
            template
        };
        let make_workspace = || template.clone();
        draws
            .par_chunks_exact(dim)
            .with_min_len(LEAST_FAVORABLE_CV_MIN_DRAWS_PER_JOB)
            .map_init(make_workspace, |workspace_result, y| {
                if let Err(err) = workspace_result.as_ref() {
                    return Err(err.clone());
                }
                let workspace = workspace_result
                    .as_mut()
                    .expect("workspace result checked above");
                Ok(workspace.solve_eta_only(y).ok())
            })
            .collect::<Result<Vec<_>, String>>()?
    } else {
        let mut workspace = ConditionalMomentLpWorkspace::new(x_matrix, sigma)?;
        draws
            .chunks_exact(dim)
            .map(|y| workspace.solve_eta_only(y).ok())
            .collect()
    };
    let failed = draw_results.iter().filter(|eta| eta.is_none()).count();
    let mut etas = draw_results.into_iter().flatten().collect::<Vec<_>>();
    let min_successful = sims / 2;
    if etas.len() < min_successful {
        return Err(format!(
            "least-favorable CV simulation: too many LP failures ({failed}/{sims}); \
             cannot estimate critical value reliably"
        ));
    }
    let p = (1.0 - hybrid_kappa).clamp(0.0, 1.0);
    let last_index = etas.len().saturating_sub(1);
    let target_rank = (usize_to_f64(last_index) * p).round();
    let idx = (0..=last_index)
        .find(|candidate| usize_to_f64(*candidate) >= target_rank)
        .unwrap_or(last_index);
    etas.select_nth_unstable_by(idx, f64::total_cmp);
    etas.get(idx)
        .copied()
        .ok_or_else(|| "failed to compute least-favorable critical value".to_string())
}
