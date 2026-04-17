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

use once_map::OnceMap;
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::sync::LazyLock;

use super::super::linear_algebra::{
    cholesky_lower, draw_standard_normal_vec_into, lower_mat_vec_mul_into, simulation_draw_seed,
};
use super::conditional_moment_lp_workspace::ConditionalMomentLpWorkspace;
use crate::util::usize_to_f64;

const LEAST_FAVORABLE_CV_PARALLEL_MIN_DRAWS: usize = 2_048;
const LEAST_FAVORABLE_CV_PARALLEL_MIN_DIM: usize = 16;

static LEAST_FAVORABLE_CV_CACHE: LazyLock<OnceMap<LeastFavorableCvCacheKey, Result<f64, String>>> =
    LazyLock::new(OnceMap::new);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct LeastFavorableCvCacheKey {
    x_rows: usize,
    x_cols: usize,
    sigma_rows: usize,
    sigma_cols: usize,
    hybrid_kappa_bits: u64,
    sims: usize,
    seed: u64,
    x_bits: Box<[u64]>,
    sigma_bits: Box<[u64]>,
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub(in crate::inference::sensitivity) fn compute_least_favorable_cv(
    x_matrix: &[Vec<f64>],
    sigma: &[Vec<f64>],
    hybrid_kappa: f64,
    sims: usize,
    seed: u64,
) -> Result<f64, String> {
    let cache_key = build_cache_key(x_matrix, sigma, hybrid_kappa, sims, seed);
    LEAST_FAVORABLE_CV_CACHE.insert_cloned(cache_key, |_| {
        compute_least_favorable_cv_uncached(x_matrix, sigma, hybrid_kappa, sims, seed)
    })
}

pub(in crate::inference::sensitivity) fn compute_least_favorable_cv_uncached(
    x_matrix: &[Vec<f64>],
    sigma: &[Vec<f64>],
    hybrid_kappa: f64,
    sims: usize,
    seed: u64,
) -> Result<f64, String> {
    let chol = cholesky_lower(sigma)?;
    let draw_results = if sims >= LEAST_FAVORABLE_CV_PARALLEL_MIN_DRAWS
        && sigma.len() >= LEAST_FAVORABLE_CV_PARALLEL_MIN_DIM
    {
        (0..sims)
            .into_par_iter()
            .map_init(
                || {
                    (
                        ConditionalMomentLpWorkspace::new(x_matrix, sigma),
                        vec![0.0; sigma.len()],
                        vec![0.0; sigma.len()],
                        vec![0.0; sigma.len()],
                    )
                },
                |(workspace_result, z, xi, y), draw_idx| {
                    if let Err(err) = workspace_result.as_ref() {
                        return Err(err.clone());
                    }
                    let workspace = workspace_result
                        .as_mut()
                        .expect("workspace result checked above");
                    let mut rng = StdRng::seed_from_u64(simulation_draw_seed(seed, draw_idx));
                    draw_standard_normal_vec_into(&mut rng, z);
                    lower_mat_vec_mul_into(&chol, z, xi);
                    y.iter_mut().zip(xi.iter()).for_each(|(y_value, xi_value)| {
                        *y_value = -*xi_value;
                    });
                    if workspace.solve_in_place(y).is_err() {
                        Ok(None)
                    } else {
                        Ok(Some(workspace.eta_star()))
                    }
                },
            )
            .collect::<Result<Vec<_>, String>>()?
    } else {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut workspace = ConditionalMomentLpWorkspace::new(x_matrix, sigma)?;
        let mut z = vec![0.0; sigma.len()];
        let mut xi = vec![0.0; sigma.len()];
        let mut y = vec![0.0; sigma.len()];
        let mut draws = Vec::with_capacity(sims);
        for _ in 0..sims {
            draw_standard_normal_vec_into(&mut rng, &mut z);
            lower_mat_vec_mul_into(&chol, &z, &mut xi);
            y.iter_mut().zip(xi.iter()).for_each(|(y_value, xi_value)| {
                *y_value = -*xi_value;
            });
            if workspace.solve_in_place(&y).is_err() {
                draws.push(None);
            } else {
                draws.push(Some(workspace.eta_star()));
            }
        }
        draws
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

fn build_cache_key(
    x_matrix: &[Vec<f64>],
    sigma: &[Vec<f64>],
    hybrid_kappa: f64,
    sims: usize,
    seed: u64,
) -> LeastFavorableCvCacheKey {
    LeastFavorableCvCacheKey {
        x_rows: x_matrix.len(),
        x_cols: x_matrix.first().map_or(0, Vec::len),
        sigma_rows: sigma.len(),
        sigma_cols: sigma.first().map_or(0, Vec::len),
        hybrid_kappa_bits: normalized_f64_bits(hybrid_kappa),
        sims,
        seed,
        x_bits: flatten_matrix_bits(x_matrix),
        sigma_bits: flatten_matrix_bits(sigma),
    }
}

fn flatten_matrix_bits(matrix: &[Vec<f64>]) -> Box<[u64]> {
    matrix
        .iter()
        .flat_map(|row| row.iter().copied().map(normalized_f64_bits))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn normalized_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}
