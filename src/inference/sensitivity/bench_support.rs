use rand::{SeedableRng, rngs::StdRng};

use super::{linear_algebra, relative_magnitude};

#[doc(hidden)]
#[must_use]
pub fn benchmark_sensitivity_normal_draws(count: usize, seed: u64) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = vec![0.0; count];
    linear_algebra::draw_standard_normal_vec_into(&mut rng, &mut out);
    out.iter().sum()
}

#[doc(hidden)]
#[must_use]
pub fn benchmark_sensitivity_normal_draws_scalar(count: usize, seed: u64) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = vec![0.0; count];
    linear_algebra::draw_standard_normal_vec_into_scalar(&mut rng, &mut out);
    out.iter().sum()
}

#[doc(hidden)]
#[must_use]
pub fn benchmark_sensitivity_matrix_rank(matrix: &[Vec<f64>], tol: f64) -> usize {
    linear_algebra::matrix_rank(matrix, tol)
}

#[doc(hidden)]
#[must_use]
pub fn benchmark_sensitivity_sandwich_covariance(left: &[Vec<f64>], sigma: &[Vec<f64>]) -> f64 {
    linear_algebra::sandwich_covariance(left, sigma)
        .iter()
        .flatten()
        .sum()
}

#[doc(hidden)]
#[must_use]
pub fn benchmark_sensitivity_rref_pivot_columns(matrix: &[Vec<f64>], tol: f64) -> usize {
    relative_magnitude::geometry::rref_pivot_columns(matrix, tol)
        .map_or(0, |pivot_columns| pivot_columns.len())
}

#[doc(hidden)]
#[must_use]
pub fn benchmark_sensitivity_multi_flci_maxima(
    chol: &[Vec<f64>],
    simulation_draws: usize,
    simulation_seed: u64,
) -> f64 {
    relative_magnitude::least_favorable_intervals::benchmark_multi_flci_maxima(
        chol,
        simulation_draws,
        simulation_seed,
    )
    .iter()
    .sum()
}

#[doc(hidden)]
#[must_use]
pub fn benchmark_sensitivity_multi_flci_maxima_scalar(
    chol: &[Vec<f64>],
    simulation_draws: usize,
    simulation_seed: u64,
) -> f64 {
    relative_magnitude::least_favorable_intervals::benchmark_multi_flci_maxima_scalar(
        chol,
        simulation_draws,
        simulation_seed,
    )
    .iter()
    .sum()
}
