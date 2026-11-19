use epistat_rng::Stream;

use super::{linear_algebra, relative_magnitude};

#[doc(hidden)]
#[must_use]
pub fn benchmark_sensitivity_normal_draws(count: usize, seed: u64) -> f64 {
    let mut rng = Stream::seeded(seed);
    let mut out = vec![0.0; count];
    rng.fill_normals(&mut out);
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
pub fn benchmark_sensitivity_multi_flci_critical_value(
    covariance: &[Vec<f64>],
    simulation_draws: usize,
    simulation_seed: u64,
) -> f64 {
    relative_magnitude::least_favorable_intervals::benchmark_multi_flci_critical_value(
        covariance,
        simulation_draws,
        simulation_seed,
    )
    .unwrap_or(f64::NAN)
}
