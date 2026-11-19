//! Shared statistical inference utilities (critical values, IF-based SE, bootstrap CI).

use epistat_inference::Clusters;
use std::hash::Hash;

use thiserror::Error;

pub mod export;
// Gated: the HonestDiD sensitivity machinery is the only consumer of the
// `clarabel` conic solver. Off by default so the register pipeline does not
// link it. Enable with `--features honest`.
#[cfg(feature = "honest")]
pub mod sensitivity;
pub mod vcov;

use crate::types::{BootstrapConfig, InferenceConfig};

/// Typed cluster specification — prevents label/data misalignment.
#[derive(Debug, Clone)]
pub enum ClusterSpec<L: Eq + Hash + Clone> {
    /// One-way clustering (e.g., by person ID).
    OneWay(Vec<L>),
}

impl<L: Eq + Hash + Clone> ClusterSpec<L> {
    #[must_use]
    pub const fn one_way(labels: Vec<L>) -> Self {
        Self::OneWay(labels)
    }
}

/// Internal compact cluster index mapping rows to integer group IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterIndex {
    pub group_assignments: Vec<usize>,
    pub n_groups: usize,
}

impl ClusterIndex {
    fn clusters(&self) -> Clusters {
        Clusters::from_labels(&self.group_assignments)
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum InferenceError {
    #[error("clustered variance requires non-empty influence-function values")]
    EmptyInfluenceFunction,
    #[error(
        "clustered variance requires aligned influence/cluster lengths ({influence_len} vs {cluster_len})"
    )]
    MisalignedClusterLengths {
        influence_len: usize,
        cluster_len: usize,
    },
    #[error("clustered variance is not finite")]
    NonFiniteClusteredVariance,
    #[error("clustered variance needs at least two clusters")]
    TooFewClusters,
    #[error("invalid confidence level for clustered CI: {0}")]
    InvalidConfidenceLevel(f64),
}

#[must_use]
pub fn validate_confidence_level(confidence_level: f64) -> bool {
    confidence_level.is_finite() && confidence_level > 0.0 && confidence_level < 1.0
}

/// The Wald interval `estimate -/+ z se` at the given confidence level, `NaN`
/// at both ends when the level is outside (0, 1).
pub(crate) fn normal_interval(estimate: f64, se: f64, confidence_level: f64) -> (f64, f64) {
    epistat_inference::interval::normal(estimate, se, 1.0 - confidence_level)
        .unwrap_or((f64::NAN, f64::NAN))
}

/// Two-sided normal critical value for a confidence level in (0, 1).
#[must_use]
pub fn z_score_for_confidence(confidence_level: f64) -> f64 {
    inverse_standard_normal_cdf(0.5 * confidence_level + 0.5)
}

/// Standard error of an estimate from its influence function: the standard
/// deviation of the values divided by `sqrt(n)`. NaN for an empty influence
/// function.
#[must_use]
pub fn standard_error_from_influence(influence_function: &[f64]) -> f64 {
    epistat_inference::influence::standard_error(influence_function).unwrap_or(f64::NAN)
}

fn map_influence_error(error: &epistat_inference::InferenceError) -> InferenceError {
    use epistat_inference::InferenceError as Shared;
    match *error {
        Shared::Length {
            expected, found, ..
        } => InferenceError::MisalignedClusterLengths {
            influence_len: expected,
            cluster_len: found,
        },
        Shared::Empty => InferenceError::EmptyInfluenceFunction,
        _ => InferenceError::NonFiniteClusteredVariance,
    }
}

/// Computes the clustered variance from influence function values using a pre-computed index.
///
/// # Errors
/// Returns an error if:
/// - Influence function is empty.
/// - Influence function and cluster index have different lengths.
/// - The resulting variance is not finite.
pub fn clustered_variance_from_index(
    influence_function: &[f64],
    cluster_index: &ClusterIndex,
) -> Result<f64, InferenceError> {
    epistat_inference::influence::clustered_variance(influence_function, &cluster_index.clusters())
        .map_err(|error| map_influence_error(&error))?
        .ok_or(InferenceError::TooFewClusters)
}

/// Computes the clustered standard error from influence function values using a pre-computed index.
///
/// # Errors
/// Returns an error if clustered variance computation fails.
pub fn clustered_standard_error_from_index(
    influence_function: &[f64],
    cluster_index: &ClusterIndex,
) -> Result<f64, InferenceError> {
    clustered_variance_from_index(influence_function, cluster_index).map(f64::sqrt)
}

/// Computes the clustered confidence interval from influence function values using a pre-computed index.
///
/// # Errors
/// Returns an error if:
/// - Confidence level is invalid.
/// - Clustered standard error computation fails.
pub fn clustered_ci_from_index(
    estimate: f64,
    influence_function: &[f64],
    cluster_index: &ClusterIndex,
    inference: InferenceConfig,
) -> Result<(f64, f64, f64), InferenceError> {
    if !validate_confidence_level(inference.confidence_level) {
        return Err(InferenceError::InvalidConfidenceLevel(
            inference.confidence_level,
        ));
    }
    let se = clustered_standard_error_from_index(influence_function, cluster_index)?;
    let (low, high) = normal_interval(estimate, se, inference.confidence_level);
    Ok((se, low, high))
}

/// Computes the clustered variance from influence function values. Clusters
/// are numbered in order of first appearance, so the sum does not depend on
/// hashing.
///
/// # Errors
/// Returns an error if:
/// - Influence function is empty.
/// - Influence function and cluster labels have different lengths.
/// - The resulting variance is not finite.
pub fn clustered_variance_from_influence<L: Eq + Hash>(
    influence_function: &[f64],
    cluster_labels: &[L],
) -> Result<f64, InferenceError> {
    epistat_inference::influence::clustered_variance(
        influence_function,
        &Clusters::from_labels(cluster_labels),
    )
    .map_err(|error| map_influence_error(&error))?
    .ok_or(InferenceError::TooFewClusters)
}

/// Computes the clustered standard error from influence function values.
///
/// # Errors
/// Returns an error if clustered variance computation fails.
pub fn clustered_standard_error_from_influence<L: Eq + Hash>(
    influence_function: &[f64],
    cluster_labels: &[L],
) -> Result<f64, InferenceError> {
    clustered_variance_from_influence(influence_function, cluster_labels).map(f64::sqrt)
}

/// Computes the clustered confidence interval from influence function values.
///
/// # Errors
/// Returns an error if:
/// - Confidence level is invalid.
/// - Clustered standard error computation fails.
pub fn clustered_ci_from_influence<L: Eq + Hash>(
    estimate: f64,
    influence_function: &[f64],
    cluster_labels: &[L],
    inference: InferenceConfig,
) -> Result<(f64, f64, f64), InferenceError> {
    if !validate_confidence_level(inference.confidence_level) {
        return Err(InferenceError::InvalidConfidenceLevel(
            inference.confidence_level,
        ));
    }
    let se = clustered_standard_error_from_influence(influence_function, cluster_labels)?;
    let (low, high) = normal_interval(estimate, se, inference.confidence_level);
    Ok((se, low, high))
}

/// Percentile interval of the Rademacher multiplier bootstrap of an estimate
/// with the given influence function. Replicate `r` is seeded with
/// `replicate_seed(seed, r)`, so the interval does not depend on the number of
/// threads. An empty influence function gives `(NaN, NaN)`.
#[must_use]
pub fn multiplier_bootstrap_ci(
    estimate: f64,
    influence_function: &[f64],
    inference: InferenceConfig,
    bootstrap: BootstrapConfig,
) -> (f64, f64) {
    let Ok(mut draws) = epistat_inference::bootstrap::multiplier(
        estimate,
        influence_function,
        bootstrap.reps,
        bootstrap.seed,
    ) else {
        return (f64::NAN, f64::NAN);
    };

    let alpha = (1.0 - inference.confidence_level).clamp(0.0, 1.0);
    let lower = quantile_unsorted(&mut draws, alpha / 2.0);
    let upper = quantile_unsorted(&mut draws, 1.0 - alpha / 2.0);
    (lower, upper)
}

/// Type 7 quantile of a sorted sample, with `q` clamped to `[0, 1]`. NaN for an
/// empty sample.
#[must_use]
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    epistat_stats::quantile::type7(sorted, q.clamp(0.0, 1.0)).unwrap_or(f64::NAN)
}

/// Type 7 quantile of an unsorted sample, which it partially reorders, with
/// `q` clamped to `[0, 1]`. NaN for an empty sample.
#[must_use]
pub fn quantile_unsorted(values: &mut [f64], q: f64) -> f64 {
    epistat_stats::quantile::type7_unsorted(values, q.clamp(0.0, 1.0)).unwrap_or(f64::NAN)
}

/// Inverse of the standard normal distribution function (Wichura's AS 241).
#[inline]
#[must_use]
pub fn inverse_standard_normal_cdf(p: f64) -> f64 {
    epistat_stats::normal::quantile(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::usize_to_f64;
    use epistat_rng::Stream;

    fn multiplier_bootstrap_ci_sorted_reference(
        estimate: f64,
        influence_function: &[f64],
        inference: InferenceConfig,
        bootstrap: BootstrapConfig,
    ) -> (f64, f64) {
        let n_f = usize_to_f64(influence_function.len());
        let mut draws = Vec::with_capacity(bootstrap.reps);

        for replicate in 0..bootstrap.reps as u64 {
            let mut stream =
                epistat_rng::Stream::seeded(epistat_rng::replicate_seed(bootstrap.seed, replicate));
            let perturbation = epistat_rng::rademacher_sum(influence_function, &mut stream);
            draws.push(estimate + perturbation / n_f);
        }
        draws.sort_by(f64::total_cmp);

        let alpha = (1.0 - inference.confidence_level).clamp(0.0, 1.0);
        let lower = quantile_sorted(&draws, alpha / 2.0);
        let upper = quantile_sorted(&draws, 1.0 - alpha / 2.0);
        (lower, upper)
    }

    fn rademacher_perturbation_scalar(influence_function: &[f64], rng: &mut Stream) -> f64 {
        use rand::RngExt;

        influence_function.iter().fold(0.0, |acc, value| {
            let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            acc + sign * value
        })
    }

    use proptest::prelude::*;
    use rstest::rstest;

    proptest! {
        #[test]
        fn standard_error_is_invariant_to_permutation(mut values in prop::collection::vec(-100.0..100.0f64, 1..100)) {
            let se1 = standard_error_from_influence(&values);
            values.reverse();
            let se2 = standard_error_from_influence(&values);
            prop_assert!((se1 - se2).abs() < 1e-12);
        }

        #[test]
        fn quantile_unsorted_matches_reference_proptest(
            mut values in prop::collection::vec(-1000.0..1000.0f64, 1..200),
            q in 0.0..1.0f64
        ) {
            let mut sorted = values.clone();
            sorted.sort_by(f64::total_cmp);
            let expected = quantile_sorted(&sorted, q);
            let actual = quantile_unsorted(&mut values, q);
            prop_assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn z_score_matches_known_95_percent_value() {
        let z = z_score_for_confidence(0.95);
        assert!((z - 1.959_963_984_540_05).abs() < 1e-8);
    }

    #[test]
    fn confidence_validation_rejects_bounds() {
        assert!(!validate_confidence_level(0.0));
        assert!(!validate_confidence_level(1.0));
        assert!(validate_confidence_level(0.95));
    }

    #[test]
    fn bootstrap_is_reproducible() {
        let influence = [1.0, -1.0, 2.0, -2.0];
        let inference = InferenceConfig::new(0.95);
        let bootstrap = BootstrapConfig {
            reps: 200,
            seed: 42,
        };
        let ci_a = multiplier_bootstrap_ci(1.0, &influence, inference, bootstrap);
        let ci_b = multiplier_bootstrap_ci(1.0, &influence, inference, bootstrap);
        assert_eq!(ci_a, ci_b);
    }

    #[rstest]
    fn quantile_unsorted_matches_sorted_interpolation_contract(
        #[values(10usize, 100, 999, 1_000, 1_001)] n: usize,
        #[values(0.025, 0.05, 0.5, 0.95, 0.975)] q: f64,
    ) {
        let values = (0..n)
            .map(|index| {
                let index_f = usize_to_f64(index);
                ((0.37 * index_f).sin() + (0.11 * index_f).cos())
                    .mul_add(0.5, usize_to_f64(index % 7) * 0.01)
            })
            .collect::<Vec<_>>();
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        let expected = quantile_sorted(&sorted, q);
        let mut unsorted = values;
        let actual = quantile_unsorted(&mut unsorted, q);
        assert_eq!(actual.to_bits(), expected.to_bits(), "n={n}, q={q}");
    }

    #[test]
    fn multiplier_bootstrap_ci_matches_sorted_reference() {
        let influence = (0..257usize)
            .map(|index| {
                let index_f = usize_to_f64(index);
                0.4f64.mul_add((0.13 * index_f).cos(), (0.07 * index_f).sin())
            })
            .collect::<Vec<_>>();
        let inference = InferenceConfig::new(0.95);
        for bootstrap in [
            BootstrapConfig { reps: 19, seed: 11 },
            BootstrapConfig {
                reps: 200,
                seed: 42,
            },
            BootstrapConfig {
                reps: 999,
                seed: 17_431,
            },
        ] {
            let expected =
                multiplier_bootstrap_ci_sorted_reference(0.7, &influence, inference, bootstrap);
            let actual = multiplier_bootstrap_ci(0.7, &influence, inference, bootstrap);
            assert_eq!(actual.0.to_bits(), expected.0.to_bits());
            assert_eq!(actual.1.to_bits(), expected.1.to_bits());
        }
    }

    #[test]
    fn multiplier_bootstrap_ci_does_not_depend_on_threads() {
        let influence = (0..5_000usize)
            .map(|index| usize_to_f64(index % 13) - 6.0)
            .collect::<Vec<_>>();
        let inference = InferenceConfig::new(0.95);
        let bootstrap = BootstrapConfig { reps: 300, seed: 5 };
        let on = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| multiplier_bootstrap_ci(0.2, &influence, inference, bootstrap))
        };
        let one = on(1);
        let four = on(4);
        assert_eq!(one.0.to_bits(), four.0.to_bits());
        assert_eq!(one.1.to_bits(), four.1.to_bits());
    }

    #[test]
    fn influence_standard_error_matches_reference() {
        let influence = [2.0, -1.0, 0.0, 1.0];
        let se = standard_error_from_influence(&influence);
        assert!((se - 0.559_016_994_374_947_5).abs() < 1e-12);
    }

    #[test]
    fn clustered_standard_error_uses_cluster_aggregation() {
        let influence = [1.0, 2.0, -1.0, -2.0];
        let labels = ["a", "a", "b", "b"];
        let se = clustered_standard_error_from_influence(&influence, &labels).unwrap();
        assert!((se - 1.5).abs() < 1e-12);
    }

    #[test]
    fn clustered_ci_rejects_misaligned_inputs() {
        let err = clustered_ci_from_influence(1.0, &[1.0, 2.0], &["a"], InferenceConfig::new(0.95))
            .unwrap_err();
        assert_eq!(
            err,
            InferenceError::MisalignedClusterLengths {
                influence_len: 2,
                cluster_len: 1,
            }
        );
    }

    #[test]
    fn covariance_matrix_matches_scalar_variance_on_diagonal() {
        let influence = vec![vec![2.0, -1.0, 0.0, 1.0], vec![1.0, 1.0, -1.0, -1.0]];
        let covariance = vcov::covariance_matrix_from_influence_matrix(&influence).unwrap();
        let se = standard_error_from_influence(&influence[0]);
        assert!((covariance[0][0] - se * se).abs() < 1e-12);
        assert_eq!(covariance.len(), 2);
        assert_eq!(covariance[0].len(), 2);
    }

    #[test]
    fn clustered_covariance_matrix_aggregates_by_cluster() {
        let influence = vec![vec![1.0, 2.0, -1.0, -2.0], vec![0.5, 0.5, -0.5, -0.5]];
        let labels = ["a", "a", "b", "b"];
        let covariance =
            vcov::clustered_covariance_matrix_from_influence_matrix(&influence, &labels).unwrap();
        let se = clustered_standard_error_from_influence(&influence[0], &labels).unwrap();
        assert!((covariance[0][0] - se * se).abs() < 1e-12);
        assert!(covariance[0][1].is_finite());
    }

    #[test]
    fn bitpacked_signs_are_statistically_equivalent_to_scalar_signs() {
        let influence = (0..251usize)
            .map(|index| {
                let index_f = usize_to_f64(index);
                0.3f64.mul_add((0.07 * index_f).cos(), (0.11 * index_f).sin())
            })
            .collect::<Vec<_>>();
        let reps = 20_000usize;
        let seed_scalar = 1_337u64;
        let seed_bitpacked = 7_331u64;

        let mut scalar_rng = Stream::seeded(seed_scalar);
        let scalar_draws = (0..reps)
            .map(|_| rademacher_perturbation_scalar(&influence, &mut scalar_rng))
            .collect::<Vec<_>>();
        let mut bitpacked_stream = epistat_rng::Stream::seeded(seed_bitpacked);
        let bitpacked_draws = (0..reps)
            .map(|_| epistat_rng::rademacher_sum(&influence, &mut bitpacked_stream))
            .collect::<Vec<_>>();

        let mean = |draws: &[f64]| draws.iter().sum::<f64>() / usize_to_f64(draws.len());
        let variance = |draws: &[f64], mean_val: f64| {
            draws
                .iter()
                .map(|value| {
                    let centered = *value - mean_val;
                    centered * centered
                })
                .sum::<f64>()
                / usize_to_f64(draws.len())
        };

        let scalar_mean = mean(&scalar_draws);
        let bitpacked_mean = mean(&bitpacked_draws);
        let scalar_var = variance(&scalar_draws, scalar_mean);
        let bitpacked_var = variance(&bitpacked_draws, bitpacked_mean);

        let mut scalar_sorted = scalar_draws;
        scalar_sorted.sort_by(f64::total_cmp);
        let mut bitpacked_sorted = bitpacked_draws;
        bitpacked_sorted.sort_by(f64::total_cmp);
        let scalar_q05 = quantile_sorted(&scalar_sorted, 0.05);
        let scalar_q50 = quantile_sorted(&scalar_sorted, 0.50);
        let scalar_q95 = quantile_sorted(&scalar_sorted, 0.95);
        let bitpacked_q05 = quantile_sorted(&bitpacked_sorted, 0.05);
        let bitpacked_q50 = quantile_sorted(&bitpacked_sorted, 0.50);
        let bitpacked_q95 = quantile_sorted(&bitpacked_sorted, 0.95);
        let quantile_tolerance = 0.1 * scalar_var.sqrt().max(1.0);

        let mean_diff = (scalar_mean - bitpacked_mean).abs();
        let mean_se = (scalar_var / usize_to_f64(reps) + bitpacked_var / usize_to_f64(reps)).sqrt();
        assert!(
            mean_diff <= 5.0f64.mul_add(mean_se, 0.05),
            "mean mismatch: scalar={scalar_mean}, bitpacked={bitpacked_mean}, se={mean_se}"
        );
        assert!(
            (scalar_var - bitpacked_var).abs() / scalar_var.max(1e-12) <= 0.08,
            "variance mismatch: scalar={scalar_var}, bitpacked={bitpacked_var}"
        );
        assert!(
            (scalar_q05 - bitpacked_q05).abs() <= quantile_tolerance,
            "q05 mismatch: scalar={scalar_q05}, bitpacked={bitpacked_q05}"
        );
        assert!(
            (scalar_q50 - bitpacked_q50).abs() <= quantile_tolerance,
            "q50 mismatch: scalar={scalar_q50}, bitpacked={bitpacked_q50}"
        );
        assert!(
            (scalar_q95 - bitpacked_q95).abs() <= quantile_tolerance,
            "q95 mismatch: scalar={scalar_q95}, bitpacked={bitpacked_q95}"
        );
    }
}
