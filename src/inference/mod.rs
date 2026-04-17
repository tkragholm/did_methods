//! Shared statistical inference utilities (critical values, IF-based SE, bootstrap CI).

use std::collections::HashMap;
use std::hash::Hash;

use rayon::prelude::*;
use thiserror::Error;

pub mod export;
pub mod sensitivity;
pub mod vcov;

use crate::types::{BootstrapConfig, InferenceConfig};
use crate::util::usize_to_f64;

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
    #[error("invalid confidence level for clustered CI: {0}")]
    InvalidConfidenceLevel(f64),
}

#[must_use]
pub fn validate_confidence_level(confidence_level: f64) -> bool {
    confidence_level.is_finite() && confidence_level > 0.0 && confidence_level < 1.0
}

#[must_use]
pub fn z_score_for_confidence(confidence_level: f64) -> f64 {
    // Two-sided normal critical value for arbitrary confidence levels in (0, 1).
    let p = 0.5f64.mul_add(confidence_level, 0.5);
    inverse_standard_normal_cdf(p)
}

#[must_use]
pub fn standard_error_from_influence(influence_function: &[f64]) -> f64 {
    let n_f = usize_to_f64(influence_function.len());
    let mean_if = influence_function.iter().sum::<f64>() / n_f;
    let centered_mean_square = influence_function
        .iter()
        .map(|value| {
            let centered = value - mean_if;
            centered * centered
        })
        .sum::<f64>()
        / n_f;
    centered_mean_square.sqrt() / n_f.sqrt()
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
    if influence_function.is_empty() {
        return Err(InferenceError::EmptyInfluenceFunction);
    }
    if influence_function.len() != cluster_index.group_assignments.len() {
        return Err(InferenceError::MisalignedClusterLengths {
            influence_len: influence_function.len(),
            cluster_len: cluster_index.group_assignments.len(),
        });
    }

    let n_f = usize_to_f64(influence_function.len());
    let mut cluster_sums = vec![0.0; cluster_index.n_groups];

    for (psi, &group_idx) in influence_function
        .iter()
        .zip(&cluster_index.group_assignments)
    {
        cluster_sums[group_idx] += *psi;
    }

    let variance = if cluster_index.n_groups >= 2 {
        let g_f = usize_to_f64(cluster_index.n_groups);
        let correction = g_f / (g_f - 1.0);
        let sum_sq = cluster_sums.iter().map(|&val| val * val).sum::<f64>();
        let scaled = correction * sum_sq;
        scaled / (n_f * n_f)
    } else {
        influence_function
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            / (n_f * n_f)
    };

    if !variance.is_finite() {
        return Err(InferenceError::NonFiniteClusteredVariance);
    }
    Ok(variance.max(0.0))
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
    let margin = z_score_for_confidence(inference.confidence_level) * se;
    Ok((se, estimate - margin, estimate + margin))
}

/// Computes the clustered variance from influence function values.
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
    if influence_function.is_empty() {
        return Err(InferenceError::EmptyInfluenceFunction);
    }
    if influence_function.len() != cluster_labels.len() {
        return Err(InferenceError::MisalignedClusterLengths {
            influence_len: influence_function.len(),
            cluster_len: cluster_labels.len(),
        });
    }

    let n_f = usize_to_f64(influence_function.len());
    let mut cluster_sum = HashMap::<&L, f64>::new();
    for (psi, label) in influence_function.iter().zip(cluster_labels) {
        let entry = cluster_sum.entry(label).or_insert(0.0);
        *entry += *psi;
    }

    let variance = if cluster_sum.len() >= 2 {
        let g_f = usize_to_f64(cluster_sum.len());
        let correction = g_f / (g_f - 1.0);
        let sum_sq = cluster_sum.values().map(|value| value * value).sum::<f64>();
        let scaled = correction * sum_sq;
        scaled / (n_f * n_f)
    } else {
        influence_function
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            / (n_f * n_f)
    };

    if !variance.is_finite() {
        return Err(InferenceError::NonFiniteClusteredVariance);
    }
    Ok(variance.max(0.0))
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
    let margin = z_score_for_confidence(inference.confidence_level) * se;
    Ok((se, estimate - margin, estimate + margin))
}

#[must_use]
pub fn multiplier_bootstrap_ci(
    estimate: f64,
    influence_function: &[f64],
    inference: InferenceConfig,
    bootstrap: BootstrapConfig,
) -> (f64, f64) {
    use rand::{SeedableRng, rngs::StdRng};

    let n_f = usize_to_f64(influence_function.len());
    let mut draws = if should_parallelize_bootstrap(bootstrap.reps, influence_function.len()) {
        (0..bootstrap.reps)
            .into_par_iter()
            .map(|draw_idx| {
                let mut rng =
                    StdRng::seed_from_u64(bootstrap_seed_for_draw(bootstrap.seed, draw_idx));
                let perturbation = rademacher_perturbation_bitpacked(influence_function, &mut rng);
                estimate + perturbation / n_f
            })
            .collect::<Vec<_>>()
    } else {
        let mut rng = StdRng::seed_from_u64(bootstrap.seed);
        let mut draws = Vec::with_capacity(bootstrap.reps);
        for _ in 0..bootstrap.reps {
            let perturbation = rademacher_perturbation_bitpacked(influence_function, &mut rng);
            draws.push(estimate + perturbation / n_f);
        }
        draws
    };

    let alpha = (1.0 - inference.confidence_level).clamp(0.0, 1.0);
    let lower = quantile_unsorted(&mut draws, alpha / 2.0);
    let upper = quantile_unsorted(&mut draws, 1.0 - alpha / 2.0);
    (lower, upper)
}

fn should_parallelize_bootstrap(reps: usize, influence_len: usize) -> bool {
    reps >= 128 && influence_len >= 4_096 && rayon::current_num_threads() > 1
}

fn bootstrap_seed_for_draw(base_seed: u64, draw_idx: usize) -> u64 {
    let idx = u64::try_from(draw_idx).unwrap_or(u64::MAX);
    splitmix64(base_seed ^ idx.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
fn rademacher_perturbation_scalar(influence_function: &[f64], rng: &mut rand::rngs::StdRng) -> f64 {
    use rand::RngExt;

    influence_function.iter().fold(0.0, |acc, value| {
        let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        acc + sign * value
    })
}

fn rademacher_perturbation_bitpacked(
    influence_function: &[f64],
    rng: &mut rand::rngs::StdRng,
) -> f64 {
    use rand::RngExt;

    let mut perturbation = 0.0;
    let mut chunk_start = 0usize;
    while chunk_start < influence_function.len() {
        let mut bits = rng.random::<u64>();
        let chunk_end = (chunk_start + 64).min(influence_function.len());
        for value in &influence_function[chunk_start..chunk_end] {
            let sign = if (bits & 1) == 1 { 1.0 } else { -1.0 };
            perturbation += sign * *value;
            bits >>= 1;
        }
        chunk_start += 64;
    }
    perturbation
}

#[must_use]
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let clamped_q = q.clamp(0.0, 1.0);
    let last = sorted.len() - 1;
    let rank = clamped_q * usize_to_f64(last);
    let lower = float_floor_to_usize(rank, last);
    let upper = float_ceil_to_usize(rank, last);
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = rank - usize_to_f64(lower);
        sorted[lower].mul_add(1.0 - fraction, sorted[upper] * fraction)
    }
}

#[must_use]
pub fn quantile_unsorted(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return values[0];
    }
    let clamped_q = q.clamp(0.0, 1.0);
    let last = values.len() - 1;
    let rank = clamped_q * usize_to_f64(last);
    let lower = float_floor_to_usize(rank, last);
    let upper = float_ceil_to_usize(rank, last);

    values.select_nth_unstable_by(lower, f64::total_cmp);
    let lower_value = values[lower];
    if lower == upper {
        return lower_value;
    }

    let right = &mut values[lower + 1..];
    let upper_in_right = upper - lower - 1;
    right.select_nth_unstable_by(upper_in_right, f64::total_cmp);
    let upper_value = right[upper_in_right];
    let fraction = rank - usize_to_f64(lower);
    lower_value.mul_add(1.0 - fraction, upper_value * fraction)
}

#[inline]
const fn poly5(x: f64, c0: f64, c1: f64, c2: f64, c3: f64, c4: f64, c5: f64) -> f64 {
    ((((c0.mul_add(x, c1)).mul_add(x, c2)).mul_add(x, c3)).mul_add(x, c4)).mul_add(x, c5)
}

#[inline]
const fn poly4(x: f64, c0: f64, c1: f64, c2: f64, c3: f64, c4: f64) -> f64 {
    (((c0.mul_add(x, c1)).mul_add(x, c2)).mul_add(x, c3)).mul_add(x, c4)
}

#[inline]
#[must_use]
pub fn inverse_standard_normal_cdf(p: f64) -> f64 {
    const A1: f64 = -3.969_683_028_665_376e1;
    const A2: f64 = 2.209_460_984_245_205e2;
    const A3: f64 = -2.759_285_104_469_687e2;
    const A4: f64 = 1.383_577_518_672_69e2;
    const A5: f64 = -3.066_479_806_614_716e1;
    const A6: f64 = 2.506_628_277_459_239;

    const B1: f64 = -5.447_609_879_822_406e1;
    const B2: f64 = 1.615_858_368_580_409e2;
    const B3: f64 = -1.556_989_798_598_866e2;
    const B4: f64 = 6.680_131_188_771_972e1;
    const B5: f64 = -1.328_068_155_288_572e1;

    const C1: f64 = -7.784_894_002_430_293e-3;
    const C2: f64 = -3.223_964_580_411_365e-1;
    const C3: f64 = -2.400_758_277_161_838;
    const C4: f64 = -2.549_732_539_343_734;
    const C5: f64 = 4.374_664_141_464_968;
    const C6: f64 = 2.938_163_982_698_783;

    const D1: f64 = 7.784_695_709_041_462e-3;
    const D2: f64 = 3.224_671_290_700_398e-1;
    const D3: f64 = 2.445_134_137_142_996;
    const D4: f64 = 3.754_408_661_907_416;

    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        let num = poly5(q, C1, C2, C3, C4, C5, C6);
        let den = poly4(q, D1, D2, D3, D4, 1.0);
        num / den
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        let num = poly5(r, A1, A2, A3, A4, A5, A6);
        let den = poly5(r, B1, B2, B3, B4, B5, 1.0);
        q * (num / den)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let num = poly5(q, C1, C2, C3, C4, C5, C6);
        let den = poly4(q, D1, D2, D3, D4, 1.0);
        -(num / den)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn float_floor_to_usize(value: f64, max: usize) -> usize {
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    let floored = value.floor();
    if floored >= usize_to_f64(max) {
        max
    } else {
        floored as usize
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn float_ceil_to_usize(value: f64, max: usize) -> usize {
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    let ceiled = value.ceil();
    if ceiled >= usize_to_f64(max) {
        max
    } else {
        ceiled as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    fn multiplier_bootstrap_ci_sorted_reference(
        estimate: f64,
        influence_function: &[f64],
        inference: InferenceConfig,
        bootstrap: BootstrapConfig,
    ) -> (f64, f64) {
        let n_f = usize_to_f64(influence_function.len());
        let mut rng = StdRng::seed_from_u64(bootstrap.seed);
        let mut draws = Vec::with_capacity(bootstrap.reps);

        for _ in 0..bootstrap.reps {
            let perturbation = rademacher_perturbation_bitpacked(influence_function, &mut rng);
            draws.push(estimate + perturbation / n_f);
        }
        draws.sort_by(f64::total_cmp);

        let alpha = (1.0 - inference.confidence_level).clamp(0.0, 1.0);
        let lower = quantile_sorted(&draws, alpha / 2.0);
        let upper = quantile_sorted(&draws, 1.0 - alpha / 2.0);
        (lower, upper)
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
    fn float_floor_to_usize_handles_edge_cases() {
        let max = 10;
        let larger_than_u32_max = f64::from(u32::MAX) + 1024.0;
        assert_eq!(float_floor_to_usize(f64::NAN, max), 0);
        assert_eq!(float_floor_to_usize(-1.0, max), 0);
        assert_eq!(float_floor_to_usize(f64::INFINITY, max), 0);
        assert_eq!(float_floor_to_usize(usize_to_f64(max), max), max);
        assert_eq!(float_floor_to_usize(usize_to_f64(max) - 0.5, max), max - 1);
        assert_eq!(float_floor_to_usize(0.0, max), 0);
        assert_eq!(float_floor_to_usize(0.1, max), 0);
        assert_eq!(float_floor_to_usize(larger_than_u32_max, max), max);
    }

    #[test]
    fn float_ceil_to_usize_handles_edge_cases() {
        let max = 10;
        let larger_than_u32_max = f64::from(u32::MAX) + 1024.0;
        assert_eq!(float_ceil_to_usize(f64::NAN, max), 0);
        assert_eq!(float_ceil_to_usize(-1.0, max), 0);
        assert_eq!(float_ceil_to_usize(f64::INFINITY, max), 0);
        assert_eq!(float_ceil_to_usize(usize_to_f64(max), max), max);
        assert_eq!(float_ceil_to_usize(usize_to_f64(max) - 0.5, max), max);
        assert_eq!(float_ceil_to_usize(0.0, max), 0);
        assert_eq!(float_ceil_to_usize(0.1, max), 1);
        assert_eq!(float_ceil_to_usize(larger_than_u32_max, max), max);
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

        let mut scalar_rng = StdRng::seed_from_u64(seed_scalar);
        let scalar_draws = (0..reps)
            .map(|_| rademacher_perturbation_scalar(&influence, &mut scalar_rng))
            .collect::<Vec<_>>();
        let mut bitpacked_rng = StdRng::seed_from_u64(seed_bitpacked);
        let bitpacked_draws = (0..reps)
            .map(|_| rademacher_perturbation_bitpacked(&influence, &mut bitpacked_rng))
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
