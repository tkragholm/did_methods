use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::HashMap;
use std::hash::Hash;

use crate::estimators::common::linalg::solve_dense_system;
use crate::util::usize_to_f64;

const VCOV_BLOCK_SIZE: usize = 32;

/// Compute the covariance matrix from a set of influence function vectors.
///
/// Returns the centered covariance estimate
/// `Σ = (1/n²) * Σ (ψ_i - ψ̄) (ψ_i - ψ̄)'`.
///
/// # Errors
/// Returns an error if:
/// - Influence vectors are empty or zero-length.
/// - Influence vectors have unequal lengths.
/// - Influence vectors contain non-finite values.
pub fn covariance_matrix_from_influence_matrix(
    influence_functions: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    let (k, n, n_f) = validate_influence_matrix(influence_functions, "covariance matrix")?;
    let means = influence_functions
        .iter()
        .map(|influence| influence.iter().sum::<f64>() / n_f)
        .collect::<Vec<_>>();
    let mut covariance = vec![vec![0.0; k]; k];

    for row_start in (0..k).step_by(VCOV_BLOCK_SIZE) {
        let row_end = (row_start + VCOV_BLOCK_SIZE).min(k);
        for col_start in (row_start..k).step_by(VCOV_BLOCK_SIZE) {
            let col_end = (col_start + VCOV_BLOCK_SIZE).min(k);
            let row_block_len = row_end - row_start;
            let col_block_len = col_end - col_start;
            let mut row_centered = vec![0.0; row_block_len * n];
            let mut col_centered = vec![0.0; col_block_len * n];

            for sample_idx in 0..n {
                for (local_row, estimate_idx) in (row_start..row_end).enumerate() {
                    row_centered[local_row * n + sample_idx] =
                        influence_functions[estimate_idx][sample_idx] - means[estimate_idx];
                }
                for (local_col, estimate_idx) in (col_start..col_end).enumerate() {
                    col_centered[local_col * n + sample_idx] =
                        influence_functions[estimate_idx][sample_idx] - means[estimate_idx];
                }
            }

            let left = MatRef::from_row_major_slice(&row_centered, row_block_len, n);
            let right = MatRef::from_row_major_slice(&col_centered, col_block_len, n);
            let mut block = Mat::<f64>::zeros(row_block_len, col_block_len);
            let scale = 1.0 / (n_f * n_f);
            matmul(
                block.as_mut(),
                Accum::Replace,
                left,
                right.transpose(),
                scale,
                Par::Seq,
            );
            for local_row in 0..row_block_len {
                let global_row = row_start + local_row;
                for local_col in 0..col_block_len {
                    let global_col = col_start + local_col;
                    let value = block[(local_row, local_col)];
                    covariance[global_row][global_col] = value;
                    covariance[global_col][global_row] = value;
                }
            }
        }
    }

    Ok(covariance)
}

/// Compute the cluster-robust covariance matrix from a set of influence function vectors.
///
/// # Errors
/// Returns an error if:
/// - Influence vectors are empty or zero-length.
/// - Cluster labels length does not match influence vector length.
pub fn clustered_covariance_matrix_from_influence_matrix<L: Eq + Hash>(
    influence_functions: &[Vec<f64>],
    cluster_labels: &[L],
) -> Result<Vec<Vec<f64>>, String> {
    if influence_functions.is_empty() {
        return Err("clustered covariance matrix requires non-empty influence vectors".to_string());
    }
    let n = influence_functions[0].len();
    if n == 0 {
        return Err("clustered covariance matrix requires non-empty influence vectors".to_string());
    }
    if cluster_labels.len() != n {
        return Err(format!(
            "clustered covariance matrix requires aligned influence/cluster lengths ({} vs {})",
            n,
            cluster_labels.len()
        ));
    }

    for influence in influence_functions {
        if influence.len() != n {
            return Err(
                "clustered covariance matrix requires equal-length influence vectors".to_string(),
            );
        }
        if influence.iter().any(|value| !value.is_finite()) {
            return Err("clustered covariance matrix requires finite influence values".to_string());
        }
    }

    let mut group_assignments = Vec::with_capacity(n);
    let mut label_to_group = HashMap::<&L, usize>::new();
    for label in cluster_labels {
        let next_group = label_to_group.len();
        let group = *label_to_group.entry(label).or_insert(next_group);
        group_assignments.push(group);
    }
    clustered_covariance_matrix_from_influence_matrix_index(
        influence_functions,
        &crate::inference::ClusterIndex {
            group_assignments,
            n_groups: label_to_group.len(),
        },
    )
}

/// Compute the cluster-robust covariance matrix from a set of influence function vectors using a pre-computed index.
///
/// # Errors
/// Returns an error if:
/// - Influence vectors are empty or zero-length.
/// - Cluster index length does not match influence vector length.
pub fn clustered_covariance_matrix_from_influence_matrix_index(
    influence_functions: &[Vec<f64>],
    cluster_index: &crate::inference::ClusterIndex,
) -> Result<Vec<Vec<f64>>, String> {
    let (k, n, n_f) =
        validate_influence_matrix(influence_functions, "clustered covariance matrix")?;
    if cluster_index.group_assignments.len() != n {
        return Err(format!(
            "clustered covariance matrix requires aligned influence/cluster lengths ({} vs {})",
            n,
            cluster_index.group_assignments.len()
        ));
    }

    let g = cluster_index.n_groups;
    let correction = if g >= 2 {
        let g_f = usize_to_f64(g);
        g_f / (g_f - 1.0)
    } else {
        1.0
    };
    let scale = correction / (n_f * n_f);
    let mut covariance = vec![vec![0.0; k]; k];

    for row_start in (0..k).step_by(VCOV_BLOCK_SIZE) {
        let row_end = (row_start + VCOV_BLOCK_SIZE).min(k);
        for col_start in (row_start..k).step_by(VCOV_BLOCK_SIZE) {
            let col_end = (col_start + VCOV_BLOCK_SIZE).min(k);
            let row_block_len = row_end - row_start;
            let col_block_len = col_end - col_start;
            let mut left_sums = vec![0.0; g * row_block_len];
            let mut right_sums = vec![0.0; g * col_block_len];

            for (sample_idx, group_idx) in cluster_index
                .group_assignments
                .iter()
                .copied()
                .enumerate()
                .take(n)
            {
                let left_base = group_idx * row_block_len;
                let right_base = group_idx * col_block_len;
                for (local_row, estimate_idx) in (row_start..row_end).enumerate() {
                    left_sums[left_base + local_row] +=
                        influence_functions[estimate_idx][sample_idx];
                }
                for (local_col, estimate_idx) in (col_start..col_end).enumerate() {
                    right_sums[right_base + local_col] +=
                        influence_functions[estimate_idx][sample_idx];
                }
            }

            let left = MatRef::from_row_major_slice(&left_sums, g, row_block_len);
            let right = MatRef::from_row_major_slice(&right_sums, g, col_block_len);
            let mut block = Mat::<f64>::zeros(row_block_len, col_block_len);
            matmul(
                block.as_mut(),
                Accum::Replace,
                left.transpose(),
                right,
                scale,
                Par::Seq,
            );
            for local_row in 0..row_block_len {
                let global_row = row_start + local_row;
                for local_col in 0..col_block_len {
                    let global_col = col_start + local_col;
                    let value = block[(local_row, local_col)];
                    covariance[global_row][global_col] = value;
                    covariance[global_col][global_row] = value;
                }
            }
        }
    }

    Ok(covariance)
}

fn validate_influence_matrix(
    influence_functions: &[Vec<f64>],
    context: &str,
) -> Result<(usize, usize, f64), String> {
    if influence_functions.is_empty() {
        return Err(format!("{context} requires non-empty influence vectors"));
    }
    let n = influence_functions[0].len();
    if n == 0 {
        return Err(format!("{context} requires non-empty influence vectors"));
    }

    for influence in influence_functions {
        if influence.len() != n {
            return Err(format!("{context} requires equal-length influence vectors"));
        }
        if influence.iter().any(|value| !value.is_finite()) {
            return Err(format!("{context} requires finite influence values"));
        }
    }

    Ok((influence_functions.len(), n, usize_to_f64(n)))
}
///
/// H0: β = 0
/// Statistic: W = β' Σ⁻¹ β
///
/// # Errors
/// Returns an error if:
/// - Estimates are empty.
/// - The covariance matrix is singular.
pub fn joint_wald_test(
    estimates: &[f64],
    covariance_matrix: &[Vec<f64>],
) -> Result<(f64, f64, usize), String> {
    let k = estimates.len();
    if k == 0 {
        return Err("Wald test requires at least one estimate".to_string());
    }

    // Flatten covariance matrix for solve_dense_system
    let mut sigma_flat = Vec::with_capacity(k * k);
    for row in covariance_matrix {
        sigma_flat.extend_from_slice(row);
    }

    let x = solve_dense_system(&sigma_flat, estimates)?;

    let mut statistic = 0.0;
    for i in 0..k {
        statistic = estimates[i].mul_add(x[i], statistic);
    }

    let chi2 = ChiSquared::new(usize_to_f64(k)).map_err(|e| e.to_string())?;
    let p_value = 1.0 - chi2.cdf(statistic);

    Ok((statistic, p_value, k))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manual_covariance(influence_functions: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let k = influence_functions.len();
        let n = influence_functions[0].len();
        let n_f = usize_to_f64(n);
        let means = influence_functions
            .iter()
            .map(|influence| influence.iter().sum::<f64>() / n_f)
            .collect::<Vec<_>>();
        let mut covariance = vec![vec![0.0; k]; k];
        for row in 0..k {
            for col in 0..k {
                let mut sum = 0.0;
                for (&row_value, &col_value) in influence_functions[row]
                    .iter()
                    .zip(influence_functions[col].iter())
                    .take(n)
                {
                    sum = (row_value - means[row]).mul_add(col_value - means[col], sum);
                }
                covariance[row][col] = sum / (n_f * n_f);
            }
        }
        covariance
    }

    fn manual_clustered_covariance(
        influence_functions: &[Vec<f64>],
        group_assignments: &[usize],
        n_groups: usize,
    ) -> Vec<Vec<f64>> {
        let k = influence_functions.len();
        let n = influence_functions[0].len();
        let n_f = usize_to_f64(n);
        let correction = if n_groups >= 2 {
            let g_f = usize_to_f64(n_groups);
            g_f / (g_f - 1.0)
        } else {
            1.0
        };
        let scale = correction / (n_f * n_f);
        let mut covariance = vec![vec![0.0; k]; k];
        for row in 0..k {
            for col in 0..k {
                let mut value = 0.0;
                for group in 0..n_groups {
                    let mut row_sum = 0.0;
                    let mut col_sum = 0.0;
                    for sample in 0..n {
                        if group_assignments[sample] == group {
                            row_sum += influence_functions[row][sample];
                            col_sum += influence_functions[col][sample];
                        }
                    }
                    value = row_sum.mul_add(col_sum, value);
                }
                covariance[row][col] = value * scale;
            }
        }
        covariance
    }

    #[test]
    fn covariance_matrix_matches_manual_reference() {
        let influence = vec![
            vec![2.0, -1.0, 0.0, 1.0, 3.0],
            vec![1.0, 1.0, -1.0, -1.0, 0.5],
            vec![0.5, -0.25, 0.75, -1.25, 2.0],
        ];
        let actual = covariance_matrix_from_influence_matrix(&influence).unwrap();
        let expected = manual_covariance(&influence);
        for row in 0..expected.len() {
            for col in 0..expected.len() {
                assert!((actual[row][col] - expected[row][col]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn clustered_covariance_matrix_matches_manual_reference() {
        let influence = vec![
            vec![1.0, 2.0, -1.0, -2.0, 0.5, 1.5],
            vec![0.5, 0.5, -0.5, -0.5, 1.0, 1.0],
            vec![2.0, -1.0, 3.0, -2.0, 0.0, 0.5],
        ];
        let cluster_index = crate::inference::ClusterIndex {
            group_assignments: vec![0, 0, 1, 1, 2, 2],
            n_groups: 3,
        };
        let actual =
            clustered_covariance_matrix_from_influence_matrix_index(&influence, &cluster_index)
                .unwrap();
        let expected = manual_clustered_covariance(
            &influence,
            &cluster_index.group_assignments,
            cluster_index.n_groups,
        );
        for row in 0..expected.len() {
            for col in 0..expected.len() {
                assert!((actual[row][col] - expected[row][col]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn wald_test_simple_case() {
        let estimates = [2.0];
        let vcov = vec![vec![1.0]];
        let (stat, p, df) = joint_wald_test(&estimates, &vcov).unwrap();
        assert_eq!(df, 1);
        assert!((stat - 4.0).abs() < 1e-10);
        // For Chi2(1), stat=4 is roughly 2 standard deviations, so p should be small (~0.0455)
        assert!(p < 0.05 && p > 0.045);
    }
}
