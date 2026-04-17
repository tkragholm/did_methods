use serde::{Deserialize, Serialize};

use crate::inference::vcov::{
    clustered_covariance_matrix_from_influence_matrix, covariance_matrix_from_influence_matrix,
};

/// Canonical long-form influence row keyed by a stable estimate identifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceInfluenceRow {
    pub estimate_id: String,
    pub unit_id: String,
    pub influence_value: f64,
}

/// Canonical pairwise covariance row keyed by stable estimate identifiers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceCovarianceRow {
    pub estimate_id_left: String,
    pub estimate_id_right: String,
    pub covariance: f64,
}

/// Expand aligned influence vectors into a stable long-form export surface.
///
/// # Errors
/// Returns an error if estimate ids, unit ids, and influence vectors are not aligned.
pub fn build_inference_influence_rows(
    estimate_ids: &[String],
    unit_ids: &[String],
    influence_functions: &[Vec<f64>],
) -> Result<Vec<InferenceInfluenceRow>, String> {
    if estimate_ids.len() != influence_functions.len() {
        return Err(format!(
            "estimate ids and influence vectors must align ({} vs {})",
            estimate_ids.len(),
            influence_functions.len()
        ));
    }
    if influence_functions
        .iter()
        .any(|values| values.len() != unit_ids.len())
    {
        return Err("all influence vectors must match the unit id length".to_string());
    }
    let mut rows = Vec::new();
    for (estimate_id, influence_values) in estimate_ids.iter().zip(influence_functions) {
        rows.reserve(influence_values.len());
        for (unit_id, influence_value) in unit_ids.iter().zip(influence_values) {
            rows.push(InferenceInfluenceRow {
                estimate_id: estimate_id.clone(),
                unit_id: unit_id.clone(),
                influence_value: *influence_value,
            });
        }
    }
    Ok(rows)
}

/// Build pairwise covariance rows from aligned influence vectors.
///
/// # Errors
/// Returns an error if estimate ids and influence vectors are not aligned or
/// covariance computation fails.
pub fn build_inference_covariance_rows(
    estimate_ids: &[String],
    influence_functions: &[Vec<f64>],
) -> Result<Vec<InferenceCovarianceRow>, String> {
    let covariance = covariance_matrix_from_estimate_influence(estimate_ids, influence_functions)?;
    Ok(flatten_covariance_rows(estimate_ids, &covariance))
}

/// Build cluster-robust pairwise covariance rows from aligned influence vectors.
///
/// # Errors
/// Returns an error if estimate ids and influence vectors are not aligned or
/// covariance computation fails.
pub fn build_clustered_inference_covariance_rows(
    estimate_ids: &[String],
    influence_functions: &[Vec<f64>],
    cluster_labels: &[String],
) -> Result<Vec<InferenceCovarianceRow>, String> {
    if estimate_ids.len() != influence_functions.len() {
        return Err(format!(
            "estimate ids and influence vectors must align ({} vs {})",
            estimate_ids.len(),
            influence_functions.len()
        ));
    }
    let covariance =
        clustered_covariance_matrix_from_influence_matrix(influence_functions, cluster_labels)?;
    Ok(flatten_covariance_rows(estimate_ids, &covariance))
}

fn covariance_matrix_from_estimate_influence(
    estimate_ids: &[String],
    influence_functions: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    if estimate_ids.len() != influence_functions.len() {
        return Err(format!(
            "estimate ids and influence vectors must align ({} vs {})",
            estimate_ids.len(),
            influence_functions.len()
        ));
    }
    covariance_matrix_from_influence_matrix(influence_functions)
}

fn flatten_covariance_rows(
    estimate_ids: &[String],
    covariance: &[Vec<f64>],
) -> Vec<InferenceCovarianceRow> {
    let mut rows = Vec::new();
    for (left_idx, left_id) in estimate_ids.iter().enumerate() {
        rows.reserve(estimate_ids.len().saturating_sub(left_idx));
        for (right_idx, right_id) in estimate_ids.iter().enumerate().skip(left_idx) {
            rows.push(InferenceCovarianceRow {
                estimate_id_left: left_id.clone(),
                estimate_id_right: right_id.clone(),
                covariance: covariance[left_idx][right_idx],
            });
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::{
        build_clustered_inference_covariance_rows, build_inference_covariance_rows,
        build_inference_influence_rows,
    };

    #[test]
    fn builds_long_form_influence_rows() {
        let estimate_ids = vec!["e1".to_string(), "e2".to_string()];
        let unit_ids = vec!["u1".to_string(), "u2".to_string()];
        let influence_functions = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let rows =
            build_inference_influence_rows(&estimate_ids, &unit_ids, &influence_functions).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].estimate_id, "e1");
        assert_eq!(rows[0].unit_id, "u1");
        assert_eq!(rows[3].estimate_id, "e2");
        assert_eq!(rows[3].unit_id, "u2");
        assert!((rows[3].influence_value - 4.0).abs() < 1e-12);
    }

    #[test]
    fn builds_pairwise_covariance_rows() {
        let estimate_ids = vec!["e1".to_string(), "e2".to_string()];
        let influence_functions = vec![vec![1.0, -1.0], vec![2.0, -2.0]];
        let rows = build_inference_covariance_rows(&estimate_ids, &influence_functions).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].estimate_id_left, "e1");
        assert_eq!(rows[0].estimate_id_right, "e1");
        assert_eq!(rows[2].estimate_id_left, "e2");
        assert_eq!(rows[2].estimate_id_right, "e2");
    }

    #[test]
    fn builds_clustered_covariance_rows() {
        let estimate_ids = vec!["e1".to_string(), "e2".to_string()];
        let influence_functions = vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]];
        let cluster_labels = vec!["a".to_string(), "a".to_string(), "b".to_string()];
        let rows = build_clustered_inference_covariance_rows(
            &estimate_ids,
            &influence_functions,
            &cluster_labels,
        )
        .unwrap();
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|row| row.covariance.is_finite()));
    }
}
