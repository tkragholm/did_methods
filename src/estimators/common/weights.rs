//! Helpers for constructing and manipulating estimator weight structures.

use tracing::warn;

/// Build a sparse diagonal matrix from weights.
///
/// The returned matrix has shape `w.len() x w.len()` and contains `w[i]` on
/// the diagonal.
///
/// # Errors
/// Returns an error if sparse matrix construction fails. A warning is emitted
/// through `tracing`.
pub fn diag_sparse_from_vec(
    w: &[f64],
) -> Result<faer::sparse::SparseColMat<usize, f64>, &'static str> {
    let mut triplets = Vec::with_capacity(w.len());
    for (i, &v) in w.iter().enumerate() {
        triplets.push(faer::sparse::Triplet::new(i, i, v));
    }
    match faer::sparse::SparseColMat::try_new_from_triplets(w.len(), w.len(), &triplets) {
        Ok(mat) => Ok(mat),
        Err(err) => {
            warn!("diag_sparse_from_vec: failed to build sparse weights: {err}");
            Err("failed to build sparse diagonal weight matrix")
        }
    }
}
