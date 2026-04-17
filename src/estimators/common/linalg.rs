//! Small linear-algebra helpers shared by estimator implementations.

use dyn_stack::{MemBuffer, MemStack};
use faer::Side;
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::prelude::*;
use faer::{Conj, Mat, MatRef, Par, Spec};

/// Solve `(A + λI) X = B` for `X`, where `A` is expected to be square.
///
/// This helper adds a ridge term `lambda` to the diagonal in-place and solves
/// with LU decomposition.
///
/// # Notes
/// - `lambda` is applied uniformly to every diagonal entry.
#[must_use]
pub fn ridge_solve_lu(mut a: Mat<f64>, b: &Mat<f64>, lambda: f64) -> Mat<f64> {
    for i in 0..a.ncols() {
        *a.get_mut(i, i) += lambda;
    }
    a.partial_piv_lu().solve(b)
}

/// Solve a dense linear system `A x = b` using Cholesky when possible.
///
/// Fallback chain:
/// 1. Cholesky (`LLT`) on SPD matrices.
/// 2. Bunch-Kaufman (`LBLT`) for symmetric-indefinite systems.
/// 3. Partial-pivot LU if previous attempts are non-finite.
/// 4. Full-pivot LU when partial-pivot output is non-finite.
///
/// # Errors
/// Returns an error if:
/// - `a.len() != b.len() * b.len()` (shape mismatch), or
/// - all solve attempts produce non-finite values.
pub fn solve_spd_system(a: &[f64], b: &[f64]) -> Result<Vec<f64>, &'static str> {
    let n = b.len();
    validate_square_system_shape(a, n)?;
    let system = MatRef::from_row_major_slice(a, n, n);
    let rhs = MatRef::from_column_major_slice(b, n, 1);

    if let Ok(cholesky) = system.llt(Side::Lower) {
        let solution = cholesky.solve(&rhs);
        if let Ok(vec) = collect_solution_column(&solution) {
            return Ok(vec);
        }
    }

    let lblt_solution = system.lblt(Side::Lower).solve(&rhs);
    if let Ok(vec) = collect_solution_column(&lblt_solution) {
        return Ok(vec);
    }

    if let Ok(solution) = solve_general_system(a, b) {
        return Ok(solution);
    }

    solve_general_system_robust(a, b)
}

/// Solve a dense linear system `A X = B` where `A` is square and `B` may have
/// multiple right-hand sides.
///
/// The same fallback chain as [`solve_spd_system`] is applied.
///
/// # Errors
/// Returns an error if:
/// - `a` does not represent a square matrix matching `rhs.nrows()`, or
/// - all solve attempts produce non-finite values.
pub fn solve_spd_system_multi_rhs(a: &[f64], rhs: &Mat<f64>) -> Result<Mat<f64>, &'static str> {
    let n = rhs.nrows();
    validate_square_system_shape(a, n)?;
    let system = MatRef::from_row_major_slice(a, n, n);

    if let Ok(cholesky) = system.llt(Side::Lower) {
        let solution = cholesky.solve(rhs);
        if matrix_is_finite(&solution) {
            return Ok(solution);
        }
    }

    let lblt_solution = system.lblt(Side::Lower).solve(rhs);
    if matrix_is_finite(&lblt_solution) {
        return Ok(lblt_solution);
    }

    let solution = system.partial_piv_lu().solve(rhs);
    if matrix_is_finite(&solution) {
        return Ok(solution);
    }

    let robust_solution = system.full_piv_lu().solve(rhs);
    if matrix_is_finite(&robust_solution) {
        return Ok(robust_solution);
    }

    Err("dense solve produced a non-finite value")
}

/// Solve a dense linear system `A x = b` with partial-pivot LU.
///
/// # Errors
/// Returns an error if:
/// - `a.len() != b.len() * b.len()` (shape mismatch), or
/// - the resulting solution contains non-finite values.
pub fn solve_general_system(a: &[f64], b: &[f64]) -> Result<Vec<f64>, &'static str> {
    let n = b.len();
    validate_square_system_shape(a, n)?;
    let system = MatRef::from_row_major_slice(a, n, n);
    let rhs = MatRef::from_column_major_slice(b, n, 1);
    let solution = system.partial_piv_lu().solve(rhs);
    collect_solution_column(&solution)
}

/// Solve a dense linear system `A x = b` with robust full-pivot LU.
///
/// This explicit path is intended for rank-deficient or near-singular systems.
///
/// # Errors
/// Returns an error if:
/// - `a.len() != b.len() * b.len()` (shape mismatch), or
/// - the resulting solution contains non-finite values.
pub fn solve_general_system_robust(a: &[f64], b: &[f64]) -> Result<Vec<f64>, &'static str> {
    let n = b.len();
    validate_square_system_shape(a, n)?;
    let system = MatRef::from_row_major_slice(a, n, n);
    let rhs = MatRef::from_column_major_slice(b, n, 1);
    let solution = system.full_piv_lu().solve(rhs);
    collect_solution_column(&solution)
}

/// Solve a dense linear system `A x = b` from flat row-major inputs.
///
/// # Arguments
/// - `a`: Flattened `n x n` coefficient matrix in row-major order.
/// - `b`: Right-hand side vector with length `n`.
///
/// # Errors
/// Returns an error if:
/// - `a.len() != b.len() * b.len()` (shape mismatch), or
/// - the resulting solution contains non-finite values.
pub fn solve_dense_system(a: &[f64], b: &[f64]) -> Result<Vec<f64>, &'static str> {
    solve_general_system_robust(a, b)
}

/// Scratch-backed SPD solver for repeated single-RHS solves with fixed dimension.
pub struct SpdCholeskyScratch {
    dim: usize,
    factor: Mat<f64>,
    rhs: Mat<f64>,
    scratch: MemBuffer,
}

impl SpdCholeskyScratch {
    #[must_use]
    pub fn new(dim: usize) -> Self {
        let factor_mem =
            llt::factor::cholesky_in_place_scratch::<f64>(dim, Par::Seq, Spec::default());
        let solve_mem = llt::solve::solve_in_place_scratch::<f64>(dim, 1, Par::Seq);
        Self {
            dim,
            factor: Mat::zeros(dim, dim),
            rhs: Mat::zeros(dim, 1),
            scratch: MemBuffer::new(factor_mem.or(solve_mem)),
        }
    }

    /// Solve `A x = b` using in-place LLT with reusable scratch memory.
    ///
    /// # Errors
    /// Returns an error if dimensions mismatch, LLT fails, or the solution is non-finite.
    pub fn solve_single_rhs_into(
        &mut self,
        a: &[f64],
        b: &[f64],
        out: &mut [f64],
    ) -> Result<(), &'static str> {
        validate_square_system_shape(a, self.dim)?;
        if b.len() != self.dim || out.len() != self.dim {
            return Err("coefficient matrix shape does not match rhs length");
        }

        self.factor
            .as_mut()
            .copy_from(MatRef::from_row_major_slice(a, self.dim, self.dim));
        self.rhs
            .as_mut()
            .copy_from(MatRef::from_column_major_slice(b, self.dim, 1));

        let factor_stack = MemStack::new(&mut self.scratch);
        if llt::factor::cholesky_in_place(
            self.factor.as_mut(),
            LltRegularization::default(),
            Par::Seq,
            factor_stack,
            Spec::default(),
        )
        .is_err()
        {
            return Err("dense solve produced a non-finite value");
        }

        let solve_stack = MemStack::new(&mut self.scratch);
        llt::solve::solve_in_place_with_conj(
            self.factor.as_ref(),
            Conj::No,
            self.rhs.as_mut(),
            Par::Seq,
            solve_stack,
        );
        for (row, value) in out.iter_mut().enumerate().take(self.dim) {
            let solved = self.rhs[(row, 0)];
            if !solved.is_finite() {
                return Err("dense solve produced a non-finite value");
            }
            *value = solved;
        }
        Ok(())
    }
}

const fn validate_square_system_shape(a: &[f64], n: usize) -> Result<(), &'static str> {
    if a.len() != n * n {
        return Err("coefficient matrix shape does not match rhs length");
    }
    Ok(())
}

fn matrix_is_finite(matrix: &Mat<f64>) -> bool {
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            if !matrix[(row, col)].is_finite() {
                return false;
            }
        }
    }
    true
}

fn collect_solution_column(solution: &Mat<f64>) -> Result<Vec<f64>, &'static str> {
    if !matrix_is_finite(solution) {
        return Err("dense solve produced a non-finite value");
    }
    let mut out = Vec::with_capacity(solution.nrows());
    for row in 0..solution.nrows() {
        out.push(solution[(row, 0)]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_spd_system_handles_positive_definite_system() {
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let b = vec![1.0, 2.0];
        let solution = solve_spd_system(&a, &b).expect("solve");
        assert!((solution[0] - 0.090_909_090_909_090_91).abs() < 1e-12);
        assert!((solution[1] - 0.636_363_636_363_636_4).abs() < 1e-12);
    }

    #[test]
    fn solve_spd_system_falls_back_to_lu_for_indefinite_matrix() {
        let a = vec![0.0, 1.0, 1.0, 0.0];
        let b = vec![1.0, 2.0];
        let solution = solve_spd_system(&a, &b).expect("solve");
        assert!((solution[0] - 2.0).abs() < 1e-12);
        assert!((solution[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn solve_spd_system_rejects_singular_matrix() {
        let a = vec![1.0, 2.0, 2.0, 4.0];
        let b = vec![1.0, 2.0];
        let err = solve_spd_system(&a, &b).expect_err("singular matrix must fail");
        assert!(err.contains("non-finite"));
    }

    #[test]
    fn solve_dense_system_preserves_robust_path_for_singular_case() {
        let singular = vec![1.0, 2.0, 2.0, 4.0];
        let b = vec![1.0, 2.0];
        let err = solve_dense_system(&singular, &b).expect_err("must fail");
        assert!(err.contains("non-finite"));
    }
}
