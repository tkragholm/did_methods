use clarabel::algebra::CscMatrix;
use dyn_stack::{MemBuffer, MemStack};
use epistat_stats::normal;
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::linalg::matmul::matmul;
use faer::prelude::Solve;
use faer::{Accum, Mat, MatMut, MatRef, Par, Side, Spec};

use crate::util::usize_to_f64;

pub(super) fn linear_grid(lb: f64, ub: f64, points: usize) -> Vec<f64> {
    if points <= 1 {
        return vec![lb];
    }
    let denom = usize_to_f64(points.saturating_sub(1));
    (0..points)
        .map(|idx| (ub - lb).mul_add(usize_to_f64(idx) / denom, lb))
        .collect()
}

pub(super) fn select_rows(matrix: &[Vec<f64>], rows: &[usize]) -> Vec<Vec<f64>> {
    rows.iter().map(|idx| matrix[*idx].clone()).collect()
}

pub(super) fn subset_square_matrix(matrix: &[Vec<f64>], rows: &[usize]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row_idx| {
            rows.iter()
                .map(|col_idx| matrix[*row_idx][*col_idx])
                .collect()
        })
        .collect()
}

pub(super) fn diag_sqrt(matrix: &[Vec<f64>]) -> Vec<f64> {
    (0..matrix.len())
        .map(|idx| matrix[idx][idx].max(0.0).sqrt())
        .collect()
}

pub(super) fn drop_zero_columns(matrix: &[Vec<f64>], tol: f64) -> Vec<Vec<f64>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let keep: Vec<bool> = (0..matrix[0].len())
        .map(|col_idx| matrix.iter().any(|row| row[col_idx].abs() > tol))
        .collect();
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter_map(|(col_idx, value)| keep[col_idx].then_some(*value))
                .collect()
        })
        .collect()
}

pub(super) fn cholesky_lower(matrix: &[Vec<f64>]) -> Result<Mat<f64>, String> {
    if matrix.iter().any(|row| row.len() != matrix.len()) {
        return Err("covariance matrix must be square".to_string());
    }
    if matrix.is_empty() {
        return Ok(Mat::zeros(0, 0));
    }
    let sym = Mat::from_fn(matrix.len(), matrix.len(), |row, col| {
        0.5 * (matrix[row][col] + matrix[col][row])
    });
    let scratch =
        llt::factor::cholesky_in_place_scratch::<f64>(matrix.len(), Par::Seq, Spec::default());
    let mut buffer = MemBuffer::new(scratch);
    if let Some(lower) = try_dense_cholesky(&sym, LltRegularization::default(), &mut buffer) {
        return Ok(lower);
    }
    if let Some(lower) = try_dense_cholesky_with_diagonal_jitter(&sym, &mut buffer) {
        return Ok(lower);
    }
    cholesky_lower_tolerant(&sym)
}

fn cholesky_lower_tolerant(matrix: &Mat<f64>) -> Result<Mat<f64>, String> {
    let n = matrix.nrows();
    let max_diag = (0..n)
        .map(|idx| matrix[(idx, idx)].abs())
        .fold(0.0_f64, f64::max);
    let neg_diag_tol = 1e-10_f64 * max_diag.max(1.0);
    let mut lower = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[(i, j)];
            for k in 0..j {
                sum = lower[(i, k)].mul_add(-lower[(j, k)], sum);
            }
            if i == j {
                if sum < -neg_diag_tol {
                    return Err("covariance matrix is not positive semidefinite".to_string());
                }
                lower[(i, j)] = sum.max(0.0).sqrt();
            } else if lower[(j, j)].abs() > 1e-12 {
                lower[(i, j)] = sum / lower[(j, j)];
            }
        }
    }
    Ok(lower)
}

fn try_dense_cholesky(
    matrix: &Mat<f64>,
    regularization: LltRegularization<f64>,
    buffer: &mut MemBuffer,
) -> Option<Mat<f64>> {
    let n = matrix.nrows();
    let mut dense = matrix.to_owned();
    llt::factor::cholesky_in_place(
        dense.as_mut(),
        regularization,
        Par::Seq,
        MemStack::new(buffer),
        Spec::default(),
    )
    .ok()?;
    for row in 0..n {
        for col in (row + 1)..n {
            dense[(row, col)] = 0.0;
        }
    }
    Some(dense)
}

fn try_dense_cholesky_with_diagonal_jitter(
    matrix: &Mat<f64>,
    buffer: &mut MemBuffer,
) -> Option<Mat<f64>> {
    let n = matrix.nrows();
    if n == 0 {
        return Some(Mat::zeros(0, 0));
    }
    let max_diag = (0..n)
        .map(|idx| matrix[(idx, idx)].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let reg = LltRegularization {
        dynamic_regularization_delta: max_diag * 1e-12,
        dynamic_regularization_epsilon: 1e-12,
    };
    if let Some(lower) = try_dense_cholesky(matrix, reg, buffer) {
        return Some(lower);
    }
    None
}

pub(super) fn lower_mat_vec_mul_into(lower: &Mat<f64>, vec: &[f64], out: &mut [f64]) {
    debug_assert_eq!(lower.nrows(), out.len());
    debug_assert_eq!(lower.ncols(), vec.len());
    if lower.nrows() >= 16 {
        let vector = MatRef::from_column_major_slice(vec, vec.len(), 1);
        let out_view = MatMut::from_column_major_slice_mut(out, out.len(), 1);
        matmul(
            out_view,
            Accum::Replace,
            lower.as_ref(),
            vector,
            1.0,
            Par::Seq,
        );
        return;
    }
    for row in 0..lower.nrows() {
        let mut sum = 0.0;
        for col in 0..=row {
            sum = lower[(row, col)].mul_add(vec[col], sum);
        }
        out[row] = sum;
    }
}

pub(super) fn pointwise_confidence_level_from_critical(critical: f64) -> Result<f64, String> {
    Ok((2.0 * normal::cdf(critical) - 1.0).clamp(1e-12, 1.0 - 1e-12))
}

pub(super) fn critical_value_from_pointwise_confidence(pointwise: f64) -> Result<f64, String> {
    Ok(normal::quantile((1.0 + pointwise) * 0.5))
}

/// The sup-t critical value of the estimator whose covariance is `covariance`,
/// for a joint confidence level of `confidence_level`.
pub(super) fn sup_t_critical_value(
    covariance: &[Vec<f64>],
    confidence_level: f64,
    simulation_draws: usize,
    simulation_seed: u64,
) -> Result<f64, String> {
    let dim = covariance.len();
    if covariance.iter().any(|row| row.len() != dim) {
        return Err("covariance matrix must be square".to_string());
    }
    let matrix = Mat::from_fn(dim, dim, |row, col| covariance[row][col]);
    epistat_inference::simultaneous::sup_t_from_covariance(
        matrix.as_ref(),
        1.0 - confidence_level,
        simulation_draws,
        simulation_seed,
    )
    .map_err(|error| format!("sup-t critical value: {error}"))
}

/// Covariance of the linear functionals `weights[i]' beta` of an estimator
/// whose covariance is `covariance`. Fails with `message` when a functional
/// has a variance that is not above `1e-14`.
pub(super) fn functional_covariance_matrix(
    covariance: &[Vec<f64>],
    weight_sets: &[Vec<f64>],
    message: &str,
) -> Result<Vec<Vec<f64>>, String> {
    let projected = weight_sets
        .iter()
        .map(|weights| {
            covariance
                .iter()
                .map(|row| dot(weights, row))
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<_>>();
    let variances = projected
        .iter()
        .zip(weight_sets.iter())
        .map(|(projection, weights)| dot(weights, projection).max(0.0))
        .collect::<Vec<_>>();
    if variances.iter().any(|variance| *variance <= 1e-14) {
        return Err(message.to_string());
    }
    let count = weight_sets.len();
    let mut result = vec![vec![0.0; count]; count];
    for i in 0..count {
        result[i][i] = variances[i];
        for j in (i + 1)..count {
            let value = dot(&weight_sets[i], &projected[j]);
            result[i][j] = value;
            result[j][i] = value;
        }
    }
    Ok(result)
}

pub(super) fn post_covariance_block(
    covariance: &[Vec<f64>],
    num_pre_periods: usize,
    num_post_periods: usize,
) -> Vec<Vec<f64>> {
    (0..num_post_periods)
        .map(|row| {
            (0..num_post_periods)
                .map(|col| covariance[num_pre_periods + row][num_pre_periods + col])
                .collect()
        })
        .collect()
}

pub(super) fn solve_square_linear_system(
    matrix: &[Vec<f64>],
    rhs: &[f64],
) -> Result<Vec<f64>, String> {
    let (rows, cols) = matrix_shape(matrix, "coefficient")?;
    if rows != cols {
        return Err(format!(
            "square linear solve requires a square coefficient matrix; got {rows}x{cols}"
        ));
    }
    if rhs.len() != rows {
        return Err(format!(
            "square linear solve rhs length {} does not match matrix size {rows}",
            rhs.len()
        ));
    }
    let dense = Mat::from_fn(rows, cols, |row, col| matrix[row][col]);
    let rhs_mat = Mat::from_fn(rows, 1, |row, _| rhs[row]);
    let solution = dense.as_ref().llt(Side::Lower).map_or_else(
        |_| dense.as_ref().partial_piv_lu().solve(&rhs_mat),
        |cholesky| cholesky.solve(&rhs_mat),
    );
    let out: Vec<f64> = (0..rows).map(|row| solution[(row, 0)]).collect();
    let residual = matrix
        .iter()
        .zip(rhs.iter())
        .map(|(row, rhs_value)| {
            row.iter()
                .zip(out.iter())
                .map(|(coeff, value)| coeff * value)
                .sum::<f64>()
                - rhs_value
        })
        .map(f64::abs)
        .fold(0.0, f64::max);
    let scale = rhs
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(1.0);
    if residual > 1e-7 * scale {
        return Err(format!(
            "square linear solve residual {residual} exceeds tolerance {}",
            1e-7 * scale
        ));
    }
    Ok(out)
}

pub(super) fn solve_square_linear_system_transposed(
    matrix: &[Vec<f64>],
    rhs: &[f64],
) -> Result<Vec<f64>, String> {
    let (rows, cols) = matrix_shape(matrix, "coefficient")?;
    if rows != cols {
        return Err(format!(
            "transposed square linear solve requires a square coefficient matrix; got {rows}x{cols}"
        ));
    }
    if rhs.len() != rows {
        return Err(format!(
            "transposed square linear solve rhs length {} does not match matrix size {rows}",
            rhs.len()
        ));
    }
    let dense_transposed = Mat::from_fn(rows, cols, |row, col| matrix[col][row]);
    let rhs_mat = Mat::from_fn(rows, 1, |row, _| rhs[row]);
    let solution = dense_transposed.as_ref().partial_piv_lu().solve(&rhs_mat);
    let out: Vec<f64> = (0..rows).map(|row| solution[(row, 0)]).collect();
    let residual = (0..rows)
        .map(|row| {
            (0..cols)
                .map(|col| matrix[col][row] * out[col])
                .sum::<f64>()
                - rhs[row]
        })
        .map(f64::abs)
        .fold(0.0, f64::max);
    let scale = rhs
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(1.0);
    if residual > 1e-7 * scale {
        return Err(format!(
            "transposed square linear solve residual {residual} exceeds tolerance {}",
            1e-7 * scale
        ));
    }
    Ok(out)
}

pub(super) fn invert_square_matrix(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let (rows, cols) = matrix_shape(matrix, "coefficient")?;
    if rows != cols {
        return Err(format!(
            "matrix inverse requires a square coefficient matrix; got {rows}x{cols}"
        ));
    }
    let dense = Mat::from_fn(rows, cols, |row, col| matrix[row][col]);
    let identity = Mat::from_fn(rows, cols, |row, col| if row == col { 1.0 } else { 0.0 });
    let inverse = dense.as_ref().partial_piv_lu().solve(&identity);
    let out: Vec<Vec<f64>> = (0..rows)
        .map(|row| (0..cols).map(|col| inverse[(row, col)]).collect())
        .collect();
    let residual = (0..rows)
        .flat_map(|row| {
            let out_ref = &out;
            (0..cols).map(move |col| {
                let expected = if row == col { 1.0 } else { 0.0 };
                matrix[row]
                    .iter()
                    .zip(out_ref.iter())
                    .map(|(left, inverse_row)| left * inverse_row[col])
                    .sum::<f64>()
                    - expected
            })
        })
        .map(f64::abs)
        .fold(0.0, f64::max);
    if residual > 1e-7 {
        return Err(format!(
            "matrix inverse residual {residual} exceeds tolerance 1e-7"
        ));
    }
    Ok(out)
}

pub(super) fn try_invert_square_matrix_row_major_into(
    matrix: &[f64],
    n: usize,
    out: &mut Vec<f64>,
) -> bool {
    let dense = MatRef::from_row_major_slice(matrix, n, n).to_owned();
    let identity = Mat::from_fn(n, n, |row, col| if row == col { 1.0 } else { 0.0 });
    let inverse = dense.as_ref().partial_piv_lu().solve(&identity);
    out.clear();
    out.resize(n * n, 0.0);
    for row in 0..n {
        for col in 0..n {
            out[row * n + col] = inverse[(row, col)];
        }
    }
    let mut residual = 0.0_f64;
    for row in 0..n {
        for col in 0..n {
            let dot = (0..n)
                .map(|k| matrix[row * n + k] * out[k * n + col])
                .sum::<f64>();
            let target = if row == col { 1.0 } else { 0.0 };
            residual = residual.max((dot - target).abs());
        }
    }
    residual <= 1e-7
}

pub(super) fn matrix_rank(matrix: &[Vec<f64>], tol: f64) -> usize {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0;
    }
    let (rows, cols) = matrix_shape_unchecked(matrix);
    let mut flat = flatten_row_major(matrix, rows * cols);
    let mut rank = 0usize;
    let mut row = 0usize;
    for col in 0..cols {
        let pivot = (row..rows).max_by(|&left, &right| {
            flat[left * cols + col]
                .abs()
                .total_cmp(&flat[right * cols + col].abs())
        });
        let Some(pivot_row) = pivot else { break };
        if flat[pivot_row * cols + col].abs() <= tol {
            continue;
        }
        if pivot_row != row {
            for offset in 0..cols {
                flat.swap(row * cols + offset, pivot_row * cols + offset);
            }
        }
        let row_start = row * cols;
        let pivot_value = flat[row_start + col];
        for value in &mut flat[(row_start + col)..(row_start + cols)] {
            *value /= pivot_value;
        }
        for r in 0..rows {
            if r == row {
                continue;
            }
            let target_start = r * cols;
            let factor = flat[target_start + col];
            if factor.abs() <= tol {
                continue;
            }
            for offset in col..cols {
                let target_idx = target_start + offset;
                flat[target_idx] = factor.mul_add(-flat[row_start + offset], flat[target_idx]);
            }
        }
        rank += 1;
        row += 1;
        if row == rows {
            break;
        }
    }
    rank
}

pub(super) fn mat_vec_mul_into(matrix: &[Vec<f64>], vector: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.reserve(matrix.len());
    for row in matrix {
        let value = row
            .iter()
            .zip(vector)
            .fold(0.0, |acc, (left, right)| left.mul_add(*right, acc));
        out.push(value);
    }
}

pub(super) fn sandwich_covariance(a: &[Vec<f64>], sigma: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (a_rows, a_cols) = matrix_shape_unchecked(a);
    if a_rows == 0 {
        return Vec::new();
    }
    let (sigma_rows, sigma_cols) = matrix_shape_unchecked(sigma);
    assert_eq!(
        a_cols, sigma_rows,
        "sandwich covariance dimension mismatch: a has {a_cols} columns, sigma has {sigma_rows} rows"
    );
    assert_eq!(
        sigma_rows, sigma_cols,
        "sandwich covariance requires square sigma; got {sigma_rows}x{sigma_cols}"
    );

    let a_flat = flatten_row_major(a, a_rows * a_cols);
    let sigma_flat = flatten_row_major(sigma, sigma_rows * sigma_cols);
    let a_mat = MatRef::from_row_major_slice(&a_flat, a_rows, a_cols).to_owned();
    let sigma_mat = MatRef::from_row_major_slice(&sigma_flat, sigma_rows, sigma_cols).to_owned();

    let mut tmp = Mat::<f64>::zeros(a_rows, sigma_cols);
    matmul(
        tmp.as_mut(),
        Accum::Replace,
        a_mat.as_ref(),
        sigma_mat.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut out = Mat::<f64>::zeros(a_rows, a_rows);
    matmul(
        out.as_mut(),
        Accum::Replace,
        tmp.as_ref(),
        a_mat.as_ref().transpose(),
        1.0,
        Par::Seq,
    );
    mat_to_row_vecs(&out)
}

pub(super) fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(l, r)| l * r).sum()
}

pub(super) fn bilinear_form_into(
    left: &[f64],
    matrix: &[Vec<f64>],
    right: &[f64],
    tmp: &mut Vec<f64>,
) -> f64 {
    let (rows, cols) = matrix_shape_unchecked(matrix);
    assert_eq!(
        left.len(),
        rows,
        "bilinear form dimension mismatch: left len {} != matrix rows {rows}",
        left.len()
    );
    assert_eq!(
        right.len(),
        cols,
        "bilinear form dimension mismatch: right len {} != matrix cols {cols}",
        right.len()
    );
    if rows == 0 || cols == 0 {
        return 0.0;
    }
    tmp.clear();
    tmp.resize(cols, 0.0);
    for col in 0..cols {
        tmp[col] = left
            .iter()
            .enumerate()
            .map(|(row, l)| l * matrix[row][col])
            .sum::<f64>();
    }
    dot(tmp, right)
}

fn matrix_shape(matrix: &[Vec<f64>], name: &str) -> Result<(usize, usize), String> {
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);
    if matrix.iter().any(|row| row.len() != cols) {
        return Err(format!("{name} matrix has non-rectangular rows"));
    }
    Ok((rows, cols))
}

fn matrix_shape_unchecked(matrix: &[Vec<f64>]) -> (usize, usize) {
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);
    debug_assert!(matrix.iter().all(|row| row.len() == cols));
    (rows, cols)
}

fn flatten_row_major(matrix: &[Vec<f64>], capacity: usize) -> Vec<f64> {
    let mut flat = Vec::with_capacity(capacity);
    for row in matrix {
        flat.extend_from_slice(row);
    }
    flat
}

fn mat_to_row_vecs(matrix: &Mat<f64>) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; matrix.ncols()]; matrix.nrows()];
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            out[row][col] = matrix[(row, col)];
        }
    }
    out
}

pub(super) fn build_clarabel_matrix(
    inequalities: &[Vec<f64>],
    equalities: &[Vec<f64>],
) -> CscMatrix<f64> {
    let total_rows = inequalities.len() + equalities.len();
    let total_cols = inequalities
        .first()
        .map_or_else(|| equalities.first().map_or(0, Vec::len), Vec::len);
    let cap = total_rows * total_cols;
    let mut row_idx = Vec::with_capacity(cap);
    let mut col_idx = Vec::with_capacity(cap);
    let mut values = Vec::with_capacity(cap);
    for (r, row) in inequalities.iter().chain(equalities.iter()).enumerate() {
        for (c, value) in row.iter().enumerate() {
            if *value != 0.0 {
                row_idx.push(r);
                col_idx.push(c);
                values.push(*value);
            }
        }
    }
    CscMatrix::new_from_triplets(total_rows, total_cols, row_idx, col_idx, values)
}

#[cfg(test)]
pub(super) fn dense_rows_to_csc(rows: &[Vec<f64>]) -> CscMatrix<f64> {
    build_clarabel_matrix(rows, &[])
}

/// Quantile of a standard normal truncated to `[l, u]`. A degenerate interval
/// `l == u` gives that point, as `TruncatedNormal::norminvp` does in R. An
/// interval with `l > u`, a `NaN` bound, a probability outside `[0, 1]` or an
/// interval whose probability underflows is an error.
pub(super) fn truncated_normal_quantile(p: f64, l: f64, u: f64) -> Result<f64, String> {
    let undefined =
        || format!("truncated normal quantile is undefined for p = {p}, interval [{l}, {u}]");
    if p.is_nan() || l.is_nan() || u.is_nan() || !(0.0..=1.0).contains(&p) {
        return Err(undefined());
    }
    if l == u {
        return Ok(l);
    }
    epistat_stats::normal::truncated_quantile(p, l, u).ok_or_else(undefined)
}

#[cfg(test)]
mod tests {
    use super::truncated_normal_quantile;

    #[test]
    fn truncated_quantile_follows_norminvp_on_degenerate_and_empty_intervals() {
        assert!((truncated_normal_quantile(0.5, -1.0, 1.0).expect("interval")).abs() < 1e-12);
        assert_eq!(truncated_normal_quantile(0.9, 0.3, 0.3), Ok(0.3));
        assert!(truncated_normal_quantile(0.9, 1.0, 0.0).is_err());
        assert!(truncated_normal_quantile(0.9, f64::NAN, 1.0).is_err());
        assert!(truncated_normal_quantile(1.5, 0.0, 1.0).is_err());
    }
}
