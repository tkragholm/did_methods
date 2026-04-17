use clarabel::algebra::CscMatrix;
use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::linalg::matmul::matmul;
use faer::prelude::Solve;
use faer::{Accum, Mat, MatMut, MatRef, Par, Side, Spec};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

use crate::inference::inverse_standard_normal_cdf;
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

pub(super) fn draw_standard_normal_vec(rng: &mut StdRng, len: usize) -> Vec<f64> {
    let mut out = vec![0.0; len];
    draw_standard_normal_vec_into(rng, &mut out);
    out
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

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(super) fn simulation_rank(sample_count: usize, confidence_level: f64) -> usize {
    (usize_to_f64(sample_count.saturating_sub(1)) * confidence_level).round() as usize
}

pub(super) fn pointwise_confidence_level_from_critical(critical: f64) -> Result<f64, String> {
    let normal = Normal::new(0.0, 1.0)
        .map_err(|err| format!("failed to create normal distribution: {err}"))?;
    Ok(2.0f64
        .mul_add(normal.cdf(critical), -1.0)
        .clamp(1e-12, 1.0 - 1e-12))
}

pub(super) fn critical_value_from_pointwise_confidence(pointwise: f64) -> Result<f64, String> {
    let normal = Normal::new(0.0, 1.0)
        .map_err(|err| format!("failed to create normal distribution: {err}"))?;
    Ok(normal.inverse_cdf((1.0 + pointwise) * 0.5))
}

const SENSITIVITY_BATCHED_CHUNK_SIZE: usize = 64;

pub(super) fn simulated_lower_cholesky_maxima_batched(
    chol: &Mat<f64>,
    simulation_draws: usize,
    simulation_seed: u64,
) -> Vec<f64> {
    let dim = chol.nrows();
    if dim < 16 || simulation_draws < 64 {
        return simulated_lower_cholesky_maxima_scalar(chol, simulation_draws, simulation_seed);
    }

    let mut maxima = vec![0.0; simulation_draws];
    let full_draws = simulation_draws - (simulation_draws % SENSITIVITY_BATCHED_CHUNK_SIZE);
    maxima[..full_draws]
        .par_chunks_mut(SENSITIVITY_BATCHED_CHUNK_SIZE)
        .enumerate()
        .for_each_init(
            || {
                (
                    Mat::<f64>::zeros(dim, SENSITIVITY_BATCHED_CHUNK_SIZE),
                    Mat::<f64>::zeros(dim, SENSITIVITY_BATCHED_CHUNK_SIZE),
                    vec![0.0; dim],
                )
            },
            |(z, draws, draw_vec), (chunk_idx, chunk_out)| {
                let start = chunk_idx * SENSITIVITY_BATCHED_CHUNK_SIZE;
                for local_idx in 0..SENSITIVITY_BATCHED_CHUNK_SIZE {
                    let draw_idx = start + local_idx;
                    let mut rng =
                        StdRng::seed_from_u64(simulation_draw_seed(simulation_seed, draw_idx));
                    draw_standard_normal_vec_into(&mut rng, draw_vec);
                    for row in 0..dim {
                        z[(row, local_idx)] = draw_vec[row];
                    }
                }
                matmul(
                    draws.as_mut(),
                    Accum::Replace,
                    chol.as_ref(),
                    z.as_ref(),
                    1.0,
                    Par::Seq,
                );
                for local_idx in 0..SENSITIVITY_BATCHED_CHUNK_SIZE {
                    chunk_out[local_idx] = (0..dim)
                        .map(|row| draws[(row, local_idx)].abs())
                        .fold(0.0, f64::max);
                }
            },
        );

    let tail_start = full_draws;
    if tail_start < simulation_draws {
        let mut z = vec![0.0; dim];
        let mut draw = vec![0.0; dim];
        for (local_idx, out) in maxima[tail_start..].iter_mut().enumerate() {
            let draw_idx = tail_start + local_idx;
            let mut rng = StdRng::seed_from_u64(simulation_draw_seed(simulation_seed, draw_idx));
            draw_standard_normal_vec_into(&mut rng, &mut z);
            lower_mat_vec_mul_into(chol, &z, &mut draw);
            *out = draw.iter().map(|value| value.abs()).fold(0.0, f64::max);
        }
    }
    maxima
}

pub(super) fn simulated_lower_cholesky_maxima_scalar(
    chol: &Mat<f64>,
    simulation_draws: usize,
    simulation_seed: u64,
) -> Vec<f64> {
    let dim = chol.nrows();
    (0..simulation_draws)
        .into_par_iter()
        .map_init(
            || (vec![0.0; dim], vec![0.0; dim]),
            |(z, draw), draw_idx| {
                let mut rng =
                    StdRng::seed_from_u64(simulation_draw_seed(simulation_seed, draw_idx));
                draw_standard_normal_vec_into(&mut rng, z);
                lower_mat_vec_mul_into(chol, z, draw);
                draw.iter().map(|value| value.abs()).fold(0.0, f64::max)
            },
        )
        .collect()
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

pub(super) fn draw_standard_normal_vec_into(rng: &mut StdRng, out: &mut [f64]) {
    draw_standard_normal_vec_into_cached(rng, out);
}

pub(super) const fn simulation_draw_seed(base_seed: u64, draw_idx: usize) -> u64 {
    let idx = draw_idx as u64;
    let mut z = base_seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(idx + 1));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn draw_standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos()
}

pub(super) fn draw_standard_normal_vec_into_scalar(rng: &mut StdRng, out: &mut [f64]) {
    for value in out {
        *value = draw_standard_normal(rng);
    }
}

pub(super) fn draw_standard_normal_vec_into_cached(rng: &mut StdRng, out: &mut [f64]) {
    let mut index = 0usize;
    while index + 1 < out.len() {
        let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2 = rng.random::<f64>();
        let radius = (-2.0_f64 * u1.ln()).sqrt();
        let theta = 2.0_f64 * std::f64::consts::PI * u2;
        out[index] = radius * theta.cos();
        out[index + 1] = radius * theta.sin();
        index += 2;
    }
    if index < out.len() {
        out[index] = draw_standard_normal(rng);
    }
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

pub(super) fn truncated_normal_quantile(p: f64, l: f64, u: f64) -> Result<f64, String> {
    let normal = Normal::new(0.0, 1.0)
        .map_err(|err| format!("failed to create normal distribution: {err}"))?;
    let pl = if l.is_infinite() && l.is_sign_negative() {
        0.0
    } else {
        normal.cdf(l)
    };
    let pu = if u.is_infinite() && u.is_sign_positive() {
        1.0
    } else {
        normal.cdf(u)
    };
    if pu <= pl {
        return Ok(0.0);
    }
    let target = p.mul_add(pu - pl, pl);
    Ok(inverse_standard_normal_cdf(
        target.clamp(1e-12, 1.0 - 1e-12),
    ))
}

#[cfg(test)]
mod tests {
    use super::{draw_standard_normal_vec_into_cached, draw_standard_normal_vec_into_scalar};
    use rand::{SeedableRng, rngs::StdRng};

    fn mean_and_variance(sample: &[f64]) -> (f64, f64) {
        let n = crate::util::usize_to_f64(sample.len());
        let mean = sample.iter().sum::<f64>() / n;
        let var = sample
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        (mean, var)
    }

    #[test]
    fn cached_and_scalar_normal_draws_are_statistically_consistent() {
        let seed = 42_042_u64;
        let len = 20_001usize;

        let mut scalar_rng = StdRng::seed_from_u64(seed);
        let mut scalar = vec![0.0; len];
        draw_standard_normal_vec_into_scalar(&mut scalar_rng, &mut scalar);
        let (scalar_mean, scalar_var) = mean_and_variance(&scalar);

        let mut cached_rng = StdRng::seed_from_u64(seed);
        let mut cached = vec![0.0; len];
        draw_standard_normal_vec_into_cached(&mut cached_rng, &mut cached);
        let (cached_mean, cached_var) = mean_and_variance(&cached);

        assert!((scalar_mean - cached_mean).abs() < 0.03);
        assert!((scalar_var - cached_var).abs() < 0.05);
        assert!(cached_mean.abs() < 0.03);
        assert!((cached_var - 1.0).abs() < 0.05);
    }
}
