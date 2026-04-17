use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT;

use super::super::linear_algebra::{
    build_clarabel_matrix, drop_zero_columns, invert_square_matrix, select_rows,
    subset_square_matrix,
};

pub(in crate::inference::sensitivity) struct RelativeMagnitudePreparedBranch {
    pub(in crate::inference::sensitivity) rows_for_arp: Vec<usize>,
    pub(in crate::inference::sensitivity) constraint_rows: Vec<Vec<f64>>,
    pub(in crate::inference::sensitivity) a_post: Vec<Vec<f64>>,
    pub(in crate::inference::sensitivity) solver_matrix: CscMatrix<f64>,
    pub(in crate::inference::sensitivity) cones: Vec<SupportedConeT<f64>>,
    pub(in crate::inference::sensitivity) inequality_len: usize,
}

pub(in crate::inference::sensitivity) fn basis_post_weights(
    num_post_periods: usize,
    post_idx: usize,
) -> Vec<f64> {
    let mut post_weights = vec![0.0; num_post_periods];
    post_weights[post_idx] = 1.0;
    post_weights
}

pub(in crate::inference::sensitivity) fn prepare_relative_magnitude_branches(
    num_pre: usize,
    num_post: usize,
    mbar: f64,
) -> Result<Vec<RelativeMagnitudePreparedBranch>, String> {
    let equality_matrix = create_pre_period_equality_matrix(num_pre, num_post);
    let min_s = -(isize::try_from(num_pre).map_err(|_| "too many pre-periods".to_string())? - 1);
    (min_s..=0)
        .flat_map(|s| {
            [true, false]
                .into_iter()
                .map(move |max_positive| (s, max_positive))
        })
        .map(|(s, max_positive)| {
            let inequality_matrix = build_relative_magnitude_constraint_matrix(
                num_pre,
                num_post,
                mbar,
                s,
                max_positive,
            )?;
            let inequality_len = inequality_matrix.len();
            let rows_for_arp = find_post_period_constraint_rows(&inequality_matrix, num_pre);
            let a_post = inequality_matrix
                .iter()
                .map(|row| row[num_pre..].to_vec())
                .collect();
            let solver_matrix = build_clarabel_matrix(&inequality_matrix, &equality_matrix);
            let cones = vec![
                SupportedConeT::NonnegativeConeT(inequality_len),
                SupportedConeT::ZeroConeT(num_pre),
            ];
            Ok(RelativeMagnitudePreparedBranch {
                rows_for_arp,
                constraint_rows: inequality_matrix,
                a_post,
                solver_matrix,
                cones,
                inequality_len,
            })
        })
        .collect()
}

pub(in crate::inference::sensitivity) fn basis_vector_index(post_weights: &[f64]) -> Option<usize> {
    let mut index = None;
    for (idx, value) in post_weights.iter().copied().enumerate() {
        if !value.is_finite() {
            return None;
        }
        if value.abs() <= 1e-12 {
            continue;
        }
        if (value - 1.0).abs() > 1e-12 || index.is_some() {
            return None;
        }
        index = Some(idx);
    }
    index
}

pub(in crate::inference::sensitivity) fn construct_gamma(
    post_weights: &[f64],
) -> Result<Vec<Vec<f64>>, String> {
    let bar_t = post_weights.len();
    let mut gamma = Vec::with_capacity(bar_t);
    let Some(pivot_idx) = post_weights.iter().position(|weight| weight.abs() > 1e-12) else {
        return Err("failed to construct Gamma basis from zero post weights".to_string());
    };
    gamma.push(post_weights.to_vec());
    for basis_idx in 0..bar_t {
        if basis_idx == pivot_idx {
            continue;
        }
        let mut row = vec![0.0; bar_t];
        row[basis_idx] = 1.0;
        gamma.push(row);
    }
    Ok(gamma)
}

type ArpViews = (Vec<Vec<f64>>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>);
const RREF_NOCLONE_MAX_ROWS: usize = 16;

pub(in crate::inference::sensitivity) enum RelativeMagnitudePreparedFunctionalTransform {
    Basis { target_idx: usize },
    General { gamma_inverse: Vec<Vec<f64>> },
}

pub(in crate::inference::sensitivity) fn prepare_relative_magnitude_functional_transform(
    post_weights: &[f64],
) -> Result<RelativeMagnitudePreparedFunctionalTransform, String> {
    if let Some(target_idx) = basis_vector_index(post_weights) {
        Ok(RelativeMagnitudePreparedFunctionalTransform::Basis { target_idx })
    } else {
        let gamma = construct_gamma(post_weights)?;
        let gamma_inverse = invert_square_matrix(&gamma)?;
        Ok(RelativeMagnitudePreparedFunctionalTransform::General { gamma_inverse })
    }
}

pub(in crate::inference::sensitivity) fn rref_pivot_columns(
    matrix: &[Vec<f64>],
    tol: f64,
) -> Result<Vec<usize>, String> {
    if matrix.len() <= RREF_NOCLONE_MAX_ROWS {
        rref_pivot_columns_noclone(matrix, tol)
    } else {
        rref_pivot_columns_clone(matrix, tol)
    }
}

fn rref_pivot_columns_noclone(matrix: &[Vec<f64>], tol: f64) -> Result<Vec<usize>, String> {
    if matrix.is_empty() {
        return Ok(Vec::new());
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut m = matrix.to_vec();
    let mut pivot_columns = Vec::new();
    let mut pivot_row = 0usize;
    for col in 0..cols {
        if pivot_row >= rows {
            break;
        }
        let Some(best_row) = (pivot_row..rows)
            .max_by(|&left, &right| m[left][col].abs().total_cmp(&m[right][col].abs()))
        else {
            break;
        };
        if m[best_row][col].abs() <= tol {
            continue;
        }
        m.swap(pivot_row, best_row);
        let pivot_value = m[pivot_row][col];
        for value in &mut m[pivot_row] {
            *value /= pivot_value;
        }
        for row in 0..rows {
            if row == pivot_row {
                continue;
            }
            let (target_row, pivot_row_ref) = if row < pivot_row {
                let (before_pivot, pivot_and_after) = m.split_at_mut(pivot_row);
                (&mut before_pivot[row], &pivot_and_after[0])
            } else {
                let (up_to_target, target_and_after) = m.split_at_mut(row);
                (&mut target_and_after[0], &up_to_target[pivot_row])
            };
            let factor = target_row[col];
            if factor.abs() <= tol {
                continue;
            }
            for c in 0..cols {
                target_row[c] = factor.mul_add(-pivot_row_ref[c], target_row[c]);
            }
        }
        pivot_columns.push(col);
        pivot_row += 1;
    }
    if pivot_columns.len() != rows {
        return Err("matrix did not yield a full pivot basis".to_string());
    }
    Ok(pivot_columns)
}

fn rref_pivot_columns_clone(matrix: &[Vec<f64>], tol: f64) -> Result<Vec<usize>, String> {
    if matrix.is_empty() {
        return Ok(Vec::new());
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut m = matrix.to_vec();
    let mut pivot_columns = Vec::new();
    let mut pivot_row = 0usize;
    for col in 0..cols {
        if pivot_row >= rows {
            break;
        }
        let Some(best_row) = (pivot_row..rows)
            .max_by(|&left, &right| m[left][col].abs().total_cmp(&m[right][col].abs()))
        else {
            break;
        };
        if m[best_row][col].abs() <= tol {
            continue;
        }
        m.swap(pivot_row, best_row);
        let pivot_value = m[pivot_row][col];
        for value in &mut m[pivot_row] {
            *value /= pivot_value;
        }
        for row in 0..rows {
            if row == pivot_row {
                continue;
            }
            let factor = m[row][col];
            if factor.abs() <= tol {
                continue;
            }
            let pivot_snapshot = m[pivot_row].clone();
            for (target, pivot_entry) in m[row].iter_mut().zip(pivot_snapshot.iter()) {
                *target = factor.mul_add(-*pivot_entry, *target);
            }
        }
        pivot_columns.push(col);
        pivot_row += 1;
    }
    if pivot_columns.len() != rows {
        return Err("matrix did not yield a full pivot basis".to_string());
    }
    Ok(pivot_columns)
}

pub(super) fn find_post_period_constraint_rows(
    a_matrix: &[Vec<f64>],
    num_pre: usize,
) -> Vec<usize> {
    a_matrix
        .iter()
        .enumerate()
        .filter_map(|(row_idx, row)| {
            row[num_pre..]
                .iter()
                .any(|value| value.abs() > 1e-12)
                .then_some(row_idx)
        })
        .collect()
}

pub(super) fn relative_magnitude_objective(
    num_pre: usize,
    num_post: usize,
    post_weights: &[f64],
) -> Vec<f64> {
    let mut objective = vec![0.0; num_pre];
    objective.extend_from_slice(&post_weights[..num_post]);
    objective
}

pub(super) fn create_pre_period_equality_matrix(num_pre: usize, num_post: usize) -> Vec<Vec<f64>> {
    (0..num_pre)
        .map(|row_idx| {
            let mut row = vec![0.0; num_pre + num_post];
            row[row_idx] = 1.0;
            row
        })
        .collect()
}

pub(super) fn build_relative_magnitude_constraint_matrix(
    num_pre: usize,
    num_post: usize,
    mbar: f64,
    s: isize,
    max_positive: bool,
) -> Result<Vec<Vec<f64>>, String> {
    let total_periods = num_pre + num_post;
    let full_cols = total_periods + 1;
    let mut a_tilde = vec![vec![0.0; full_cols]; total_periods];
    for (row_idx, row) in a_tilde.iter_mut().enumerate() {
        row[row_idx] = -1.0;
        row[row_idx + 1] = 1.0;
    }

    let start = isize::try_from(num_pre).map_err(|_| "too many pre-periods".to_string())? + s - 1;
    if start < 0
        || start + 1 >= isize::try_from(full_cols).map_err(|_| "too many periods".to_string())?
    {
        return Err(format!(
            "invalid relative-magnitude branch index s={s} for num_pre={num_pre}"
        ));
    }
    let start = usize::try_from(start)
        .map_err(|_| "invalid relative-magnitude branch index".to_string())?;
    let mut v_max = vec![0.0; full_cols];
    v_max[start] = -1.0;
    v_max[start + 1] = 1.0;
    if !max_positive {
        for value in &mut v_max {
            *value = -*value;
        }
    }

    let mut a_ub = Vec::with_capacity(total_periods);
    for _ in 0..num_pre {
        a_ub.push(v_max.clone());
    }
    for _ in 0..num_post {
        a_ub.push(v_max.iter().map(|value| mbar * value).collect());
    }

    let mut constraints = Vec::with_capacity(total_periods * 2);
    for (tilde_row, ub_row) in a_tilde.iter().zip(&a_ub) {
        constraints.push(
            tilde_row
                .iter()
                .zip(ub_row)
                .map(|(left, right)| left - right)
                .collect::<Vec<_>>(),
        );
        constraints.push(
            tilde_row
                .iter()
                .zip(ub_row)
                .map(|(left, right)| -left - right)
                .collect::<Vec<_>>(),
        );
    }

    let zero_col = num_pre;
    let mut dropped = Vec::new();
    for row in constraints {
        let mut compact = Vec::with_capacity(total_periods);
        for (col_idx, value) in row.into_iter().enumerate() {
            if col_idx != zero_col {
                compact.push(value);
            }
        }
        if compact.iter().any(|value| value.abs() > 1e-10) {
            dropped.push(compact);
        }
    }
    Ok(dropped)
}

pub(in crate::inference::sensitivity) fn build_target_and_design(
    post_weights: &[f64],
    a_post: &[Vec<f64>],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
    let prepared = prepare_relative_magnitude_functional_transform(post_weights)?;
    Ok(build_target_and_design_from_transform(a_post, &prepared))
}

pub(in crate::inference::sensitivity) fn build_target_and_design_from_transform(
    a_post: &[Vec<f64>],
    transform: &RelativeMagnitudePreparedFunctionalTransform,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    match transform {
        RelativeMagnitudePreparedFunctionalTransform::Basis { target_idx } => {
            let mut x_matrix = Vec::with_capacity(a_post.len());
            let mut a_target = Vec::with_capacity(a_post.len());
            for row in a_post {
                a_target.push(row[*target_idx]);
                let mut reduced = Vec::with_capacity(a_post[0].len().saturating_sub(1));
                reduced.extend_from_slice(&row[..*target_idx]);
                reduced.extend_from_slice(&row[*target_idx + 1..]);
                x_matrix.push(reduced);
            }
            (a_target, x_matrix)
        }
        RelativeMagnitudePreparedFunctionalTransform::General { gamma_inverse } => {
            let projected = multiply_rows_by_square_matrix(a_post, gamma_inverse);
            let a_target: Vec<f64> = projected.iter().map(|row| row[0]).collect();
            let x_matrix: Vec<Vec<f64>> = projected.iter().map(|row| row[1..].to_vec()).collect();
            (a_target, x_matrix)
        }
    }
}

pub(in crate::inference::sensitivity) fn build_selected_target_and_design_from_transform(
    a_post: &[Vec<f64>],
    selected_rows: &[usize],
    transform: &RelativeMagnitudePreparedFunctionalTransform,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    match transform {
        RelativeMagnitudePreparedFunctionalTransform::Basis { target_idx } => {
            let mut x_matrix = Vec::with_capacity(selected_rows.len());
            let mut a_target = Vec::with_capacity(selected_rows.len());
            for row_idx in selected_rows {
                let row = &a_post[*row_idx];
                a_target.push(row[*target_idx]);
                let mut reduced = Vec::with_capacity(a_post[0].len().saturating_sub(1));
                reduced.extend_from_slice(&row[..*target_idx]);
                reduced.extend_from_slice(&row[*target_idx + 1..]);
                x_matrix.push(reduced);
            }
            (a_target, x_matrix)
        }
        RelativeMagnitudePreparedFunctionalTransform::General { gamma_inverse } => {
            let mut projected = vec![vec![0.0; gamma_inverse[0].len()]; selected_rows.len()];
            for (selected_idx, row_idx) in selected_rows.iter().copied().enumerate() {
                let left_row = &a_post[row_idx];
                for (shared_idx, left_value) in left_row.iter().copied().enumerate() {
                    if left_value.abs() <= 1e-12 {
                        continue;
                    }
                    for (col_idx, right_value) in
                        gamma_inverse[shared_idx].iter().copied().enumerate()
                    {
                        projected[selected_idx][col_idx] =
                            left_value.mul_add(right_value, projected[selected_idx][col_idx]);
                    }
                }
            }
            let a_target: Vec<f64> = projected.iter().map(|row| row[0]).collect();
            let x_matrix: Vec<Vec<f64>> = projected.iter().map(|row| row[1..].to_vec()).collect();
            (a_target, x_matrix)
        }
    }
}

fn multiply_rows_by_square_matrix(
    left_rows: &[Vec<f64>],
    right_square: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; right_square[0].len()]; left_rows.len()];
    for (row_idx, left_row) in left_rows.iter().enumerate() {
        for (shared_idx, left_value) in left_row.iter().copied().enumerate() {
            if left_value.abs() <= 1e-12 {
                continue;
            }
            for (col_idx, right_value) in right_square[shared_idx].iter().copied().enumerate() {
                out[row_idx][col_idx] = left_value.mul_add(right_value, out[row_idx][col_idx]);
            }
        }
    }
    out
}

pub(in crate::inference::sensitivity) fn prepare_arp_views(
    x_matrix: &[Vec<f64>],
    y_vec: &[f64],
    a_target: &[f64],
    sigma_y: &[Vec<f64>],
    rows_for_arp: &[usize],
) -> ArpViews {
    let sigma_arp = subset_square_matrix(sigma_y, rows_for_arp);
    let x_arp = drop_zero_columns(&select_rows(x_matrix, rows_for_arp), 1e-12);
    let y_arp_base: Vec<f64> = rows_for_arp.iter().map(|&row_idx| y_vec[row_idx]).collect();
    let a_target_arp: Vec<f64> = rows_for_arp
        .iter()
        .map(|&row_idx| a_target[row_idx])
        .collect();
    (x_arp, y_arp_base, a_target_arp, sigma_arp)
}
