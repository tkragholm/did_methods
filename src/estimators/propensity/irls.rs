use faer::prelude::*;
use faer::{Mat, MatRef};

use super::common::{debug_matrix_ranges, empirical_logit, group_binary_design_rows, safe_sigmoid};
use crate::error::InternalDidError;

/// Fit logistic regression via iteratively reweighted least squares.
///
/// # Errors
/// Returns `InternalDidError::Convergence` if IRLS fails to converge or errors encountered during weighting.
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn irls(
    design: MatRef<'_, f64>,
    target: MatRef<'_, f64>,
    max_iter: u64,
    tol: f64,
    min_weight: f64,
) -> Result<Mat<f64>, InternalDidError> {
    let grouped = group_binary_design_rows(design, target);
    let grouped_design = grouped.design.as_ref();
    let grouped_target = grouped.target_means.as_ref();
    let mut beta = initial_beta(grouped_design, grouped_target);
    for iter in 0..max_iter {
        let x_beta = grouped_design * &beta;
        let p_hat = prob(x_beta.as_ref());
        let w = weights(p_hat.as_ref(), min_weight);
        let z = working_response(grouped_target, x_beta.as_ref(), p_hat.as_ref(), w.as_ref());
        let new_beta = solve_wls(
            grouped_design,
            &grouped.total_counts,
            w.as_ref(),
            z.as_ref(),
        );
        let change = (&new_beta - &beta).norm_l2();
        if iter < 5 || iter % 10 == 0 {
            debug_matrix_ranges("p_hat", p_hat.as_ref());
            debug_matrix_ranges("w", w.as_ref());
        }
        if change < tol {
            return Ok(new_beta);
        }
        beta = new_beta;
    }
    Err(InternalDidError::Convergence(
        "IRLS did not converge".to_string(),
    ))
}

fn initial_beta(design: MatRef<'_, f64>, target: MatRef<'_, f64>) -> Mat<f64> {
    let mut beta0 = Mat::zeros(design.ncols(), 1);
    *beta0.get_mut(0, 0) = empirical_logit(target);
    beta0
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn prob(x_beta: MatRef<'_, f64>) -> Mat<f64> {
    let mut prob_vec = Mat::zeros(x_beta.nrows(), 1);
    for row in 0..x_beta.nrows() {
        *prob_vec.get_mut(row, 0) = safe_sigmoid(*x_beta.get(row, 0));
    }
    prob_vec
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn weights(prob_vec: MatRef<'_, f64>, min_weight: f64) -> Mat<f64> {
    let mut w_mat = Mat::zeros(prob_vec.nrows(), 1);
    for row in 0..prob_vec.nrows() {
        let pval = prob_vec.get(row, 0);
        *w_mat.get_mut(row, 0) = (pval * (1.0 - pval)).max(min_weight);
    }
    w_mat
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn working_response(
    d: MatRef<'_, f64>,
    x_beta: MatRef<'_, f64>,
    p_hat: MatRef<'_, f64>,
    w: MatRef<'_, f64>,
) -> Mat<f64> {
    let mut z = Mat::zeros(x_beta.nrows(), 1);
    for i in 0..x_beta.nrows() {
        let eta = x_beta.get(i, 0);
        let p = p_hat.get(i, 0);
        let di = *d.get(i, 0);
        let wi = *w.get(i, 0);
        *z.get_mut(i, 0) = eta + (di - p) / wi;
    }
    z
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn solve_wls(
    x: MatRef<'_, f64>,
    row_counts: &[f64],
    w: MatRef<'_, f64>,
    z: MatRef<'_, f64>,
) -> Mat<f64> {
    let feature_count = x.ncols();
    let mut x_t_w_x: Mat<f64> = Mat::zeros(feature_count, feature_count);
    let mut x_t_w_z: Mat<f64> = Mat::zeros(feature_count, 1);

    for (row, row_count) in row_counts.iter().copied().enumerate().take(x.nrows()) {
        let row_weight = row_count * *w.get(row, 0);
        let response = *z.get(row, 0);
        if row_weight <= 0.0 {
            continue;
        }
        for lhs_col in 0..feature_count {
            let lhs = *x.get(row, lhs_col);
            *x_t_w_z.get_mut(lhs_col, 0) =
                (row_weight * lhs).mul_add(response, *x_t_w_z.get_mut(lhs_col, 0));
            for rhs_col in 0..feature_count {
                *x_t_w_x.get_mut(lhs_col, rhs_col) = (row_weight * lhs)
                    .mul_add(*x.get(row, rhs_col), *x_t_w_x.get_mut(lhs_col, rhs_col));
            }
        }
    }

    let reg = 1e-8;
    for i in 0..x_t_w_x.ncols() {
        *x_t_w_x.get_mut(i, i) += reg;
    }
    x_t_w_x.partial_piv_lu().solve(&x_t_w_z)
}
