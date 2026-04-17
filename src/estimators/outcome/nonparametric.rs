//! Nonparametric outcome regression estimators.
//!
//! The local-linear estimator implemented here is intended for semiparametric
//! nuisance estimation in repeated-cross-section workflows such as
//! [`DiD_CC`](crate::methods::did_cc). For a query point `x`, it fits a local
//! weighted least-squares regression of `Y` on `[1, X - x]`, where weights are
//! the product of sampling weights and Gaussian kernel weights. The fitted
//! intercept is the estimated conditional mean `m(x)`.
//!
//! References:
//! - Fan, J. and Gijbels, I. (1996). *Local Polynomial Modelling and Its
//!   Applications*.
//! - Sant'Anna, P. H. C. and Xu, L. (2026). `DiD_CC`.

use faer::prelude::SolveLstsq;
use faer::{Mat, MatRef};
use std::collections::HashMap;

use crate::estimators::common::linalg::{SpdCholeskyScratch, solve_spd_system};

/// Local-linear nonparametric outcome regression with Gaussian kernel weights.
#[derive(Debug, Clone, Copy)]
pub struct LocalLinearOutcome {
    /// Ridge stabilization added to the weighted normal equations.
    pub ridge: f64,
    /// Multiplicative factor applied to the rule-of-thumb bandwidth.
    pub bandwidth_scale: f64,
    /// Lower bound used for every feature bandwidth.
    pub min_bandwidth: f64,
}

impl Default for LocalLinearOutcome {
    fn default() -> Self {
        Self {
            ridge: 1e-8,
            bandwidth_scale: 1.0,
            min_bandwidth: 1e-3,
        }
    }
}

impl LocalLinearOutcome {
    /// Predict conditional means at `query_design` using local-linear
    /// regressions fit on `train_design`, `train_outcome`, and `train_weight`.
    ///
    /// The first column is assumed to be an intercept and is excluded from the
    /// kernel distance metric.
    ///
    /// # Errors
    /// Returns an error when the training arrays have inconsistent lengths or
    /// when the local linear system cannot be solved.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn predict_from_training(
        self,
        train_design: MatRef<'_, f64>,
        train_outcome: &[f64],
        train_weight: &[f64],
        query_design: MatRef<'_, f64>,
        scratch: &mut LocalLinearScratch,
    ) -> Result<Vec<f64>, &'static str> {
        if train_design.nrows() != train_outcome.len() || train_design.nrows() != train_weight.len()
        {
            return Err("training arrays must have matching row counts");
        }
        if train_design.ncols() != query_design.ncols() {
            return Err("training and query designs must have the same column count");
        }
        if train_design.nrows() == 0 {
            return Err("local linear outcome regression requires at least one training row");
        }

        let feature_count = train_design.ncols().saturating_sub(1);
        if feature_count == 0 {
            let denominator = train_weight.iter().sum::<f64>();
            if denominator <= 0.0 || !denominator.is_finite() {
                return Err("training weights must sum to a positive finite value");
            }
            let mean = train_outcome
                .iter()
                .zip(train_weight.iter())
                .map(|(y, w)| y * w)
                .sum::<f64>()
                / denominator;
            return Ok(vec![mean; query_design.nrows()]);
        }

        let local_param_count = feature_count + 1;
        prepare_local_linear_inputs(
            self,
            train_design,
            feature_count,
            local_param_count,
            scratch,
        );
        predict_query_rows_local_linear(
            self,
            train_design,
            train_outcome,
            train_weight,
            query_design,
            scratch,
            feature_count,
            local_param_count,
        )
    }

    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn refill_bandwidths(self, train_design: MatRef<'_, f64>, out: &mut Vec<f64>) {
        let row_count = train_design.nrows();
        let feature_count = train_design.ncols().saturating_sub(1);
        let sample_scale = crate::util::usize_to_f64(row_count)
            .powf(-1.0 / (crate::util::usize_to_f64(feature_count) + 4.0));
        out.clear();
        out.reserve(feature_count.saturating_sub(out.capacity()));

        for col in 1..train_design.ncols() {
            let mean = (0..row_count)
                .map(|row| *train_design.get(row, col))
                .sum::<f64>()
                / crate::util::usize_to_f64(row_count);
            let variance = (0..row_count)
                .map(|row| {
                    let centered = *train_design.get(row, col) - mean;
                    centered * centered
                })
                .sum::<f64>()
                / crate::util::usize_to_f64(row_count.max(1));
            let std_dev = variance.max(0.0).sqrt();
            let bandwidth =
                (self.bandwidth_scale * std_dev.max(1.0) * sample_scale).max(self.min_bandwidth);
            out.push(bandwidth);
        }
    }
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn prepare_local_linear_inputs(
    model: LocalLinearOutcome,
    train_design: MatRef<'_, f64>,
    feature_count: usize,
    local_param_count: usize,
    scratch: &mut LocalLinearScratch,
) {
    scratch.refill_covariates_flat(train_design, feature_count);
    model.refill_bandwidths(train_design, &mut scratch.bandwidths);
    scratch.ensure_solver(local_param_count);
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
#[allow(
    clippy::too_many_arguments,
    reason = "local-linear kernel prediction needs these inputs"
)]
fn predict_query_rows_local_linear(
    model: LocalLinearOutcome,
    train_design: MatRef<'_, f64>,
    train_outcome: &[f64],
    train_weight: &[f64],
    query_design: MatRef<'_, f64>,
    scratch: &mut LocalLinearScratch,
    feature_count: usize,
    local_param_count: usize,
) -> Result<Vec<f64>, &'static str> {
    let grouped_queries = build_query_covariate_groups(query_design, feature_count);
    let mut unique_predictions = Vec::with_capacity(grouped_queries.unique_covariates.len());
    let mut basis = vec![0.0; local_param_count];
    let mut normal_matrix = vec![0.0; local_param_count * local_param_count];
    let mut normal_rhs = vec![0.0; local_param_count];
    let mut spd_solution = vec![0.0; local_param_count];

    for query_covariates in &grouped_queries.unique_covariates {
        unique_predictions.push(predict_query_covariates(
            model,
            train_design,
            train_outcome,
            train_weight,
            scratch,
            query_covariates,
            &mut basis,
            &mut normal_matrix,
            &mut normal_rhs,
            &mut spd_solution,
        )?);
    }

    let mut predictions = Vec::with_capacity(query_design.nrows());
    for &unique_idx in &grouped_queries.row_to_unique {
        predictions.push(unique_predictions[unique_idx]);
    }

    Ok(predictions)
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn build_query_covariate_groups(
    query_design: MatRef<'_, f64>,
    feature_count: usize,
) -> QueryCovariateGroups {
    let mut key_to_unique = HashMap::<Vec<u64>, usize>::new();
    let mut unique_covariates = Vec::<Vec<f64>>::new();
    let mut row_to_unique = Vec::with_capacity(query_design.nrows());
    let mut query_covariates = vec![0.0; feature_count];
    let mut query_bits = vec![0_u64; feature_count];

    for query_row in 0..query_design.nrows() {
        fill_query_covariates(query_design, query_row, &mut query_covariates);
        for (bit, value) in query_bits.iter_mut().zip(query_covariates.iter().copied()) {
            *bit = normalized_f64_bits(value);
        }
        if let Some(&unique_idx) = key_to_unique.get(query_bits.as_slice()) {
            row_to_unique.push(unique_idx);
            continue;
        }

        let unique_idx = unique_covariates.len();
        key_to_unique.insert(query_bits.clone(), unique_idx);
        unique_covariates.push(query_covariates.clone());
        row_to_unique.push(unique_idx);
    }

    QueryCovariateGroups {
        unique_covariates,
        row_to_unique,
    }
}

struct QueryCovariateGroups {
    unique_covariates: Vec<Vec<f64>>,
    row_to_unique: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn predict_query_covariates(
    model: LocalLinearOutcome,
    train_design: MatRef<'_, f64>,
    train_outcome: &[f64],
    train_weight: &[f64],
    scratch: &mut LocalLinearScratch,
    query_covariates: &[f64],
    basis: &mut [f64],
    normal_matrix: &mut [f64],
    normal_rhs: &mut [f64],
    spd_solution: &mut [f64],
) -> Result<f64, &'static str> {
    let weight_sum = accumulate_local_linear_system(
        train_design,
        train_outcome,
        train_weight,
        &scratch.covariates_flat,
        &scratch.bandwidths,
        query_covariates,
        basis,
        normal_matrix,
        normal_rhs,
    );
    solve_local_linear_system(
        model,
        train_design,
        train_outcome,
        train_weight,
        &scratch.covariates_flat,
        &scratch.bandwidths,
        query_covariates,
        weight_sum,
        &mut scratch.solver,
        normal_matrix,
        normal_rhs,
        spd_solution,
    )
}

fn fill_query_covariates(
    query_design: MatRef<'_, f64>,
    query_row: usize,
    query_covariates: &mut [f64],
) {
    for (feature_idx, query_value) in query_covariates.iter_mut().enumerate() {
        *query_value = *query_design.get(query_row, feature_idx + 1);
    }
}

fn normalized_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
#[allow(
    clippy::too_many_arguments,
    reason = "normal-equation accumulation needs these inputs"
)]
fn accumulate_local_linear_system(
    train_design: MatRef<'_, f64>,
    train_outcome: &[f64],
    train_weight: &[f64],
    train_covariates_flat: &[f64],
    bandwidths: &[f64],
    query_covariates: &[f64],
    basis: &mut [f64],
    normal_matrix: &mut [f64],
    normal_rhs: &mut [f64],
) -> f64 {
    normal_matrix.fill(0.0);
    normal_rhs.fill(0.0);
    let local_param_count = basis.len();
    let feature_count = query_covariates.len();
    let mut weight_sum = 0.0;

    for train_row in 0..train_design.nrows() {
        let row_start = train_row * feature_count;
        let train_covariates = &train_covariates_flat[row_start..row_start + feature_count];
        let kernel_weight = gaussian_kernel_weight(train_covariates, query_covariates, bandwidths);
        let weight = train_weight[train_row] * kernel_weight;
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        weight_sum += weight;

        basis[0] = 1.0;
        for (feature_idx, query_value) in query_covariates.iter().enumerate() {
            basis[feature_idx + 1] = train_covariates[feature_idx] - *query_value;
        }

        for row in 0..local_param_count {
            normal_rhs[row] =
                (weight * basis[row]).mul_add(train_outcome[train_row], normal_rhs[row]);
            for col in 0..local_param_count {
                normal_matrix[row * local_param_count + col] = (weight * basis[row])
                    .mul_add(basis[col], normal_matrix[row * local_param_count + col]);
            }
        }
    }

    weight_sum
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn solve_local_linear_system(
    model: LocalLinearOutcome,
    train_design: MatRef<'_, f64>,
    train_outcome: &[f64],
    train_weight: &[f64],
    train_covariates_flat: &[f64],
    bandwidths: &[f64],
    query_covariates: &[f64],
    weight_sum: f64,
    solver: &mut SpdCholeskyScratch,
    normal_matrix: &mut [f64],
    normal_rhs: &[f64],
    spd_solution: &mut [f64],
) -> Result<f64, &'static str> {
    let local_param_count = query_covariates.len() + 1;
    if weight_sum <= 0.0 || !weight_sum.is_finite() {
        return Err("local linear fit produced zero effective kernel weight");
    }
    for diagonal in 0..local_param_count {
        normal_matrix[diagonal * local_param_count + diagonal] += model.ridge;
    }
    if solver
        .solve_single_rhs_into(normal_matrix, normal_rhs, spd_solution)
        .is_ok()
    {
        return Ok(spd_solution[0]);
    }
    match solve_spd_system(normal_matrix, normal_rhs) {
        Ok(solution) => Ok(solution[0]),
        Err(_) if model.ridge > 0.0 && model.ridge <= 1e-6 => {
            let solution = solve_local_qr_lstsq(
                train_design,
                train_covariates_flat,
                train_outcome,
                train_weight,
                query_covariates,
                bandwidths,
                local_param_count,
            )?;
            Ok(solution[0])
        }
        Err(err) => Err(err),
    }
}

/// Reusable local-linear work buffers for repeated prediction calls.
pub struct LocalLinearScratch {
    covariates_flat: Vec<f64>,
    bandwidths: Vec<f64>,
    solver: SpdCholeskyScratch,
    solver_dim: usize,
}

impl LocalLinearScratch {
    #[must_use]
    pub fn new(feature_count: usize) -> Self {
        let solver_dim = feature_count + 1;
        Self {
            covariates_flat: Vec::new(),
            bandwidths: Vec::with_capacity(feature_count),
            solver: SpdCholeskyScratch::new(solver_dim),
            solver_dim,
        }
    }

    fn ensure_solver(&mut self, local_param_count: usize) {
        if self.solver_dim != local_param_count {
            self.solver = SpdCholeskyScratch::new(local_param_count);
            self.solver_dim = local_param_count;
        }
    }

    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn refill_covariates_flat(&mut self, design: MatRef<'_, f64>, feature_count: usize) {
        self.covariates_flat.clear();
        let needed = design.nrows() * feature_count;
        self.covariates_flat
            .reserve(needed.saturating_sub(self.covariates_flat.capacity()));
        for row in 0..design.nrows() {
            for feature_idx in 0..feature_count {
                self.covariates_flat.push(*design.get(row, feature_idx + 1));
            }
        }
    }
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn solve_local_qr_lstsq(
    train_design: MatRef<'_, f64>,
    train_covariates_flat: &[f64],
    train_outcome: &[f64],
    train_weight: &[f64],
    query_covariates: &[f64],
    bandwidths: &[f64],
    local_param_count: usize,
) -> Result<Vec<f64>, &'static str> {
    let mut weighted_design_flat = Vec::with_capacity(train_design.nrows() * local_param_count);
    let mut weighted_outcome = Vec::with_capacity(train_design.nrows());

    for train_row in 0..train_design.nrows() {
        let row_start = train_row * query_covariates.len();
        let train_covariates =
            &train_covariates_flat[row_start..row_start + query_covariates.len()];
        let kernel_weight = gaussian_kernel_weight(train_covariates, query_covariates, bandwidths);
        let weight = train_weight[train_row] * kernel_weight;
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        let sqrt_weight = weight.sqrt();
        weighted_design_flat.push(sqrt_weight);
        for (train_value, query_value) in train_covariates.iter().zip(query_covariates.iter()) {
            weighted_design_flat.push(sqrt_weight * (train_value - *query_value));
        }
        weighted_outcome.push(sqrt_weight * train_outcome[train_row]);
    }

    let row_count = weighted_outcome.len();
    if row_count < local_param_count {
        return Err("local linear system cannot be solved");
    }

    let weighted_design = Mat::from_fn(row_count, local_param_count, |row, col| {
        weighted_design_flat[row * local_param_count + col]
    });
    let weighted_outcome_mat = Mat::from_fn(row_count, 1, |row, _| weighted_outcome[row]);
    let solution = weighted_design
        .col_piv_qr()
        .solve_lstsq(&weighted_outcome_mat);

    let mut out = vec![0.0; local_param_count];
    for row in 0..local_param_count {
        let value = solution[(row, 0)];
        if !value.is_finite() {
            return Err("local linear system cannot be solved");
        }
        out[row] = value;
    }
    Ok(out)
}

fn gaussian_kernel_weight(
    train_covariates: &[f64],
    query_covariates: &[f64],
    bandwidths: &[f64],
) -> f64 {
    train_covariates
        .iter()
        .zip(query_covariates.iter())
        .zip(bandwidths.iter())
        .map(|((train_value, query_value), bandwidth)| {
            let distance = (train_value - query_value) / bandwidth.max(f64::EPSILON);
            (-0.5 * distance * distance).exp()
        })
        .product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_linear_recovers_smooth_signal() {
        let train_design = Mat::from_fn(5, 2, |row, col| match (row, col) {
            (0, 1) => -1.0,
            (1, 1) => -0.5,
            (2, 1) => 0.0,
            (3, 1) => 0.5,
            (_, 0) | (4, 1) => 1.0,
            _ => unreachable!(),
        });
        let train_outcome = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let train_weight = vec![1.0; 5];
        let query_design = Mat::from_fn(3, 2, |row, col| match (row, col) {
            (_, 0) => 1.0,
            (0, 1) => -0.25,
            (1, 1) => 0.25,
            (2, 1) => 0.75,
            _ => unreachable!(),
        });

        let estimator = LocalLinearOutcome::default();
        let mut scratch = LocalLinearScratch::new(1);
        let predictions = estimator
            .predict_from_training(
                train_design.as_ref(),
                &train_outcome,
                &train_weight,
                query_design.as_ref(),
                &mut scratch,
            )
            .expect("predict local-linear");

        assert!((predictions[0] - 0.75).abs() < 0.25);
        assert!((predictions[1] - 1.25).abs() < 0.25);
        assert!((predictions[2] - 1.75).abs() < 0.25);
    }

    #[test]
    fn local_linear_reduces_to_weighted_mean_without_covariates() {
        let train_design = Mat::from_fn(3, 1, |_, _| 1.0);
        let train_outcome = vec![1.0, 2.0, 5.0];
        let train_weight = vec![1.0, 2.0, 1.0];
        let query_design = Mat::from_fn(2, 1, |_, _| 1.0);

        let estimator = LocalLinearOutcome::default();
        let mut scratch = LocalLinearScratch::new(0);
        let predictions = estimator
            .predict_from_training(
                train_design.as_ref(),
                &train_outcome,
                &train_weight,
                query_design.as_ref(),
                &mut scratch,
            )
            .expect("predict weighted mean");

        assert_eq!(predictions.len(), 2);
        assert!((predictions[0] - 2.5).abs() < 1e-12);
        assert!((predictions[1] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn local_linear_handles_duplicate_query_rows() {
        let train_design = Mat::from_fn(5, 2, |row, col| match (row, col) {
            (_, 0) | (4, 1) => 1.0,
            (0, 1) => -1.0,
            (1, 1) => -0.5,
            (2, 1) => 0.0,
            (3, 1) => 0.5,
            _ => unreachable!(),
        });
        let train_outcome = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let train_weight = vec![1.0; 5];
        let query_design = Mat::from_fn(5, 2, |row, col| match (row, col) {
            (_, 0) => 1.0,
            (0 | 1, 1) => -0.25,
            (2 | 4, 1) => 0.25,
            (3, 1) => 0.75,
            _ => unreachable!(),
        });

        let estimator = LocalLinearOutcome::default();
        let mut scratch = LocalLinearScratch::new(1);
        let predictions = estimator
            .predict_from_training(
                train_design.as_ref(),
                &train_outcome,
                &train_weight,
                query_design.as_ref(),
                &mut scratch,
            )
            .expect("predict local-linear with duplicates");

        assert_eq!(predictions.len(), 5);
        assert!((predictions[0] - predictions[1]).abs() < 1e-12);
        assert!((predictions[2] - predictions[4]).abs() < 1e-12);
        assert!((predictions[0] - 0.75).abs() < 0.25);
        assert!((predictions[2] - 1.25).abs() < 0.25);
        assert!((predictions[3] - 1.75).abs() < 0.25);
    }
}
