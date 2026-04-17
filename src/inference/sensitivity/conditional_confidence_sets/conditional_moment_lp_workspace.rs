#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

//! Reusable ARP auxiliary LP workspace for `HonestDiD` conditional inference.
//!
//! This workspace is shared infrastructure across sensitivity families. It is
//! the generic conditional-moment program solved repeatedly during:
//!
//! - ARP test inversion
//! - least-favorable critical-value simulation
//! - hybrid conditional confidence-set construction
//!
//! The family-specific geometry enters through the supplied `x_matrix` and
//! covariance surface; the LP itself is otherwise generic.

use highs::{ColProblem, HighsModelStatus, Model, Sense};
use highs_sys::{HighsInt, STATUS_ERROR, STATUS_OK, STATUS_WARNING};

use super::super::linear_algebra::diag_sqrt;

/// Reusable workspace for the ARP auxiliary LP.
///
/// The primal variables are
/// `(\eta^+, \eta^-, \delta^+, \delta^-)`,
/// where `\eta = \eta^+ - \eta^-` and `\delta = \delta^+ - \delta^-`.
/// Solving this program yields the ARP statistic `\eta^*`, the nuisance
/// regression coefficients `\delta^*`, and—when available—the nonnegative dual
/// multipliers used in the conditional acceptance test.
pub(in crate::inference::sensitivity) struct ConditionalMomentLpWorkspace {
    model: Option<Model>,
    k: usize,
    /// Normalization scale applied to the constraint matrix and rhs.
    ///
    /// The ARP LP is built with `A_scaled = A / scale` and rhs `b_scaled = b /
    /// scale`, which is an equivalent LP (same primal optimum). This brings
    /// large-magnitude outcomes to O(1) scale so that solver iterations do not
    /// encounter avoidable numerical issues.
    ///
    /// After solving, the primal solution (η*, δ*) is identical to the
    /// unscaled optimum. The row duals are defined on the scaled system, so we
    /// divide them back by `scale` when storing `lambda` to restore the
    /// original dual geometry expected by the conditional test.
    scale: f64,
    row_indices: Vec<HighsInt>,
    row_lower_bounds: Vec<f64>,
    rhs_scratch: Vec<f64>,
    eta_star: f64,
    delta_star: Vec<f64>,
    lambda: Vec<f64>,
    column_values: Vec<f64>,
    row_values: Vec<f64>,
    column_duals: Vec<f64>,
    row_duals: Vec<f64>,
    column_basis_status: Vec<HighsInt>,
    row_basis_status: Vec<HighsInt>,
    has_solution: bool,
    has_basis: bool,
}

impl ConditionalMomentLpWorkspace {
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub(in crate::inference::sensitivity) fn new(
        x_matrix: &[Vec<f64>],
        sigma: &[Vec<f64>],
    ) -> Result<Self, String> {
        let sd_vec = diag_sqrt(sigma);
        let scale = sd_vec.iter().copied().fold(0.0_f64, f64::max).max(1.0);
        let k = x_matrix.first().map_or(0, Vec::len);
        let num_vars = 2 + 2 * k;

        let mut problem = ColProblem::new();
        let mut rows = Vec::with_capacity(x_matrix.len());
        let row_indices = (0..x_matrix.len())
            .map(|row_idx| {
                rows.push(problem.add_row(..=0.0));
                row_idx.try_into().map_err(|_| {
                    format!("HonestDiD eta/delta LP has too many rows for HiGHS: {row_idx}")
                })
            })
            .collect::<Result<Vec<HighsInt>, String>>()?;

        let eta_plus_factors = sd_vec
            .iter()
            .enumerate()
            .map(|(row_idx, sd)| (rows[row_idx], -*sd / scale))
            .collect::<Vec<_>>();
        problem.add_column(1.0, 0.0.., eta_plus_factors);

        let eta_minus_factors = sd_vec
            .iter()
            .enumerate()
            .map(|(row_idx, sd)| (rows[row_idx], *sd / scale))
            .collect::<Vec<_>>();
        problem.add_column(-1.0, 0.0.., eta_minus_factors);

        for col_idx in 0..k {
            let delta_plus_factors = x_matrix
                .iter()
                .enumerate()
                .map(|(row_idx, row)| (rows[row_idx], -row[col_idx] / scale))
                .collect::<Vec<_>>();
            problem.add_column(0.0, 0.0.., delta_plus_factors);
        }

        for col_idx in 0..k {
            let delta_minus_factors = x_matrix
                .iter()
                .enumerate()
                .map(|(row_idx, row)| (rows[row_idx], row[col_idx] / scale))
                .collect::<Vec<_>>();
            problem.add_column(0.0, 0.0.., delta_minus_factors);
        }

        let mut model = problem.optimise(Sense::Minimise);
        model.make_quiet();
        model.set_option("solver", "simplex");
        model.set_option("presolve", "off");

        Ok(Self {
            model: Some(model),
            k,
            scale,
            row_indices,
            row_lower_bounds: vec![f64::NEG_INFINITY; x_matrix.len()],
            rhs_scratch: vec![0.0; x_matrix.len()],
            eta_star: 0.0,
            delta_star: vec![0.0; k],
            lambda: vec![0.0; x_matrix.len()],
            column_values: vec![0.0; num_vars],
            row_values: vec![0.0; x_matrix.len()],
            column_duals: vec![0.0; num_vars],
            row_duals: vec![0.0; x_matrix.len()],
            column_basis_status: vec![0; num_vars],
            row_basis_status: vec![0; x_matrix.len()],
            has_solution: false,
            has_basis: false,
        })
    }

    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub(in crate::inference::sensitivity) fn solve_in_place(
        &mut self,
        y_vec: &[f64],
    ) -> Result<(), String> {
        if self.rhs_scratch.len() != y_vec.len() {
            return Err(format!(
                "HonestDiD eta/delta LP rhs length mismatch: expected {}, got {}",
                self.rhs_scratch.len(),
                y_vec.len()
            ));
        }

        for (dst, y) in self.rhs_scratch.iter_mut().zip(y_vec.iter()) {
            *dst = -y / self.scale;
        }

        update_row_upper_bounds(
            self.model
                .as_mut()
                .ok_or_else(|| "HonestDiD eta/delta LP model is unavailable".to_string())?,
            &self.row_indices,
            &self.row_lower_bounds,
            &self.rhs_scratch,
        )?;

        let model = self
            .model
            .as_mut()
            .ok_or_else(|| "HonestDiD eta/delta LP model is unavailable".to_string())?;
        if self.has_basis {
            set_basis(model, &self.column_basis_status, &self.row_basis_status)?;
        } else if self.has_solution {
            model.set_solution(
                Some(&self.column_values),
                Some(&self.row_values),
                Some(&self.column_duals),
                Some(&self.row_duals),
            );
        }
        let status = solve_highs_model(model)?;
        match status {
            HighsModelStatus::Optimal
            | HighsModelStatus::ObjectiveBound
            | HighsModelStatus::ObjectiveTarget => {}
            status => {
                return Err(format!(
                    "failed to solve HonestDiD eta/delta LP: {status:?}"
                ));
            }
        }

        populate_solution_buffers(
            model,
            &mut self.column_values,
            &mut self.row_values,
            &mut self.column_duals,
            &mut self.row_duals,
        )?;
        populate_basis_buffers(
            model,
            &mut self.column_basis_status,
            &mut self.row_basis_status,
        )?;
        self.has_solution = true;
        self.has_basis = true;

        self.eta_star = self.column_values[0] - self.column_values[1];
        self.delta_star.clear();
        self.delta_star.extend((0..self.k).map(|col_idx| {
            self.column_values[2 + col_idx] - self.column_values[2 + self.k + col_idx]
        }));

        let inv_scale = 1.0 / self.scale;
        self.lambda.clear();
        self.lambda.extend(
            self.row_duals
                .iter()
                .map(|value| (-*value).max(0.0) * inv_scale),
        );

        Ok(())
    }

    pub(in crate::inference::sensitivity) const fn eta_star(&self) -> f64 {
        self.eta_star
    }

    pub(in crate::inference::sensitivity) fn delta_star(&self) -> &[f64] {
        &self.delta_star
    }

    pub(in crate::inference::sensitivity) fn lambda(&self) -> &[f64] {
        &self.lambda
    }
}

fn update_row_upper_bounds(
    model: &mut Model,
    row_indices: &[HighsInt],
    row_lower_bounds: &[f64],
    row_upper_bounds: &[f64],
) -> Result<(), String> {
    let row_count: HighsInt = row_indices
        .len()
        .try_into()
        .map_err(|_| "HonestDiD eta/delta LP has too many rows for HiGHS".to_string())?;
    let status = unsafe {
        highs_sys::Highs_changeRowsBoundsBySet(
            model.as_mut_ptr(),
            row_count,
            row_indices.as_ptr(),
            row_lower_bounds.as_ptr(),
            row_upper_bounds.as_ptr(),
        )
    };
    try_highs_status(status, "update HonestDiD eta/delta LP rhs")
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn solve_highs_model(model: &mut Model) -> Result<HighsModelStatus, String> {
    let status = unsafe { highs_sys::Highs_run(model.as_mut_ptr()) };
    try_highs_status(status, "solve HonestDiD eta/delta LP")?;
    let model_status = unsafe { highs_sys::Highs_getModelStatus(model.as_mut_ptr()) };
    HighsModelStatus::try_from(model_status)
        .map_err(|_| format!("failed to read HonestDiD eta/delta LP model status: {model_status}"))
}

fn populate_solution_buffers(
    model: &mut Model,
    column_values: &mut [f64],
    row_values: &mut [f64],
    column_duals: &mut [f64],
    row_duals: &mut [f64],
) -> Result<(), String> {
    let status = unsafe {
        highs_sys::Highs_getSolution(
            model.as_mut_ptr(),
            column_values.as_mut_ptr(),
            column_duals.as_mut_ptr(),
            row_values.as_mut_ptr(),
            row_duals.as_mut_ptr(),
        )
    };
    try_highs_status(status, "read HonestDiD eta/delta LP solution")
}

fn set_basis(
    model: &mut Model,
    column_basis_status: &[HighsInt],
    row_basis_status: &[HighsInt],
) -> Result<(), String> {
    let status = unsafe {
        highs_sys::Highs_setBasis(
            model.as_mut_ptr(),
            column_basis_status.as_ptr(),
            row_basis_status.as_ptr(),
        )
    };
    try_highs_status(status, "apply HonestDiD eta/delta LP basis")
}

fn populate_basis_buffers(
    model: &mut Model,
    column_basis_status: &mut [HighsInt],
    row_basis_status: &mut [HighsInt],
) -> Result<(), String> {
    let status = unsafe {
        highs_sys::Highs_getBasis(
            model.as_mut_ptr(),
            column_basis_status.as_mut_ptr(),
            row_basis_status.as_mut_ptr(),
        )
    };
    try_highs_status(status, "read HonestDiD eta/delta LP basis")
}

fn try_highs_status(status: HighsInt, context: &str) -> Result<(), String> {
    match status {
        STATUS_OK | STATUS_WARNING => Ok(()),
        STATUS_ERROR => Err(format!("failed to {context}: HiGHS returned STATUS_ERROR")),
        other => Err(format!(
            "failed to {context}: HiGHS returned unexpected status {other}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use clarabel::algebra::CscMatrix;
    use clarabel::solver::{
        DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
    };

    use super::super::super::linear_algebra::{dense_rows_to_csc, diag_sqrt};
    use super::ConditionalMomentLpWorkspace;

    struct ClarabelReferenceResult {
        eta_star: f64,
        delta_star: Vec<f64>,
        lambda: Vec<f64>,
    }

    fn solve_with_clarabel_reference(
        x_matrix: &[Vec<f64>],
        sigma: &[Vec<f64>],
        y_vec: &[f64],
    ) -> ClarabelReferenceResult {
        let sd_vec = diag_sqrt(sigma);
        let scale = sd_vec.iter().copied().fold(0.0_f64, f64::max).max(1.0);
        let k = x_matrix.first().map_or(0, Vec::len);
        let num_vars = 2 + 2 * k;
        let constraint_rows: Vec<Vec<f64>> = x_matrix
            .iter()
            .zip(&sd_vec)
            .map(|(x_row, sd)| {
                let mut row = vec![0.0; num_vars];
                row[0] = -sd / scale;
                row[1] = sd / scale;
                for (col_idx, x_value) in x_row.iter().enumerate() {
                    row[2 + col_idx] = -x_value / scale;
                    row[2 + k + col_idx] = x_value / scale;
                }
                row
            })
            .collect();
        let constraint_matrix = dense_rows_to_csc(&constraint_rows);
        let rhs = y_vec.iter().map(|value| -value / scale).collect::<Vec<_>>();
        let cones = vec![SupportedConeT::NonnegativeConeT(constraint_rows.len())];
        let quadratic = CscMatrix::<f64>::zeros((num_vars, num_vars));
        let mut q = vec![0.0; num_vars];
        q[0] = 1.0;
        q[1] = -1.0;
        let settings = DefaultSettingsBuilder::<f64>::default()
            .verbose(false)
            .presolve_enable(false)
            .input_sparse_dropzeros(false)
            .build()
            .expect("Clarabel settings");
        let mut solver =
            DefaultSolver::new(&quadratic, &q, &constraint_matrix, &rhs, &cones, settings)
                .expect("Clarabel workspace");
        solver.solve();
        assert!(matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ));
        ClarabelReferenceResult {
            eta_star: solver.solution.x[0] - solver.solution.x[1],
            delta_star: (0..k)
                .map(|col_idx| solver.solution.x[2 + col_idx] - solver.solution.x[2 + k + col_idx])
                .collect(),
            lambda: solver
                .solution
                .z
                .iter()
                .map(|value| value.max(0.0) / scale)
                .collect(),
        }
    }

    #[test]
    fn highs_conditional_workspace_matches_clarabel_reference() {
        let x_matrix = vec![Vec::new(), Vec::new(), Vec::new()];
        let sigma = vec![
            vec![1.2, 0.1, 0.05],
            vec![0.1, 0.9, 0.08],
            vec![0.05, 0.08, 1.1],
        ];
        let y_vec = vec![0.7, 0.3, 0.45];

        let reference = solve_with_clarabel_reference(&x_matrix, &sigma, &y_vec);
        let mut workspace = ConditionalMomentLpWorkspace::new(&x_matrix, &sigma).unwrap();
        workspace.solve_in_place(&y_vec).unwrap();

        assert!((workspace.eta_star() - reference.eta_star).abs() < 1e-8);
        for (observed, expected) in workspace
            .delta_star()
            .iter()
            .zip(reference.delta_star.iter())
        {
            assert!((observed - expected).abs() < 1e-8);
        }
        for (observed, expected) in workspace.lambda().iter().zip(reference.lambda.iter()) {
            assert!((observed - expected).abs() < 1e-7);
        }
    }
}
