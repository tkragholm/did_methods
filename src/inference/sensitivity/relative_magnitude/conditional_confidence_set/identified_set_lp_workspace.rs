//! Reusable Clarabel workspaces for `DeltaRM` identified-set solves.
//!
//! The generic ARP auxiliary workspace now lives under the shared
//! `sensitivity::conditional` module. This module retains only the
//! branch-specific identified-set LP workspace used by the `DeltaRM` geometry.
//!
//! The numeric casts allowed in this file are limited to solver sizing and
//! indexing-adjacent code; they do not change the statistical formulas.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
};

/// Reusable workspace for a branch-specific identified-set LP.
///
/// The matrix and cone geometry are fixed within a `(s, max_positive)` branch,
/// so identified-set computation only needs to swap the linear objective
/// between maximization and minimization of the target functional.
pub(super) struct RelativeMagnitudeIdentifiedSetWorkspace {
    solver: DefaultSolver<f64>,
    base_objective: Vec<f64>,
    current_q: Vec<f64>,
}

impl RelativeMagnitudeIdentifiedSetWorkspace {
    pub(super) fn new(
        quadratic: &CscMatrix<f64>,
        constraint_matrix: &CscMatrix<f64>,
        rhs: &[f64],
        cones: &[SupportedConeT<f64>],
        objective: &[f64],
    ) -> Result<Self, String> {
        let settings = DefaultSettingsBuilder::<f64>::default()
            .verbose(false)
            .presolve_enable(false)
            .input_sparse_dropzeros(false)
            .build()
            .map_err(|err| format!("failed to build Clarabel settings: {err}"))?;
        let solver = DefaultSolver::new(
            quadratic,
            objective,
            constraint_matrix,
            rhs,
            cones,
            settings,
        )
        .map_err(|err| format!("failed to initialize Clarabel solver: {err}"))?;
        Ok(Self {
            solver,
            base_objective: objective.to_vec(),
            current_q: objective.to_vec(),
        })
    }

    pub(super) fn solve_with_q(&mut self, q: &[f64]) -> Result<Option<f64>, String> {
        self.current_q.clear();
        self.current_q.extend_from_slice(q);
        self.solver
            .update_q(&self.current_q)
            .map_err(|err| format!("failed to update relative-magnitude LP objective: {err}"))?;
        self.solver.solve();
        match self.solver.solution.status {
            SolverStatus::Solved | SolverStatus::AlmostSolved => Ok(Some(
                self.base_objective
                    .iter()
                    .zip(self.solver.solution.x.iter())
                    .map(|(left, right)| left * right)
                    .sum::<f64>(),
            )),
            SolverStatus::PrimalInfeasible
            | SolverStatus::DualInfeasible
            | SolverStatus::AlmostPrimalInfeasible
            | SolverStatus::AlmostDualInfeasible => Ok(None),
            status => Err(format!(
                "Clarabel failed to solve relative-magnitude LP: {status:?}"
            )),
        }
    }
}
