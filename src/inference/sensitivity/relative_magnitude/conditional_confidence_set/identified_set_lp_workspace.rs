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
    /// What a second solver needs if the first one stops on a numerical
    /// error: the problem, kept so the retry can rebuild rather than reuse a
    /// factorisation that has already gone wrong.
    problem: IdentifiedSetProblem,
}

struct IdentifiedSetProblem {
    quadratic: CscMatrix<f64>,
    constraint_matrix: CscMatrix<f64>,
    rhs: Vec<f64>,
    cones: Vec<SupportedConeT<f64>>,
}

/// The solver's own defaults, which is how every parity fixture was solved.
fn default_settings() -> Result<clarabel::solver::DefaultSettings<f64>, String> {
    DefaultSettingsBuilder::<f64>::default()
        .verbose(false)
        .presolve_enable(false)
        .input_sparse_dropzeros(false)
        .build()
        .map_err(|err| format!("failed to build Clarabel settings: {err}"))
}

/// The settings for a retry after `NumericalError` or `InsufficientProgress`.
///
/// Those two statuses are the KKT factorisation losing accuracy, not the
/// program being infeasible, and on this LP they turned up once on Study I's
/// first 0.18.1 server run, on one anchor of one slice. Presolve removes the
/// redundant rows the branch geometry carries, a larger static regularisation
/// keeps the factorisation away from the degenerate face, and a longer
/// iteration budget with tolerances one order looser lets an interior point
/// method finish on a vertex it was already close to. The identified set is
/// then reported to `1e-7` rather than `1e-8`, which is far inside the grid
/// step anything reads it at.
fn retry_settings() -> Result<clarabel::solver::DefaultSettings<f64>, String> {
    DefaultSettingsBuilder::<f64>::default()
        .verbose(false)
        .presolve_enable(true)
        .input_sparse_dropzeros(false)
        .static_regularization_constant(1e-7)
        .max_iter(500)
        .tol_gap_abs(1e-7)
        .tol_gap_rel(1e-7)
        .tol_feas(1e-7)
        .build()
        .map_err(|err| format!("failed to build Clarabel retry settings: {err}"))
}

impl RelativeMagnitudeIdentifiedSetWorkspace {
    pub(super) fn new(
        quadratic: &CscMatrix<f64>,
        constraint_matrix: &CscMatrix<f64>,
        rhs: &[f64],
        cones: &[SupportedConeT<f64>],
        objective: &[f64],
    ) -> Result<Self, String> {
        let solver = DefaultSolver::new(
            quadratic,
            objective,
            constraint_matrix,
            rhs,
            cones,
            default_settings()?,
        )
        .map_err(|err| format!("failed to initialize Clarabel solver: {err}"))?;
        Ok(Self {
            solver,
            base_objective: objective.to_vec(),
            current_q: objective.to_vec(),
            problem: IdentifiedSetProblem {
                quadratic: quadratic.clone(),
                constraint_matrix: constraint_matrix.clone(),
                rhs: rhs.to_vec(),
                cones: cones.to_vec(),
            },
        })
    }

    pub(super) fn solve_with_q(&mut self, q: &[f64]) -> Result<Option<f64>, String> {
        self.current_q.clear();
        self.current_q.extend_from_slice(q);
        self.solver
            .update_q(&self.current_q)
            .map_err(|err| format!("failed to update relative-magnitude LP objective: {err}"))?;
        self.solver.solve();
        let status = self.solver.solution.status;
        if matches!(
            status,
            SolverStatus::NumericalError | SolverStatus::InsufficientProgress
        ) {
            // A fresh solver on the same problem, under `retry_settings`. The
            // workspace keeps it, so a branch that needed the retry once solves
            // its second objective the same way.
            self.solver = DefaultSolver::new(
                &self.problem.quadratic,
                &self.current_q,
                &self.problem.constraint_matrix,
                &self.problem.rhs,
                &self.problem.cones,
                retry_settings()?,
            )
            .map_err(|err| format!("failed to initialize Clarabel retry solver: {err}"))?;
            self.solver.solve();
        }
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
