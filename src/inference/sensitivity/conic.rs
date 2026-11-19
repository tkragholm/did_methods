//! The one place Clarabel is called for the sensitivity programs.
//!
//! A program is `min q'x + x'Px / 2` subject to `Ax + s = b`, `s` in a product
//! of cones. [`ConicWorkspace`] solves it for a sequence of objectives `q`
//! with the same constraints.
//!
//! Every solve starts with Clarabel's defaults, with presolve and zero-dropping
//! off. A `NumericalError` or `InsufficientProgress` is the factorisation
//! losing accuracy on a feasible but thin cone, not infeasibility, so the
//! program is solved once more by a new solver with a static regularisation of
//! `1e-6`, a dynamic one of `1e-12`, 1,000 iterations and tolerances of `1e-6`.
//! The workspace keeps the retry solver, so a program that needed it once is
//! solved the same way for its next objective. A program solved under the
//! retry is accurate to `1e-6` and not to Clarabel's default `1e-8`. Every
//! solve is reported to the [`SolverCounter`], which counts the retries and the
//! programs that end `AlmostSolved`.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
};

use super::SolverCounter;

fn default_settings() -> Result<DefaultSettings<f64>, String> {
    DefaultSettingsBuilder::<f64>::default()
        .verbose(false)
        .presolve_enable(false)
        .input_sparse_dropzeros(false)
        .build()
        .map_err(|err| format!("failed to build Clarabel settings: {err}"))
}

fn conservative_settings() -> Result<DefaultSettings<f64>, String> {
    DefaultSettingsBuilder::<f64>::default()
        .verbose(false)
        .presolve_enable(false)
        .input_sparse_dropzeros(false)
        .static_regularization_constant(1e-6)
        .dynamic_regularization_eps(1e-12)
        .max_iter(1000)
        .tol_gap_abs(1e-6)
        .tol_gap_rel(1e-6)
        .tol_feas(1e-6)
        .build()
        .map_err(|err| format!("failed to build Clarabel retry settings: {err}"))
}

/// A program with fixed constraints whose objective can be replaced.
pub(super) struct ConicWorkspace {
    solver: DefaultSolver<f64>,
    quadratic: CscMatrix<f64>,
    constraint_matrix: CscMatrix<f64>,
    rhs: Vec<f64>,
    cones: Vec<SupportedConeT<f64>>,
    q: Vec<f64>,
    what: &'static str,
}

impl ConicWorkspace {
    /// `what` names the program in error messages.
    pub(super) fn new(
        what: &'static str,
        quadratic: &CscMatrix<f64>,
        constraint_matrix: &CscMatrix<f64>,
        rhs: &[f64],
        cones: &[SupportedConeT<f64>],
        q: &[f64],
    ) -> Result<Self, String> {
        let solver = DefaultSolver::new(
            quadratic,
            q,
            constraint_matrix,
            rhs,
            cones,
            default_settings()?,
        )
        .map_err(|err| format!("failed to initialize {what} solver: {err}"))?;
        Ok(Self {
            solver,
            quadratic: quadratic.clone(),
            constraint_matrix: constraint_matrix.clone(),
            rhs: rhs.to_vec(),
            cones: cones.to_vec(),
            q: q.to_vec(),
            what,
        })
    }

    /// Solves for the objective `q`. `Ok(None)` is an infeasible or unbounded
    /// program, `Ok(Some(x))` its solution.
    pub(super) fn solve_with_q(
        &mut self,
        q: &[f64],
        counter: &SolverCounter,
    ) -> Result<Option<&[f64]>, String> {
        self.q.clear();
        self.q.extend_from_slice(q);
        let what = self.what;
        self.solver
            .update_q(&self.q)
            .map_err(|err| format!("failed to update {what} objective: {err}"))?;
        self.solver.solve();
        let status = self.solver.solution.status;
        counter.note_status(status);
        if matches!(
            status,
            SolverStatus::NumericalError | SolverStatus::InsufficientProgress
        ) {
            counter.note_retry();
            self.solver = DefaultSolver::new(
                &self.quadratic,
                &self.q,
                &self.constraint_matrix,
                &self.rhs,
                &self.cones,
                conservative_settings()?,
            )
            .map_err(|err| format!("failed to initialize {what} retry solver: {err}"))?;
            self.solver.solve();
            counter.note_status(self.solver.solution.status);
        }
        match self.solver.solution.status {
            SolverStatus::Solved | SolverStatus::AlmostSolved => {
                Ok(Some(self.solver.solution.x.as_slice()))
            }
            SolverStatus::PrimalInfeasible
            | SolverStatus::DualInfeasible
            | SolverStatus::AlmostPrimalInfeasible
            | SolverStatus::AlmostDualInfeasible => Ok(None),
            status => Err(format!("Clarabel failed to solve {what}: {status:?}")),
        }
    }
}

/// Solves one program.
pub(super) fn solve(
    what: &'static str,
    quadratic: &CscMatrix<f64>,
    q: &[f64],
    constraint_matrix: &CscMatrix<f64>,
    rhs: &[f64],
    cones: &[SupportedConeT<f64>],
    counter: &SolverCounter,
) -> Result<Option<Vec<f64>>, String> {
    let mut workspace = ConicWorkspace::new(what, quadratic, constraint_matrix, rhs, cones, q)?;
    Ok(workspace.solve_with_q(q, counter)?.map(<[f64]>::to_vec))
}
