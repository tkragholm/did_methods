//! Reusable solver workspace for `DeltaRM` identified-set solves.
//!
//! The matrix and cone geometry are fixed within a `(s, max_positive)` branch,
//! so identified-set computation only needs to swap the linear objective
//! between maximization and minimization of the target functional.

use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT;

use crate::inference::sensitivity::SolverCounter;
use crate::inference::sensitivity::conic::ConicWorkspace;

pub(super) struct RelativeMagnitudeIdentifiedSetWorkspace {
    program: ConicWorkspace,
    base_objective: Vec<f64>,
}

impl RelativeMagnitudeIdentifiedSetWorkspace {
    pub(super) fn new(
        quadratic: &CscMatrix<f64>,
        constraint_matrix: &CscMatrix<f64>,
        rhs: &[f64],
        cones: &[SupportedConeT<f64>],
        objective: &[f64],
    ) -> Result<Self, String> {
        Ok(Self {
            program: ConicWorkspace::new(
                "relative-magnitude LP",
                quadratic,
                constraint_matrix,
                rhs,
                cones,
                objective,
            )?,
            base_objective: objective.to_vec(),
        })
    }

    pub(super) fn solve_with_q(
        &mut self,
        q: &[f64],
        counter: &SolverCounter,
    ) -> Result<Option<f64>, String> {
        Ok(self.program.solve_with_q(q, counter)?.map(|solution| {
            self.base_objective
                .iter()
                .zip(solution)
                .map(|(left, right)| left * right)
                .sum::<f64>()
        }))
    }
}
