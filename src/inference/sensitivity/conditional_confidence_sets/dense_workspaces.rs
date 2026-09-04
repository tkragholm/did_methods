//! The two LP workspaces on the dense simplex instead of HiGHS.
//!
//! Same contracts as the HiGHS-backed structs of the same names, so the
//! consumers do not know which one they have. Selected by the `dense-lp`
//! feature.

use super::super::linear_algebra::diag_sqrt;

/// Optimal bases each workspace keeps as starting points; see
/// [`DenseSimplex::set_memory`].
const BASIS_MEMORY: usize = 32;
use super::dense_simplex::{DenseSimplex, LpError};
use super::dual_geometry::solve_dual_max_with_clarabel_fallback;

/// Reusable workspace for the ARP auxiliary LP,
/// `min η  s.t.  sd_i η + x_i'δ ≥ y_i`, with `(η, δ)` free.
///
/// Solved in the multiplier space: the columns of the simplex are the scaled
/// rows `(sd_i, x_i) / scale`, its costs are `y / scale`, and its equality
/// right-hand side is `e_0`. Its duals are `(η*, δ*)` and its `λ` are the
/// nonnegative row multipliers the conditional test reads. Only the costs
/// change between solves, so every solve after the first starts from the
/// previous basis.
#[derive(Clone)]
pub(in crate::inference::sensitivity) struct ConditionalMomentLpWorkspace {
    lp: DenseSimplex,
    k: usize,
    /// The constraint rows and rhs are divided by the largest standard error,
    /// capped below at one, so the program is solved at unit scale whatever
    /// the outcome is denominated in.
    scale: f64,
    cost_scratch: Vec<f64>,
    eta_star: f64,
    delta_star: Vec<f64>,
    lambda: Vec<f64>,
}

impl ConditionalMomentLpWorkspace {
    pub(in crate::inference::sensitivity) fn new(
        x_matrix: &[Vec<f64>],
        sigma: &[Vec<f64>],
    ) -> Result<Self, String> {
        let sd_vec = diag_sqrt(sigma);
        let scale = sd_vec.iter().copied().fold(0.0_f64, f64::max).max(1.0);
        let k = x_matrix.first().map_or(0, Vec::len);
        let rows = x_matrix.len();
        if sd_vec.len() != rows {
            return Err(format!(
                "HonestDiD eta/delta LP has {rows} design rows but {} variances",
                sd_vec.len()
            ));
        }
        let columns: Vec<Vec<f64>> = x_matrix
            .iter()
            .zip(&sd_vec)
            .map(|(x_row, sd)| {
                let mut column = Vec::with_capacity(k + 1);
                column.push(sd / scale);
                column.extend(x_row.iter().map(|x| x / scale));
                column
            })
            .collect();
        let mut g = vec![0.0; k + 1];
        g[0] = 1.0;
        let mut lp = DenseSimplex::new(&columns, &g);
        lp.set_memory(BASIS_MEMORY);
        Ok(Self {
            lp,
            k,
            scale,
            cost_scratch: vec![0.0; rows],
            eta_star: 0.0,
            delta_star: vec![0.0; k],
            lambda: vec![0.0; rows],
        })
    }

    /// Run phase one now, so clones of this workspace start from a feasible
    /// basis. The least-favorable simulation clones one prepared workspace per
    /// rayon job rather than building and feasibility-solving one per job.
    /// Infeasibility is not an error here: the solves report it, as they would
    /// have anyway.
    pub(in crate::inference::sensitivity) fn prepare(&mut self) {
        let _ = self.lp.prepare();
    }

    fn load(&mut self, y_vec: &[f64]) -> Result<(), String> {
        if self.cost_scratch.len() != y_vec.len() {
            return Err(format!(
                "HonestDiD eta/delta LP rhs length mismatch: expected {}, got {}",
                self.cost_scratch.len(),
                y_vec.len()
            ));
        }
        for (dst, y) in self.cost_scratch.iter_mut().zip(y_vec) {
            *dst = y / self.scale;
        }
        self.lp.set_cost(&self.cost_scratch);
        self.lp
            .solve()
            .map_err(|status| format!("failed to solve HonestDiD eta/delta LP: {status}"))
    }

    /// # Errors
    /// Returns an error if the rhs length is wrong or the program has no
    /// bounded optimum.
    pub(in crate::inference::sensitivity) fn solve_in_place(
        &mut self,
        y_vec: &[f64],
    ) -> Result<(), String> {
        self.load(y_vec)?;
        let duals = self.lp.duals();
        self.eta_star = duals[0];
        self.delta_star.clear();
        self.delta_star.extend_from_slice(&duals[1..=self.k]);
        let inv_scale = 1.0 / self.scale;
        self.lambda.clear();
        self.lambda
            .extend(self.lp.lambda().iter().map(|value| value * inv_scale));
        Ok(())
    }

    /// The statistic alone.
    ///
    /// # Errors
    /// As [`Self::solve_in_place`].
    pub(in crate::inference::sensitivity) fn solve_eta_only(
        &mut self,
        y_vec: &[f64],
    ) -> Result<f64, String> {
        self.load(y_vec)?;
        Ok(self.lp.dual_0())
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

/// Pieces a dual-max workspace remembers per affine cost family; see
/// [`DualMaxLpWorkspace::solve_for_c`].
const DUAL_PIECES: usize = 64;

/// One basis of the dual max program with the interval of `c` on which it is
/// optimal for the current affine cost family.
#[derive(Clone)]
struct Piece {
    lo: f64,
    hi: f64,
    basis: Vec<usize>,
    xb: Vec<f64>,
}

/// Reusable workspace for the dual max LP,
/// `max f'x  s.t.  W' x = e_0,  x ≥ 0`, which is the simplex's own form.
pub(in crate::inference::sensitivity) struct DualMaxLpWorkspace<'a> {
    lp: DenseSimplex,
    /// Read only when the simplex fails, to hand the Clarabel fallback its
    /// equalities.
    w_t: &'a [Vec<f64>],
    f_scratch: Vec<f64>,
    /// The affine family `f = f0 + c f1` the pieces belong to. A call with a
    /// different family clears them.
    f0: Vec<f64>,
    f1: Vec<f64>,
    has_family: bool,
    pieces: Vec<Piece>,
    pieces_next: usize,
    y0_scratch: Vec<f64>,
    y1_scratch: Vec<f64>,
}

impl<'a> DualMaxLpWorkspace<'a> {
    pub(in crate::inference::sensitivity) fn new(w_t: &'a [Vec<f64>]) -> Result<Self, String> {
        let dim = w_t.len();
        let width = w_t.first().map_or(0, Vec::len);
        if w_t.iter().any(|row| row.len() != width) {
            return Err("HonestDiD dual max LP has ragged rows".to_string());
        }
        let mut g = vec![0.0; width];
        if width > 0 {
            g[0] = 1.0;
        }
        let mut lp = DenseSimplex::new(w_t, &g);
        lp.set_memory(BASIS_MEMORY);
        Ok(Self {
            lp,
            w_t,
            f_scratch: vec![0.0; dim],
            f0: vec![0.0; dim],
            f1: vec![0.0; dim],
            has_family: false,
            pieces: Vec::new(),
            pieces_next: 0,
            y0_scratch: Vec::with_capacity(width),
            y1_scratch: Vec::with_capacity(width),
        })
    }

    /// The optimum `f' x` for `f = s_t + c σ_γ / σ_b²`.
    ///
    /// The bisection above calls this twenty-odd times per acceptance test with
    /// the same `s_t`, `σ_γ` and `σ_b²` and a moving `c`, so the cost vector is
    /// affine in `c` and the optimum is convex piecewise-linear in it. Each
    /// solve therefore records the basis it ended on together with the interval
    /// of `c` on which that basis stays optimal, read off two reduced-cost
    /// vectors. A later `c` inside a recorded interval is answered from that
    /// basis's values with no solve, and with the same arithmetic the solve
    /// would have used, so the number is the same one.
    pub(in crate::inference::sensitivity) fn solve_for_c(
        &mut self,
        s_t: &[f64],
        sigma_gamma: &[f64],
        sigma_b2: f64,
        c: f64,
    ) -> Result<f64, String> {
        if sigma_b2.abs() < f64::EPSILON {
            return Err("HonestDiD dual max program encountered zero gamma variance".to_string());
        }
        for ((f_value, s_value), sigma_gamma_value) in self
            .f_scratch
            .iter_mut()
            .zip(s_t.iter())
            .zip(sigma_gamma.iter())
        {
            *f_value = c.mul_add(*sigma_gamma_value / sigma_b2, *s_value);
        }
        let same_family = self.has_family
            && self.f0 == s_t
            && self
                .f1
                .iter()
                .zip(sigma_gamma)
                .all(|(f1, sigma_gamma_value)| *f1 == sigma_gamma_value / sigma_b2);
        if same_family {
            if let Some(piece) = self
                .pieces
                .iter()
                .find(|piece| piece.lo <= c && c <= piece.hi)
            {
                let n = self.lp.num_columns();
                let mut objective = 0.0;
                for (&j, &x) in piece.basis.iter().zip(&piece.xb) {
                    if j < n {
                        objective += self.f_scratch[j] * x.max(0.0);
                    }
                }
                return Ok(objective);
            }
        } else {
            self.f0.copy_from_slice(s_t);
            for (f1, sigma_gamma_value) in self.f1.iter_mut().zip(sigma_gamma) {
                *f1 = sigma_gamma_value / sigma_b2;
            }
            self.has_family = true;
            self.pieces.clear();
            self.pieces_next = 0;
        }
        self.lp.set_cost(&self.f_scratch);
        match self.lp.solve() {
            Ok(()) => {
                let (lo, hi) = self.lp.optimality_interval(
                    &self.f0,
                    &self.f1,
                    &mut self.y0_scratch,
                    &mut self.y1_scratch,
                );
                let piece = Piece {
                    lo,
                    hi,
                    basis: self.lp.basis().to_vec(),
                    xb: self.lp.basic_values().to_vec(),
                };
                if self.pieces.len() < DUAL_PIECES {
                    self.pieces.push(piece);
                } else {
                    self.pieces[self.pieces_next] = piece;
                    self.pieces_next = (self.pieces_next + 1) % DUAL_PIECES;
                }
                Ok(self.lp.objective())
            }
            Err(LpError::Infeasible) => Err("HonestDiD dual max program is infeasible".to_string()),
            Err(LpError::Unbounded) => Err("HonestDiD dual max program is unbounded".to_string()),
            Err(status) => solve_dual_max_with_clarabel_fallback(self.w_t, &self.f_scratch)
                .map_err(|fallback_err| {
                    format!(
                        "dense simplex failed to solve HonestDiD dual max LP: {status}; \
                         Clarabel fallback also failed: {fallback_err}"
                    )
                }),
        }
    }
}

#[cfg(test)]
mod tests {
    use clarabel::algebra::CscMatrix;
    use clarabel::solver::{
        DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
    };
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    use super::super::super::linear_algebra::{
        build_clarabel_matrix, dense_rows_to_csc, diag_sqrt, dot,
    };
    use super::{ConditionalMomentLpWorkspace, DualMaxLpWorkspace};

    struct ClarabelReferenceResult {
        eta_star: f64,
        delta_star: Vec<f64>,
        lambda: Vec<f64>,
    }

    /// The ARP program as the split-variable conic problem the HiGHS tests use.
    fn solve_arp_with_clarabel(
        x_matrix: &[Vec<f64>],
        sigma: &[Vec<f64>],
        y_vec: &[f64],
    ) -> Option<ClarabelReferenceResult> {
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
        if !matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ) {
            return None;
        }
        Some(ClarabelReferenceResult {
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
        })
    }

    #[test]
    fn conditional_workspace_matches_clarabel_reference() {
        let x_matrix = vec![Vec::new(), Vec::new(), Vec::new()];
        let sigma = vec![
            vec![1.2, 0.1, 0.05],
            vec![0.1, 0.9, 0.08],
            vec![0.05, 0.08, 1.1],
        ];
        let y_vec = vec![0.7, 0.3, 0.45];

        let reference = solve_arp_with_clarabel(&x_matrix, &sigma, &y_vec).unwrap();
        let mut workspace = ConditionalMomentLpWorkspace::new(&x_matrix, &sigma).unwrap();
        workspace.solve_in_place(&y_vec).unwrap();

        assert!((workspace.eta_star() - reference.eta_star).abs() < 1e-8);
        for (observed, expected) in workspace.delta_star().iter().zip(&reference.delta_star) {
            assert!((observed - expected).abs() < 1e-8);
        }
        for (observed, expected) in workspace.lambda().iter().zip(&reference.lambda) {
            assert!((observed - expected).abs() < 1e-7);
        }
    }

    #[test]
    fn conditional_workspace_matches_clarabel_with_nuisance_columns_across_rhs_changes() {
        let mut rng = StdRng::seed_from_u64(20260905);
        for _ in 0..40 {
            let rows = rng.random_range(6..30);
            let k = rng.random_range(1..rows.min(8));
            let x_matrix: Vec<Vec<f64>> = (0..rows)
                .map(|_| (0..k).map(|_| rng.random_range(-1.0..1.0)).collect())
                .collect();
            let sigma: Vec<Vec<f64>> = (0..rows)
                .map(|i| {
                    (0..rows)
                        .map(|j| {
                            if i == j {
                                rng.random_range(0.2..3.0)
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect();
            let mut workspace = ConditionalMomentLpWorkspace::new(&x_matrix, &sigma).unwrap();
            for _ in 0..5 {
                let y_vec: Vec<f64> = (0..rows).map(|_| rng.random_range(-2.0..2.0)).collect();
                // A design with a direction `Xδ > 0` makes `η` unbounded below;
                // both solvers must then say so.
                let Some(reference) = solve_arp_with_clarabel(&x_matrix, &sigma, &y_vec) else {
                    let error = workspace.solve_in_place(&y_vec).unwrap_err();
                    assert!(error.contains("Infeasible"), "{error}");
                    continue;
                };
                workspace.solve_in_place(&y_vec).unwrap();
                assert!(
                    (workspace.eta_star() - reference.eta_star).abs() < 1e-6,
                    "{} vs {}",
                    workspace.eta_star(),
                    reference.eta_star
                );
                // The objective of the multiplier program equals eta*.
                let eta_only = workspace.solve_eta_only(&y_vec).unwrap();
                assert!((eta_only - reference.eta_star).abs() < 1e-6);
                // The multipliers reproduce eta* through the rhs.
                let from_lambda: f64 = workspace
                    .lambda()
                    .iter()
                    .zip(&y_vec)
                    .map(|(l, y)| l * y)
                    .sum();
                assert!((from_lambda - reference.eta_star).abs() < 1e-6);
            }
        }
    }

    fn solve_dual_max_with_clarabel(w_t: &[Vec<f64>], f: &[f64]) -> f64 {
        let dim = w_t.len();
        let width = w_t.first().map_or(0, Vec::len);
        let inequalities: Vec<Vec<f64>> = (0..dim)
            .map(|idx| {
                let mut row = vec![0.0; dim];
                row[idx] = -1.0;
                row
            })
            .collect();
        let mut equalities = Vec::with_capacity(width);
        for col_idx in 0..width {
            equalities.push(w_t.iter().map(|row| row[col_idx]).collect());
        }
        let constraint_matrix = build_clarabel_matrix(&inequalities, &equalities);
        let mut rhs = vec![0.0; dim];
        rhs.push(1.0);
        rhs.extend(std::iter::repeat_n(0.0, width.saturating_sub(1)));
        let cones = vec![
            SupportedConeT::NonnegativeConeT(dim),
            SupportedConeT::ZeroConeT(width),
        ];
        let quadratic = CscMatrix::<f64>::zeros((dim, dim));
        let q = f.iter().map(|value| -*value).collect::<Vec<_>>();
        let settings = DefaultSettingsBuilder::<f64>::default()
            .verbose(false)
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
        dot(&solver.solution.x, f)
    }

    #[test]
    fn dual_workspace_matches_clarabel_reference() {
        let w_t = vec![vec![1.0, 0.0], vec![1.0, 1.0], vec![1.0, -1.0]];
        let mut workspace = DualMaxLpWorkspace::new(&w_t).unwrap();
        let s_t = vec![0.2, -0.1, 0.05];
        let sigma_gamma = vec![0.4, -0.2, 0.1];
        let sigma_b2 = 0.75;
        for c in [0.3_f64, -0.2, 1.7, 0.0] {
            let f = s_t
                .iter()
                .zip(sigma_gamma.iter())
                .map(|(s_value, gamma_value)| c.mul_add(*gamma_value / sigma_b2, *s_value))
                .collect::<Vec<_>>();
            let reference = solve_dual_max_with_clarabel(&w_t, &f);
            let observed = workspace
                .solve_for_c(&s_t, &sigma_gamma, sigma_b2, c)
                .unwrap();
            assert!(
                (observed - reference).abs() < 1e-8,
                "c={c}: {observed} vs {reference}"
            );
        }
    }

    #[test]
    fn dual_workspace_reports_an_infeasible_program() {
        // Every column has a negative first entry, so no x >= 0 reaches e_0.
        let w_t = vec![vec![-1.0, 0.0], vec![-0.5, 1.0]];
        let mut workspace = DualMaxLpWorkspace::new(&w_t).unwrap();
        let error = workspace
            .solve_for_c(&[0.1, 0.2], &[0.3, 0.1], 1.0, 0.5)
            .unwrap_err();
        assert!(error.contains("infeasible"), "{error}");
    }
}
