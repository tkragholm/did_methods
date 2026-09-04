//! A dense primal simplex for the multiplier programs behind the ARP test.
//!
//! Solves
//!
//! ```text
//! max  c'λ    subject to    A λ = g,    λ ≥ 0
//! ```
//!
//! with `A` an `m × n` dense matrix, `m` small (the number of free variables in
//! the program this is the dual of) and `n` a few dozen (the number of moment
//! inequalities). Both LP workspaces in this module are instances of it:
//!
//! - The ARP auxiliary program `min η  s.t.  a_i'(η, δ) ≥ b_i`, with `(η, δ)`
//!   free, is the dual of this problem with the `a_i` as columns, `g = e_0`
//!   and `c = b`. Its optimal `(η, δ)` are this problem's duals `y`, and its
//!   nonnegative row multipliers are this problem's `λ`.
//! - The dual max program `max f'x  s.t.  W' x = e_0,  x ≥ 0` is this problem
//!   directly.
//!
//! In both uses the constraint matrix is fixed for the life of a workspace and
//! only `c` changes between solves, so a basis stays primal feasible from one
//! solve to the next and the next solve starts from it. The first solve goes
//! through a phase one on artificial columns.
//!
//! The basis inverse is held explicitly and updated by the pivot's elementary
//! row operations, with a fresh Gauss-Jordan factorisation every
//! [`REFACTOR_EVERY`] pivots. At these sizes that is a few hundred floating
//! point operations per pivot, which is where the win over a general solver
//! comes from: nothing here is proportional to anything but `m` and `n`.

use std::fmt;

const TOL_REDUCED_COST: f64 = 1e-9;
const TOL_PIVOT: f64 = 1e-10;
const TOL_DEGENERATE: f64 = 1e-12;
const TOL_PHASE_ONE: f64 = 1e-8;
const REFACTOR_EVERY: usize = 40;
/// Consecutive degenerate pivots before pricing switches to Bland's rule,
/// which cannot cycle.
const BLAND_AFTER_DEGENERATE: usize = 25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::inference::sensitivity) enum LpError {
    /// No `λ ≥ 0` satisfies `A λ = g`.
    Infeasible,
    /// The objective increases without bound.
    Unbounded,
    /// The pivot budget ran out.
    IterationLimit,
    /// The basis could not be factorised.
    Singular,
}

impl fmt::Display for LpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Infeasible => "Infeasible",
            Self::Unbounded => "Unbounded",
            Self::IterationLimit => "IterationLimit",
            Self::Singular => "Singular",
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    One,
    Two,
}

pub(in crate::inference::sensitivity) struct DenseSimplex {
    m: usize,
    n: usize,
    /// Column-major, `m * n`: column `j` occupies `a[j * m..(j + 1) * m]`.
    a: Vec<f64>,
    cost: Vec<f64>,
    g: Vec<f64>,
    /// Artificial column `i` is `art_sign[i] * e_i`, so the all-artificial
    /// basis carries `|g|`.
    art_sign: Vec<f64>,
    /// Column index per basis position: `< n` is an original column, `n + i`
    /// the artificial for row `i`.
    basis: Vec<usize>,
    is_basic: Vec<bool>,
    /// Row-major `m * m`.
    binv: Vec<f64>,
    xb: Vec<f64>,
    y: Vec<f64>,
    col: Vec<f64>,
    lambda: Vec<f64>,
    objective: f64,
    has_feasible_basis: bool,
    pivots_since_refactor: usize,
}

impl DenseSimplex {
    /// `columns[j]` is column `j` of `A`, each of length `g.len()`.
    pub(in crate::inference::sensitivity) fn new(columns: &[Vec<f64>], g: &[f64]) -> Self {
        let m = g.len();
        let n = columns.len();
        let mut a = Vec::with_capacity(m * n);
        for column in columns {
            debug_assert_eq!(column.len(), m);
            a.extend_from_slice(column);
        }
        Self {
            m,
            n,
            a,
            cost: vec![0.0; n],
            g: g.to_vec(),
            art_sign: vec![1.0; m],
            basis: vec![0; m],
            is_basic: vec![false; n + m],
            binv: vec![0.0; m * m],
            xb: vec![0.0; m],
            y: vec![0.0; m],
            col: vec![0.0; m],
            lambda: vec![0.0; n],
            objective: 0.0,
            has_feasible_basis: false,
            pivots_since_refactor: 0,
        }
    }

    pub(in crate::inference::sensitivity) fn set_cost(&mut self, cost: &[f64]) {
        debug_assert_eq!(cost.len(), self.n);
        self.cost.copy_from_slice(cost);
    }

    /// `c'λ` at the last optimum.
    pub(in crate::inference::sensitivity) const fn objective(&self) -> f64 {
        self.objective
    }

    /// The optimal `λ`, length `n`.
    pub(in crate::inference::sensitivity) fn lambda(&self) -> &[f64] {
        &self.lambda
    }

    /// The optimal duals `y`, length `m`: the solution of the program this is
    /// the dual of.
    pub(in crate::inference::sensitivity) fn duals(&self) -> &[f64] {
        &self.y
    }

    /// Solve for the current cost vector, from the last basis when there is
    /// one.
    ///
    /// # Errors
    /// See [`LpError`]. After an error the basis is still primal feasible
    /// (except after `Singular`), so a later call can start from it.
    pub(in crate::inference::sensitivity) fn solve(&mut self) -> Result<(), LpError> {
        if !self.has_feasible_basis {
            self.install_artificial_basis();
            self.iterate(Phase::One)?;
            let infeasibility: f64 = self
                .basis
                .iter()
                .zip(&self.xb)
                .filter(|(j, _)| **j >= self.n)
                .map(|(_, x)| *x)
                .sum();
            if infeasibility > TOL_PHASE_ONE {
                return Err(LpError::Infeasible);
            }
            self.has_feasible_basis = true;
        }
        self.iterate(Phase::Two)?;
        self.compute_duals(Phase::Two);
        self.lambda.fill(0.0);
        let mut objective = 0.0;
        for (position, &j) in self.basis.iter().enumerate() {
            if j < self.n {
                let value = self.xb[position].max(0.0);
                self.lambda[j] = value;
                objective += self.cost[j] * value;
            }
        }
        self.objective = objective;
        Ok(())
    }

    fn install_artificial_basis(&mut self) {
        let m = self.m;
        self.is_basic.fill(false);
        self.binv.fill(0.0);
        for i in 0..m {
            self.art_sign[i] = if self.g[i] < 0.0 { -1.0 } else { 1.0 };
            self.basis[i] = self.n + i;
            self.is_basic[self.n + i] = true;
            self.binv[i * m + i] = self.art_sign[i];
            self.xb[i] = self.g[i].abs();
        }
        self.pivots_since_refactor = 0;
    }

    fn cost_of(&self, j: usize, phase: Phase) -> f64 {
        match phase {
            Phase::One => {
                if j < self.n {
                    0.0
                } else {
                    -1.0
                }
            }
            Phase::Two => {
                if j < self.n {
                    self.cost[j]
                } else {
                    0.0
                }
            }
        }
    }

    /// `y = B^{-T} c_B`.
    fn compute_duals(&mut self, phase: Phase) {
        let m = self.m;
        self.y.fill(0.0);
        for r in 0..m {
            let c_b = self.cost_of(self.basis[r], phase);
            if c_b == 0.0 {
                continue;
            }
            let row = &self.binv[r * m..(r + 1) * m];
            for (y, b) in self.y.iter_mut().zip(row) {
                *y += c_b * b;
            }
        }
    }

    fn column(&self, j: usize) -> &[f64] {
        &self.a[j * self.m..(j + 1) * self.m]
    }

    fn iterate(&mut self, phase: Phase) -> Result<(), LpError> {
        let m = self.m;
        let n = self.n;
        let limit = 100 * (m + n) + 1_000;
        let mut degenerate_streak = 0usize;
        let mut bland = false;
        for _ in 0..limit {
            self.compute_duals(phase);
            // Pricing. Only original columns may enter: artificials start
            // basic and, once out, have no business coming back.
            let mut entering = None;
            let mut best = TOL_REDUCED_COST;
            for j in 0..n {
                if self.is_basic[j] {
                    continue;
                }
                let mut d = self.cost_of(j, phase);
                for (a, y) in self.column(j).iter().zip(&self.y) {
                    d -= a * y;
                }
                if bland {
                    if d > TOL_REDUCED_COST {
                        entering = Some(j);
                        break;
                    }
                } else if d > best {
                    best = d;
                    entering = Some(j);
                }
            }
            let Some(q) = entering else {
                return Ok(());
            };
            // col = B^{-1} a_q
            for i in 0..m {
                let row = &self.binv[i * m..(i + 1) * m];
                self.col[i] = row.iter().zip(self.column(q)).map(|(b, a)| b * a).sum();
            }
            // Ratio test. In phase two a basic artificial sits at zero and must
            // not be allowed to grow, so it blocks at zero whichever way the
            // pivot would move it.
            let mut leaving: Option<usize> = None;
            let mut theta = f64::INFINITY;
            for i in 0..m {
                let alpha = self.col[i];
                if phase == Phase::Two && self.basis[i] >= n && alpha.abs() > TOL_PIVOT {
                    leaving = Some(i);
                    theta = 0.0;
                    break;
                }
                if alpha > TOL_PIVOT {
                    let ratio = self.xb[i] / alpha;
                    let better = match leaving {
                        None => true,
                        Some(r) => {
                            ratio < theta - TOL_DEGENERATE
                                || ((ratio - theta).abs() <= TOL_DEGENERATE
                                    && self.basis[i] < self.basis[r])
                        }
                    };
                    if better {
                        theta = ratio;
                        leaving = Some(i);
                    }
                }
            }
            let Some(r) = leaving else {
                return Err(LpError::Unbounded);
            };
            if theta <= TOL_DEGENERATE {
                degenerate_streak += 1;
                if degenerate_streak >= BLAND_AFTER_DEGENERATE {
                    bland = true;
                }
            } else {
                degenerate_streak = 0;
            }
            self.pivot(r, q, theta);
            if self.pivots_since_refactor >= REFACTOR_EVERY {
                self.refactor()?;
            }
        }
        Err(LpError::IterationLimit)
    }

    fn pivot(&mut self, r: usize, q: usize, theta: f64) {
        let m = self.m;
        let p = self.col[r];
        for i in 0..m {
            if i != r {
                self.xb[i] -= theta * self.col[i];
                if self.xb[i] < 0.0 {
                    self.xb[i] = 0.0;
                }
            }
        }
        self.xb[r] = theta;
        let inv_p = 1.0 / p;
        for value in &mut self.binv[r * m..(r + 1) * m] {
            *value *= inv_p;
        }
        for i in 0..m {
            if i == r {
                continue;
            }
            let factor = self.col[i];
            if factor == 0.0 {
                continue;
            }
            let (head, tail) = self.binv.split_at_mut(r.max(i) * m);
            let (row_i, row_r) = if i < r {
                (&mut head[i * m..(i + 1) * m], &tail[..m])
            } else {
                (&mut tail[..m], &head[r * m..(r + 1) * m])
            };
            for (target, pivot_row) in row_i.iter_mut().zip(row_r) {
                *target -= factor * pivot_row;
            }
        }
        let old = self.basis[r];
        self.is_basic[old] = false;
        self.is_basic[q] = true;
        self.basis[r] = q;
        self.pivots_since_refactor += 1;
    }

    /// Rebuild `B^{-1}` from the basis by Gauss-Jordan elimination with partial
    /// pivoting, and `x_B = B^{-1} g` with it.
    fn refactor(&mut self) -> Result<(), LpError> {
        let m = self.m;
        let mut b = vec![0.0; m * m];
        for (position, &j) in self.basis.iter().enumerate() {
            if j < self.n {
                for (i, &value) in self.column(j).iter().enumerate() {
                    b[i * m + position] = value;
                }
            } else {
                let row = j - self.n;
                b[row * m + position] = self.art_sign[row];
            }
        }
        let mut inv = vec![0.0; m * m];
        for i in 0..m {
            inv[i * m + i] = 1.0;
        }
        for k in 0..m {
            let mut pivot_row = k;
            let mut pivot_abs = b[k * m + k].abs();
            for i in k + 1..m {
                let candidate = b[i * m + k].abs();
                if candidate > pivot_abs {
                    pivot_abs = candidate;
                    pivot_row = i;
                }
            }
            if pivot_abs < 1e-13 {
                return Err(LpError::Singular);
            }
            if pivot_row != k {
                for c in 0..m {
                    b.swap(k * m + c, pivot_row * m + c);
                    inv.swap(k * m + c, pivot_row * m + c);
                }
            }
            let inv_p = 1.0 / b[k * m + k];
            for c in 0..m {
                b[k * m + c] *= inv_p;
                inv[k * m + c] *= inv_p;
            }
            for i in 0..m {
                if i == k {
                    continue;
                }
                let factor = b[i * m + k];
                if factor == 0.0 {
                    continue;
                }
                for c in 0..m {
                    b[i * m + c] -= factor * b[k * m + c];
                    inv[i * m + c] -= factor * inv[k * m + c];
                }
            }
        }
        self.binv = inv;
        for i in 0..m {
            let row = &self.binv[i * m..(i + 1) * m];
            let value: f64 = row.iter().zip(&self.g).map(|(b, g)| b * g).sum();
            self.xb[i] = value.max(0.0);
        }
        self.pivots_since_refactor = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use clarabel::algebra::CscMatrix;
    use clarabel::solver::{
        DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
    };
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    use super::super::super::linear_algebra::build_clarabel_matrix;
    use super::{DenseSimplex, LpError};

    /// `max c'λ s.t. A λ = g, λ ≥ 0` by interior point, as the oracle.
    fn clarabel_reference(columns: &[Vec<f64>], g: &[f64], c: &[f64]) -> Option<f64> {
        let n = columns.len();
        let m = g.len();
        let inequalities: Vec<Vec<f64>> = (0..n)
            .map(|idx| {
                let mut row = vec![0.0; n];
                row[idx] = -1.0;
                row
            })
            .collect();
        let equalities: Vec<Vec<f64>> = (0..m)
            .map(|i| columns.iter().map(|column| column[i]).collect())
            .collect();
        let constraint_matrix = build_clarabel_matrix(&inequalities, &equalities);
        let mut rhs = vec![0.0; n];
        rhs.extend_from_slice(g);
        let cones = vec![
            SupportedConeT::NonnegativeConeT(n),
            SupportedConeT::ZeroConeT(m),
        ];
        let quadratic = CscMatrix::<f64>::zeros((n, n));
        let q = c.iter().map(|value| -*value).collect::<Vec<_>>();
        let settings = DefaultSettingsBuilder::<f64>::default()
            .verbose(false)
            .tol_gap_abs(1e-10)
            .tol_gap_rel(1e-10)
            .tol_feas(1e-10)
            .build()
            .expect("Clarabel settings");
        let mut solver =
            DefaultSolver::new(&quadratic, &q, &constraint_matrix, &rhs, &cones, settings)
                .expect("Clarabel workspace");
        solver.solve();
        match solver.solution.status {
            SolverStatus::Solved | SolverStatus::AlmostSolved => {
                Some(solver.solution.x.iter().zip(c).map(|(x, c)| x * c).sum())
            }
            _ => None,
        }
    }

    /// Costs that keep `max c'λ` bounded on these columns: each `c_j` sits
    /// below `a_j'y` for one `y`, so `y` is dual feasible.
    fn bounded_cost(rng: &mut StdRng, columns: &[Vec<f64>], m: usize) -> Vec<f64> {
        let y: Vec<f64> = (0..m).map(|_| rng.random_range(-1.0..1.0)).collect();
        columns
            .iter()
            .map(|column| {
                let ay: f64 = column.iter().zip(&y).map(|(a, y)| a * y).sum();
                ay - rng.random_range(0.0..1.0)
            })
            .collect()
    }

    fn random_feasible_problem(
        rng: &mut StdRng,
        m: usize,
        n: usize,
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let columns: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..m).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();
        // g = A λ0 for a sparse nonnegative λ0, so the problem is feasible.
        let mut g = vec![0.0; m];
        for column in &columns {
            if rng.random_bool(0.4) {
                let weight: f64 = rng.random_range(0.0..2.0);
                for (g, a) in g.iter_mut().zip(column) {
                    *g += weight * a;
                }
            }
        }
        let c = bounded_cost(rng, &columns, m);
        (columns, g, c)
    }

    #[test]
    fn matches_clarabel_on_random_feasible_bounded_programs() {
        let mut rng = StdRng::seed_from_u64(20260904);
        for trial in 0..200 {
            let m = rng.random_range(2..9);
            let n = rng.random_range(m..m + 24);
            let (columns, g, c) = random_feasible_problem(&mut rng, m, n);
            let Some(reference) = clarabel_reference(&columns, &g, &c) else {
                continue;
            };
            let mut lp = DenseSimplex::new(&columns, &g);
            lp.set_cost(&c);
            lp.solve().unwrap_or_else(|e| panic!("trial {trial}: {e}"));
            assert!(
                (lp.objective() - reference).abs() < 1e-6,
                "trial {trial}: simplex {} vs clarabel {reference}",
                lp.objective()
            );
            // Primal feasibility and dual feasibility of the reported answer.
            for i in 0..m {
                let activity: f64 = columns
                    .iter()
                    .zip(lp.lambda())
                    .map(|(column, l)| column[i] * l)
                    .sum();
                assert!((activity - g[i]).abs() < 1e-9, "trial {trial}: row {i}");
            }
            assert!(lp.lambda().iter().all(|l| *l >= 0.0));
            for (j, column) in columns.iter().enumerate() {
                let ay: f64 = column.iter().zip(lp.duals()).map(|(a, y)| a * y).sum();
                assert!(ay >= c[j] - 1e-8, "trial {trial}: reduced cost of {j}");
            }
        }
    }

    #[test]
    fn warm_start_after_a_cost_change_matches_a_cold_solve() {
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..100 {
            let m = rng.random_range(2..8);
            let n = rng.random_range(m + 2..m + 20);
            let (columns, g, c1) = random_feasible_problem(&mut rng, m, n);
            let c2 = bounded_cost(&mut rng, &columns, m);
            let mut warm = DenseSimplex::new(&columns, &g);
            warm.set_cost(&c1);
            warm.solve().unwrap();
            warm.set_cost(&c2);
            warm.solve().unwrap();
            let mut cold = DenseSimplex::new(&columns, &g);
            cold.set_cost(&c2);
            cold.solve().unwrap();
            assert!((warm.objective() - cold.objective()).abs() < 1e-9);
        }
    }

    #[test]
    fn reports_infeasibility() {
        // Columns all in the positive orthant cannot reach a negative g.
        let columns = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let mut lp = DenseSimplex::new(&columns, &[-1.0, 1.0]);
        lp.set_cost(&[1.0, 1.0, 1.0]);
        assert_eq!(lp.solve(), Err(LpError::Infeasible));
    }

    #[test]
    fn reports_unboundedness() {
        // λ_1 - λ_2 = 1 with both free to grow, and a cost that rewards it.
        let columns = vec![vec![1.0], vec![-1.0]];
        let mut lp = DenseSimplex::new(&columns, &[1.0]);
        lp.set_cost(&[1.0, 1.0]);
        assert_eq!(lp.solve(), Err(LpError::Unbounded));
    }

    #[test]
    fn survives_a_degenerate_start_and_many_pivots() {
        // g = e_0 is the production shape: every artificial but one starts at
        // zero, so phase one is degenerate from the first pivot.
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..50 {
            let m = 20;
            let n = 40;
            let columns: Vec<Vec<f64>> = (0..n)
                .map(|_| {
                    let mut column: Vec<f64> =
                        (0..m).map(|_| rng.random_range(-1.0..1.0)).collect();
                    column[0] = rng.random_range(0.1..1.0);
                    column
                })
                .collect();
            let mut g = vec![0.0; m];
            g[0] = 1.0;
            let c: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
            let mut lp = DenseSimplex::new(&columns, &g);
            lp.set_cost(&c);
            let reference = clarabel_reference(&columns, &g, &c);
            match (lp.solve(), reference) {
                (Ok(()), Some(reference)) => {
                    assert!((lp.objective() - reference).abs() < 1e-6);
                }
                (Err(LpError::Infeasible), None) | (Err(LpError::Unbounded), None) => {}
                (result, reference) => panic!("simplex {result:?} vs clarabel {reference:?}"),
            }
            // A hundred further cost changes from the same basis.
            if lp.lambda().iter().any(|l| *l > 0.0) {
                for _ in 0..100 {
                    let c: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
                    lp.set_cost(&c);
                    if let Some(reference) = clarabel_reference(&columns, &g, &c) {
                        lp.solve().unwrap();
                        assert!((lp.objective() - reference).abs() < 1e-6);
                    }
                }
            }
        }
    }
}
