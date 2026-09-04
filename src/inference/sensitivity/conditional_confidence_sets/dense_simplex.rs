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

#[derive(Clone)]
struct RememberedBasis {
    /// The basis as a sorted set, for the membership test.
    key: Vec<usize>,
    basis: Vec<usize>,
    binv: Vec<f64>,
    xb: Vec<f64>,
    /// How many pivots the stored inverse is past its last factorisation, so
    /// a reload keeps counting from there rather than from zero.
    pivots_since_refactor: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    One,
    Two,
}

#[derive(Clone)]
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
    pivot_row: Vec<f64>,
    lambda: Vec<f64>,
    objective: f64,
    has_feasible_basis: bool,
    pivots_since_refactor: usize,
    /// Optimal bases seen by earlier solves, each with its inverse and basic
    /// values, so a new cost vector can start from whichever of them it likes
    /// best rather than from wherever the last solve stopped. See
    /// [`Self::set_memory`].
    memory: Vec<RememberedBasis>,
    memory_cap: usize,
    memory_next: usize,
    key_scratch: Vec<usize>,
    duals_current: bool,
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
            pivot_row: vec![0.0; m],
            lambda: vec![0.0; n],
            objective: 0.0,
            has_feasible_basis: false,
            pivots_since_refactor: 0,
            memory: Vec::new(),
            memory_cap: 0,
            memory_next: 0,
            key_scratch: vec![0; m],
            duals_current: false,
        }
    }

    /// Remember up to `cap` optimal bases across solves, and start each solve
    /// from the remembered basis with the best objective under the new costs
    /// when that beats the current one.
    ///
    /// For a sequence of unrelated cost vectors over one polytope, which is what
    /// the least-favorable simulation is, the optimum lands on a small set of
    /// vertices far more often than not: measured on Study I's shape, a run of
    /// 57 draws visited 12 distinct optimal bases. Starting from the best known
    /// vertex turns most of those solves into a single pricing pass. Zero
    /// disables it.
    pub(in crate::inference::sensitivity) fn set_memory(&mut self, cap: usize) {
        self.memory_cap = cap;
        self.memory.clear();
        self.memory_next = 0;
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
    /// the dual of. Computed from the final basis on demand.
    pub(in crate::inference::sensitivity) fn duals(&mut self) -> &[f64] {
        if !self.duals_current {
            self.compute_duals(Phase::Two);
            self.duals_current = true;
        }
        &self.y
    }

    /// The first dual alone, `Σ_r c_Br (B^{-1})_{r0}`, in the same order of
    /// summation as [`Self::duals`] so the two agree to the bit. It is the
    /// optimal value of the program this is the dual of when `g = e_0`.
    pub(in crate::inference::sensitivity) fn dual_0(&self) -> f64 {
        if self.duals_current {
            return self.y[0];
        }
        let m = self.m;
        let mut y0 = 0.0;
        for r in 0..m {
            let c_b = self.cost_of(self.basis[r], Phase::Two);
            if c_b == 0.0 {
                continue;
            }
            y0 += c_b * self.binv[r * m];
        }
        y0
    }

    /// Solve for the current cost vector, from the last basis when there is
    /// one.
    ///
    /// # Errors
    /// See [`LpError`]. After an error the basis is still primal feasible
    /// (except after `Singular`), so a later call can start from it.
    pub(in crate::inference::sensitivity) fn solve(&mut self) -> Result<(), LpError> {
        self.prepare()?;
        if !self.memory.is_empty() {
            self.start_from_best_remembered();
        }
        self.iterate(Phase::Two)?;
        if self.memory_cap > 0 {
            self.remember_current()?;
        }
        self.duals_current = false;
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

    /// The current basis, one column index per position; `>= n` is an
    /// artificial column.
    pub(in crate::inference::sensitivity) fn basis(&self) -> &[usize] {
        &self.basis
    }

    /// The basic values, by basis position.
    pub(in crate::inference::sensitivity) fn basic_values(&self) -> &[f64] {
        &self.xb
    }

    pub(in crate::inference::sensitivity) const fn num_columns(&self) -> usize {
        self.n
    }

    /// The interval of `t` on which the current basis stays optimal for the
    /// costs `c0 + t c1`.
    ///
    /// A basis is optimal when every nonbasic reduced cost is at most the
    /// pricing tolerance, and for affine costs each reduced cost is affine in
    /// `t`, so the interval is read off two reduced-cost vectors. Meant to be
    /// called right after a solve at some `t` in the interval; `y0` and `y1`
    /// are scratch of length `m`.
    pub(in crate::inference::sensitivity) fn optimality_interval(
        &self,
        c0: &[f64],
        c1: &[f64],
        y0: &mut Vec<f64>,
        y1: &mut Vec<f64>,
    ) -> (f64, f64) {
        let m = self.m;
        y0.clear();
        y0.resize(m, 0.0);
        y1.clear();
        y1.resize(m, 0.0);
        for (r, &j) in self.basis.iter().enumerate() {
            if j >= self.n {
                continue;
            }
            let row = &self.binv[r * m..(r + 1) * m];
            let (cb0, cb1) = (c0[j], c1[j]);
            if cb0 != 0.0 {
                for (y, b) in y0.iter_mut().zip(row) {
                    *y += cb0 * b;
                }
            }
            if cb1 != 0.0 {
                for (y, b) in y1.iter_mut().zip(row) {
                    *y += cb1 * b;
                }
            }
        }
        let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
        for j in 0..self.n {
            if self.is_basic[j] {
                continue;
            }
            let d0 = c0[j] - dot(self.column(j), y0);
            let d1 = c1[j] - dot(self.column(j), y1);
            // d0 + t d1 <= TOL
            if d1 > 0.0 {
                hi = hi.min((TOL_REDUCED_COST - d0) / d1);
            } else if d1 < 0.0 {
                lo = lo.max((TOL_REDUCED_COST - d0) / d1);
            }
        }
        (lo, hi)
    }

    /// The objective of a basis under the current costs: `Σ c_j x_Bj` over its
    /// original columns.
    fn basis_objective(&self, basis: &[usize], xb: &[f64]) -> f64 {
        basis
            .iter()
            .zip(xb)
            .filter(|(j, _)| **j < self.n)
            .map(|(j, x)| self.cost[*j] * x)
            .sum()
    }

    fn start_from_best_remembered(&mut self) {
        let current = self.basis_objective(&self.basis, &self.xb);
        let mut best = None;
        let mut best_value = current;
        for (index, entry) in self.memory.iter().enumerate() {
            let value = self.basis_objective(&entry.basis, &entry.xb);
            if value > best_value {
                best_value = value;
                best = Some(index);
            }
        }
        let Some(index) = best else {
            return;
        };
        let entry = &self.memory[index];
        self.basis.copy_from_slice(&entry.basis);
        self.binv.copy_from_slice(&entry.binv);
        self.xb.copy_from_slice(&entry.xb);
        self.pivots_since_refactor = entry.pivots_since_refactor;
        self.is_basic.fill(false);
        for &j in &self.basis {
            self.is_basic[j] = true;
        }
    }

    fn remember_current(&mut self) -> Result<(), LpError> {
        self.key_scratch.copy_from_slice(&self.basis);
        self.key_scratch.sort_unstable();
        if self
            .memory
            .iter()
            .any(|entry| entry.key == self.key_scratch)
        {
            return Ok(());
        }
        // A stored inverse is reloaded and pivoted on again and again, so it
        // goes in fresh when it is more than half way to its next
        // factorisation; otherwise drift could accumulate across cycles.
        if self.pivots_since_refactor > REFACTOR_EVERY / 2 {
            self.refactor()?;
        }
        let entry = RememberedBasis {
            key: self.key_scratch.clone(),
            basis: self.basis.clone(),
            binv: self.binv.clone(),
            xb: self.xb.clone(),
            pivots_since_refactor: self.pivots_since_refactor,
        };
        if self.memory.len() < self.memory_cap {
            self.memory.push(entry);
        } else {
            self.memory[self.memory_next] = entry;
            self.memory_next = (self.memory_next + 1) % self.memory_cap;
        }
        Ok(())
    }

    /// Phase one on its own: find a feasible basis without a cost vector, so
    /// that a workspace can be cloned into many with the work done once.
    /// Feasibility does not depend on the costs, so this is never wasted.
    ///
    /// # Errors
    /// `Infeasible` when no `λ ≥ 0` satisfies `A λ = g`.
    pub(in crate::inference::sensitivity) fn prepare(&mut self) -> Result<(), LpError> {
        if self.has_feasible_basis {
            return Ok(());
        }
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
        // `y` is rebuilt from the basis here and after every refactorisation,
        // and moved along with each pivot in between: the entering column's
        // reduced cost times the scaled pivot row is exactly the change in
        // `B^{-T} c_B`, and it costs `m` operations instead of `m²`.
        let mut duals_stale = true;
        for _ in 0..limit {
            if duals_stale {
                self.compute_duals(phase);
                duals_stale = false;
            }
            // Pricing. Only original columns may enter: artificials start
            // basic and, once out, have no business coming back.
            let mut entering = None;
            let mut best = TOL_REDUCED_COST;
            for j in 0..n {
                if self.is_basic[j] {
                    continue;
                }
                let d = self.cost_of(j, phase) - dot(self.column(j), &self.y);
                if bland {
                    if d > TOL_REDUCED_COST {
                        entering = Some(j);
                        best = d;
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
            let d_q = best;
            // col = B^{-1} a_q
            for i in 0..m {
                self.col[i] = dot(&self.binv[i * m..(i + 1) * m], &self.a[q * m..(q + 1) * m]);
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
            for (y, pivot_row) in self.y.iter_mut().zip(&self.pivot_row) {
                *y += d_q * pivot_row;
            }
            if self.pivots_since_refactor >= REFACTOR_EVERY {
                self.refactor()?;
                duals_stale = true;
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
        for (scratch, value) in self
            .pivot_row
            .iter_mut()
            .zip(&self.binv[r * m..(r + 1) * m])
        {
            *scratch = value * inv_p;
        }
        for i in 0..m {
            let factor = self.col[i];
            if i == r || factor == 0.0 {
                continue;
            }
            for (target, pivot_row) in self.binv[i * m..(i + 1) * m]
                .iter_mut()
                .zip(&self.pivot_row)
            {
                *target -= factor * pivot_row;
            }
        }
        self.binv[r * m..(r + 1) * m].copy_from_slice(&self.pivot_row);
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

/// Four independent accumulators, so the multiply-adds pipeline instead of
/// serialising on one sum. The summation order differs from a plain fold, which
/// moves a reduced cost by a rounding unit and nothing a pivot choice can see
/// except on an exact tie.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0_f64; 4];
    let (chunks_a, rest_a) = a.as_chunks::<4>();
    let (chunks_b, rest_b) = b.as_chunks::<4>();
    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        acc[0] += ca[0] * cb[0];
        acc[1] += ca[1] * cb[1];
        acc[2] += ca[2] * cb[2];
        acc[3] += ca[3] * cb[3];
    }
    let mut tail = 0.0;
    for (x, y) in rest_a.iter().zip(rest_b) {
        tail += x * y;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
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
