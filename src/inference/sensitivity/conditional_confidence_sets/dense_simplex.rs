//! The dense primal simplex behind the ARP test, from `epistat-optim`.
//!
//! The solver maximises `c'λ` subject to `A λ = g`, `λ ≥ 0`. Both LP
//! workspaces in this module are instances of it. The tests here check it
//! against the `clarabel` interior-point solver, which the shared crate does
//! not depend on.

pub(super) use epistat_optim::simplex::{DenseSimplex, LpError};

#[cfg(test)]
mod tests {
    use clarabel::algebra::CscMatrix;
    use clarabel::solver::{
        DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
    };
    use epistat_rng::Stream;
    use rand::RngExt;

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
    fn bounded_cost(rng: &mut Stream, columns: &[Vec<f64>], m: usize) -> Vec<f64> {
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
        rng: &mut Stream,
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
        let mut rng = Stream::seeded(20260904);
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
        let mut rng = Stream::seeded(7);
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
        let mut rng = Stream::seeded(99);
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
