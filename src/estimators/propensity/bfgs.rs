//! BFGS-based propensity score fitters.
//!
//! Replaces the raw Newton steps used by the calibration and IPT paths with
//! BFGS + More-Thuente line search via the `argmin` framework. Wolfe conditions
//! on each line search guarantee sufficient decrease and curvature, giving
//! global convergence that the original Newton-only code could not ensure.
//!
//! Both objectives are strictly convex, so BFGS converges to the unique global
//! minimum from any starting point.

use argmin::core::{CostFunction, Error as ArgminError, Executor, Gradient, IterState, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::BFGS;
use faer::{Mat, MatRef};

use crate::error::InternalDidError;
use crate::estimators::propensity::common::group_binary_design_rows;

// ─────────────────────────────────────────────────────────
// Faer ↔ vector conversions (boundary only)
// ─────────────────────────────────────────────────────────

fn mat_to_vec(m: MatRef<'_, f64>) -> Vec<f64> {
    (0..m.nrows()).map(|row| *m.get(row, 0)).collect()
}

fn vec_to_mat(values: &[f64]) -> Mat<f64> {
    Mat::from_fn(values.len(), 1, |i, _| values[i])
}

type BfgsIterState = IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64>;

fn argmin_to_did(e: &ArgminError) -> InternalDidError {
    InternalDidError::Estimation(e.to_string())
}

// ─────────────────────────────────────────────────────────
// Calibration objective  (replaces the fake trust-region Newton)
//
// min_β  −Σ_treated x′β / n  +  Σ_control exp(x′β) / n
//
// This is the calibrated propensity score criterion from
// Sant'Anna & Zhao (2020). The Hessian Σ_control exp(x′β) x x′ / n is PSD,
// so the objective is convex and BFGS converges globally.
// ─────────────────────────────────────────────────────────

struct CalibrationProblem<'a> {
    x: MatRef<'a, f64>,
    treated_counts: &'a [f64],
    control_counts: &'a [f64],
    n_treated: f64,
    n_control: f64,
}

impl CalibrationProblem<'_> {
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn eval(&self, param: &[f64]) -> (f64, Vec<f64>) {
        let p = self.x.ncols();
        let n_treated = self.n_treated.max(1.0);
        let n_control = self.n_control.max(1.0);
        let mut grad = vec![0.0; p];
        let mut loss = 0.0;

        for i in 0..self.x.nrows() {
            let lp: f64 = (0..p).map(|j| *self.x.get(i, j) * param[j]).sum();
            let treated_count = self.treated_counts[i];
            let control_count = self.control_counts[i];
            if treated_count > 0.0 {
                let scale = treated_count / n_treated;
                loss = lp.mul_add(-scale, loss);
                for (j, grad_j) in grad.iter_mut().enumerate().take(p) {
                    *grad_j = (*self.x.get(i, j)).mul_add(-scale, *grad_j);
                }
            }
            if control_count > 0.0 {
                let e = lp.exp();
                let scale = control_count / n_control;
                loss = e.mul_add(scale, loss);
                for (j, grad_j) in grad.iter_mut().enumerate().take(p) {
                    *grad_j = (*self.x.get(i, j) * e).mul_add(scale, *grad_j);
                }
            }
        }
        (loss, grad)
    }
}

impl CostFunction for CalibrationProblem<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Vec<f64>) -> Result<f64, ArgminError> {
        Ok(self.eval(p).0)
    }
}

impl Gradient for CalibrationProblem<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, ArgminError> {
        Ok(self.eval(p).1)
    }
}

/// Fit calibrated propensity scores via BFGS + More-Thuente line search.
///
/// # Errors
/// Returns [`InternalDidError`] if the line search or solver setup fails, or if
/// BFGS produces no usable parameter estimate within `max_iter` iterations.
pub fn fit_bfgs_calibration(
    x: MatRef<'_, f64>,
    d: MatRef<'_, f64>,
    beta0: MatRef<'_, f64>,
    max_iter: u64,
) -> Result<Mat<f64>, InternalDidError> {
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn run_calibration_bfgs(
        problem: CalibrationProblem<'_>,
        solver: BFGS<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>, f64>,
        initial: Vec<f64>,
        max_iter: u64,
    ) -> Result<Vec<f64>, InternalDidError> {
        let result = Executor::new(problem, solver)
            .configure(|state: BfgsIterState| state.param(initial).max_iters(max_iter))
            .run()
            .map_err(|err| argmin_to_did(&err))?;
        result.state().get_best_param().cloned().ok_or_else(|| {
            InternalDidError::Convergence("BFGS calibration produced no solution".to_string())
        })
    }

    let grouped = group_binary_design_rows(x, d);
    let linesearch = MoreThuenteLineSearch::<Vec<f64>, Vec<f64>, f64>::new()
        .with_c(1e-4, 0.9)
        .map_err(|err| argmin_to_did(&err))?;
    let solver = BFGS::new(linesearch);
    let beta = run_calibration_bfgs(
        CalibrationProblem {
            x: grouped.design.as_ref(),
            treated_counts: &grouped.treated_counts,
            control_counts: &grouped.control_counts,
            n_treated: grouped.treated_counts.iter().sum::<f64>(),
            n_control: grouped.control_counts.iter().sum::<f64>(),
        },
        solver,
        mat_to_vec(beta0),
        max_iter,
    )?;
    Ok(vec_to_mat(&beta))
}

// ─────────────────────────────────────────────────────────
// IPT (inverse probability tilting) objective
//
// Uses a piecewise-quadratic stabilisation of exp(−v) for v > vstar to avoid
// overflow in the exponential (vstar = 700 by default).
// ─────────────────────────────────────────────────────────

struct IptProblem<'a> {
    x: MatRef<'a, f64>,
    treated_counts: &'a [f64],
    control_counts: &'a [f64],
    vstar: f64,
}

impl IptProblem<'_> {
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn eval(&self, param: &[f64]) -> (f64, Vec<f64>) {
        let n = self.x.nrows();
        let p = self.x.ncols();
        let vstar = self.vstar;
        let ev = (-vstar).exp();
        let an = -vstar - ev;
        let bn = -1.0 + ev;
        let cn = -ev;
        let mut loss = 0.0;
        let mut grad = vec![0.0; p];
        for i in 0..n {
            let lp: f64 = (0..p).map(|j| *self.x.get(i, j) * param[j]).sum();
            let treated_count = self.treated_counts[i];
            let control_count = self.control_counts[i];
            let (phi, phi_p) = if lp < vstar {
                let e = (-lp).exp();
                (-lp - e, -1.0 + e)
            } else {
                (cn.mul_add(0.5 * lp, bn).mul_add(lp, an), cn.mul_add(lp, bn))
            };
            loss -= control_count.mul_add(phi, treated_count * lp);
            let gs = control_count.mul_add(-phi_p, treated_count);
            for (j, grad_j) in grad.iter_mut().enumerate().take(p) {
                *grad_j = gs.mul_add(*self.x.get(i, j), *grad_j);
            }
        }
        (loss, grad)
    }
}

impl CostFunction for IptProblem<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Vec<f64>) -> Result<f64, ArgminError> {
        Ok(self.eval(p).0)
    }
}

impl Gradient for IptProblem<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, ArgminError> {
        Ok(self.eval(p).1)
    }
}

/// Fit IPT propensity scores via BFGS + More-Thuente line search.
///
/// # Errors
/// Returns [`InternalDidError`] if setup or the solver itself fails.
pub fn fit_bfgs_ipt(
    x: MatRef<'_, f64>,
    d: MatRef<'_, f64>,
    beta0: MatRef<'_, f64>,
    vstar: f64,
    max_iter: u64,
) -> Result<Mat<f64>, InternalDidError> {
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn run_ipt_bfgs(
        problem: IptProblem<'_>,
        solver: BFGS<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>, f64>,
        initial: Vec<f64>,
        max_iter: u64,
    ) -> Result<Vec<f64>, InternalDidError> {
        let result = Executor::new(problem, solver)
            .configure(|state: BfgsIterState| state.param(initial).max_iters(max_iter))
            .run()
            .map_err(|err| argmin_to_did(&err))?;
        result.state().get_best_param().cloned().ok_or_else(|| {
            InternalDidError::Convergence("BFGS IPT produced no solution".to_string())
        })
    }

    let grouped = group_binary_design_rows(x, d);
    let linesearch = MoreThuenteLineSearch::<Vec<f64>, Vec<f64>, f64>::new()
        .with_c(1e-4, 0.9)
        .map_err(|err| argmin_to_did(&err))?;
    let solver = BFGS::new(linesearch);
    let beta = run_ipt_bfgs(
        IptProblem {
            x: grouped.design.as_ref(),
            treated_counts: &grouped.treated_counts,
            control_counts: &grouped.control_counts,
            vstar,
        },
        solver,
        mat_to_vec(beta0),
        max_iter,
    )?;
    Ok(vec_to_mat(&beta))
}
