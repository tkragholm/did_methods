use faer::{Mat, MatRef};

use super::bfgs::{fit_bfgs_calibration, fit_bfgs_ipt};
use super::common::empirical_logit;
use super::irls::irls;
use super::types::{Config, Params, PropensityEstimator};
use crate::error::InternalDidError;

pub struct LogisticPS {
    pub cfg: Config,
}

impl LogisticPS {
    #[must_use]
    pub const fn new(cfg: Config) -> Self {
        Self { cfg }
    }
}

impl PropensityEstimator for LogisticPS {
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn fit(
        &self,
        design: MatRef<'_, f64>,
        target: MatRef<'_, f64>,
    ) -> Result<Params, InternalDidError> {
        let beta0 = initial_beta(design, target);

        // Primary: BFGS on the calibration (Sant'Anna–Zhao) objective.
        // Wolfe line search guarantees descent; converges globally on the
        // convex objective unlike the previous Newton-only path.
        let calibration_result =
            fit_bfgs_calibration(design, target, beta0.as_ref(), self.cfg.max_iter);
        if let Ok(beta) = calibration_result {
            return Ok(Params { beta });
        }
        let calibration_err = calibration_result.err().unwrap_or_else(|| {
            InternalDidError::Estimation("unknown calibration failure".to_string())
        });

        // Secondary: BFGS on the stabilised IPT objective.
        // The piecewise-quadratic extension keeps the objective smooth even
        // for large linear predictors.
        let ipt_result = fit_bfgs_ipt(
            design,
            target,
            beta0.as_ref(),
            self.cfg.vstar,
            self.cfg.max_iter,
        );
        if let Ok(beta) = ipt_result {
            return Ok(Params { beta });
        }
        let ipt_err = ipt_result
            .err()
            .unwrap_or_else(|| InternalDidError::Estimation("unknown IPT failure".to_string()));

        // Final fallback: IRLS (Fisher scoring). Always returns a finite
        // estimate or an explicit convergence error.
        let beta = irls(
            design,
            target,
            self.cfg.max_iter,
            self.cfg.tol,
            self.cfg.min_weight,
        )
        .map_err(|irls_err| {
            InternalDidError::Convergence(format!(
                "all logistic propensity solvers failed (calibration: {calibration_err}; IPT: {ipt_err}; IRLS: {irls_err})"
            ))
        })?;
        Ok(Params { beta })
    }
}

fn initial_beta(x: MatRef<'_, f64>, d: MatRef<'_, f64>) -> Mat<f64> {
    let mut b = Mat::zeros(x.ncols(), 1);
    *b.get_mut(0, 0) = empirical_logit(d);
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_design() -> Mat<f64> {
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        Mat::from_fn(x.len(), 2, |row, col| if col == 0 { 1.0 } else { x[row] })
    }

    #[test]
    fn initial_beta_handles_empirical_logit_branches() {
        let design = simple_design();
        let all_zero = Mat::from_fn(5, 1, |_, _| 0.0);
        let all_one = Mat::from_fn(5, 1, |_, _| 1.0);
        let half = Mat::from_fn(5, 1, |row, _| if row < 2 { 0.0 } else { 1.0 });

        let b0 = initial_beta(design.as_ref(), all_zero.as_ref());
        let b1 = initial_beta(design.as_ref(), all_one.as_ref());
        let bh = initial_beta(design.as_ref(), half.as_ref());

        assert!((b0.get(0, 0) + 6.907).abs() < 1e-12);
        assert!((b1.get(0, 0) - 6.907).abs() < 1e-12);
        assert!(bh.get(0, 0).is_finite());
    }

    #[test]
    fn logistic_fit_converges_on_separable_data() {
        let design = simple_design();
        let target = Mat::from_fn(5, 1, |row, _| if row < 2 { 0.0 } else { 1.0 });
        let cfg = Config::default();
        let est = LogisticPS::new(cfg);
        let params = est
            .fit(design.as_ref(), target.as_ref())
            .expect("should converge");
        assert_eq!(params.beta.nrows(), 2);
        assert!(params.beta.col_as_slice(0).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn logistic_fit_reaches_irls_fallback_on_zero_iters() {
        let design = simple_design();
        let target = Mat::from_fn(5, 1, |row, _| if row < 3 { 0.0 } else { 1.0 });
        // max_iter=0 forces BFGS to fail immediately (no iterations allowed),
        // so the IRLS fallback is exercised.
        let cfg = Config {
            max_iter: 0,
            tol: 1e-8,
            min_weight: 1e-8,
            vstar: 700.0,
        };
        let est = LogisticPS::new(cfg);
        // IRLS with 0 iters returns Convergence error
        assert!(est.fit(design.as_ref(), target.as_ref()).is_err());
    }
}
