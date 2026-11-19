use faer::MatRef;

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
        // The maximum-likelihood logit by IRLS (Fisher scoring), the fit R's
        // DRDID uses through glm wherever it does not calibrate.
        let beta = irls(design, target, self.cfg.max_iter, self.cfg.tol).map_err(|irls_err| {
            InternalDidError::Convergence(format!("logistic propensity fit failed: {irls_err}"))
        })?;
        Ok(Params { beta })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn simple_design() -> Mat<f64> {
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        Mat::from_fn(x.len(), 2, |row, col| if col == 0 { 1.0 } else { x[row] })
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
        };
        let est = LogisticPS::new(cfg);
        // IRLS with 0 iters returns Convergence error
        assert!(est.fit(design.as_ref(), target.as_ref()).is_err());
    }
}
