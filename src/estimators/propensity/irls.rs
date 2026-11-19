use faer::{Mat, MatRef};

use crate::error::InternalDidError;
use crate::estimators::common::logit::fit_logit_collapsed;

/// Fit logistic regression by maximum likelihood.
///
/// This is `glm(family = binomial)` of R, with `tol` as its `epsilon`: the
/// iterations stop when the relative change in the deviance falls below it.
/// Repeated design rows are collapsed first, which does not change the
/// iterates or the stopping iteration.
///
/// # Errors
/// Returns `InternalDidError::Convergence` if the fit fails or does not
/// converge in `max_iter` iterations.
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn irls(
    design: MatRef<'_, f64>,
    target: MatRef<'_, f64>,
    max_iter: u64,
    tol: f64,
) -> Result<Mat<f64>, InternalDidError> {
    if max_iter == 0 {
        return Err(InternalDidError::Convergence(
            "IRLS did not converge".to_string(),
        ));
    }
    let treated: Vec<f64> = (0..target.nrows()).map(|row| target[(row, 0)]).collect();
    let max_iterations = usize::try_from(max_iter).unwrap_or(usize::MAX);
    let fit = fit_logit_collapsed(design, &treated, tol, max_iterations)
        .map_err(|error| InternalDidError::Convergence(format!("logit fit failed: {error}")))?;
    if !fit.converged {
        return Err(InternalDidError::Convergence(
            "IRLS did not converge".to_string(),
        ));
    }
    Ok(Mat::from_fn(fit.coefficients.len(), 1, |row, _| {
        fit.coefficients[row]
    }))
}
