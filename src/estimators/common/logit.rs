//! Maximum-likelihood logistic regression, through `epistat_glm`.

use std::collections::HashMap;

use epistat_glm::{Family, Glm, GlmError, Link};
use epistat_linalg::{Attempt, solve};
use faer::{Mat, MatRef};

/// The coefficients of a logistic fit and how the iterations ended.
#[derive(Debug, Clone)]
pub struct LogitFit {
    /// One per column of the design. A column that is a combination of the
    /// columns before it is aliased and has coefficient 0, which leaves the
    /// fitted values as they are.
    pub coefficients: Vec<f64>,
    pub iterations: usize,
    /// Whether the relative change in the deviance fell below the tolerance.
    pub converged: bool,
}

/// The logit fit of `treated` (0 or 1) on `design`, with optional prior
/// weights, as R's `glm(family = binomial)` does it with its `epsilon` set to
/// `tolerance` and `maxit` to `max_iterations`.
///
/// # Errors
/// Returns the [`GlmError`] of the fit.
pub fn fit_logit(
    design: MatRef<'_, f64>,
    treated: &[f64],
    weights: Option<&[f64]>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<LogitFit, GlmError> {
    let model = Glm {
        tolerance,
        max_iterations,
        ..Glm::new(Family::Binomial, Link::Logit)
    };
    let fit = model.fit(design, treated, weights, None, None)?;
    Ok(LogitFit {
        coefficients: fit
            .coefficients
            .iter()
            .map(|value| value.unwrap_or(0.0))
            .collect(),
        iterations: fit.iterations,
        converged: fit.converged,
    })
}

/// The unweighted logit fit of rows that repeat, with R's iterates.
///
/// Rows with the same design row are collapsed to one row with a count and a
/// number of successes. Grouped and ungrouped IRLS take the same steps, but
/// the grouped deviance lacks the saturated term of the ungrouped one, so
/// `glm`'s test `|D - D_old| / (|D| + 0.1) < epsilon` would stop at another
/// iteration. This fit takes the first step from R's starting means
/// `(y + 0.5) / 2`, then steps with [`Glm::fit`] from the coefficients so far,
/// and applies the test to the ungrouped deviance. It therefore stops at the
/// iteration `glm` stops at on the ungrouped rows, and gives the same
/// coefficients. Step halving in a step is limited to one halving.
///
/// # Errors
/// Returns the [`GlmError`] of a step, or of the ungrouped fit that replaces
/// this one when the first step cannot be solved.
pub fn fit_logit_collapsed(
    design: MatRef<'_, f64>,
    treated: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<LogitFit, GlmError> {
    let (grouped, counts, successes) = group_rows(design, treated);
    let (groups, p) = (grouped.nrows(), grouped.ncols());
    let shares: Vec<f64> = (0..groups).map(|g| successes[g] / counts[g]).collect();
    let fallback = || fit_logit(design, treated, None, tolerance, max_iterations);

    // The first step, from the starting means: W = 3/16 for every row and
    // z = +-(ln 3 + 4/3) by the response.
    let c = 3.0_f64.ln() + 4.0 / 3.0;
    let w0 = 0.1875;
    let weights: Vec<f64> = counts.iter().map(|n| w0 * n).collect();
    let Ok(gram) = epistat_linalg::weighted_gram(grouped.as_ref(), &weights) else {
        return fallback();
    };
    let z: Vec<f64> = (0..groups)
        .map(|g| c * (2.0 * successes[g] - counts[g]) / counts[g])
        .collect();
    let Ok(rhs) = epistat_linalg::weighted_xtz(grouped.as_ref(), &weights, &z) else {
        return fallback();
    };
    let Ok(first) = solve(gram.as_ref(), rhs.as_ref(), &Attempt::SYMMETRIC) else {
        return fallback();
    };
    let mut beta: Vec<f64> = (0..p).map(|j| first.x[(j, 0)]).collect();

    let total: f64 = counts.iter().sum();
    let mut previous = total * 2.0 * (4.0_f64 / 3.0).ln();
    let model = Glm {
        max_iterations: 1,
        ..Glm::new(Family::Binomial, Link::Logit)
    };
    for iteration in 1..=max_iterations {
        if iteration > 1 {
            let step = model.fit(grouped.as_ref(), &shares, Some(&counts), None, Some(&beta))?;
            beta = step
                .coefficients
                .iter()
                .map(|value| value.unwrap_or(0.0))
                .collect();
        }
        let deviance = ungrouped_deviance(grouped.as_ref(), &beta, &counts, &successes);
        if !deviance.is_finite() {
            return fallback();
        }
        if (deviance - previous).abs() / (deviance.abs() + 0.1) < tolerance {
            return Ok(LogitFit {
                coefficients: beta,
                iterations: iteration,
                converged: true,
            });
        }
        previous = deviance;
    }
    Ok(LogitFit {
        coefficients: beta,
        iterations: max_iterations,
        converged: false,
    })
}

/// `-2 sum_g (s_g ln mu_g + (n_g - s_g) ln(1 - mu_g))`, the deviance of the
/// ungrouped rows, with the means kept off 0 and 1 by machine epsilon as R's
/// logit inverse does.
fn ungrouped_deviance(grouped: MatRef<'_, f64>, beta: &[f64], n: &[f64], s: &[f64]) -> f64 {
    let mut total = 0.0;
    for g in 0..grouped.nrows() {
        let eta: f64 = (0..grouped.ncols())
            .map(|j| grouped[(g, j)] * beta[j])
            .sum();
        let mu = if eta < -30.0 {
            f64::EPSILON
        } else if eta > 30.0 {
            1.0 / f64::EPSILON
        } else {
            eta.exp()
        };
        let mu = mu / (1.0 + mu);
        let mu = mu.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
        total += s[g] * mu.ln() + (n[g] - s[g]) * (1.0 - mu).ln();
    }
    -2.0 * total
}

fn normalized_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}

/// The distinct design rows, with the number of rows and of successes of each.
fn group_rows(design: MatRef<'_, f64>, treated: &[f64]) -> (Mat<f64>, Vec<f64>, Vec<f64>) {
    let p = design.ncols();
    let mut index = HashMap::<Vec<u64>, usize>::new();
    let mut rows = Vec::<Vec<f64>>::new();
    let (mut counts, mut successes) = (Vec::<f64>::new(), Vec::<f64>::new());
    let mut key = vec![0_u64; p];
    for row in 0..design.nrows() {
        for col in 0..p {
            key[col] = normalized_bits(design[(row, col)]);
        }
        let group = *index.entry(key.clone()).or_insert_with(|| {
            rows.push((0..p).map(|col| design[(row, col)]).collect());
            counts.push(0.0);
            successes.push(0.0);
            rows.len() - 1
        });
        counts[group] += 1.0;
        if treated[row] > 0.5 {
            successes[group] += 1.0;
        }
    }
    let grouped = Mat::from_fn(rows.len(), p, |row, col| rows[row][col]);
    (grouped, counts, successes)
}
