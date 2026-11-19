//! Calibrated propensity score shared by the improved `DR-DiD` estimators.
//!
//! The score follows `DRDID::pscore.cal` (Sant'Anna and Zhao, 2020). The
//! weighted logit maximum likelihood fit is the starting point. The score is
//! then the minimiser of the calibration loss
//!
//! ```text
//! L(g) = -mean_i( w_i * (D_i * x_i'g - (1 - D_i) * exp(x_i'g)) ),
//! ```
//!
//! which is convex with an analytic gradient and Hessian, so Newton steps with
//! step halving reach its minimiser. When that iteration does not converge,
//! the inverse probability tilting loss of Graham, Pinto and Egel (2012), with
//! its sample-size dependent quadratic extension beyond `ln(n - 1)`, is
//! minimised from the same start. When that fails as well, the logit
//! coefficients are used. Designs are row-major with the intercept in column 0.

use epistat_linalg::{weighted_gram, weighted_xtz};
use faer::MatRef;

use crate::estimators::common::linalg::solve_dense_system;
use crate::estimators::common::logit::fit_logit;
use crate::types::DrDidError;
use crate::util::usize_to_f64;

const NEWTON_MAX_ITER: usize = 1_000;
const HALVINGS: usize = 40;
const STEP_TOL: f64 = 1e-12;
const LOGIT_MAX_ITER: usize = 200;
const LOGIT_TOLERANCE: f64 = 1e-12;

/// Which fit supplied the coefficients of a propensity score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PscoreMethod {
    Calibration,
    InverseProbabilityTilting,
    Logit,
}

pub(super) struct CalibratedPscore {
    pub(super) scores: Vec<f64>,
    // Which branch produced the scores; read by tests.
    #[allow(dead_code)]
    pub(super) method: PscoreMethod,
}

/// Fit the calibrated propensity score of every row.
///
/// `weights` are the sampling weights, scaled to mean one by the caller.
pub(super) fn calibrated_propensity_scores(
    design: &[f64],
    p: usize,
    treated: &[f64],
    weights: &[f64],
) -> Result<CalibratedPscore, DrDidError> {
    let logit = fit_weighted_logit(design, p, treated, weights)?;
    let (coefficients, method) =
        if let Some(g) = minimise_calibration(design, p, treated, weights, &logit) {
            (g, PscoreMethod::Calibration)
        } else if let Some(g) = minimise_tilting(design, p, treated, weights, &logit) {
            (g, PscoreMethod::InverseProbabilityTilting)
        } else {
            (logit, PscoreMethod::Logit)
        };
    let scores = linear_index(design, p, &coefficients)
        .into_iter()
        .map(|v| 1.0 / (1.0 + (-v).exp()))
        .collect::<Vec<_>>();
    if scores.iter().any(|s| s.is_nan()) {
        return Err(DrDidError::SingularSystem);
    }
    Ok(CalibratedPscore { scores, method })
}

/// Weighted least squares of `y` on the design over the rows where `keep`
/// holds. Returns the coefficients.
pub(super) fn weighted_least_squares(
    design: &[f64],
    p: usize,
    y: &[f64],
    weights: &[f64],
    keep: impl Fn(usize) -> bool,
) -> Result<Vec<f64>, DrDidError> {
    let n = y.len();
    let kept_weights: Vec<f64> = (0..n)
        .map(|i| if keep(i) { weights[i] } else { 0.0 })
        .collect();
    let x = MatRef::from_row_major_slice(design, n, p);
    let gram = weighted_gram(x, &kept_weights).map_err(|_| DrDidError::SingularSystem)?;
    let rhs = weighted_xtz(x, &kept_weights, y).map_err(|_| DrDidError::SingularSystem)?;
    let gram_flat: Vec<f64> = (0..p * p).map(|k| gram[(k / p, k % p)]).collect();
    let rhs_flat: Vec<f64> = (0..p).map(|k| rhs[(k, 0)]).collect();
    solve_dense_system(&gram_flat, &rhs_flat).map_err(|_| DrDidError::SingularSystem)
}

/// The fitted values `x_i'b` of every row.
pub(super) fn linear_index(design: &[f64], p: usize, beta: &[f64]) -> Vec<f64> {
    design
        .chunks_exact(p)
        .map(|row| row.iter().zip(beta).map(|(x, b)| x * b).sum())
        .collect()
}

fn symmetrise(gram: &mut [f64], p: usize) {
    for a in 0..p {
        for b in 0..a {
            gram[b * p + a] = gram[a * p + b];
        }
    }
}

/// The weighted logit maximum likelihood fit, with the iterations stopped by
/// the relative change in the deviance or after `LOGIT_MAX_ITER`.
fn fit_weighted_logit(
    design: &[f64],
    p: usize,
    treated: &[f64],
    weights: &[f64],
) -> Result<Vec<f64>, DrDidError> {
    let x = MatRef::from_row_major_slice(design, treated.len(), p);
    fit_logit(x, treated, Some(weights), LOGIT_TOLERANCE, LOGIT_MAX_ITER)
        .map(|fit| fit.coefficients)
        .map_err(|_| DrDidError::SingularSystem)
}

/// A twice differentiable loss with its value, gradient and Hessian.
trait Loss {
    fn value(&self, gamma: &[f64]) -> f64;
    fn derivatives(&self, gamma: &[f64]) -> (Vec<f64>, Vec<f64>);
}

fn newton(loss: &impl Loss, start: &[f64]) -> Option<Vec<f64>> {
    let p = start.len();
    let mut gamma = start.to_vec();
    let mut value = loss.value(&gamma);
    if !value.is_finite() {
        return None;
    }
    for _ in 0..NEWTON_MAX_ITER {
        let (gradient, hessian) = loss.derivatives(&gamma);
        if gradient.iter().any(|g| !g.is_finite()) || hessian.iter().any(|h| !h.is_finite()) {
            return None;
        }
        let step = solve_dense_system(&hessian, &gradient).ok()?;
        let scale = 1.0 + gamma.iter().map(|g| g.abs()).fold(0.0, f64::max);
        let step_size = step.iter().map(|s| s.abs()).fold(0.0, f64::max);
        let mut factor = 1.0;
        let mut moved = false;
        for _ in 0..HALVINGS {
            let candidate: Vec<f64> = (0..p).map(|k| gamma[k] - factor * step[k]).collect();
            let candidate_value = loss.value(&candidate);
            if candidate_value.is_finite() && candidate_value <= value {
                gamma = candidate;
                value = candidate_value;
                moved = true;
                break;
            }
            factor *= 0.5;
        }
        if step_size <= STEP_TOL * scale {
            return Some(gamma);
        }
        if !moved {
            let gradient_size = gradient.iter().map(|g| g.abs()).fold(0.0, f64::max);
            return (gradient_size < 1e-9).then_some(gamma);
        }
    }
    None
}

struct Calibration<'a> {
    design: &'a [f64],
    p: usize,
    treated: &'a [f64],
    weights: &'a [f64],
}

impl Loss for Calibration<'_> {
    fn value(&self, gamma: &[f64]) -> f64 {
        let n = usize_to_f64(self.treated.len());
        let mut total = 0.0;
        for (i, row) in self.design.chunks_exact(self.p).enumerate() {
            let index: f64 = row.iter().zip(gamma).map(|(x, g)| x * g).sum();
            let term = if self.treated[i] > 0.5 {
                index
            } else {
                -index.exp()
            };
            total += term * self.weights[i];
        }
        -total / n
    }

    fn derivatives(&self, gamma: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = usize_to_f64(self.treated.len());
        let p = self.p;
        let mut gradient = vec![0.0; p];
        let mut hessian = vec![0.0; p * p];
        for (i, row) in self.design.chunks_exact(p).enumerate() {
            let index: f64 = row.iter().zip(gamma).map(|(x, g)| x * g).sum();
            let e = index.exp();
            let w = self.weights[i];
            let (g_scalar, h_scalar) = if self.treated[i] > 0.5 {
                (-w, 0.0)
            } else {
                (w * e, w * e)
            };
            for a in 0..p {
                gradient[a] += g_scalar * row[a];
                if h_scalar != 0.0 {
                    for b in 0..=a {
                        hessian[a * p + b] += h_scalar * row[a] * row[b];
                    }
                }
            }
        }
        for g in &mut gradient {
            *g /= n;
        }
        for h in &mut hessian {
            *h /= n;
        }
        symmetrise(&mut hessian, p);
        (gradient, hessian)
    }
}

fn minimise_calibration(
    design: &[f64],
    p: usize,
    treated: &[f64],
    weights: &[f64],
    start: &[f64],
) -> Option<Vec<f64>> {
    newton(
        &Calibration {
            design,
            p,
            treated,
            weights,
        },
        start,
    )
}

struct Tilting<'a> {
    design: &'a [f64],
    p: usize,
    treated: &'a [f64],
    weights: &'a [f64],
    a_n: f64,
    b_n: f64,
    c_n: f64,
    v_star: f64,
}

impl<'a> Tilting<'a> {
    fn new(design: &'a [f64], p: usize, treated: &'a [f64], weights: &'a [f64]) -> Self {
        let n = usize_to_f64(treated.len());
        let l = (n - 1.0).ln();
        Self {
            design,
            p,
            treated,
            weights,
            c_n: -(n - 1.0),
            b_n: -n + (n - 1.0) * l,
            a_n: -(n - 1.0) * (1.0 - l + 0.5 * l * l),
            v_star: l,
        }
    }
}

impl Loss for Tilting<'_> {
    fn value(&self, gamma: &[f64]) -> f64 {
        let mut total = 0.0;
        for (i, row) in self.design.chunks_exact(self.p).enumerate() {
            let v: f64 = row.iter().zip(gamma).map(|(x, g)| x * g).sum();
            let phi = if v < self.v_star {
                -v - v.exp()
            } else {
                self.a_n + self.b_n * v + 0.5 * self.c_n * v * v
            };
            total += self.weights[i] * ((1.0 - self.treated[i]) * phi + v);
        }
        -total
    }

    fn derivatives(&self, gamma: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let p = self.p;
        let mut gradient = vec![0.0; p];
        let mut hessian = vec![0.0; p * p];
        for (i, row) in self.design.chunks_exact(p).enumerate() {
            let v: f64 = row.iter().zip(gamma).map(|(x, g)| x * g).sum();
            let (phi1, phi2) = if v < self.v_star {
                (-1.0 - v.exp(), -v.exp())
            } else {
                (self.b_n + self.c_n * v, self.c_n)
            };
            let c = 1.0 - self.treated[i];
            let g_scalar = -self.weights[i] * (c * phi1 + 1.0);
            let h_scalar = -c * self.weights[i] * phi2;
            for a in 0..p {
                gradient[a] += g_scalar * row[a];
                for b in 0..=a {
                    hessian[a * p + b] += h_scalar * row[a] * row[b];
                }
            }
        }
        symmetrise(&mut hessian, p);
        (gradient, hessian)
    }
}

fn minimise_tilting(
    design: &[f64],
    p: usize,
    treated: &[f64],
    weights: &[f64],
    start: &[f64],
) -> Option<Vec<f64>> {
    newton(&Tilting::new(design, p, treated, weights), start)
}
