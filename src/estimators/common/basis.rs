//! Basis expansions shared by nonparametric and sieve estimators.
//!
//! These basis functions provide the approximation space used by continuous
//! treatment estimators, most notably the sieve ACRT implementation. The trait
//! is intentionally small: evaluate the basis, evaluate its derivative, and
//! report the number of basis functions.
//!
//! Included bases:
//! - [`PolynomialBasis`], a simple global polynomial basis useful for tests and
//!   small problems.
//! - [`BSplineBasis`], a local spline basis evaluated by the Cox-de Boor
//!   recursion.
//!
//! References:
//! - de Boor, C. (1978). *A Practical Guide to Splines*.
//! - standard sieve / series-estimation treatments in semiparametric
//!   econometrics.

use epistat_linalg::basis::{BSpline, Basis as SharedBasis, Polynomial};
use faer::Col;

/// Trait for functional basis expansions (e.g., B-Splines, Polynomials).
pub trait Basis {
    /// Evaluate the basis functions at point x: [`ψ_1(x)`, ..., `ψ_K(x)`]
    fn eval(&self, x: f64) -> Col<f64>;

    /// Evaluate the derivative of the basis functions at point x.
    /// Used for calculating the Average Causal Response (ACRT).
    fn eval_deriv(&self, x: f64) -> Col<f64>;

    /// Number of basis functions (K).
    fn num_basis(&self) -> usize;
}

fn column(values: &[f64]) -> Col<f64> {
    Col::from_fn(values.len(), |i| values[i])
}

/// A simple polynomial basis: [1, x, x^2, ..., x^(degree)]
#[derive(Debug, Clone, Copy)]
pub struct PolynomialBasis {
    pub degree: usize,
}

impl PolynomialBasis {
    #[must_use]
    pub const fn new(degree: usize) -> Self {
        Self { degree }
    }

    const fn shared(self) -> Polynomial {
        Polynomial {
            degree: self.degree,
        }
    }
}

impl Basis for PolynomialBasis {
    fn eval(&self, x: f64) -> Col<f64> {
        column(&self.shared().values(x))
    }

    fn eval_deriv(&self, x: f64) -> Col<f64> {
        column(&self.shared().derivatives(x))
    }

    fn num_basis(&self) -> usize {
        self.degree + 1
    }
}

/// B-Spline basis implementation, evaluated by the Cox-de Boor recursion.
#[derive(Debug, Clone)]
pub struct BSplineBasis {
    pub degree: usize,
    pub internal_knots: Vec<f64>,
    pub boundary_knots: (f64, f64),
    spline: BSpline,
}

impl BSplineBasis {
    /// Create a new B-Spline basis.
    ///
    /// # Arguments
    /// * `degree` - Degree of the spline (e.g., 3 for cubic).
    /// * `internal_knots` - Sorted list of internal knot positions.
    /// * `boundary_knots` - (min, max) boundary knots.
    ///
    /// # Errors
    /// If the boundary knots are not finite and increasing, or the internal
    /// knots are not sorted or do not lie strictly inside the boundary.
    pub fn new(
        degree: usize,
        internal_knots: Vec<f64>,
        boundary_knots: (f64, f64),
    ) -> Result<Self, epistat_linalg::basis::KnotError> {
        let spline = BSpline::new(degree, &internal_knots, boundary_knots.0, boundary_knots.1)?;
        Ok(Self {
            degree,
            internal_knots,
            boundary_knots,
            spline,
        })
    }
}

impl Basis for BSplineBasis {
    fn eval(&self, x: f64) -> Col<f64> {
        column(&self.spline.values(x))
    }

    fn eval_deriv(&self, x: f64) -> Col<f64> {
        column(&self.spline.derivatives(x))
    }

    fn num_basis(&self) -> usize {
        self.internal_knots.len() + self.degree + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_basis_eval_works() {
        let basis = PolynomialBasis::new(2);
        let vals = basis.eval(2.0);
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 2.0).abs() < 1e-12);
        assert!((vals[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn bspline_basis_eval_sums_to_one() {
        // B-splines partition of unity property
        let basis = BSplineBasis::new(3, vec![0.5], (0.0, 1.0)).expect("valid knots");
        for x in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let vals = basis.eval(x);
            let sum: f64 = (0..vals.nrows()).map(|i| vals[i]).sum();
            assert!((sum - 1.0).abs() < 1e-12, "Sum at x={x} was {sum}");
        }
    }
}
