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
}

impl Basis for PolynomialBasis {
    fn eval(&self, x: f64) -> Col<f64> {
        let k = self.num_basis();
        let mut vals = Vec::with_capacity(k);
        let mut power = 1.0;
        for _ in 0..k {
            vals.push(power);
            power *= x;
        }
        Col::from_fn(k, |i| vals[i])
    }

    fn eval_deriv(&self, x: f64) -> Col<f64> {
        let k = self.num_basis();
        let mut vals = Vec::with_capacity(k);
        vals.push(0.0);

        let mut coeff = 1.0;
        let mut power = 1.0;
        for _ in 1..k {
            vals.push(coeff * power);
            coeff += 1.0;
            power *= x;
        }

        Col::from_fn(k, |i| vals[i])
    }

    fn num_basis(&self) -> usize {
        self.degree + 1
    }
}

/// B-Spline basis implementation.
///
/// Uses the Cox-de Boor recursion formula for evaluation.
#[derive(Debug, Clone)]
pub struct BSplineBasis {
    pub degree: usize,
    pub internal_knots: Vec<f64>,
    pub boundary_knots: (f64, f64),
    /// The full knot vector (augmented with boundary knots).
    knots: Vec<f64>,
}

impl BSplineBasis {
    /// Create a new B-Spline basis.
    ///
    /// # Arguments
    /// * `degree` - Degree of the spline (e.g., 3 for cubic).
    /// * `internal_knots` - Sorted list of internal knot positions.
    /// * `boundary_knots` - (min, max) boundary knots.
    ///
    /// # Notes
    /// This constructor does not validate knot ordering or boundary
    /// consistency; callers are expected to provide a valid knot sequence.
    #[must_use]
    pub fn new(degree: usize, internal_knots: Vec<f64>, boundary_knots: (f64, f64)) -> Self {
        let mut knots = Vec::with_capacity(internal_knots.len() + 2 * (degree + 1));
        // Augmented knots at boundaries
        for _ in 0..=degree {
            knots.push(boundary_knots.0);
        }
        knots.extend_from_slice(&internal_knots);
        for _ in 0..=degree {
            knots.push(boundary_knots.1);
        }
        Self {
            degree,
            internal_knots,
            boundary_knots,
            knots,
        }
    }

    /// Cox-de Boor recursion for B-spline evaluation.
    fn b_spline(&self, i: usize, k: usize, x: f64) -> f64 {
        if k == 0 {
            if x >= self.knots[i] && x < self.knots[i + 1] {
                return 1.0;
            }
            // Handle last point boundary
            if i == self.num_basis() - 1 && (x - self.knots[i + 1]).abs() < 1e-12 {
                return 1.0;
            }
            return 0.0;
        }

        let mut val = 0.0;
        let den1 = self.knots[i + k] - self.knots[i];
        if den1 > 0.0 {
            val = ((x - self.knots[i]) / den1).mul_add(self.b_spline(i, k - 1, x), val);
        }
        let den2 = self.knots[i + k + 1] - self.knots[i + 1];
        if den2 > 0.0 {
            val = ((self.knots[i + k + 1] - x) / den2).mul_add(self.b_spline(i + 1, k - 1, x), val);
        }
        val
    }
}

impl Basis for BSplineBasis {
    fn eval(&self, x: f64) -> Col<f64> {
        let n = self.num_basis();
        Col::from_fn(n, |i| self.b_spline(i, self.degree, x))
    }

    fn eval_deriv(&self, x: f64) -> Col<f64> {
        // Derivative of B-spline of degree p is a linear combination of B-splines of degree p-1
        let n = self.num_basis();
        if self.degree == 0 {
            return Col::zeros(n);
        }

        let p = f64::from(u32::try_from(self.degree).expect("spline degree exceeds u32::MAX"));
        // Cache lower-degree basis values once to avoid repeated recursive work.
        let lower_degree_vals: Vec<f64> = (0..=n)
            .map(|i| self.b_spline(i, self.degree - 1, x))
            .collect();

        Col::from_fn(n, |i| {
            let mut val = 0.0;
            let den1 = self.knots[i + self.degree] - self.knots[i];
            if den1 > 0.0 {
                // d/dx B_{i,p}(x) = p * [ B_{i,p-1}/(t_{i+p}-t_i) - B_{i+1,p-1}/(t_{i+p+1}-t_{i+1}) ]
                val = (p / den1).mul_add(lower_degree_vals[i], val);
            }
            let den2 = self.knots[i + self.degree + 1] - self.knots[i + 1];
            if den2 > 0.0 {
                val = (p / den2).mul_add(-lower_degree_vals[i + 1], val);
            }
            val
        })
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
        let basis = BSplineBasis::new(3, vec![0.5], (0.0, 1.0));
        for x in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let vals = basis.eval(x);
            let sum: f64 = (0..vals.nrows()).map(|i| vals[i]).sum();
            assert!((sum - 1.0).abs() < 1e-12, "Sum at x={x} was {sum}");
        }
    }
}
