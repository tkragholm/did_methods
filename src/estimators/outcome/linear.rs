use epistat_linalg::{weighted_gram, weighted_xtz};
use faer::{Mat, MatRef};

use crate::estimators::common::linalg::ridge_solve_lu;
use crate::estimators::outcome::model::OutcomeModel;

const INVALID_WEIGHTS: &str = "weights must be finite and non-negative, one per row";

pub struct LinearOutcome {
    pub ridge: f64,
}

impl Default for LinearOutcome {
    fn default() -> Self {
        Self { ridge: 1e-8 }
    }
}

impl OutcomeModel for LinearOutcome {
    fn fit(
        &self,
        design: MatRef<'_, f64>,
        target: MatRef<'_, f64>,
        weights: Option<&[f64]>,
    ) -> Result<Mat<f64>, &'static str> {
        let Some(weights) = weights else {
            let a = design.transpose() * design;
            let b = design.transpose() * target;
            return ridge_solve_lu(a, &b, self.ridge);
        };
        let a = weighted_gram(design, weights).map_err(|_| INVALID_WEIGHTS)?;
        let mut b = Mat::<f64>::zeros(design.ncols(), target.ncols());
        let mut column = vec![0.0; target.nrows()];
        for col in 0..target.ncols() {
            for (row, value) in column.iter_mut().enumerate() {
                *value = target[(row, col)];
            }
            let product = weighted_xtz(design, weights, &column).map_err(|_| INVALID_WEIGHTS)?;
            for row in 0..design.ncols() {
                b[(row, col)] = product[(row, 0)];
            }
        }
        ridge_solve_lu(a, &b, self.ridge)
    }
    fn predict(&self, design: MatRef<'_, f64>, beta: MatRef<'_, f64>) -> Vec<f64> {
        let n = design.nrows();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut pred = 0.0;
            for j in 0..design.ncols() {
                pred += design.get(i, j) * beta.get(j, 0);
            }
            out.push(pred);
        }
        out
    }
}
