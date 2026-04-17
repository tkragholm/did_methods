use faer::{Mat, MatRef};

use crate::error::InternalDidError;

pub trait PropensityEstimator {
    fn fit(&self, x: MatRef<'_, f64>, d: MatRef<'_, f64>) -> Result<Params, InternalDidError>;
}

pub trait MultinomialPropensityEstimator {
    fn fit_multinomial(
        &self,
        x: MatRef<'_, f64>,
        class_labels: &[usize],
        sampling_weights: &[f64],
        class_count: usize,
    ) -> Result<MultinomialParams, InternalDidError>;
}

#[derive(Clone, Debug)]
pub struct Config {
    pub max_iter: u64,
    pub tol: f64,
    pub min_weight: f64,
    pub vstar: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-6,
            min_weight: 1e-10,
            vstar: 700.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Params {
    pub beta: Mat<f64>,
}

#[derive(Clone, Debug)]
pub struct MultinomialParams {
    pub beta: Mat<f64>,
}
