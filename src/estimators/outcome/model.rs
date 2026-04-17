use faer::{Mat, MatRef};

pub trait OutcomeModel {
    fn fit(&self, x: MatRef<'_, f64>, y: MatRef<'_, f64>, w: Option<&[f64]>) -> Mat<f64>;
    fn predict(&self, x: MatRef<'_, f64>, beta: MatRef<'_, f64>) -> Vec<f64>;
}
