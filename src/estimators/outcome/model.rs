use faer::{Mat, MatRef};

pub trait OutcomeModel {
    /// Fit the model and return its coefficients.
    ///
    /// # Errors
    /// Returns an error if the coefficients are not finite.
    fn fit(
        &self,
        x: MatRef<'_, f64>,
        y: MatRef<'_, f64>,
        w: Option<&[f64]>,
    ) -> Result<Mat<f64>, &'static str>;
    fn predict(&self, x: MatRef<'_, f64>, beta: MatRef<'_, f64>) -> Vec<f64>;
}
