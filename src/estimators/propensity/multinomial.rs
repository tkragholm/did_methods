use epistat_optim::bfgs::Bfgs;
use faer::{Mat, MatRef};
use std::collections::HashMap;

use super::common::softmax_scores;
use super::types::{Config, MultinomialParams, MultinomialPropensityEstimator};
use crate::error::InternalDidError;

pub struct MultinomialSoftmaxPS {
    pub cfg: Config,
    pub ridge: f64,
}

impl MultinomialSoftmaxPS {
    #[must_use]
    pub const fn new(cfg: Config, ridge: f64) -> Self {
        Self { cfg, ridge }
    }
}

struct MultinomialProblem {
    x: Mat<f64>,
    class_weight_sums: Vec<f64>,
    row_total_weights: Vec<f64>,
    class_count: usize,
    ridge: f64,
    min_weight: f64,
}

impl MultinomialProblem {
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn evaluate(&self, params: &[f64]) -> (f64, Vec<f64>) {
        let design = self.x.as_ref();
        let coefficients = vec_to_mat(params, design.ncols(), self.class_count);
        let probabilities = softmax_scores(
            design,
            coefficients.as_ref(),
            self.class_count,
            self.min_weight,
        );

        let mut loss = 0.0;
        let mut gradient = vec![0.0; params.len()];

        for (row_index, row_probabilities) in probabilities.iter().enumerate().take(design.nrows())
        {
            for (class_index, probability) in
                row_probabilities.iter().enumerate().take(self.class_count)
            {
                let class_weight =
                    self.class_weight_sums[row_index * self.class_count + class_index];
                if class_weight > 0.0 {
                    loss = class_weight.mul_add(-probability.max(1e-12).ln(), loss);
                }
            }

            for class_index in 0..self.class_count {
                let class_weight =
                    self.class_weight_sums[row_index * self.class_count + class_index];
                let residual = self.row_total_weights[row_index]
                    .mul_add(row_probabilities[class_index], -class_weight);
                for feature_index in 0..design.ncols() {
                    gradient[param_index(feature_index, class_index, self.class_count)] = residual
                        .mul_add(
                            design[(row_index, feature_index)],
                            gradient[param_index(feature_index, class_index, self.class_count)],
                        );
                }
            }
        }

        for feature_index in 1..design.ncols() {
            for class_index in 0..self.class_count {
                let index = param_index(feature_index, class_index, self.class_count);
                loss = (0.5 * self.ridge).mul_add(params[index].powi(2), loss);
                gradient[index] = self.ridge.mul_add(params[index], gradient[index]);
            }
        }

        (loss, gradient)
    }
}

impl MultinomialPropensityEstimator for MultinomialSoftmaxPS {
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn fit_multinomial(
        &self,
        x: MatRef<'_, f64>,
        class_labels: &[usize],
        sampling_weights: &[f64],
        class_count: usize,
    ) -> Result<MultinomialParams, InternalDidError> {
        if x.nrows() == 0
            || class_labels.len() != x.nrows()
            || sampling_weights.len() != x.nrows()
            || class_count < 2
        {
            return Err(InternalDidError::Estimation(
                "invalid multinomial input shape".to_string(),
            ));
        }
        if class_labels.iter().any(|label| *label >= class_count) {
            return Err(InternalDidError::Estimation(
                "multinomial class label exceeds class count".to_string(),
            ));
        }

        let ridge = self.ridge.max(1e-10);
        let problem = build_grouped_problem(
            x,
            class_labels,
            sampling_weights,
            class_count,
            ridge,
            self.cfg.min_weight,
        );
        let initial = vec![0.0; problem.x.ncols() * class_count];
        let solver = Bfgs {
            max_iterations: usize::try_from(self.cfg.max_iter.max(25)).unwrap_or(usize::MAX),
            ..Bfgs::default()
        };
        let best = solver
            .minimize(&initial, |params| problem.evaluate(params), None)
            .map_err(|err| InternalDidError::Estimation(err.to_string()))?
            .x;

        Ok(MultinomialParams {
            beta: vec_to_mat(&best, x.ncols(), class_count),
        })
    }
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn build_grouped_problem(
    x: MatRef<'_, f64>,
    class_labels: &[usize],
    sampling_weights: &[f64],
    class_count: usize,
    ridge: f64,
    min_weight: f64,
) -> MultinomialProblem {
    let feature_count = x.ncols();
    let mut key_to_group = HashMap::<Vec<u64>, usize>::new();
    let mut grouped_rows = Vec::<Vec<f64>>::new();
    let mut class_weight_sums = Vec::<f64>::new();
    let mut row_total_weights = Vec::<f64>::new();
    let mut key_bits = vec![0_u64; feature_count];
    let mut row_values = vec![0.0; feature_count];

    for row_index in 0..x.nrows() {
        for feature_index in 0..feature_count {
            let value = x[(row_index, feature_index)];
            row_values[feature_index] = value;
            key_bits[feature_index] = normalized_f64_bits(value);
        }

        let group_index = if let Some(&group_index) = key_to_group.get(key_bits.as_slice()) {
            group_index
        } else {
            let group_index = grouped_rows.len();
            key_to_group.insert(key_bits.clone(), group_index);
            grouped_rows.push(row_values.clone());
            class_weight_sums.extend(std::iter::repeat_n(0.0, class_count));
            row_total_weights.push(0.0);
            group_index
        };

        let class_label = class_labels[row_index];
        let row_weight = sampling_weights[row_index];
        class_weight_sums[group_index * class_count + class_label] += row_weight;
        row_total_weights[group_index] += row_weight;
    }

    let group_count = grouped_rows.len();
    let grouped_design = Mat::from_fn(group_count, feature_count, |row, col| {
        grouped_rows[row][col]
    });

    MultinomialProblem {
        x: grouped_design,
        class_weight_sums,
        row_total_weights,
        class_count,
        ridge,
        min_weight,
    }
}

const fn param_index(feature_index: usize, class_index: usize, class_count: usize) -> usize {
    feature_index * class_count + class_index
}

fn normalized_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}

fn vec_to_mat(values: &[f64], feature_count: usize, class_count: usize) -> Mat<f64> {
    Mat::from_fn(feature_count, class_count, |feature, class| {
        values[param_index(feature, class, class_count)]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multinomial_softmax_probabilities_sum_to_one() {
        let x = Mat::from_fn(6, 2, |row, col| {
            if col == 0 {
                1.0
            } else {
                f64::from(u32::try_from(row).expect("row fits u32")) * 0.5
            }
        });
        let class_labels = vec![0, 1, 2, 3, 0, 1];
        let weights = vec![1.0; 6];
        let estimator = MultinomialSoftmaxPS::new(Config::default(), 1e-8);

        let params = estimator
            .fit_multinomial(x.as_ref(), &class_labels, &weights, 4)
            .expect("multinomial fit");
        let probabilities = softmax_scores(
            x.as_ref(),
            params.beta.as_ref(),
            params.beta.ncols(),
            estimator.cfg.min_weight,
        );

        for row in probabilities {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-10);
            assert!(row.iter().all(|value| value.is_finite() && *value > 0.0));
        }
    }
}
