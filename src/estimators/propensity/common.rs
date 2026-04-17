use crate::util::usize_to_f64;
use faer::{Mat, MatRef};
use std::collections::HashMap;
use tracing::trace;

pub struct GroupedBinaryRows {
    pub design: Mat<f64>,
    pub treated_counts: Vec<f64>,
    pub control_counts: Vec<f64>,
    pub total_counts: Vec<f64>,
    pub target_means: Mat<f64>,
}

#[must_use]
pub fn safe_sigmoid(v: f64) -> f64 {
    let z = v.clamp(-700.0, 700.0);
    1.0 / (1.0 + (-z).exp())
}

#[must_use]
pub fn empirical_logit(y: MatRef<'_, f64>) -> f64 {
    let y_mean = (0..y.nrows()).map(|row| *y.get(row, 0)).sum::<f64>() / usize_to_f64(y.nrows());
    if y_mean <= 0.001 {
        -6.907
    } else if y_mean >= 0.999 {
        6.907
    } else {
        (y_mean / (1.0 - y_mean)).ln()
    }
}

#[must_use]
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn group_binary_design_rows(
    design: MatRef<'_, f64>,
    target: MatRef<'_, f64>,
) -> GroupedBinaryRows {
    let feature_count = design.ncols();
    let mut key_to_group = HashMap::<Vec<u64>, usize>::new();
    let mut grouped_rows = Vec::<Vec<f64>>::new();
    let mut treated_counts = Vec::<f64>::new();
    let mut control_counts = Vec::<f64>::new();
    let mut total_counts = Vec::<f64>::new();
    let mut key_bits = vec![0_u64; feature_count];
    let mut row_values = vec![0.0; feature_count];

    for row_index in 0..design.nrows() {
        for feature_index in 0..feature_count {
            let value = design[(row_index, feature_index)];
            row_values[feature_index] = value;
            key_bits[feature_index] = normalized_f64_bits(value);
        }

        let group_index = if let Some(&group_index) = key_to_group.get(key_bits.as_slice()) {
            group_index
        } else {
            let group_index = grouped_rows.len();
            key_to_group.insert(key_bits.clone(), group_index);
            grouped_rows.push(row_values.clone());
            treated_counts.push(0.0);
            control_counts.push(0.0);
            total_counts.push(0.0);
            group_index
        };

        let treated = target[(row_index, 0)] > 0.5;
        if treated {
            treated_counts[group_index] += 1.0;
        } else {
            control_counts[group_index] += 1.0;
        }
        total_counts[group_index] += 1.0;
    }

    let group_count = grouped_rows.len();
    let grouped_design = Mat::from_fn(group_count, feature_count, |row, col| {
        grouped_rows[row][col]
    });
    let target_means = Mat::from_fn(group_count, 1, |row, _| {
        treated_counts[row] / total_counts[row].max(1.0)
    });

    GroupedBinaryRows {
        design: grouped_design,
        treated_counts,
        control_counts,
        total_counts,
        target_means,
    }
}

#[must_use]
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn logistic_scores(covariates: MatRef<'_, f64>, beta: MatRef<'_, f64>) -> Vec<f64> {
    let n = covariates.nrows();
    let k = covariates.ncols();
    let mut scores = Vec::with_capacity(n);

    for i in 0..n {
        let mut linear_pred = 0.0;
        for j in 0..k {
            linear_pred += covariates.get(i, j) * beta.get(j, 0);
        }
        let ps = safe_sigmoid(linear_pred).clamp(1e-6, 0.995);
        scores.push(ps);
    }

    scores
}

#[must_use]
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn softmax_scores(
    covariates: MatRef<'_, f64>,
    beta: MatRef<'_, f64>,
    class_count: usize,
    clip: f64,
) -> Vec<Vec<f64>> {
    let row_count = covariates.nrows();
    let feature_count = covariates.ncols();
    (0..row_count)
        .map(|row_index| {
            let mut logits = vec![0.0; class_count];
            for (class_index, logit) in logits.iter_mut().enumerate().take(class_count) {
                *logit = (0..feature_count)
                    .map(|feature_index| {
                        covariates.get(row_index, feature_index)
                            * beta.get(feature_index, class_index)
                    })
                    .sum::<f64>();
            }
            let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut probabilities = logits
                .into_iter()
                .map(|value| (value - max_logit).exp())
                .collect::<Vec<_>>();
            let total = probabilities.iter().sum::<f64>().max(1e-12);
            for probability in &mut probabilities {
                *probability = (*probability / total).clamp(clip, 1.0);
            }
            let normalized_total = probabilities.iter().sum::<f64>().max(1e-12);
            for probability in &mut probabilities {
                *probability /= normalized_total;
            }
            probabilities
        })
        .collect()
}

#[must_use]
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn softmax_scores_fixed4(
    covariates: MatRef<'_, f64>,
    beta: MatRef<'_, f64>,
    clip: f64,
) -> Vec<[f64; 4]> {
    debug_assert_eq!(beta.ncols(), 4);
    let row_count = covariates.nrows();
    let feature_count = covariates.ncols();
    let mut scores = Vec::with_capacity(row_count);

    for row_index in 0..row_count {
        let mut logits = [0.0; 4];
        for (class_index, logit) in logits.iter_mut().enumerate() {
            *logit = (0..feature_count)
                .map(|feature_index| {
                    covariates.get(row_index, feature_index) * beta.get(feature_index, class_index)
                })
                .sum::<f64>();
        }
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut probabilities = logits.map(|value| (value - max_logit).exp());
        let total = probabilities.iter().sum::<f64>().max(1e-12);
        for probability in &mut probabilities {
            *probability = (*probability / total).clamp(clip, 1.0);
        }
        let normalized_total = probabilities.iter().sum::<f64>().max(1e-12);
        for probability in &mut probabilities {
            *probability /= normalized_total;
        }
        scores.push(probabilities);
    }

    scores
}

pub fn debug_matrix_ranges(name: &str, m: MatRef<'_, f64>) {
    let min = (0..m.nrows())
        .map(|row| *m.get(row, 0))
        .fold(f64::INFINITY, f64::min);
    let max = (0..m.nrows())
        .map(|row| *m.get(row, 0))
        .fold(f64::NEG_INFINITY, f64::max);
    trace!("{} range = [{:.6}, {:.6}]", name, min, max);
}

fn normalized_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}
