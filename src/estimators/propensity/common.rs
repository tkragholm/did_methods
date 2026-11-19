use faer::MatRef;

#[must_use]
pub fn safe_sigmoid(v: f64) -> f64 {
    let z = v.clamp(-700.0, 700.0);
    1.0 / (1.0 + (-z).exp())
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
