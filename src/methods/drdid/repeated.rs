use faer::Mat;
use itertools::izip;

use crate::estimators::outcome::linear::LinearOutcome;
use crate::estimators::outcome::model::OutcomeModel;
use crate::estimators::propensity::common::logistic_scores;
use crate::estimators::propensity::logistic::LogisticPS;
use crate::estimators::propensity::types::{Config as PropensityConfig, PropensityEstimator};
use crate::inference::{
    multiplier_bootstrap_ci, standard_error_from_influence, validate_confidence_level,
};
use crate::methods::drdid::moments::{
    RepeatedMomentError, RepeatedMomentEstimate, RepeatedMomentInputs,
    normalize_weights_to_n as normalize_weights_to_n_shared, repeated_att_moments,
};
use crate::types::{DidCell, DrDidConfig, DrDidError, DrDidEstimate, DrDidRepeatedObservation};

/// Estimate ATT in a repeated cross-section `DR-DiD` design.
///
/// This implements the repeated cross-section doubly robust moment in Eq. (2.8)
/// of Callaway & Sant'Anna (2020), using control-group outcome models and
/// propensity-score reweighting.
///
/// Let `D_i` be treatment status and `T_i` the post-period indicator. The
/// repeated-cross-section ATT is identified from the moment condition
///
/// ```text
/// E[ ψ(W_i; ATT, η) ] = 0
/// ```
///
/// where `η` collects the nuisance functions:
///
/// - `π(X_i) = P(D_i = 1 | X_i)`,
/// - `m_{0,pre}(X_i) = E[Y_i | D_i = 0, T_i = 0, X_i]`,
/// - `m_{0,post}(X_i) = E[Y_i | D_i = 0, T_i = 1, X_i]`.
///
/// The implementation here follows the standard doubly robust repeated-cross-
/// section score and treats the improved / calibrated repeated estimator as a
/// separate surface in [`super::improved`].
///
/// References:
/// - Callaway, B. and Sant'Anna, P. H. C. (2021). "Difference-in-Differences
///   with Multiple Time Periods". *Journal of Econometrics*.
/// - Sant'Anna, P. H. C. and Zhao, J. (2020). "Doubly Robust Difference-in-
///   Differences Estimators". *Journal of Econometrics*.
///
/// # Errors
///
/// Returns [`DrDidError`] when inputs/configuration are invalid or nuisance
/// models cannot be fit.
pub fn estimate_drdid_repeated_cross_section(
    observations: &[DrDidRepeatedObservation],
    config: DrDidConfig,
) -> Result<DrDidEstimate, DrDidError> {
    validate_drdid_config(config)?;
    if observations.is_empty() {
        return Err(DrDidError::EmptyInput);
    }

    let prepared = prepare_repeated_data(observations)?;
    let nuisance = fit_repeated_nuisance_models(&prepared, config)?;
    let moments = estimate_repeated_att_moments(
        &nuisance.normalized_weights,
        &prepared.treated_indicator,
        &prepared.post_indicator,
        &nuisance.propensity_scores,
        &nuisance.residualized_outcome,
    )?;
    let att = moments.att;
    let influence_function = moments.influence_function;

    let se = standard_error_from_influence(&influence_function);
    let (ci_low, ci_high) = multiplier_bootstrap_ci(
        att,
        &influence_function,
        config.inference(),
        config.bootstrap(),
    );

    Ok(DrDidEstimate {
        att,
        se,
        ci_low,
        ci_high,
        treated_n: prepared.treated_n,
        control_n: prepared.control_n,
        total_weight: prepared.total_weight,
        influence_function,
    })
}

struct RepeatedPreparedData {
    feature_count: usize,
    treated_n: usize,
    control_n: usize,
    total_weight: f64,
    treated_indicator: Vec<f64>,
    post_indicator: Vec<f64>,
    outcome: Vec<f64>,
    sampling_weights: Vec<f64>,
    design_matrix_flat: Vec<f64>,
}

struct RepeatedNuisanceFits {
    normalized_weights: Vec<f64>,
    propensity_scores: Vec<f64>,
    residualized_outcome: Vec<f64>,
}

fn prepare_repeated_data(
    observations: &[DrDidRepeatedObservation],
) -> Result<RepeatedPreparedData, DrDidError> {
    let covariate_count = observations.first().map_or(0, |row| row.covariates.len());
    let feature_count = covariate_count + 1;
    let observation_count = observations.len();

    let mut treated_n = 0_usize;
    let mut control_n = 0_usize;
    let mut total_weight = 0.0_f64;

    let mut has_treated_pre = false;
    let mut has_treated_post = false;
    let mut has_control_pre = false;
    let mut has_control_post = false;

    let mut treated_indicator = Vec::with_capacity(observation_count);
    let mut post_indicator = Vec::with_capacity(observation_count);
    let mut outcome = Vec::with_capacity(observation_count);
    let mut sampling_weights = Vec::with_capacity(observation_count);
    let mut design_matrix_flat = Vec::with_capacity(observation_count * feature_count);

    for row in observations {
        if row.covariates.len() != covariate_count {
            return Err(DrDidError::InconsistentCovariateCount {
                expected: covariate_count,
                actual: row.covariates.len(),
            });
        }
        if !row.outcome.is_finite() {
            return Err(DrDidError::InvalidOutcome { value: row.outcome });
        }
        if !row.weight.is_finite() || row.weight <= 0.0 {
            return Err(DrDidError::InvalidWeight { value: row.weight });
        }
        for covariate in &row.covariates {
            if !covariate.is_finite() {
                return Err(DrDidError::InvalidCovariate { value: *covariate });
            }
        }

        if row.treated {
            treated_n += 1;
            if row.post_period {
                has_treated_post = true;
            } else {
                has_treated_pre = true;
            }
            treated_indicator.push(1.0);
        } else {
            control_n += 1;
            if row.post_period {
                has_control_post = true;
            } else {
                has_control_pre = true;
            }
            treated_indicator.push(0.0);
        }

        post_indicator.push(if row.post_period { 1.0 } else { 0.0 });
        outcome.push(row.outcome);
        total_weight += row.weight;
        sampling_weights.push(row.weight);
        design_matrix_flat.push(1.0);
        design_matrix_flat.extend_from_slice(&row.covariates);
    }

    if treated_n == 0 {
        return Err(DrDidError::NoTreated);
    }
    if control_n == 0 {
        return Err(DrDidError::NoControl);
    }
    if !has_treated_pre {
        return Err(DrDidError::MissingCell {
            cell: DidCell::TreatedPre,
        });
    }
    if !has_treated_post {
        return Err(DrDidError::MissingCell {
            cell: DidCell::TreatedPost,
        });
    }
    if !has_control_pre {
        return Err(DrDidError::MissingCell {
            cell: DidCell::ControlPre,
        });
    }
    if !has_control_post {
        return Err(DrDidError::MissingCell {
            cell: DidCell::ControlPost,
        });
    }

    Ok(RepeatedPreparedData {
        feature_count,
        treated_n,
        control_n,
        total_weight,
        treated_indicator,
        post_indicator,
        outcome,
        sampling_weights,
        design_matrix_flat,
    })
}

fn fit_repeated_nuisance_models(
    prepared: &RepeatedPreparedData,
    config: DrDidConfig,
) -> Result<RepeatedNuisanceFits, DrDidError> {
    let observation_count = prepared.treated_indicator.len();
    let normalized_weights = normalize_weights_to_n(&prepared.sampling_weights)?;
    let design_matrix = Mat::from_fn(observation_count, prepared.feature_count, |row, col| {
        prepared.design_matrix_flat[row * prepared.feature_count + col]
    });
    let treated_target = Mat::from_fn(observation_count, 1, |row, _| {
        prepared.treated_indicator[row]
    });

    let propensity_cfg = PropensityConfig {
        max_iter: u64::try_from(config.max_iter)
            .map_err(|_| DrDidError::InvalidConfig("max_iter exceeds u64".to_string()))?,
        tol: config.tol,
        min_weight: config.propensity_clip,
        vstar: 700.0,
    };
    let propensity = LogisticPS::new(propensity_cfg);
    let propensity_params = propensity
        .fit(design_matrix.as_ref(), treated_target.as_ref())
        .map_err(|err| DrDidError::InvalidConfig(err.to_string()))?;
    let propensity_scores =
        logistic_scores(design_matrix.as_ref(), propensity_params.beta.as_ref())
            .into_iter()
            .map(|score| score.clamp(config.propensity_clip, 1.0 - config.propensity_clip))
            .collect::<Vec<_>>();

    let control_outcome_pre = fit_control_outcome_by_period(
        &design_matrix,
        &prepared.treated_indicator,
        &prepared.post_indicator,
        &prepared.outcome,
        &normalized_weights,
        false,
        config.ridge,
    )?;
    let control_outcome_post = fit_control_outcome_by_period(
        &design_matrix,
        &prepared.treated_indicator,
        &prepared.post_indicator,
        &prepared.outcome,
        &normalized_weights,
        true,
        config.ridge,
    )?;

    let residualized_outcome = izip!(
        prepared.post_indicator.iter(),
        prepared.outcome.iter(),
        control_outcome_pre.iter(),
        control_outcome_post.iter()
    )
    .map(
        |(post_indicator, outcome, control_pre_prediction, control_post_prediction)| {
            let control_prediction = if *post_indicator > 0.5 {
                *control_post_prediction
            } else {
                *control_pre_prediction
            };
            outcome - control_prediction
        },
    )
    .collect::<Vec<_>>();

    Ok(RepeatedNuisanceFits {
        normalized_weights,
        propensity_scores,
        residualized_outcome,
    })
}

fn estimate_repeated_att_moments(
    normalized_weights: &[f64],
    treated_indicator: &[f64],
    post_indicator: &[f64],
    propensity_scores: &[f64],
    residualized_outcome: &[f64],
) -> Result<RepeatedMomentEstimate, DrDidError> {
    repeated_att_moments(RepeatedMomentInputs {
        normalized_weights,
        treated: treated_indicator,
        post_period: post_indicator,
        propensity: propensity_scores,
        signal: residualized_outcome,
    })
    .map_err(|err| match err {
        RepeatedMomentError::MissingCell { cell } => DrDidError::MissingCell { cell },
        RepeatedMomentError::InvalidInputShape => {
            DrDidError::InvalidConfig("moment input shape mismatch".to_string())
        }
    })
}

fn fit_control_outcome_by_period(
    design_matrix: &Mat<f64>,
    treated_indicator: &[f64],
    post_indicator: &[f64],
    outcome: &[f64],
    normalized_weights: &[f64],
    post_period: bool,
    ridge: f64,
) -> Result<Vec<f64>, DrDidError> {
    let period_value = if post_period { 1.0 } else { 0.0 };
    let control_indices = treated_indicator
        .iter()
        .enumerate()
        .filter_map(|(row_index, treated_value)| {
            if *treated_value < 0.5
                && (post_indicator[row_index] - period_value).abs() < f64::EPSILON
            {
                Some(row_index)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    if control_indices.is_empty() {
        return Err(DrDidError::MissingCell {
            cell: if post_period {
                DidCell::ControlPost
            } else {
                DidCell::ControlPre
            },
        });
    }

    let control_design = Mat::from_fn(control_indices.len(), design_matrix.ncols(), |row, col| {
        *design_matrix.get(control_indices[row], col)
    });
    let control_outcome = Mat::from_fn(control_indices.len(), 1, |row, _| {
        outcome[control_indices[row]]
    });
    let control_weights = control_indices
        .iter()
        .map(|row_index| normalized_weights[*row_index])
        .collect::<Vec<_>>();

    let model = LinearOutcome { ridge };
    let control_coefficients = model.fit(
        control_design.as_ref(),
        control_outcome.as_ref(),
        Some(&control_weights),
    );
    Ok(model.predict(design_matrix.as_ref(), control_coefficients.as_ref()))
}

fn validate_drdid_config(config: DrDidConfig) -> Result<(), DrDidError> {
    if !validate_confidence_level(config.confidence_level) {
        return Err(DrDidError::InvalidConfig(
            "confidence_level must be finite and in (0, 1)".to_string(),
        ));
    }
    if !config.propensity_clip.is_finite()
        || config.propensity_clip <= 0.0
        || config.propensity_clip >= 0.5
    {
        return Err(DrDidError::InvalidConfig(
            "propensity_clip must be finite and in (0, 0.5)".to_string(),
        ));
    }
    if !config.ridge.is_finite() || config.ridge < 0.0 {
        return Err(DrDidError::InvalidConfig(
            "ridge must be finite and non-negative".to_string(),
        ));
    }
    if config.max_iter == 0 {
        return Err(DrDidError::InvalidConfig(
            "max_iter must be > 0".to_string(),
        ));
    }
    if !config.tol.is_finite() || config.tol <= 0.0 {
        return Err(DrDidError::InvalidConfig(
            "tol must be finite and > 0".to_string(),
        ));
    }
    if config.bootstrap_reps == 0 {
        return Err(DrDidError::InvalidConfig(
            "bootstrap_reps must be > 0".to_string(),
        ));
    }
    Ok(())
}

fn normalize_weights_to_n(weights: &[f64]) -> Result<Vec<f64>, DrDidError> {
    normalize_weights_to_n_shared(weights)
        .map_err(|_| DrDidError::InvalidConfig("sum of weights must be > 0".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DidCell, TimePeriod, TreatmentGroup};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    fn rc_row(treated: bool, post_period: bool, outcome: f64, x1: f64) -> DrDidRepeatedObservation {
        DrDidRepeatedObservation {
            covariates: vec![x1],
            ..DrDidRepeatedObservation::new(
                DidCell::from_parts(
                    TreatmentGroup::from_bool(treated),
                    TimePeriod::from_bool(post_period),
                ),
                outcome,
            )
        }
    }

    fn draw_standard_normal(rng: &mut StdRng) -> f64 {
        let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2 = rng.random::<f64>();
        (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos()
    }

    fn logistic(v: f64) -> f64 {
        1.0 / (1.0 + (-v).exp())
    }

    fn simulate_repeated_cross_section(
        rng: &mut StdRng,
        n_per_period: usize,
        true_att: f64,
    ) -> Vec<DrDidRepeatedObservation> {
        let mut rows = Vec::with_capacity(2 * n_per_period);
        for post_period in [false, true] {
            for _ in 0..n_per_period {
                let x = draw_standard_normal(rng);
                let p_treat = logistic(0.5f64.mul_add(x, -0.3)).clamp(0.05, 0.95);
                let treated = rng.random::<f64>() < p_treat;
                let noise = draw_standard_normal(rng);
                let time_trend = if post_period { 0.75 } else { 0.0 };
                let treatment_effect = if treated && post_period {
                    true_att
                } else {
                    0.0
                };
                let outcome = 1.1f64.mul_add(x, 2.0) + time_trend + treatment_effect + noise;
                rows.push(rc_row(treated, post_period, outcome, x));
            }
        }
        rows
    }

    #[test]
    fn estimates_repeated_cross_section_att() {
        let mut rows = Vec::new();
        for _ in 0..20 {
            rows.push(rc_row(false, false, 5.0, 0.0));
            rows.push(rc_row(false, false, 6.0, 1.0));
            rows.push(rc_row(false, true, 7.0, 0.0));
            rows.push(rc_row(false, true, 8.0, 1.0));
            rows.push(rc_row(true, false, 5.0, 0.0));
            rows.push(rc_row(true, false, 6.0, 1.0));
            rows.push(rc_row(true, true, 10.0, 0.0));
            rows.push(rc_row(true, true, 11.0, 1.0));
        }

        let config = DrDidConfig::builder()
            .bootstrap_reps(199)
            .bootstrap_seed(42)
            .build();
        let estimate = estimate_drdid_repeated_cross_section(&rows, config).expect("estimate");

        assert_eq!(estimate.treated_n, 80);
        assert_eq!(estimate.control_n, 80);
        assert!((estimate.att - 3.0).abs() < 0.2);
        assert!(estimate.se.is_finite());
        assert!(estimate.ci_low <= estimate.ci_high);
    }

    #[test]
    fn repeated_cross_section_rejects_missing_cells() {
        let rows = vec![
            rc_row(false, false, 5.0, 0.0),
            rc_row(false, true, 7.0, 0.0),
            rc_row(true, true, 10.0, 0.0),
        ];
        let err = estimate_drdid_repeated_cross_section(&rows, DrDidConfig::default())
            .expect_err("missing treated-pre cell should fail");
        assert_eq!(
            err,
            DrDidError::MissingCell {
                cell: DidCell::TreatedPre
            }
        );
    }

    #[test]
    fn repeated_cross_section_rejects_invalid_values_and_config() {
        let valid = vec![
            rc_row(false, false, 5.0, 0.0),
            rc_row(false, true, 7.0, 0.0),
            rc_row(true, false, 5.0, 0.0),
            rc_row(true, true, 10.0, 0.0),
        ];

        let bad_cfg = DrDidConfig::builder().confidence_level(1.0).build();
        let err = estimate_drdid_repeated_cross_section(&valid, bad_cfg).expect_err("bad config");
        assert!(matches!(err, DrDidError::InvalidConfig(msg) if msg.contains("confidence_level")));

        let bad_weight = vec![
            rc_row(false, false, 5.0, 0.0),
            rc_row(false, true, 7.0, 0.0),
            rc_row(true, false, 5.0, 0.0),
            DrDidRepeatedObservation {
                weight: 0.0,
                ..rc_row(true, true, 10.0, 0.0)
            },
        ];
        let err = estimate_drdid_repeated_cross_section(&bad_weight, DrDidConfig::default())
            .expect_err("bad weight");
        assert_eq!(err, DrDidError::InvalidWeight { value: 0.0 });

        let bad_outcome = vec![
            rc_row(false, false, f64::INFINITY, 0.0),
            rc_row(false, true, 7.0, 0.0),
            rc_row(true, false, 5.0, 0.0),
            rc_row(true, true, 10.0, 0.0),
        ];
        let err = estimate_drdid_repeated_cross_section(&bad_outcome, DrDidConfig::default())
            .expect_err("bad outcome");
        assert!(matches!(err, DrDidError::InvalidOutcome { value } if value.is_infinite()));

        let bad_cov = vec![
            rc_row(false, false, 5.0, f64::NAN),
            rc_row(false, true, 7.0, 0.0),
            rc_row(true, false, 5.0, 0.0),
            rc_row(true, true, 10.0, 0.0),
        ];
        let err = estimate_drdid_repeated_cross_section(&bad_cov, DrDidConfig::default())
            .expect_err("bad covariate");
        assert!(matches!(err, DrDidError::InvalidCovariate { value } if value.is_nan()));
    }

    #[test]
    fn repeated_cross_section_rejects_no_treated_or_control_and_inconsistent_covariates() {
        let no_treated = vec![
            rc_row(false, false, 5.0, 0.0),
            rc_row(false, true, 7.0, 0.0),
        ];
        let err = estimate_drdid_repeated_cross_section(&no_treated, DrDidConfig::default())
            .expect_err("must fail");
        assert_eq!(err, DrDidError::NoTreated);

        let no_control = vec![rc_row(true, false, 5.0, 0.0), rc_row(true, true, 10.0, 0.0)];
        let err = estimate_drdid_repeated_cross_section(&no_control, DrDidConfig::default())
            .expect_err("must fail");
        assert_eq!(err, DrDidError::NoControl);

        let inconsistent = vec![
            rc_row(false, false, 5.0, 0.0),
            DrDidRepeatedObservation {
                covariates: vec![0.0, 1.0],
                ..DrDidRepeatedObservation::new(DidCell::ControlPost, 7.0)
            },
            rc_row(true, false, 5.0, 0.0),
            rc_row(true, true, 10.0, 0.0),
        ];
        let err = estimate_drdid_repeated_cross_section(&inconsistent, DrDidConfig::default())
            .expect_err("must fail");
        assert_eq!(
            err,
            DrDidError::InconsistentCovariateCount {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn repeated_cross_section_simulation_recovers_true_att() {
        let true_att = 2.0;
        let draws = 25;
        let mut rng = StdRng::seed_from_u64(11_003);
        let mut estimates = Vec::with_capacity(draws);

        for draw in 0..draws {
            let rows = simulate_repeated_cross_section(&mut rng, 900, true_att);
            let config = DrDidConfig::builder()
                .bootstrap_reps(49)
                .bootstrap_seed(200 + u64::try_from(draw).expect("draw index fits in u64"))
                .build();
            let est = estimate_drdid_repeated_cross_section(&rows, config).expect("simulation fit");
            estimates.push(est);
        }

        let mean_att =
            estimates.iter().map(|e| e.att).sum::<f64>() / crate::util::usize_to_f64(draws);
        let rmse = (estimates
            .iter()
            .map(|e| (e.att - true_att).powi(2))
            .sum::<f64>()
            / crate::util::usize_to_f64(draws))
        .sqrt();
        let coverage = estimates
            .iter()
            .filter(|e| e.ci_low <= true_att && true_att <= e.ci_high)
            .count();
        let coverage_rate = crate::util::usize_to_f64(coverage) / crate::util::usize_to_f64(draws);

        assert!((mean_att - true_att).abs() < 0.12, "mean_att={mean_att}");
        assert!(rmse < 0.30, "rmse={rmse}");
        assert!(coverage_rate > 0.70, "coverage_rate={coverage_rate}");
    }
}
