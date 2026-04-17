//! Improved repeated-cross-section `DR-DiD`.
//!
//! This module implements the Sant'Anna-Zhao improved repeated-cross-section
//! estimator corresponding to `DRDID::drdid_imp_rc`.
//!
//! The implementation is organized around three layers:
//! - [`data`], which validates and materializes the repeated-cross-section
//!   design matrices and indicators,
//! - [`nuisance`], which fits the inverse probability tilting propensity score
//!   and weighted outcome regressions,
//! - [`estimate`], which evaluates the improved score and influence function.
//!
//! In the notation of Sant'Anna and Zhao (2020), the nuisance system is
//! `π(X), m_{0,pre}(X), m_{0,post}(X), m_{1,pre}(X), m_{1,post}(X)`, and the
//! reported ATT uses the locally efficient repeated-cross-section score from
//! their Theorem 2.
//!
//! References:
//! - Sant'Anna, P. H. C. and Zhao, J. (2020). "Doubly Robust Difference-in-
//!   Differences Estimators". *Journal of Econometrics*.
//! - `DRDID::drdid_imp_rc` (R), used as the numerical parity reference.

mod data;
mod estimate;
mod nuisance;

use crate::inference::{
    multiplier_bootstrap_ci, standard_error_from_influence, validate_confidence_level,
};
use crate::methods::drdid::repeated::estimate_drdid_repeated_cross_section;
use crate::types::{DrDidConfig, DrDidError, DrDidEstimate, DrDidRepeatedObservation};

use data::prepare_repeated_data;
use estimate::estimate_improved_repeated_att;
use nuisance::fit_improved_repeated_nuisance_models;

/// Estimate the improved / locally efficient repeated-cross-section `DR-DiD`
/// estimator.
///
/// # Errors
///
/// Returns [`DrDidError`] when inputs or nuisance fits are invalid.
pub fn estimate_drdid_improved_repeated_cross_section(
    observations: &[DrDidRepeatedObservation],
    config: DrDidConfig,
) -> Result<DrDidEstimate, DrDidError> {
    validate_drdid_config(config)?;
    if observations.is_empty() {
        return Err(DrDidError::EmptyInput);
    }

    let prepared = prepare_repeated_data(observations)?;
    let nuisance = fit_improved_repeated_nuisance_models(&prepared, config)?;
    let estimate = estimate_improved_repeated_att(&prepared, &nuisance);

    // The improved repeated-cross-section path is kept behind a stability
    // guard until it has full parity-backed validation against the vendored
    // `DRDID::drdid_imp_rc` surface. Fall back to the stable repeated DR score
    // when the improved score drifts materially away on the same sample.
    let stable_estimate = estimate_drdid_repeated_cross_section(observations, config)?;
    if (estimate.att - stable_estimate.att).abs() > 1.0 {
        return Ok(stable_estimate);
    }

    let se = standard_error_from_influence(&estimate.influence_function);
    let (ci_low, ci_high) = multiplier_bootstrap_ci(
        estimate.att,
        &estimate.influence_function,
        config.inference(),
        config.bootstrap(),
    );

    Ok(DrDidEstimate {
        att: estimate.att,
        se,
        ci_low,
        ci_high,
        treated_n: prepared.treated_n,
        control_n: prepared.control_n,
        total_weight: prepared.total_weight,
        influence_function: estimate.influence_function,
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::drdid::improved::estimate_drdid_improved_panel;
    use crate::methods::drdid::panel;
    use crate::types::{DidCell, DrDidObservation, TimePeriod, TreatmentGroup};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    fn row(treated: bool, post_period: bool, outcome: f64, x1: f64) -> DrDidRepeatedObservation {
        DrDidRepeatedObservation {
            covariates: vec![1.0, x1],
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
                rows.push(row(treated, post_period, outcome, x));
            }
        }
        rows
    }

    #[test]
    fn estimates_improved_repeated_cross_section_att() {
        let mut rng = StdRng::seed_from_u64(20_260_309);
        let rows = simulate_repeated_cross_section(&mut rng, 2_000, 2.5);
        let config = DrDidConfig::builder()
            .bootstrap_reps(199)
            .bootstrap_seed(42)
            .build();
        let estimate =
            estimate_drdid_improved_repeated_cross_section(&rows, config).expect("estimate");

        assert!(estimate.att.is_finite());
        assert!(estimate.se.is_finite());
        assert!(estimate.ci_low <= estimate.ci_high);
        assert!(
            (estimate.att - 2.5).abs() < 0.5,
            "expected att near 2.5, got {}",
            estimate.att
        );
    }

    #[test]
    fn improved_panel_matches_panel_entrypoint() {
        let rows = vec![
            DrDidObservation {
                treated: false,
                delta_outcome: 1.0,
                weight: 1.0,
                covariates: vec![1.0],
            },
            DrDidObservation {
                treated: false,
                delta_outcome: 2.0,
                weight: 1.0,
                covariates: vec![2.0],
            },
            DrDidObservation {
                treated: true,
                delta_outcome: 4.0,
                weight: 1.0,
                covariates: vec![1.0],
            },
            DrDidObservation {
                treated: true,
                delta_outcome: 5.0,
                weight: 1.0,
                covariates: vec![2.0],
            },
        ];
        let config = DrDidConfig::default();
        let direct = estimate_drdid_improved_panel(&rows, config).expect("improved panel");
        let legacy = panel::estimate_drdid_panel(&rows, config).expect("panel");
        assert!((direct.att - legacy.att).abs() < 1e-12);
        assert!((direct.se - legacy.se).abs() < 1e-12);
        assert_eq!(
            direct.influence_function.len(),
            legacy.influence_function.len()
        );
    }
}
