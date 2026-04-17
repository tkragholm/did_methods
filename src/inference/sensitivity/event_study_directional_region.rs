use super::super::super::linear_algebra::{
    cholesky_lower, critical_value_from_pointwise_confidence, dot,
    pointwise_confidence_level_from_critical, post_covariance_block,
    simulated_lower_cholesky_maxima_batched, simulation_rank,
};
use super::super::super::relative_magnitude::geometry::basis_vector_index;
use super::super::super::{
    HonestDirectionalFunctionalPoint, HonestDirectionalRegion, HonestDirectionalRegionDiagnostics,
    HonestEventStudyInput, HonestJointPathConfig, HonestJointPathMethod, HonestSensitivity,
    RelativeMagnitudeConfidenceSetConfig,
};
use crate::inference::validate_confidence_level;
use crate::types::InferenceConfig;
use crate::util::usize_to_f64;
use rayon::prelude::*;
use std::time::Instant;
use tracing::debug;

const fn sensitivity_label(sensitivity: HonestSensitivity) -> &'static str {
    match sensitivity {
        HonestSensitivity::RelativeMagnitude(_) => "relative_magnitude",
        HonestSensitivity::Smoothness(_) => "smoothness",
    }
}

fn emit_directional_region_timing(
    step: &str,
    sensitivity: HonestSensitivity,
    directions: usize,
    duration_ms: u128,
) {
    debug!(
        target: "did_profile",
        scope = "directional_region",
        step,
        sensitivity_kind = sensitivity_label(sensitivity),
        directions,
        duration_ms,
        "directional region timing"
    );
}

fn validate_direction(direction: &[f64], num_post: usize) -> Result<(), String> {
    if direction.len() != num_post {
        return Err(format!(
            "direction length {} does not match number of post periods {}",
            direction.len(),
            num_post
        ));
    }
    if direction.iter().any(|value| !value.is_finite()) {
        return Err("direction vector values must be finite".to_string());
    }
    let l1: f64 = direction.iter().map(|value| value.abs()).sum();
    if l1 <= 1e-12 {
        return Err("direction vector must have non-zero weight".to_string());
    }
    Ok(())
}

fn direction_correlation_matrix(
    input: &HonestEventStudyInput,
    directions: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    let sigma_post = post_covariance_block(
        &input.covariance,
        input.num_pre_periods(),
        input.num_post_periods(),
    );
    let projected = directions
        .iter()
        .map(|direction| {
            sigma_post
                .iter()
                .map(|row| dot(direction, row))
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<_>>();
    let variances = projected
        .iter()
        .zip(directions.iter())
        .map(|(sigma_l, direction)| dot(direction, sigma_l).max(0.0))
        .collect::<Vec<_>>();
    if variances.iter().any(|variance| *variance <= 1e-14) {
        return Err(
            "one or more direction vectors have near-zero variance under the post covariance"
                .to_string(),
        );
    }
    let stddev = variances
        .iter()
        .map(|value| value.sqrt())
        .collect::<Vec<_>>();
    let n = directions.len();
    let mut corr = vec![vec![0.0; n]; n];
    for i in 0..n {
        corr[i][i] = 1.0;
        for j in (i + 1)..n {
            let cov_ij = dot(&directions[i], &projected[j]);
            let c = (cov_ij / (stddev[i] * stddev[j])).clamp(-1.0, 1.0);
            corr[i][j] = c;
            corr[j][i] = c;
        }
    }
    Ok(corr)
}

fn simulated_directional_pointwise_confidence_level(
    input: &HonestEventStudyInput,
    confidence_level: f64,
    simulation_draws: usize,
    simulation_seed: u64,
    directions: &[Vec<f64>],
) -> Result<f64, String> {
    if simulation_draws == 0 {
        return Err("directional simulation_draws must be positive".to_string());
    }
    let corr = direction_correlation_matrix(input, directions)?;
    let chol = cholesky_lower(&corr)?;
    let mut maxima =
        simulated_lower_cholesky_maxima_batched(&chol, simulation_draws, simulation_seed);
    let n = maxima.len();
    let rank = simulation_rank(n, confidence_level);
    maxima.select_nth_unstable_by(rank.min(n - 1), f64::total_cmp);
    let critical = maxima[rank.min(n - 1)];
    Ok(critical)
}

fn validate_direction_set(
    input: &HonestEventStudyInput,
    directions: &[Vec<f64>],
) -> Result<(), String> {
    input.validate()?;
    if directions.is_empty() {
        return Err("directional region requires at least one direction".to_string());
    }
    let num_post = input.num_post_periods();
    for direction in directions {
        validate_direction(direction, num_post)?;
    }
    Ok(())
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub(super) fn calibrate_directional_region(
    input: &HonestEventStudyInput,
    directions: &[Vec<f64>],
    inference: InferenceConfig,
    joint_config: HonestJointPathConfig,
) -> Result<(f64, f64), String> {
    validate_direction_set(input, directions)?;
    let alpha = 1.0 - inference.confidence_level;
    match joint_config.method {
        HonestJointPathMethod::Bonferroni => {
            let pointwise = 1.0 - alpha / usize_to_f64(directions.len());
            if !validate_confidence_level(pointwise) {
                return Err(format!(
                    "invalid Bonferroni-adjusted confidence level {pointwise}"
                ));
            }
            Ok((
                pointwise,
                critical_value_from_pointwise_confidence(pointwise)?,
            ))
        }
        HonestJointPathMethod::GaussianSimulated => {
            let critical = simulated_directional_pointwise_confidence_level(
                input,
                inference.confidence_level,
                joint_config.simulation_draws,
                joint_config.simulation_seed,
                directions,
            )?;
            Ok((
                pointwise_confidence_level_from_critical(critical)?,
                critical,
            ))
        }
    }
}

/// Simultaneous `HonestDiD` region over a finite set of post-treatment linear
/// functionals.
///
/// # Errors
/// Returns an error if any direction is malformed, if calibration fails, or if
/// any directional scalar assessment fails.
pub fn assess_honest_event_study_directional_region_with_config(
    input: &HonestEventStudyInput,
    directions: &[Vec<f64>],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
) -> Result<HonestDirectionalRegion, String> {
    assess_honest_event_study_directional_region_with_optional_prepared_relative_magnitude(
        input,
        directions,
        sensitivity,
        inference,
        null_value,
        relative_magnitude_config,
        joint_config,
        None,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub(in crate::inference::sensitivity) fn assess_honest_event_study_directional_region_with_optional_prepared_relative_magnitude(
    input: &HonestEventStudyInput,
    directions: &[Vec<f64>],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
    prepared_relative_magnitude: Option<&super::super::PreparedRelativeMagnitudeBaseContext>,
) -> Result<HonestDirectionalRegion, String> {
    let calibrate_started = Instant::now();
    let (pointwise_confidence_level, calibrated_max_t_critical_value) =
        calibrate_directional_region(input, directions, inference, joint_config)?;
    emit_directional_region_timing(
        "calibrate",
        sensitivity,
        directions.len(),
        calibrate_started.elapsed().as_millis(),
    );
    assess_honest_event_study_directional_region_with_precalibrated_pointwise(
        input,
        directions,
        sensitivity,
        null_value,
        relative_magnitude_config,
        prepared_relative_magnitude,
        joint_config.method,
        inference.confidence_level,
        pointwise_confidence_level,
        calibrated_max_t_critical_value,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(in crate::inference::sensitivity) fn assess_honest_event_study_directional_region_with_precalibrated_pointwise(
    input: &HonestEventStudyInput,
    directions: &[Vec<f64>],
    sensitivity: HonestSensitivity,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    prepared_relative_magnitude: Option<&super::super::PreparedRelativeMagnitudeBaseContext>,
    method: HonestJointPathMethod,
    confidence_level: f64,
    pointwise_confidence_level: f64,
    calibrated_max_t_critical_value: f64,
) -> Result<HonestDirectionalRegion, String> {
    let total_started = Instant::now();
    let pointwise_inference = InferenceConfig::new(pointwise_confidence_level);
    let pointwise_relative_magnitude_config =
        RelativeMagnitudeConfidenceSetConfig::from_inference(pointwise_inference);
    let owned_relative_magnitude = match (sensitivity, prepared_relative_magnitude) {
        (HonestSensitivity::RelativeMagnitude(mbar), None) => Some(
            super::super::prepare_relative_magnitude_base_context(input, mbar)?,
        ),
        _ => None,
    };
    let prepared_relative_magnitude =
        prepared_relative_magnitude.or(owned_relative_magnitude.as_ref());
    let basis_direction_indices = directions
        .iter()
        .map(|direction| basis_vector_index(direction))
        .collect::<Option<Vec<_>>>();
    let prepare_started = Instant::now();
    let prepared_functional_branch_sets =
        match (prepared_relative_magnitude, &basis_direction_indices) {
            (Some(prepared), None) => Some(
                directions
                    .par_iter()
                    .map(|direction| {
                        super::super::prepare_relative_magnitude_functional_branches(
                            direction,
                            &prepared.branches,
                            &prepared.input_branches,
                        )
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            ),
            _ => None,
        };
    emit_directional_region_timing(
        "prepare_functional_branches",
        sensitivity,
        directions.len(),
        prepare_started.elapsed().as_millis(),
    );

    let scalar_started = Instant::now();
    let points = directions
        .par_iter()
        .enumerate()
        .map(|(direction_index, direction)| {
            let assessment = match prepared_relative_magnitude {
                Some(prepared) => match &basis_direction_indices {
                    Some(basis_indices) => {
                        super::super::assess_relative_magnitude_basis_period_with_prepared_input(
                            input,
                            basis_indices[direction_index],
                            prepared.mbar,
                            pointwise_inference,
                            null_value,
                            RelativeMagnitudeConfidenceSetConfig {
                                hybrid: relative_magnitude_config.hybrid,
                                hybrid_kappa: pointwise_relative_magnitude_config.hybrid_kappa,
                            },
                            &prepared.branches,
                            &prepared.input_branches,
                        )?
                    }
                    None => super::super::assess_relative_magnitude_functional_with_prepared_functional_branches(
                        input,
                        direction,
                        prepared.mbar,
                        pointwise_inference,
                        null_value,
                        RelativeMagnitudeConfidenceSetConfig {
                            hybrid: relative_magnitude_config.hybrid,
                            hybrid_kappa: pointwise_relative_magnitude_config.hybrid_kappa,
                        },
                        &prepared.branches,
                        &prepared.input_branches,
                        &prepared_functional_branch_sets
                            .as_ref()
                            .ok_or("missing prepared functional branch sets")?[direction_index],
                    )?,
                },
                None => super::super::assess_honest_event_study_functional_with_config(
                    input,
                    direction,
                    sensitivity,
                    pointwise_inference,
                    null_value,
                    relative_magnitude_config,
                )?,
            };
            Ok(HonestDirectionalFunctionalPoint {
                post_weights: direction.clone(),
                assessment,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    emit_directional_region_timing(
        "scalar_evaluations",
        sensitivity,
        directions.len(),
        scalar_started.elapsed().as_millis(),
    );
    emit_directional_region_timing(
        "total_precalibrated",
        sensitivity,
        directions.len(),
        total_started.elapsed().as_millis(),
    );

    Ok(HonestDirectionalRegion {
        confidence_level,
        pointwise_confidence_level,
        calibrated_max_t_critical_value,
        method,
        points,
        diagnostics: HonestDirectionalRegionDiagnostics::fixed(directions.len(), 0),
    })
}

/// Default wrapper for the finite-direction simultaneous region.
///
/// # Errors
/// Returns an error if direction validation, calibration, or scalar directional
/// assessment fails.
pub fn assess_honest_event_study_directional_region(
    input: &HonestEventStudyInput,
    directions: &[Vec<f64>],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
) -> Result<HonestDirectionalRegion, String> {
    assess_honest_event_study_directional_region_with_config(
        input,
        directions,
        sensitivity,
        inference,
        null_value,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        HonestJointPathConfig::default_for_production(),
    )
}
