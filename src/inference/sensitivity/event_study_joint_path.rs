use super::super::super::linear_algebra::{
    cholesky_lower, critical_value_from_pointwise_confidence,
    pointwise_confidence_level_from_critical, simulated_lower_cholesky_maxima_batched,
    simulation_rank,
};
use super::super::super::{
    HonestEventStudyInput, HonestJointPathConfig, HonestJointPathMethod, HonestJointPathPoint,
    HonestJointPathRegion, HonestPostFunctional, HonestSensitivity,
    RelativeMagnitudeConfidenceSetConfig,
};
use crate::inference::validate_confidence_level;
use crate::types::InferenceConfig;
use crate::util::usize_to_f64;
use rayon::prelude::*;

/// Construct a conservative joint confidence region for the full
/// post-treatment path.
///
/// The current implementation computes a simultaneous rectangle across all
/// post-treatment periods and then evaluates the corresponding scalar
/// `HonestDiD` interval for each period at an adjusted pointwise confidence
/// level. The adjustment can be Bonferroni or a tighter covariance-aware
/// simulated max-`t` calibration.
///
/// # Errors
/// Returns an error if the event-study input is inconsistent or if any scalar
/// period-level `HonestDiD` assessment fails.
pub fn assess_honest_event_study_joint_path_region(
    input: &HonestEventStudyInput,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
) -> Result<HonestJointPathRegion, String> {
    assess_honest_event_study_joint_path_region_with_config(
        input,
        sensitivity,
        inference,
        null_value,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        HonestJointPathConfig::default_for_production(),
    )
}

fn build_post_correlation_block(input: &HonestEventStudyInput) -> Vec<Vec<f64>> {
    let num_post = input.num_post_periods();
    let start = input.num_pre_periods();
    let mut corr = vec![vec![0.0; num_post]; num_post];
    let stddevs: Vec<f64> = (0..num_post)
        .map(|idx| input.covariance[start + idx][start + idx].max(0.0).sqrt())
        .collect();
    for i in 0..num_post {
        for j in 0..num_post {
            let denom = stddevs[i] * stddevs[j];
            corr[i][j] = if i == j {
                1.0
            } else if denom <= 1e-12 {
                0.0
            } else {
                (input.covariance[start + i][start + j] / denom).clamp(-1.0, 1.0)
            };
        }
    }
    corr
}

fn simulated_joint_pointwise_confidence_level(
    input: &HonestEventStudyInput,
    confidence_level: f64,
    simulation_draws: usize,
    simulation_seed: u64,
) -> Result<f64, String> {
    if simulation_draws == 0 {
        return Err("joint path simulation_draws must be positive".to_string());
    }
    let corr = build_post_correlation_block(input);
    let chol = cholesky_lower(&corr)?;
    let mut maxima =
        simulated_lower_cholesky_maxima_batched(&chol, simulation_draws, simulation_seed);
    let n = maxima.len();
    let rank = simulation_rank(n, confidence_level);
    maxima.select_nth_unstable_by(rank.min(n - 1), f64::total_cmp);
    let critical = maxima[rank.min(n - 1)];
    Ok(critical)
}

fn joint_pointwise_confidence_level(
    input: &HonestEventStudyInput,
    inference: InferenceConfig,
    joint_config: HonestJointPathConfig,
) -> Result<(f64, f64), String> {
    let num_post = input.num_post_periods();
    if num_post == 0 {
        return Err("joint path region requires at least one post-treatment period".to_string());
    }
    if num_post == 1 {
        return Ok((
            inference.confidence_level,
            critical_value_from_pointwise_confidence(inference.confidence_level)?,
        ));
    }
    let alpha = 1.0 - inference.confidence_level;
    match joint_config.method {
        HonestJointPathMethod::Bonferroni => {
            let pointwise = 1.0 - alpha / usize_to_f64(num_post);
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
            let critical = simulated_joint_pointwise_confidence_level(
                input,
                inference.confidence_level,
                joint_config.simulation_draws,
                joint_config.simulation_seed,
            )?;
            Ok((
                pointwise_confidence_level_from_critical(critical)?,
                critical,
            ))
        }
    }
}

/// Version of [`assess_honest_event_study_joint_path_region`] with explicit
/// `DeltaRM` hybrid control and joint-path calibration control.
///
/// # Errors
/// Returns an error if the event-study input is inconsistent, the implied
/// simultaneous pointwise confidence level is invalid, or any scalar period-level
/// `HonestDiD` assessment fails.
pub fn assess_honest_event_study_joint_path_region_with_config(
    input: &HonestEventStudyInput,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
) -> Result<HonestJointPathRegion, String> {
    assess_honest_event_study_joint_path_region_with_optional_prepared_relative_magnitude(
        input,
        sensitivity,
        inference,
        null_value,
        relative_magnitude_config,
        joint_config,
        None,
    )
}

pub(in crate::inference::sensitivity::event_study_assessment) fn assess_honest_event_study_joint_path_region_with_optional_prepared_relative_magnitude(
    input: &HonestEventStudyInput,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
    prepared_relative_magnitude: Option<&super::super::PreparedRelativeMagnitudeBaseContext>,
) -> Result<HonestJointPathRegion, String> {
    input.validate()?;
    let (pointwise_confidence_level, calibrated_max_t_critical_value) =
        joint_pointwise_confidence_level(input, inference, joint_config)?;
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
    let basis_weights = prepared_relative_magnitude.map(|_| {
        (0..input.num_post_periods())
            .map(|post_idx| super::super::basis_post_weights(input.num_post_periods(), post_idx))
            .collect::<Vec<_>>()
    });
    let prepared_functional_branch_sets =
        match (prepared_relative_magnitude, basis_weights.as_ref()) {
            (Some(prepared), Some(basis_weights)) => Some(
                basis_weights
                    .par_iter()
                    .map(|post_weights| {
                        super::super::prepare_relative_magnitude_functional_branches(
                            post_weights,
                            &prepared.branches,
                            &prepared.input_branches,
                        )
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            ),
            _ => None,
        };
    let relative_magnitude_pointwise_config = RelativeMagnitudeConfidenceSetConfig {
        hybrid: relative_magnitude_config.hybrid,
        hybrid_kappa: pointwise_relative_magnitude_config.hybrid_kappa,
    };
    let points = input
        .post_periods
        .par_iter()
        .enumerate()
        .map(|(point_index, post_period)| {
            let functional = HonestPostFunctional::Period(*post_period);
            let assessment = match (
                prepared_relative_magnitude,
                basis_weights.as_ref(),
                prepared_functional_branch_sets.as_ref(),
            ) {
                (Some(prepared), Some(basis_weights), Some(prepared_functional_branch_sets)) => {
                    super::super::assess_relative_magnitude_basis_period_with_prepared_functional_branches(
                        input,
                        &basis_weights[point_index],
                        prepared.mbar,
                        pointwise_inference,
                        null_value,
                        relative_magnitude_pointwise_config,
                        &prepared.branches,
                        &prepared.input_branches,
                        &prepared_functional_branch_sets[point_index],
                    )?
                }
                _ => super::super::assess_honest_event_study_post_functional_with_config(
                    input,
                    &functional,
                    sensitivity,
                    pointwise_inference,
                    null_value,
                    relative_magnitude_config,
                )?,
            };
            Ok(HonestJointPathPoint {
                post_period: *post_period,
                assessment,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(HonestJointPathRegion {
        confidence_level: inference.confidence_level,
        pointwise_confidence_level,
        calibrated_max_t_critical_value,
        method: joint_config.method,
        points,
    })
}
