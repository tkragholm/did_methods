use super::super::relative_magnitude::conditional_confidence_set::RelativeMagnitudePreparedFunctionalBranch;
use super::super::relative_magnitude::least_favorable_intervals::{
    build_relative_magnitude_multi_flci_problem_with_precomputed_sets,
    compute_relative_magnitude_multi_flci_with_config,
    compute_relative_magnitude_multi_flci_with_prepared_branches,
    compute_relative_magnitude_multi_flci_with_prepared_functional_branches,
};
use super::super::smoothness::least_favorable_intervals::{
    SmoothnessFlciConfig, build_smoothness_multi_flci_problem,
    compute_smoothness_multi_flci_with_config,
};
use super::super::{
    HonestDirectionalFunctionalPoint, HonestDirectionalRegion, HonestDirectionalRegionDiagnostics,
    HonestFunctionalAssessmentPoint, HonestIdentifiedSet, HonestJointPathConfig,
    HonestJointPathRegion, HonestMultiFlciResult, HonestOptimizationSurfaceAdaptiveRunConfig,
    HonestOriginalConfidenceSet, HonestPostFunctional, HonestSensitivity,
    HonestStudyWorkflowResult, HonestWorkflowConfig, HonestWorkflowDirectionMode,
};
use super::HonestEventStudyInput;
use crate::types::InferenceConfig;
use rayon::prelude::*;
use std::time::Instant;
use tracing::debug;

const fn sensitivity_label(sensitivity: HonestSensitivity) -> &'static str {
    match sensitivity {
        HonestSensitivity::RelativeMagnitude(_) => "relative_magnitude",
        HonestSensitivity::Smoothness(_) => "smoothness",
    }
}

fn emit_workflow_timing(
    step: &str,
    sensitivity: HonestSensitivity,
    functionals: usize,
    post_periods: usize,
    duration_ms: u128,
) {
    debug!(
        target: "did_profile",
        scope = "honest_workflow",
        step,
        sensitivity_kind = sensitivity_label(sensitivity),
        functionals,
        post_periods,
        duration_ms,
        "honest workflow timing"
    );
}

/// High-level one-shot `HonestDiD` workflow for study-facing robustness outputs.
///
/// This computes:
/// - scalar assessments for requested functionals
/// - a simultaneous joint post-treatment path region
/// - a simultaneous directional region
/// - optional multi-functional `DeltaRM` least-favorable intervals (when
///   `sensitivity` is `RelativeMagnitude`)
///
/// # Errors
/// Returns an error if any underlying component computation fails.
pub fn assess_honest_event_study_workflow(
    input: &HonestEventStudyInput,
    functionals: &[HonestPostFunctional],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
) -> Result<HonestStudyWorkflowResult, String> {
    assess_honest_event_study_workflow_with_config(
        input,
        functionals,
        sensitivity,
        inference,
        null_value,
        &HonestWorkflowConfig {
            relative_magnitude: super::RelativeMagnitudeConfidenceSetConfig::from_inference(
                inference,
            ),
            joint: HonestJointPathConfig::default_for_production(),
            direction_mode: HonestWorkflowDirectionMode::default_for_production(),
        },
    )
}

/// Configurable version of [`assess_honest_event_study_workflow`].
///
/// # Errors
/// Returns an error if any underlying component computation fails.
#[allow(clippy::too_many_lines)]
#[cfg_attr(feature = "hotpath", hotpath::measure)]
pub fn assess_honest_event_study_workflow_with_config(
    input: &HonestEventStudyInput,
    functionals: &[HonestPostFunctional],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    config: &HonestWorkflowConfig,
) -> Result<HonestStudyWorkflowResult, String> {
    let workflow_started = Instant::now();
    if functionals.is_empty() {
        return Err("workflow requires at least one functional".to_string());
    }
    let workflow_post_weights = functionals
        .iter()
        .map(|functional| super::post_weights_for_functional(input, functional))
        .collect::<Result<Vec<_>, String>>()?;
    let prepare_started = Instant::now();
    let prepared_relative_magnitude = match sensitivity {
        HonestSensitivity::RelativeMagnitude(mbar) => {
            Some(super::prepare_relative_magnitude_base_context(input, mbar)?)
        }
        HonestSensitivity::Smoothness(_) => None,
    };
    let prepared_functional_branch_sets = match &prepared_relative_magnitude {
        Some(prepared) => Some(
            workflow_post_weights
                .par_iter()
                .map(|post_weights| {
                    super::prepare_relative_magnitude_functional_branches(
                        post_weights,
                        &prepared.branches,
                        &prepared.input_branches,
                    )
                })
                .collect::<Result<Vec<_>, String>>()?,
        ),
        None => None,
    };
    emit_workflow_timing(
        "prepare_relative_magnitude",
        sensitivity,
        functionals.len(),
        input.num_post_periods(),
        prepare_started.elapsed().as_millis(),
    );

    let functional_started = Instant::now();
    let functional_assessments = match (
        &prepared_relative_magnitude,
        &prepared_functional_branch_sets,
    ) {
        (Some(prepared), Some(prepared_functional_branches)) => functionals
            .par_iter()
            .zip(workflow_post_weights.par_iter())
            .zip(prepared_functional_branches.par_iter())
            .map(|((functional, post_weights), prepared_functional_branch)| {
                let assessment =
                    super::assess_relative_magnitude_functional_with_prepared_functional_branches(
                        input,
                        post_weights,
                        prepared.mbar,
                        inference,
                        null_value,
                        config.relative_magnitude,
                        &prepared.branches,
                        &prepared.input_branches,
                        prepared_functional_branch,
                    )?;
                Ok(HonestFunctionalAssessmentPoint {
                    functional: functional.clone(),
                    assessment,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
        _ => functionals
            .par_iter()
            .zip(workflow_post_weights.par_iter())
            .map(|(functional, post_weights)| {
                let assessment = super::assess_honest_event_study_functional_with_config(
                    input,
                    post_weights,
                    sensitivity,
                    inference,
                    null_value,
                    config.relative_magnitude,
                )?;
                Ok(HonestFunctionalAssessmentPoint {
                    functional: functional.clone(),
                    assessment,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    };
    emit_workflow_timing(
        "functional_assessments",
        sensitivity,
        functionals.len(),
        input.num_post_periods(),
        functional_started.elapsed().as_millis(),
    );

    let joint_started = Instant::now();
    let joint_path_region =
        super::regions::assess_honest_event_study_joint_path_region_with_optional_prepared_relative_magnitude(
            input,
            sensitivity,
            inference,
            null_value,
            config.relative_magnitude,
            config.joint,
            prepared_relative_magnitude.as_ref(),
        )?;
    emit_workflow_timing(
        "joint_path_region",
        sensitivity,
        functionals.len(),
        input.num_post_periods(),
        joint_started.elapsed().as_millis(),
    );

    let directional_started = Instant::now();
    let directional_region = match config.direction_mode {
        HonestWorkflowDirectionMode::Basis => {
            directional_region_from_joint_path_region(input, &joint_path_region)
        }
        _ => build_workflow_directional_region(
            input,
            sensitivity,
            inference,
            null_value,
            config,
            prepared_relative_magnitude.as_ref(),
        )?,
    };
    emit_workflow_timing(
        "directional_region",
        sensitivity,
        functionals.len(),
        input.num_post_periods(),
        directional_started.elapsed().as_millis(),
    );
    let multi_flci_started = Instant::now();
    let multi_flci = build_workflow_multi_function_intervals(
        input,
        &workflow_post_weights,
        &functional_assessments,
        sensitivity,
        inference,
        config,
        prepared_relative_magnitude.as_ref(),
        prepared_functional_branch_sets.as_deref(),
    )?;
    emit_workflow_timing(
        "multi_flci",
        sensitivity,
        functionals.len(),
        input.num_post_periods(),
        multi_flci_started.elapsed().as_millis(),
    );
    emit_workflow_timing(
        "total",
        sensitivity,
        functionals.len(),
        input.num_post_periods(),
        workflow_started.elapsed().as_millis(),
    );

    Ok(HonestStudyWorkflowResult {
        functional_assessments,
        joint_path_region,
        directional_region,
        multi_flci,
    })
}

fn directional_region_from_joint_path_region(
    input: &HonestEventStudyInput,
    joint_path_region: &HonestJointPathRegion,
) -> HonestDirectionalRegion {
    let points = joint_path_region
        .points
        .iter()
        .map(|point| {
            let post_weights = super::post_weights_for_functional(
                input,
                &HonestPostFunctional::Period(point.post_period),
            )
            .expect("joint path post period should always map to a basis functional");
            HonestDirectionalFunctionalPoint {
                post_weights,
                assessment: point.assessment.clone(),
            }
        })
        .collect();
    HonestDirectionalRegion {
        confidence_level: joint_path_region.confidence_level,
        pointwise_confidence_level: joint_path_region.pointwise_confidence_level,
        calibrated_max_t_critical_value: joint_path_region.calibrated_max_t_critical_value,
        method: joint_path_region.method,
        points,
        diagnostics: HonestDirectionalRegionDiagnostics::fixed(joint_path_region.points.len(), 0),
    }
}

#[cfg_attr(feature = "hotpath", hotpath::measure)]
fn build_workflow_directional_region(
    input: &HonestEventStudyInput,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    config: &HonestWorkflowConfig,
    prepared_relative_magnitude: Option<&super::PreparedRelativeMagnitudeBaseContext>,
) -> Result<HonestDirectionalRegion, String> {
    match &config.direction_mode {
        HonestWorkflowDirectionMode::Basis => {
            let mut directions =
                vec![vec![0.0; input.num_post_periods()]; input.num_post_periods()];
            for (idx, direction) in directions.iter_mut().enumerate() {
                direction[idx] = 1.0;
            }
            super::regions::assess_honest_event_study_directional_region_with_optional_prepared_relative_magnitude(
                input,
                &directions,
                sensitivity,
                inference,
                null_value,
                config.relative_magnitude,
                config.joint,
                prepared_relative_magnitude,
            )
        }
        HonestWorkflowDirectionMode::Custom(directions) => {
            super::regions::assess_honest_event_study_directional_region_with_optional_prepared_relative_magnitude(
                input,
                directions,
                sensitivity,
                inference,
                null_value,
                config.relative_magnitude,
                config.joint,
                prepared_relative_magnitude,
            )
        }
        HonestWorkflowDirectionMode::OptimizationSurface(surface) => {
            super::regions::assess_honest_event_study_optimization_surface_region_with_config(
                input,
                sensitivity,
                inference,
                null_value,
                config.relative_magnitude,
                config.joint,
                *surface,
            )
        }
        HonestWorkflowDirectionMode::OptimizationSurfaceAdaptive { surface, adaptive } => {
            super::regions::assess_honest_event_study_optimization_surface_region_adaptive_with_config(
                input,
                sensitivity,
                inference,
                null_value,
                HonestOptimizationSurfaceAdaptiveRunConfig {
                    relative_magnitude: config.relative_magnitude,
                    joint: config.joint,
                    surface: *surface,
                    adaptive: *adaptive,
                },
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_workflow_multi_function_intervals(
    input: &HonestEventStudyInput,
    post_weight_sets: &[Vec<f64>],
    functional_assessments: &[HonestFunctionalAssessmentPoint],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    config: &HonestWorkflowConfig,
    prepared_relative_magnitude: Option<&super::PreparedRelativeMagnitudeBaseContext>,
    prepared_functional_branch_sets: Option<&[Vec<RelativeMagnitudePreparedFunctionalBranch>]>,
) -> Result<Option<HonestMultiFlciResult>, String> {
    match sensitivity {
        HonestSensitivity::RelativeMagnitude(mbar) => {
            let originals = functional_assessments
                .iter()
                .map(|point| HonestOriginalConfidenceSet {
                    estimate: point.assessment.estimate,
                    se: point.assessment.se,
                    ci: point.assessment.original_ci,
                })
                .collect();
            let identified_sets = functional_assessments
                .iter()
                .map(|point| HonestIdentifiedSet {
                    lb: point.assessment.identified_set.0,
                    ub: point.assessment.identified_set.1,
                })
                .collect();
            let problem = build_relative_magnitude_multi_flci_problem_with_precomputed_sets(
                input,
                post_weight_sets,
                mbar,
                inference,
                originals,
                identified_sets,
            )?;
            let result = match prepared_relative_magnitude {
                Some(prepared) => match prepared_functional_branch_sets {
                    Some(prepared_functional_branches) => {
                        compute_relative_magnitude_multi_flci_with_prepared_functional_branches(
                            &problem,
                            config.relative_magnitude,
                            config.joint,
                            &prepared.branches,
                            &prepared.input_branches,
                            prepared_functional_branches,
                        )?
                    }
                    None => compute_relative_magnitude_multi_flci_with_prepared_branches(
                        &problem,
                        config.relative_magnitude,
                        config.joint,
                        &prepared.branches,
                        &prepared.input_branches,
                    )?,
                },
                None => compute_relative_magnitude_multi_flci_with_config(
                    &problem,
                    config.relative_magnitude,
                    config.joint,
                )?,
            };
            Ok(Some(result))
        }
        HonestSensitivity::Smoothness(m) => {
            let problem =
                build_smoothness_multi_flci_problem(input, post_weight_sets, m, inference)?;
            let result = compute_smoothness_multi_flci_with_config(
                &problem,
                SmoothnessFlciConfig::default_for_production(),
                config.joint,
            )?;
            Ok(Some(result))
        }
    }
}
