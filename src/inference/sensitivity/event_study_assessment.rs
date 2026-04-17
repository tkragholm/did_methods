#[path = "event_study_regions.rs"]
mod regions;
#[path = "event_study_workflow.rs"]
mod workflow;

use crate::types::{AttGtEstimate, InferenceConfig};
use rayon::prelude::*;

use super::relative_magnitude::conditional_confidence_set::{
    RelativeMagnitudePreparedFunctionalBranch, RelativeMagnitudePreparedInputBranch,
    compute_relative_magnitude_branch_identified_sets_with_prepared_branches,
    compute_relative_magnitude_confidence_set_for_basis_period_with_prepared_input,
    compute_relative_magnitude_confidence_set_with_prepared_functional_branches_and_identified_sets,
    compute_relative_magnitude_identified_set_for_basis_period,
    global_identified_set_from_branch_identified_sets,
    prepare_relative_magnitude_functional_branches, prepare_relative_magnitude_input_branches,
};
use super::relative_magnitude::geometry::{
    RelativeMagnitudePreparedBranch, basis_post_weights, prepare_relative_magnitude_branches,
};
use super::smoothness::pre_periods_satisfy_smoothness_bound;
use super::{
    HonestAssessment, HonestEventStudyInput, HonestEventStudyPoint, HonestJointPathConfig,
    HonestMultiFlciResult, HonestPostFunctional, HonestResult, HonestSensitivity,
    RelativeMagnitudeConfidenceSetConfig, SmoothnessConfidenceSetConfig,
    relative_magnitude::{
        compute_original_confidence_set,
        compute_relative_magnitude_confidence_set_with_precomputed_sets,
        least_favorable_intervals::{
            build_relative_magnitude_flci_problem, build_relative_magnitude_multi_flci_problem,
            compute_relative_magnitude_flci_with_config,
            compute_relative_magnitude_multi_flci_with_config,
        },
    },
    smoothness::{
        compute_smoothness_confidence_set_with_config, compute_smoothness_identified_set,
    },
};
use crate::inference::z_score_for_confidence;
use crate::util::usize_to_f64;

pub use regions::{
    assess_honest_event_study_directional_region,
    assess_honest_event_study_directional_region_with_config,
    assess_honest_event_study_joint_path_region,
    assess_honest_event_study_joint_path_region_with_config,
    assess_honest_event_study_optimization_surface_region,
    assess_honest_event_study_optimization_surface_region_adaptive,
    assess_honest_event_study_optimization_surface_region_adaptive_with_config,
    assess_honest_event_study_optimization_surface_region_with_config,
};
pub use workflow::{
    assess_honest_event_study_workflow, assess_honest_event_study_workflow_with_config,
};

/// Compute "Honest" confidence intervals for a scalar ATT.
///
/// This implements the Rambachan and Roth (2023) robust inference for a single
/// post-treatment period, given pre-trend estimates.
///
/// # Mathematical implementation (scalar case)
/// 1. Identify the maximum possible bias `δ` in the post-treatment period
///    consistent with the pre-trends and the restriction `M`.
/// 2. Construct the identified set `[τ - δ_max, τ - δ_min]`.
/// 3. Construct the robust confidence interval by adding uncertainty to the
///    identified set.
///
/// # Arguments
/// * `target_estimate` - The ATT estimate for the period of interest.
/// * `pre_period_estimates` - Estimates for the pre-treatment periods.
/// * `vcov` - Variance-covariance matrix of
///   `(pre_period_estimates..., target_estimate)`.
/// * `sensitivity` - The sensitivity bound.
/// * `inference` - Confidence level.
///
/// # Errors
/// Returns an error if covariance dimensions are inconsistent or the
/// sensitivity problem is numerically invalid for the supplied target and
/// pre-period estimates.
pub fn compute_honest_ci_scalar(
    target_estimate: &AttGtEstimate,
    pre_period_estimates: &[AttGtEstimate],
    vcov: &[Vec<f64>],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
) -> Result<HonestResult, String> {
    let k = pre_period_estimates.len();
    if vcov.len() != k + 1 || vcov[0].len() != k + 1 {
        return Err("VCOV matrix dimensions must match pre_periods + 1".to_string());
    }

    let (identified_lower, identified_upper) = match sensitivity {
        HonestSensitivity::Smoothness(m) => {
            let last_pre = pre_period_estimates
                .last()
                .ok_or("requires at least one pre-period")?;
            let beta_pre = last_pre.att;
            let delta_min = -beta_pre - m;
            let delta_max = -beta_pre + m;
            (
                target_estimate.att - delta_max,
                target_estimate.att - delta_min,
            )
        }
        HonestSensitivity::RelativeMagnitude(m) => {
            let max_pre_bias = pre_period_estimates
                .iter()
                .map(|estimate| estimate.att.abs())
                .fold(0.0, f64::max);

            let delta_bound = m * max_pre_bias;
            (
                target_estimate.att - delta_bound,
                target_estimate.att + delta_bound,
            )
        }
    };

    let se = target_estimate.se;
    let z = z_score_for_confidence(inference.confidence_level);
    let robust_lower = z.mul_add(-se, identified_lower);
    let robust_upper = z.mul_add(se, identified_upper);

    Ok(HonestResult {
        original_estimate: target_estimate.att,
        original_se: target_estimate.se,
        robust_ci: (robust_lower, robust_upper),
        identified_set: (identified_lower, identified_upper),
        m: match sensitivity {
            HonestSensitivity::Smoothness(m) | HonestSensitivity::RelativeMagnitude(m) => m,
        },
    })
}

/// Compute scalar `HonestDiD` results for each post-treatment event-study
/// coefficient.
///
/// This assumes the event-study coefficient vector is ordered as pre-period
/// estimates followed by post-period estimates.
///
/// # Errors
/// Returns an error if dimensions are inconsistent or covariance entries are
/// invalid.
pub fn compute_honest_event_study_scalar(
    betahat: &[f64],
    covariance: &[Vec<f64>],
    pre_periods: &[i32],
    post_periods: &[i32],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
) -> Result<Vec<HonestEventStudyPoint>, String> {
    let input = HonestEventStudyInput {
        betahat: betahat.to_vec(),
        covariance: covariance.to_vec(),
        pre_periods: pre_periods.to_vec(),
        post_periods: post_periods.to_vec(),
    };
    compute_honest_event_study_scalar_for_input(&input, sensitivity, inference)
}

/// Compute scalar `HonestDiD` results for each post-treatment coefficient in a
/// typed event-study input.
///
/// # Errors
/// Returns an error if `input` is inconsistent.
pub fn compute_honest_event_study_scalar_for_input(
    input: &HonestEventStudyInput,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
) -> Result<Vec<HonestEventStudyPoint>, String> {
    input.validate()?;

    let prepared_branches: Option<Vec<RelativeMagnitudePreparedBranch>> = match sensitivity {
        HonestSensitivity::RelativeMagnitude(mbar) => Some(prepare_relative_magnitude_branches(
            input.num_pre_periods(),
            input.num_post_periods(),
            mbar,
        )?),
        HonestSensitivity::Smoothness(_) => None,
    };
    let prepared_input_branches = prepared_branches
        .as_ref()
        .map(|prepared| prepare_relative_magnitude_input_branches(input, prepared));

    input
        .post_periods
        .par_iter()
        .enumerate()
        .map(|(post_idx, period)| {
            let result = match sensitivity {
                HonestSensitivity::RelativeMagnitude(mbar) => {
                    let prepared = prepared_branches
                        .as_ref()
                        .ok_or("missing prepared relative-magnitude branch geometry")?;
                    let original = compute_original_confidence_set(
                        input,
                        &basis_post_weights(input.num_post_periods(), post_idx),
                        inference,
                    )?;
                    let identified = compute_relative_magnitude_identified_set_for_basis_period(
                        input, post_idx, prepared,
                    )?;
                    let prepared_input = prepared_input_branches
                        .as_ref()
                        .ok_or("missing prepared relative-magnitude input branch geometry")?;
                    let conditional =
                        compute_relative_magnitude_confidence_set_for_basis_period_with_prepared_input(
                            input,
                            post_idx,
                            inference,
                            RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
                            prepared,
                            prepared_input,
                        )?;
                    HonestResult {
                        original_estimate: original.estimate,
                        original_se: original.se,
                        robust_ci: (conditional.lb, conditional.ub),
                        identified_set: (identified.lb, identified.ub),
                        m: mbar,
                    }
                }
                HonestSensitivity::Smoothness(m) => {
                    let post_weights = basis_post_weights(input.num_post_periods(), post_idx);
                    let original =
                        compute_original_confidence_set(input, &post_weights, inference)?;
                    let identified = compute_smoothness_identified_set(input, &post_weights, m)?;
                    let conditional = compute_smoothness_confidence_set_with_config(
                        input,
                        &post_weights,
                        m,
                        inference,
                        SmoothnessConfidenceSetConfig::from_inference(inference),
                    )?;
                    HonestResult {
                        original_estimate: original.estimate,
                        original_se: original.se,
                        robust_ci: (conditional.lb, conditional.ub),
                        identified_set: (identified.lb, identified.ub),
                        m,
                    }
                }
            };
            Ok(HonestEventStudyPoint {
                post_period: *period,
                result,
            })
        })
        .collect()
}

/// Assess whether a post-treatment event-study linear functional remains
/// statistically distinguishable from `null_value` under the requested
/// `HonestDiD` sensitivity restriction.
///
/// This is the study-facing entrypoint for robustness claims about aggregated
/// post-treatment effects, for example:
///
/// - one specific post period: `post_weights = e_t`
/// - an average over years 11--15: `post_weights = (0, ..., 0, 1/5, ..., 1/5)`
/// - any other fixed linear functional of `τ_post`
///
/// Under `RelativeMagnitude(Mbar)`, this function already uses the exact
/// `ΔRM(Mbar)` identified-set and conditional confidence-set path for the
/// supplied `post_weights`; it is not restricted to basis-vector / first-post
/// functionals. The conditional confidence set uses the default
/// least-favorable hybrid configuration.
///
/// # Errors
/// Returns an error if the event-study input is inconsistent or if the chosen
/// sensitivity routine fails.
pub fn assess_honest_event_study_functional(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
) -> Result<HonestAssessment, String> {
    assess_honest_event_study_functional_with_config(
        input,
        post_weights,
        sensitivity,
        inference,
        null_value,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
    )
}

/// Assess whether a post-treatment event-study linear functional remains
/// statistically distinguishable from `null_value`.
///
/// This variant uses the requested `HonestDiD` sensitivity restriction with
/// explicit `DeltaRM` hybrid control.
///
/// This is the same study-facing assessment as
/// [`assess_honest_event_study_functional`], but it allows callers to choose
/// the `DeltaRM` conditional confidence-set hybrid mode explicitly. This only
/// affects [`HonestSensitivity::RelativeMagnitude`]; smoothness calculations
/// ignore `relative_magnitude_config`.
///
/// The default wrapper uses
/// [`RelativeMagnitudeConfidenceSetConfig::from_inference`] which corresponds to the
/// least-favorable (`LF`) hybrid path in the reference `HonestDiD` package.
///
/// # Errors
/// Returns an error if the event-study input is inconsistent or if the chosen
/// sensitivity routine fails.
#[allow(clippy::too_many_lines)]
pub fn assess_honest_event_study_functional_with_config(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestAssessment, String> {
    let original = compute_original_confidence_set(input, post_weights, inference)?;
    let (robust_ci, identified_set, sensitivity_value) = match sensitivity {
        HonestSensitivity::RelativeMagnitude(mbar) => {
            let identified = super::relative_magnitude::compute_relative_magnitude_identified_set(
                input,
                post_weights,
                mbar,
            )?;
            let conditional = compute_relative_magnitude_confidence_set_with_precomputed_sets(
                input,
                post_weights,
                mbar,
                inference,
                relative_magnitude_config,
                &original,
                &identified,
            )?;
            (
                (conditional.lb, conditional.ub),
                (identified.lb, identified.ub),
                mbar,
            )
        }
        HonestSensitivity::Smoothness(m) => {
            if post_weights.len() != input.num_post_periods() {
                return Err(format!(
                    "post_weights length {} does not match number of post periods {}",
                    post_weights.len(),
                    input.num_post_periods()
                ));
            }
            if post_weights.iter().all(|weight| weight.abs() <= 1e-12) {
                return Err(
                    "Smoothness sensitivity requires at least one non-zero post_weights weight"
                        .to_string(),
                );
            }
            if pre_periods_satisfy_smoothness_bound(&input.betahat[..input.num_pre_periods()], m) {
                let identified = compute_smoothness_identified_set(input, post_weights, m)?;
                let conditional = compute_smoothness_confidence_set_with_config(
                    input,
                    post_weights,
                    m,
                    inference,
                    SmoothnessConfidenceSetConfig::from_inference(inference),
                )?;
                let identified_ci = (identified.lb, identified.ub);
                let robust_ci = (conditional.lb, conditional.ub);
                (robust_ci, identified_ci, m)
            } else {
                (
                    (f64::NEG_INFINITY, f64::INFINITY),
                    (f64::NEG_INFINITY, f64::INFINITY),
                    m,
                )
            }
        }
    };
    Ok(HonestAssessment {
        original_ci: original.ci,
        robust_ci,
        identified_set,
        estimate: original.estimate,
        se: original.se,
        null_value,
        sensitivity_value,
        robustly_significant: robust_ci.0 > null_value || robust_ci.1 < null_value,
    })
}

/// Convenience wrapper for a single post-treatment event-study period.
///
/// # Errors
/// Returns an error if `post_period` is not present in the event-study input or
/// if the underlying `HonestDiD` calculation fails.
pub fn assess_honest_event_study_period(
    input: &HonestEventStudyInput,
    post_period: i32,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
) -> Result<HonestAssessment, String> {
    let Some(post_idx) = input
        .post_periods
        .iter()
        .position(|period| *period == post_period)
    else {
        return Err(format!(
            "post period {post_period} is not present in HonestEventStudyInput"
        ));
    };
    let mut post_weights = vec![0.0; input.num_post_periods()];
    post_weights[post_idx] = 1.0;
    assess_honest_event_study_functional(input, &post_weights, sensitivity, inference, null_value)
}

/// Version of [`assess_honest_event_study_period`] with explicit `DeltaRM`
/// conditional confidence-set hybrid control.
///
/// # Errors
/// Returns an error if `post_period` is not present in the event-study input or
/// if the underlying `HonestDiD` calculation fails.
pub fn assess_honest_event_study_period_with_config(
    input: &HonestEventStudyInput,
    post_period: i32,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestAssessment, String> {
    let Some(post_idx) = input
        .post_periods
        .iter()
        .position(|period| *period == post_period)
    else {
        return Err(format!(
            "post period {post_period} is not present in HonestEventStudyInput"
        ));
    };
    let mut post_weights = vec![0.0; input.num_post_periods()];
    post_weights[post_idx] = 1.0;
    assess_honest_event_study_functional_with_config(
        input,
        &post_weights,
        sensitivity,
        inference,
        null_value,
        relative_magnitude_config,
    )
}

fn post_weights_for_functional(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
) -> Result<Vec<f64>, String> {
    let mut post_weights = vec![0.0; input.num_post_periods()];
    match functional {
        HonestPostFunctional::Period(post_period) => {
            let Some(post_idx) = input
                .post_periods
                .iter()
                .position(|period| period == post_period)
            else {
                return Err(format!(
                    "post period {post_period} is not present in HonestEventStudyInput"
                ));
            };
            post_weights[post_idx] = 1.0;
        }
        HonestPostFunctional::AverageWindow {
            start_period,
            end_period,
        } => {
            if start_period > end_period {
                return Err(format!(
                    "average window start_period {start_period} exceeds end_period {end_period}"
                ));
            }
            let covered: Vec<usize> = input
                .post_periods
                .iter()
                .enumerate()
                .filter_map(|(idx, period)| {
                    (*period >= *start_period && *period <= *end_period).then_some(idx)
                })
                .collect();
            if covered.is_empty() {
                return Err(format!(
                    "average window [{start_period}, {end_period}] covers no post periods"
                ));
            }
            let weight = 1.0 / usize_to_f64(covered.len());
            for idx in covered {
                post_weights[idx] = weight;
            }
        }
        HonestPostFunctional::Weighted(weights) => {
            if weights.is_empty() {
                return Err(
                    "weighted functional requires at least one post-period weight".to_string(),
                );
            }
            for (post_period, weight) in weights {
                let Some(post_idx) = input
                    .post_periods
                    .iter()
                    .position(|period| period == post_period)
                else {
                    return Err(format!(
                        "post period {post_period} is not present in HonestEventStudyInput"
                    ));
                };
                post_weights[post_idx] += *weight;
            }
        }
    }
    Ok(post_weights)
}

/// Assess a typed post-treatment linear functional under `HonestDiD`
/// sensitivity restrictions.
///
/// This is the preferred study-facing surface for multi-period claims such as:
///
/// - one specific post-treatment year
/// - an average over years 11--15
/// - a custom weighted long-run effect
///
/// # Errors
/// Returns an error if the requested functional cannot be represented on the
/// supplied post-period support or if the underlying `HonestDiD` calculation
/// fails.
pub fn assess_honest_event_study_post_functional(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
) -> Result<HonestAssessment, String> {
    let post_weights = post_weights_for_functional(input, functional)?;
    assess_honest_event_study_functional(input, &post_weights, sensitivity, inference, null_value)
}

/// Version of [`assess_honest_event_study_post_functional`] with explicit
/// `DeltaRM` conditional confidence-set hybrid control.
///
/// # Errors
/// Returns an error if the requested functional cannot be represented on the
/// supplied post-period support or if the underlying `HonestDiD` calculation
/// fails.
pub fn assess_honest_event_study_post_functional_with_config(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestAssessment, String> {
    let post_weights = post_weights_for_functional(input, functional)?;
    assess_honest_event_study_functional_with_config(
        input,
        &post_weights,
        sensitivity,
        inference,
        null_value,
        relative_magnitude_config,
    )
}

#[allow(clippy::too_many_arguments)]
fn assess_relative_magnitude_functional_with_prepared_functional_branches(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
    prepared_functional_branches: &[RelativeMagnitudePreparedFunctionalBranch],
) -> Result<HonestAssessment, String> {
    let original = compute_original_confidence_set(input, post_weights, inference)?;
    let branch_identified_sets =
        compute_relative_magnitude_branch_identified_sets_with_prepared_branches(
            input,
            post_weights,
            prepared_branches,
        )?;
    let identified = global_identified_set_from_branch_identified_sets(
        input,
        post_weights,
        &branch_identified_sets,
    )?;
    let conditional =
        compute_relative_magnitude_confidence_set_with_prepared_functional_branches_and_identified_sets(
        input,
        post_weights,
        inference,
        relative_magnitude_config,
        &original,
        &identified,
        prepared_branches,
        prepared_input_branches,
        prepared_functional_branches,
        &branch_identified_sets,
    )?;
    let robust_ci = (conditional.lb, conditional.ub);
    Ok(HonestAssessment {
        original_ci: original.ci,
        robust_ci,
        identified_set: (identified.lb, identified.ub),
        estimate: original.estimate,
        se: original.se,
        null_value,
        sensitivity_value: mbar,
        robustly_significant: robust_ci.0 > null_value || robust_ci.1 < null_value,
    })
}

#[allow(clippy::too_many_arguments)]
fn assess_relative_magnitude_basis_period_with_prepared_input(
    input: &HonestEventStudyInput,
    post_idx: usize,
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
) -> Result<HonestAssessment, String> {
    let post_weights = basis_post_weights(input.num_post_periods(), post_idx);
    let prepared_functional_branches = prepare_relative_magnitude_functional_branches(
        &post_weights,
        prepared_branches,
        prepared_input_branches,
    )?;
    assess_relative_magnitude_basis_period_with_prepared_functional_branches(
        input,
        &post_weights,
        mbar,
        inference,
        null_value,
        relative_magnitude_config,
        prepared_branches,
        prepared_input_branches,
        &prepared_functional_branches,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::inference::sensitivity) fn assess_relative_magnitude_basis_period_with_prepared_functional_branches(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
    prepared_functional_branches: &[RelativeMagnitudePreparedFunctionalBranch],
) -> Result<HonestAssessment, String> {
    let original = compute_original_confidence_set(input, post_weights, inference)?;
    let branch_identified_sets =
        compute_relative_magnitude_branch_identified_sets_with_prepared_branches(
            input,
            post_weights,
            prepared_branches,
        )?;
    let identified = global_identified_set_from_branch_identified_sets(
        input,
        post_weights,
        &branch_identified_sets,
    )?;
    let conditional =
        compute_relative_magnitude_confidence_set_with_prepared_functional_branches_and_identified_sets(
        input,
        post_weights,
        inference,
        relative_magnitude_config,
        &original,
        &identified,
        prepared_branches,
        prepared_input_branches,
        prepared_functional_branches,
        &branch_identified_sets,
    )?;
    let robust_ci = (conditional.lb, conditional.ub);
    Ok(HonestAssessment {
        original_ci: original.ci,
        robust_ci,
        identified_set: (identified.lb, identified.ub),
        estimate: original.estimate,
        se: original.se,
        null_value,
        sensitivity_value: mbar,
        robustly_significant: robust_ci.0 > null_value || robust_ci.1 < null_value,
    })
}

struct PreparedRelativeMagnitudeBaseContext {
    mbar: f64,
    branches: Vec<RelativeMagnitudePreparedBranch>,
    input_branches: Vec<RelativeMagnitudePreparedInputBranch>,
}

fn prepare_relative_magnitude_base_context(
    input: &HonestEventStudyInput,
    mbar: f64,
) -> Result<PreparedRelativeMagnitudeBaseContext, String> {
    let branches = prepare_relative_magnitude_branches(
        input.num_pre_periods(),
        input.num_post_periods(),
        mbar,
    )?;
    let input_branches = prepare_relative_magnitude_input_branches(input, &branches);
    Ok(PreparedRelativeMagnitudeBaseContext {
        mbar,
        branches,
        input_branches,
    })
}

/// Assess a typed post-treatment functional using the current least-favorable
/// `DeltaRM` interval surface.
///
/// This is the study-facing FLCI entrypoint for `DeltaRM`. It currently
/// surfaces the validated least-favorable conditional interval for the target
/// functional, packaged into the same [`HonestAssessment`] shape used by the
/// other event-study sensitivity helpers.
///
/// # Errors
/// Returns an error if the functional is invalid for the supplied event-study
/// input or if the underlying `DeltaRM` least-favorable interval path fails.
pub fn assess_honest_event_study_post_functional_flci(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
) -> Result<HonestAssessment, String> {
    assess_honest_event_study_post_functional_flci_with_config(
        input,
        functional,
        mbar,
        inference,
        null_value,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
    )
}

/// Assess a typed post-treatment functional using the current least-favorable
/// `DeltaRM` interval surface with an explicit hybrid configuration.
///
/// # Errors
/// Returns an error if the functional is invalid for the supplied event-study
/// input or if the underlying `DeltaRM` least-favorable interval path fails.
pub fn assess_honest_event_study_post_functional_flci_with_config(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestAssessment, String> {
    let post_weights = post_weights_for_functional(input, functional)?;
    let problem = build_relative_magnitude_flci_problem(input, &post_weights, mbar, inference)?;
    let flci = compute_relative_magnitude_flci_with_config(&problem, relative_magnitude_config)?;
    Ok(HonestAssessment {
        original_ci: flci.original_ci,
        robust_ci: flci.flci,
        identified_set: flci.identified_set,
        estimate: problem.original.estimate,
        se: problem.original.se,
        null_value,
        sensitivity_value: mbar,
        robustly_significant: flci.flci.0 > null_value || flci.flci.1 < null_value,
    })
}

/// Assess multiple typed post-treatment functionals simultaneously using a
/// jointly calibrated least-favorable `DeltaRM` interval surface.
///
/// # Errors
/// Returns an error if any functional is invalid for the supplied input or if
/// the underlying joint multi-functional FLCI solve fails.
pub fn assess_honest_event_study_post_functional_multi_flci(
    input: &HonestEventStudyInput,
    functionals: &[HonestPostFunctional],
    mbar: f64,
    inference: InferenceConfig,
) -> Result<HonestMultiFlciResult, String> {
    assess_honest_event_study_post_functional_multi_flci_with_config(
        input,
        functionals,
        mbar,
        inference,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        HonestJointPathConfig::default_for_production(),
    )
}

/// Version of [`assess_honest_event_study_post_functional_multi_flci`] with
/// explicit scalar-hybrid and joint-calibration controls.
///
/// # Errors
/// Returns an error if any functional is invalid for the supplied input or if
/// the underlying joint multi-functional FLCI solve fails.
pub fn assess_honest_event_study_post_functional_multi_flci_with_config(
    input: &HonestEventStudyInput,
    functionals: &[HonestPostFunctional],
    mbar: f64,
    inference: InferenceConfig,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
) -> Result<HonestMultiFlciResult, String> {
    if functionals.is_empty() {
        return Err("multi-functional FLCI requires at least one functional".to_string());
    }
    let post_weight_sets = functionals
        .iter()
        .map(|functional| post_weights_for_functional(input, functional))
        .collect::<Result<Vec<_>, String>>()?;
    let problem =
        build_relative_magnitude_multi_flci_problem(input, &post_weight_sets, mbar, inference)?;
    compute_relative_magnitude_multi_flci_with_config(
        &problem,
        relative_magnitude_config,
        joint_config,
    )
}
