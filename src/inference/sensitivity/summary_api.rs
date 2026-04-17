use statrs::distribution::{ContinuousCDF, Normal};

use super::linear_algebra::{diag_sqrt, linear_grid, mat_vec_mul_into, sandwich_covariance};
use super::relative_magnitude::{
    compute_linear_trend_relative_magnitude_confidence_set_with_config,
    compute_monotone_linear_trend_relative_magnitude_confidence_set_with_config,
    compute_monotone_relative_magnitude_confidence_set_with_config,
    compute_original_confidence_set, compute_relative_magnitude_confidence_set_with_config,
    compute_relative_magnitude_identified_set,
    compute_signed_linear_trend_relative_magnitude_confidence_set_with_config,
    compute_signed_relative_magnitude_confidence_set_with_config, geometry::basis_post_weights,
};
use super::smoothness::least_favorable_intervals::{
    SmoothnessFlciConfig, build_smoothness_flci_problem, compute_smoothness_flci_with_config,
};
use super::smoothness::{
    build_smoothness_constraint_matrix, compute_monotone_smoothness_confidence_set_with_config,
    compute_signed_smoothness_confidence_set_with_config,
    compute_smoothness_confidence_set_with_config,
};
use super::{
    HonestBiasDirection, HonestConditionalConfidenceSet, HonestEventStudyInput,
    HonestMonotonicityDirection, HonestOriginalConfidenceSet, HonestRelativeMagnitudeBound,
    RelativeMagnitudeConfidenceSetConfig, RelativeMagnitudeHybrid, SmoothnessConfidenceSetConfig,
    SmoothnessHybrid,
};
use crate::inference::validate_confidence_level;
use crate::types::InferenceConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensitivitySummaryMethod {
    Flci,
    Conditional,
    ConditionalFlci,
    ConditionalLeastFavorable,
}

impl SensitivitySummaryMethod {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Flci => "FLCI",
            Self::Conditional => "Conditional",
            Self::ConditionalFlci => "C-F",
            Self::ConditionalLeastFavorable => "C-LF",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensitivitySummaryRow {
    pub lb: f64,
    pub ub: f64,
    pub method: SensitivitySummaryMethod,
    pub delta: &'static str,
    pub sensitivity_name: &'static str,
    pub sensitivity_value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensitivitySummary {
    pub rows: Vec<SensitivitySummaryRow>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensitivityRestrictionKind {
    Smoothness,
    RelativeMagnitude,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PostPeriodSensitivitySummary {
    pub robust_ci: SensitivitySummary,
    pub orig_ci: HonestOriginalConfidenceSet,
    pub sensitivity_type: SensitivityRestrictionKind,
    pub post_period: i32,
}

fn smoothness_variant_label(
    bias_direction: Option<HonestBiasDirection>,
    monotonicity_direction: Option<HonestMonotonicityDirection>,
) -> Result<&'static str, String> {
    match (bias_direction, monotonicity_direction) {
        (Some(_), Some(_)) => Err(
            "smoothness wrapper supports either a sign restriction or a monotonicity restriction, not both"
                .to_string(),
        ),
        (None, None) => Ok("DeltaSD"),
        (Some(HonestBiasDirection::Positive), None) => Ok("DeltaSDPB"),
        (Some(HonestBiasDirection::Negative), None) => Ok("DeltaSDNB"),
        (None, Some(HonestMonotonicityDirection::Increasing)) => Ok("DeltaSDI"),
        (None, Some(HonestMonotonicityDirection::Decreasing)) => Ok("DeltaSDD"),
    }
}

const fn default_smoothness_method(
    bias_direction: Option<HonestBiasDirection>,
    monotonicity_direction: Option<HonestMonotonicityDirection>,
) -> SensitivitySummaryMethod {
    if bias_direction.is_some() || monotonicity_direction.is_some() {
        SensitivitySummaryMethod::ConditionalFlci
    } else {
        SensitivitySummaryMethod::Flci
    }
}

fn validate_api_inputs(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    inference: InferenceConfig,
) -> Result<(), String> {
    input.validate()?;
    if !validate_confidence_level(inference.confidence_level) {
        return Err(format!(
            "invalid confidence level {}",
            inference.confidence_level
        ));
    }
    if post_weights.len() != input.num_post_periods() {
        return Err(format!(
            "post_weights length {} does not match number of post periods {}",
            post_weights.len(),
            input.num_post_periods()
        ));
    }
    if post_weights.iter().any(|weight| !weight.is_finite()) {
        return Err("post_weights weights must be finite".to_string());
    }
    if post_weights.iter().all(|weight| weight.abs() <= 1e-12) {
        return Err("post_weights must contain at least one non-zero weight".to_string());
    }
    Ok(())
}

/// Estimate an upper end for the default smoothness grid from the observed
/// pre-treatment second differences.
///
/// # Errors
/// Returns an error if the event-study input is invalid, contains too few
/// pre-periods, or `alpha` lies outside `(0, 1)`.
pub fn estimate_smoothness_upper_bound_from_pretrends(
    input: &HonestEventStudyInput,
    alpha: f64,
) -> Result<f64, String> {
    input.validate()?;
    let num_pre = input.num_pre_periods();
    if num_pre <= 1 {
        return Err("smoothness upper-bound M requires at least two pre periods".to_string());
    }
    if !(0.0..1.0).contains(&alpha) {
        return Err(format!("alpha must lie in (0,1), got {alpha}"));
    }
    let constraint_matrix = build_smoothness_constraint_matrix(num_pre, 0, false);
    let pre_coef = &input.betahat[..num_pre];
    let pre_sigma: Vec<Vec<f64>> = input.covariance[..num_pre]
        .iter()
        .map(|row| row[..num_pre].to_vec())
        .collect();
    let mut diffs = Vec::with_capacity(constraint_matrix.len());
    mat_vec_mul_into(&constraint_matrix, pre_coef, &mut diffs);
    let sigma_diffs = sandwich_covariance(&constraint_matrix, &pre_sigma);
    let se_diffs = diag_sqrt(&sigma_diffs);
    let normal = Normal::new(0.0, 1.0)
        .map_err(|err| format!("failed to create normal distribution: {err}"))?;
    let z = normal.inverse_cdf(1.0 - alpha);
    Ok(diffs
        .iter()
        .zip(se_diffs.iter())
        .map(|(diff, se)| diff + z * se)
        .fold(0.0_f64, f64::max))
}

fn default_smoothness_m_values(input: &HonestEventStudyInput) -> Result<Vec<f64>, String> {
    let num_pre = input.num_pre_periods();
    if num_pre == 0 {
        return Err("smoothness summary requires at least one pre period".to_string());
    }
    let upper = if num_pre == 1 {
        input.covariance[0][0].max(0.0).sqrt()
    } else {
        estimate_smoothness_upper_bound_from_pretrends(input, 0.05)?
    };
    Ok(linear_grid(0.0, upper.max(0.0), 10))
}

fn default_relative_magnitude_mbar_values() -> Vec<f64> {
    linear_grid(0.0, 2.0, 10)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelativeMagnitudeApiVariant {
    Base,
    SignedBase(HonestBiasDirection),
    MonotoneBase(HonestMonotonicityDirection),
    Linear,
    SignedLinear(HonestBiasDirection),
    MonotoneLinear(HonestMonotonicityDirection),
}

impl RelativeMagnitudeApiVariant {
    fn from_wrapper_options(
        bound: HonestRelativeMagnitudeBound,
        bias_direction: Option<HonestBiasDirection>,
        monotonicity_direction: Option<HonestMonotonicityDirection>,
    ) -> Result<Self, String> {
        match (bound, bias_direction, monotonicity_direction) {
            (_, Some(_), Some(_)) => Err(
                "relative-magnitude wrapper supports either a sign restriction or a monotonicity restriction, not both"
                    .to_string(),
            ),
            (HonestRelativeMagnitudeBound::ParallelTrendsDeviation, None, None) => {
                Ok(Self::Base)
            }
            (
                HonestRelativeMagnitudeBound::ParallelTrendsDeviation,
                Some(direction),
                None,
            ) => Ok(Self::SignedBase(direction)),
            (
                HonestRelativeMagnitudeBound::ParallelTrendsDeviation,
                None,
                Some(direction),
            ) => Ok(Self::MonotoneBase(direction)),
            (HonestRelativeMagnitudeBound::LinearTrendDeviation, None, None) => {
                Ok(Self::Linear)
            }
            (HonestRelativeMagnitudeBound::LinearTrendDeviation, Some(direction), None) => {
                Ok(Self::SignedLinear(direction))
            }
            (HonestRelativeMagnitudeBound::LinearTrendDeviation, None, Some(direction)) => {
                Ok(Self::MonotoneLinear(direction))
            }
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Base => "DeltaRM",
            Self::SignedBase(HonestBiasDirection::Positive) => "DeltaRMPB",
            Self::SignedBase(HonestBiasDirection::Negative) => "DeltaRMNB",
            Self::MonotoneBase(HonestMonotonicityDirection::Increasing) => "DeltaRMI",
            Self::MonotoneBase(HonestMonotonicityDirection::Decreasing) => "DeltaRMD",
            Self::Linear => "DeltaSDRM",
            Self::SignedLinear(HonestBiasDirection::Positive) => "DeltaSDRMPB",
            Self::SignedLinear(HonestBiasDirection::Negative) => "DeltaSDRMNB",
            Self::MonotoneLinear(HonestMonotonicityDirection::Increasing) => "DeltaSDRMI",
            Self::MonotoneLinear(HonestMonotonicityDirection::Decreasing) => "DeltaSDRMD",
        }
    }

    fn compute_identified_set(
        self,
        input: &HonestEventStudyInput,
        post_weights: &[f64],
        mbar: f64,
    ) -> Result<super::HonestIdentifiedSet, String> {
        match self {
            Self::Base => compute_relative_magnitude_identified_set(input, post_weights, mbar),
            Self::SignedBase(direction) => super::compute_signed_relative_magnitude_identified_set(
                input,
                post_weights,
                mbar,
                direction,
            ),
            Self::MonotoneBase(direction) => {
                super::compute_monotone_relative_magnitude_identified_set(
                    input,
                    post_weights,
                    mbar,
                    direction,
                )
            }
            Self::Linear => super::compute_linear_trend_relative_magnitude_identified_set(
                input,
                post_weights,
                mbar,
            ),
            Self::SignedLinear(direction) => {
                super::compute_signed_linear_trend_relative_magnitude_identified_set(
                    input,
                    post_weights,
                    mbar,
                    direction,
                )
            }
            Self::MonotoneLinear(direction) => {
                super::compute_monotone_linear_trend_relative_magnitude_identified_set(
                    input,
                    post_weights,
                    mbar,
                    direction,
                )
            }
        }
    }

    fn compute_conditional_cs(
        self,
        input: &HonestEventStudyInput,
        post_weights: &[f64],
        mbar: f64,
        inference: InferenceConfig,
        config: RelativeMagnitudeConfidenceSetConfig,
    ) -> Result<HonestConditionalConfidenceSet, String> {
        match self {
            Self::Base => compute_relative_magnitude_confidence_set_with_config(
                input,
                post_weights,
                mbar,
                inference,
                config,
            ),
            Self::SignedBase(direction) => {
                compute_signed_relative_magnitude_confidence_set_with_config(
                    input,
                    post_weights,
                    mbar,
                    direction,
                    inference,
                    config,
                )
            }
            Self::MonotoneBase(direction) => {
                compute_monotone_relative_magnitude_confidence_set_with_config(
                    input,
                    post_weights,
                    mbar,
                    direction,
                    inference,
                    config,
                )
            }
            Self::Linear => compute_linear_trend_relative_magnitude_confidence_set_with_config(
                input,
                post_weights,
                mbar,
                inference,
                config,
            ),
            Self::SignedLinear(direction) => {
                compute_signed_linear_trend_relative_magnitude_confidence_set_with_config(
                    input,
                    post_weights,
                    mbar,
                    direction,
                    inference,
                    config,
                )
            }
            Self::MonotoneLinear(direction) => {
                compute_monotone_linear_trend_relative_magnitude_confidence_set_with_config(
                    input,
                    post_weights,
                    mbar,
                    direction,
                    inference,
                    config,
                )
            }
        }
    }
}

const fn collect_interval(
    conditional: &HonestConditionalConfidenceSet,
    method: SensitivitySummaryMethod,
    delta: &'static str,
    sensitivity_name: &'static str,
    sensitivity_value: f64,
) -> SensitivitySummaryRow {
    SensitivitySummaryRow {
        lb: conditional.lb,
        ub: conditional.ub,
        method,
        delta,
        sensitivity_name,
        sensitivity_value,
    }
}

/// Summarize smoothness-based sensitivity intervals over an `M` grid for one
/// post-treatment functional.
///
/// # Errors
/// Returns an error if the event-study input is invalid, the wrapper options
/// are inconsistent, or any underlying identified-set / confidence-set solve
/// fails.
pub fn summarize_smoothness_sensitivity(
    input: &HonestEventStudyInput,
    inference: InferenceConfig,
    post_weights: &[f64],
    method: Option<SensitivitySummaryMethod>,
    m_values: Option<&[f64]>,
    bias_direction: Option<HonestBiasDirection>,
    monotonicity_direction: Option<HonestMonotonicityDirection>,
) -> Result<SensitivitySummary, String> {
    validate_api_inputs(input, post_weights, inference)?;
    let delta = smoothness_variant_label(bias_direction, monotonicity_direction)?;
    let method =
        method.unwrap_or_else(|| default_smoothness_method(bias_direction, monotonicity_direction));
    let m_values = match m_values {
        Some(values) => values.to_vec(),
        None => default_smoothness_m_values(input)?,
    };
    let rows = m_values
        .into_iter()
        .map(|m| {
            if !m.is_finite() || m < 0.0 {
                return Err(format!(
                    "smoothness sensitivity requires finite non-negative M, got {m}"
                ));
            }
            let row = match method {
                SensitivitySummaryMethod::Flci => {
                    let problem = build_smoothness_flci_problem(input, post_weights, m, inference)?;
                    let flci = compute_smoothness_flci_with_config(
                        &problem,
                        SmoothnessFlciConfig::default_for_production(),
                    )?;
                    SensitivitySummaryRow {
                        lb: flci.flci.0,
                        ub: flci.flci.1,
                        method,
                        delta,
                        sensitivity_name: "M",
                        sensitivity_value: m,
                    }
                }
                SensitivitySummaryMethod::Conditional
                | SensitivitySummaryMethod::ConditionalFlci
                | SensitivitySummaryMethod::ConditionalLeastFavorable => {
                    let hybrid = match method {
                        SensitivitySummaryMethod::Conditional => SmoothnessHybrid::ArpOnly,
                        SensitivitySummaryMethod::ConditionalFlci => SmoothnessHybrid::Flci,
                        SensitivitySummaryMethod::ConditionalLeastFavorable => {
                            SmoothnessHybrid::LeastFavorable
                        }
                        SensitivitySummaryMethod::Flci => unreachable!(),
                    };
                    let config = SmoothnessConfidenceSetConfig {
                        hybrid,
                        ..SmoothnessConfidenceSetConfig::from_inference(inference)
                    };
                    let conditional = match (bias_direction, monotonicity_direction) {
                        (None, None) => compute_smoothness_confidence_set_with_config(
                            input,
                            post_weights,
                            m,
                            inference,
                            config,
                        )?,
                        (Some(direction), None) => {
                            compute_signed_smoothness_confidence_set_with_config(
                                input,
                                post_weights,
                                m,
                                direction,
                                inference,
                                config,
                            )?
                        }
                        (None, Some(direction)) => {
                            compute_monotone_smoothness_confidence_set_with_config(
                                input,
                                post_weights,
                                m,
                                direction,
                                inference,
                                config,
                            )?
                        }
                        (Some(_), Some(_)) => unreachable!("validated above"),
                    };
                    collect_interval(&conditional, method, delta, "M", m)
                }
            };
            Ok(row)
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(SensitivitySummary { rows })
}

/// Summarize relative-magnitude sensitivity intervals over an `Mbar` grid for
/// one post-treatment functional.
///
/// # Errors
/// Returns an error if the event-study input is invalid, the wrapper options
/// are inconsistent, or any underlying identified-set / confidence-set solve
/// fails.
#[allow(clippy::too_many_arguments)]
pub fn summarize_relative_magnitude_sensitivity(
    input: &HonestEventStudyInput,
    inference: InferenceConfig,
    post_weights: &[f64],
    method: Option<SensitivitySummaryMethod>,
    mbar_values: Option<&[f64]>,
    bound: Option<HonestRelativeMagnitudeBound>,
    bias_direction: Option<HonestBiasDirection>,
    monotonicity_direction: Option<HonestMonotonicityDirection>,
) -> Result<SensitivitySummary, String> {
    validate_api_inputs(input, post_weights, inference)?;
    let bound = bound.unwrap_or(HonestRelativeMagnitudeBound::ParallelTrendsDeviation);
    let variant = RelativeMagnitudeApiVariant::from_wrapper_options(
        bound,
        bias_direction,
        monotonicity_direction,
    )?;
    let delta = variant.label();
    let method = method.unwrap_or(SensitivitySummaryMethod::ConditionalLeastFavorable);
    if matches!(
        method,
        SensitivitySummaryMethod::Flci | SensitivitySummaryMethod::ConditionalFlci
    ) {
        return Err(
            "relative-magnitude wrapper currently supports only Conditional and C-LF methods"
                .to_string(),
        );
    }
    let mbar_values = mbar_values.map_or_else(default_relative_magnitude_mbar_values, |values| {
        values.to_vec()
    });
    let rows = mbar_values
        .into_iter()
        .map(|mbar| {
            if !mbar.is_finite() || mbar < 0.0 {
                return Err(format!(
                    "relative-magnitude sensitivity requires finite non-negative Mbar, got {mbar}"
                ));
            }
            let config = RelativeMagnitudeConfidenceSetConfig {
                hybrid: match method {
                    SensitivitySummaryMethod::Conditional => RelativeMagnitudeHybrid::ArpOnly,
                    SensitivitySummaryMethod::ConditionalLeastFavorable => {
                        RelativeMagnitudeHybrid::LeastFavorable
                    }
                    SensitivitySummaryMethod::Flci | SensitivitySummaryMethod::ConditionalFlci => {
                        unreachable!()
                    }
                },
                ..RelativeMagnitudeConfidenceSetConfig::from_inference(inference)
            };
            let identified = variant.compute_identified_set(input, post_weights, mbar)?;
            let conditional = variant
                .compute_conditional_cs(input, post_weights, mbar, inference, config)
                .or_else(|err| {
                    if !identified.lb.is_finite() || !identified.ub.is_finite() {
                        Ok(HonestConditionalConfidenceSet {
                            lb: identified.lb,
                            ub: identified.ub,
                        })
                    } else {
                        Err(err)
                    }
                })?;
            Ok(collect_interval(&conditional, method, delta, "Mbar", mbar))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(SensitivitySummary { rows })
}

/// Summarize sensitivity results for a named post-treatment period by
/// converting it to the corresponding basis functional.
///
/// # Errors
/// Returns an error if the requested post period is absent, the input is
/// invalid, or the underlying summary routine fails.
#[allow(clippy::too_many_arguments)]
pub fn summarize_post_period_sensitivity(
    input: &HonestEventStudyInput,
    post_period: i32,
    sensitivity_type: SensitivityRestrictionKind,
    inference: InferenceConfig,
    method: Option<SensitivitySummaryMethod>,
    sensitivity_values: Option<&[f64]>,
    bound: Option<HonestRelativeMagnitudeBound>,
    bias_direction: Option<HonestBiasDirection>,
    monotonicity_direction: Option<HonestMonotonicityDirection>,
) -> Result<PostPeriodSensitivitySummary, String> {
    input.validate()?;
    let Some(post_idx) = input
        .post_periods
        .iter()
        .position(|period| *period == post_period)
    else {
        return Err(format!(
            "post period {post_period} not found in event-study input"
        ));
    };
    let post_weights = basis_post_weights(input.num_post_periods(), post_idx);
    let orig_ci = compute_original_confidence_set(input, &post_weights, inference)?;
    let robust_ci = match sensitivity_type {
        SensitivityRestrictionKind::Smoothness => summarize_smoothness_sensitivity(
            input,
            inference,
            &post_weights,
            method,
            sensitivity_values,
            bias_direction,
            monotonicity_direction,
        )?,
        SensitivityRestrictionKind::RelativeMagnitude => summarize_relative_magnitude_sensitivity(
            input,
            inference,
            &post_weights,
            method,
            sensitivity_values,
            bound,
            bias_direction,
            monotonicity_direction,
        )?,
    };
    Ok(PostPeriodSensitivitySummary {
        robust_ci,
        orig_ci,
        sensitivity_type,
        post_period,
    })
}
