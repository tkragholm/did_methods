use epistat_stats::normal;

use super::linear_algebra::{diag_sqrt, linear_grid, mat_vec_mul_into, sandwich_covariance};
use super::relative_magnitude::{
    RelativeMagnitudeFamily, compute_original_confidence_set,
    compute_relative_magnitude_family_conditional_cs_with_precomputed_sets,
    compute_relative_magnitude_family_identified_set, geometry::basis_post_weights,
    prepare_relative_magnitude_family_draws,
};
use super::smoothness::build_smoothness_constraint_matrix;
use super::smoothness::least_favorable_intervals::SmoothnessFlciConfig;
use super::solver_tally::SolverCounts;
use super::{
    HonestBiasDirection, HonestConditionalConfidenceSet, HonestEventStudyInput,
    HonestIdentifiedSet, HonestMonotonicityDirection, HonestOriginalConfidenceSet,
    HonestRelativeMagnitudeBound, LeastFavorableSimulation, RelativeMagnitudeConfidenceSetConfig,
    RelativeMagnitudeHybrid, SmoothnessConfidenceSetConfig, SmoothnessHybrid,
};
use rayon::prelude::*;

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
    /// The identified set this row's interval was built around, where the
    /// restriction family has a closed form for one.
    ///
    /// It is on the row because the caller would otherwise ask for it again.
    /// Constructing the confidence set requires the identified set -- it is what
    /// the inversion grid is centred on -- so it has already been solved by the
    /// time the row exists, and a consumer that wants to report "a loss of up to
    /// this much remains consistent with the data" beside the interval was
    /// re-running eight branch LPs to learn something this function had just
    /// thrown away. Study I's bridge did exactly that, which made three
    /// identified-set solves per row where one was needed.
    ///
    /// `None` on the smoothness path, which does not compute one here.
    pub identified: Option<HonestIdentifiedSet>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensitivitySummary {
    pub rows: Vec<SensitivitySummaryRow>,
    /// Conic programs that needed the retry with looser tolerances, and programs
    /// that ended `AlmostSolved`, while the summary was computed. See
    /// [`SolverCounts`]. A summary from `summarize_relative_magnitude_sensitivity_many`
    /// carries the counts of the whole call.
    pub solver: SolverCounts,
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
    let z = normal::quantile(1.0 - alpha);
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

    /// The internal family this wrapper variant names.
    const fn family(self) -> RelativeMagnitudeFamily {
        match self {
            Self::Base => RelativeMagnitudeFamily::Base,
            Self::SignedBase(direction) => RelativeMagnitudeFamily::SignedBase(direction),
            Self::MonotoneBase(direction) => RelativeMagnitudeFamily::MonotoneBase(direction),
            Self::Linear => RelativeMagnitudeFamily::Linear,
            Self::SignedLinear(direction) => RelativeMagnitudeFamily::SignedLinear(direction),
            Self::MonotoneLinear(direction) => RelativeMagnitudeFamily::MonotoneLinear(direction),
        }
    }
}

const fn collect_interval(
    conditional: &HonestConditionalConfidenceSet,
    method: SensitivitySummaryMethod,
    delta: &'static str,
    sensitivity_name: &'static str,
    sensitivity_value: f64,
    identified: Option<HonestIdentifiedSet>,
) -> SensitivitySummaryRow {
    SensitivitySummaryRow {
        lb: conditional.lb,
        ub: conditional.ub,
        method,
        delta,
        sensitivity_name,
        sensitivity_value,
        identified,
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
    summarize_smoothness_sensitivity_counted(
        input,
        inference,
        post_weights,
        method,
        m_values,
        bias_direction,
        monotonicity_direction,
        LeastFavorableSimulation::default(),
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

/// Same as [`summarize_smoothness_sensitivity`], with explicit least-favorable simulation settings.
///
/// # Errors
/// Returns an error under the conditions of [`summarize_smoothness_sensitivity`], or if `simulation` is
/// invalid.
#[allow(clippy::too_many_arguments)]
pub fn summarize_smoothness_sensitivity_with_simulation(
    input: &HonestEventStudyInput,
    inference: InferenceConfig,
    post_weights: &[f64],
    method: Option<SensitivitySummaryMethod>,
    m_values: Option<&[f64]>,
    bias_direction: Option<HonestBiasDirection>,
    monotonicity_direction: Option<HonestMonotonicityDirection>,
    simulation: LeastFavorableSimulation,
) -> Result<SensitivitySummary, String> {
    simulation.validate()?;
    summarize_smoothness_sensitivity_counted(
        input,
        inference,
        post_weights,
        method,
        m_values,
        bias_direction,
        monotonicity_direction,
        simulation,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::inference::sensitivity) fn summarize_smoothness_sensitivity_counted(
    input: &HonestEventStudyInput,
    inference: InferenceConfig,
    post_weights: &[f64],
    method: Option<SensitivitySummaryMethod>,
    m_values: Option<&[f64]>,
    bias_direction: Option<HonestBiasDirection>,
    monotonicity_direction: Option<HonestMonotonicityDirection>,
    simulation: LeastFavorableSimulation,
    counter: &crate::inference::sensitivity::SolverCounter,
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
                    let problem = crate::inference::sensitivity::smoothness::least_favorable_intervals::build_smoothness_flci_problem_counted(input, post_weights, m, inference, counter)?;
                    let flci = crate::inference::sensitivity::smoothness::least_favorable_intervals::compute_smoothness_flci_with_config_counted(
                        &problem,
                        SmoothnessFlciConfig::default_for_production(), counter,
                    )?;
                    SensitivitySummaryRow {
                        lb: flci.flci.0,
                        ub: flci.flci.1,
                        method,
                        delta,
                        sensitivity_name: "M",
                        sensitivity_value: m,
                        identified: None,
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
                        (None, None) => crate::inference::sensitivity::smoothness::compute_smoothness_confidence_set_with_config_counted(
                            input,
                            post_weights,
                            m,
                            inference,
                            config, simulation, counter,
                        )?,
                        (Some(direction), None) => {
                            crate::inference::sensitivity::smoothness::compute_signed_smoothness_confidence_set_with_config_counted(
                                input,
                                post_weights,
                                m,
                                direction,
                                inference,
                                config, simulation, counter,
                            )?
                        }
                        (None, Some(direction)) => {
                            crate::inference::sensitivity::smoothness::compute_monotone_smoothness_confidence_set_with_config_counted(
                                input,
                                post_weights,
                                m,
                                direction,
                                inference,
                                config, simulation, counter,
                            )?
                        }
                        (Some(_), Some(_)) => unreachable!("validated above"),
                    };
                    collect_interval(&conditional, method, delta, "M", m, None)
                }
            };
            Ok(row)
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(SensitivitySummary {
        rows,
        solver: counter.counts(),
    })
}

/// Summarize relative-magnitude sensitivity intervals over an `Mbar` grid for
/// one post-treatment functional.
///
/// One functional of [`summarize_relative_magnitude_sensitivity_many`].
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
    summarize_relative_magnitude_sensitivity_counted(
        input,
        inference,
        post_weights,
        method,
        mbar_values,
        bound,
        bias_direction,
        monotonicity_direction,
        LeastFavorableSimulation::default(),
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

/// Same as [`summarize_relative_magnitude_sensitivity`], with explicit least-favorable simulation settings.
///
/// # Errors
/// Returns an error under the conditions of [`summarize_relative_magnitude_sensitivity`], or if `simulation` is
/// invalid.
#[allow(clippy::too_many_arguments)]
pub fn summarize_relative_magnitude_sensitivity_with_simulation(
    input: &HonestEventStudyInput,
    inference: InferenceConfig,
    post_weights: &[f64],
    method: Option<SensitivitySummaryMethod>,
    mbar_values: Option<&[f64]>,
    bound: Option<HonestRelativeMagnitudeBound>,
    bias_direction: Option<HonestBiasDirection>,
    monotonicity_direction: Option<HonestMonotonicityDirection>,
    simulation: LeastFavorableSimulation,
) -> Result<SensitivitySummary, String> {
    simulation.validate()?;
    summarize_relative_magnitude_sensitivity_counted(
        input,
        inference,
        post_weights,
        method,
        mbar_values,
        bound,
        bias_direction,
        monotonicity_direction,
        simulation,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::inference::sensitivity) fn summarize_relative_magnitude_sensitivity_counted(
    input: &HonestEventStudyInput,
    inference: InferenceConfig,
    post_weights: &[f64],
    method: Option<SensitivitySummaryMethod>,
    mbar_values: Option<&[f64]>,
    bound: Option<HonestRelativeMagnitudeBound>,
    bias_direction: Option<HonestBiasDirection>,
    monotonicity_direction: Option<HonestMonotonicityDirection>,
    simulation: LeastFavorableSimulation,
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<SensitivitySummary, String> {
    let mut summaries = summarize_relative_magnitude_sensitivity_many_counted(
        input,
        inference,
        &[post_weights],
        method,
        mbar_values,
        bound,
        bias_direction,
        monotonicity_direction,
        simulation,
        counter,
    )?;
    Ok(summaries.remove(0))
}

/// Summarize relative-magnitude sensitivity intervals over an `Mbar` grid for
/// several post-treatment functionals of one event study, in the order given.
///
/// The functionals share everything that does not depend on the weights: at
/// each `Mbar`, every branch's constraint matrix, its ARP covariance, and the
/// thousand simulation draws behind its least-favorable critical value. Those
/// draws were the tenth of the surface's CPU that twenty functionals spent
/// making the same numbers twenty times. Made once here and handed to each
/// functional, they give the same critical values. The functionals themselves
/// run in parallel.
///
/// # Errors
/// Returns an error if the event-study input is invalid for any functional,
/// the wrapper options are inconsistent, or any underlying identified-set /
/// confidence-set solve fails.
#[allow(clippy::too_many_arguments)]
pub fn summarize_relative_magnitude_sensitivity_many(
    input: &HonestEventStudyInput,
    inference: InferenceConfig,
    functionals: &[&[f64]],
    method: Option<SensitivitySummaryMethod>,
    mbar_values: Option<&[f64]>,
    bound: Option<HonestRelativeMagnitudeBound>,
    bias_direction: Option<HonestBiasDirection>,
    monotonicity_direction: Option<HonestMonotonicityDirection>,
) -> Result<Vec<SensitivitySummary>, String> {
    summarize_relative_magnitude_sensitivity_many_counted(
        input,
        inference,
        functionals,
        method,
        mbar_values,
        bound,
        bias_direction,
        monotonicity_direction,
        LeastFavorableSimulation::default(),
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

/// Same as [`summarize_relative_magnitude_sensitivity_many`], with explicit least-favorable simulation settings.
///
/// # Errors
/// Returns an error under the conditions of [`summarize_relative_magnitude_sensitivity_many`], or if `simulation` is
/// invalid.
#[allow(clippy::too_many_arguments)]
pub fn summarize_relative_magnitude_sensitivity_many_with_simulation(
    input: &HonestEventStudyInput,
    inference: InferenceConfig,
    functionals: &[&[f64]],
    method: Option<SensitivitySummaryMethod>,
    mbar_values: Option<&[f64]>,
    bound: Option<HonestRelativeMagnitudeBound>,
    bias_direction: Option<HonestBiasDirection>,
    monotonicity_direction: Option<HonestMonotonicityDirection>,
    simulation: LeastFavorableSimulation,
) -> Result<Vec<SensitivitySummary>, String> {
    simulation.validate()?;
    summarize_relative_magnitude_sensitivity_many_counted(
        input,
        inference,
        functionals,
        method,
        mbar_values,
        bound,
        bias_direction,
        monotonicity_direction,
        simulation,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::inference::sensitivity) fn summarize_relative_magnitude_sensitivity_many_counted(
    input: &HonestEventStudyInput,
    inference: InferenceConfig,
    functionals: &[&[f64]],
    method: Option<SensitivitySummaryMethod>,
    mbar_values: Option<&[f64]>,
    bound: Option<HonestRelativeMagnitudeBound>,
    bias_direction: Option<HonestBiasDirection>,
    monotonicity_direction: Option<HonestMonotonicityDirection>,
    simulation: LeastFavorableSimulation,
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<Vec<SensitivitySummary>, String> {
    for post_weights in functionals {
        validate_api_inputs(input, post_weights, inference)?;
    }
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
        least_favorable_draws: simulation.draws,
        least_favorable_seed: simulation.seed,
        ..RelativeMagnitudeConfidenceSetConfig::from_inference(inference)
    };
    config.validate(inference)?;
    let family = variant.family();
    // None of these depends on `Mbar`, so none belongs inside the loop.
    let originals = functionals
        .iter()
        .map(|post_weights| compute_original_confidence_set(input, post_weights, inference))
        .collect::<Result<Vec<_>, String>>()?;
    let mut rows: Vec<Vec<SensitivitySummaryRow>> = functionals
        .iter()
        .map(|_| Vec::with_capacity(mbar_values.len()))
        .collect();
    for mbar in mbar_values {
        if !mbar.is_finite() || mbar < 0.0 {
            return Err(format!(
                "relative-magnitude sensitivity requires finite non-negative Mbar, got {mbar}"
            ));
        }
        let shared_draws = match config.hybrid {
            RelativeMagnitudeHybrid::LeastFavorable => {
                Some(prepare_relative_magnitude_family_draws(
                    input,
                    mbar,
                    family,
                    config.least_favorable_draws,
                    config.least_favorable_seed,
                )?)
            }
            RelativeMagnitudeHybrid::ArpOnly => None,
        };
        let at_this_mbar = functionals
            .par_iter()
            .zip(originals.par_iter())
            .map(|(post_weights, original)| {
                {
                    // Solved once and threaded through. The confidence set is built
                    // around the identified set, so computing it here and handing
                    // it on is not an optimisation of the wrapper so much as the
                    // removal of a second solve of the same LPs one call further
                    // down.
                    let identified = compute_relative_magnitude_family_identified_set(
                        input,
                        post_weights,
                        mbar,
                        family,
                        counter,
                    )?;
                    let conditional =
                        compute_relative_magnitude_family_conditional_cs_with_precomputed_sets(
                            input,
                            post_weights,
                            mbar,
                            inference,
                            config,
                            original,
                            &identified,
                            family,
                            shared_draws.as_deref(),
                            counter,
                        )
                        .or_else(|err| {
                            if identified.lb.is_finite() && identified.ub.is_finite() {
                                Err(err)
                            } else {
                                Ok(HonestConditionalConfidenceSet {
                                    lb: identified.lb,
                                    ub: identified.ub,
                                })
                            }
                        })?;
                    Ok(collect_interval(
                        &conditional,
                        method,
                        delta,
                        "Mbar",
                        mbar,
                        Some(identified),
                    ))
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        for (target, row) in rows.iter_mut().zip(at_this_mbar) {
            target.push(row);
        }
    }
    let solver_counts = counter.counts();
    Ok(rows
        .into_iter()
        .map(|rows| SensitivitySummary {
            rows,
            solver: solver_counts,
        })
        .collect())
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
    summarize_post_period_sensitivity_counted(
        input,
        post_period,
        sensitivity_type,
        inference,
        method,
        sensitivity_values,
        bound,
        bias_direction,
        monotonicity_direction,
        LeastFavorableSimulation::default(),
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

/// Same as [`summarize_post_period_sensitivity`], with explicit least-favorable simulation settings.
///
/// # Errors
/// Returns an error under the conditions of [`summarize_post_period_sensitivity`], or if `simulation` is
/// invalid.
#[allow(clippy::too_many_arguments)]
pub fn summarize_post_period_sensitivity_with_simulation(
    input: &HonestEventStudyInput,
    post_period: i32,
    sensitivity_type: SensitivityRestrictionKind,
    inference: InferenceConfig,
    method: Option<SensitivitySummaryMethod>,
    sensitivity_values: Option<&[f64]>,
    bound: Option<HonestRelativeMagnitudeBound>,
    bias_direction: Option<HonestBiasDirection>,
    monotonicity_direction: Option<HonestMonotonicityDirection>,
    simulation: LeastFavorableSimulation,
) -> Result<PostPeriodSensitivitySummary, String> {
    simulation.validate()?;
    summarize_post_period_sensitivity_counted(
        input,
        post_period,
        sensitivity_type,
        inference,
        method,
        sensitivity_values,
        bound,
        bias_direction,
        monotonicity_direction,
        simulation,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::inference::sensitivity) fn summarize_post_period_sensitivity_counted(
    input: &HonestEventStudyInput,
    post_period: i32,
    sensitivity_type: SensitivityRestrictionKind,
    inference: InferenceConfig,
    method: Option<SensitivitySummaryMethod>,
    sensitivity_values: Option<&[f64]>,
    bound: Option<HonestRelativeMagnitudeBound>,
    bias_direction: Option<HonestBiasDirection>,
    monotonicity_direction: Option<HonestMonotonicityDirection>,
    simulation: LeastFavorableSimulation,
    counter: &crate::inference::sensitivity::SolverCounter,
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
        SensitivityRestrictionKind::Smoothness => summarize_smoothness_sensitivity_counted(
            input,
            inference,
            &post_weights,
            method,
            sensitivity_values,
            bias_direction,
            monotonicity_direction,
            simulation,
            counter,
        )?,
        SensitivityRestrictionKind::RelativeMagnitude => {
            summarize_relative_magnitude_sensitivity_counted(
                input,
                inference,
                &post_weights,
                method,
                sensitivity_values,
                bound,
                bias_direction,
                monotonicity_direction,
                simulation,
                counter,
            )?
        }
    };
    Ok(PostPeriodSensitivitySummary {
        robust_ci,
        orig_ci,
        sensitivity_type,
        post_period,
    })
}
