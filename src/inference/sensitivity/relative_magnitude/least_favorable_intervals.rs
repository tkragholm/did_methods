//! Typed problem surface for `DeltaRM` least-favorable confidence intervals.
//!
//! This module exposes the current least-favorable `DeltaRM` interval as a
//! typed FLCI result. The implementation reuses the validated scalar
//! `DeltaRM` conditional confidence-set path already present in this crate:
//!
//! - the original parallel-trends confidence set for `l' τ_post`
//! - the exact `DeltaRM(Mbar)` identified set for the same linear functional
//! - the least-favorable conditional confidence interval for the same target
//!
//! This is not yet the full multi-period FLCI optimizer surface of the R
//! `HonestDiD` package. It is the current scalar/linear-functional
//! least-favorable interval for `DeltaRM`, expressed as a first-class Rust API
//! instead of an internal conditional-CS detail.
//!
//! References:
//! - Rambachan, A. and Roth, J. (2023), "A More Credible Approach to Parallel
//!   Trends", especially the least-favorable / FLCI discussion
//! - `HonestDiD` R package `flci.R`

use crate::types::InferenceConfig;
use faer::Mat;
use rayon::prelude::*;

use super::super::linear_algebra::{
    cholesky_lower, critical_value_from_pointwise_confidence, dot,
    pointwise_confidence_level_from_critical, post_covariance_block,
    simulated_lower_cholesky_maxima_batched, simulated_lower_cholesky_maxima_scalar,
    simulation_rank,
};
use super::super::{
    HonestConditionalConfidenceSet, HonestEventStudyInput, HonestIdentifiedSet,
    HonestJointPathConfig, HonestJointPathMethod, HonestOriginalConfidenceSet,
    RelativeMagnitudeConfidenceSetConfig, RelativeMagnitudeMultiFlciPoint,
    RelativeMagnitudeMultiFlciResult,
};
use super::conditional_confidence_set::{
    RelativeMagnitudePreparedFunctionalBranch, RelativeMagnitudePreparedInputBranch,
    compute_relative_magnitude_confidence_set_with_prepared_functional_branches,
    prepare_relative_magnitude_functional_branches, prepare_relative_magnitude_input_branches,
};
use super::geometry::{RelativeMagnitudePreparedBranch, prepare_relative_magnitude_branches};
use super::{
    compute_original_confidence_set, compute_relative_magnitude_confidence_set_with_config,
    compute_relative_magnitude_identified_set,
};
use crate::inference::validate_confidence_level;
use crate::util::usize_to_f64;

/// Typed `DeltaRM` FLCI problem for a post-treatment linear functional.
///
/// The target is the scalar functional
///
/// ```text
/// θ = l' τ_post
/// ```
///
/// under the relative-magnitude restriction `ΔRM(Mbar)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativeMagnitudeFlciProblem {
    /// Typed event-study input.
    pub input: HonestEventStudyInput,
    /// Post-treatment linear functional weights.
    pub post_weights: Vec<f64>,
    /// Relative-magnitude bound `Mbar`.
    pub mbar: f64,
    /// Inference configuration used for the original confidence set.
    pub inference: InferenceConfig,
    /// Original parallel-trends confidence set for `l' τ_post`.
    pub original: HonestOriginalConfidenceSet,
    /// Exact `DeltaRM(Mbar)` identified set for `l' τ_post`.
    pub identified_set: HonestIdentifiedSet,
}

/// Least-favorable `DeltaRM` interval result for a post-treatment linear
/// functional.
///
/// The `flci` field is the current least-favorable `DeltaRM` conditional
/// confidence interval for `l' τ_post`, exposed as a typed result alongside
/// the original and identified-set quantities.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativeMagnitudeFlciResult {
    /// Least-favorable confidence interval.
    pub flci: (f64, f64),
    /// Original parallel-trends confidence interval.
    pub original_ci: (f64, f64),
    /// Exact `DeltaRM(Mbar)` identified set.
    pub identified_set: (f64, f64),
}

/// Typed `DeltaRM` simultaneous FLCI problem over multiple post-treatment
/// linear functionals.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativeMagnitudeMultiFlciProblem {
    /// Typed event-study input.
    pub input: HonestEventStudyInput,
    /// Functional weights in post-period order.
    pub post_weight_sets: Vec<Vec<f64>>,
    /// Relative-magnitude bound `Mbar`.
    pub mbar: f64,
    /// Family-wise inference configuration.
    pub inference: InferenceConfig,
    /// Original parallel-trends confidence sets for each functional.
    pub originals: Vec<HonestOriginalConfidenceSet>,
    /// Exact identified sets for each functional.
    pub identified_sets: Vec<HonestIdentifiedSet>,
}

/// Build a typed `DeltaRM` FLCI problem for a post-treatment linear
/// functional.
///
/// This packages the exact quantities the future least-favorable optimizer
/// needs, while reusing the validated original-CS and identified-set paths.
///
/// # Errors
/// Returns an error if the event-study input is inconsistent, `post_weights` is
/// incompatible with the post-period support, or the underlying original-CS /
/// identified-set calculations fail.
pub fn build_relative_magnitude_flci_problem(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
) -> Result<RelativeMagnitudeFlciProblem, String> {
    let original = compute_original_confidence_set(input, post_weights, inference)?;
    let identified_set = compute_relative_magnitude_identified_set(input, post_weights, mbar)?;
    Ok(RelativeMagnitudeFlciProblem {
        input: input.clone(),
        post_weights: post_weights.to_vec(),
        mbar,
        inference,
        original,
        identified_set,
    })
}

/// Solve the current least-favorable `DeltaRM` interval for a typed problem
/// using the default least-favorable hybrid configuration implied by
/// `problem.inference`.
///
/// # Errors
/// Returns an error if the underlying `DeltaRM` conditional interval fails.
pub fn compute_relative_magnitude_flci(
    problem: &RelativeMagnitudeFlciProblem,
) -> Result<RelativeMagnitudeFlciResult, String> {
    compute_relative_magnitude_flci_with_config(
        problem,
        RelativeMagnitudeConfidenceSetConfig::from_inference(problem.inference),
    )
}

/// Solve the current least-favorable `DeltaRM` interval for a typed problem
/// with an explicit hybrid configuration.
///
/// This surfaces the same scalar/linear-functional least-favorable interval as
/// the lower-level `DeltaRM` conditional confidence-set path, but packages the
/// result with the original confidence interval and identified set.
///
/// # Errors
/// Returns an error if the underlying `DeltaRM` conditional interval fails.
pub fn compute_relative_magnitude_flci_with_config(
    problem: &RelativeMagnitudeFlciProblem,
    config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<RelativeMagnitudeFlciResult, String> {
    let conditional: HonestConditionalConfidenceSet =
        compute_relative_magnitude_confidence_set_with_config(
            &problem.input,
            &problem.post_weights,
            problem.mbar,
            problem.inference,
            config,
        )?;
    Ok(RelativeMagnitudeFlciResult {
        flci: (conditional.lb, conditional.ub),
        original_ci: problem.original.ci,
        identified_set: (problem.identified_set.lb, problem.identified_set.ub),
    })
}

/// Build a typed simultaneous `DeltaRM` FLCI problem over multiple functionals.
///
/// # Errors
/// Returns an error if any functional is invalid or if the underlying original
/// and identified-set builders fail.
pub fn build_relative_magnitude_multi_flci_problem(
    input: &HonestEventStudyInput,
    post_weight_sets: &[Vec<f64>],
    mbar: f64,
    inference: InferenceConfig,
) -> Result<RelativeMagnitudeMultiFlciProblem, String> {
    if post_weight_sets.is_empty() {
        return Err("multi FLCI requires at least one functional".to_string());
    }
    let mut originals = Vec::with_capacity(post_weight_sets.len());
    let mut identified_sets = Vec::with_capacity(post_weight_sets.len());
    for post_weights in post_weight_sets {
        originals.push(compute_original_confidence_set(
            input,
            post_weights,
            inference,
        )?);
        identified_sets.push(compute_relative_magnitude_identified_set(
            input,
            post_weights,
            mbar,
        )?);
    }
    Ok(RelativeMagnitudeMultiFlciProblem {
        input: input.clone(),
        post_weight_sets: post_weight_sets.to_vec(),
        mbar,
        inference,
        originals,
        identified_sets,
    })
}

/// Build a typed simultaneous `DeltaRM` FLCI problem from precomputed
/// original confidence sets and identified sets.
///
/// # Errors
/// Returns an error if the inputs are empty or if the supplied precomputed
/// vectors do not match `post_weight_sets`.
pub fn build_relative_magnitude_multi_flci_problem_with_precomputed_sets(
    input: &HonestEventStudyInput,
    post_weight_sets: &[Vec<f64>],
    mbar: f64,
    inference: InferenceConfig,
    originals: Vec<HonestOriginalConfidenceSet>,
    identified_sets: Vec<HonestIdentifiedSet>,
) -> Result<RelativeMagnitudeMultiFlciProblem, String> {
    if post_weight_sets.is_empty() {
        return Err("multi FLCI requires at least one functional".to_string());
    }
    if originals.len() != post_weight_sets.len() {
        return Err(format!(
            "original confidence-set count {} does not match functional count {}",
            originals.len(),
            post_weight_sets.len()
        ));
    }
    if identified_sets.len() != post_weight_sets.len() {
        return Err(format!(
            "identified-set count {} does not match functional count {}",
            identified_sets.len(),
            post_weight_sets.len()
        ));
    }
    Ok(RelativeMagnitudeMultiFlciProblem {
        input: input.clone(),
        post_weight_sets: post_weight_sets.to_vec(),
        mbar,
        inference,
        originals,
        identified_sets,
    })
}

fn functional_correlation_matrix(
    input: &HonestEventStudyInput,
    post_weight_sets: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    let sigma_post = post_covariance_block(
        &input.covariance,
        input.num_pre_periods(),
        input.num_post_periods(),
    );
    let projected = post_weight_sets
        .iter()
        .map(|post_weights| {
            sigma_post
                .iter()
                .map(|row| dot(post_weights, row))
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<_>>();
    let variances = projected
        .iter()
        .zip(post_weight_sets.iter())
        .map(|(sigma_l, post_weights)| dot(post_weights, sigma_l).max(0.0))
        .collect::<Vec<_>>();
    if variances.iter().any(|variance| *variance <= 1e-14) {
        return Err(
            "one or more functionals have near-zero variance under post covariance".to_string(),
        );
    }
    let stddevs = variances
        .iter()
        .map(|value| value.sqrt())
        .collect::<Vec<_>>();
    let n = post_weight_sets.len();
    let mut corr = vec![vec![0.0; n]; n];
    for i in 0..n {
        corr[i][i] = 1.0;
        for j in (i + 1)..n {
            let cov_ij = dot(&post_weight_sets[i], &projected[j]);
            let c = (cov_ij / (stddevs[i] * stddevs[j])).clamp(-1.0, 1.0);
            corr[i][j] = c;
            corr[j][i] = c;
        }
    }
    Ok(corr)
}

fn simulated_joint_pointwise_confidence_level(
    input: &HonestEventStudyInput,
    confidence_level: f64,
    post_weight_sets: &[Vec<f64>],
    simulation_draws: usize,
    simulation_seed: u64,
) -> Result<f64, String> {
    if simulation_draws == 0 {
        return Err("multi FLCI simulation_draws must be positive".to_string());
    }
    let corr = functional_correlation_matrix(input, post_weight_sets)?;
    let chol = cholesky_lower(&corr)?;
    let mut maxima =
        simulated_lower_cholesky_maxima_batched(&chol, simulation_draws, simulation_seed);
    let n = maxima.len();
    let rank = simulation_rank(n, confidence_level);
    maxima.select_nth_unstable_by(rank.min(n - 1), f64::total_cmp);
    let critical = maxima[rank.min(n - 1)];
    Ok(critical)
}

#[doc(hidden)]
pub fn benchmark_multi_flci_maxima(
    chol: &[Vec<f64>],
    simulation_draws: usize,
    simulation_seed: u64,
) -> Vec<f64> {
    let lower = Mat::from_fn(chol.len(), chol.len(), |row, col| chol[row][col]);
    simulated_lower_cholesky_maxima_batched(&lower, simulation_draws, simulation_seed)
}

#[doc(hidden)]
pub fn benchmark_multi_flci_maxima_scalar(
    chol: &[Vec<f64>],
    simulation_draws: usize,
    simulation_seed: u64,
) -> Vec<f64> {
    let lower = Mat::from_fn(chol.len(), chol.len(), |row, col| chol[row][col]);
    simulated_lower_cholesky_maxima_scalar(&lower, simulation_draws, simulation_seed)
}

fn joint_pointwise_confidence_level(
    problem: &RelativeMagnitudeMultiFlciProblem,
    joint_config: HonestJointPathConfig,
) -> Result<(f64, f64), String> {
    let n = problem.post_weight_sets.len();
    if n == 1 {
        return Ok((
            problem.inference.confidence_level,
            critical_value_from_pointwise_confidence(problem.inference.confidence_level)?,
        ));
    }
    let alpha = 1.0 - problem.inference.confidence_level;
    match joint_config.method {
        HonestJointPathMethod::Bonferroni => {
            let pointwise = 1.0 - alpha / usize_to_f64(n);
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
                &problem.input,
                problem.inference.confidence_level,
                &problem.post_weight_sets,
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

/// Compute simultaneous `DeltaRM` FLCIs over a functional set using default
/// least-favorable and production joint calibration settings.
///
/// # Errors
/// Returns an error if joint calibration or any scalar FLCI solve fails.
pub fn compute_relative_magnitude_multi_flci(
    problem: &RelativeMagnitudeMultiFlciProblem,
) -> Result<RelativeMagnitudeMultiFlciResult, String> {
    compute_relative_magnitude_multi_flci_with_config(
        problem,
        RelativeMagnitudeConfidenceSetConfig::from_inference(problem.inference),
        HonestJointPathConfig::default_for_production(),
    )
}

/// Compute simultaneous `DeltaRM` FLCIs over a functional set with explicit
/// scalar hybrid and joint-calibration configurations.
///
/// # Errors
/// Returns an error if joint calibration or any scalar FLCI solve fails.
pub fn compute_relative_magnitude_multi_flci_with_config(
    problem: &RelativeMagnitudeMultiFlciProblem,
    config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
) -> Result<RelativeMagnitudeMultiFlciResult, String> {
    let prepared_branches = prepare_relative_magnitude_branches(
        problem.input.num_pre_periods(),
        problem.input.num_post_periods(),
        problem.mbar,
    )?;
    let prepared_input_branches =
        prepare_relative_magnitude_input_branches(&problem.input, &prepared_branches);
    compute_relative_magnitude_multi_flci_with_prepared_branches(
        problem,
        config,
        joint_config,
        &prepared_branches,
        &prepared_input_branches,
    )
}

pub(in crate::inference::sensitivity) fn compute_relative_magnitude_multi_flci_with_prepared_branches(
    problem: &RelativeMagnitudeMultiFlciProblem,
    config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
) -> Result<RelativeMagnitudeMultiFlciResult, String> {
    let prepared_functional_branch_sets = problem
        .post_weight_sets
        .iter()
        .map(|post_weights| {
            prepare_relative_magnitude_functional_branches(
                post_weights,
                prepared_branches,
                prepared_input_branches,
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    compute_relative_magnitude_multi_flci_with_prepared_functional_branches(
        problem,
        config,
        joint_config,
        prepared_branches,
        prepared_input_branches,
        &prepared_functional_branch_sets,
    )
}

pub(in crate::inference::sensitivity) fn compute_relative_magnitude_multi_flci_with_prepared_functional_branches(
    problem: &RelativeMagnitudeMultiFlciProblem,
    config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
    prepared_functional_branch_sets: &[Vec<RelativeMagnitudePreparedFunctionalBranch>],
) -> Result<RelativeMagnitudeMultiFlciResult, String> {
    let (pointwise_confidence_level, calibrated_max_t_critical_value) =
        joint_pointwise_confidence_level(problem, joint_config)?;
    let pointwise_inference = InferenceConfig::new(pointwise_confidence_level);
    let pointwise_default =
        RelativeMagnitudeConfidenceSetConfig::from_inference(pointwise_inference);
    // Intentional override: `hybrid_kappa` is re-derived from the
    // pointwise-adjusted confidence level so first-stage size tracks the
    // simultaneous calibration.
    let pointwise_config = RelativeMagnitudeConfidenceSetConfig {
        hybrid: config.hybrid,
        hybrid_kappa: pointwise_default.hybrid_kappa,
    };
    if prepared_functional_branch_sets.len() != problem.post_weight_sets.len() {
        return Err(format!(
            "prepared functional set count {} does not match functional count {}",
            prepared_functional_branch_sets.len(),
            problem.post_weight_sets.len()
        ));
    }

    // A single functional has nothing to be simultaneous about, so this must
    // return exactly what the scalar FLCI returns -- and before this it did not.
    //
    // Both paths end up choosing a grid point from a 1000-point grid spanning the
    // identified set padded by 20 standard errors, so the answer is quantised to
    // roughly 1e-3 on this data. The prepared-branch path and the direct path
    // reached that grid slightly differently and landed about two steps apart:
    // lower bounds agreed to 12 decimals while upper bounds differed by 1.8e-3.
    // Quantised disagreement cannot be closed by tightening anything, only by
    // having one path stop re-deriving what the other already computes.
    //
    // `joint_pointwise_confidence_level` already special-cases `n == 1` and
    // returns the nominal level with no calibration, so delegating here changes
    // no inference: the returned `pointwise_confidence_level` and critical value
    // are the ones this function would have reported anyway.
    if problem.post_weight_sets.len() == 1 {
        let post_weights = &problem.post_weight_sets[0];
        let original = &problem.originals[0];
        let identified_set = &problem.identified_sets[0];
        let conditional = compute_relative_magnitude_confidence_set_with_config(
            &problem.input,
            post_weights,
            problem.mbar,
            pointwise_inference,
            pointwise_config,
        )?;
        return Ok(RelativeMagnitudeMultiFlciResult {
            confidence_level: problem.inference.confidence_level,
            pointwise_confidence_level,
            calibrated_max_t_critical_value,
            method: joint_config.method,
            points: vec![RelativeMagnitudeMultiFlciPoint {
                post_weights: post_weights.clone(),
                flci: (conditional.lb, conditional.ub),
                original_ci: original.ci,
                identified_set: (identified_set.lb, identified_set.ub),
                null_value: 0.0,
                robustly_significant: conditional.lb > 0.0 || conditional.ub < 0.0,
            }],
        });
    }

    let points = problem
        .post_weight_sets
        .par_iter()
        .zip(problem.originals.par_iter())
        .zip(problem.identified_sets.par_iter())
        .zip(prepared_functional_branch_sets.par_iter())
        .map(
            |(((post_weights, original), identified_set), prepared_functional_branches)| {
                let conditional =
                    compute_relative_magnitude_confidence_set_with_prepared_functional_branches(
                        &problem.input,
                        post_weights,
                        pointwise_inference,
                        pointwise_config,
                        original,
                        identified_set,
                        prepared_branches,
                        prepared_input_branches,
                        prepared_functional_branches,
                    )?;
                Ok(RelativeMagnitudeMultiFlciPoint {
                    post_weights: post_weights.clone(),
                    flci: (conditional.lb, conditional.ub),
                    original_ci: original.ci,
                    identified_set: (identified_set.lb, identified_set.ub),
                    null_value: 0.0,
                    robustly_significant: conditional.lb > 0.0 || conditional.ub < 0.0,
                })
            },
        )
        .collect::<Result<Vec<_>, String>>()?;

    Ok(RelativeMagnitudeMultiFlciResult {
        confidence_level: problem.inference.confidence_level,
        pointwise_confidence_level,
        calibrated_max_t_critical_value,
        method: joint_config.method,
        points,
    })
}
