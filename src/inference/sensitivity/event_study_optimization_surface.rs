use argmin::core::{CostFunction, Error as ArgminError, Executor, IterState, State};
use argmin::solver::neldermead::NelderMead;
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;
use tracing::debug;

use super::super::super::linear_algebra::{draw_standard_normal_vec, post_covariance_block};
use super::super::super::{
    HonestDirectionalRegion, HonestDirectionalRegionDiagnostics, HonestEventStudyInput,
    HonestJointPathConfig, HonestOptimizationSurfaceAdaptiveRunConfig,
    HonestOptimizationSurfaceConfig, HonestSensitivity, RelativeMagnitudeConfidenceSetConfig,
};
use super::directional_region::{
    assess_honest_event_study_directional_region_with_config, calibrate_directional_region,
};
use crate::types::InferenceConfig;

const DIRECTION_DEDUP_SCALE: f64 = 1e10;

const fn sensitivity_label(sensitivity: HonestSensitivity) -> &'static str {
    match sensitivity {
        HonestSensitivity::RelativeMagnitude(_) => "relative_magnitude",
        HonestSensitivity::Smoothness(_) => "smoothness",
    }
}

fn emit_optimization_surface_timing(
    step: &str,
    sensitivity: HonestSensitivity,
    directions: usize,
    random_directions: usize,
    iteration: usize,
    duration_ms: u128,
) {
    debug!(
        target: "did_profile",
        scope = "optimization_surface",
        step,
        sensitivity_kind = sensitivity_label(sensitivity),
        directions,
        random_directions,
        iteration,
        duration_ms,
        "optimization surface timing"
    );
}

struct OptimizationSurfaceDirections {
    directions: Vec<Vec<f64>>,
    random_direction_count: usize,
}

fn projected_variance(direction: &[f64], sigma_post: &[Vec<f64>]) -> f64 {
    sigma_post
        .iter()
        .enumerate()
        .map(|(row_idx, row)| {
            direction[row_idx]
                * row
                    .iter()
                    .enumerate()
                    .map(|(col_idx, value)| value * direction[col_idx])
                    .sum::<f64>()
        })
        .sum::<f64>()
        .max(0.0)
}

fn canonicalize_unit_direction(direction: &[f64]) -> Option<Vec<f64>> {
    let norm2 = direction.iter().map(|value| value * value).sum::<f64>();
    if !norm2.is_finite() || norm2 <= 1e-14 {
        return None;
    }
    let norm = norm2.sqrt();
    let mut unit = direction
        .iter()
        .map(|value| value / norm)
        .collect::<Vec<_>>();
    if let Some(first_nonzero) = unit.iter().find(|value| value.abs() > 1e-12)
        && *first_nonzero < 0.0
    {
        for value in &mut unit {
            *value = -*value;
        }
    }
    Some(unit)
}

struct DirectionVarianceObjective<'a> {
    sigma_post: &'a [Vec<f64>],
}

impl CostFunction for DirectionVarianceObjective<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, ArgminError> {
        let Some(direction) = canonicalize_unit_direction(param) else {
            return Ok(1e6);
        };
        Ok(-projected_variance(&direction, self.sigma_post))
    }
}

fn build_nelder_mead_simplex(seed: &[f64]) -> Vec<Vec<f64>> {
    let dim = seed.len();
    let mut simplex = Vec::with_capacity(dim + 1);
    simplex.push(seed.to_vec());
    for axis in 0..dim {
        let mut vertex = seed.to_vec();
        vertex[axis] += 0.15;
        simplex.push(vertex);
    }
    simplex
}

fn refine_direction_with_argmin(
    seed_direction: &[f64],
    sigma_post: &[Vec<f64>],
) -> Result<Option<Vec<f64>>, String> {
    let seed = canonicalize_unit_direction(seed_direction)
        .ok_or_else(|| "seed direction is degenerate and cannot be refined".to_string())?;
    if seed.len() <= 1 {
        return Ok(Some(seed));
    }
    let simplex = build_nelder_mead_simplex(&seed);
    let solver = NelderMead::new(simplex)
        .with_sd_tolerance(1e-5)
        .map_err(|err| format!("failed to configure Nelder-Mead tolerance: {err}"))?;
    let problem = DirectionVarianceObjective { sigma_post };
    let result = Executor::new(problem, solver)
        .configure(|state: IterState<Vec<f64>, (), (), (), (), f64>| state.max_iters(35))
        .run()
        .map_err(|err| format!("Nelder-Mead refinement failed: {err}"))?;
    Ok(result
        .state()
        .get_best_param()
        .and_then(|vector| canonicalize_unit_direction(vector)))
}

fn top_ranked_indices_by_variance(
    directions: &[Vec<f64>],
    sigma_post: &[Vec<f64>],
    max_count: usize,
) -> Vec<(usize, f64)> {
    let mut ranked = directions
        .iter()
        .enumerate()
        .map(|(index, direction)| (index, projected_variance(direction, sigma_post)))
        .collect::<Vec<_>>();
    let keep = ranked.len().min(max_count);
    if keep > 0 && keep < ranked.len() {
        let cutoff = keep - 1;
        ranked.select_nth_unstable_by(cutoff, |left, right| right.1.total_cmp(&left.1));
        ranked.truncate(keep);
    }
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    ranked
}

fn refine_optimization_surface_directions_with_argmin(
    input: &HonestEventStudyInput,
    directions: &mut [Vec<f64>],
    warm_starts_by_rank: Option<&[Vec<f64>]>,
) -> Result<Vec<Vec<f64>>, String> {
    if directions.is_empty() {
        return Ok(Vec::new());
    }
    let sigma_post = post_covariance_block(
        &input.covariance,
        input.num_pre_periods(),
        input.num_post_periods(),
    );
    let ranked = top_ranked_indices_by_variance(directions, &sigma_post, 8);
    let refinements = ranked.len();
    let selected = ranked
        .into_iter()
        .take(refinements)
        .enumerate()
        .map(|(rank, (index, _))| {
            let seed = warm_starts_by_rank
                .and_then(|warm| warm.get(rank))
                .filter(|candidate| candidate.len() == directions[index].len())
                .cloned()
                .unwrap_or_else(|| directions[index].clone());
            (rank, index, seed)
        })
        .collect::<Vec<_>>();

    let mut refined = selected
        .par_iter()
        .map(|(rank, index, seed)| {
            let candidate = refine_direction_with_argmin(seed, &sigma_post)?
                .or_else(|| canonicalize_unit_direction(seed))
                .ok_or_else(|| "refined direction is degenerate".to_string())?;
            Ok((*rank, *index, candidate))
        })
        .collect::<Result<Vec<_>, String>>()?;
    refined.sort_by_key(|(rank, _, _)| *rank);

    for (_, index, candidate) in &refined {
        directions[*index].clone_from(candidate);
    }

    Ok(refined
        .into_iter()
        .map(|(_, _, candidate)| candidate)
        .collect())
}

fn generate_optimization_surface_directions(
    num_post: usize,
    config: HonestOptimizationSurfaceConfig,
) -> Result<OptimizationSurfaceDirections, String> {
    if num_post == 0 {
        return Err("optimization surface region requires at least one post period".to_string());
    }
    if !config.include_basis
        && !config.include_pairwise_contrasts
        && config.random_unit_directions == 0
    {
        return Err("optimization surface direction set is empty; enable basis, pairwise, or random directions".to_string());
    }

    let mut directions = Vec::new();
    let mut is_random = Vec::new();

    if config.include_basis {
        for idx in 0..num_post {
            let mut direction = vec![0.0; num_post];
            direction[idx] = 1.0;
            directions.push(direction);
            is_random.push(false);
        }
    }

    if config.include_pairwise_contrasts && num_post >= 2 {
        for left in 0..num_post {
            for right in (left + 1)..num_post {
                let mut direction = vec![0.0; num_post];
                direction[left] = 1.0;
                direction[right] = -1.0;
                if let Some(unit) = canonicalize_unit_direction(&direction) {
                    directions.push(unit);
                    is_random.push(false);
                }
            }
        }
    }

    if config.random_unit_directions > 0 {
        let mut rng = StdRng::seed_from_u64(config.random_seed);
        for _ in 0..config.random_unit_directions {
            let raw = draw_standard_normal_vec(&mut rng, num_post);
            if let Some(unit) = canonicalize_unit_direction(&raw) {
                directions.push(unit);
                is_random.push(true);
            }
        }
    }

    let random_direction_count = deduplicate_unit_directions(&mut directions, &mut is_random);
    Ok(OptimizationSurfaceDirections {
        directions,
        random_direction_count,
    })
}

fn deduplicate_unit_directions(directions: &mut Vec<Vec<f64>>, is_random: &mut Vec<bool>) -> usize {
    debug_assert_eq!(directions.len(), is_random.len());
    let mut seen = HashSet::<Vec<i64>>::new();
    let mut deduped_directions = Vec::with_capacity(directions.len());
    let mut deduped_random_flags = Vec::with_capacity(is_random.len());

    for (direction, random_flag) in directions.drain(..).zip(is_random.drain(..)) {
        if seen.insert(direction_dedup_key(&direction)) {
            deduped_directions.push(direction);
            deduped_random_flags.push(random_flag);
        }
    }

    let random_direction_count = deduped_random_flags.iter().filter(|flag| **flag).count();
    *directions = deduped_directions;
    *is_random = deduped_random_flags;
    random_direction_count
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "rounded direction buckets intentionally map onto compact integer dedup keys"
)]
fn direction_dedup_key(direction: &[f64]) -> Vec<i64> {
    direction
        .iter()
        .map(|value| {
            if value.abs() < 0.5 / DIRECTION_DEDUP_SCALE {
                0
            } else {
                (value * DIRECTION_DEDUP_SCALE).round() as i64
            }
        })
        .collect()
}

/// Approximate the full multi-period `HonestDiD` optimization surface via a
/// dense directional envelope and return a simultaneous joint region.
///
/// # Errors
/// Returns an error if the event-study input is inconsistent, the implied
/// direction set is empty, calibration fails, or any scalar directional
/// assessment fails.
pub fn assess_honest_event_study_optimization_surface_region_with_config(
    input: &HonestEventStudyInput,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
    surface_config: HonestOptimizationSurfaceConfig,
) -> Result<HonestDirectionalRegion, String> {
    let direction_set =
        generate_optimization_surface_directions(input.num_post_periods(), surface_config)?;
    let mut region = assess_honest_event_study_directional_region_with_config(
        input,
        &direction_set.directions,
        sensitivity,
        inference,
        null_value,
        relative_magnitude_config,
        joint_config,
    )?;
    region.diagnostics = HonestDirectionalRegionDiagnostics::fixed(
        direction_set.directions.len(),
        direction_set.random_direction_count,
    );
    Ok(region)
}

/// Approximate the full optimization surface with adaptive direction-set
/// enrichment.
///
/// # Errors
/// Returns an error if adaptive configuration validation, direction generation,
/// calibration, or directional scalar assessment fails.
#[cfg_attr(feature = "hotpath", hotpath::measure)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "surface approximation uses rounded dedup keys and a single orchestration pass"
)]
pub fn assess_honest_event_study_optimization_surface_region_adaptive_with_config(
    input: &HonestEventStudyInput,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    config: HonestOptimizationSurfaceAdaptiveRunConfig,
) -> Result<HonestDirectionalRegion, String> {
    let total_started = Instant::now();
    config.adaptive.validate()?;

    if config.surface.random_unit_directions == 0 {
        return assess_honest_event_study_optimization_surface_region_with_config(
            input,
            sensitivity,
            inference,
            null_value,
            config.relative_magnitude,
            config.joint,
            config.surface,
        );
    }

    let total_random = config.surface.random_unit_directions;
    let mut previous_pointwise: Option<f64> = None;
    let mut random_count = config.adaptive.random_batch_size.min(total_random);
    let mut iterations = 0usize;
    let mut did_converge = false;
    let mut chosen_random_draw = random_count;
    let mut warm_starts_by_rank: Option<Vec<Vec<f64>>> = None;

    while iterations < config.adaptive.max_iterations {
        let mut cfg = config.surface;
        cfg.random_unit_directions = random_count;
        let generate_started = Instant::now();
        let OptimizationSurfaceDirections {
            mut directions,
            mut random_direction_count,
        } = generate_optimization_surface_directions(input.num_post_periods(), cfg)?;
        emit_optimization_surface_timing(
            "generate_directions",
            sensitivity,
            directions.len(),
            random_direction_count,
            iterations,
            generate_started.elapsed().as_millis(),
        );
        let refine_started = Instant::now();
        let refined = refine_optimization_surface_directions_with_argmin(
            input,
            &mut directions,
            warm_starts_by_rank.as_deref(),
        )?;
        warm_starts_by_rank = Some(refined);
        let mut is_random = vec![false; directions.len().saturating_sub(random_direction_count)];
        is_random.extend(std::iter::repeat_n(true, random_direction_count));
        random_direction_count = deduplicate_unit_directions(&mut directions, &mut is_random);
        emit_optimization_surface_timing(
            "refine_directions",
            sensitivity,
            directions.len(),
            random_direction_count,
            iterations,
            refine_started.elapsed().as_millis(),
        );
        let calibrate_started = Instant::now();
        // Only the pointwise level is read here: it is what the convergence
        // test compares. The critical value used to be carried out of the loop
        // as well, and is not any more -- the region is recomputed at the chosen
        // count once the loop has decided, so this calibration exists purely to
        // answer "draw more?".
        let (current_pointwise, _) =
            calibrate_directional_region(input, &directions, inference, config.joint)?;
        emit_optimization_surface_timing(
            "calibrate",
            sensitivity,
            directions.len(),
            random_direction_count,
            iterations,
            calibrate_started.elapsed().as_millis(),
        );
        let converged_this_iter = previous_pointwise.is_some_and(|previous| {
            (current_pointwise - previous).abs() <= config.adaptive.pointwise_tolerance
                && random_direction_count >= config.adaptive.min_random_for_convergence
        });
        if converged_this_iter || random_count >= total_random {
            did_converge = converged_this_iter;
            chosen_random_draw = random_count;
            break;
        }
        previous_pointwise = Some(current_pointwise);
        random_count = (random_count + config.adaptive.random_batch_size).min(total_random);
        iterations += 1;
    }

    // THE ANSWER IS COMPUTED BY THE FULL-GRID PATH, at the number of random
    // directions this loop chose.
    //
    // The loop warm-starts each iteration's refinement from the previous one's
    // solutions. That is a real speedup and the right thing to do while deciding
    // whether to draw more directions; it must not decide the REGION. Refinement
    // is an iterative optimisation, so a warm start lands somewhere slightly
    // different from a cold one, `deduplicate_unit_directions` then merges a
    // near-coincident pair in one path and not the other, and the region comes
    // back with a different number of directions depending on how many batches
    // it took to arrive.
    //
    // Measured on the parity fixture, forced to the same 40 random directions:
    // warm-started, 45 directions and a pointwise level of 0.998888888888889;
    // full grid, 46 and 0.9989130434782608. The gap is immaterial and the
    // principle is not -- a confidence region has to be a function of the data
    // rather than of the search path that found it, and this repository has
    // fixed the same class of defect twice before, in the matched cohort's row
    // order and in the diagnosis-group tie-break.
    //
    // Re-refining the loop's OWN directions cold is not enough and was tried:
    // they have already been moved by the warm passes, so cold-refining them
    // starts somewhere the full path never stands. What restores the contract is
    // computing the final region the way the full path computes it.
    //
    // The saving the adaptive path exists for survives: it stops at the count it
    // chose rather than at `total_random`, which is the expensive number. What it
    // costs is one extra evaluation at that chosen count.
    let mut cfg = config.surface;
    cfg.random_unit_directions = chosen_random_draw;
    let final_started = Instant::now();
    let mut region = assess_honest_event_study_optimization_surface_region_with_config(
        input,
        sensitivity,
        inference,
        null_value,
        config.relative_magnitude,
        config.joint,
        cfg,
    )?;
    let settled_random_count = region.diagnostics.random_direction_count;
    emit_optimization_surface_timing(
        "final_directional_assessment",
        sensitivity,
        region.points.len(),
        settled_random_count,
        iterations + 1,
        final_started.elapsed().as_millis(),
    );
    region.diagnostics = HonestDirectionalRegionDiagnostics::adaptive(
        region.points.len(),
        settled_random_count,
        iterations + 1,
        did_converge,
    );
    emit_optimization_surface_timing(
        "total",
        sensitivity,
        region.points.len(),
        settled_random_count,
        iterations + 1,
        total_started.elapsed().as_millis(),
    );
    Ok(region)
}

/// Default wrapper for the adaptive optimization-surface region.
///
/// # Errors
/// Returns an error if adaptive configuration validation, direction generation,
/// calibration, or directional scalar assessment fails.
pub fn assess_honest_event_study_optimization_surface_region_adaptive(
    input: &HonestEventStudyInput,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
) -> Result<HonestDirectionalRegion, String> {
    assess_honest_event_study_optimization_surface_region_adaptive_with_config(
        input,
        sensitivity,
        inference,
        null_value,
        HonestOptimizationSurfaceAdaptiveRunConfig::from_inference(inference),
    )
}

/// Default wrapper for the fixed optimization-surface region.
///
/// # Errors
/// Returns an error if direction generation, calibration, or directional
/// scalar assessment fails.
pub fn assess_honest_event_study_optimization_surface_region(
    input: &HonestEventStudyInput,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
) -> Result<HonestDirectionalRegion, String> {
    let surface_config = HonestOptimizationSurfaceConfig::default_for_production();
    assess_honest_event_study_optimization_surface_region_with_config(
        input,
        sensitivity,
        inference,
        null_value,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        HonestJointPathConfig::default_for_production(),
        surface_config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_ranked_indices_by_variance_matches_full_sort_selection() {
        let sigma_post = vec![
            vec![1.0, 0.2, 0.1],
            vec![0.2, 1.2, 0.3],
            vec![0.1, 0.3, 0.8],
        ];
        let directions = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, -1.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![2.0, -1.0, 0.5],
        ]
        .into_iter()
        .map(|direction| canonicalize_unit_direction(&direction).expect("unit direction"))
        .collect::<Vec<_>>();

        let mut expected = directions
            .iter()
            .enumerate()
            .map(|(index, direction)| (index, projected_variance(direction, &sigma_post)))
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| right.1.total_cmp(&left.1));
        expected.truncate(4);

        let actual = top_ranked_indices_by_variance(&directions, &sigma_post, 4);
        assert_eq!(actual.len(), expected.len());
        for (actual_item, expected_item) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual_item.0, expected_item.0);
            assert_eq!(actual_item.1.to_bits(), expected_item.1.to_bits());
        }
    }

    #[test]
    fn deduplicate_unit_directions_preserves_unique_order_and_random_count() {
        let mut directions = vec![
            vec![1.0, 0.0],
            vec![1.0 + 1e-12, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
        ];
        let mut is_random = vec![false, true, true, false];
        let random_count = deduplicate_unit_directions(&mut directions, &mut is_random);
        assert_eq!(directions.len(), 2);
        assert_eq!(is_random, vec![false, true]);
        assert_eq!(random_count, 1);
    }
}
