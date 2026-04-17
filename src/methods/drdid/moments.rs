use itertools::izip;

use crate::types::DidCell;
use crate::util::usize_to_f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightNormalizationError {
    NonPositiveOrNonFiniteTotal,
}

pub fn normalize_weights_to_n(weights: &[f64]) -> Result<Vec<f64>, WeightNormalizationError> {
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(WeightNormalizationError::NonPositiveOrNonFiniteTotal);
    }
    let scale = usize_to_f64(weights.len()) / total;
    Ok(weights.iter().map(|weight| weight * scale).collect())
}

#[derive(Debug, Clone, Copy)]
pub struct RepeatedMomentInputs<'a> {
    pub normalized_weights: &'a [f64],
    pub treated: &'a [f64],
    pub post_period: &'a [f64],
    pub propensity: &'a [f64],
    pub signal: &'a [f64],
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepeatedMomentEstimate {
    pub att: f64,
    pub influence_function: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatedMomentError {
    InvalidInputShape,
    MissingCell { cell: DidCell },
}

#[derive(Debug, Clone)]
struct CellMomentVectors {
    treated_post_weights: Vec<f64>,
    treated_pre_weights: Vec<f64>,
    control_post_weights: Vec<f64>,
    control_pre_weights: Vec<f64>,
    treated_post_contributions: Vec<f64>,
    treated_pre_contributions: Vec<f64>,
    control_post_contributions: Vec<f64>,
    control_pre_contributions: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct CellWeightMeans {
    treated_post: f64,
    treated_pre: f64,
    control_post: f64,
    control_pre: f64,
}

#[derive(Debug, Clone, Copy)]
struct CellSignalMeans {
    treated_post: f64,
    treated_pre: f64,
    control_post: f64,
    control_pre: f64,
}

pub fn repeated_att_moments(
    inputs: RepeatedMomentInputs<'_>,
) -> Result<RepeatedMomentEstimate, RepeatedMomentError> {
    validate_input_shapes(inputs)?;

    let observation_count = inputs.normalized_weights.len();
    let observation_count_f64 = usize_to_f64(observation_count);
    let cell_moments = build_cell_moment_vectors(inputs);
    let weight_means = cell_weight_means(&cell_moments, observation_count_f64)?;
    let signal_means = cell_signal_means(&cell_moments, weight_means, observation_count_f64);

    let att = signal_means.treated_post - signal_means.treated_pre - signal_means.control_post
        + signal_means.control_pre;
    let influence_function = influence_function(&cell_moments, weight_means, signal_means);

    Ok(RepeatedMomentEstimate {
        att,
        influence_function,
    })
}

const fn validate_input_shapes(
    inputs: RepeatedMomentInputs<'_>,
) -> Result<(), RepeatedMomentError> {
    let observation_count = inputs.normalized_weights.len();
    if observation_count == 0
        || inputs.treated.len() != observation_count
        || inputs.post_period.len() != observation_count
        || inputs.propensity.len() != observation_count
        || inputs.signal.len() != observation_count
    {
        return Err(RepeatedMomentError::InvalidInputShape);
    }
    Ok(())
}

fn build_cell_moment_vectors(inputs: RepeatedMomentInputs<'_>) -> CellMomentVectors {
    let observation_count = inputs.normalized_weights.len();

    let mut treated_post_weights = Vec::with_capacity(observation_count);
    let mut treated_pre_weights = Vec::with_capacity(observation_count);
    let mut control_post_weights = Vec::with_capacity(observation_count);
    let mut control_pre_weights = Vec::with_capacity(observation_count);
    let mut treated_post_contributions = Vec::with_capacity(observation_count);
    let mut treated_pre_contributions = Vec::with_capacity(observation_count);
    let mut control_post_contributions = Vec::with_capacity(observation_count);
    let mut control_pre_contributions = Vec::with_capacity(observation_count);

    for (treated_indicator, post_indicator, propensity_score, signal_value, normalized_weight) in izip!(
        inputs.treated.iter(),
        inputs.post_period.iter(),
        inputs.propensity.iter(),
        inputs.signal.iter(),
        inputs.normalized_weights.iter()
    ) {
        let control_indicator = 1.0 - treated_indicator;
        let pre_indicator = 1.0 - post_indicator;
        let control_weight_divisor = (1.0 - propensity_score).max(1e-12);

        let treated_post_weight = normalized_weight * treated_indicator * post_indicator;
        let treated_pre_weight = normalized_weight * treated_indicator * pre_indicator;
        let control_post_weight =
            normalized_weight * propensity_score * control_indicator * post_indicator
                / control_weight_divisor;
        let control_pre_weight =
            normalized_weight * propensity_score * control_indicator * pre_indicator
                / control_weight_divisor;

        treated_post_weights.push(treated_post_weight);
        treated_pre_weights.push(treated_pre_weight);
        control_post_weights.push(control_post_weight);
        control_pre_weights.push(control_pre_weight);

        treated_post_contributions.push(treated_post_weight * signal_value);
        treated_pre_contributions.push(treated_pre_weight * signal_value);
        control_post_contributions.push(control_post_weight * signal_value);
        control_pre_contributions.push(control_pre_weight * signal_value);
    }

    CellMomentVectors {
        treated_post_weights,
        treated_pre_weights,
        control_post_weights,
        control_pre_weights,
        treated_post_contributions,
        treated_pre_contributions,
        control_post_contributions,
        control_pre_contributions,
    }
}

fn cell_weight_means(
    cell_moments: &CellMomentVectors,
    observation_count_f64: f64,
) -> Result<CellWeightMeans, RepeatedMomentError> {
    let means = CellWeightMeans {
        treated_post: cell_moments.treated_post_weights.iter().sum::<f64>() / observation_count_f64,
        treated_pre: cell_moments.treated_pre_weights.iter().sum::<f64>() / observation_count_f64,
        control_post: cell_moments.control_post_weights.iter().sum::<f64>() / observation_count_f64,
        control_pre: cell_moments.control_pre_weights.iter().sum::<f64>() / observation_count_f64,
    };

    if means.treated_post <= 0.0 {
        return Err(RepeatedMomentError::MissingCell {
            cell: DidCell::TreatedPost,
        });
    }
    if means.treated_pre <= 0.0 {
        return Err(RepeatedMomentError::MissingCell {
            cell: DidCell::TreatedPre,
        });
    }
    if means.control_post <= 0.0 {
        return Err(RepeatedMomentError::MissingCell {
            cell: DidCell::ControlPost,
        });
    }
    if means.control_pre <= 0.0 {
        return Err(RepeatedMomentError::MissingCell {
            cell: DidCell::ControlPre,
        });
    }

    Ok(means)
}

fn cell_signal_means(
    cell_moments: &CellMomentVectors,
    weight_means: CellWeightMeans,
    observation_count_f64: f64,
) -> CellSignalMeans {
    CellSignalMeans {
        treated_post: (cell_moments.treated_post_contributions.iter().sum::<f64>()
            / observation_count_f64)
            / weight_means.treated_post,
        treated_pre: (cell_moments.treated_pre_contributions.iter().sum::<f64>()
            / observation_count_f64)
            / weight_means.treated_pre,
        control_post: (cell_moments.control_post_contributions.iter().sum::<f64>()
            / observation_count_f64)
            / weight_means.control_post,
        control_pre: (cell_moments.control_pre_contributions.iter().sum::<f64>()
            / observation_count_f64)
            / weight_means.control_pre,
    }
}

fn influence_function(
    cell_moments: &CellMomentVectors,
    weight_means: CellWeightMeans,
    signal_means: CellSignalMeans,
) -> Vec<f64> {
    let observation_count = cell_moments.treated_post_weights.len();

    (0..observation_count)
        .map(|row_index| {
            let treated_post_term = signal_means.treated_post.mul_add(
                -cell_moments.treated_post_weights[row_index],
                cell_moments.treated_post_contributions[row_index],
            ) / weight_means.treated_post;
            let treated_pre_term = signal_means.treated_pre.mul_add(
                -cell_moments.treated_pre_weights[row_index],
                cell_moments.treated_pre_contributions[row_index],
            ) / weight_means.treated_pre;
            let control_post_term = signal_means.control_post.mul_add(
                -cell_moments.control_post_weights[row_index],
                cell_moments.control_post_contributions[row_index],
            ) / weight_means.control_post;
            let control_pre_term = signal_means.control_pre.mul_add(
                -cell_moments.control_pre_weights[row_index],
                cell_moments.control_pre_contributions[row_index],
            ) / weight_means.control_pre;
            treated_post_term - treated_pre_term - control_post_term + control_pre_term
        })
        .collect::<Vec<_>>()
}
