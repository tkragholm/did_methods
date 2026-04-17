use crate::types::{DidCell, DrDidError, DrDidRepeatedObservation};

pub(super) struct RepeatedPreparedData {
    pub(super) feature_count: usize,
    pub(super) treated_n: usize,
    pub(super) control_n: usize,
    pub(super) total_weight: f64,
    pub(super) treated_indicator: Vec<f64>,
    pub(super) post_indicator: Vec<f64>,
    pub(super) outcome: Vec<f64>,
    pub(super) sampling_weights: Vec<f64>,
    pub(super) design_matrix_flat: Vec<f64>,
}

#[allow(clippy::too_many_lines)]
pub(super) fn prepare_repeated_data(
    observations: &[DrDidRepeatedObservation],
) -> Result<RepeatedPreparedData, DrDidError> {
    let covariate_count = observations.first().map_or(0, |row| row.covariates.len());
    let feature_count = covariate_count.max(1);
    let observation_count = observations.len();

    let mut treated_n = 0_usize;
    let mut control_n = 0_usize;
    let mut total_weight = 0.0_f64;
    let mut has_treated_pre = false;
    let mut has_treated_post = false;
    let mut has_control_pre = false;
    let mut has_control_post = false;

    let mut treated_indicator = Vec::with_capacity(observation_count);
    let mut post_indicator = Vec::with_capacity(observation_count);
    let mut outcome = Vec::with_capacity(observation_count);
    let mut sampling_weights = Vec::with_capacity(observation_count);
    let mut design_matrix_flat = Vec::with_capacity(observation_count * feature_count);

    for row in observations {
        if row.covariates.len() != covariate_count {
            return Err(DrDidError::InconsistentCovariateCount {
                expected: covariate_count,
                actual: row.covariates.len(),
            });
        }
        if !row.outcome.is_finite() {
            return Err(DrDidError::InvalidOutcome { value: row.outcome });
        }
        if !row.weight.is_finite() || row.weight <= 0.0 {
            return Err(DrDidError::InvalidWeight { value: row.weight });
        }
        for covariate in &row.covariates {
            if !covariate.is_finite() {
                return Err(DrDidError::InvalidCovariate { value: *covariate });
            }
        }

        if row.treated {
            treated_n += 1;
            if row.post_period {
                has_treated_post = true;
            } else {
                has_treated_pre = true;
            }
            treated_indicator.push(1.0);
        } else {
            control_n += 1;
            if row.post_period {
                has_control_post = true;
            } else {
                has_control_pre = true;
            }
            treated_indicator.push(0.0);
        }

        post_indicator.push(if row.post_period { 1.0 } else { 0.0 });
        outcome.push(row.outcome);
        total_weight += row.weight;
        sampling_weights.push(row.weight);
        if covariate_count == 0 {
            design_matrix_flat.push(1.0);
        } else {
            design_matrix_flat.extend_from_slice(&row.covariates);
        }
    }

    if treated_n == 0 {
        return Err(DrDidError::NoTreated);
    }
    if control_n == 0 {
        return Err(DrDidError::NoControl);
    }
    if !has_treated_pre {
        return Err(DrDidError::MissingCell {
            cell: DidCell::TreatedPre,
        });
    }
    if !has_treated_post {
        return Err(DrDidError::MissingCell {
            cell: DidCell::TreatedPost,
        });
    }
    if !has_control_pre {
        return Err(DrDidError::MissingCell {
            cell: DidCell::ControlPre,
        });
    }
    if !has_control_post {
        return Err(DrDidError::MissingCell {
            cell: DidCell::ControlPost,
        });
    }

    Ok(RepeatedPreparedData {
        feature_count,
        treated_n,
        control_n,
        total_weight,
        treated_indicator,
        post_indicator,
        outcome,
        sampling_weights,
        design_matrix_flat,
    })
}
