use crate::inference::{multiplier_bootstrap_ci, standard_error_from_influence};
use crate::methods::drdid::panel::estimate_drdid_panel;
use crate::types::{DrDidConfig, DrDidObservation, TripleDidObservation, TripleDidResult};

/// Estimates the Doubly Robust Triple-Difference (DR-DDD) ATT.
///
/// # Mathematical Implementation (OVS 2025)
/// `ATT_DDD = ATT(4 vs 3) + ATT(4 vs 2) - ATT(4 vs 1)`
///
/// The subgroup layout follows the `triplediff` package:
/// - `4`: `(S=1, Q=1)` focal treated-eligible group
/// - `3`: `(S=1, Q=0)` treated-ineligible group
/// - `2`: `(S=0, Q=1)` untreated-eligible group
/// - `1`: `(S=0, Q=0)` untreated-ineligible group
///
/// Each comparison runs a panel DR-DiD on subgroup `4` against one comparator
/// subgroup, then rescales the resulting influence function by `n / n_k`, matching
/// `triplediff::att_dr`.
///
/// # Errors
/// Returns an error string if:
/// - Observations are empty.
/// - A comparison subgroup is empty.
/// - The underlying DR-DiD estimation fails.
pub fn estimate_dr_ddd(
    observations: &[TripleDidObservation],
    config: DrDidConfig,
) -> Result<TripleDidResult, String> {
    if observations.is_empty() {
        return Err("no observations for DDD".to_string());
    }

    let subgroups = observations.iter().map(subgroup_code).collect::<Vec<_>>();
    let full_sample_n = crate::util::usize_to_f64(observations.len());

    let mut att_ddd = 0.0;
    let mut influence_function = vec![0.0; observations.len()];

    for (comparison_subgroup, sign) in [(3_u8, 1.0_f64), (2_u8, 1.0_f64), (1_u8, -1.0_f64)] {
        let selected_indices = subgroups
            .iter()
            .enumerate()
            .filter_map(|(idx, subgroup)| {
                if *subgroup == 4 || *subgroup == comparison_subgroup {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if selected_indices.is_empty() {
            return Err(format!("empty comparison subgroup {comparison_subgroup}"));
        }

        let comparison_n = crate::util::usize_to_f64(selected_indices.len());
        let cell_obs = selected_indices
            .iter()
            .map(|&idx| DrDidObservation {
                treated: subgroups[idx] == 4,
                delta_outcome: observations[idx].delta_outcome,
                covariates: observations[idx].covariates.clone(),
                weight: observations[idx].weight,
            })
            .collect::<Vec<_>>();

        let res = estimate_drdid_panel(&cell_obs, config).map_err(|e| {
            format!("DR-DiD failed for comparison subgroup {comparison_subgroup}: {e:?}")
        })?;

        let scale = full_sample_n / comparison_n;
        att_ddd = sign.mul_add(res.att, att_ddd);
        for (local_idx, &global_idx) in selected_indices.iter().enumerate() {
            influence_function[global_idx] = (sign * scale).mul_add(
                res.influence_function[local_idx],
                influence_function[global_idx],
            );
        }
    }

    let se = standard_error_from_influence(&influence_function);
    let (ci_low, ci_high) = multiplier_bootstrap_ci(
        att_ddd,
        &influence_function,
        config.inference(),
        config.bootstrap(),
    );

    Ok(TripleDidResult {
        att_ddd,
        se,
        ci_low,
        ci_high,
        influence_function,
    })
}

const fn subgroup_code(observation: &TripleDidObservation) -> u8 {
    match (observation.group_s, observation.partition_q) {
        (true, true) => 4,
        (true, false) => 3,
        (false, true) => 2,
        (false, false) => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_comparison_subgroup_fails_fast() {
        let obs = vec![
            TripleDidObservation {
                treated: true,
                group_s: true,
                partition_q: true,
                delta_outcome: 10.0,
                weight: 1.0,
                covariates: vec![],
            },
            TripleDidObservation {
                treated: false,
                group_s: false,
                partition_q: true,
                delta_outcome: 2.0,
                weight: 1.0,
                covariates: vec![],
            },
            TripleDidObservation {
                treated: false,
                group_s: false,
                partition_q: false,
                delta_outcome: 3.0,
                weight: 1.0,
                covariates: vec![],
            },
        ];

        let err = estimate_dr_ddd(&obs, DrDidConfig::default()).expect_err("must fail");
        assert!(err.contains("comparison subgroup 3"));
    }

    #[test]
    fn subgroup_coding_matches_triplediff_layout() {
        assert_eq!(
            subgroup_code(&TripleDidObservation {
                treated: true,
                group_s: true,
                partition_q: true,
                delta_outcome: 0.0,
                weight: 1.0,
                covariates: vec![],
            }),
            4
        );
        assert_eq!(
            subgroup_code(&TripleDidObservation {
                treated: true,
                group_s: true,
                partition_q: false,
                delta_outcome: 0.0,
                weight: 1.0,
                covariates: vec![],
            }),
            3
        );
        assert_eq!(
            subgroup_code(&TripleDidObservation {
                treated: false,
                group_s: false,
                partition_q: true,
                delta_outcome: 0.0,
                weight: 1.0,
                covariates: vec![],
            }),
            2
        );
        assert_eq!(
            subgroup_code(&TripleDidObservation {
                treated: false,
                group_s: false,
                partition_q: false,
                delta_outcome: 0.0,
                weight: 1.0,
                covariates: vec![],
            }),
            1
        );
    }
}
