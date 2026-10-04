#![allow(clippy::float_cmp)]

use super::relative_magnitude::geometry::{
    build_target_and_design, build_target_and_design_from_transform,
    prepare_relative_magnitude_branches, prepare_relative_magnitude_functional_transform,
};
use super::relative_magnitude::{
    compute_original_confidence_set, compute_relative_magnitude_identified_set,
    conditional_confidence_set::{
        compute_relative_magnitude_branch_identified_sets_with_prepared_branches,
        compute_relative_magnitude_confidence_set_with_prepared_branches,
        compute_relative_magnitude_confidence_set_with_prepared_functional_branches,
        compute_relative_magnitude_confidence_set_with_prepared_functional_branches_full_grid,
        prepare_relative_magnitude_functional_branches, prepare_relative_magnitude_input_branches,
    },
};
use super::*;
use crate::{AttGtEstimate, InferenceConfig};

fn assert_same_interval(left: (f64, f64), right: (f64, f64)) {
    for (left_bound, right_bound) in [(left.0, right.0), (left.1, right.1)] {
        if left_bound.is_infinite() || right_bound.is_infinite() {
            assert_eq!(left_bound, right_bound);
        } else {
            assert!((left_bound - right_bound).abs() < 1e-8);
        }
    }
}

#[test]
fn honest_ci_smoothness_bounds_identified_set() {
    let target = AttGtEstimate {
        att: 10.0,
        se: 1.0,
        ..Default::default()
    };
    let pre_trend = vec![AttGtEstimate {
        att: 1.0,
        se: 0.5,
        ..Default::default()
    }];
    let vcov = vec![vec![0.25, 0.0], vec![0.0, 1.0]];
    let inference = InferenceConfig::new(0.95);

    let result = compute_honest_ci_scalar(
        &target,
        &pre_trend,
        &vcov,
        HonestSensitivity::Smoothness(0.0),
        inference,
    )
    .unwrap();
    assert_eq!(result.identified_set, (11.0, 11.0));

    let result_m2 = compute_honest_ci_scalar(
        &target,
        &pre_trend,
        &vcov,
        HonestSensitivity::Smoothness(2.0),
        inference,
    )
    .unwrap();
    assert_eq!(result_m2.identified_set, (9.0, 13.0));
}

#[test]
fn honest_ci_relative_magnitude_bounds_identified_set() {
    let target = AttGtEstimate {
        att: 10.0,
        se: 1.0,
        ..Default::default()
    };
    let pre_trend = vec![AttGtEstimate {
        att: 2.0,
        se: 0.5,
        ..Default::default()
    }];
    let vcov = vec![vec![0.25, 0.0], vec![0.0, 1.0]];
    let inference = InferenceConfig::new(0.95);

    let result = compute_honest_ci_scalar(
        &target,
        &pre_trend,
        &vcov,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
    )
    .unwrap();
    assert_eq!(result.identified_set, (8.0, 12.0));
}

#[test]
fn honest_event_study_scalar_returns_one_result_per_post_period() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 2.0, 10.0, 20.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 4.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1],
    };
    let results = compute_honest_event_study_scalar_for_input(
        &input,
        HonestSensitivity::RelativeMagnitude(1.0),
        InferenceConfig::new(0.95),
    )
    .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].post_period, 0);
    assert_eq!(results[1].post_period, 1);
    assert!(
        results
            .iter()
            .all(|point| point.result.robust_ci.0.is_finite())
    );
    assert!(
        results
            .iter()
            .all(|point| point.result.robust_ci.1.is_finite())
    );
}

#[test]
fn construct_honest_original_cs_matches_manual_single_post_period() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 2.0, 10.0, 20.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 4.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1],
    };
    let result = compute_original_confidence_set(&input, &[1.0, 0.0], InferenceConfig::new(0.95))
        .expect("original cs");
    assert_eq!(result.estimate, 10.0);
    assert_eq!(result.se, 1.0);
    assert!(result.ci.0 < 10.0);
    assert!(result.ci.1 > 10.0);
}

#[test]
fn delta_rm_identified_set_matches_single_post_scalar_case() {
    let input = HonestEventStudyInput {
        betahat: vec![2.0, 10.0],
        covariance: vec![vec![0.25, 0.0], vec![0.0, 1.0]],
        pre_periods: vec![-1],
        post_periods: vec![0],
    };
    let identified =
        compute_relative_magnitude_identified_set(&input, &[1.0], 1.0).expect("DeltaRM idset");
    assert!((identified.lb - 8.0).abs() < 1e-6);
    assert!((identified.ub - 12.0).abs() < 1e-6);
}

#[test]
fn assess_honest_event_study_period_reports_significance_against_zero() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 2.0, 10.0, 20.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 4.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1],
    };

    let assessment = assess_honest_event_study_period(
        &input,
        0,
        HonestSensitivity::RelativeMagnitude(1.0),
        InferenceConfig::new(0.95),
        0.0,
    )
    .expect("period assessment");

    assert_eq!(assessment.null_value, 0.0);
    assert!(assessment.robust_ci.0 <= assessment.robust_ci.1);
    assert!(assessment.robustly_significant);
}

#[test]
fn assess_honest_event_study_functional_supports_average_post_effects() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 2.0, 10.0, 20.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 4.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1],
    };

    let assessment = assess_honest_event_study_functional(
        &input,
        &[0.5, 0.5],
        HonestSensitivity::RelativeMagnitude(1.0),
        InferenceConfig::new(0.95),
        0.0,
    )
    .expect("functional assessment");

    assert!((assessment.estimate - 15.0).abs() < 1e-8);
    assert!(assessment.robust_ci.0 <= assessment.robust_ci.1);
    assert!(assessment.original_ci.0 <= assessment.original_ci.1);
}

#[test]
fn assess_honest_event_study_functional_supports_smoothness_non_basis_functional() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 1.2, 10.0, 20.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 4.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1],
    };
    let assessment = assess_honest_event_study_functional(
        &input,
        &[0.5, 0.5],
        HonestSensitivity::Smoothness(0.5),
        InferenceConfig::new(0.95),
        0.0,
    )
    .expect("smoothness should support non-basis functionals");
    assert!((assessment.estimate - 15.0).abs() < 1e-8);
    assert!(assessment.robust_ci.0 <= assessment.robust_ci.1);
    assert!(assessment.identified_set.0 <= assessment.identified_set.1);
}

#[test]
fn honest_event_study_scalar_smoothness_matches_delta_sd_conditional_cs_for_basis_periods() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 1.2, 10.0, 20.0],
        covariance: vec![
            vec![0.25, 0.05, 0.0, 0.0],
            vec![0.05, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.2],
            vec![0.0, 0.0, 0.2, 4.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1],
    };
    let inference = InferenceConfig::new(0.95);
    let m = 0.5;

    let results = compute_honest_event_study_scalar_for_input(
        &input,
        HonestSensitivity::Smoothness(m),
        inference,
    )
    .expect("smoothness scalar results");

    for (post_idx, point) in results.iter().enumerate() {
        let post_weights =
            relative_magnitude::geometry::basis_post_weights(input.num_post_periods(), post_idx);
        let original =
            relative_magnitude::compute_original_confidence_set(&input, &post_weights, inference)
                .expect("original");
        let identified =
            smoothness::compute_smoothness_identified_set(&input, &post_weights, m).expect("idset");
        let conditional = smoothness::compute_smoothness_confidence_set_with_config(
            &input,
            &post_weights,
            m,
            inference,
            SmoothnessConfidenceSetConfig::from_inference(inference),
        )
        .expect("conditional cs");
        assert!((point.result.original_estimate - original.estimate).abs() < 1e-8);
        assert!((point.result.original_se - original.se).abs() < 1e-8);
        assert_same_interval(point.result.robust_ci, (conditional.lb, conditional.ub));
        assert_same_interval(point.result.identified_set, (identified.lb, identified.ub));
    }
}

#[test]
fn delta_sd_conditional_moments_match_r_post_period_row_filtering() {
    let full = smoothness::build_smoothness_moment_system(2, 2, 0.5, false);
    assert_eq!(full.constraint_matrix.len(), 6);
    assert_eq!(full.constraint_bounds, vec![0.5; 6]);
    assert_eq!(full.rows_for_arp, vec![0, 1, 2, 3, 4, 5]);

    let filtered = smoothness::build_smoothness_moment_system(2, 2, 0.5, true);
    assert_eq!(filtered.constraint_matrix, full.constraint_matrix);
    assert_eq!(filtered.constraint_bounds, full.constraint_bounds);
    assert_eq!(filtered.rows_for_arp, vec![1, 2, 4, 5]);
}

#[test]
fn delta_sd_conditional_cs_returns_unbounded_interval_when_pretrends_violate_bound() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 10.0, 5.0],
        covariance: vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0],
    };
    let result = compute_smoothness_confidence_set(&input, &[1.0], 1.0, InferenceConfig::new(0.95))
        .expect("conditional cs");
    assert_eq!(result.lb, f64::NEG_INFINITY);
    assert_eq!(result.ub, f64::INFINITY);
}

#[test]
fn delta_sd_conditional_cs_default_matches_explicit_flci_config() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 10.0, 5.0],
        covariance: vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0],
    };
    let inference = InferenceConfig::new(0.95);
    let default_result = compute_smoothness_confidence_set(&input, &[1.0], 1.0, inference)
        .expect("default conditional cs");
    let explicit = compute_smoothness_confidence_set_with_config(
        &input,
        &[1.0],
        1.0,
        inference,
        SmoothnessConfidenceSetConfig::from_inference(inference),
    )
    .expect("explicit conditional cs");
    assert_same_interval(
        (default_result.lb, default_result.ub),
        (explicit.lb, explicit.ub),
    );
}

#[test]
fn delta_sd_conditional_cs_least_favorable_is_nested_in_arp() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.2, 1.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0],
            vec![0.0, 0.25, 0.0],
            vec![0.0, 0.0, 0.5],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0],
    };
    let inference = InferenceConfig::new(0.95);
    let arp = compute_smoothness_confidence_set_with_config(
        &input,
        &[1.0],
        0.5,
        inference,
        SmoothnessConfidenceSetConfig {
            hybrid: SmoothnessHybrid::ArpOnly,
            ..SmoothnessConfidenceSetConfig::from_inference(inference)
        },
    )
    .expect("arp conditional cs");
    let lf = compute_smoothness_confidence_set_with_config(
        &input,
        &[1.0],
        0.5,
        inference,
        SmoothnessConfidenceSetConfig {
            hybrid: SmoothnessHybrid::LeastFavorable,
            ..SmoothnessConfidenceSetConfig::from_inference(inference)
        },
    )
    .expect("lf conditional cs");
    assert!(arp.lb <= lf.lb);
    assert!(lf.ub <= arp.ub);
}

#[test]
fn delta_sd_adaptive_grid_matches_full_grid_interval() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.2, 1.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0],
            vec![0.0, 0.25, 0.0],
            vec![0.0, 0.0, 0.5],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0],
    };
    let inference = InferenceConfig::new(0.95);
    let config = SmoothnessConfidenceSetConfig {
        hybrid: SmoothnessHybrid::ArpOnly,
        ..SmoothnessConfidenceSetConfig::from_inference(inference)
    };
    let post_weights = [1.0];
    let moments = smoothness::build_smoothness_moment_system(
        input.num_pre_periods(),
        input.num_post_periods(),
        0.5,
        config.post_period_moments_only,
    );
    let identified =
        smoothness::compute_smoothness_identified_set(&input, &post_weights, 0.5).expect("idset");
    let adaptive = compute_smoothness_confidence_set_with_config(
        &input,
        &post_weights,
        0.5,
        inference,
        config,
    )
    .expect("adaptive conditional cs");
    let full_grid = smoothness::compute_honest_linear_sensitivity_conditional_cs_full_grid(
        &input,
        &post_weights,
        0.5,
        inference,
        config,
        &moments,
        &identified,
    )
    .expect("full-grid conditional cs");

    assert_same_interval((adaptive.lb, adaptive.ub), (full_grid.lb, full_grid.ub));
}

#[test]
fn delta_sd_upper_bound_mpre_matches_manual_second_difference_bound() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 2.0, 4.0, 10.0],
        covariance: vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ],
        pre_periods: vec![-3, -2, -1],
        post_periods: vec![0],
    };
    let mub = estimate_smoothness_upper_bound_from_pretrends(&input, 0.05).expect("upper bound");
    assert!(mub.is_finite());
    assert!(mub > 0.0);
}

#[test]
fn create_honest_sensitivity_results_defaults_to_flci_and_default_m_grid() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 10.0, 5.0],
        covariance: vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0],
    };
    let result = summarize_smoothness_sensitivity(
        &input,
        InferenceConfig::new(0.95),
        &[1.0],
        None,
        None,
        None,
        None,
    )
    .expect("wrapper results");
    assert_eq!(result.rows.len(), 10);
    assert!(
        result
            .rows
            .iter()
            .all(|row| row.method == SensitivitySummaryMethod::Flci)
    );
    assert!(result.rows.iter().all(|row| row.delta == "DeltaSD"));
    assert!(result.rows.iter().all(|row| row.sensitivity_name == "M"));
}

#[test]
fn create_honest_sensitivity_results_conditional_method_uses_requested_labels() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.2, 1.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0],
            vec![0.0, 0.25, 0.0],
            vec![0.0, 0.0, 0.5],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0],
    };
    let result = summarize_smoothness_sensitivity(
        &input,
        InferenceConfig::new(0.95),
        &[1.0],
        Some(SensitivitySummaryMethod::ConditionalLeastFavorable),
        Some(&[0.0, 0.5]),
        None,
        None,
    )
    .expect("conditional wrapper results");
    assert_eq!(result.rows.len(), 2);
    assert!(
        result
            .rows
            .iter()
            .all(|row| row.method == SensitivitySummaryMethod::ConditionalLeastFavorable)
    );
}

#[test]
fn create_honest_sensitivity_results_relative_magnitudes_defaults_to_c_lf() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 2.0, 10.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0],
            vec![0.0, 0.25, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0],
    };
    let result = summarize_relative_magnitude_sensitivity(
        &input,
        InferenceConfig::new(0.95),
        &[1.0],
        None,
        None,
        None,
        None,
        None,
    )
    .expect("relative magnitude wrapper results");
    assert_eq!(result.rows.len(), 10);
    assert!(
        result
            .rows
            .iter()
            .all(|row| row.method == SensitivitySummaryMethod::ConditionalLeastFavorable)
    );
    assert!(result.rows.iter().all(|row| row.delta == "DeltaRM"));
    assert!(result.rows.iter().all(|row| row.sensitivity_name == "Mbar"));
}

#[test]
fn create_honest_sensitivity_results_relative_magnitudes_rejects_flci_method() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 2.0, 10.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0],
            vec![0.0, 0.25, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0],
    };
    let err = summarize_relative_magnitude_sensitivity(
        &input,
        InferenceConfig::new(0.95),
        &[1.0],
        Some(SensitivitySummaryMethod::Flci),
        Some(&[0.0]),
        None,
        None,
        None,
    )
    .expect_err("unsupported method should error");
    assert!(err.contains("supports only Conditional and C-LF"));
}

#[test]
fn delta_sdb_and_sdm_conditional_moments_match_r_row_counts() {
    let sdb = smoothness::build_signed_smoothness_moment_system(
        2,
        2,
        0.5,
        HonestBiasDirection::Positive,
        true,
    );
    assert_eq!(sdb.constraint_matrix.len(), 8);
    assert_eq!(
        sdb.constraint_bounds,
        vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]
    );
    assert_eq!(sdb.rows_for_arp, vec![1, 2, 4, 5, 6, 7]);

    let sdm = smoothness::build_monotone_smoothness_moment_system(
        2,
        2,
        0.5,
        HonestMonotonicityDirection::Increasing,
        true,
    );
    assert_eq!(sdm.constraint_matrix.len(), 10);
    assert_eq!(
        sdm.constraint_bounds,
        vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(sdm.rows_for_arp, vec![1, 2, 4, 5, 8, 9]);
}

#[test]
fn delta_sdb_and_sdm_identified_sets_are_well_formed() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.1, 0.4, 0.6],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 0.5, 0.0],
            vec![0.0, 0.0, 0.0, 0.5],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1],
    };
    let post_weights = [1.0, 0.0];
    let sdb = compute_signed_smoothness_identified_set(
        &input,
        &post_weights,
        0.5,
        HonestBiasDirection::Positive,
    )
    .expect("sdb idset");
    let sdm = compute_monotone_smoothness_identified_set(
        &input,
        &post_weights,
        0.5,
        HonestMonotonicityDirection::Increasing,
    )
    .expect("sdm idset");
    assert!(sdb.lb <= sdb.ub);
    assert!(sdm.lb <= sdm.ub);
    assert!(sdb.lb.is_finite() || sdb.lb == f64::NEG_INFINITY);
    assert!(sdb.ub.is_finite() || sdb.ub == f64::INFINITY);
    assert!(sdm.lb.is_finite() || sdm.lb == f64::NEG_INFINITY);
    assert!(sdm.ub.is_finite() || sdm.ub == f64::INFINITY);
}

#[test]
fn create_honest_sensitivity_results_supports_sign_and_monotone_variants() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.1, 0.4],
        covariance: vec![
            vec![0.25, 0.0, 0.0],
            vec![0.0, 0.25, 0.0],
            vec![0.0, 0.0, 0.5],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0],
    };
    let sign_restricted = summarize_smoothness_sensitivity(
        &input,
        InferenceConfig::new(0.95),
        &[1.0],
        None,
        Some(&[0.0]),
        Some(HonestBiasDirection::Positive),
        None,
    )
    .expect("sign-restricted wrapper");
    assert_eq!(
        sign_restricted.rows[0].method,
        SensitivitySummaryMethod::ConditionalFlci
    );
    assert_eq!(sign_restricted.rows[0].delta, "DeltaSDPB");

    let monotone = summarize_smoothness_sensitivity(
        &input,
        InferenceConfig::new(0.95),
        &[1.0],
        Some(SensitivitySummaryMethod::ConditionalLeastFavorable),
        Some(&[0.0]),
        None,
        Some(HonestMonotonicityDirection::Decreasing),
    )
    .expect("monotone wrapper");
    assert_eq!(
        monotone.rows[0].method,
        SensitivitySummaryMethod::ConditionalLeastFavorable
    );
    assert_eq!(monotone.rows[0].delta, "DeltaSDD");
}

#[test]
fn create_honest_sensitivity_results_relative_magnitudes_supports_sign_and_monotone_variants() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.1, 0.4],
        covariance: vec![
            vec![0.25, 0.0, 0.0],
            vec![0.0, 0.25, 0.0],
            vec![0.0, 0.0, 0.5],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0],
    };
    let sign_restricted = summarize_relative_magnitude_sensitivity(
        &input,
        InferenceConfig::new(0.95),
        &[1.0],
        None,
        Some(&[0.0]),
        None,
        Some(HonestBiasDirection::Positive),
        None,
    )
    .expect("relative sign-restricted wrapper");
    assert_eq!(
        sign_restricted.rows[0].method,
        SensitivitySummaryMethod::ConditionalLeastFavorable
    );
    assert_eq!(sign_restricted.rows[0].delta, "DeltaRMPB");

    let monotone = summarize_relative_magnitude_sensitivity(
        &input,
        InferenceConfig::new(0.95),
        &[1.0],
        Some(SensitivitySummaryMethod::Conditional),
        Some(&[0.0]),
        None,
        None,
        Some(HonestMonotonicityDirection::Decreasing),
    )
    .expect("relative monotone wrapper");
    assert_eq!(
        monotone.rows[0].method,
        SensitivitySummaryMethod::Conditional
    );
    assert_eq!(monotone.rows[0].delta, "DeltaRMD");
}

#[test]
fn create_honest_sensitivity_results_relative_magnitudes_supports_linear_trend_variants() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.1, 0.2, 0.4],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 0.5, 0.0],
            vec![0.0, 0.0, 0.0, 0.5],
        ],
        pre_periods: vec![-3, -2],
        post_periods: vec![0, 1],
    };
    let base = summarize_relative_magnitude_sensitivity(
        &input,
        InferenceConfig::new(0.95),
        &[1.0, 0.0],
        None,
        Some(&[0.0]),
        Some(HonestRelativeMagnitudeBound::LinearTrendDeviation),
        None,
        None,
    )
    .expect("linear-trend wrapper");
    assert_eq!(base.rows[0].delta, "DeltaSDRM");

    let sign_restricted = summarize_relative_magnitude_sensitivity(
        &input,
        InferenceConfig::new(0.95),
        &[1.0, 0.0],
        None,
        Some(&[0.0]),
        Some(HonestRelativeMagnitudeBound::LinearTrendDeviation),
        Some(HonestBiasDirection::Positive),
        None,
    )
    .expect("linear-trend sign wrapper");
    assert_eq!(sign_restricted.rows[0].delta, "DeltaSDRMPB");

    let monotone = summarize_relative_magnitude_sensitivity(
        &input,
        InferenceConfig::new(0.95),
        &[1.0, 0.0],
        Some(SensitivitySummaryMethod::Conditional),
        Some(&[0.0]),
        Some(HonestRelativeMagnitudeBound::LinearTrendDeviation),
        None,
        Some(HonestMonotonicityDirection::Increasing),
    )
    .expect("linear-trend monotone wrapper");
    assert_eq!(monotone.rows[0].delta, "DeltaSDRMI");
}

#[test]
fn honest_did_returns_original_and_robust_results_for_requested_post_period() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.1, 0.2, 0.4],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 0.5, 0.0],
            vec![0.0, 0.0, 0.0, 0.5],
        ],
        pre_periods: vec![-3, -2],
        post_periods: vec![0, 1],
    };
    let result = summarize_post_period_sensitivity(
        &input,
        1,
        SensitivityRestrictionKind::RelativeMagnitude,
        InferenceConfig::new(0.95),
        None,
        Some(&[0.0, 1.0]),
        Some(HonestRelativeMagnitudeBound::LinearTrendDeviation),
        None,
        None,
    )
    .expect("summarize_post_period_sensitivity result");
    assert_eq!(result.post_period, 1);
    assert_eq!(
        result.sensitivity_type,
        SensitivityRestrictionKind::RelativeMagnitude
    );
    assert_eq!(result.orig_ci.estimate, 0.4);
    assert_eq!(result.robust_ci.rows.len(), 2);
    assert!(
        result
            .robust_ci
            .rows
            .iter()
            .all(|row| row.delta == "DeltaSDRM")
    );
}

#[test]
fn delta_sdrm_family_results_are_well_formed() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.1, 0.2, 0.4],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 0.5, 0.0],
            vec![0.0, 0.0, 0.0, 0.5],
        ],
        pre_periods: vec![-3, -2],
        post_periods: vec![0, 1],
    };
    let post_weights = [1.0, 0.0];
    let sdrm = compute_linear_trend_relative_magnitude_identified_set(&input, &post_weights, 0.0)
        .expect("sdrm idset");
    let signed_linear = compute_signed_linear_trend_relative_magnitude_identified_set(
        &input,
        &post_weights,
        0.0,
        HonestBiasDirection::Positive,
    )
    .expect("sdrmb idset");
    let monotone_linear = compute_monotone_linear_trend_relative_magnitude_identified_set(
        &input,
        &post_weights,
        0.0,
        HonestMonotonicityDirection::Increasing,
    )
    .expect("sdrmm idset");
    assert!(sdrm.lb <= sdrm.ub);
    assert!(signed_linear.lb <= signed_linear.ub);
    assert!(monotone_linear.lb <= monotone_linear.ub);
}

#[test]
fn linear_trend_relative_magnitude_requires_two_pre_periods() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.4],
        covariance: vec![vec![0.25, 0.0], vec![0.0, 0.5]],
        pre_periods: vec![-2],
        post_periods: vec![0],
    };
    let err = summarize_relative_magnitude_sensitivity(
        &input,
        InferenceConfig::new(0.95),
        &[1.0],
        None,
        Some(&[0.0]),
        Some(HonestRelativeMagnitudeBound::LinearTrendDeviation),
        None,
        None,
    )
    .expect_err("linear-trend bound should require two pre-periods");
    assert!(err.contains("at least two pre-periods"));
}

#[test]
fn honest_did_errors_for_missing_post_period() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.1, 0.2],
        covariance: vec![
            vec![0.25, 0.0, 0.0],
            vec![0.0, 0.25, 0.0],
            vec![0.0, 0.0, 0.5],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0],
    };
    let err = summarize_post_period_sensitivity(
        &input,
        5,
        SensitivityRestrictionKind::Smoothness,
        InferenceConfig::new(0.95),
        None,
        Some(&[0.0]),
        None,
        None,
        None,
    )
    .expect_err("missing post period should error");
    assert!(err.contains("post period 5 not found"));
}

#[test]
fn honest_did_supports_smoothness_path() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.1, 0.2],
        covariance: vec![
            vec![0.25, 0.0, 0.0],
            vec![0.0, 0.25, 0.0],
            vec![0.0, 0.0, 0.5],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0],
    };
    let result = summarize_post_period_sensitivity(
        &input,
        0,
        SensitivityRestrictionKind::Smoothness,
        InferenceConfig::new(0.95),
        Some(SensitivitySummaryMethod::ConditionalLeastFavorable),
        Some(&[0.0, 0.5]),
        None,
        None,
        None,
    )
    .expect("smoothness summarize_post_period_sensitivity");
    assert_eq!(
        result.sensitivity_type,
        SensitivityRestrictionKind::Smoothness
    );
    assert_eq!(result.robust_ci.rows.len(), 2);
    assert!(
        result
            .robust_ci
            .rows
            .iter()
            .all(|row| row.delta == "DeltaSD")
    );
}

#[test]
fn honest_joint_path_reporting_emits_stable_csv_and_json() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 2.0, 10.0, 20.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 4.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1],
    };
    let region = assess_honest_event_study_joint_path_region(
        &input,
        HonestSensitivity::RelativeMagnitude(1.0),
        InferenceConfig::new(0.95),
        0.0,
    )
    .expect("joint region");
    let csv = render_honest_joint_path_report_csv(&region);
    assert!(csv.starts_with("schema_version,method,confidence_level"));
    assert!(csv.contains("honestdid_joint_report_v1"));

    let json = render_honest_joint_path_report_json(&region).expect("joint json");
    let payload: serde_json::Value = serde_json::from_str(&json).expect("parse json");
    assert_eq!(
        payload["schema_version"].as_str(),
        Some("honestdid_joint_report_v1")
    );
}

#[test]
fn honest_directional_reporting_emits_rows_with_direction_vectors() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 2.0, 10.0, 20.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 4.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1],
    };
    let directions = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let region = assess_honest_event_study_directional_region(
        &input,
        &directions,
        HonestSensitivity::RelativeMagnitude(1.0),
        InferenceConfig::new(0.95),
        0.0,
    )
    .expect("directional region");
    let rows = build_honest_directional_report_rows(&region);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].functional_id, 0);
    assert_eq!(rows[0].post_weights, vec![1.0, 0.0]);
}

#[test]
fn honest_multi_flci_reporting_emits_stable_schema() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 2.0, 10.0, 20.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 4.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1],
    };
    let problem = build_relative_magnitude_multi_flci_problem(
        &input,
        &[vec![1.0, 0.0], vec![0.0, 1.0]],
        1.0,
        InferenceConfig::new(0.95),
    )
    .expect("multi flci problem");
    let result = compute_relative_magnitude_multi_flci(&problem).expect("multi flci");
    let csv = render_honest_multi_flci_report_csv(&result);
    assert!(csv.starts_with(
        "schema_version,method,confidence_level,pointwise_confidence_level,calibrated_max_t_critical_value"
    ));
    assert!(csv.contains("null_value,robustly_significant"));
    let json = render_honest_multi_flci_report_json(&result).expect("multi flci json");
    let payload: serde_json::Value = serde_json::from_str(&json).expect("parse json");
    assert_eq!(
        payload["schema_version"].as_str(),
        Some("honestdid_joint_report_v1")
    );
    assert_eq!(payload["rows"][0]["null_value"].as_f64(), Some(0.0));
}

#[test]
fn honest_workflow_relative_magnitude_populates_multi_flci() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 2.0, 10.0, 20.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 4.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1],
    };
    let result = assess_honest_event_study_workflow_with_config(
        &input,
        &[
            HonestPostFunctional::Period(0),
            HonestPostFunctional::AverageWindow {
                start_period: 0,
                end_period: 1,
            },
        ],
        HonestSensitivity::RelativeMagnitude(1.0),
        InferenceConfig::new(0.95),
        0.0,
        &HonestWorkflowConfig {
            relative_magnitude: RelativeMagnitudeConfidenceSetConfig::from_inference(
                InferenceConfig::new(0.95),
            ),
            joint: HonestJointPathConfig::default_for_production(),
            direction_mode: HonestWorkflowDirectionMode::Basis,
        },
    )
    .expect("workflow");
    assert_eq!(result.functional_assessments.len(), 2);
    assert!(result.multi_flci.is_some());
}

#[test]
fn delta_rm_conditional_cs_handles_large_target_estimates() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 2.0, 150.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0],
            vec![0.0, 0.25, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0],
    };

    let result =
        compute_relative_magnitude_confidence_set(&input, &[1.0], 1.0, InferenceConfig::new(0.95))
            .expect("conditional cs");

    assert!(result.lb.is_finite());
    assert!(result.ub.is_finite());
    assert!(result.lb <= 150.0);
    assert!(result.ub >= 150.0);
}

#[test]
fn honest_workflow_smoothness_populates_multi_flci() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.2, 10.0, 20.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 4.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1],
    };
    let result = assess_honest_event_study_workflow_with_config(
        &input,
        &[HonestPostFunctional::Period(0)],
        HonestSensitivity::Smoothness(0.5),
        InferenceConfig::new(0.95),
        0.0,
        &HonestWorkflowConfig {
            relative_magnitude: RelativeMagnitudeConfidenceSetConfig::from_inference(
                InferenceConfig::new(0.95),
            ),
            joint: HonestJointPathConfig::default_for_production(),
            direction_mode: HonestWorkflowDirectionMode::Basis,
        },
    )
    .expect("workflow");
    assert_eq!(result.functional_assessments.len(), 1);
    assert!(result.multi_flci.is_some());
    let flci = result.multi_flci.expect("smoothness multi flci");
    assert_eq!(flci.points.len(), 1);
    assert!(flci.points[0].flci.0.is_finite());
    assert!(flci.points[0].flci.1.is_finite());
}

#[test]
fn honest_workflow_basis_matches_manual_components() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 2.0, 10.0, 20.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 4.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1],
    };
    let inference = InferenceConfig::new(0.95);
    let functionals = vec![HonestPostFunctional::Period(0)];
    let workflow = assess_honest_event_study_workflow_with_config(
        &input,
        &functionals,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
        &HonestWorkflowConfig {
            relative_magnitude: RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
            joint: HonestJointPathConfig::default_for_production(),
            direction_mode: HonestWorkflowDirectionMode::Basis,
        },
    )
    .expect("workflow");

    let manual_assessment = assess_honest_event_study_post_functional(
        &input,
        &HonestPostFunctional::Period(0),
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
    )
    .expect("manual assessment");
    let manual_joint = assess_honest_event_study_joint_path_region(
        &input,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
    )
    .expect("manual joint");

    assert_eq!(workflow.functional_assessments.len(), 1);
    assert_eq!(
        workflow.functional_assessments[0].assessment,
        manual_assessment
    );
    assert_eq!(
        workflow.joint_path_region.points.len(),
        manual_joint.points.len()
    );
    assert_eq!(
        workflow.directional_region.points.len(),
        input.num_post_periods()
    );
}

#[test]
fn relative_magnitude_cached_functional_transform_matches_direct_projection() {
    let a_post = vec![
        vec![1.0, 2.0, 3.0],
        vec![0.5, -1.0, 4.0],
        vec![2.0, 0.0, -0.25],
    ];
    let post_weights = vec![0.2, 0.3, 0.5];
    let direct =
        build_target_and_design(&post_weights, &a_post).expect("direct target/design projection");
    let prepared =
        prepare_relative_magnitude_functional_transform(&post_weights).expect("prepared transform");
    let cached = build_target_and_design_from_transform(&a_post, &prepared);

    assert_eq!(direct.0.len(), cached.0.len());
    assert_eq!(direct.1.len(), cached.1.len());
    for (left, right) in direct.0.iter().zip(cached.0.iter()) {
        assert!((left - right).abs() < 1e-10);
    }
    for (left_row, right_row) in direct.1.iter().zip(cached.1.iter()) {
        assert_eq!(left_row.len(), right_row.len());
        for (left, right) in left_row.iter().zip(right_row.iter()) {
            assert!((left - right).abs() < 1e-10);
        }
    }
}

#[test]
fn relative_magnitude_multi_flci_precomputed_builder_matches_direct_builder() {
    let input = HonestEventStudyInput {
        betahat: vec![1.0, 2.0, 10.0, 20.0],
        covariance: vec![
            vec![0.25, 0.0, 0.0, 0.0],
            vec![0.0, 0.25, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 4.0],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1],
    };
    let post_weight_sets = vec![vec![1.0, 0.0], vec![0.5, 0.5]];
    let inference = InferenceConfig::new(0.95);
    let direct =
        build_relative_magnitude_multi_flci_problem(&input, &post_weight_sets, 1.0, inference)
            .expect("direct multi-flci problem");
    let precomputed = build_relative_magnitude_multi_flci_problem_with_precomputed_sets(
        &input,
        &post_weight_sets,
        1.0,
        inference,
        direct.originals.clone(),
        direct.identified_sets.clone(),
    )
    .expect("precomputed multi-flci problem");

    assert_eq!(direct, precomputed);
}

#[test]
fn relative_magnitude_prepared_functional_branches_match_direct_conditional_confidence_set() {
    let input = HonestEventStudyInput {
        betahat: vec![0.25, -0.10, 0.40, 0.85, 1.10],
        covariance: vec![
            vec![0.16, 0.01, 0.00, 0.00, 0.00],
            vec![0.01, 0.20, 0.01, 0.00, 0.00],
            vec![0.00, 0.01, 0.25, 0.02, 0.01],
            vec![0.00, 0.00, 0.02, 0.36, 0.03],
            vec![0.00, 0.00, 0.01, 0.03, 0.49],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1, 2],
    };
    let post_weights = vec![0.2, 0.3, 0.5];
    let inference = InferenceConfig::new(0.95);
    let config = RelativeMagnitudeConfidenceSetConfig::from_inference(inference);
    let original = compute_original_confidence_set(&input, &post_weights, inference)
        .expect("original confidence set");
    let identified = compute_relative_magnitude_identified_set(&input, &post_weights, 1.0)
        .expect("identified set");
    let prepared_branches =
        prepare_relative_magnitude_branches(input.num_pre_periods(), input.num_post_periods(), 1.0)
            .expect("prepared branches");
    let prepared_input_branches =
        prepare_relative_magnitude_input_branches(&input, &prepared_branches);
    let prepared_functional_branches = prepare_relative_magnitude_functional_branches(
        &post_weights,
        &prepared_branches,
        &prepared_input_branches,
    )
    .expect("prepared functional branches");

    let direct = compute_relative_magnitude_confidence_set_with_prepared_branches(
        &input,
        &post_weights,
        inference,
        config,
        &original,
        &identified,
        &prepared_branches,
        &prepared_input_branches,
    )
    .expect("direct conditional confidence set");
    let cached = compute_relative_magnitude_confidence_set_with_prepared_functional_branches(
        &input,
        &post_weights,
        inference,
        config,
        &original,
        &identified,
        &prepared_branches,
        &prepared_input_branches,
        &prepared_functional_branches,
    )
    .expect("cached conditional confidence set");

    assert_same_interval((direct.lb, direct.ub), (cached.lb, cached.ub));
}

#[test]
fn relative_magnitude_adaptive_grid_matches_full_grid_interval() {
    let input = HonestEventStudyInput {
        betahat: vec![0.25, -0.10, 0.40, 0.85, 1.10],
        covariance: vec![
            vec![0.16, 0.01, 0.00, 0.00, 0.00],
            vec![0.01, 0.20, 0.01, 0.00, 0.00],
            vec![0.00, 0.01, 0.25, 0.02, 0.01],
            vec![0.00, 0.00, 0.02, 0.36, 0.03],
            vec![0.00, 0.00, 0.01, 0.03, 0.49],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1, 2],
    };
    let post_weights = vec![0.2, 0.3, 0.5];
    let inference = InferenceConfig::new(0.95);
    let config = RelativeMagnitudeConfidenceSetConfig::from_inference(inference);
    let original = compute_original_confidence_set(&input, &post_weights, inference)
        .expect("original confidence set");
    let identified = compute_relative_magnitude_identified_set(&input, &post_weights, 1.0)
        .expect("identified set");
    let prepared_branches =
        prepare_relative_magnitude_branches(input.num_pre_periods(), input.num_post_periods(), 1.0)
            .expect("prepared branches");
    let prepared_input_branches =
        prepare_relative_magnitude_input_branches(&input, &prepared_branches);
    let prepared_functional_branches = prepare_relative_magnitude_functional_branches(
        &post_weights,
        &prepared_branches,
        &prepared_input_branches,
    )
    .expect("prepared functional branches");

    let adaptive = compute_relative_magnitude_confidence_set_with_prepared_functional_branches(
        &input,
        &post_weights,
        inference,
        config,
        &original,
        &identified,
        &prepared_branches,
        &prepared_input_branches,
        &prepared_functional_branches,
    )
    .expect("adaptive conditional confidence set");
    let full_grid =
        compute_relative_magnitude_confidence_set_with_prepared_functional_branches_full_grid(
            inference,
            config,
            &original,
            &identified,
            &prepared_input_branches,
            &prepared_functional_branches,
        )
        .expect("full-grid conditional confidence set");

    assert_same_interval((adaptive.lb, adaptive.ub), (full_grid.lb, full_grid.ub));
}

/// An event study shaped like Study I's: a short pre-period, a long post-period,
/// and income-scale coefficients whose horizons are strongly correlated because
/// they share families.
fn study_shaped_input() -> HonestEventStudyInput {
    let betahat = vec![
        -83.27, 6.82, 105.78, 79.47, 227.29, 195.36, -212.48, -449.14, -685.27, -853.38,
    ];
    let se = [
        90.25, 84.31, 75.91, 60.43, 61.5, 74.55, 82.48, 90.93, 97.42, 102.79,
    ];
    let dim = betahat.len();
    let covariance = (0..dim)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let lag = f64::from(u32::try_from(i.abs_diff(j)).expect("lag fits"));
                    se[i] * se[j] * 0.5f64.mul_add(0.85f64.powf(lag), 0.5)
                })
                .collect()
        })
        .collect();
    HonestEventStudyInput {
        betahat,
        covariance,
        pre_periods: vec![-4, -3, -2, -1],
        post_periods: vec![0, 1, 2, 3, 4, 5],
    }
}

#[test]
fn most_relative_magnitude_branches_are_infeasible_at_the_observed_pre_trend() {
    // The precondition the next test rests on, asserted rather than assumed.
    // The identified-set LP fixes `delta_pre = betahat_pre`, and the branch
    // index names which pre-period attains the largest violation, so at one
    // realised pre-trend only one branch can be feasible.
    let input = study_shaped_input();
    let branches =
        prepare_relative_magnitude_branches(input.num_pre_periods(), input.num_post_periods(), 1.0)
            .expect("prepared branches");
    let mut post_weights = vec![0.0; input.num_post_periods()];
    post_weights[0] = 1.0;
    let sets = compute_relative_magnitude_branch_identified_sets_with_prepared_branches(
        &input,
        &post_weights,
        &branches,
    )
    .expect("branch identified sets");

    assert_eq!(sets.len(), 8);
    assert_eq!(sets.iter().filter(|set| set.is_some()).count(), 1);
}

#[test]
fn the_prepared_basis_period_path_agrees_with_the_matrix_path() {
    // Two implementations of one object. The prepared path prepares the branch
    // geometry once and walks the branches in order; the matrix path rebuilds
    // each branch's constraint matrix and solves them in parallel. They must
    // return the same confidence set.
    //
    // The prepared path SKIPPED any branch whose identified set was infeasible
    // until 4 September 2026, and by the test above that is seven branches of
    // eight here. Skipping them conditions the confidence set on the realised
    // pre-trend, which the method exists to avoid, and returns intervals that
    // are too narrow. `branch_anchor_idx` carries the argument.
    let input = study_shaped_input();
    let inference = InferenceConfig::new(0.95);
    let mbar = 1.0;

    let prepared = compute_honest_event_study_scalar_for_input(
        &input,
        HonestSensitivity::RelativeMagnitude(mbar),
        inference,
    )
    .expect("prepared basis-period path");
    assert_eq!(prepared.len(), input.num_post_periods());

    for (post_idx, point) in prepared.iter().enumerate() {
        let mut post_weights = vec![0.0; input.num_post_periods()];
        post_weights[post_idx] = 1.0;
        let matrix = summarize_relative_magnitude_sensitivity(
            &input,
            inference,
            &post_weights,
            None,
            Some(&[mbar]),
            None,
            None,
            None,
        )
        .expect("matrix path");
        assert_eq!(matrix.rows.len(), 1);
        assert_same_interval(
            point.result.robust_ci,
            (matrix.rows[0].lb, matrix.rows[0].ub),
        );
    }
}

/// An AR(1) covariance with unit-scale standard errors that grow over the
/// post-period, as the caller's rescaling to a largest SE of one leaves them.
fn ar1_covariance(dim: usize, rho: f64) -> Vec<Vec<f64>> {
    let last = f64::from(u32::try_from(dim - 1).expect("dimension fits"));
    let se: Vec<f64> = (0..dim)
        .map(|i| {
            0.65f64.mul_add(
                f64::from(u32::try_from(i).expect("index fits")) / last,
                0.35,
            )
        })
        .collect();
    (0..dim)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let lag = i32::try_from(i.abs_diff(j)).expect("lag fits");
                    se[i] * se[j] * rho.powi(lag)
                })
                .collect()
        })
        .collect()
}

/// Four pre-periods before the reference period and sixteen post-periods,
/// with `betahat_pre` as given and a declining post-period path.
fn near_tie_input(betahat_pre: [f64; 4]) -> HonestEventStudyInput {
    let mut betahat = betahat_pre.to_vec();
    betahat.extend((0..16).map(|t| {
        let t = f64::from(t);
        (0.006 * t).mul_add(t, 0.28f64.mul_add(-t, 0.3))
    }));
    let dim = betahat.len();
    HonestEventStudyInput {
        betahat,
        covariance: ar1_covariance(dim, 0.6),
        pre_periods: vec![-5, -4, -3, -2],
        post_periods: (0..16).collect(),
    }
}

/// The largest absolute step of the pre-period path, the reference period's
/// zero included.
fn largest_pre_step(betahat_pre: &[f64]) -> f64 {
    let mut path = betahat_pre.to_vec();
    path.push(0.0);
    path.windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(0.0, f64::max)
}

/// The base relative-magnitude identified set in closed form,
/// `l'beta_post +/- Mbar * max_step * sum_j |sum_{t >= j} l_t|`.
fn closed_form_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
) -> (f64, f64) {
    let num_pre = input.num_pre_periods();
    let estimate: f64 = post_weights
        .iter()
        .zip(&input.betahat[num_pre..])
        .map(|(weight, beta)| weight * beta)
        .sum();
    let reach: f64 = (0..post_weights.len())
        .map(|j| post_weights[j..].iter().sum::<f64>().abs())
        .sum();
    let half_width = mbar * largest_pre_step(&input.betahat[..num_pre]) * reach;
    (estimate - half_width, estimate + half_width)
}

#[test]
fn a_near_tie_of_the_two_largest_pre_steps_solves_on_the_surface() {
    // The two largest pre-period steps are 0.6016 (-3 to -2) and 0.6 (-2 to
    // the reference period), 0.27 % apart. The branch that names the smaller
    // one as the largest is infeasible by a small margin, which Clarabel
    // cannot certify: it stopped with InsufficientProgress at the larger Mbar
    // values of the late horizons.
    let input = near_tie_input([-1.3916, -1.0816, -1.2016, -0.6]);
    let inference = InferenceConfig::new(0.95);
    let mbars = [0.75, 1.0, 1.5];
    for horizon in [3, 9, 15] {
        let mut post_weights = vec![0.0; input.num_post_periods()];
        post_weights[horizon] = 1.0;
        let summaries = summarize_relative_magnitude_sensitivity_many(
            &input,
            inference,
            &[post_weights.as_slice()],
            None,
            Some(&mbars),
            None,
            None,
            None,
        )
        .unwrap_or_else(|error| panic!("t+{horizon}: {error}"));
        assert_eq!(summaries[0].rows.len(), mbars.len());
        for (row, mbar) in summaries[0].rows.iter().zip(mbars) {
            let identified = row.identified.as_ref().expect("identified set");
            let (lb, ub) = closed_form_identified_set(&input, &post_weights, mbar);
            assert!((identified.lb - lb).abs() < 1e-6, "t+{horizon} Mbar {mbar}");
            assert!((identified.ub - ub).abs() < 1e-6, "t+{horizon} Mbar {mbar}");
            assert!(row.lb <= identified.lb && identified.ub <= row.ub);
        }
    }
}

#[test]
fn a_near_tie_of_the_two_largest_pre_steps_solves_on_the_joint_path() {
    // The prepared-branch route of the joint path solves the same branch LPs.
    // Here the two largest steps are -0.1845 (-4 to -3) and -0.183 (-3 to
    // -2), 0.8 % apart.
    let input = near_tie_input([0.40, 0.43, 0.2455, 0.0625]);
    let inference = InferenceConfig::new(0.95);
    let joint_config = HonestJointPathConfig {
        method: HonestJointPathMethod::GaussianSimulated,
        simulation_draws: 2_000,
        simulation_seed: 7,
    };
    for mbar in [0.25, 0.5] {
        let region = assess_honest_event_study_joint_path_region_with_config(
            &input,
            HonestSensitivity::RelativeMagnitude(mbar),
            inference,
            0.0,
            RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
            joint_config,
        )
        .unwrap_or_else(|error| panic!("Mbar {mbar}: {error}"));
        assert_eq!(region.points.len(), input.num_post_periods());
        for (idx, point) in region.points.iter().enumerate() {
            let mut post_weights = vec![0.0; input.num_post_periods()];
            post_weights[idx] = 1.0;
            let (lb, ub) = closed_form_identified_set(&input, &post_weights, mbar);
            let (got_lb, got_ub) = point.assessment.identified_set;
            assert!((got_lb - lb).abs() < 1e-6, "t+{idx} Mbar {mbar}");
            assert!((got_ub - ub).abs() < 1e-6, "t+{idx} Mbar {mbar}");
        }
    }
}

mod near_tie_property {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn the_identified_set_matches_the_closed_form_when_the_largest_steps_nearly_tie(
            largest in 0.2f64..1.0,
            log_gap in -4.0f64..-2.0,
            tied_at in 0usize..4,
            other_offset in 1usize..4,
            small_steps in proptest::collection::vec(-1.0f64..1.0, 2),
            signs in proptest::collection::vec(any::<bool>(), 4),
            random_weights in proptest::collection::vec(-1.0f64..1.0, 16),
            one_hot in proptest::option::of(0usize..16),
            mbar in 0.05f64..2.0,
        ) {
            // A single horizon, as most of the surface's functionals are, or
            // a random linear combination of all of them.
            let weights: Vec<f64> = match one_hot {
                Some(horizon) => (0..16).map(|t| if t == horizon { 1.0 } else { 0.0 }).collect(),
                None => random_weights,
            };
            prop_assume!(weights.iter().any(|weight| weight.abs() > 1e-3));
            // Steps of the pre-period path, the last one into the reference
            // period. Two are the largest, a relative gap of 1e-4 to 1e-2
            // apart, and the other two are at most half the largest.
            let other_at = (tied_at + other_offset) % 4;
            let mut steps = [0.0f64; 4];
            let mut rest = small_steps.iter();
            for (idx, step) in steps.iter_mut().enumerate() {
                let magnitude = if idx == tied_at {
                    largest
                } else if idx == other_at {
                    largest * (1.0 - 10f64.powf(log_gap))
                } else {
                    0.5 * largest * rest.next().expect("two small steps").abs()
                };
                *step = if signs[idx] { magnitude } else { -magnitude };
            }
            // betahat_pre from the steps, walking back from the zero at the
            // reference period.
            let mut betahat_pre = [0.0f64; 4];
            let mut level = 0.0;
            for idx in (0..4).rev() {
                level -= steps[idx];
                betahat_pre[idx] = level;
            }
            let input = near_tie_input(betahat_pre);
            let identified = compute_relative_magnitude_identified_set(&input, &weights, mbar)
                .map_err(TestCaseError::fail)?;
            let (lb, ub) = closed_form_identified_set(&input, &weights, mbar);
            let tolerance = 1e-6 * (1.0 + lb.abs().max(ub.abs()));
            prop_assert!((identified.lb - lb).abs() < tolerance, "lb {} vs {}", identified.lb, lb);
            prop_assert!((identified.ub - ub).abs() < tolerance, "ub {} vs {}", identified.ub, ub);
        }
    }
}
