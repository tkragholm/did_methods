//! The solver counts in the diagnostics of the optimisation surface and the
//! smoothness confidence set. This file holds a single test so that no other
//! computation in the process moves the counters.

use did_methods::{
    HonestEventStudyInput, HonestSensitivity, InferenceConfig,
    assess_honest_event_study_optimization_surface_region, summarize_smoothness_sensitivity,
};

#[test]
fn counts_are_reported_and_bounded() {
    let input = HonestEventStudyInput {
        betahat: vec![0.02, -0.01, 0.03, 0.5, 0.7],
        covariance: (0..5)
            .map(|i| (0..5).map(|j| if i == j { 0.04 } else { 0.005 }).collect())
            .collect(),
        pre_periods: vec![-3, -2, -1],
        post_periods: vec![0, 1],
    };
    let inference = InferenceConfig::new(0.95);
    let summarize = || {
        summarize_smoothness_sensitivity(
            &input,
            inference,
            &[1.0, 0.0],
            None,
            Some(&[0.05]),
            None,
            None,
        )
        .expect("smoothness summary")
    };
    let summary = summarize();
    assert_eq!(summary.rows.len(), 1);
    assert_eq!(
        summary.solver,
        summarize().solver,
        "the counts of a deterministic call repeat"
    );

    let run = || {
        assess_honest_event_study_optimization_surface_region(
            &input,
            HonestSensitivity::Smoothness(0.05),
            inference,
            0.0,
        )
        .expect("surface region")
        .diagnostics
    };
    let first = run();
    let second = run();
    assert_eq!(first, second, "the counts of a deterministic call repeat");
    println!(
        "smoothness summary: {:?}; surface: {} retried, {} almost solved",
        summary.solver, first.solver_retries, first.solver_almost_solved
    );
}
