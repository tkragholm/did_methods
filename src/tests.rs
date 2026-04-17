use super::*;
use faer::Mat;

use crate::error::InternalDidError as PortedDidError;
use crate::estimators::common::linalg::ridge_solve_lu;
use crate::estimators::common::testing::{
    horvitz_thompson, normalize_weights, stabilize, trim, xt_w_b, xt_w_x,
};
use crate::estimators::common::weights::diag_sparse_from_vec;
use crate::estimators::outcome::linear::LinearOutcome;
use crate::estimators::outcome::model::OutcomeModel;
use crate::estimators::propensity::common::logistic_scores;
use crate::estimators::propensity::ipt::ipt_loss_grad_hess;
use crate::estimators::propensity::irls::irls as irls_fit;
use crate::types::{DidCell, TimePeriod, TreatmentGroup};

use rstest::rstest;

fn panel_row(treated: bool, post_period: bool, outcome: f64) -> PanelObservation {
    PanelObservation::new(did_cell(treated, post_period), outcome)
}

fn did_cell(treated: bool, post_period: bool) -> DidCell {
    DidCell::from_parts(
        TreatmentGroup::from_bool(treated),
        TimePeriod::from_bool(post_period),
    )
}

fn treatment_group(treated: bool) -> TreatmentGroup {
    TreatmentGroup::from_bool(treated)
}

#[test]
fn estimates_att_for_simple_2x2_panel() {
    let panel = [
        panel_row(true, false, 10.0),
        panel_row(true, false, 10.0),
        panel_row(true, true, 18.0),
        panel_row(true, true, 18.0),
        panel_row(false, false, 7.0),
        panel_row(false, false, 7.0),
        panel_row(false, true, 12.0),
        panel_row(false, true, 12.0),
    ];

    let result = estimate_att_two_by_two(&panel, DidConfig::default()).expect("estimate");
    assert!((result.att - 3.0).abs() < 1e-12);
}

#[rstest]
#[case::missing_cells(
    &[
        panel_row(true, false, 10.0),
        panel_row(true, true, 11.0),
        panel_row(false, false, 9.0),
    ],
    DidError::EmptyCell {
        cell: DidCell::ControlPost,
    }
)]
#[case::invalid_weights(
    &[
        PanelObservation { weight: 0.0, ..panel_row(true, false, 10.0) },
        panel_row(true, true, 11.0),
        panel_row(false, false, 9.0),
        panel_row(false, true, 9.0),
    ],
    DidError::InvalidWeight { value: 0.0 }
)]
#[case::nan_outcome(
    &[
        panel_row(true, false, 10.0),
        panel_row(true, true, f64::NAN),
        panel_row(false, false, 9.0),
        panel_row(false, true, 9.0),
    ],
    DidError::InvalidOutcome { value: f64::NAN }
)]
fn validate_two_by_two_inputs(
    #[case] panel: &[PanelObservation],
    #[case] expected_error: DidError,
) {
    let error = estimate_att_two_by_two(panel, DidConfig::default()).expect_err("must fail");
    match (error, expected_error) {
        (DidError::InvalidOutcome { value: a }, DidError::InvalidOutcome { value: b }) => {
            assert!(a.is_nan() && b.is_nan());
        }
        (a, b) => assert_eq!(a, b),
    }
}

#[test]
fn summarizes_two_by_two_cells() {
    let panel = [
        panel_row(true, false, 10.0),
        panel_row(true, true, 14.0),
        panel_row(false, false, 8.0),
        panel_row(false, true, 10.0),
    ];
    let summary = summarize_two_by_two(&panel).expect("summary");
    assert_eq!(summary.total_observations(), 4);
    assert_eq!(summary.cell(DidCell::TreatedPre).observations, 1);
    assert!((summary.cell(DidCell::TreatedPost).mean_outcome - 14.0).abs() < 1e-12);
    assert!((summary.cell(DidCell::ControlPre).mean_outcome - 8.0).abs() < 1e-12);
    assert!((summary.cell(DidCell::ControlPost).mean_outcome - 10.0).abs() < 1e-12);
}

#[test]
fn aggregates_event_time_equal_weights() {
    let points = [
        EventTimePoint::new(-1, 0.1, 0.2),
        EventTimePoint::new(-1, 0.3, 0.4),
        EventTimePoint::new(0, 0.6, 0.5),
    ];
    let output = aggregate_event_time(&points, 0.95, EventTimeWeighting::Equal)
        .expect("aggregate event-time");
    assert_eq!(output.len(), 2);
    assert!((output[0].estimate - 0.2).abs() < 1e-12);
    assert_eq!(output[0].event_time, -1);
    assert_eq!(output[1].event_time, 0);
}

#[test]
fn aggregates_event_time_by_weight() {
    let points = [
        EventTimePoint {
            weight: 1.0,
            ..EventTimePoint::new(0, 1.0, 0.2)
        },
        EventTimePoint {
            weight: 3.0,
            ..EventTimePoint::new(0, 3.0, 0.4)
        },
    ];
    let output = aggregate_event_time(&points, 0.95, EventTimeWeighting::ByWeight)
        .expect("aggregate event-time");
    assert_eq!(output.len(), 1);
    assert!((output[0].estimate - 2.5).abs() < 1e-12);
}

#[test]
fn uses_generic_confidence_levels_for_event_time_ci() {
    let points = [EventTimePoint::new(0, 0.0, 1.0)];
    let output = aggregate_event_time(&points, 0.92, EventTimeWeighting::Equal)
        .expect("aggregate event-time");
    let margin = output[0].ci_high - output[0].estimate;
    // Two-sided z critical value at 92% confidence.
    assert!((margin - 1.750_686_071_252_17).abs() < 1e-6);
}

#[test]
fn confidence_interval_width_increases_with_confidence_level() {
    let points = [EventTimePoint::new(0, 0.0, 1.0)];
    let ci90 = aggregate_event_time(&points, 0.90, EventTimeWeighting::Equal).expect("ci90")[0];
    let ci95 = aggregate_event_time(&points, 0.95, EventTimeWeighting::Equal).expect("ci95")[0];
    let ci99 = aggregate_event_time(&points, 0.99, EventTimeWeighting::Equal).expect("ci99")[0];
    let width90 = ci90.ci_high - ci90.ci_low;
    let width95 = ci95.ci_high - ci95.ci_low;
    let width99 = ci99.ci_high - ci99.ci_low;
    assert!(width90 < width95);
    assert!(width95 < width99);
}

#[test]
fn rejects_invalid_event_time_weight() {
    let points = [EventTimePoint {
        weight: 0.0,
        ..EventTimePoint::new(0, 1.0, 0.1)
    }];
    let error =
        aggregate_event_time(&points, 0.95, EventTimeWeighting::ByWeight).expect_err("must fail");
    assert_eq!(error, EventTimeError::InvalidPointWeight);
}

#[test]
fn drdid_estimates_att_with_covariates() {
    let rows = vec![
        DrDidObservation {
            covariates: vec![1.0],
            ..DrDidObservation::new(treatment_group(true), 9.0)
        },
        DrDidObservation {
            covariates: vec![0.0],
            ..DrDidObservation::new(treatment_group(true), 8.0)
        },
        DrDidObservation {
            covariates: vec![1.0],
            ..DrDidObservation::new(treatment_group(true), 7.0)
        },
        DrDidObservation {
            covariates: vec![1.0],
            ..DrDidObservation::new(treatment_group(false), 6.0)
        },
        DrDidObservation {
            covariates: vec![0.0],
            ..DrDidObservation::new(treatment_group(false), 5.0)
        },
        DrDidObservation {
            covariates: vec![1.0],
            ..DrDidObservation::new(treatment_group(false), 4.0)
        },
    ];
    let config = DrDidConfig::builder()
        .bootstrap_reps(199)
        .bootstrap_seed(42)
        .build();
    let estimate = estimate_drdid_panel(&rows, config).expect("estimate drdid");
    assert_eq!(estimate.treated_n, 3);
    assert_eq!(estimate.control_n, 3);
    assert!((estimate.att - 3.0).abs() < 0.35);
    assert!(estimate.se.is_finite());
    assert!(estimate.ci_low <= estimate.ci_high);
}

#[test]
fn drdid_rejects_inconsistent_covariates() {
    let rows = vec![
        DrDidObservation {
            covariates: vec![1.0],
            ..DrDidObservation::new(treatment_group(true), 8.0)
        },
        DrDidObservation {
            covariates: vec![1.0, 0.0],
            ..DrDidObservation::new(treatment_group(false), 5.0)
        },
    ];
    let error = estimate_drdid_panel(&rows, DrDidConfig::default()).expect_err("must fail");
    assert_eq!(
        error,
        DrDidError::InconsistentCovariateCount {
            expected: 1,
            actual: 2
        }
    );
}

#[test]
fn drdid_rejects_missing_treated_or_control_groups() {
    let no_treated = vec![
        DrDidObservation::new(treatment_group(false), 4.0),
        DrDidObservation::new(treatment_group(false), 5.0),
    ];
    let err = estimate_drdid_panel(&no_treated, DrDidConfig::default()).expect_err("must fail");
    assert_eq!(err, DrDidError::NoTreated);

    let no_control = vec![
        DrDidObservation::new(treatment_group(true), 7.0),
        DrDidObservation::new(treatment_group(true), 8.0),
    ];
    let err = estimate_drdid_panel(&no_control, DrDidConfig::default()).expect_err("must fail");
    assert_eq!(err, DrDidError::NoControl);
}

#[test]
fn drdid_rejects_invalid_config_and_values() {
    let rows = vec![
        DrDidObservation::new(treatment_group(true), 9.0),
        DrDidObservation::new(treatment_group(false), 6.0),
    ];

    let err = estimate_drdid_panel(&rows, DrDidConfig::builder().confidence_level(1.0).build())
        .expect_err("must fail");
    assert_eq!(
        err,
        DrDidError::InvalidConfig("confidence_level must be finite and in (0, 1)".to_string())
    );

    let bad_weight = vec![
        DrDidObservation {
            weight: 1.0,
            ..DrDidObservation::new(treatment_group(true), 9.0)
        },
        DrDidObservation {
            weight: 0.0,
            ..DrDidObservation::new(treatment_group(false), 6.0)
        },
    ];
    let err = estimate_drdid_panel(&bad_weight, DrDidConfig::default()).expect_err("must fail");
    assert_eq!(err, DrDidError::InvalidWeight { value: 0.0 });

    let bad_outcome = vec![
        DrDidObservation::new(treatment_group(true), f64::INFINITY),
        DrDidObservation::new(treatment_group(false), 6.0),
    ];
    let err = estimate_drdid_panel(&bad_outcome, DrDidConfig::default()).expect_err("must fail");
    assert_eq!(
        err,
        DrDidError::InvalidOutcome {
            value: f64::INFINITY
        }
    );

    let bad_covariate = vec![
        DrDidObservation {
            covariates: vec![f64::NAN],
            ..DrDidObservation::new(treatment_group(true), 9.0)
        },
        DrDidObservation {
            covariates: vec![0.0],
            ..DrDidObservation::new(treatment_group(false), 6.0)
        },
    ];
    let err = estimate_drdid_panel(&bad_covariate, DrDidConfig::default()).expect_err("must fail");
    assert!(matches!(err, DrDidError::InvalidCovariate { value } if value.is_nan()));
}

fn make_logit_design() -> Mat<f64> {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
    Mat::from_fn(x.len(), 2, |row, col| if col == 0 { 1.0 } else { x[row] })
}

fn make_logit_target() -> Mat<f64> {
    let d = [0.0, 0.0, 0.0, 1.0, 1.0];
    Mat::from_fn(d.len(), 1, |row, _| d[row])
}

#[test]
fn irls_converges_for_simple_logistic_problem() {
    let design = make_logit_design();
    let target = make_logit_target();

    let beta =
        irls_fit(design.as_ref(), target.as_ref(), 200, 1e-8, 1e-8).expect("irls should converge");
    assert_eq!(beta.nrows(), 2);
    assert_eq!(beta.ncols(), 1);
    assert!(beta.col_as_slice(0).iter().all(|v| v.is_finite()));

    let ps = logistic_scores(design.as_ref(), beta.as_ref());
    assert_eq!(ps.len(), design.nrows());
    assert!(ps.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn irls_reports_non_convergence_when_no_iterations_allowed() {
    let design = make_logit_design();
    let target = make_logit_target();
    let err = irls_fit(design.as_ref(), target.as_ref(), 0, 1e-8, 1e-8).expect_err("must fail");
    assert_eq!(
        err,
        PortedDidError::Convergence("IRLS did not converge".to_string())
    );
}

#[test]
fn ipt_loss_gradient_hessian_are_finite_and_well_shaped() {
    let design = Mat::from_fn(2, 2, |row, col| match (row, col) {
        (0, 1) => -1.0,
        (0 | 1, 0) | (1, 1) => 1.0,
        _ => 0.0,
    });
    let target = Mat::from_fn(2, 1, |row, _| if row == 0 { 0.0 } else { 1.0 });
    let beta = Mat::from_fn(2, 1, |row, _| if row == 0 { 0.0 } else { 2.0 });

    // vstar=0.5 ensures one row follows the lower branch and one row the upper branch.
    let (loss, grad, hess) = ipt_loss_grad_hess(&design, &target, &beta, 0.5);
    assert!(loss.is_finite());
    assert_eq!(grad.nrows(), 2);
    assert_eq!(grad.ncols(), 1);
    assert_eq!(hess.nrows(), 2);
    assert_eq!(hess.ncols(), 2);
    assert!(grad.col_as_slice(0).iter().all(|v| v.is_finite()));
    for i in 0..2 {
        for j in 0..2 {
            assert!(hess.get(i, j).is_finite());
            assert!((hess.get(i, j) - hess.get(j, i)).abs() < 1e-12);
        }
    }
}

#[test]
fn ported_error_display_is_stable() {
    let estimation = PortedDidError::Estimation("bad matrix".to_string());
    let convergence = PortedDidError::Convergence("max iter".to_string());
    assert_eq!(format!("{estimation}"), "Estimation failed: bad matrix");
    assert_eq!(format!("{convergence}"), "Convergence error: max iter");
}

#[test]
fn weights_trim_stabilize_and_normalize_cover_edge_cases() {
    assert_eq!(trim(&[0.0, 0.5, 1.0], 0.1), vec![0.1, 0.5, 0.9]);

    assert!(stabilize(&[]).is_empty());
    assert_eq!(stabilize(&[1.0, -1.0]), vec![1.0, -1.0]);
    let stabilized = stabilize(&[1.0, 3.0]);
    assert!((stabilized[0] - 0.5).abs() < 1e-12);
    assert!((stabilized[1] - 1.5).abs() < 1e-12);

    let empty = Mat::<f64>::zeros(0, 1);
    assert!(normalize_weights(&empty).is_empty());

    let zero_sum = Mat::from_fn(2, 1, |row, _| if row == 0 { 1.0 } else { -1.0 });
    assert_eq!(normalize_weights(&zero_sum), vec![1.0, 1.0]);

    let base = Mat::from_fn(2, 1, |row, _| if row == 0 { 2.0 } else { 6.0 });
    let norm = normalize_weights(&base);
    assert!((norm[0] - 0.5).abs() < 1e-12);
    assert!((norm[1] - 1.5).abs() < 1e-12);
}

#[test]
fn horvitz_thompson_handles_regular_and_invalid_inputs() {
    let ht = horvitz_thompson(&[2.0, 4.0], &[1.0, 0.0], &[0.5, 0.5]).expect("ht");
    assert!((ht + 2.0).abs() < 1e-12);

    let clamped = horvitz_thompson(&[1.0, 1.0], &[1.0, 0.0], &[0.0, 1.0]).expect("clamped");
    assert!(clamped.is_finite());

    assert!(horvitz_thompson(&[], &[], &[]).is_err());
    assert!(horvitz_thompson(&[1.0], &[1.0, 0.0], &[0.5]).is_err());
    assert!(horvitz_thompson(&[1.0], &[1.0], &[f64::NAN]).is_err());
}

#[test]
fn diag_sparse_from_vec_builds_diagonal_matrix() {
    let mat = diag_sparse_from_vec(&[2.0, 4.0, 8.0]).expect("sparse diagonal");
    assert_eq!(mat.nrows(), 3);
    assert_eq!(mat.ncols(), 3);
}

#[test]
fn linear_outcome_fit_and_predict_cover_weighted_and_unweighted_paths() {
    let design = Mat::from_fn(
        3,
        2,
        |row, col| {
            if col == 0 { 1.0 } else { [0.0, 1.0, 2.0][row] }
        },
    );
    let target = Mat::from_fn(3, 1, |row, _| [1.0, 3.0, 5.0][row]);
    let model = LinearOutcome::default();

    let beta_unweighted = model.fit(design.as_ref(), target.as_ref(), None);
    assert!((beta_unweighted.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((beta_unweighted.get(1, 0) - 2.0).abs() < 1e-6);

    let beta_weighted = model.fit(design.as_ref(), target.as_ref(), Some(&[1.0, 2.0, 1.0]));
    assert!((beta_weighted.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((beta_weighted.get(1, 0) - 2.0).abs() < 1e-6);

    let preds = model.predict(design.as_ref(), beta_unweighted.as_ref());
    assert_eq!(preds.len(), 3);
    assert!((preds[0] - 1.0).abs() < 1e-6);
    assert!((preds[1] - 3.0).abs() < 1e-6);
    assert!((preds[2] - 5.0).abs() < 1e-6);
}

#[test]
fn linalg_helpers_cover_matrix_products_and_ridge_solver() {
    let x_t_w = Mat::from_fn(2, 2, |row, col| match (row, col) {
        (0, 0) => 1.0,
        (0, 1) => 2.0,
        (1, 0) => 3.0,
        (1, 1) => 4.0,
        _ => 0.0,
    });
    let x = Mat::from_fn(2, 2, |row, col| match (row, col) {
        (0, 0) => 5.0,
        (0, 1) => 6.0,
        (1, 0) => 7.0,
        (1, 1) => 8.0,
        _ => 0.0,
    });
    let b = Mat::from_fn(2, 1, |row, _| if row == 0 { 4.0 } else { 8.0 });

    let prod_x = xt_w_x(&x_t_w, &x);
    assert!((prod_x.get(0, 0) - 19.0).abs() < 1e-12);
    assert!((prod_x.get(0, 1) - 22.0).abs() < 1e-12);
    assert!((prod_x.get(1, 0) - 43.0).abs() < 1e-12);
    assert!((prod_x.get(1, 1) - 50.0).abs() < 1e-12);

    let prod_b = xt_w_b(&x_t_w, &b);
    assert!((prod_b.get(0, 0) - 20.0).abs() < 1e-12);
    assert!((prod_b.get(1, 0) - 44.0).abs() < 1e-12);

    let a = Mat::from_fn(2, 2, |row, col| match (row, col) {
        (0, 0) => 2.0,
        (1, 1) => 4.0,
        _ => 0.0,
    });
    let sol = ridge_solve_lu(a, &b, 2.0);
    assert!((sol.get(0, 0) - 1.0).abs() < 1e-12);
    assert!((sol.get(1, 0) - (8.0 / 6.0)).abs() < 1e-12);
}
