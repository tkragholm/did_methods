#![allow(dead_code, clippy::float_cmp)]

use did_methods::{
    AttGtConfig, AttGtObservation, ContinuousObservation, DidCcConfig, DrDidConfig,
    DrDidObservation, DrDidRepeatedObservation, HonestEventStudyInput, HonestJointPathConfig,
    HonestJointPathMethod, HonestOptimizationSurfaceAdaptiveConfig,
    HonestOptimizationSurfaceAdaptiveRunConfig, HonestOptimizationSurfaceConfig,
    HonestPostFunctional, HonestSensitivity, InferenceConfig, PolynomialBasis,
    RelativeMagnitudeConfidenceSetConfig, RelativeMagnitudeHybrid, TripleDidObservation,
    assess_honest_event_study_directional_region_with_config,
    assess_honest_event_study_joint_path_region,
    assess_honest_event_study_joint_path_region_with_config,
    assess_honest_event_study_optimization_surface_region_adaptive_with_config,
    assess_honest_event_study_optimization_surface_region_with_config,
    assess_honest_event_study_post_functional_flci,
    assess_honest_event_study_post_functional_flci_with_config,
    assess_honest_event_study_post_functional_multi_flci_with_config,
    build_relative_magnitude_flci_problem, build_relative_magnitude_multi_flci_problem,
    compute_honest_event_study_scalar_for_input, compute_original_confidence_set,
    compute_relative_magnitude_confidence_set,
    compute_relative_magnitude_confidence_set_with_config, compute_relative_magnitude_flci,
    compute_relative_magnitude_flci_with_config, compute_relative_magnitude_identified_set,
    compute_relative_magnitude_multi_flci, compute_relative_magnitude_multi_flci_with_config,
    estimate_acrt_sieve, estimate_att_gt, estimate_did_cc_robust, estimate_did_cc_stationary,
    estimate_dr_ddd, estimate_drdid_panel, inference::vcov::joint_wald_test,
    test_did_cc_stationarity,
};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct DrDidRef {
    att: f64,
    se: f64,
    y1: Vec<f64>,
    y0: Vec<f64>,
    treat: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    influence: Vec<f64>,
}

#[test]
fn parity_drdid_panel_lalonde() {
    let data = fs::read_to_string("tests/drdid_panel_ref.json").expect("read ref");
    let reference: DrDidRef = serde_json::from_str(&data).expect("parse json");

    let n = reference.treat.len();
    let mut observations = Vec::with_capacity(n);
    for i in 0..n {
        observations.push(DrDidObservation {
            treated: reference.treat[i] == 1,
            delta_outcome: reference.y1[i] - reference.y0[i],
            covariates: reference.covariates[i].clone(),
            weight: 1.0,
        });
    }

    let config = DrDidConfig::default();
    let result = estimate_drdid_panel(&observations, config).expect("estimate");
    assert!((result.att - reference.att).abs() < 1e-6);
}

#[derive(Deserialize)]
struct MpdtaRow {
    year: i32,
    #[serde(rename = "first.treat")]
    first_treat: i32,
    lemp: f64,
    countyreal: i64,
}

#[derive(Deserialize)]
struct DidRef {
    group: Vec<i32>,
    t: Vec<i32>,
    att: Vec<f64>,
    se: Vec<f64>,
    data_subset: Vec<MpdtaRow>,
}

#[test]
fn parity_did_attgt_mpdta() {
    let data = fs::read_to_string("tests/did_attgt_ref.json").expect("read ref");
    let reference: DidRef = serde_json::from_str(&data).expect("parse json");

    let mut observations = Vec::new();
    for row in reference.data_subset {
        let g = if row.first_treat == 0 {
            None
        } else {
            Some(row.first_treat)
        };
        observations.push(AttGtObservation::with_unit_id(
            row.countyreal,
            g,
            row.year,
            row.lemp,
        ));
    }

    let config = AttGtConfig::default();
    let results = estimate_att_gt(&observations, config).expect("estimate");

    for rust_res in &results {
        let ref_idx = reference
            .group
            .iter()
            .zip(reference.t.iter())
            .position(|(&g, &t)| g == rust_res.group && t == rust_res.time);

        if let Some(idx) = ref_idx {
            assert!((rust_res.att - reference.att[idx]).abs() < 1e-8);
            assert!((rust_res.se - reference.se[idx]).abs() < 1e-8);
        }
    }
}

#[derive(Deserialize)]
struct ContDidRef {
    delta_y: Vec<f64>,
    dose: Vec<f64>,
    treated: Vec<bool>,
}

#[test]
fn parity_contdid_simulation() {
    let data = fs::read_to_string("tests/contdid_ref.json").expect("read ref");
    let reference: ContDidRef = serde_json::from_str(&data).expect("parse json");

    let mut observations = Vec::new();
    for i in 0..reference.delta_y.len() {
        observations.push(ContinuousObservation {
            dose: reference.dose[i],
            delta_outcome: reference.delta_y[i],
            weight: 1.0,
        });
    }

    let basis = PolynomialBasis::new(3);
    let result = estimate_acrt_sieve(&observations, &basis).expect("sieve");

    assert!(result.acrt_glob.is_finite());
}

#[derive(Deserialize)]
struct TripleDidDataRow {
    id: i32,
    year: i32,
    partition: i32,
    x1: f64,
    x2: f64,
    treat: i32,
    outcome: f64,
}

#[derive(Deserialize)]
struct TripleDidRef {
    att: f64,
    se: f64,
    data: Vec<TripleDidDataRow>,
}

#[test]
fn parity_triple_did_panel() {
    let data = fs::read_to_string("tests/triple_did_ref.json").expect("read ref");
    let reference: TripleDidRef = serde_json::from_str(&data).expect("parse json");

    // Reconstruct delta outcomes from 2-period panel
    let mut obs = Vec::new();
    let n = reference.data.len();
    for i in (0..n).step_by(2) {
        let row_pre = &reference.data[i];
        let row_post = &reference.data[i + 1];
        assert_eq!(row_pre.id, row_post.id);

        obs.push(TripleDidObservation {
            treated: row_post.treat == 1,
            group_s: row_post.treat == 1,
            partition_q: row_post.partition == 1,
            delta_outcome: row_post.outcome - row_pre.outcome,
            weight: 1.0,
            covariates: vec![1.0, row_post.x1, row_post.x2],
        });
    }

    let config = DrDidConfig::default();
    let result = estimate_dr_ddd(&obs, config).expect("ddd");

    println!(
        "R Triple-DiD ATT: {}, Rust Triple-DiD ATT: {}",
        reference.att, result.att_ddd
    );
    assert!((result.att_ddd - reference.att).abs() < 1e-6);
}

#[derive(Deserialize)]
struct WaldRef {
    estimates: Vec<f64>,
    vcov: Vec<Vec<f64>>,
    statistic: f64,
    p_value: f64,
}

#[derive(Deserialize)]
struct DidCcScaffoldRow {
    treated: bool,
    post_period: bool,
    outcome: f64,
    covariates: Vec<f64>,
}

#[derive(Deserialize)]
struct DidCcScaffoldDesign {
    rows: Vec<DidCcScaffoldRow>,
    robust: DidCcEstimateRef,
    stationary: DidCcEstimateRef,
    hausman: DidCcHausmanRef,
}

#[derive(Deserialize)]
struct DidCcEstimateRef {
    att: f64,
    se: f64,
}

#[derive(Deserialize)]
struct DidCcHausmanRef {
    difference: f64,
    difference_se: f64,
    statistic: f64,
    p_value: f64,
}

#[derive(Deserialize)]
struct DidCcScaffoldFixture {
    aligned_constant_effect: DidCcScaffoldDesign,
    composition_shift: DidCcScaffoldDesign,
}

#[test]
fn parity_wald_test() {
    let data = fs::read_to_string("tests/wald_test_ref.json").expect("read ref");
    let reference: WaldRef = serde_json::from_str(&data).expect("parse json");

    let (stat, p, _) = joint_wald_test(&reference.estimates, &reference.vcov).expect("wald");
    assert!((stat - reference.statistic).abs() < 1e-10);
    assert!((p - reference.p_value).abs() < 1e-10);
}

#[test]
fn did_cc_scaffold_fixture_schema_loads() {
    let data =
        fs::read_to_string("tests/did_cc_ref_scaffold.json").expect("read did_cc scaffold ref");
    let fixture: DidCcScaffoldFixture = serde_json::from_str(&data).expect("parse did_cc scaffold");

    assert!(!fixture.aligned_constant_effect.rows.is_empty());
    assert!(!fixture.composition_shift.rows.is_empty());
    assert!(
        fixture
            .aligned_constant_effect
            .rows
            .iter()
            .all(|row| row.outcome.is_finite() && !row.covariates.is_empty())
    );
    assert!(
        fixture
            .composition_shift
            .rows
            .iter()
            .any(|row| row.treated && row.post_period)
    );
}

fn did_cc_rows(rows: &[DidCcScaffoldRow]) -> Vec<DrDidRepeatedObservation> {
    rows.iter()
        .map(|row| DrDidRepeatedObservation {
            treated: row.treated,
            post_period: row.post_period,
            outcome: row.outcome,
            weight: 1.0,
            covariates: row.covariates.clone(),
        })
        .collect()
}

#[test]
fn parity_did_cc_authored_reference_fixture() {
    let data =
        fs::read_to_string("tests/did_cc_ref_scaffold.json").expect("read did_cc scaffold ref");
    let fixture: DidCcScaffoldFixture = serde_json::from_str(&data).expect("parse did_cc scaffold");

    for design in [&fixture.aligned_constant_effect, &fixture.composition_shift] {
        let rows = did_cc_rows(&design.rows);
        let robust = estimate_did_cc_robust(&rows, DidCcConfig::default()).expect("robust did_cc");
        let stationary =
            estimate_did_cc_stationary(&rows, DidCcConfig::default()).expect("stationary did_cc");
        let hausman =
            test_did_cc_stationarity(&rows, DidCcConfig::default()).expect("hausman did_cc");

        assert!((robust.att - design.robust.att).abs() < 1e-6);
        assert!((robust.se - design.robust.se).abs() < 1e-6);
        assert!((stationary.att - design.stationary.att).abs() < 1e-6);
        assert!((stationary.se - design.stationary.se).abs() < 1e-6);
        assert!((hausman.difference - design.hausman.difference).abs() < 1e-6);
        assert!((hausman.difference_se - design.hausman.difference_se).abs() < 1e-6);
        assert!((hausman.statistic - design.hausman.statistic).abs() < 1e-6);
        assert!((hausman.p_value - design.hausman.p_value).abs() < 1e-6);
    }
}

#[derive(Deserialize)]
struct HonestRef {
    betahat: Vec<f64>,
    sigma: Vec<Vec<f64>>,
    pre_indices: Vec<usize>,
    post_indices: Vec<usize>,
}

#[derive(Deserialize)]
struct HonestOriginalRef {
    lb: f64,
    ub: f64,
    method: String,
}

#[derive(Deserialize)]
struct HonestRelativeMagnitudeRef {
    lb: f64,
    ub: f64,
    id_lb: f64,
    id_ub: f64,
    method: String,
    #[serde(rename = "Delta")]
    delta: String,
    #[serde(rename = "Mbar")]
    mbar: f64,
}

#[derive(Deserialize)]
struct HonestRelativeMagnitudeFixture {
    // `l_vec` is the R name for this vector, and what generate_reference_data.R
    // writes. Without the alias serde found no `post_weights`, `default` handed
    // back an empty Vec, and eight tests died downstream at "post_weights length
    // 0 does not match number of post periods 4" — a fixture-schema mismatch
    // wearing the costume of an estimator bug.
    //
    // `default` stays: honest_did_rm_ref.json genuinely carries no weight vector
    // and the tests reading it supply their own.
    #[serde(default, alias = "l_vec")]
    post_weights: Vec<f64>,
    original: HonestOriginalRef,
    relative_magnitude: Vec<HonestRelativeMagnitudeRef>,
}

#[derive(Deserialize)]
struct HonestJointPathRefMeta {
    mbar: f64,
    alpha_joint: f64,
    pointwise_confidence_level: f64,
}

#[derive(Deserialize)]
struct HonestJointPathRefPoint {
    post_period: i32,
    lb: f64,
    ub: f64,
}

#[derive(Deserialize)]
struct HonestJointPathRef {
    meta: HonestJointPathRefMeta,
    #[serde(alias = "l_vecs")]
    post_weight_sets: Vec<Vec<f64>>,
    points: Vec<HonestJointPathRefPoint>,
}

#[derive(Deserialize)]
struct HonestDirectionalRefDirection {
    name: String,
    #[serde(alias = "l_vec")]
    post_weights: Vec<f64>,
}

#[derive(Deserialize)]
struct HonestDirectionalRefPoint {
    name: String,
    lb: f64,
    ub: f64,
}

#[derive(Deserialize)]
struct HonestDirectionalRef {
    meta: HonestJointPathRefMeta,
    directions: Vec<HonestDirectionalRefDirection>,
    points: Vec<HonestDirectionalRefPoint>,
}

#[derive(Deserialize)]
struct HonestMultiFlciScaffold {
    #[serde(alias = "l_vecs")]
    post_weight_sets: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct HonestGaussianScaffoldMeta {
    confidence_level: f64,
}

#[derive(Deserialize)]
struct HonestGaussianCalibrationRef {
    calibrated_max_t_critical_value: f64,
    pointwise_confidence_level: f64,
}

#[derive(Deserialize)]
struct HonestGaussianScaffold {
    meta: HonestGaussianScaffoldMeta,
    joint: HonestGaussianCalibrationRef,
    directional: HonestGaussianCalibrationRef,
    directions: Vec<Vec<f64>>,
    mbar: f64,
    joint_points: Vec<HonestJointPathRefPoint>,
    directional_points: Vec<HonestDirectionalRefPoint>,
}

#[test]
fn honest_event_study_scalar_fixture_smoke() {
    let data = fs::read_to_string("tests/honest_did_ref.json").expect("read ref");
    let reference: HonestRef = serde_json::from_str(&data).expect("parse json");

    let pre_periods: Vec<i32> = reference
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..reference.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: reference.betahat,
        covariance: reference.sigma,
        pre_periods,
        post_periods,
    };

    let results = compute_honest_event_study_scalar_for_input(
        &input,
        HonestSensitivity::RelativeMagnitude(1.0),
        InferenceConfig::new(0.95),
    )
    .expect("honest event-study scalar");

    assert_eq!(results.len(), reference.post_indices.len());
    assert!(
        results
            .iter()
            .all(|result| result.result.original_estimate.is_finite())
    );
    assert!(
        results
            .iter()
            .all(|result| result.result.original_se.is_finite())
    );
    assert!(
        results
            .iter()
            .all(|result| result.result.identified_set.0.is_finite())
    );
    assert!(
        results
            .iter()
            .all(|result| result.result.identified_set.1.is_finite())
    );
    assert!(
        results
            .iter()
            .all(|result| result.result.robust_ci.0 <= result.result.robust_ci.1)
    );
}

#[test]
fn honest_original_cs_matches_reference_package() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let reference_data = fs::read_to_string("tests/honest_did_rm_ref.json").expect("read rm ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let rm_ref: HonestRelativeMagnitudeFixture =
        serde_json::from_str(&reference_data).expect("parse rm ref");
    assert_eq!(rm_ref.original.method, "Original");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    let original =
        compute_original_confidence_set(&input, &[1.0, 0.0, 0.0, 0.0], InferenceConfig::new(0.95))
            .expect("original cs");

    assert!((original.ci.0 - rm_ref.original.lb).abs() < 1e-8);
    assert!((original.ci.1 - rm_ref.original.ub).abs() < 1e-8);
}

#[test]
fn honest_relative_magnitude_fixture_is_available_for_future_parity() {
    let reference_data = fs::read_to_string("tests/honest_did_rm_ref.json").expect("read rm ref");
    let rm_ref: HonestRelativeMagnitudeFixture =
        serde_json::from_str(&reference_data).expect("parse rm ref");
    assert_eq!(rm_ref.relative_magnitude.len(), 2);
    assert!(
        rm_ref
            .relative_magnitude
            .iter()
            .all(|row| row.method == "C-LF")
    );
    assert!(
        rm_ref
            .relative_magnitude
            .iter()
            .all(|row| row.delta == "DeltaRM")
    );
    assert!(
        rm_ref
            .relative_magnitude
            .iter()
            .all(|row| row.id_lb <= row.id_ub)
    );
    assert_eq!(rm_ref.relative_magnitude[0].mbar, 0.5);
    assert_eq!(rm_ref.relative_magnitude[1].mbar, 1.0);
}

#[test]
fn honest_delta_rm_identified_set_matches_reference_package_first_post() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let reference_data = fs::read_to_string("tests/honest_did_rm_ref.json").expect("read rm ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let rm_ref: HonestRelativeMagnitudeFixture =
        serde_json::from_str(&reference_data).expect("parse rm ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    for row in &rm_ref.relative_magnitude {
        let identified =
            compute_relative_magnitude_identified_set(&input, &[1.0, 0.0, 0.0, 0.0], row.mbar)
                .expect("DeltaRM identified set");
        assert!((identified.lb - row.id_lb).abs() < 1e-6);
        assert!((identified.ub - row.id_ub).abs() < 1e-6);
    }
}

#[test]
fn honest_delta_rm_conditional_cs_matches_reference_package_first_post() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let reference_data = fs::read_to_string("tests/honest_did_rm_ref.json").expect("read rm ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let rm_ref: HonestRelativeMagnitudeFixture =
        serde_json::from_str(&reference_data).expect("parse rm ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    for row in &rm_ref.relative_magnitude {
        let conditional = compute_relative_magnitude_confidence_set(
            &input,
            &[1.0, 0.0, 0.0, 0.0],
            row.mbar,
            InferenceConfig::new(0.95),
        )
        .expect("DeltaRM conditional cs");
        assert!((conditional.lb - row.lb).abs() < 5e-3);
        assert!((conditional.ub - row.ub).abs() < 5e-3);
    }
}

#[test]
fn honest_delta_rm_average_post_matches_reference_package() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let reference_data =
        fs::read_to_string("tests/honest_did_rm_avg_ref.json").expect("read avg rm ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let rm_ref: HonestRelativeMagnitudeFixture =
        serde_json::from_str(&reference_data).expect("parse avg rm ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    let original =
        compute_original_confidence_set(&input, &rm_ref.post_weights, InferenceConfig::new(0.95))
            .expect("average-post original cs");
    assert!((original.ci.0 - rm_ref.original.lb).abs() < 1e-8);
    assert!((original.ci.1 - rm_ref.original.ub).abs() < 1e-8);

    for row in &rm_ref.relative_magnitude {
        let identified =
            compute_relative_magnitude_identified_set(&input, &rm_ref.post_weights, row.mbar)
                .expect("avg identified set");
        assert!((identified.lb - row.id_lb).abs() < 1e-6);
        assert!((identified.ub - row.id_ub).abs() < 1e-6);

        let conditional = compute_relative_magnitude_confidence_set(
            &input,
            &rm_ref.post_weights,
            row.mbar,
            InferenceConfig::new(0.95),
        )
        .expect("avg conditional cs");
        assert!((conditional.lb - row.lb).abs() < 5e-3);
        assert!((conditional.ub - row.ub).abs() < 5e-3);
    }
}

#[test]
fn honest_delta_rm_conditional_default_wrapper_matches_explicit_lf_config() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let reference_data =
        fs::read_to_string("tests/honest_did_rm_avg_ref.json").expect("read avg rm ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let rm_ref: HonestRelativeMagnitudeFixture =
        serde_json::from_str(&reference_data).expect("parse avg rm ref");
    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();
    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };
    let inference = InferenceConfig::default();

    let wrapped =
        compute_relative_magnitude_confidence_set(&input, &rm_ref.post_weights, 0.5, inference)
            .expect("wrapped conditional cs");
    let explicit = compute_relative_magnitude_confidence_set_with_config(
        &input,
        &rm_ref.post_weights,
        0.5,
        inference,
        RelativeMagnitudeConfidenceSetConfig {
            hybrid: RelativeMagnitudeHybrid::LeastFavorable,
            hybrid_kappa: (1.0 - inference.confidence_level) / 20.0,
        },
    )
    .expect("explicit LF conditional cs");

    assert!((wrapped.lb - explicit.lb).abs() < 1e-10);
    assert!((wrapped.ub - explicit.ub).abs() < 1e-10);
}

#[test]
fn honest_assessment_average_post_matches_reference_package() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let reference_data =
        fs::read_to_string("tests/honest_did_rm_avg_ref.json").expect("read avg rm ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let rm_ref: HonestRelativeMagnitudeFixture =
        serde_json::from_str(&reference_data).expect("parse avg rm ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    for row in &rm_ref.relative_magnitude {
        let assessment = did_methods::assess_honest_event_study_functional(
            &input,
            &rm_ref.post_weights,
            did_methods::HonestSensitivity::RelativeMagnitude(row.mbar),
            InferenceConfig::new(0.95),
            0.0,
        )
        .expect("avg assessment");

        assert!((assessment.original_ci.0 - rm_ref.original.lb).abs() < 1e-8);
        assert!((assessment.original_ci.1 - rm_ref.original.ub).abs() < 1e-8);
        assert!((assessment.identified_set.0 - row.id_lb).abs() < 1e-6);
        assert!((assessment.identified_set.1 - row.id_ub).abs() < 1e-6);
        assert!((assessment.robust_ci.0 - row.lb).abs() < 5e-3);
        assert!((assessment.robust_ci.1 - row.ub).abs() < 5e-3);
    }
}

#[test]
fn honest_assessment_default_wrapper_matches_explicit_lf_config() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let reference_data =
        fs::read_to_string("tests/honest_did_rm_avg_ref.json").expect("read avg rm ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let rm_ref: HonestRelativeMagnitudeFixture =
        serde_json::from_str(&reference_data).expect("parse avg rm ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    let inference = InferenceConfig::new(0.95);
    let wrapped = did_methods::assess_honest_event_study_functional(
        &input,
        &rm_ref.post_weights,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
    )
    .expect("wrapped assessment");

    let explicit = did_methods::assess_honest_event_study_functional_with_config(
        &input,
        &rm_ref.post_weights,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
        RelativeMagnitudeConfidenceSetConfig {
            hybrid: RelativeMagnitudeHybrid::LeastFavorable,
            hybrid_kappa: (1.0 - inference.confidence_level) / 20.0,
        },
    )
    .expect("explicit LF assessment");

    assert_eq!(wrapped, explicit);
}

#[test]
fn honest_post_functional_average_window_matches_reference_package() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let reference_data =
        fs::read_to_string("tests/honest_did_rm_avg_ref.json").expect("read avg rm ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let rm_ref: HonestRelativeMagnitudeFixture =
        serde_json::from_str(&reference_data).expect("parse avg rm ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    let functional = HonestPostFunctional::AverageWindow {
        start_period: 0,
        end_period: 3,
    };

    for row in &rm_ref.relative_magnitude {
        let assessment = did_methods::assess_honest_event_study_post_functional(
            &input,
            &functional,
            did_methods::HonestSensitivity::RelativeMagnitude(row.mbar),
            InferenceConfig::new(0.95),
            0.0,
        )
        .expect("average-window assessment");

        assert!((assessment.original_ci.0 - rm_ref.original.lb).abs() < 1e-8);
        assert!((assessment.original_ci.1 - rm_ref.original.ub).abs() < 1e-8);
        assert!((assessment.identified_set.0 - row.id_lb).abs() < 1e-6);
        assert!((assessment.identified_set.1 - row.id_ub).abs() < 1e-6);
        assert!((assessment.robust_ci.0 - row.lb).abs() < 5e-3);
        assert!((assessment.robust_ci.1 - row.ub).abs() < 5e-3);
    }
}

#[test]
fn honest_post_functional_period_matches_period_wrapper() {
    let data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let input_ref: HonestRef = serde_json::from_str(&data).expect("parse input ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    let direct = did_methods::assess_honest_event_study_period(
        &input,
        0,
        HonestSensitivity::RelativeMagnitude(1.0),
        InferenceConfig::new(0.95),
        0.0,
    )
    .expect("direct period");

    let typed = did_methods::assess_honest_event_study_post_functional(
        &input,
        &HonestPostFunctional::Period(0),
        HonestSensitivity::RelativeMagnitude(1.0),
        InferenceConfig::new(0.95),
        0.0,
    )
    .expect("typed period");

    assert_eq!(direct, typed);
}

#[test]
fn honest_post_functional_with_config_matches_explicit_functional_config() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let reference_data =
        fs::read_to_string("tests/honest_did_rm_avg_ref.json").expect("read avg rm ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let rm_ref: HonestRelativeMagnitudeFixture =
        serde_json::from_str(&reference_data).expect("parse avg rm ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    let inference = InferenceConfig::new(0.95);
    let functional = HonestPostFunctional::AverageWindow {
        start_period: 0,
        end_period: 3,
    };
    let config = RelativeMagnitudeConfidenceSetConfig {
        hybrid: RelativeMagnitudeHybrid::LeastFavorable,
        hybrid_kappa: (1.0 - inference.confidence_level) / 20.0,
    };

    let direct = did_methods::assess_honest_event_study_functional_with_config(
        &input,
        &rm_ref.post_weights,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
        config,
    )
    .expect("direct functional config");

    let typed = did_methods::assess_honest_event_study_post_functional_with_config(
        &input,
        &functional,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
        config,
    )
    .expect("typed functional config");

    assert_eq!(direct, typed);
}

#[test]
fn honest_delta_rm_flci_problem_builder_matches_average_post_reference() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let reference_data =
        fs::read_to_string("tests/honest_did_rm_avg_ref.json").expect("read avg rm ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let rm_ref: HonestRelativeMagnitudeFixture =
        serde_json::from_str(&reference_data).expect("parse avg rm ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    let problem = build_relative_magnitude_flci_problem(
        &input,
        &rm_ref.post_weights,
        rm_ref.relative_magnitude[1].mbar,
        InferenceConfig::new(0.95),
    )
    .expect("flci problem");

    assert!((problem.original.ci.0 - rm_ref.original.lb).abs() < 1e-8);
    assert!((problem.original.ci.1 - rm_ref.original.ub).abs() < 1e-8);
    assert!((problem.identified_set.lb - rm_ref.relative_magnitude[1].id_lb).abs() < 1e-6);
    assert!((problem.identified_set.ub - rm_ref.relative_magnitude[1].id_ub).abs() < 1e-6);
}

#[test]
fn honest_delta_rm_flci_matches_average_post_reference_package() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let reference_data =
        fs::read_to_string("tests/honest_did_rm_avg_ref.json").expect("read avg rm ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let rm_ref: HonestRelativeMagnitudeFixture =
        serde_json::from_str(&reference_data).expect("parse avg rm ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    for row in &rm_ref.relative_magnitude {
        let problem = build_relative_magnitude_flci_problem(
            &input,
            &rm_ref.post_weights,
            row.mbar,
            InferenceConfig::new(0.95),
        )
        .expect("flci problem");

        let result = compute_relative_magnitude_flci(&problem).expect("flci result");
        assert!((result.original_ci.0 - rm_ref.original.lb).abs() < 1e-8);
        assert!((result.original_ci.1 - rm_ref.original.ub).abs() < 1e-8);
        assert!((result.identified_set.0 - row.id_lb).abs() < 1e-6);
        assert!((result.identified_set.1 - row.id_ub).abs() < 1e-6);
        assert!((result.flci.0 - row.lb).abs() < 5e-3);
        assert!((result.flci.1 - row.ub).abs() < 5e-3);
    }
}

#[test]
fn honest_delta_rm_flci_default_matches_explicit_lf_config() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let reference_data =
        fs::read_to_string("tests/honest_did_rm_avg_ref.json").expect("read avg rm ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let rm_ref: HonestRelativeMagnitudeFixture =
        serde_json::from_str(&reference_data).expect("parse avg rm ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    let inference = InferenceConfig::new(0.95);
    let problem = build_relative_magnitude_flci_problem(
        &input,
        &rm_ref.post_weights,
        rm_ref.relative_magnitude[1].mbar,
        inference,
    )
    .expect("flci problem");

    let wrapped = compute_relative_magnitude_flci(&problem).expect("wrapped flci");
    let explicit = compute_relative_magnitude_flci_with_config(
        &problem,
        RelativeMagnitudeConfidenceSetConfig {
            hybrid: RelativeMagnitudeHybrid::LeastFavorable,
            hybrid_kappa: (1.0 - inference.confidence_level) / 20.0,
        },
    )
    .expect("explicit LF flci");

    assert_eq!(wrapped, explicit);
}

#[test]
fn honest_post_functional_flci_average_window_matches_reference_package() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let reference_data =
        fs::read_to_string("tests/honest_did_rm_avg_ref.json").expect("read avg rm ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let rm_ref: HonestRelativeMagnitudeFixture =
        serde_json::from_str(&reference_data).expect("parse avg rm ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    let inference = InferenceConfig::new(0.95);
    let result = assess_honest_event_study_post_functional_flci(
        &input,
        &HonestPostFunctional::AverageWindow {
            start_period: 0,
            end_period: 3,
        },
        rm_ref.relative_magnitude[1].mbar,
        inference,
        0.0,
    )
    .expect("flci wrapper");

    let ref_row = &rm_ref.relative_magnitude[1];
    assert!((result.original_ci.0 - rm_ref.original.lb).abs() < 1e-8);
    assert!((result.original_ci.1 - rm_ref.original.ub).abs() < 1e-8);
    assert!((result.identified_set.0 - ref_row.id_lb).abs() < 1e-6);
    assert!((result.identified_set.1 - ref_row.id_ub).abs() < 1e-6);
    assert!((result.robust_ci.0 - ref_row.lb).abs() < 5e-3);
    assert!((result.robust_ci.1 - ref_row.ub).abs() < 5e-3);
}

#[test]
fn honest_post_functional_flci_default_matches_explicit_lf_config() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();

    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    let inference = InferenceConfig::new(0.95);
    let wrapped = assess_honest_event_study_post_functional_flci(
        &input,
        &HonestPostFunctional::AverageWindow {
            start_period: 0,
            end_period: 3,
        },
        2.0,
        inference,
        0.0,
    )
    .expect("wrapped flci");

    let explicit = assess_honest_event_study_post_functional_flci_with_config(
        &input,
        &HonestPostFunctional::AverageWindow {
            start_period: 0,
            end_period: 3,
        },
        2.0,
        inference,
        0.0,
        RelativeMagnitudeConfidenceSetConfig {
            hybrid: RelativeMagnitudeHybrid::LeastFavorable,
            hybrid_kappa: (1.0 - inference.confidence_level) / 20.0,
        },
    )
    .expect("explicit LF flci");

    assert_eq!(wrapped, explicit);
}

#[test]
fn honest_joint_path_region_single_post_matches_scalar_period() {
    let input = HonestEventStudyInput {
        betahat: vec![0.1, -0.2],
        covariance: vec![vec![0.04, 0.0], vec![0.0, 0.09]],
        pre_periods: vec![-1],
        post_periods: vec![0],
    };
    let inference = InferenceConfig::new(0.95);

    let joint = assess_honest_event_study_joint_path_region(
        &input,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
    )
    .expect("joint path region");
    let scalar = did_methods::assess_honest_event_study_period(
        &input,
        0,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
    )
    .expect("scalar period");

    assert_eq!(joint.points.len(), 1);
    assert!((joint.pointwise_confidence_level - inference.confidence_level).abs() < 1e-12);
    assert_eq!(joint.points[0].post_period, 0);
    assert_eq!(joint.points[0].assessment, scalar);
}

#[test]
fn honest_joint_path_region_default_matches_explicit_lf_config() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();
    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };
    let inference = InferenceConfig::new(0.95);

    let wrapped = assess_honest_event_study_joint_path_region(
        &input,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
    )
    .expect("wrapped joint region");
    let explicit = assess_honest_event_study_joint_path_region_with_config(
        &input,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
        RelativeMagnitudeConfidenceSetConfig {
            hybrid: RelativeMagnitudeHybrid::LeastFavorable,
            hybrid_kappa: (1.0 - inference.confidence_level) / 20.0,
        },
        HonestJointPathConfig::default_for_production(),
    )
    .expect("explicit joint region");

    assert!(
        (wrapped.pointwise_confidence_level - explicit.pointwise_confidence_level).abs() < 1e-12
    );
    assert_eq!(wrapped.points.len(), explicit.points.len());
    for (left, right) in wrapped.points.iter().zip(explicit.points.iter()) {
        assert_eq!(left.post_period, right.post_period);
        assert_eq!(left.assessment, right.assessment);
    }
}

#[test]
fn honest_joint_path_region_gaussian_is_no_wider_than_bonferroni() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();
    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };
    let inference = InferenceConfig::new(0.95);

    let bonf = assess_honest_event_study_joint_path_region_with_config(
        &input,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        HonestJointPathConfig {
            method: HonestJointPathMethod::Bonferroni,
            simulation_draws: 1_000,
            simulation_seed: 42,
        },
    )
    .expect("bonf joint region");

    let gaussian = assess_honest_event_study_joint_path_region_with_config(
        &input,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        HonestJointPathConfig {
            method: HonestJointPathMethod::GaussianSimulated,
            simulation_draws: 10_000,
            simulation_seed: 42,
        },
    )
    .expect("gaussian joint region");

    assert_eq!(bonf.points.len(), gaussian.points.len());
    for (bonf_point, gaussian_point) in bonf.points.iter().zip(gaussian.points.iter()) {
        let bonf_width = bonf_point.assessment.robust_ci.1 - bonf_point.assessment.robust_ci.0;
        let gaussian_width =
            gaussian_point.assessment.robust_ci.1 - gaussian_point.assessment.robust_ci.0;
        assert!(
            gaussian_width <= bonf_width + 1e-8,
            "expected gaussian joint region to be no wider than Bonferroni for post period {}",
            bonf_point.post_period
        );
    }
}

#[test]
fn honest_delta_rm_supports_signed_linear_functionals() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();
    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    let mut post_weights = vec![0.0; input.num_post_periods()];
    post_weights[0] = 1.0;
    post_weights[1] = -1.0;
    let inference = InferenceConfig::new(0.95);

    let identified = compute_relative_magnitude_identified_set(&input, &post_weights, 1.0)
        .expect("identified set for signed post_weights");
    let conditional = compute_relative_magnitude_confidence_set_with_config(
        &input,
        &post_weights,
        1.0,
        inference,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
    )
    .expect("conditional cs for signed post_weights");

    assert!(identified.lb.is_finite() && identified.ub.is_finite());
    assert!(conditional.lb.is_finite() && conditional.ub.is_finite());
    assert!(identified.lb <= identified.ub);
    assert!(conditional.lb <= conditional.ub);
}

#[test]
fn honest_directional_region_basis_matches_joint_path_points() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();
    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };
    let inference = InferenceConfig::new(0.95);
    let joint_config = HonestJointPathConfig {
        method: HonestJointPathMethod::GaussianSimulated,
        simulation_draws: 8_000,
        simulation_seed: 42,
    };

    let joint = assess_honest_event_study_joint_path_region_with_config(
        &input,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        joint_config,
    )
    .expect("joint path region");

    let directions = (0..input.num_post_periods())
        .map(|idx| {
            let mut l = vec![0.0; input.num_post_periods()];
            l[idx] = 1.0;
            l
        })
        .collect::<Vec<_>>();
    let directional = assess_honest_event_study_directional_region_with_config(
        &input,
        &directions,
        HonestSensitivity::RelativeMagnitude(1.0),
        inference,
        0.0,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        joint_config,
    )
    .expect("directional region");

    assert!(
        (joint.pointwise_confidence_level - directional.pointwise_confidence_level).abs() < 1e-12
    );
    assert_eq!(joint.points.len(), directional.points.len());
    for (joint_point, directional_point) in joint.points.iter().zip(directional.points.iter()) {
        assert_eq!(directional_point.assessment, joint_point.assessment);
    }
}

#[test]
fn honest_optimization_surface_pairwise_only_builds_expected_direction_count() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.1, -0.1, 0.2, -0.2],
        covariance: vec![
            vec![0.04, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.04, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.09, 0.01, 0.0],
            vec![0.0, 0.0, 0.01, 0.09, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.09],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1, 2],
    };
    let inference = InferenceConfig::new(0.95);
    let region = assess_honest_event_study_optimization_surface_region_with_config(
        &input,
        HonestSensitivity::Smoothness(0.5),
        inference,
        0.0,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        HonestJointPathConfig {
            method: HonestJointPathMethod::Bonferroni,
            simulation_draws: 1_000,
            simulation_seed: 7,
        },
        HonestOptimizationSurfaceConfig {
            include_basis: false,
            include_pairwise_contrasts: true,
            random_unit_directions: 0,
            random_seed: 7,
        },
    )
    .expect("optimization surface region");

    assert_eq!(region.points.len(), 3);
}

#[test]
fn honest_optimization_surface_adaptive_matches_full_grid_when_forced_to_max_random() {
    let input = HonestEventStudyInput {
        betahat: vec![0.0, 0.1, -0.1, 0.2, -0.2],
        covariance: vec![
            vec![0.04, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.04, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.09, 0.01, 0.0],
            vec![0.0, 0.0, 0.01, 0.09, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.09],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1, 2],
    };
    let inference = InferenceConfig::new(0.95);
    let surface_config = HonestOptimizationSurfaceConfig {
        include_basis: true,
        include_pairwise_contrasts: true,
        random_unit_directions: 40,
        random_seed: 11,
    };
    let joint_config = HonestJointPathConfig {
        method: HonestJointPathMethod::Bonferroni,
        simulation_draws: 1_000,
        simulation_seed: 17,
    };

    let full = assess_honest_event_study_optimization_surface_region_with_config(
        &input,
        HonestSensitivity::Smoothness(0.5),
        inference,
        0.0,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        joint_config,
        surface_config,
    )
    .expect("full optimization surface region");

    let adaptive = assess_honest_event_study_optimization_surface_region_adaptive_with_config(
        &input,
        HonestSensitivity::Smoothness(0.5),
        inference,
        0.0,
        HonestOptimizationSurfaceAdaptiveRunConfig {
            relative_magnitude: RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
            joint: joint_config,
            surface: surface_config,
            adaptive: HonestOptimizationSurfaceAdaptiveConfig {
                random_batch_size: 10,
                min_random_for_convergence: 10_000,
                pointwise_tolerance: 0.0,
                max_iterations: 10,
            },
        },
    )
    .expect("adaptive optimization surface region");

    assert!((full.pointwise_confidence_level - adaptive.pointwise_confidence_level).abs() < 1e-12);
    assert_eq!(full.points.len(), adaptive.points.len());
    assert_eq!(full.diagnostics.random_direction_count, 40);
    assert_eq!(adaptive.diagnostics.random_direction_count, 40);
    assert!(adaptive.diagnostics.iterations_run >= 1);
}

#[test]
fn honest_delta_rm_multi_flci_single_functional_matches_scalar_flci() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();
    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };
    let inference = InferenceConfig::new(0.95);
    let mut post_weights = vec![0.0; input.num_post_periods()];
    post_weights[0] = 1.0;
    let problem = build_relative_magnitude_flci_problem(&input, &post_weights, 1.0, inference)
        .expect("scalar flci problem");
    let scalar = compute_relative_magnitude_flci(&problem).expect("scalar flci");

    let multi_problem = build_relative_magnitude_multi_flci_problem(
        &input,
        std::slice::from_ref(&post_weights),
        1.0,
        inference,
    )
    .expect("multi flci problem");
    let multi = compute_relative_magnitude_multi_flci(&multi_problem).expect("multi flci");

    assert_eq!(multi.points.len(), 1);
    assert!((multi.pointwise_confidence_level - inference.confidence_level).abs() < 1e-12);
    assert!((multi.points[0].flci.0 - scalar.flci.0).abs() < 1e-10);
    assert!((multi.points[0].flci.1 - scalar.flci.1).abs() < 1e-10);
}

#[test]
fn honest_delta_rm_multi_flci_gaussian_is_no_wider_than_bonferroni() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();
    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };
    let inference = InferenceConfig::new(0.95);
    let mut first = vec![0.0; input.num_post_periods()];
    first[0] = 1.0;
    let mut second = vec![0.0; input.num_post_periods()];
    second[1] = 1.0;
    let mut diff = vec![0.0; input.num_post_periods()];
    diff[0] = 1.0;
    diff[1] = -1.0;
    let post_weight_sets = vec![first, second, diff];
    let problem =
        build_relative_magnitude_multi_flci_problem(&input, &post_weight_sets, 1.0, inference)
            .expect("multi flci problem");

    let bonf = compute_relative_magnitude_multi_flci_with_config(
        &problem,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        HonestJointPathConfig {
            method: HonestJointPathMethod::Bonferroni,
            simulation_draws: 1_000,
            simulation_seed: 42,
        },
    )
    .expect("bonf multi flci");
    let gaussian = compute_relative_magnitude_multi_flci_with_config(
        &problem,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        HonestJointPathConfig {
            method: HonestJointPathMethod::GaussianSimulated,
            simulation_draws: 10_000,
            simulation_seed: 42,
        },
    )
    .expect("gaussian multi flci");

    assert_eq!(bonf.points.len(), gaussian.points.len());
    for (bonf_point, gaussian_point) in bonf.points.iter().zip(gaussian.points.iter()) {
        let bonf_width = bonf_point.flci.1 - bonf_point.flci.0;
        let gaussian_width = gaussian_point.flci.1 - gaussian_point.flci.0;
        assert!(gaussian_width <= bonf_width + 1e-8);
    }
}

#[test]
fn honest_post_functional_multi_flci_wrapper_matches_direct_problem_path() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();
    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };
    let inference = InferenceConfig::new(0.95);
    let functionals = vec![
        HonestPostFunctional::Period(0),
        HonestPostFunctional::AverageWindow {
            start_period: 0,
            end_period: 1,
        },
    ];
    let config = RelativeMagnitudeConfidenceSetConfig::from_inference(inference);
    let joint = HonestJointPathConfig {
        method: HonestJointPathMethod::Bonferroni,
        simulation_draws: 1_000,
        simulation_seed: 99,
    };

    let wrapped = assess_honest_event_study_post_functional_multi_flci_with_config(
        &input,
        &functionals,
        1.0,
        inference,
        config,
        joint,
    )
    .expect("wrapped multi flci");

    let mut l_period = vec![0.0; input.num_post_periods()];
    l_period[0] = 1.0;
    let mut l_avg = vec![0.0; input.num_post_periods()];
    l_avg[0] = 0.5;
    l_avg[1] = 0.5;
    let direct_problem =
        build_relative_magnitude_multi_flci_problem(&input, &[l_period, l_avg], 1.0, inference)
            .expect("direct problem");
    let direct = compute_relative_magnitude_multi_flci_with_config(&direct_problem, config, joint)
        .expect("direct multi flci");

    assert!((wrapped.pointwise_confidence_level - direct.pointwise_confidence_level).abs() < 1e-12);
    assert_eq!(wrapped.points.len(), direct.points.len());
    for (left, right) in wrapped.points.iter().zip(direct.points.iter()) {
        assert!((left.flci.0 - right.flci.0).abs() < 1e-10);
        assert!((left.flci.1 - right.flci.1).abs() < 1e-10);
    }
}

#[test]
fn honest_joint_path_region_bonferroni_matches_r_fixture() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let fixture_data =
        fs::read_to_string("tests/honest_did_joint_path_ref.json").expect("read joint ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let fixture: HonestJointPathRef = serde_json::from_str(&fixture_data).expect("parse joint ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();
    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    assert_eq!(fixture.post_weight_sets.len(), input.num_post_periods());
    let inference = InferenceConfig::new(1.0 - fixture.meta.alpha_joint);
    let rust = assess_honest_event_study_joint_path_region_with_config(
        &input,
        HonestSensitivity::RelativeMagnitude(fixture.meta.mbar),
        inference,
        0.0,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        HonestJointPathConfig {
            method: HonestJointPathMethod::Bonferroni,
            simulation_draws: 1_000,
            simulation_seed: 123,
        },
    )
    .expect("joint path region");

    assert!(
        (rust.pointwise_confidence_level - fixture.meta.pointwise_confidence_level).abs() < 1e-12
    );
    assert_eq!(rust.points.len(), fixture.points.len());
    for (rust_point, fixture_point) in rust.points.iter().zip(fixture.points.iter()) {
        assert_eq!(rust_point.post_period, fixture_point.post_period);
        assert!((rust_point.assessment.robust_ci.0 - fixture_point.lb).abs() < 1e-2);
        assert!((rust_point.assessment.robust_ci.1 - fixture_point.ub).abs() < 1e-2);
    }
}

#[test]
fn honest_directional_region_bonferroni_matches_r_fixture() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let fixture_data =
        fs::read_to_string("tests/honest_did_directional_ref.json").expect("read directional ref");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let fixture: HonestDirectionalRef =
        serde_json::from_str(&fixture_data).expect("parse directional ref");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();
    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    let directions = fixture
        .directions
        .iter()
        .map(|entry| entry.post_weights.clone())
        .collect::<Vec<_>>();
    let inference = InferenceConfig::new(1.0 - fixture.meta.alpha_joint);
    let rust = assess_honest_event_study_directional_region_with_config(
        &input,
        &directions,
        HonestSensitivity::RelativeMagnitude(fixture.meta.mbar),
        inference,
        0.0,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        HonestJointPathConfig {
            method: HonestJointPathMethod::Bonferroni,
            simulation_draws: 1_000,
            simulation_seed: 123,
        },
    )
    .expect("directional region");

    assert!(
        (rust.pointwise_confidence_level - fixture.meta.pointwise_confidence_level).abs() < 1e-12
    );
    assert_eq!(rust.points.len(), fixture.points.len());
    for ((rust_point, fixture_point), fixture_direction) in rust
        .points
        .iter()
        .zip(fixture.points.iter())
        .zip(fixture.directions.iter())
    {
        assert_eq!(fixture_point.name, fixture_direction.name);
        assert!((rust_point.assessment.robust_ci.0 - fixture_point.lb).abs() < 1e-2);
        assert!((rust_point.assessment.robust_ci.1 - fixture_point.ub).abs() < 1e-2);
    }
}

#[test]
fn honest_multi_flci_scaffold_fixture_drives_joint_solver() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let fixture_data = fs::read_to_string("tests/honest_did_multi_flci_scaffold.json")
        .expect("read multi flci scaffold");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let fixture: HonestMultiFlciScaffold =
        serde_json::from_str(&fixture_data).expect("parse multi scaffold");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();
    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };
    assert!(!fixture.post_weight_sets.is_empty());
    assert!(
        fixture
            .post_weight_sets
            .iter()
            .all(|l| l.len() == input.num_post_periods())
    );

    let inference = InferenceConfig::new(0.95);
    let problem = build_relative_magnitude_multi_flci_problem(
        &input,
        &fixture.post_weight_sets,
        1.0,
        inference,
    )
    .expect("build multi flci problem");
    let result = compute_relative_magnitude_multi_flci(&problem).expect("solve multi flci");

    assert_eq!(result.points.len(), fixture.post_weight_sets.len());
    assert!(
        result
            .points
            .iter()
            .all(|point| point.flci.0.is_finite() && point.flci.1.is_finite())
    );
}

#[test]
fn honest_joint_gaussian_calibration_matches_r_scaffold() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let fixture_data = fs::read_to_string("tests/honest_did_gaussian_scaffold.json")
        .expect("read gaussian scaffold");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let fixture: HonestGaussianScaffold =
        serde_json::from_str(&fixture_data).expect("parse gaussian scaffold");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();
    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    let inference = InferenceConfig::new(fixture.meta.confidence_level);
    let region = assess_honest_event_study_joint_path_region_with_config(
        &input,
        HonestSensitivity::RelativeMagnitude(fixture.mbar),
        inference,
        0.0,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        HonestJointPathConfig {
            method: HonestJointPathMethod::GaussianSimulated,
            simulation_draws: 200_000,
            simulation_seed: 20_260_310,
        },
    )
    .expect("gaussian joint region");

    assert!(
        (region.calibrated_max_t_critical_value - fixture.joint.calibrated_max_t_critical_value)
            .abs()
            < 0.08
    );
    assert!(
        (region.pointwise_confidence_level - fixture.joint.pointwise_confidence_level).abs() < 0.01
    );
    assert_eq!(region.points.len(), fixture.joint_points.len());
    for (rust_point, ref_point) in region.points.iter().zip(fixture.joint_points.iter()) {
        assert_eq!(rust_point.post_period, ref_point.post_period);
        assert!((rust_point.assessment.robust_ci.0 - ref_point.lb).abs() < 2e-2);
        assert!((rust_point.assessment.robust_ci.1 - ref_point.ub).abs() < 2e-2);
    }
}

#[test]
fn honest_directional_gaussian_calibration_matches_r_scaffold() {
    let input_data = fs::read_to_string("tests/honest_did_ref.json").expect("read input ref");
    let fixture_data = fs::read_to_string("tests/honest_did_gaussian_scaffold.json")
        .expect("read gaussian scaffold");
    let input_ref: HonestRef = serde_json::from_str(&input_data).expect("parse input ref");
    let fixture: HonestGaussianScaffold =
        serde_json::from_str(&fixture_data).expect("parse gaussian scaffold");

    let pre_periods: Vec<i32> = input_ref
        .pre_indices
        .iter()
        .map(|idx| -i32::try_from(*idx).expect("pre index fits"))
        .collect();
    let post_periods: Vec<i32> = (0..input_ref.post_indices.len())
        .map(|idx| i32::try_from(idx).expect("post index fits"))
        .collect();
    let input = HonestEventStudyInput {
        betahat: input_ref.betahat,
        covariance: input_ref.sigma,
        pre_periods,
        post_periods,
    };

    let inference = InferenceConfig::new(fixture.meta.confidence_level);
    let region = assess_honest_event_study_directional_region_with_config(
        &input,
        &fixture.directions,
        HonestSensitivity::RelativeMagnitude(fixture.mbar),
        inference,
        0.0,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        HonestJointPathConfig {
            method: HonestJointPathMethod::GaussianSimulated,
            simulation_draws: 200_000,
            simulation_seed: 20_260_310,
        },
    )
    .expect("gaussian directional region");

    assert!(
        (region.calibrated_max_t_critical_value
            - fixture.directional.calibrated_max_t_critical_value)
            .abs()
            < 0.08
    );
    assert!(
        (region.pointwise_confidence_level - fixture.directional.pointwise_confidence_level).abs()
            < 0.01
    );
    assert_eq!(region.points.len(), fixture.directional_points.len());
    for (rust_point, ref_point) in region.points.iter().zip(fixture.directional_points.iter()) {
        assert!((rust_point.assessment.robust_ci.0 - ref_point.lb).abs() < 2e-2);
        assert!((rust_point.assessment.robust_ci.1 - ref_point.ub).abs() < 2e-2);
    }
}
