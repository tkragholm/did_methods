//! Parity of the improved doubly robust estimators against
//! `DRDID::drdid_imp_rc` and `DRDID::drdid_imp_panel` 1.3.0, which fit the
//! propensity score by Sant'Anna and Zhao's calibration (`pscore.cal`).
//! Fixtures from `tests/generate_drdid_improved_reference.R`.

use std::fs;

use did_methods::{
    DrDidConfig, DrDidObservation, DrDidRepeatedObservation, estimate_drdid_improved_panel,
    estimate_drdid_improved_repeated_cross_section,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    rc: Vec<RcCase>,
    panel: Vec<PanelCase>,
}

#[derive(Deserialize)]
struct RcData {
    y: Vec<f64>,
    post: Vec<f64>,
    d: Vec<f64>,
    x: Vec<Vec<f64>>,
    weights: Vec<f64>,
}

#[derive(Deserialize)]
struct RcCase {
    name: String,
    att: f64,
    se: f64,
    inffunc: Vec<f64>,
    data: RcData,
}

#[derive(Deserialize)]
struct PanelData {
    y1: Vec<f64>,
    y0: Vec<f64>,
    d: Vec<f64>,
    x: Vec<Vec<f64>>,
    weights: Vec<f64>,
}

#[derive(Deserialize)]
struct PanelCase {
    name: String,
    att: f64,
    se: f64,
    inffunc: Vec<f64>,
    data: PanelData,
}

fn fixture() -> Fixture {
    let raw = fs::read_to_string("tests/drdid_improved_ref.json").expect("read improved fixture");
    serde_json::from_str(&raw).expect("parse improved fixture")
}

fn worst(ours: &[f64], theirs: &[f64]) -> f64 {
    assert_eq!(ours.len(), theirs.len());
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn config() -> DrDidConfig {
    DrDidConfig::builder().ridge(0.0).build()
}

#[test]
fn improved_rc_matches_drdid_imp_rc() {
    for case in fixture().rc {
        let d = &case.data;
        // The repeated estimators prepend the intercept themselves.
        let rows: Vec<DrDidRepeatedObservation> = (0..d.y.len())
            .map(|i| DrDidRepeatedObservation {
                treated: d.d[i] > 0.5,
                post_period: d.post[i] > 0.5,
                outcome: d.y[i],
                weight: d.weights[i],
                covariates: d.x[i].clone(),
            })
            .collect();
        let fit = estimate_drdid_improved_repeated_cross_section(&rows, config())
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let (att, se) = ((fit.att - case.att).abs(), (fit.se - case.se).abs());
        let influence = worst(&fit.influence_function, &case.inffunc);
        println!(
            "{}: att {att:e}, se {se:e}, influence {influence:e}",
            case.name
        );
        assert!(att < 1e-8 && se < 1e-8 && influence < 1e-6, "{}", case.name);
    }
}

#[test]
fn improved_panel_matches_drdid_imp_panel() {
    for case in fixture().panel {
        let d = &case.data;
        // The panel estimator takes a finished design matrix, intercept first.
        let rows: Vec<DrDidObservation> = (0..d.y1.len())
            .map(|i| DrDidObservation {
                treated: d.d[i] > 0.5,
                delta_outcome: d.y1[i] - d.y0[i],
                weight: d.weights[i],
                covariates: std::iter::once(1.0).chain(d.x[i].iter().copied()).collect(),
            })
            .collect();
        let fit = estimate_drdid_improved_panel(&rows, config())
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let (att, se) = ((fit.att - case.att).abs(), (fit.se - case.se).abs());
        let influence = worst(&fit.influence_function, &case.inffunc);
        println!(
            "{}: att {att:e}, se {se:e}, influence {influence:e}",
            case.name
        );
        assert!(att < 1e-8 && se < 1e-8 && influence < 1e-6, "{}", case.name);
    }
}
