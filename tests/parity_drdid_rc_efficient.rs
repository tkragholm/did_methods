//! Parity of the locally efficient repeated-cross-section estimator against
//! `DRDID::drdid_rc` 1.3.0, which is what `did::att_gt(est_method = "dr")` calls
//! on repeated cross sections.

#![allow(clippy::float_cmp)]

use std::fs;

use did_methods::{DrDidConfig, DrDidRepeatedObservation, estimate_drdid_repeated_efficient};
use serde::Deserialize;

#[derive(Deserialize)]
struct Cell {
    group: i32,
    time: i32,
    att: f64,
    se: f64,
    inffunc: Vec<f64>,
    y: Vec<f64>,
    post: Vec<f64>,
    d: Vec<f64>,
    lpop: Vec<f64>,
}

fn rows(cell: &Cell) -> Vec<DrDidRepeatedObservation> {
    (0..cell.y.len())
        .map(|i| DrDidRepeatedObservation {
            treated: cell.d[i] > 0.5,
            post_period: cell.post[i] > 0.5,
            outcome: cell.y[i],
            weight: 1.0,
            // The intercept is NOT passed here. Unlike `estimate_drdid_panel`,
            // whose covariates are a finished design matrix, the repeated
            // estimators prepend one themselves (`feature_count = covariates + 1`).
            // R was given cbind(1, lpop) to match.
            covariates: vec![cell.lpop[i]],
        })
        .collect()
}

#[test]
fn locally_efficient_rc_matches_drdid_rc() {
    let raw =
        fs::read_to_string("tests/drdid_rc_efficient_ref.json").expect("read efficient fixture");
    let cells: Vec<Cell> = serde_json::from_str(&raw).expect("parse efficient fixture");
    assert_eq!(cells.len(), 12);

    let mut worst_att = 0.0_f64;
    let mut worst_se = 0.0_f64;
    let mut worst_influence = 0.0_f64;

    for cell in &cells {
        let fit = estimate_drdid_repeated_efficient(
            &rows(cell),
            DrDidConfig::builder().ridge(0.0).build(),
        )
        .unwrap_or_else(|e| panic!("cell ({}, {}): {e}", cell.group, cell.time));

        worst_att = worst_att.max((fit.att - cell.att).abs());
        worst_se = worst_se.max((fit.se - cell.se).abs());
        assert_eq!(fit.influence_function.len(), cell.inffunc.len());
        for (ours, theirs) in fit.influence_function.iter().zip(&cell.inffunc) {
            worst_influence = worst_influence.max((ours - theirs).abs());
        }
    }

    println!(
        "worst vs DRDID::drdid_rc -- att {worst_att:e}, se {worst_se:e}, \
         influence {worst_influence:e}"
    );
    assert!(worst_att < 1e-9, "worst att deviation {worst_att:e}");
    assert!(worst_se < 1e-9, "worst se deviation {worst_se:e}");
    assert!(
        worst_influence < 1e-7,
        "worst influence deviation {worst_influence:e}"
    );
}

/// The two RC estimators agree on the point estimate and disagree on the
/// standard error, which is the whole reason both exist.
#[test]
fn the_two_rc_estimators_differ_only_in_precision() {
    let raw =
        fs::read_to_string("tests/drdid_rc_efficient_ref.json").expect("read efficient fixture");
    let cells: Vec<Cell> = serde_json::from_str(&raw).expect("parse efficient fixture");

    let mut differed = 0_usize;
    for cell in &cells {
        let observations = rows(cell);
        let config = DrDidConfig::builder().build();
        let efficient = estimate_drdid_repeated_efficient(&observations, config).expect("eff");
        let traditional = did_methods::estimate_drdid_repeated_cross_section(&observations, config)
            .expect("trad");

        assert!(
            (efficient.att - traditional.att).abs() < 1e-8,
            "({}, {}) point estimates should agree: {} vs {}",
            cell.group,
            cell.time,
            efficient.att,
            traditional.att
        );
        if (efficient.se - traditional.se).abs() / traditional.se > 1e-3 {
            differed += 1;
        }
    }
    assert!(
        differed >= cells.len() / 2,
        "the two estimators gave near-identical standard errors in all but {differed} cells; \
         one of them may have been routed to the other"
    );
}
