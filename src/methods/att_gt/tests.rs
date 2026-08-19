use std::collections::BTreeMap;

use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::inference::{quantile_sorted, standard_error_from_influence};
use crate::types::{
    AttGtAggregationConfig, AttGtBandConfig, AttGtBandError, AttGtBandEstimate, BasePeriod,
    DrDidConfig,
};

use super::*;

fn row(g: Option<i32>, t: i32, y: f64) -> AttGtObservation {
    AttGtObservation::new(g, t, y)
}

fn panel_row(id: i64, g: Option<i32>, t: i32, y: f64) -> AttGtObservation {
    AttGtObservation::with_unit_id(id, g, t, y)
}

fn reference_bands_sorted_quantile(
    estimates: &[AttGtEstimate],
    config: AttGtBandConfig,
) -> Vec<AttGtBandEstimate> {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut maxima = Vec::with_capacity(config.reps);
    let k = estimates.len();
    for _ in 0..config.reps {
        let mut max_abs = 0.0_f64;
        for _ in 0..k {
            let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
            let u2 = rng.random::<f64>();
            let z = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
            max_abs = max_abs.max(z.abs());
        }
        maxima.push(max_abs);
    }
    maxima.sort_by(f64::total_cmp);
    let alpha = (1.0 - config.confidence_level.confidence_level).clamp(0.0, 1.0);
    let critical = quantile_sorted(&maxima, 1.0 - alpha);

    estimates
        .iter()
        .map(|estimate| {
            let margin = critical * estimate.se;
            AttGtBandEstimate {
                group: estimate.group,
                time: estimate.time,
                event_time: estimate.event_time,
                att: estimate.att,
                se: estimate.se,
                band_low: estimate.att - margin,
                band_high: estimate.att + margin,
            }
        })
        .collect()
}

fn reference_bands_with_influence_sorted_quantile(
    estimates: &[AttGtEstimate],
    influence_functions: &[Vec<f64>],
    config: AttGtBandConfig,
) -> Vec<AttGtBandEstimate> {
    let n = influence_functions[0].len();
    let se_if = influence_functions
        .iter()
        .map(|influence| standard_error_from_influence(influence))
        .collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut maxima = Vec::with_capacity(config.reps);
    let mut signs = vec![1.0; n];
    for _ in 0..config.reps {
        for sign in &mut signs {
            *sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        }
        let mut max_abs = 0.0_f64;
        for influence in influence_functions {
            let numerator = influence
                .iter()
                .zip(signs.iter())
                .map(|(value, sign)| value * sign)
                .sum::<f64>();
            let denominator = influence
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            let t_stat = numerator / denominator;
            max_abs = max_abs.max(t_stat.abs());
        }
        maxima.push(max_abs);
    }
    maxima.sort_by(f64::total_cmp);
    let alpha = (1.0 - config.confidence_level.confidence_level).clamp(0.0, 1.0);
    let critical = quantile_sorted(&maxima, 1.0 - alpha);

    estimates
        .iter()
        .zip(se_if.iter())
        .map(|(estimate, se_if)| {
            let margin = critical * *se_if;
            AttGtBandEstimate {
                group: estimate.group,
                time: estimate.time,
                event_time: estimate.event_time,
                att: estimate.att,
                se: *se_if,
                band_low: estimate.att - margin,
                band_high: estimate.att + margin,
            }
        })
        .collect()
}

fn row_dr(g: Option<i32>, t: i32, y: f64, x: f64) -> AttGtDrObservation {
    AttGtDrObservation {
        covariates: vec![x],
        ..AttGtDrObservation::new(g, t, y)
    }
}

#[test]
fn estimates_group_time_att_with_never_treated_controls() {
    let mut rows = Vec::new();
    for _ in 0..50 {
        for t in 1..=4 {
            let t_f = f64::from(t);
            rows.push(row(None, t, 10.0 + t_f));
            rows.push(row(Some(2), t, 10.0 + t_f + if t >= 2 { 2.0 } else { 0.0 }));
            rows.push(row(Some(3), t, 10.0 + t_f + if t >= 3 { 1.0 } else { 0.0 }));
        }
    }

    let estimates = estimate_att_gt(
        &rows,
        AttGtConfig {
            base_period: BasePeriod::Universal,
            ..AttGtConfig::default()
        },
    )
    .expect("estimate att(g,t)");
    let att_map = estimates
        .iter()
        .map(|est| ((est.group, est.time), est.att))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(att_map.len(), 6);
    assert!((att_map[&(2, 2)] - 2.0).abs() < 1e-12);
    assert!((att_map[&(2, 3)] - 2.0).abs() < 1e-12);
    assert!((att_map[&(2, 4)] - 2.0).abs() < 1e-12);
    assert!(att_map[&(3, 1)].abs() < 1e-12);
    assert!((att_map[&(3, 3)] - 1.0).abs() < 1e-12);
    assert!((att_map[&(3, 4)] - 1.0).abs() < 1e-12);
}

#[test]
fn rejects_missing_never_treated_group() {
    let rows = vec![row(Some(2), 1, 10.0), row(Some(2), 2, 12.0)];
    let err = estimate_att_gt(&rows, AttGtConfig::default()).expect_err("must fail");
    assert_eq!(err, AttGtError::MissingNeverTreatedGroup);
}

#[test]
fn strict_mode_errors_on_missing_cells() {
    let rows = vec![
        row(None, 1, 11.0),
        row(None, 2, 12.0),
        row(Some(2), 1, 10.0),
        // missing treated_post at t=2
    ];
    let err = estimate_att_gt(
        &rows,
        AttGtConfig {
            base_period: BasePeriod::Universal,
            skip_incomplete_pairs: false,
            ..AttGtConfig::default()
        },
    )
    .expect_err("must fail");
    assert!(matches!(err, AttGtError::MissingCell { cell, .. } if cell == "treated_post"));
}

#[test]
fn estimates_group_time_att_with_not_yet_treated_controls() {
    let mut rows = Vec::new();
    for _ in 0..50 {
        for t in 1..=4 {
            let t_f = f64::from(t);
            rows.push(row(Some(2), t, 10.0 + t_f + if t >= 2 { 2.0 } else { 0.0 }));
            rows.push(row(Some(3), t, 10.0 + t_f + if t >= 3 { 1.0 } else { 0.0 }));
            rows.push(row(Some(4), t, 10.0 + t_f));
        }
    }
    let estimates = estimate_att_gt(
        &rows,
        AttGtConfig {
            base_period: BasePeriod::Universal,
            comparison_group: ComparisonGroup::NotYetTreated,
            ..AttGtConfig::default()
        },
    )
    .expect("estimate att(g,t)");
    let att_map = estimates
        .iter()
        .map(|est| ((est.group, est.time), est.att))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(att_map.len(), 6);
    assert!((att_map[&(2, 2)] - 2.0).abs() < 1e-12);
    assert!((att_map[&(2, 3)] - 2.0).abs() < 1e-12);
    assert!((att_map[&(3, 1)] - (2.0 / 3.0)).abs() < 1e-12);
    assert!((att_map[&(3, 3)] - 1.0).abs() < 1e-12);
    assert!((att_map[&(4, 1)] - 1.0).abs() < 1e-12);
    assert!((att_map[&(4, 2)] - 0.5).abs() < 1e-12);
}

#[test]
fn uses_panel_unit_differences_for_standard_errors_when_ids_are_present() {
    let rows = vec![
        panel_row(1, Some(2), 1, 10.0),
        panel_row(1, Some(2), 2, 13.0),
        panel_row(2, Some(2), 1, 8.0),
        panel_row(2, Some(2), 2, 12.0),
        panel_row(3, None, 1, 7.0),
        panel_row(3, None, 2, 8.0),
        panel_row(4, None, 1, 9.0),
        panel_row(4, None, 2, 11.0),
    ];

    let estimates = estimate_att_gt(&rows, AttGtConfig::default()).expect("panel att(g,t)");
    let estimate = estimates
        .iter()
        .find(|estimate| estimate.group == 2 && estimate.time == 2)
        .expect("g=2,t=2 estimate");

    assert!((estimate.att - 2.0).abs() < 1e-12);
    assert!((estimate.se - 0.5).abs() < 1e-12);
}

#[test]
fn aggregation_helpers_work() {
    let estimates = vec![
        AttGtEstimate {
            group: 2,
            time: 2,
            event_time: 0,
            att: 2.0,
            se: 0.1,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 10,
            control_n: 10,
            total_weight: 20.0,
        },
        AttGtEstimate {
            group: 2,
            time: 3,
            event_time: 1,
            att: 2.0,
            se: 0.1,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 10,
            control_n: 10,
            total_weight: 20.0,
        },
        AttGtEstimate {
            group: 2,
            time: 4,
            event_time: 2,
            att: 2.0,
            se: 0.1,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 10,
            control_n: 10,
            total_weight: 20.0,
        },
        AttGtEstimate {
            group: 3,
            time: 3,
            event_time: 0,
            att: 1.0,
            se: 0.1,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 10,
            control_n: 10,
            total_weight: 20.0,
        },
        AttGtEstimate {
            group: 3,
            time: 4,
            event_time: 1,
            att: 1.0,
            se: 0.1,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 10,
            control_n: 10,
            total_weight: 20.0,
        },
    ];

    let cfg = AttGtAggregationConfig::default();
    let by_event = aggregate_att_gt_event_time(&estimates, cfg).expect("event aggregation");
    let event_map = by_event
        .into_iter()
        .map(|entry| (entry.event_time, entry.summary.estimate))
        .collect::<BTreeMap<_, _>>();
    assert!((event_map[&0] - 1.5).abs() < 1e-12);
    assert!((event_map[&1] - 1.5).abs() < 1e-12);
    assert!((event_map[&2] - 2.0).abs() < 1e-12);

    let by_group = aggregate_att_gt_by_cohort(&estimates, cfg).expect("cohort aggregation");
    let group_map = by_group
        .into_iter()
        .map(|entry| (entry.group, entry.summary.estimate))
        .collect::<BTreeMap<_, _>>();
    assert!((group_map[&2] - 2.0).abs() < 1e-12);
    assert!((group_map[&3] - 1.0).abs() < 1e-12);

    let by_time = aggregate_att_gt_by_calendar_time(&estimates, cfg).expect("calendar aggregation");
    let time_map = by_time
        .into_iter()
        .map(|entry| (entry.time, entry.summary.estimate))
        .collect::<BTreeMap<_, _>>();
    assert!((time_map[&2] - 2.0).abs() < 1e-12);
    assert!((time_map[&3] - 1.5).abs() < 1e-12);
    assert!((time_map[&4] - 1.5).abs() < 1e-12);

    let overall = aggregate_att_gt_overall(&estimates, cfg).expect("overall aggregation");
    assert!((overall.estimate - 1.6).abs() < 1e-12);
}

#[test]
fn estimates_dr_group_time_att_with_never_treated_controls() {
    let mut rows = Vec::new();
    for _ in 0..40 {
        for t in 1..=4 {
            let t_f = f64::from(t);
            for x in [0.0, 1.0] {
                let base = 0.5f64.mul_add(x, 10.0 + t_f);
                rows.push(row_dr(None, t, base, x));
                rows.push(row_dr(Some(2), t, base + if t >= 2 { 2.0 } else { 0.0 }, x));
                rows.push(row_dr(Some(3), t, base + if t >= 3 { 1.0 } else { 0.0 }, x));
            }
        }
    }

    let estimates = estimate_att_gt_dr(
        &rows,
        AttGtDrConfig {
            att_gt: AttGtConfig {
                base_period: BasePeriod::Universal,
                ..AttGtConfig::default()
            },
            drdid: DrDidConfig::builder()
                .bootstrap_reps(49)
                .bootstrap_seed(101)
                .build(),
        },
    )
    .expect("dr att(g,t)");
    let att_map = estimates
        .iter()
        .map(|est| ((est.group, est.time), est.att))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(att_map.len(), 6);
    assert!((att_map[&(2, 2)] - 2.0).abs() < 0.2);
    assert!((att_map[&(2, 3)] - 2.0).abs() < 0.2);
    assert!((att_map[&(2, 4)] - 2.0).abs() < 0.2);
    assert!((att_map[&(3, 3)] - 1.0).abs() < 0.2);
    assert!((att_map[&(3, 4)] - 1.0).abs() < 0.2);
}

#[test]
fn estimates_dr_group_time_att_with_not_yet_treated_controls() {
    let mut rows = Vec::new();
    for _ in 0..40 {
        for t in 1..=4 {
            let t_f = f64::from(t);
            for x in [0.0, 1.0] {
                let base = 0.4f64.mul_add(x, 10.0 + t_f);
                rows.push(row_dr(Some(2), t, base + if t >= 2 { 2.0 } else { 0.0 }, x));
                rows.push(row_dr(Some(3), t, base + if t >= 3 { 1.0 } else { 0.0 }, x));
                rows.push(row_dr(Some(4), t, base, x));
            }
        }
    }

    let estimates = estimate_att_gt_dr(
        &rows,
        AttGtDrConfig {
            att_gt: AttGtConfig::builder()
                .comparison_group(ComparisonGroup::NotYetTreated)
                .base_period(BasePeriod::Universal)
                .build(),
            drdid: DrDidConfig::builder()
                .bootstrap_reps(49)
                .bootstrap_seed(102)
                .build(),
        },
    )
    .expect("dr att(g,t)");
    let att_map = estimates
        .iter()
        .map(|est| ((est.group, est.time), est.att))
        .collect::<BTreeMap<_, _>>();
    // Four cells, not six. Every unit here is eventually treated (groups 2, 3
    // and 4; there is no never-treated arm), and a not-yet-treated control must
    // be untreated at the LATER of the two periods a cell compares. That empties
    // the cells whose later period is 4, since no group has G > 4:
    //   (2,4) base 1 -> needs G > 4   gone
    //   (3,4) base 2 -> needs G > 4   gone
    // and all of group 4's own cells, whose base is 3, need G > 3 with only
    // group 4 itself above it. What survives is (2,2), (2,3), (3,1), (3,3).
    // Six was the count under the older rule that judged eligibility at `time`
    // alone, which admitted units already treated by the baseline period.
    assert_eq!(att_map.len(), 4);
    assert!((att_map[&(2, 2)] - 2.0).abs() < 0.2);
    assert!((att_map[&(2, 3)] - 2.0).abs() < 0.2);
    assert!((att_map[&(3, 3)] - 1.0).abs() < 0.2);
    // The surviving pre-treatment placebo, which must be flat.
    assert!(att_map[&(3, 1)].abs() < 0.2);
}

#[test]
fn dr_group_time_rejects_inconsistent_covariates() {
    let rows = vec![
        row_dr(None, 1, 10.0, 0.0),
        AttGtDrObservation {
            covariates: vec![0.0, 1.0],
            ..AttGtDrObservation::new(Some(2), 1, 10.0)
        },
    ];
    let err = estimate_att_gt_dr(&rows, AttGtDrConfig::default()).expect_err("must fail");
    assert_eq!(
        err,
        AttGtError::InconsistentCovariateCount {
            expected: 1,
            actual: 2
        }
    );
}

#[test]
fn dr_with_influence_outputs_aligned_vectors() {
    let mut rows = Vec::new();
    for _ in 0..20 {
        for t in 1..=4 {
            let t_f = f64::from(t);
            for x in [0.0, 1.0] {
                let base = 0.5f64.mul_add(x, 10.0 + t_f);
                rows.push(row_dr(None, t, base, x));
                rows.push(row_dr(Some(2), t, base + if t >= 2 { 2.0 } else { 0.0 }, x));
                rows.push(row_dr(Some(3), t, base + if t >= 3 { 1.0 } else { 0.0 }, x));
            }
        }
    }

    let out = estimate_att_gt_dr_with_influence(
        &rows,
        AttGtDrConfig {
            drdid: DrDidConfig::builder()
                .bootstrap_reps(49)
                .bootstrap_seed(111)
                .build(),
            ..AttGtDrConfig::default()
        },
    )
    .expect("dr with influence");

    assert_eq!(out.estimates.len(), out.influence_functions.len());
    let n = rows.len();
    assert!(
        out.influence_functions
            .iter()
            .all(|influence| influence.len() == n)
    );
    assert!(
        out.influence_functions
            .iter()
            .all(|influence| influence.iter().any(|value| value.abs() > 0.0))
    );
}

#[test]
fn or_with_influence_outputs_aligned_vectors() {
    let mut rows = Vec::new();
    for _ in 0..20 {
        for t in 1..=4 {
            let t_f = f64::from(t);
            for x in [0.0, 1.0] {
                let base = 0.75f64.mul_add(x, 20.0 + t_f);
                rows.push(row_dr(None, t, base, x));
                rows.push(row_dr(Some(2), t, base + if t >= 2 { 2.0 } else { 0.0 }, x));
                rows.push(row_dr(Some(3), t, base + if t >= 3 { 1.0 } else { 0.0 }, x));
            }
        }
    }
    let out = estimate_att_gt_or_with_influence(&rows, AttGtDrConfig::default())
        .expect("or with influence");
    assert_eq!(out.estimates.len(), out.influence_functions.len());
    assert!(
        out.influence_functions
            .iter()
            .all(|influence| influence.len() == rows.len())
    );
}

#[test]
fn influence_bands_work_with_ipw_estimator_output() {
    let mut rows = Vec::new();
    for i in 0..300 {
        let x = if i % 2 == 0 { 0.0 } else { 1.0 };
        let noise = if i % 5 == 0 { -0.1 } else { 0.1 };
        for t in 1..=4 {
            let t_f = f64::from(t);
            let base = 0.8f64.mul_add(x, 10.0 + t_f) + noise;
            rows.push(row_dr(None, t, base, x));
            if i % 4 != 0 {
                rows.push(row_dr(Some(2), t, base + if t >= 2 { 2.0 } else { 0.0 }, x));
            }
            if i % 3 == 0 {
                rows.push(row_dr(Some(3), t, base + if t >= 3 { 1.0 } else { 0.0 }, x));
            }
        }
    }

    let out = estimate_att_gt_ipw_with_influence(
        &rows,
        AttGtDrConfig {
            drdid: DrDidConfig::builder()
                .bootstrap_reps(49)
                .bootstrap_seed(222)
                .propensity_clip(0.01)
                .build(),
            ..AttGtDrConfig::default()
        },
    )
    .expect("ipw with influence");
    let bands = att_gt_simultaneous_bands_with_influence(
        &out.estimates,
        &out.influence_functions,
        AttGtBandConfig {
            reps: 999,
            seed: 33,
            ..AttGtBandConfig::default()
        },
    )
    .expect("bands from estimator output");
    assert_eq!(bands.len(), out.estimates.len());
}

#[test]
fn estimates_or_group_time_att() {
    let mut rows = Vec::new();
    for _ in 0..50 {
        for t in 1..=4 {
            let t_f = f64::from(t);
            for x in [0.0, 1.0] {
                let base = 0.75f64.mul_add(x, 20.0 + t_f);
                rows.push(row_dr(None, t, base, x));
                rows.push(row_dr(Some(2), t, base + if t >= 2 { 2.0 } else { 0.0 }, x));
                rows.push(row_dr(Some(3), t, base + if t >= 3 { 1.0 } else { 0.0 }, x));
            }
        }
    }

    let estimates = estimate_att_gt_or(
        &rows,
        AttGtDrConfig {
            att_gt: AttGtConfig {
                base_period: BasePeriod::Universal,
                ..AttGtConfig::default()
            },
            ..AttGtDrConfig::default()
        },
    )
    .expect("or att(g,t)");
    let att_map = estimates
        .iter()
        .map(|est| ((est.group, est.time), est.att))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(att_map.len(), 6);
    assert!((att_map[&(2, 2)] - 2.0).abs() < 0.15);
    assert!((att_map[&(2, 3)] - 2.0).abs() < 0.15);
    assert!((att_map[&(2, 4)] - 2.0).abs() < 0.15);
    assert!((att_map[&(3, 3)] - 1.0).abs() < 0.15);
    assert!((att_map[&(3, 4)] - 1.0).abs() < 0.15);
}

#[test]
fn estimates_ipw_group_time_att() {
    let mut rows = Vec::new();
    for i in 0..800 {
        let x = if i % 2 == 0 { 0.0 } else { 1.0 };
        let noise = if i % 5 == 0 { -0.1 } else { 0.1 };
        for t in 1..=4 {
            let t_f = f64::from(t);
            let base = 0.8f64.mul_add(x, 10.0 + t_f) + noise;
            rows.push(row_dr(None, t, base, x));
            if i % 4 != 0 {
                rows.push(row_dr(Some(2), t, base + if t >= 2 { 2.0 } else { 0.0 }, x));
            }
            if i % 3 == 0 {
                rows.push(row_dr(Some(3), t, base + if t >= 3 { 1.0 } else { 0.0 }, x));
            }
        }
    }

    let estimates = estimate_att_gt_ipw(
        &rows,
        AttGtDrConfig {
            att_gt: AttGtConfig {
                base_period: BasePeriod::Universal,
                ..AttGtConfig::default()
            },
            drdid: DrDidConfig::builder()
                .propensity_clip(0.01)
                .max_iter(300)
                .tol(1e-8)
                .build(),
        },
    )
    .expect("ipw att(g,t)");

    let att_map = estimates
        .iter()
        .map(|est| ((est.group, est.time), est.att))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(att_map.len(), 6);
    assert!((att_map[&(2, 2)] - 2.0).abs() < 0.35);
    assert!((att_map[&(2, 3)] - 2.0).abs() < 0.35);
    assert!((att_map[&(2, 4)] - 2.0).abs() < 0.35);
    assert!((att_map[&(3, 3)] - 1.0).abs() < 0.35);
    assert!((att_map[&(3, 4)] - 1.0).abs() < 0.35);
}

#[test]
fn computes_simultaneous_bands() {
    let estimates = vec![
        AttGtEstimate {
            group: 2,
            time: 2,
            event_time: 0,
            att: 2.0,
            se: 0.2,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 100,
            control_n: 120,
            total_weight: 220.0,
        },
        AttGtEstimate {
            group: 3,
            time: 3,
            event_time: 0,
            att: 1.0,
            se: 0.2,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 100,
            control_n: 120,
            total_weight: 220.0,
        },
    ];

    let bands = att_gt_simultaneous_bands(
        &estimates,
        AttGtBandConfig {
            reps: 1_999,
            seed: 7,
            ..AttGtBandConfig::default()
        },
    )
    .expect("bands");

    assert_eq!(bands.len(), 2);
    assert!(bands[0].band_low < bands[0].att);
    assert!(bands[0].band_high > bands[0].att);
    let width0 = bands[0].band_high - bands[0].band_low;
    let width1 = bands[1].band_high - bands[1].band_low;
    assert!((width0 - width1).abs() < 1e-12);
}

#[test]
fn computes_simultaneous_bands_with_influence() {
    let influences = vec![
        vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.5, -1.5, 0.25],
        vec![0.5, -0.25, 1.0, -1.0, 0.75, -0.5, 0.25, -0.75],
    ];
    let estimates = vec![
        AttGtEstimate {
            group: 2,
            time: 2,
            event_time: 0,
            att: 2.0,
            se: 0.1,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 100,
            control_n: 120,
            total_weight: 220.0,
        },
        AttGtEstimate {
            group: 3,
            time: 3,
            event_time: 0,
            att: 1.0,
            se: 0.1,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 100,
            control_n: 120,
            total_weight: 220.0,
        },
    ];
    let bands = att_gt_simultaneous_bands_with_influence(
        &estimates,
        &influences,
        AttGtBandConfig {
            reps: 2_499,
            seed: 19,
            ..AttGtBandConfig::default()
        },
    )
    .expect("bands with influence");

    assert_eq!(bands.len(), 2);
    for band in &bands {
        assert!(band.band_low < band.att);
        assert!(band.band_high > band.att);
    }
    let se0 = standard_error_from_influence(&influences[0]);
    let se1 = standard_error_from_influence(&influences[1]);
    assert!((bands[0].se - se0).abs() < 1e-12);
    assert!((bands[1].se - se1).abs() < 1e-12);
}

#[test]
fn simultaneous_bands_matches_sorted_reference() {
    let estimates = vec![
        AttGtEstimate {
            group: 2,
            time: 2,
            event_time: 0,
            att: 2.0,
            se: 0.2,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 100,
            control_n: 120,
            total_weight: 220.0,
        },
        AttGtEstimate {
            group: 3,
            time: 3,
            event_time: 0,
            att: 1.0,
            se: 0.2,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 100,
            control_n: 120,
            total_weight: 220.0,
        },
    ];
    let config = AttGtBandConfig {
        reps: 2_999,
        seed: 1_337,
        ..AttGtBandConfig::default()
    };

    let actual = att_gt_simultaneous_bands(&estimates, config).expect("bands");
    let expected = reference_bands_sorted_quantile(&estimates, config);

    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_eq!(a.band_low.to_bits(), e.band_low.to_bits());
        assert_eq!(a.band_high.to_bits(), e.band_high.to_bits());
    }
}

#[test]
fn simultaneous_bands_with_influence_matches_sorted_reference() {
    let influences = vec![
        vec![1.0, -1.0, 0.5, -0.5, 0.0, 1.5, -1.5, 0.25],
        vec![0.5, -0.25, 1.0, -1.0, 0.75, -0.5, 0.25, -0.75],
    ];
    let estimates = vec![
        AttGtEstimate {
            group: 2,
            time: 2,
            event_time: 0,
            att: 2.0,
            se: 0.1,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 100,
            control_n: 120,
            total_weight: 220.0,
        },
        AttGtEstimate {
            group: 3,
            time: 3,
            event_time: 0,
            att: 1.0,
            se: 0.1,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 100,
            control_n: 120,
            total_weight: 220.0,
        },
    ];
    let config = AttGtBandConfig {
        reps: 2_499,
        seed: 19,
        ..AttGtBandConfig::default()
    };

    let actual =
        att_gt_simultaneous_bands_with_influence(&estimates, &influences, config).expect("bands");
    let expected = reference_bands_with_influence_sorted_quantile(&estimates, &influences, config);

    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_eq!(a.band_low.to_bits(), e.band_low.to_bits());
        assert_eq!(a.band_high.to_bits(), e.band_high.to_bits());
    }
}

#[test]
fn bands_with_influence_reject_mismatched_counts() {
    let estimates = vec![AttGtEstimate {
        group: 2,
        time: 2,
        event_time: 0,
        att: 1.0,
        se: 0.1,
        ci_low: 0.0,
        ci_high: 0.0,
        treated_n: 10,
        control_n: 10,
        total_weight: 20.0,
    }];
    let err = att_gt_simultaneous_bands_with_influence(&estimates, &[], AttGtBandConfig::default())
        .expect_err("must fail");
    assert_eq!(err, AttGtBandError::InfluenceCountMismatch);
}

#[test]
fn simultaneous_bands_reject_invalid_reps() {
    let estimates = vec![AttGtEstimate {
        group: 2,
        time: 2,
        event_time: 0,
        att: 1.0,
        se: 0.1,
        ci_low: 0.0,
        ci_high: 0.0,
        treated_n: 10,
        control_n: 10,
        total_weight: 20.0,
    }];
    let err = att_gt_simultaneous_bands(
        &estimates,
        AttGtBandConfig {
            reps: 0,
            ..AttGtBandConfig::default()
        },
    )
    .expect_err("must fail");
    assert_eq!(err, AttGtBandError::InvalidReps);
}
