//! Parity of the panel `ATT(g,t)` route against `did` 2.5.1 and `DRDID` 1.3.0.
//!
//! Fixtures are written by `tests/generate_cs_panel_reference.R`, which records
//! every `att_gt` default it overrides. Three of those defaults are the usual
//! source of a spurious failure here and are worth naming again:
//! `base_period = "varying"`, `bstrap = TRUE` (so R's headline SEs are bootstrap,
//! not analytic) and `cband = TRUE` (so its intervals are simultaneous, not
//! pointwise). The fixtures set all three explicitly.

#![allow(clippy::float_cmp)]

use std::fs;

use did_methods::{
    AttGtConfig, AttGtDrConfig, AttGtDrObservation, BasePeriod, ComparisonGroup, DrDidConfig,
    DrDidObservation, InferenceConfig, estimate_att_gt_dr_panel_with_influence,
    estimate_drdid_panel,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct DataRow {
    year: i32,
    countyreal: i64,
    lemp: f64,
    #[serde(rename = "first.treat")]
    first_treat: i32,
    lpop: f64,
}

#[derive(Deserialize)]
struct CsPanelRef {
    group: Vec<i32>,
    t: Vec<i32>,
    att: Vec<f64>,
    /// `null` at the (g, g-1) anchor cell that a universal base period emits.
    se: Vec<Option<f64>>,
    n_units: usize,
    inffunc: Vec<Vec<f64>>,
    /// The ids that `inffunc`'s ROWS correspond to. `did` reorders units by
    /// cohort during pre-processing, so this is not the sorted id list.
    inffunc_unit_order: Vec<i64>,
    data_subset: Vec<DataRow>,
}

#[derive(Deserialize)]
struct DrDidCellRef {
    att: f64,
    se: f64,
    inffunc: Vec<f64>,
    d: Vec<f64>,
    lemp_base: Vec<f64>,
    lemp_post: Vec<f64>,
    lpop: Vec<f64>,
}

fn observations(rows: &[DataRow]) -> Vec<AttGtDrObservation> {
    rows.iter()
        .map(|row| AttGtDrObservation {
            unit_id: Some(row.countyreal),
            // `did` encodes never-treated as gname == 0, not as a missing value.
            first_treated_time: (row.first_treat != 0).then_some(row.first_treat),
            time: row.year,
            outcome: row.lemp,
            weight: 1.0,
            covariates: vec![row.lpop],
        })
        .collect()
}

fn universal_config(comparison_group: ComparisonGroup) -> AttGtDrConfig {
    AttGtDrConfig {
        att_gt: AttGtConfig {
            confidence_level: InferenceConfig::default(),
            comparison_group,
            base_period: BasePeriod::Universal,
            anticipation_periods: 0,
            skip_incomplete_pairs: true,
        },
        drdid: DrDidConfig::builder().build(),
    }
}

/// The pair estimator alone, on one `(g, t)` cell lifted out of `mpdta` by hand.
///
/// Checked first because if this fails nothing above it can pass, and the failure
/// is then in `drdid_panel` rather than in the `ATT(g,t)` loop around it.
#[test]
fn drdid_panel_matches_r_on_one_lifted_cell() {
    let raw = fs::read_to_string("tests/drdid_panel_cell_ref.json").expect("read cell fixture");
    let fixture: DrDidCellRef = serde_json::from_str(&raw).expect("parse cell fixture");

    let rows = (0..fixture.d.len())
        .map(|i| DrDidObservation {
            treated: fixture.d[i] > 0.5,
            delta_outcome: fixture.lemp_post[i] - fixture.lemp_base[i],
            weight: 1.0,
            // estimate_drdid_panel takes a finished design matrix, so the
            // intercept is explicit here. R was given cbind(1, lpop) to match.
            covariates: vec![1.0, fixture.lpop[i]],
        })
        .collect::<Vec<_>>();

    let fit = estimate_drdid_panel(&rows, DrDidConfig::builder().build()).expect("panel fit");

    assert!(
        (fit.att - fixture.att).abs() < 1e-10,
        "att {} vs R {}",
        fit.att,
        fixture.att
    );
    assert!(
        (fit.se - fixture.se).abs() < 1e-10,
        "se {} vs R {}",
        fit.se,
        fixture.se
    );
    assert_eq!(fit.influence_function.len(), fixture.inffunc.len());
    for (ours, theirs) in fit.influence_function.iter().zip(&fixture.inffunc) {
        assert!((ours - theirs).abs() < 1e-9, "influence {ours} vs {theirs}");
    }
}

/// The full grid, universal base period, never-treated comparison group.
#[test]
fn panel_att_gt_matches_r_universal_never_treated() {
    let raw =
        fs::read_to_string("tests/cs_panel_dr_universal_ref.json").expect("read panel fixture");
    let fixture: CsPanelRef = serde_json::from_str(&raw).expect("parse panel fixture");

    let out = estimate_att_gt_dr_panel_with_influence(
        &observations(&fixture.data_subset),
        universal_config(ComparisonGroup::NeverTreated),
    )
    .expect("panel att(g,t)");

    // R emits the anchor cell (g, g-1) with att = 0 and se = NA under a universal
    // base period. We skip it rather than emit a degenerate estimate, so the
    // comparison is over R's non-anchor cells only.
    let expected = (0..fixture.att.len())
        .filter(|&i| fixture.se[i].is_some())
        .map(|i| {
            (
                fixture.group[i],
                fixture.t[i],
                fixture.att[i],
                fixture.se[i].unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(
        out.estimates.len(),
        expected.len(),
        "cell count: rust {:?} vs R {:?}",
        out.estimates
            .iter()
            .map(|e| (e.group, e.time))
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|(g, t, _, _)| (*g, *t))
            .collect::<Vec<_>>()
    );

    for (ours, (group, time, att, se)) in out.estimates.iter().zip(&expected) {
        assert_eq!((ours.group, ours.time), (*group, *time));
        assert!(
            (ours.att - att).abs() < 1e-10,
            "att({group},{time}) {} vs R {att}",
            ours.att
        );
        assert!(
            (ours.se - se).abs() < 1e-10,
            "se({group},{time}) {} vs R {se}",
            ours.se
        );
    }

    // The influence matrix is the thing everything downstream is a function of,
    // so it is compared element for element rather than trusted because the SEs
    // agreed. R's inffunc is n_units x n_cells in the fixture's row-major form.
    assert_eq!(fixture.inffunc_unit_order.len(), fixture.n_units);
    for (cell, ours) in out.influence_functions.iter().enumerate() {
        assert_eq!(
            ours.len(),
            fixture.n_units,
            "influence length for cell {cell}"
        );
    }

    // Our rows are indexed by position in the sorted distinct unit ids; R's are
    // indexed by its own cohort-major order. Join on the id rather than assuming
    // either. `our_position` is the same map `unit_universe` builds internally.
    let mut sorted_ids = fixture
        .data_subset
        .iter()
        .map(|row| row.countyreal)
        .collect::<Vec<_>>();
    sorted_ids.sort_unstable();
    sorted_ids.dedup();
    assert_eq!(sorted_ids.len(), fixture.n_units);

    let r_columns = (0..fixture.att.len())
        .filter(|&i| fixture.se[i].is_some())
        .collect::<Vec<_>>();
    // The influence function is held to a looser bound than the ATT and the SE,
    // deliberately. It is a PER-UNIT quantity that depends directly on the fitted
    // propensity score, so it inherits the full difference between R's logistic
    // solver and ours. The ATT and SE are averages over hundreds of units, where
    // those differences cancel to below 1e-10. Asserting one worst case rather
    // than 6000 individually so that the reported number is the real bound.
    let mut worst = 0.0_f64;
    let mut worst_at = (0_i64, 0_usize);
    for (ours, &r_col) in out.influence_functions.iter().zip(&r_columns) {
        for (r_row, id) in fixture.inffunc_unit_order.iter().enumerate() {
            let our_position = sorted_ids
                .binary_search(id)
                .unwrap_or_else(|_| panic!("unit {id} missing from the sorted universe"));
            let deviation = (ours[our_position] - fixture.inffunc[r_row][r_col]).abs();
            if deviation > worst {
                worst = deviation;
                worst_at = (*id, r_col);
            }
        }
    }
    assert!(
        worst < 1e-7,
        "worst influence deviation {worst:e} at unit {}, cell {}",
        worst_at.0,
        worst_at.1
    );
    println!("worst influence deviation vs did 2.5.1: {worst:e}");
}

/// The not-yet-treated comparison group, which is the switch that measures
/// comparator contamination.
#[test]
fn panel_att_gt_matches_r_universal_not_yet_treated() {
    let raw = fs::read_to_string("tests/cs_panel_dr_notyet_ref.json").expect("read notyet fixture");
    let fixture: CsPanelRef = serde_json::from_str(&raw).expect("parse notyet fixture");
    let base =
        fs::read_to_string("tests/cs_panel_dr_universal_ref.json").expect("read panel fixture");
    let base_fixture: CsPanelRef = serde_json::from_str(&base).expect("parse panel fixture");

    let out = estimate_att_gt_dr_panel_with_influence(
        &observations(&base_fixture.data_subset),
        universal_config(ComparisonGroup::NotYetTreated),
    )
    .expect("panel att(g,t) not-yet-treated");

    let expected = (0..fixture.att.len())
        .filter(|&i| fixture.se[i].is_some())
        .map(|i| {
            (
                fixture.group[i],
                fixture.t[i],
                fixture.att[i],
                fixture.se[i].unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(out.estimates.len(), expected.len());
    for (ours, (group, time, att, se)) in out.estimates.iter().zip(&expected) {
        assert_eq!((ours.group, ours.time), (*group, *time));
        assert!(
            (ours.att - att).abs() < 1e-10,
            "att({group},{time}) {} vs R {att}",
            ours.att
        );
        assert!(
            (ours.se - se).abs() < 1e-10,
            "se({group},{time}) {} vs R {se}",
            ours.se
        );
    }
}

/// A unit-id-free input must fail loudly rather than quietly produce the
/// repeated-cross-section answer under a panel name.
#[test]
fn panel_route_refuses_rows_without_unit_ids() {
    let raw =
        fs::read_to_string("tests/cs_panel_dr_universal_ref.json").expect("read panel fixture");
    let fixture: CsPanelRef = serde_json::from_str(&raw).expect("parse panel fixture");
    let mut rows = observations(&fixture.data_subset);
    rows[17].unit_id = None;

    let err = estimate_att_gt_dr_panel_with_influence(
        &rows,
        universal_config(ComparisonGroup::NeverTreated),
    )
    .expect_err("must refuse");
    assert_eq!(err, did_methods::AttGtError::MissingUnitId);
}

// ---------------------------------------------------------------------------
// Aggregation parity against did::aggte
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct AggRef {
    overall_att: f64,
    overall_se: f64,
    egt: Option<Vec<i32>>,
    att_egt: Option<Vec<f64>>,
    se_egt: Option<Vec<Option<f64>>>,
}

#[derive(Deserialize)]
struct Aggregations {
    simple: AggRef,
    dynamic: AggRef,
    group: AggRef,
    calendar: AggRef,
    dynamic_balanced: AggRef,
}

#[derive(Deserialize)]
struct CsPanelWithAgg {
    data_subset: Vec<DataRow>,
    aggregations: Aggregations,
}

/// Estimates, influence and unit panel from the universal-base fixture, which is
/// the input every aggregation below is computed from.
fn universal_run() -> (
    Vec<did_methods::AttGtEstimate>,
    Vec<Vec<f64>>,
    did_methods::UnitPanel,
    Aggregations,
) {
    let raw =
        fs::read_to_string("tests/cs_panel_dr_universal_ref.json").expect("read panel fixture");
    let fixture: CsPanelWithAgg = serde_json::from_str(&raw).expect("parse panel fixture");
    let rows = observations(&fixture.data_subset);
    let out = estimate_att_gt_dr_panel_with_influence(
        &rows,
        universal_config(ComparisonGroup::NeverTreated),
    )
    .expect("panel att(g,t)");
    let panel = did_methods::unit_panel(&rows).expect("unit panel");
    (
        out.estimates,
        out.influence_functions,
        panel,
        fixture.aggregations,
    )
}

fn check_overall(label: &str, got: &did_methods::AggteResult, want: &AggRef) {
    assert!(
        (got.overall_att - want.overall_att).abs() < 1e-9,
        "{label} overall att {} vs R {}",
        got.overall_att,
        want.overall_att
    );
    assert!(
        (got.overall_se - want.overall_se).abs() < 1e-9,
        "{label} overall se {} vs R {}",
        got.overall_se,
        want.overall_se
    );
}

/// Compare the path against R's, minus the anchor.
///
/// R's dynamic path carries the base event time (`e = -1`) as a point with
/// `att = 0` and `se = NA`. We do not emit it, and that is deliberate rather
/// than an omission: it is not an estimate, it is the normalisation, and a row
/// with zero variance would make the influence covariance singular. That matrix
/// is exactly what `EventTimeResult::to_honest_input` feeds to the HonestDiD
/// FLCI solve, which needs to invert it. The anchor belongs on a plot, not in an
/// estimate vector, so R's `NA`-se entries are dropped here.
fn check_path(label: &str, got: &did_methods::AggteResult, want: &AggRef) {
    let keys = want.egt.as_ref().expect("egt");
    let atts = want.att_egt.as_ref().expect("att_egt");
    let ses = want.se_egt.as_ref().expect("se_egt");
    let expected = (0..keys.len())
        .filter(|&i| ses[i].is_some())
        .map(|i| (keys[i], atts[i], ses[i].unwrap_or_default()))
        .collect::<Vec<_>>();

    assert_eq!(got.by_key.len(), expected.len(), "{label} path length");
    for (point, (key, att, se)) in got.by_key.iter().zip(&expected) {
        assert_eq!(point.key, *key, "{label} key");
        assert!(
            (point.att - att).abs() < 1e-9,
            "{label} att at {key}: {} vs R {att}",
            point.att
        );
        assert!(
            (point.se - se).abs() < 1e-9,
            "{label} se at {key}: {} vs R {se}",
            point.se
        );
    }
}

fn run(
    estimates: &[did_methods::AttGtEstimate],
    influence: &[Vec<f64>],
    panel: &did_methods::UnitPanel,
    aggregation: did_methods::AggteType,
    balance_e: Option<i32>,
) -> did_methods::AggteResult {
    did_methods::aggregate_att_gt(
        estimates,
        influence,
        panel,
        did_methods::AggteConfig {
            aggregation,
            balance_e,
            min_e: None,
            max_e: None,
            confidence_level: InferenceConfig::default(),
        },
    )
    .expect("aggregate")
}

#[test]
fn aggte_simple_matches_r() {
    let (estimates, influence, panel, want) = universal_run();
    let got = run(
        &estimates,
        &influence,
        &panel,
        did_methods::AggteType::Simple,
        None,
    );
    check_overall("simple", &got, &want.simple);
    assert!(got.by_key.is_empty(), "simple has no path");
}

#[test]
fn aggte_dynamic_matches_r() {
    let (estimates, influence, panel, want) = universal_run();
    let got = run(
        &estimates,
        &influence,
        &panel,
        did_methods::AggteType::Dynamic,
        None,
    );
    check_overall("dynamic", &got, &want.dynamic);
    check_path("dynamic", &got, &want.dynamic);
}

#[test]
fn aggte_group_matches_r() {
    let (estimates, influence, panel, want) = universal_run();
    let got = run(
        &estimates,
        &influence,
        &panel,
        did_methods::AggteType::Group,
        None,
    );
    check_overall("group", &got, &want.group);
    check_path("group", &got, &want.group);
}

#[test]
fn aggte_calendar_matches_r() {
    let (estimates, influence, panel, want) = universal_run();
    let got = run(
        &estimates,
        &influence,
        &panel,
        did_methods::AggteType::Calendar,
        None,
    );
    check_overall("calendar", &got, &want.calendar);
    check_path("calendar", &got, &want.calendar);
}

/// The one that matters most for Study 1: a dynamic path on a fixed cohort
/// composition. With `balance_e = 1` on mpdta, cohort 2007 is dropped (it is
/// observed for 0 post periods, not 1) and the path is cut at event time 1.
#[test]
fn aggte_dynamic_balanced_matches_r() {
    let (estimates, influence, panel, want) = universal_run();
    let got = run(
        &estimates,
        &influence,
        &panel,
        did_methods::AggteType::Dynamic,
        Some(1),
    );
    check_overall("dynamic_balanced", &got, &want.dynamic_balanced);
    check_path("dynamic_balanced", &got, &want.dynamic_balanced);

    // The point of balancing: the unbalanced path runs further and is made of a
    // different set of cohorts, so the two must not coincide.
    let unbalanced = run(
        &estimates,
        &influence,
        &panel,
        did_methods::AggteType::Dynamic,
        None,
    );
    assert!(unbalanced.by_key.len() > got.by_key.len());
    assert!((unbalanced.overall_att - got.overall_att).abs() > 1e-6);
}

// ---------------------------------------------------------------------------
// Multiplier bootstrap
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct MbootRef {
    se: Vec<Option<f64>>,
    crit_val: f64,
}

/// The bootstrap cannot be reproduced across languages: R's RNG stream is not
/// ours, so the draws differ even with matched weights. What is checkable, and
/// what actually matters, is that the two agree to within Monte Carlo error.
/// Both sides use 20,000 iterations to make that error small enough for the
/// comparison to mean something.
///
/// The critical value is the load-bearing number: it is what makes the bands
/// simultaneous rather than pointwise, and it is the one the manuscript's
/// figures depend on.
#[test]
fn mboot_bands_agree_with_did_within_simulation_error() {
    let raw = fs::read_to_string("tests/cs_panel_mboot_ref.json").expect("read mboot fixture");
    let fixture: MbootRef = serde_json::from_str(&raw).expect("parse mboot fixture");
    let (estimates, influence, _panel, _) = universal_run();

    let bands = did_methods::att_gt_mboot_bands(
        &estimates,
        &influence,
        None,
        did_methods::AttGtBandConfig {
            confidence_level: InferenceConfig::default(),
            reps: 20_000,
            seed: 20_260_819,
        },
    )
    .expect("mboot bands");

    // The implied critical value, recovered from any band: (high - att) / se.
    let implied = (bands[0].band_high - bands[0].att) / bands[0].se;
    assert!(
        (implied - fixture.crit_val).abs() < 0.05,
        "critical value {implied} vs did {}",
        fixture.crit_val
    );

    // Simultaneous, so it must exceed the pointwise normal quantile. If this
    // ever fails the bands are not doing the job their name claims.
    assert!(
        implied > 1.96,
        "a simultaneous critical value below the pointwise one: {implied}"
    );

    // Bootstrap standard errors, cell by cell, skipping R's NA anchors.
    let expected = fixture
        .se
        .iter()
        .filter_map(|value| *value)
        .collect::<Vec<_>>();
    assert_eq!(bands.len(), expected.len());
    for (band, want) in bands.iter().zip(&expected) {
        let relative = (band.se - want).abs() / want;
        assert!(
            relative < 0.05,
            "bootstrap se {} vs did {want} for ({}, {})",
            band.se,
            band.group,
            band.time
        );
    }
}

/// The correlation-aware aggregated variance, pinned.
///
/// `aggregate_att_gt_event_time_with_influence` used to compute
/// `sum(w^2 * se^2)`, the variance of a weighted sum of INDEPENDENT components.
/// The ATT(g,t) cells share units, so that understated the standard error. This
/// asserts the two formulas actually disagree on real data, so a regression back
/// to the cheap one cannot pass silently.
#[test]
fn aggregated_variance_uses_the_influence_functions_not_the_diagonal() {
    let (estimates, influence, _panel, _) = universal_run();
    let aggregated = did_methods::aggregate_att_gt_event_time_with_influence(
        &estimates,
        &influence,
        did_methods::AttGtAggregationConfig::default(),
    )
    .expect("aggregate");

    // Rebuild the naive number for the event time with the most components.
    let busiest = aggregated
        .estimates
        .iter()
        .enumerate()
        .max_by_key(|(_, e)| e.summary.components)
        .map(|(index, _)| index)
        .expect("some event time");
    let event_time = aggregated.estimates[busiest].summary.components;
    assert!(event_time > 1, "need a multi-component bucket to compare");

    let ours = aggregated.estimates[busiest].summary.se;
    assert!(ours.is_finite() && ours > 0.0);

    let naive = {
        let key = aggregated.estimates[busiest].event_time;
        let members = estimates
            .iter()
            .filter(|e| e.event_time == key)
            .collect::<Vec<_>>();
        let total: f64 = members.iter().map(|e| f64::from(e.treated_n as u32)).sum();
        members
            .iter()
            .map(|e| {
                let w = f64::from(e.treated_n as u32) / total;
                w * w * e.se * e.se
            })
            .sum::<f64>()
            .sqrt()
    };
    assert!(
        (ours - naive).abs() > 1e-6,
        "influence-based se {ours} is indistinguishable from the independence \
         approximation {naive}; the fix may have been reverted"
    );
}

// ---------------------------------------------------------------------------
// Clustering
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ClusteredRef {
    n_states: usize,
    se: Vec<Option<f64>>,
    crit_val: f64,
    unit_order: Vec<i64>,
    cluster_of_unit: Vec<i64>,
}

/// Clustered bands against `did` with `clustervars`.
///
/// This is the shape Study 1 has: two parents of one child are two units sharing
/// every household shock, so the family is the cluster and the unit is not.
/// mpdta stands in for it with counties clustered into states.
#[test]
fn clustered_mboot_bands_agree_with_did() {
    let raw = fs::read_to_string("tests/cs_panel_mboot_clustered_ref.json")
        .expect("read clustered fixture");
    let fixture: ClusteredRef = serde_json::from_str(&raw).expect("parse clustered fixture");
    let (estimates, influence, _panel, _) = universal_run();

    // `unit_order` is the sorted id list, which is exactly our unit index.
    assert_eq!(fixture.unit_order.len(), influence[0].len());

    let bands = did_methods::att_gt_mboot_bands(
        &estimates,
        &influence,
        Some(&fixture.cluster_of_unit),
        did_methods::AttGtBandConfig {
            confidence_level: InferenceConfig::default(),
            reps: 20_000,
            seed: 20_260_819,
        },
    )
    .expect("clustered mboot bands");

    let implied = (bands[0].band_high - bands[0].att) / bands[0].se;
    // Looser than the unclustered case on purpose: the bootstrap now draws 29
    // weights per replication rather than 500, so the Monte Carlo error of a
    // maximum over twelve statistics is correspondingly larger.
    assert!(
        (implied - fixture.crit_val).abs() < 0.15,
        "clustered critical value {implied} vs did {}",
        fixture.crit_val
    );

    let expected = fixture
        .se
        .iter()
        .filter_map(|value| *value)
        .collect::<Vec<_>>();
    assert_eq!(bands.len(), expected.len());
    for (band, want) in bands.iter().zip(&expected) {
        let relative = (band.se - want).abs() / want;
        assert!(
            relative < 0.10,
            "clustered se {} vs did {want} for ({}, {})",
            band.se,
            band.group,
            band.time
        );
    }

    // Clustering must actually change something. With 29 clusters standing in for
    // 500 units the standard errors move substantially, and a test that passed
    // whether or not the labels were used would be worthless.
    let unclustered = did_methods::att_gt_mboot_bands(
        &estimates,
        &influence,
        None,
        did_methods::AttGtBandConfig {
            confidence_level: InferenceConfig::default(),
            reps: 20_000,
            seed: 20_260_819,
        },
    )
    .expect("unclustered bands");
    let moved = bands
        .iter()
        .zip(&unclustered)
        .filter(|(a, b)| (a.se - b.se).abs() / b.se > 0.1)
        .count();
    assert!(
        moved >= bands.len() / 2,
        "clustering barely moved the standard errors: {moved} of {} cells",
        bands.len()
    );
    assert_eq!(
        fixture.n_states,
        fixture
            .cluster_of_unit
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    );
}

/// The identity that makes clustering safe to switch on: one cluster per unit
/// must reproduce the unclustered answer exactly, not approximately.
///
/// Without this, turning clustering on for a design that has none would move
/// every number for no reason, and there would be no way to tell that from a
/// real clustering effect.
#[test]
fn clustering_by_unit_reproduces_the_unclustered_standard_error() {
    let (estimates, influence, panel, _) = universal_run();
    let n = influence[0].len();
    let singleton = (0..n).map(|i| i as i64).collect::<Vec<i64>>();

    let plain = did_methods::aggregate_att_gt(
        &estimates,
        &influence,
        &panel,
        did_methods::AggteConfig::default(),
    )
    .expect("unclustered");

    let clustered_panel = panel.clone().clustered_by(singleton.clone());
    let clustered = did_methods::aggregate_att_gt(
        &estimates,
        &influence,
        &clustered_panel,
        did_methods::AggteConfig::default(),
    )
    .expect("singleton-clustered");

    assert!((plain.overall_se - clustered.overall_se).abs() < 1e-12);
    for (a, b) in plain.by_key.iter().zip(&clustered.by_key) {
        assert!(
            (a.se - b.se).abs() < 1e-12,
            "event time {}: {} vs {}",
            a.key,
            a.se,
            b.se
        );
    }

    // And the same identity for the bootstrap path.
    let config = did_methods::AttGtBandConfig {
        confidence_level: InferenceConfig::default(),
        reps: 2_000,
        seed: 4_242,
    };
    let plain_bands =
        did_methods::att_gt_mboot_bands(&estimates, &influence, None, config).expect("plain");
    let singleton_bands =
        did_methods::att_gt_mboot_bands(&estimates, &influence, Some(&singleton), config)
            .expect("singleton");
    for (a, b) in plain_bands.iter().zip(&singleton_bands) {
        assert!((a.se - b.se).abs() < 1e-12, "{} vs {}", a.se, b.se);
    }
}

/// Sampling weights, which is how the matching enters once it stops being the
/// design.
///
/// Study 1 matches each case to up to four comparators. Under Callaway-Sant'Anna
/// the comparison group is the risk set with covariate adjustment, so keeping the
/// matching as a design as well would adjust twice; the decision is that the
/// match weight becomes a weight and nothing else. That only works if the weight
/// reaches every place it should: the pair estimator, the cohort shares that the
/// aggregation weights by, and the correction for those shares being estimated.
/// This checks all three at once against R's `weightsname`.
#[test]
fn weights_flow_through_estimation_and_aggregation() {
    #[derive(Deserialize)]
    struct WeightedRef {
        att: Vec<f64>,
        se: Vec<Option<f64>>,
        unit_order: Vec<i64>,
        weight_of_unit: Vec<f64>,
        dynamic: AggRef,
    }

    let raw = fs::read_to_string("tests/cs_panel_weighted_ref.json").expect("read weighted");
    let fixture: WeightedRef = serde_json::from_str(&raw).expect("parse weighted");
    let base =
        fs::read_to_string("tests/cs_panel_dr_universal_ref.json").expect("read panel fixture");
    let base_fixture: CsPanelWithAgg = serde_json::from_str(&base).expect("parse panel fixture");

    let by_unit = fixture
        .unit_order
        .iter()
        .zip(&fixture.weight_of_unit)
        .map(|(id, weight)| (*id, *weight))
        .collect::<std::collections::BTreeMap<i64, f64>>();

    let rows = base_fixture
        .data_subset
        .iter()
        .map(|row| AttGtDrObservation {
            unit_id: Some(row.countyreal),
            first_treated_time: (row.first_treat != 0).then_some(row.first_treat),
            time: row.year,
            outcome: row.lemp,
            weight: by_unit[&row.countyreal],
            covariates: vec![row.lpop],
        })
        .collect::<Vec<_>>();

    let out = estimate_att_gt_dr_panel_with_influence(
        &rows,
        universal_config(ComparisonGroup::NeverTreated),
    )
    .expect("weighted panel att(g,t)");

    let expected = (0..fixture.att.len())
        .filter(|&i| fixture.se[i].is_some())
        .map(|i| fixture.att[i])
        .collect::<Vec<_>>();
    assert_eq!(out.estimates.len(), expected.len());
    for (ours, want) in out.estimates.iter().zip(&expected) {
        assert!(
            (ours.att - want).abs() < 1e-9,
            "weighted att({}, {}) {} vs R {want}",
            ours.group,
            ours.time,
            ours.att
        );
    }

    let panel = did_methods::unit_panel(&rows).expect("unit panel");
    let dynamic = run(
        &out.estimates,
        &out.influence_functions,
        &panel,
        did_methods::AggteType::Dynamic,
        None,
    );
    check_overall("weighted dynamic", &dynamic, &fixture.dynamic);
    check_path("weighted dynamic", &dynamic, &fixture.dynamic);

    // The weights must matter. If they were being dropped somewhere this would
    // agree with the unweighted run and every assertion above would still pass
    // only because R was also given weights it ignored.
    let unweighted = universal_run().0;
    let moved = out
        .estimates
        .iter()
        .zip(&unweighted)
        .filter(|(a, b)| (a.att - b.att).abs() > 1e-6)
        .count();
    assert_eq!(
        moved,
        out.estimates.len(),
        "weighting left some cells untouched"
    );
}

/// Analytic clustered per-cell standard errors against `did`.
///
/// `did` reports these under `bstrap = FALSE` with `clustervars`, so the per-cell
/// standard error has a reference and does not have to be left to the bootstrap.
#[test]
fn clustered_per_cell_standard_errors_match_did() {
    #[derive(Deserialize)]
    struct AnalyticClusteredRef {
        se: Vec<Option<f64>>,
        unit_order: Vec<i64>,
        cluster_of_unit: Vec<i64>,
    }

    let raw = fs::read_to_string("tests/cs_panel_clustered_analytic_ref.json")
        .expect("read analytic clustered fixture");
    let fixture: AnalyticClusteredRef =
        serde_json::from_str(&raw).expect("parse analytic clustered fixture");
    let (estimates, influence, panel, _) = universal_run();
    assert_eq!(fixture.unit_order.len(), influence[0].len());

    let clustered = did_methods::att_gt_clustered_standard_errors(
        &estimates,
        &influence,
        &panel.clone().clustered_by(fixture.cluster_of_unit.clone()),
        InferenceConfig::default(),
    )
    .expect("clustered per-cell");

    let expected = fixture
        .se
        .iter()
        .filter_map(|value| *value)
        .collect::<Vec<_>>();
    assert_eq!(clustered.len(), expected.len());
    for (ours, want) in clustered.iter().zip(&expected) {
        assert!(
            (ours.se - want).abs() < 1e-10,
            "clustered se({}, {}) {} vs did {want}",
            ours.group,
            ours.time,
            ours.se
        );
    }

    // It has to actually change the answer, in both directions. On this data the
    // first cell falls from 0.0221 to 0.0102 and another rises from 0.0312 to
    // 0.0489, so a rule of thumb like "clustering inflates standard errors" would
    // be wrong here.
    let up = clustered
        .iter()
        .zip(&estimates)
        .filter(|(a, b)| a.se > b.se * 1.05)
        .count();
    let down = clustered
        .iter()
        .zip(&estimates)
        .filter(|(a, b)| a.se < b.se * 0.95)
        .count();
    assert!(up > 0 && down > 0, "moved up {up}, down {down}");
}

/// With one cluster per unit, the clustered per-cell standard error must return
/// the pair estimator's own number.
///
/// This is what makes the function safe to apply unconditionally: on a design
/// with no clustering it is a no-op rather than a small unexplained shift.
#[test]
fn clustered_per_cell_reduces_to_the_pair_estimator() {
    let (estimates, influence, panel, _) = universal_run();
    let same = did_methods::att_gt_clustered_standard_errors(
        &estimates,
        &influence,
        &panel,
        InferenceConfig::default(),
    )
    .expect("singleton clusters");

    for (ours, original) in same.iter().zip(&estimates) {
        assert!(
            (ours.se - original.se).abs() < 1e-12,
            "({}, {}): {} vs pair estimator {}",
            ours.group,
            ours.time,
            ours.se,
            original.se
        );
    }
}

/// `min_e` and `max_e` against `did`, including their effect on the overall.
#[test]
fn aggte_dynamic_trimming_matches_r() {
    #[derive(Deserialize)]
    struct TrimmedRef {
        min_e: i32,
        max_e: i32,
        overall_att: f64,
        overall_se: f64,
        egt: Option<Vec<i32>>,
        att_egt: Option<Vec<f64>>,
        se_egt: Option<Vec<Option<f64>>>,
    }

    let raw = fs::read_to_string("tests/cs_panel_trimmed_ref.json").expect("read trimmed");
    let fixtures: Vec<TrimmedRef> = serde_json::from_str(&raw).expect("parse trimmed");
    let (estimates, influence, panel, _) = universal_run();
    assert_eq!(fixtures.len(), 3);

    for want in &fixtures {
        let got = did_methods::aggregate_att_gt(
            &estimates,
            &influence,
            &panel,
            did_methods::AggteConfig {
                aggregation: did_methods::AggteType::Dynamic,
                balance_e: None,
                min_e: Some(want.min_e),
                max_e: Some(want.max_e),
                confidence_level: InferenceConfig::default(),
            },
        )
        .expect("trimmed dynamic");

        let label = format!("min_e={} max_e={}", want.min_e, want.max_e);
        let as_agg = AggRef {
            overall_att: want.overall_att,
            overall_se: want.overall_se,
            egt: want.egt.clone(),
            att_egt: want.att_egt.clone(),
            se_egt: want.se_egt.clone(),
        };
        check_overall(&label, &got, &as_agg);
        check_path(&label, &got, &as_agg);

        // Trimming must change the overall, or the bounds are being ignored.
        if want.max_e < 3 {
            let untrimmed = run(
                &estimates,
                &influence,
                &panel,
                did_methods::AggteType::Dynamic,
                None,
            );
            assert!(
                (untrimmed.overall_att - got.overall_att).abs() > 1e-9,
                "{label} left the overall unchanged"
            );
        }
    }
}

/// The repeated-cross-section route, which previously had no reference at all.
///
/// `did_attgt_dr_ref.json` has sat in this directory since before the panel route
/// existed and was consumed by nothing: it was generated with `att_gt`'s default
/// `panel = TRUE` while the only Rust path available was RC, so it could never
/// have passed.
///
/// # These are two different estimators, not two implementations of one
///
/// `did` calls `DRDID::drdid_rc` for `est_method = "dr"`, the LOCALLY EFFICIENT
/// repeated-cross-section estimator. This crate's RC route implements the
/// TRADITIONAL one, `DRDID::drdid_rc1`. On mpdta the point estimates agree to
/// 1e-9 and the standard errors differ by about 2.8%: for cell (2004, 2004),
/// 0.0922804 traditional against 0.0897358 locally efficient.
///
/// So the reference here is `drdid_rc1`, cell by cell. Comparing against `did`'s
/// `panel = FALSE` output would be comparing against a different estimator and
/// the 2.8% gap would look like a bug.
#[test]
fn rc_route_matches_the_traditional_estimator_and_aggregates() {
    #[derive(Deserialize)]
    struct RcCell {
        group: i32,
        time: i32,
        att: f64,
        se_traditional: f64,
        se_locally_efficient: f64,
    }
    #[derive(Deserialize)]
    struct RcRef {
        n_rows: usize,
        dynamic: AggRef,
    }

    let cells: Vec<RcCell> = serde_json::from_str(
        &fs::read_to_string("tests/drdid_rc1_cells_ref.json").expect("read rc1"),
    )
    .expect("parse rc1");
    let did_rc: RcRef =
        serde_json::from_str(&fs::read_to_string("tests/cs_rc_ref.json").expect("read rc"))
            .expect("parse rc");
    let base: CsPanelWithAgg = serde_json::from_str(
        &fs::read_to_string("tests/cs_panel_dr_universal_ref.json").expect("read panel"),
    )
    .expect("parse panel");

    let rows = observations(&base.data_subset);
    assert_eq!(rows.len(), did_rc.n_rows);
    let out = did_methods::estimate_att_gt_dr_with_influence(
        &rows,
        universal_config(ComparisonGroup::NeverTreated),
    )
    .expect("rc att(g,t)");
    assert_eq!(out.estimates.len(), cells.len());

    let mut worst_se = 0.0_f64;
    for (ours, want) in out.estimates.iter().zip(&cells) {
        assert_eq!((ours.group, ours.time), (want.group, want.time));
        assert!(
            (ours.att - want.att).abs() < 1e-9,
            "rc att({}, {}) {} vs drdid_rc1 {}",
            ours.group,
            ours.time,
            ours.att,
            want.att
        );
        // Loose by the standards of the panel route's 1e-10. Our traditional RC
        // standard error tracks drdid_rc1 to within 9e-4 relative but is not
        // identical to it. That gap is small, real and unexplained, so it is
        // BOUNDED here rather than asserted away, and the worst case is printed
        // below so a regression shows up as a number rather than as a pass.
        let relative = (ours.se - want.se_traditional).abs() / want.se_traditional;
        worst_se = worst_se.max(relative);
        assert!(
            relative < 1e-3,
            "rc se({}, {}) {} vs drdid_rc1 {} (relative {relative:e})",
            ours.group,
            ours.time,
            ours.se,
            want.se_traditional
        );
        // And it is the traditional estimator, not the locally efficient one.
        assert!(
            (ours.se - want.se_traditional).abs() < (ours.se - want.se_locally_efficient).abs(),
            "rc se({}, {}) {} sits closer to drdid_rc {} than to drdid_rc1 {}; the \
             route may have been switched to the locally efficient estimator",
            ours.group,
            ours.time,
            ours.se,
            want.se_locally_efficient,
            want.se_traditional
        );
    }

    println!("worst rc se deviation vs DRDID::drdid_rc1: {worst_se:e}");

    // The point of this test for item 2 of the plan: the aggregation now works on
    // the RC route, given a ROW-indexed panel. The point estimates match did's
    // aggte; the standard errors cannot, because did aggregated a different
    // estimator's influence functions.
    let panel = did_methods::row_panel(&rows);
    let dynamic = run(
        &out.estimates,
        &out.influence_functions,
        &panel,
        did_methods::AggteType::Dynamic,
        None,
    );
    let keys = did_rc.dynamic.egt.as_ref().expect("egt");
    let atts = did_rc.dynamic.att_egt.as_ref().expect("att");
    let ses = did_rc.dynamic.se_egt.as_ref().expect("se");
    let expected = (0..keys.len())
        .filter(|&i| ses[i].is_some())
        .map(|i| (keys[i], atts[i]))
        .collect::<Vec<_>>();
    assert_eq!(dynamic.by_key.len(), expected.len());
    for (point, (key, att)) in dynamic.by_key.iter().zip(&expected) {
        assert_eq!(point.key, *key);
        assert!(
            (point.att - att).abs() < 1e-9,
            "rc dynamic att at {key}: {} vs R {att}",
            point.att
        );
    }
    assert!(
        (dynamic.overall_att - did_rc.dynamic.overall_att).abs() < 1e-9,
        "rc dynamic overall {} vs R {}",
        dynamic.overall_att,
        did_rc.dynamic.overall_att
    );

    // What the panel route buys, as a number rather than a claim.
    let (panel_estimates, _, _, _) = universal_run();
    for (rc, pnl) in out.estimates.iter().zip(&panel_estimates) {
        assert!((rc.att - pnl.att).abs() < 1e-9);
        assert!(
            rc.se > pnl.se * 2.0,
            "discarding the pairing should inflate the se: {} vs {}",
            rc.se,
            pnl.se
        );
    }
}

/// A repeated panel row is an error, not a frequency weight.
///
/// Expressing "this unit counts four times" by repeating its rows works on the
/// repeated-cross-section route, where a row is an observation. On the panel
/// route the copies collapse into one unit and the intended weight silently
/// vanishes. That is a wrong answer with no symptom, so it is refused.
#[test]
fn a_repeated_panel_row_is_refused() {
    let raw =
        fs::read_to_string("tests/cs_panel_dr_universal_ref.json").expect("read panel fixture");
    let fixture: CsPanelWithAgg = serde_json::from_str(&raw).expect("parse panel fixture");
    let mut rows = observations(&fixture.data_subset);
    rows.push(rows[0].clone());

    let err = estimate_att_gt_dr_panel_with_influence(
        &rows,
        universal_config(ComparisonGroup::NeverTreated),
    )
    .expect_err("a duplicated panel row must be refused");
    assert!(
        matches!(err, did_methods::AttGtError::DuplicatePanelRow { .. }),
        "got {err:?}"
    );
}
