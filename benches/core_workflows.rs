use std::time::Duration;

use criterion::{
    BenchmarkGroup, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use did_methods::inference::sensitivity::{
    benchmark_sensitivity_matrix_rank, benchmark_sensitivity_multi_flci_maxima,
    benchmark_sensitivity_multi_flci_maxima_scalar, benchmark_sensitivity_normal_draws,
    benchmark_sensitivity_normal_draws_scalar, benchmark_sensitivity_rref_pivot_columns,
    benchmark_sensitivity_sandwich_covariance,
};
use did_methods::inference::{
    multiplier_bootstrap_ci, vcov::covariance_matrix_from_influence_matrix,
};
use did_methods::{
    AttGtConfig, AttGtDrConfig, AttGtDrObservation, BootstrapConfig, DidCcConfig, DrDidConfig,
    DrDidObservation, DrDidRepeatedObservation, HonestEventStudyInput, HonestJointPathConfig,
    HonestPostFunctional, HonestSensitivity, HonestWorkflowConfig, HonestWorkflowDirectionMode,
    InferenceConfig, RelativeMagnitudeConfidenceSetConfig,
    assess_honest_event_study_directional_region_with_config,
    assess_honest_event_study_joint_path_region_with_config,
    assess_honest_event_study_post_functional_multi_flci_with_config,
    assess_honest_event_study_post_functional_with_config, assess_honest_event_study_workflow,
    estimate_att_gt_dr, estimate_did_cc_robust, estimate_did_cc_stationary,
    estimate_drdid_improved_repeated_cross_section, estimate_drdid_panel, test_did_cc_stationarity,
};
use faer::Mat;
use faer::prelude::Solve;

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("benchmark size fits u32"))
}

fn build_drdid_panel_rows(num_units: usize, num_covariates: usize) -> Vec<DrDidObservation> {
    (0..num_units)
        .map(|unit_index| {
            let treated = unit_index % 4 == 0;
            let unit_index_f64 = usize_to_f64(unit_index);
            let covariates = (0..num_covariates)
                .map(|covariate_index| {
                    let covariate_index_f64 = usize_to_f64(covariate_index);
                    (unit_index_f64 * (covariate_index_f64 + 1.0)).sin()
                })
                .collect::<Vec<_>>();
            let baseline = unit_index_f64.rem_euclid(23.0) * 0.1;
            let covariate_signal = covariates
                .iter()
                .enumerate()
                .map(|(covariate_index, value)| (usize_to_f64(covariate_index) + 1.0) * value)
                .sum::<f64>()
                * 0.05;
            let treatment_effect = if treated { 1.75 } else { 0.0 };
            DrDidObservation {
                treated,
                delta_outcome: baseline + covariate_signal + treatment_effect,
                weight: 1.0,
                covariates,
            }
        })
        .collect()
}

fn build_att_gt_rows(num_units: usize, start_year: i32, periods: usize) -> Vec<AttGtDrObservation> {
    let mut observations = Vec::with_capacity(num_units * periods);
    for unit_index in 0..num_units {
        let first_treated_time = match unit_index % 5 {
            0 => Some(start_year + 2),
            1 => Some(start_year + 3),
            2 => Some(start_year + 4),
            _ => None,
        };
        let baseline = usize_to_f64(unit_index % 17) * 0.2;
        for period_offset in 0..periods {
            let time = start_year + i32::try_from(period_offset).expect("period offset fits i32");
            let event_time = first_treated_time.map_or(-99, |group| time - group);
            let treated_effect = if event_time >= 0 {
                0.8 + f64::from(event_time) * 0.15
            } else {
                0.0
            };
            let trend = f64::from(time - start_year) * 0.05;
            observations.push(AttGtDrObservation {
                first_treated_time,
                time,
                outcome: baseline + trend + treated_effect,
                weight: 1.0,
                covariates: vec![
                    usize_to_f64(unit_index % 7) * 0.1,
                    usize_to_f64(period_offset) * 0.05,
                ],
            });
        }
    }
    observations
}

fn build_honest_input(num_pre_periods: usize, num_post_periods: usize) -> HonestEventStudyInput {
    let total_periods = num_pre_periods + num_post_periods;
    let betahat = (0..total_periods)
        .map(|index| {
            if index < num_pre_periods {
                (usize_to_f64(index) - usize_to_f64(num_pre_periods)) * 0.05
            } else {
                usize_to_f64(index - num_pre_periods).mul_add(0.3, 0.4)
            }
        })
        .collect::<Vec<_>>();

    let covariance = (0..total_periods)
        .map(|row_index| {
            (0..total_periods)
                .map(|column_index| {
                    let distance = usize_to_f64(row_index.abs_diff(column_index));
                    if row_index == column_index {
                        usize_to_f64(row_index).mul_add(0.01, 0.16)
                    } else {
                        0.03 / (1.0 + distance)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    HonestEventStudyInput {
        betahat,
        covariance,
        pre_periods: (0..num_pre_periods)
            .map(|index| -i32::try_from(num_pre_periods - index).expect("pre index fits i32"))
            .collect(),
        post_periods: (0..num_post_periods)
            .map(|index| i32::try_from(index).expect("post index fits i32"))
            .collect(),
    }
}

fn build_did_cc_rows(num_units_per_cell: usize) -> Vec<DrDidRepeatedObservation> {
    let mut observations = Vec::with_capacity(num_units_per_cell * 4);
    for unit_index in 0..num_units_per_cell {
        let unit_index_f64 = usize_to_f64(unit_index);
        let control_x = usize_to_f64(unit_index % 9) * 0.25;
        let treated_pre_x = usize_to_f64((unit_index + 2) % 9) * 0.25;
        let treated_post_x = usize_to_f64((unit_index + 5) % 9) * 0.25;

        observations.push(DrDidRepeatedObservation {
            treated: false,
            post_period: false,
            outcome: unit_index_f64.mul_add(0.001, control_x),
            weight: 1.0,
            covariates: vec![control_x, control_x * control_x],
        });
        observations.push(DrDidRepeatedObservation {
            treated: false,
            post_period: true,
            outcome: unit_index_f64.mul_add(0.001, control_x + 0.9),
            weight: 1.0,
            covariates: vec![control_x, control_x * control_x],
        });
        observations.push(DrDidRepeatedObservation {
            treated: true,
            post_period: false,
            outcome: unit_index_f64.mul_add(0.001, treated_pre_x + 0.2),
            weight: 1.0,
            covariates: vec![treated_pre_x, treated_pre_x * treated_pre_x],
        });
        observations.push(DrDidRepeatedObservation {
            treated: true,
            post_period: true,
            outcome: unit_index_f64.mul_add(0.001, treated_post_x + 2.1 + treated_post_x * 0.6),
            weight: 1.0,
            covariates: vec![treated_post_x, treated_post_x * treated_post_x],
        });
    }
    observations
}

fn build_improved_repeated_rows(num_units_per_period: usize) -> Vec<DrDidRepeatedObservation> {
    let mut rows = Vec::with_capacity(2 * num_units_per_period);
    for post_period in [false, true] {
        for unit_index in 0..num_units_per_period {
            let unit_f = usize_to_f64(unit_index + 1);
            let x1 = (0.013 * unit_f).sin();
            let x2 = (0.021 * unit_f).cos();
            let score = 0.4f64.mul_add(x1, -(0.25 * x2)) + if post_period { 0.12 } else { -0.08 };
            let treated = score > 0.0;
            let base =
                0.7f64.mul_add(-x2, 1.2f64.mul_add(x1, 1.8)) + if post_period { 0.9 } else { 0.0 };
            let treatment_effect = if treated && post_period { 1.5 } else { 0.0 };
            let noise = (0.033 * unit_f).sin() * 0.1;
            rows.push(DrDidRepeatedObservation {
                treated,
                post_period,
                outcome: base + treatment_effect + noise,
                weight: 1.0,
                covariates: vec![1.0, x1, x2],
            });
        }
    }
    rows
}

fn bench_drdid_panel(criterion: &mut Criterion) {
    let rows = build_drdid_panel_rows(20_000, 6);
    let throughput = u64::try_from(rows.len()).expect("row count fits u64");
    let mut group = criterion.benchmark_group("drdid_panel");
    group.throughput(Throughput::Elements(throughput));
    group.bench_function("20k_rows_6_covariates", |bencher| {
        bencher.iter(|| estimate_drdid_panel(&rows, DrDidConfig::default()).expect("panel drdid"));
    });
    group.finish();
}

fn bench_drdid_improved_repeated(criterion: &mut Criterion) {
    let rows = build_improved_repeated_rows(8_000);
    let throughput = u64::try_from(rows.len()).expect("row count fits u64");
    let mut group = criterion.benchmark_group("drdid_improved_repeated");
    group.throughput(Throughput::Elements(throughput));
    group.bench_function("8k_per_period_3_covariates", |bencher| {
        bencher.iter(|| {
            estimate_drdid_improved_repeated_cross_section(&rows, DrDidConfig::default())
                .expect("improved repeated drdid")
        });
    });
    group.finish();
}

fn bench_att_gt_dr(criterion: &mut Criterion) {
    let rows = build_att_gt_rows(8_000, 2000, 6);
    let throughput = u64::try_from(rows.len()).expect("row count fits u64");
    let mut group = criterion.benchmark_group("att_gt_dr");
    group.throughput(Throughput::Elements(throughput));
    group.bench_function("8k_units_6_periods", |bencher| {
        bencher.iter(|| {
            estimate_att_gt_dr(
                &rows,
                AttGtDrConfig {
                    att_gt: AttGtConfig::default(),
                    drdid: DrDidConfig::default(),
                },
            )
            .expect("att_gt dr")
        });
    });
    group.finish();
}

fn bench_honest_workflow(criterion: &mut Criterion) {
    let input = build_honest_input(3, 2);
    let functionals = vec![HonestPostFunctional::Period(0)];
    let mut group = criterion.benchmark_group("honest_workflow");
    group.throughput(Throughput::Elements(
        u64::try_from(input.post_periods.len()).expect("post period count fits u64"),
    ));
    group.bench_function("smoothness_3pre_2post", |bencher| {
        bencher.iter(|| {
            assess_honest_event_study_workflow(
                &input,
                &functionals,
                HonestSensitivity::Smoothness(0.5),
                InferenceConfig::new(0.95),
                0.0,
            )
            .expect("honest workflow")
        });
    });
    group.finish();
}

fn build_relative_magnitude_functionals(num_post_periods: usize) -> Vec<HonestPostFunctional> {
    let mid_period = i32::try_from(num_post_periods / 2).expect("mid period fits i32");
    let last_period = i32::try_from(num_post_periods.saturating_sub(1)).expect("last fits i32");
    let tail_start =
        i32::try_from(num_post_periods.saturating_sub(4)).expect("tail start fits i32");
    vec![
        HonestPostFunctional::Period(0),
        HonestPostFunctional::Period(mid_period),
        HonestPostFunctional::AverageWindow {
            start_period: 0,
            end_period: 3.min(last_period),
        },
        HonestPostFunctional::AverageWindow {
            start_period: tail_start,
            end_period: last_period,
        },
        HonestPostFunctional::Weighted(vec![(0, 0.15), (mid_period, 0.35), (last_period, 0.50)]),
    ]
}

fn configure_heavy_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

fn build_basis_directions(num_post_periods: usize) -> Vec<Vec<f64>> {
    let mut directions = vec![vec![0.0; num_post_periods]; num_post_periods];
    for (index, direction) in directions.iter_mut().enumerate() {
        direction[index] = 1.0;
    }
    directions
}

fn bench_relative_magnitude_scalar_functionals(criterion: &mut Criterion) {
    let inference = InferenceConfig::new(0.95);
    let relative_magnitude_config = RelativeMagnitudeConfidenceSetConfig::from_inference(inference);
    let mut group = criterion.benchmark_group("relative_magnitude_scalar_functionals");
    configure_heavy_group(&mut group);
    for (num_pre, num_post) in [(3usize, 6usize), (4usize, 8usize)] {
        let input = build_honest_input(num_pre, num_post);
        let functionals = build_relative_magnitude_functionals(num_post);
        let selected = [
            ("period0", functionals[0].clone()),
            ("mid_period", functionals[1].clone()),
            ("weighted_tail", functionals[4].clone()),
        ];
        for (label, functional) in selected {
            group.throughput(Throughput::Elements(1));
            group.bench_function(format!("{num_pre}pre_{num_post}post/{label}"), |bencher| {
                bencher.iter(|| {
                    let output = assess_honest_event_study_post_functional_with_config(
                        &input,
                        &functional,
                        HonestSensitivity::RelativeMagnitude(1.0),
                        inference,
                        0.0,
                        relative_magnitude_config,
                    )
                    .expect("relative-magnitude scalar functional");
                    std::hint::black_box(output);
                });
            });
        }
    }
    group.finish();
}

fn bench_smoothness_scalar_functionals(criterion: &mut Criterion) {
    let inference = InferenceConfig::new(0.95);
    let mut group = criterion.benchmark_group("smoothness_scalar_functionals");
    configure_heavy_group(&mut group);
    for (num_pre, num_post) in [(3usize, 6usize), (4usize, 8usize)] {
        let input = build_honest_input(num_pre, num_post);
        let functionals = build_relative_magnitude_functionals(num_post);
        let selected = [
            ("period0", functionals[0].clone()),
            ("mid_period", functionals[1].clone()),
            ("weighted_tail", functionals[4].clone()),
        ];
        for (label, functional) in selected {
            group.throughput(Throughput::Elements(1));
            group.bench_function(format!("{num_pre}pre_{num_post}post/{label}"), |bencher| {
                bencher.iter(|| {
                    let output = did_methods::assess_honest_event_study_post_functional(
                        &input,
                        &functional,
                        HonestSensitivity::Smoothness(0.5),
                        inference,
                        0.0,
                    )
                    .expect("smoothness scalar functional");
                    std::hint::black_box(output);
                });
            });
        }
    }
    group.finish();
}

fn bench_relative_magnitude_multi_flci(criterion: &mut Criterion) {
    let inference = InferenceConfig::new(0.95);
    let relative_magnitude_config = RelativeMagnitudeConfidenceSetConfig::from_inference(inference);
    let joint = HonestJointPathConfig::default_for_production();
    let mut group = criterion.benchmark_group("relative_magnitude_multi_flci");
    configure_heavy_group(&mut group);
    for (num_pre, num_post) in [(3usize, 6usize), (4usize, 8usize)] {
        let input = build_honest_input(num_pre, num_post);
        let functionals = build_relative_magnitude_functionals(num_post)
            .into_iter()
            .take(3)
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(
            u64::try_from(functionals.len()).expect("functional count fits u64"),
        ));
        group.bench_function(format!("{num_pre}pre_{num_post}post"), |bencher| {
            bencher.iter(|| {
                let output = assess_honest_event_study_post_functional_multi_flci_with_config(
                    &input,
                    &functionals,
                    1.0,
                    inference,
                    relative_magnitude_config,
                    joint,
                )
                .expect("relative-magnitude multi flci");
                std::hint::black_box(output);
            });
        });
    }
    group.finish();
}

fn bench_relative_magnitude_workflow(criterion: &mut Criterion) {
    let inference = InferenceConfig::new(0.95);
    let relative_magnitude_config = RelativeMagnitudeConfidenceSetConfig::from_inference(inference);
    let joint = HonestJointPathConfig::default_for_production();
    let mut group = criterion.benchmark_group("relative_magnitude_workflow");
    configure_heavy_group(&mut group);
    for (num_pre, num_post) in [(3usize, 6usize), (4usize, 8usize)] {
        let input = build_honest_input(num_pre, num_post);
        let functionals = build_relative_magnitude_functionals(num_post)
            .into_iter()
            .take(3)
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(
            u64::try_from(functionals.len() * num_post).expect("throughput fits u64"),
        ));
        group.bench_function(format!("{num_pre}pre_{num_post}post"), |bencher| {
            bencher.iter(|| {
                let output = assess_honest_event_study_workflow(
                    &input,
                    &functionals,
                    HonestSensitivity::RelativeMagnitude(1.0),
                    inference,
                    0.0,
                )
                .expect("relative-magnitude workflow");
                std::hint::black_box(output);
            });
        });
        group.bench_function(
            format!("{num_pre}pre_{num_post}post_basis_cfg"),
            |bencher| {
                bencher.iter(|| {
                    let output = did_methods::assess_honest_event_study_workflow_with_config(
                        &input,
                        &functionals,
                        HonestSensitivity::RelativeMagnitude(1.0),
                        inference,
                        0.0,
                        &HonestWorkflowConfig {
                            relative_magnitude: relative_magnitude_config,
                            joint,
                            direction_mode: HonestWorkflowDirectionMode::Basis,
                        },
                    )
                    .expect("relative-magnitude configured workflow");
                    std::hint::black_box(output);
                });
            },
        );
    }
    group.finish();
}

fn bench_relative_magnitude_joint_path_region(criterion: &mut Criterion) {
    let inference = InferenceConfig::new(0.95);
    let relative_magnitude_config = RelativeMagnitudeConfidenceSetConfig::from_inference(inference);
    let joint = HonestJointPathConfig::default_for_production();
    let mut group = criterion.benchmark_group("relative_magnitude_joint_path_region");
    configure_heavy_group(&mut group);
    for (num_pre, num_post) in [(3usize, 6usize), (4usize, 8usize)] {
        let input = build_honest_input(num_pre, num_post);
        group.throughput(Throughput::Elements(
            u64::try_from(num_post).expect("post-period count fits u64"),
        ));
        group.bench_function(format!("{num_pre}pre_{num_post}post"), |bencher| {
            bencher.iter(|| {
                let output = assess_honest_event_study_joint_path_region_with_config(
                    &input,
                    HonestSensitivity::RelativeMagnitude(1.0),
                    inference,
                    0.0,
                    relative_magnitude_config,
                    joint,
                )
                .expect("relative-magnitude joint path region");
                std::hint::black_box(output);
            });
        });
    }
    group.finish();
}

fn bench_relative_magnitude_directional_basis_region(criterion: &mut Criterion) {
    let inference = InferenceConfig::new(0.95);
    let relative_magnitude_config = RelativeMagnitudeConfidenceSetConfig::from_inference(inference);
    let joint = HonestJointPathConfig::default_for_production();
    let mut group = criterion.benchmark_group("relative_magnitude_directional_basis_region");
    configure_heavy_group(&mut group);
    for (num_pre, num_post) in [(3usize, 6usize), (4usize, 8usize)] {
        let input = build_honest_input(num_pre, num_post);
        let directions = build_basis_directions(num_post);
        group.throughput(Throughput::Elements(
            u64::try_from(directions.len()).expect("direction count fits u64"),
        ));
        group.bench_function(format!("{num_pre}pre_{num_post}post"), |bencher| {
            bencher.iter(|| {
                let output = assess_honest_event_study_directional_region_with_config(
                    &input,
                    &directions,
                    HonestSensitivity::RelativeMagnitude(1.0),
                    inference,
                    0.0,
                    relative_magnitude_config,
                    joint,
                )
                .expect("relative-magnitude directional basis region");
                std::hint::black_box(output);
            });
        });
    }
    group.finish();
}

fn bench_did_cc(criterion: &mut Criterion) {
    let rows = build_did_cc_rows(4_000);
    let throughput = u64::try_from(rows.len()).expect("row count fits u64");
    let config = DidCcConfig::default();
    let cross_fit_config = DidCcConfig {
        drdid: DrDidConfig {
            bootstrap_reps: 19,
            max_iter: 1_000,
            tol: 1e-6,
            ..DrDidConfig::default()
        },
        cross_fit_folds: 4,
        cross_fit_seed: 17,
        ..DidCcConfig::default()
    };
    let cross_fit_rows = build_did_cc_rows(1_000);
    let mut group = criterion.benchmark_group("did_cc");
    group.throughput(Throughput::Elements(throughput));

    group.bench_function("robust_4k_per_cell", |bencher| {
        bencher.iter(|| estimate_did_cc_robust(&rows, config).expect("did_cc robust"));
    });
    group.bench_function("stationary_4k_per_cell", |bencher| {
        bencher.iter(|| estimate_did_cc_stationary(&rows, config).expect("did_cc stationary"));
    });
    group.bench_function("hausman_4k_per_cell", |bencher| {
        bencher.iter(|| test_did_cc_stationarity(&rows, config).expect("did_cc hausman"));
    });
    group.bench_function("robust_cross_fit_1k_per_cell", |bencher| {
        bencher.iter(|| {
            estimate_did_cc_robust(&cross_fit_rows, cross_fit_config)
                .expect("did_cc robust cross-fit")
        });
    });
    group.bench_function("stationary_cross_fit_1k_per_cell", |bencher| {
        bencher.iter(|| {
            estimate_did_cc_stationary(&cross_fit_rows, cross_fit_config)
                .expect("did_cc stationary cross-fit")
        });
    });
    group.bench_function("hausman_cross_fit_1k_per_cell", |bencher| {
        bencher.iter(|| {
            test_did_cc_stationarity(&cross_fit_rows, cross_fit_config)
                .expect("did_cc hausman cross-fit")
        });
    });

    group.finish();
}

fn bench_multiplier_bootstrap_ci(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("multiplier_bootstrap_ci");
    for n in [200usize, 1_000, 5_000] {
        let influence_function = (0..n)
            .map(|index| {
                let index_f = usize_to_f64(index);
                0.25f64.mul_add((0.17 * index_f).cos(), (0.03 * index_f).sin())
            })
            .collect::<Vec<_>>();
        let throughput = u64::try_from(n).expect("influence length fits u64");
        group.throughput(Throughput::Elements(throughput));
        group.bench_function(format!("n{n}_reps999"), |bencher| {
            bencher.iter(|| {
                multiplier_bootstrap_ci(
                    0.7,
                    &influence_function,
                    InferenceConfig::new(0.95),
                    BootstrapConfig {
                        reps: 999,
                        seed: 17_431,
                    },
                )
            });
        });
    }
    group.finish();
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn quantile_two_selection(mut values: Vec<f64>, alpha: f64) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let n_minus_one_f = usize_to_f64(n.saturating_sub(1));
    let lower_index = ((alpha / 2.0) * n_minus_one_f).floor() as usize;
    let upper_index = ((1.0 - alpha / 2.0) * n_minus_one_f).floor() as usize;

    let (_, lower, _) = values.select_nth_unstable_by(lower_index, f64::total_cmp);
    let lower_value = *lower;
    let (_, upper, _) = values.select_nth_unstable_by(upper_index, f64::total_cmp);
    let upper_value = *upper;
    (lower_value, upper_value)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn quantile_full_sort(mut values: Vec<f64>, alpha: f64) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    values.sort_by(f64::total_cmp);
    let n_minus_one_f = usize_to_f64(n.saturating_sub(1));
    let lower_index = ((alpha / 2.0) * n_minus_one_f).floor() as usize;
    let upper_index = ((1.0 - alpha / 2.0) * n_minus_one_f).floor() as usize;
    (values[lower_index], values[upper_index])
}

fn bench_quantile_extraction(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("quantile_extraction");
    for reps in [200usize, 500, 1_000] {
        let samples = (0..reps)
            .map(|idx| {
                let idx_f = usize_to_f64(idx + 1);
                0.43f64.mul_add((0.017 * idx_f).cos(), (0.031 * idx_f).sin())
            })
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(
            u64::try_from(reps).expect("reps fits u64"),
        ));
        group.bench_function(format!("two_selection/r{reps}"), |bencher| {
            bencher.iter(|| {
                let out = quantile_two_selection(samples.clone(), 0.05);
                std::hint::black_box(out);
            });
        });
        group.bench_function(format!("full_sort/r{reps}"), |bencher| {
            bencher.iter(|| {
                let out = quantile_full_sort(samples.clone(), 0.05);
                std::hint::black_box(out);
            });
        });
    }
    group.finish();
}

fn covariance_matrix_reference(influence_functions: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = influence_functions[0].len();
    let n_f = usize_to_f64(n);
    let k = influence_functions.len();
    let means = influence_functions
        .iter()
        .map(|influence| influence.iter().sum::<f64>() / n_f)
        .collect::<Vec<_>>();
    let mut covariance = vec![vec![0.0; k]; k];
    for left in 0..k {
        for right in left..k {
            let sum_prod = influence_functions[left]
                .iter()
                .zip(influence_functions[right].iter())
                .map(|(l, r)| (l - means[left]) * (r - means[right]))
                .sum::<f64>();
            let val = sum_prod / (n_f * n_f);
            covariance[left][right] = val;
            covariance[right][left] = val;
        }
    }
    covariance
}

fn bench_covariance_matrix(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("covariance_matrix_from_influence_matrix");
    for (k, n) in [(8usize, 2_000usize), (16, 10_000)] {
        let influence_functions = (0..k)
            .map(|estimate_idx| {
                let estimate_f = usize_to_f64(estimate_idx + 1);
                (0..n)
                    .map(|sample_idx| {
                        let sample_f = usize_to_f64(sample_idx + 1);
                        0.01f64.mul_add(estimate_f, (0.003 * estimate_f * sample_f).sin())
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(
            u64::try_from(k.saturating_mul(n)).expect("k*n fits u64"),
        ));
        group.bench_function(format!("current/k{k}_n{n}"), |bencher| {
            bencher.iter(|| {
                let _ = covariance_matrix_from_influence_matrix(&influence_functions)
                    .expect("covariance matrix");
            });
        });
        group.bench_function(format!("reference/k{k}_n{n}"), |bencher| {
            bencher.iter(|| {
                let _ = covariance_matrix_reference(&influence_functions);
            });
        });
    }
    group.finish();
}

fn lower_tri_mul_nested(lower: &[Vec<f64>], vec: &[f64], out: &mut [f64]) {
    for (row_idx, row) in lower.iter().enumerate() {
        out[row_idx] = row[..=row_idx]
            .iter()
            .zip(&vec[..=row_idx])
            .map(|(l, v)| l * v)
            .sum();
    }
}

fn pack_lower_triangular(lower: &[Vec<f64>]) -> Vec<f64> {
    let n = lower.len();
    let mut packed = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..=row {
            packed[row * n + col] = lower[row][col];
        }
    }
    packed
}

fn lower_tri_mul_packed(lower_packed: &[f64], dim: usize, vec: &[f64], out: &mut [f64]) {
    for (row, out_cell) in out.iter_mut().enumerate().take(dim) {
        let row_offset = row * dim;
        let mut sum = 0.0;
        for col in 0..=row {
            sum = lower_packed[row_offset + col].mul_add(vec[col], sum);
        }
        *out_cell = sum;
    }
}

fn bench_sensitivity_lower_triangular_kernel(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sensitivity_lower_triangular_kernel");
    for dim in [16usize, 64, 128] {
        let lower = (0..dim)
            .map(|row| {
                (0..dim)
                    .map(|col| {
                        if col <= row {
                            (0.01 * usize_to_f64(row + 1) * usize_to_f64(col + 1)).sin()
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let packed = pack_lower_triangular(&lower);
        let v = (0..dim)
            .map(|idx| (0.03 * usize_to_f64(idx + 1)).cos())
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(
            u64::try_from(dim * dim).expect("dim^2 fits u64"),
        ));
        group.bench_function(format!("nested/dim{dim}"), |bencher| {
            bencher.iter(|| {
                let mut out = vec![0.0; dim];
                lower_tri_mul_nested(&lower, &v, &mut out);
                std::hint::black_box(out);
            });
        });
        group.bench_function(format!("packed/dim{dim}"), |bencher| {
            bencher.iter(|| {
                let mut out = vec![0.0; dim];
                lower_tri_mul_packed(&packed, dim, &v, &mut out);
                std::hint::black_box(out);
            });
        });
    }
    group.finish();
}

fn bench_sensitivity_lower_triangular_batched(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sensitivity_lower_triangular_batched");
    for (dim, draws) in [(16usize, 64usize), (64, 64), (64, 256)] {
        let lower = (0..dim)
            .map(|row| {
                (0..dim)
                    .map(|col| {
                        if col <= row {
                            (0.01 * usize_to_f64(row + 1) * usize_to_f64(col + 1)).sin()
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let draws_mat = Mat::from_fn(dim, draws, |row, col| {
            (0.001 * usize_to_f64((row + 1) * (col + 3))).cos()
        });
        let lower_mat = Mat::from_fn(dim, dim, |row, col| lower[row][col]);
        group.throughput(Throughput::Elements(
            u64::try_from(dim * draws).expect("dim*draws fits u64"),
        ));
        group.bench_function(format!("scalar_loop/dim{dim}_draws{draws}"), |bencher| {
            bencher.iter(|| {
                let mut out = vec![vec![0.0; dim]; draws];
                let mut draw_vec = vec![0.0; dim];
                for draw in 0..draws {
                    for row in 0..dim {
                        draw_vec[row] = draws_mat[(row, draw)];
                    }
                    lower_tri_mul_nested(&lower, &draw_vec, &mut out[draw]);
                }
                std::hint::black_box(out);
            });
        });
        group.bench_function(format!("faer_batched/dim{dim}_draws{draws}"), |bencher| {
            bencher.iter(|| {
                let out = &lower_mat * &draws_mat;
                std::hint::black_box(out);
            });
        });
    }
    group.finish();
}

#[allow(clippy::many_single_char_names)]
fn normal_equations_loops(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    w: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![0.0; p * p];
    let mut b = vec![0.0; p];
    for row in 0..n {
        let x_row = &x[row * p..(row + 1) * p];
        let row_weight = w[row];
        for i in 0..p {
            b[i] = (row_weight * x_row[i]).mul_add(y[row], b[i]);
            for j in i..p {
                let v = row_weight * x_row[i] * x_row[j];
                a[i * p + j] += v;
                if i != j {
                    a[j * p + i] += v;
                }
            }
        }
    }
    (a, b)
}

#[allow(clippy::many_single_char_names)]
fn normal_equations_faer(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    w: &[f64],
) -> (Mat<f64>, Mat<f64>) {
    let design = Mat::from_fn(n, p, |row, col| x[row * p + col]);
    let weighted_design = Mat::from_fn(n, p, |row, col| design[(row, col)] * w[row].sqrt());
    let weighted_outcome = Mat::from_fn(n, 1, |row, _| w[row] * y[row]);
    let a = weighted_design.transpose() * &weighted_design;
    let b = design.transpose() * weighted_outcome;
    (a, b)
}

fn bench_panel_weighted_cross_products(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("panel_weighted_cross_products");
    for (n, p) in [(5_000usize, 6usize), (20_000, 6), (20_000, 12)] {
        let x = (0..(n * p))
            .map(|idx| {
                let idx_f = usize_to_f64(idx + 1);
                0.5f64.mul_add((0.0003 * idx_f).cos(), (0.001 * idx_f).sin())
            })
            .collect::<Vec<_>>();
        let y = (0..n)
            .map(|idx| {
                let idx_f = usize_to_f64(idx + 1);
                (0.004 * idx_f).cos()
            })
            .collect::<Vec<_>>();
        let w = (0..n)
            .map(|idx| {
                let idx_f = usize_to_f64(idx + 1);
                0.9f64.mul_add((0.002 * idx_f).sin().abs(), 0.1)
            })
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(
            u64::try_from(n * p).expect("n*p fits u64"),
        ));
        group.bench_function(format!("loops/n{n}_p{p}"), |bencher| {
            bencher.iter(|| {
                let out = normal_equations_loops(&x, n, p, &y, &w);
                std::hint::black_box(out);
            });
        });
        group.bench_function(format!("faer/n{n}_p{p}"), |bencher| {
            bencher.iter(|| {
                let out = normal_equations_faer(&x, n, p, &y, &w);
                std::hint::black_box(out);
            });
        });
    }
    group.finish();
}

fn weighted_empirical_logit_local(treated: &[f64], weights: &[f64], clip: f64) -> f64 {
    let treated_weight = treated
        .iter()
        .zip(weights.iter())
        .map(|(treated_value, weight)| treated_value * weight)
        .sum::<f64>();
    let total_weight = weights.iter().sum::<f64>();
    let share = (treated_weight / total_weight).clamp(clip, 1.0 - clip);
    (share / (1.0 - share)).ln()
}

fn weighted_logit_initial_loops(
    design_matrix_flat: &[f64],
    feature_count: usize,
    treated_indicator: &[f64],
    normalized_weights: &[f64],
    clip: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let mut coefficients = vec![0.0; feature_count];
    coefficients[0] = weighted_empirical_logit_local(treated_indicator, normalized_weights, clip);
    let observation_count = treated_indicator.len();
    let mut linear_index = vec![0.0; observation_count];
    let mut probabilities = vec![0.0; observation_count];
    let mut working_response = vec![0.0; observation_count];

    for _ in 0..max_iter {
        for row_index in 0..observation_count {
            let row =
                &design_matrix_flat[row_index * feature_count..(row_index + 1) * feature_count];
            let index = row
                .iter()
                .zip(coefficients.iter())
                .map(|(x, b)| x * b)
                .sum::<f64>()
                .clamp(-35.0, 35.0);
            linear_index[row_index] = index;
            let probability = 1.0 / (1.0 + (-index).exp());
            probabilities[row_index] = probability.clamp(clip, 1.0 - clip);
        }
        let mut weighted_crossprod = vec![0.0; feature_count * feature_count];
        let mut weighted_response = vec![0.0; feature_count];
        for row_index in 0..observation_count {
            let row =
                &design_matrix_flat[row_index * feature_count..(row_index + 1) * feature_count];
            let probability = probabilities[row_index];
            let variance = (probability * (1.0 - probability)).max(1e-8);
            let weight = normalized_weights[row_index] * variance;
            working_response[row_index] =
                linear_index[row_index] + (treated_indicator[row_index] - probability) / variance;
            for left in 0..feature_count {
                weighted_response[left] = (weight * row[left])
                    .mul_add(working_response[row_index], weighted_response[left]);
                for right in left..feature_count {
                    let outer = weight * row[left] * row[right];
                    weighted_crossprod[left * feature_count + right] += outer;
                    if left != right {
                        weighted_crossprod[right * feature_count + left] += outer;
                    }
                }
            }
        }
        for diagonal_index in 0..feature_count {
            weighted_crossprod[diagonal_index * feature_count + diagonal_index] += 1e-8;
        }
        let Some(next) = solve_system_faer(&weighted_crossprod, &weighted_response) else {
            break;
        };
        let max_change = next
            .iter()
            .zip(coefficients.iter())
            .map(|(new_value, old_value)| (new_value - old_value).abs())
            .fold(0.0, f64::max);
        coefficients = next;
        if max_change <= tol.max(1e-10) {
            break;
        }
    }
    coefficients
}

fn solve_system_faer(a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let system = Mat::from_fn(n, n, |row, col| a[row * n + col]);
    let rhs = Mat::from_fn(n, 1, |row, _| b[row]);
    let solution = system.full_piv_lu().solve(rhs);
    let mut out = vec![0.0; n];
    for row in 0..n {
        let value = solution[(row, 0)];
        if !value.is_finite() {
            return None;
        }
        out[row] = value;
    }
    Some(out)
}

fn weighted_logit_initial_faer(
    design_matrix_flat: &[f64],
    feature_count: usize,
    treated_indicator: &[f64],
    normalized_weights: &[f64],
    clip: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let mut coefficients = vec![0.0; feature_count];
    coefficients[0] = weighted_empirical_logit_local(treated_indicator, normalized_weights, clip);
    let observation_count = treated_indicator.len();
    let design = Mat::from_fn(observation_count, feature_count, |row, col| {
        design_matrix_flat[row * feature_count + col]
    });
    let mut linear_index = vec![0.0; observation_count];
    let mut probabilities = vec![0.0; observation_count];
    let mut working_response = vec![0.0; observation_count];
    let mut row_weights = vec![0.0; observation_count];

    for _ in 0..max_iter {
        for row_index in 0..observation_count {
            let row =
                &design_matrix_flat[row_index * feature_count..(row_index + 1) * feature_count];
            let index = row
                .iter()
                .zip(coefficients.iter())
                .map(|(x, b)| x * b)
                .sum::<f64>()
                .clamp(-35.0, 35.0);
            linear_index[row_index] = index;
            let probability = 1.0 / (1.0 + (-index).exp());
            probabilities[row_index] = probability.clamp(clip, 1.0 - clip);
        }
        for row_index in 0..observation_count {
            let probability = probabilities[row_index];
            let variance = (probability * (1.0 - probability)).max(1e-8);
            let weight = normalized_weights[row_index] * variance;
            row_weights[row_index] = weight;
            working_response[row_index] =
                linear_index[row_index] + (treated_indicator[row_index] - probability) / variance;
        }
        let weighted_design = Mat::from_fn(observation_count, feature_count, |row, col| {
            design[(row, col)] * row_weights[row].sqrt()
        });
        let weighted_working_response = Mat::from_fn(observation_count, 1, |row, _| {
            row_weights[row] * working_response[row]
        });
        let weighted_crossprod_mat = weighted_design.transpose() * &weighted_design;
        let weighted_response_mat = design.transpose() * weighted_working_response;
        let mut weighted_crossprod = vec![0.0; feature_count * feature_count];
        for row in 0..feature_count {
            for col in 0..feature_count {
                weighted_crossprod[row * feature_count + col] = weighted_crossprod_mat[(row, col)];
            }
        }
        for diagonal_index in 0..feature_count {
            weighted_crossprod[diagonal_index * feature_count + diagonal_index] += 1e-8;
        }
        let mut weighted_response = vec![0.0; feature_count];
        for row in 0..feature_count {
            weighted_response[row] = weighted_response_mat[(row, 0)];
        }
        let Some(next) = solve_system_faer(&weighted_crossprod, &weighted_response) else {
            break;
        };
        let max_change = next
            .iter()
            .zip(coefficients.iter())
            .map(|(new_value, old_value)| (new_value - old_value).abs())
            .fold(0.0, f64::max);
        coefficients = next;
        if max_change <= tol.max(1e-10) {
            break;
        }
    }
    coefficients
}

fn bench_improved_repeated_logit_initial_kernel(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("improved_repeated_logit_initial_kernel");
    for (n, p) in [(2_000usize, 6usize), (8_000, 6), (8_000, 12)] {
        let design_matrix_flat = (0..(n * p))
            .map(|idx| {
                let idx_f = usize_to_f64(idx + 1);
                0.35f64.mul_add((0.0007 * idx_f).cos(), (0.0017 * idx_f).sin())
            })
            .collect::<Vec<_>>();
        let treated_indicator = (0..n)
            .map(|idx| if idx % 3 == 0 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let normalized_weights = vec![1.0; n];
        group.throughput(Throughput::Elements(
            u64::try_from(n * p).expect("n*p fits u64"),
        ));
        group.bench_function(format!("loops/n{n}_p{p}"), |bencher| {
            bencher.iter(|| {
                let out = weighted_logit_initial_loops(
                    &design_matrix_flat,
                    p,
                    &treated_indicator,
                    &normalized_weights,
                    1e-6,
                    50,
                    1e-8,
                );
                std::hint::black_box(out);
            });
        });
        group.bench_function(format!("faer/n{n}_p{p}"), |bencher| {
            bencher.iter(|| {
                let out = weighted_logit_initial_faer(
                    &design_matrix_flat,
                    p,
                    &treated_indicator,
                    &normalized_weights,
                    1e-6,
                    50,
                    1e-8,
                );
                std::hint::black_box(out);
            });
        });
    }
    group.finish();
}

fn bench_sensitivity_normal_draws(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sensitivity_normal_draws");
    for len in [32usize, 128, 512, 2_048] {
        group.throughput(Throughput::Elements(
            u64::try_from(len).expect("len fits u64"),
        ));
        group.bench_function(format!("current/len{len}"), |bencher| {
            bencher.iter(|| {
                let out = benchmark_sensitivity_normal_draws(len, 27_019);
                std::hint::black_box(out);
            });
        });
        group.bench_function(format!("scalar_ref/len{len}"), |bencher| {
            bencher.iter(|| {
                let out = benchmark_sensitivity_normal_draws_scalar(len, 27_019);
                std::hint::black_box(out);
            });
        });
    }
    group.finish();
}

fn matrix_rank_reference(matrix: &[Vec<f64>], tol: f64) -> usize {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0;
    }
    let mut m = matrix.to_vec();
    let rows = m.len();
    let cols = m[0].len();
    let mut rank = 0usize;
    let mut row = 0usize;
    for col in 0..cols {
        let pivot =
            (row..rows).max_by(|&left, &right| m[left][col].abs().total_cmp(&m[right][col].abs()));
        let Some(pivot_row) = pivot else { break };
        if m[pivot_row][col].abs() <= tol {
            continue;
        }
        m.swap(row, pivot_row);
        let pivot_value = m[row][col];
        for value in &mut m[row][col..] {
            *value /= pivot_value;
        }
        for r in 0..rows {
            if r == row {
                continue;
            }
            let factor = m[r][col];
            if factor.abs() <= tol {
                continue;
            }
            let pivot_snapshot: Vec<f64> = m[row][col..].to_vec();
            for (target, pivot_entry) in m[r][col..].iter_mut().zip(pivot_snapshot.iter()) {
                *target = factor.mul_add(-*pivot_entry, *target);
            }
        }
        rank += 1;
        row += 1;
        if row == rows {
            break;
        }
    }
    rank
}

fn bench_sensitivity_matrix_rank(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sensitivity_matrix_rank");
    for dim in [16usize, 32, 64] {
        let matrix = (0..dim)
            .map(|row| {
                (0..dim)
                    .map(|col| {
                        let base = (0.013 * usize_to_f64(row + 1) * usize_to_f64(col + 1)).sin();
                        if row == col { base + 1.0 } else { base }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(
            u64::try_from(dim * dim).expect("dim^2 fits u64"),
        ));
        group.bench_function(format!("current/dim{dim}"), |bencher| {
            bencher.iter(|| {
                let out = benchmark_sensitivity_matrix_rank(&matrix, 1e-10);
                std::hint::black_box(out);
            });
        });
        group.bench_function(format!("reference/dim{dim}"), |bencher| {
            bencher.iter(|| {
                let out = matrix_rank_reference(&matrix, 1e-10);
                std::hint::black_box(out);
            });
        });
    }
    group.finish();
}

fn bench_sensitivity_sandwich_covariance(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sensitivity_sandwich_covariance");
    for (rows, cols) in [(8usize, 16usize), (16, 32), (24, 48)] {
        let left = (0..rows)
            .map(|row| {
                (0..cols)
                    .map(|col| {
                        (0.017 * usize_to_f64(row + 1) * usize_to_f64(col + 2)).sin()
                            + if row == col % rows { 0.25 } else { 0.0 }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let sigma = (0..cols)
            .map(|row| {
                (0..cols)
                    .map(|col| {
                        let distance = usize_to_f64(row.abs_diff(col));
                        if row == col {
                            usize_to_f64(row).mul_add(0.01, 1.0)
                        } else {
                            0.12 / (1.0 + distance)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(
            u64::try_from(rows * cols * cols).expect("benchmark size fits u64"),
        ));
        group.bench_function(format!("rows{rows}_cols{cols}"), |bencher| {
            bencher.iter(|| {
                let out = benchmark_sensitivity_sandwich_covariance(&left, &sigma);
                std::hint::black_box(out);
            });
        });
    }
    group.finish();
}

fn rref_pivot_columns_reference(matrix: &[Vec<f64>], tol: f64) -> usize {
    if matrix.is_empty() {
        return 0;
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut m = matrix.to_vec();
    let mut pivot_columns = Vec::new();
    let mut pivot_row = 0usize;
    for col in 0..cols {
        if pivot_row >= rows {
            break;
        }
        let Some(best_row) = (pivot_row..rows)
            .max_by(|&left, &right| m[left][col].abs().total_cmp(&m[right][col].abs()))
        else {
            break;
        };
        if m[best_row][col].abs() <= tol {
            continue;
        }
        m.swap(pivot_row, best_row);
        let pivot_value = m[pivot_row][col];
        for value in &mut m[pivot_row] {
            *value /= pivot_value;
        }
        for row in 0..rows {
            if row == pivot_row {
                continue;
            }
            let factor = m[row][col];
            if factor.abs() <= tol {
                continue;
            }
            let pivot_snapshot = m[pivot_row].clone();
            for (target, pivot_entry) in m[row].iter_mut().zip(pivot_snapshot.iter()) {
                *target = factor.mul_add(-*pivot_entry, *target);
            }
        }
        pivot_columns.push(col);
        pivot_row += 1;
    }
    pivot_columns.len()
}

fn bench_sensitivity_rref_pivot_columns(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sensitivity_rref_pivot_columns");
    for dim in [8usize, 16, 32] {
        let matrix = (0..dim)
            .map(|row| {
                (0..(dim * 2))
                    .map(|col| {
                        (0.021 * usize_to_f64(row + 1) * usize_to_f64(col + 1)).sin()
                            + if row == col % dim { 0.7 } else { 0.0 }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(
            u64::try_from(dim * dim * 2).expect("dim^2*2 fits u64"),
        ));
        group.bench_function(format!("current/dim{dim}"), |bencher| {
            bencher.iter(|| {
                let out = benchmark_sensitivity_rref_pivot_columns(&matrix, 1e-10);
                std::hint::black_box(out);
            });
        });
        group.bench_function(format!("reference/dim{dim}"), |bencher| {
            bencher.iter(|| {
                let out = rref_pivot_columns_reference(&matrix, 1e-10);
                std::hint::black_box(out);
            });
        });
    }
    group.finish();
}

fn bench_sensitivity_multi_flci_maxima(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sensitivity_multi_flci_maxima");
    for (dim, draws) in [(8usize, 64usize), (16, 64), (32, 128), (32, 512)] {
        let chol = (0..dim)
            .map(|row| {
                (0..dim)
                    .map(|col| match col.cmp(&row) {
                        std::cmp::Ordering::Greater => 0.0,
                        std::cmp::Ordering::Equal => 0.005f64.mul_add(usize_to_f64(row + 1), 1.0),
                        std::cmp::Ordering::Less => 0.02 * usize_to_f64(row - col + 1),
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(
            u64::try_from(dim * draws).expect("dim*draws fits u64"),
        ));
        group.bench_function(format!("current/dim{dim}_draws{draws}"), |bencher| {
            bencher.iter(|| {
                let out = benchmark_sensitivity_multi_flci_maxima(&chol, draws, 31_415);
                std::hint::black_box(out);
            });
        });
        group.bench_function(format!("scalar_ref/dim{dim}_draws{draws}"), |bencher| {
            bencher.iter(|| {
                let out = benchmark_sensitivity_multi_flci_maxima_scalar(&chol, draws, 31_415);
                std::hint::black_box(out);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_drdid_panel,
    bench_drdid_improved_repeated,
    bench_att_gt_dr,
    bench_honest_workflow,
    bench_relative_magnitude_scalar_functionals,
    bench_smoothness_scalar_functionals,
    bench_relative_magnitude_multi_flci,
    bench_relative_magnitude_workflow,
    bench_relative_magnitude_joint_path_region,
    bench_relative_magnitude_directional_basis_region,
    bench_did_cc,
    bench_multiplier_bootstrap_ci,
    bench_quantile_extraction,
    bench_covariance_matrix,
    bench_sensitivity_lower_triangular_kernel,
    bench_sensitivity_lower_triangular_batched,
    bench_panel_weighted_cross_products,
    bench_improved_repeated_logit_initial_kernel,
    bench_sensitivity_normal_draws,
    bench_sensitivity_matrix_rank,
    bench_sensitivity_sandwich_covariance,
    bench_sensitivity_rref_pivot_columns,
    bench_sensitivity_multi_flci_maxima
);
criterion_main!(benches);
