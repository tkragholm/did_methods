//! One Study I-shaped slice-anchor surface, serially, for profiling.
//!
//! Four pre-periods, sixteen post, twenty functionals (every horizon and four
//! windows), the production nine-point `Mbar` grid, the least-favorable
//! hybrid. This is what the pipeline's bridge runs per functional in parallel;
//! serial here so a profiler attributes time to functions rather than to the
//! rayon pool.
//!
//! `cargo run --release --example surface_profile --features dense-lp`
//! `cargo run --release --example surface_profile --features dense-lp,hotpath`

use std::time::Instant;

use did_methods::{
    HonestEventStudyInput, HonestRelativeMagnitudeBound, InferenceConfig,
    summarize_relative_magnitude_sensitivity, summarize_relative_magnitude_sensitivity_many,
};

fn study() -> HonestEventStudyInput {
    let pre: Vec<i32> = vec![-4, -3, -2, -1];
    let post: Vec<i32> = (0..16).collect();
    let dim = pre.len() + post.len();
    let mut betahat = Vec::with_capacity(dim);
    for i in 0..pre.len() {
        betahat.push(0.004 * (i as f64 - 2.0));
    }
    for h in 0..post.len() {
        betahat.push(-0.03 - 0.004 * h as f64);
    }
    let se: Vec<f64> = (0..dim).map(|i| 0.05 + 0.002 * i as f64).collect();
    let covariance = (0..dim)
        .map(|i| {
            (0..dim)
                .map(|j| se[i] * se[j] * 0.6f64.powi(i.abs_diff(j) as i32))
                .collect()
        })
        .collect();
    HonestEventStudyInput {
        betahat,
        covariance,
        pre_periods: pre,
        post_periods: post,
    }
}

fn functionals() -> Vec<(String, Vec<f64>)> {
    let mut out: Vec<(String, Vec<f64>)> = (0..16)
        .map(|h| {
            let mut w = vec![0.0; 16];
            w[h] = 1.0;
            (format!("h{h}"), w)
        })
        .collect();
    for (name, lo, hi) in [
        ("w1_3", 1, 3),
        ("w4_6", 4, 6),
        ("w7_10", 7, 10),
        ("w11_16", 11, 15),
    ] {
        let w = (0..16)
            .map(|k| {
                if (lo..=hi).contains(&k) {
                    1.0 / (hi - lo + 1) as f64
                } else {
                    0.0
                }
            })
            .collect();
        out.push((name.to_string(), w));
    }
    out
}

#[cfg_attr(feature = "hotpath", hotpath::main)]
fn main() {
    let input = study();
    let grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5];
    let inference = InferenceConfig::new(0.95);
    let repeats: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let many = std::env::args().nth(2).is_some_and(|mode| mode == "many");
    let start = Instant::now();
    let mut rows = 0usize;
    let mut checksum = 0.0;
    for _ in 0..repeats {
        if many {
            let functionals = functionals();
            let weights: Vec<&[f64]> = functionals.iter().map(|(_, w)| w.as_slice()).collect();
            let summaries = summarize_relative_magnitude_sensitivity_many(
                &input,
                inference,
                &weights,
                None,
                Some(&grid),
                Some(HonestRelativeMagnitudeBound::ParallelTrendsDeviation),
                None,
                None,
            )
            .expect("surface");
            for summary in summaries {
                rows += summary.rows.len();
                checksum += summary.rows.iter().map(|r| r.lb + r.ub).sum::<f64>();
            }
            continue;
        }
        for (_, weights) in functionals() {
            let summary = summarize_relative_magnitude_sensitivity(
                &input,
                inference,
                &weights,
                None,
                Some(&grid),
                Some(HonestRelativeMagnitudeBound::ParallelTrendsDeviation),
                None,
                None,
            )
            .expect("surface");
            rows += summary.rows.len();
            checksum += summary.rows.iter().map(|r| r.lb + r.ub).sum::<f64>();
        }
    }
    let elapsed = start.elapsed();
    println!(
        "{repeats} surface(s){}: {rows} rows, checksum {checksum:.9}, {:.3} s per surface",
        if many { ", many" } else { "" },
        elapsed.as_secs_f64() / repeats as f64
    );
}
