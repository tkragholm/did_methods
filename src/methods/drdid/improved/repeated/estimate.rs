//! Final ATT and influence-function assembly for improved repeated `DR-DiD`.
//!
//! This module implements the final combination step after nuisance models have
//! been estimated. Using Sant'Anna and Zhao's notation for repeated
//! cross-sections, it assembles the normalized sample analogues of the four
//! building blocks:
//!
//! - treated pre-period outcome contrast,
//! - treated post-period outcome contrast,
//! - control pre-period outcome contrast,
//! - control post-period outcome contrast,
//!
//! together with the locally efficient augmentation terms that arise from the
//! outcome-regression projections. The ATT is then
//!
//! ```text
//! ATT = (T_post - T_pre) - (C_post - C_pre) + (E_post - E_pre),
//! ```
//!
//! where each term is formed from the normalized weighted moments described in
//! Sant'Anna and Zhao (2020, Section 3.2). The influence function mirrors the
//! same decomposition and is returned explicitly so the shared inference layer
//! can construct clustered or heteroskedasticity-robust variance estimates.
//!
//! Reference:
//! - Sant'Anna, P. H. C. and Zhao, J. (2020). "Doubly Robust Difference-in-
//!   Differences Estimators". *Journal of Econometrics*.

use super::data::RepeatedPreparedData;
use super::nuisance::ImprovedRepeatedNuisanceFits;

pub(super) struct ImprovedRepeatedAttEstimate {
    pub(super) att: f64,
    pub(super) influence_function: Vec<f64>,
}

pub(super) fn estimate_improved_repeated_att(
    prepared: &RepeatedPreparedData,
    nuisance: &ImprovedRepeatedNuisanceFits,
) -> ImprovedRepeatedAttEstimate {
    let n = prepared.treated_indicator.len();
    let out_y_cont = nuisance.post_indicator_mix(&prepared.post_indicator);

    let weighted_terms = collect_weighted_terms(prepared, nuisance, &out_y_cont, n);
    let means = WeightedMeans::from_terms(&weighted_terms);
    let normalized = normalize_weighted_terms(weighted_terms, &means);
    let att = means.att();
    let influence_function = build_influence_function(&normalized, &means, n);

    ImprovedRepeatedAttEstimate {
        att,
        influence_function,
    }
}

struct WeightedTerms {
    w_treat_pre: Vec<f64>,
    w_treat_post: Vec<f64>,
    w_cont_pre: Vec<f64>,
    w_cont_post: Vec<f64>,
    w_d: Vec<f64>,
    w_dt1: Vec<f64>,
    w_dt0: Vec<f64>,
    eta_treat_pre: Vec<f64>,
    eta_treat_post: Vec<f64>,
    eta_cont_pre: Vec<f64>,
    eta_cont_post: Vec<f64>,
    eta_d_post: Vec<f64>,
    eta_dt1_post: Vec<f64>,
    eta_d_pre: Vec<f64>,
    eta_dt0_pre: Vec<f64>,
}

struct WeightedMeans {
    w_treat_pre: f64,
    w_treat_post: f64,
    w_cont_pre: f64,
    w_cont_post: f64,
    w_d: f64,
    w_dt1: f64,
    w_dt0: f64,
    att_treat_pre: f64,
    att_treat_post: f64,
    att_cont_pre: f64,
    att_cont_post: f64,
    att_d_post: f64,
    att_dt1_post: f64,
    att_d_pre: f64,
    att_dt0_pre: f64,
}

impl WeightedMeans {
    fn from_terms(terms: &WeightedTerms) -> Self {
        Self {
            w_treat_pre: mean(&terms.w_treat_pre),
            w_treat_post: mean(&terms.w_treat_post),
            w_cont_pre: mean(&terms.w_cont_pre),
            w_cont_post: mean(&terms.w_cont_post),
            w_d: mean(&terms.w_d),
            w_dt1: mean(&terms.w_dt1),
            w_dt0: mean(&terms.w_dt0),
            att_treat_pre: mean(&terms.eta_treat_pre),
            att_treat_post: mean(&terms.eta_treat_post),
            att_cont_pre: mean(&terms.eta_cont_pre),
            att_cont_post: mean(&terms.eta_cont_post),
            att_d_post: mean(&terms.eta_d_post),
            att_dt1_post: mean(&terms.eta_dt1_post),
            att_d_pre: mean(&terms.eta_d_pre),
            att_dt0_pre: mean(&terms.eta_dt0_pre),
        }
    }

    fn att(&self) -> f64 {
        (self.att_treat_post - self.att_treat_pre) - (self.att_cont_post - self.att_cont_pre)
            + (self.att_d_post - self.att_dt1_post)
            - (self.att_d_pre - self.att_dt0_pre)
    }
}

fn collect_weighted_terms(
    prepared: &RepeatedPreparedData,
    nuisance: &ImprovedRepeatedNuisanceFits,
    out_y_cont: &[f64],
    n: usize,
) -> WeightedTerms {
    let mut w_treat_pre = Vec::with_capacity(n);
    let mut w_treat_post = Vec::with_capacity(n);
    let mut w_cont_pre = Vec::with_capacity(n);
    let mut w_cont_post = Vec::with_capacity(n);
    let mut w_d = Vec::with_capacity(n);
    let mut w_dt1 = Vec::with_capacity(n);
    let mut w_dt0 = Vec::with_capacity(n);

    let mut eta_treat_pre = Vec::with_capacity(n);
    let mut eta_treat_post = Vec::with_capacity(n);
    let mut eta_cont_pre = Vec::with_capacity(n);
    let mut eta_cont_post = Vec::with_capacity(n);
    let mut eta_d_post = Vec::with_capacity(n);
    let mut eta_dt1_post = Vec::with_capacity(n);
    let mut eta_d_pre = Vec::with_capacity(n);
    let mut eta_dt0_pre = Vec::with_capacity(n);

    for (row_index, out_cont) in out_y_cont.iter().copied().enumerate().take(n) {
        let i_weight = nuisance.normalized_weights[row_index];
        let treated = prepared.treated_indicator[row_index];
        let post = prepared.post_indicator[row_index];
        let propensity_score = nuisance.propensity_scores[row_index];
        let trim = nuisance.trim_indicator[row_index];
        let outcome = prepared.outcome[row_index];
        let out_treat_pre = nuisance.out_y_treat_pre[row_index];
        let out_treat_post = nuisance.out_y_treat_post[row_index];
        let out_cont_pre = nuisance.out_y_cont_pre[row_index];
        let out_cont_post = nuisance.out_y_cont_post[row_index];

        let wtp = i_weight * treated * (1.0 - post);
        let wtx = i_weight * treated * post;
        let wcp = trim * i_weight * propensity_score * (1.0 - treated) * (1.0 - post)
            / (1.0 - propensity_score);
        let wcx =
            trim * i_weight * propensity_score * (1.0 - treated) * post / (1.0 - propensity_score);
        let wd = i_weight * treated;
        let wd1 = i_weight * treated * post;
        let wd0 = i_weight * treated * (1.0 - post);

        w_treat_pre.push(wtp);
        w_treat_post.push(wtx);
        w_cont_pre.push(wcp);
        w_cont_post.push(wcx);
        w_d.push(wd);
        w_dt1.push(wd1);
        w_dt0.push(wd0);

        eta_treat_pre.push(wtp * (outcome - out_cont));
        eta_treat_post.push(wtx * (outcome - out_cont));
        eta_cont_pre.push(wcp * (outcome - out_cont));
        eta_cont_post.push(wcx * (outcome - out_cont));
        eta_d_post.push(wd * (out_treat_post - out_cont_post));
        eta_dt1_post.push(wd1 * (out_treat_post - out_cont_post));
        eta_d_pre.push(wd * (out_treat_pre - out_cont_pre));
        eta_dt0_pre.push(wd0 * (out_treat_pre - out_cont_pre));
    }

    WeightedTerms {
        w_treat_pre,
        w_treat_post,
        w_cont_pre,
        w_cont_post,
        w_d,
        w_dt1,
        w_dt0,
        eta_treat_pre,
        eta_treat_post,
        eta_cont_pre,
        eta_cont_post,
        eta_d_post,
        eta_dt1_post,
        eta_d_pre,
        eta_dt0_pre,
    }
}

fn normalize_weighted_terms(mut terms: WeightedTerms, means: &WeightedMeans) -> WeightedTerms {
    scale_by_mean(&mut terms.eta_treat_pre, means.w_treat_pre);
    scale_by_mean(&mut terms.eta_treat_post, means.w_treat_post);
    scale_by_mean(&mut terms.eta_cont_pre, means.w_cont_pre);
    scale_by_mean(&mut terms.eta_cont_post, means.w_cont_post);
    scale_by_mean(&mut terms.eta_d_post, means.w_d);
    scale_by_mean(&mut terms.eta_dt1_post, means.w_dt1);
    scale_by_mean(&mut terms.eta_d_pre, means.w_d);
    scale_by_mean(&mut terms.eta_dt0_pre, means.w_dt0);
    terms
}

fn build_influence_function(terms: &WeightedTerms, means: &WeightedMeans, n: usize) -> Vec<f64> {
    let mut influence_function = Vec::with_capacity(n);
    for row_index in 0..n {
        let inf_treat_pre = terms.eta_treat_pre[row_index]
            - terms.w_treat_pre[row_index] * means.att_treat_pre / means.w_treat_pre;
        let inf_treat_post = terms.eta_treat_post[row_index]
            - terms.w_treat_post[row_index] * means.att_treat_post / means.w_treat_post;
        let inf_treat = inf_treat_post - inf_treat_pre;

        let inf_cont_pre = terms.eta_cont_pre[row_index]
            - terms.w_cont_pre[row_index] * means.att_cont_pre / means.w_cont_pre;
        let inf_cont_post = terms.eta_cont_post[row_index]
            - terms.w_cont_post[row_index] * means.att_cont_post / means.w_cont_post;
        let inf_cont = inf_cont_post - inf_cont_pre;

        let inf_eff1 =
            terms.eta_d_post[row_index] - terms.w_d[row_index] * means.att_d_post / means.w_d;
        let inf_eff2 = terms.eta_dt1_post[row_index]
            - terms.w_dt1[row_index] * means.att_dt1_post / means.w_dt1;
        let inf_eff3 =
            terms.eta_d_pre[row_index] - terms.w_d[row_index] * means.att_d_pre / means.w_d;
        let inf_eff4 =
            terms.eta_dt0_pre[row_index] - terms.w_dt0[row_index] * means.att_dt0_pre / means.w_dt0;
        let inf_eff = (inf_eff1 - inf_eff2) - (inf_eff3 - inf_eff4);

        influence_function.push((inf_treat - inf_cont) + inf_eff);
    }
    influence_function
}

fn scale_by_mean(values: &mut [f64], mean_value: f64) {
    for value in values {
        *value /= mean_value;
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / crate::util::usize_to_f64(values.len())
}
