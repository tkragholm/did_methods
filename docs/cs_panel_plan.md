# Full Callaway-Sant'Anna on panel data: what is done and what is left

Written 19 August 2026, when the panel DR route was added.

## Why

Study 1's `docs/estimator_choice.md` (12 August) ruled that the matched risk-set
design already removes the forbidden comparison, so Callaway-Sant'Anna had
nothing to add, and kept the fixed-baseline pairwise estimator as primary. That
reasoning is correct for the estimand the paper reports.

It stops being sufficient the moment the claim is about **how an effect evolves
with duration**, because in a pooled design duration is confounded with
diagnosis cohort: the years 11-15 window can only draw on families diagnosed by
2012, while the years 0-2 window draws on all of 2000-2018. Three things only
the group-time decomposition can give:

- `balance_e`: a dynamic path on a **fixed cohort composition**, which the
  fixed-baseline design cannot produce at all.
- calendar-time aggregation, which isolates a policy vintage from a duration.
- cohort aggregation, the vintage view itself.

Study 2 (`societal_costs`) commits to staggered group-time DiD in its draft
methods, so this is shared infrastructure rather than one paper's sensitivity.

## Reference and fixtures

`did` 2.5.1 and `DRDID` 1.3.0 are the reference. Python (`csdid`, `differences`)
are ports of the same R code, so they are a second opinion, never a tie-breaker.

`tests/generate_cs_panel_reference.R` writes four fixtures from `mpdta` and
records, in the file, every `att_gt` default it overrides. Two of those defaults
will otherwise produce failures that look like bugs and are not:
`bstrap = TRUE` (R's headline SEs are bootstrap, ours are analytic) and
`cband = TRUE` (R's intervals are simultaneous, ours pointwise). `base_period`
defaults to `"varying"` and must be `"universal"` to mean a fixed t-1 anchor;
`cs_panel_dr_varying_ref.json` exists so the difference is visible in numbers
rather than taken on trust.

## Done

`estimate_att_gt_dr_panel[_with_influence]` in
`src/methods/att_gt/panel_pairs.rs`, matching `did::att_gt(panel = TRUE,
est_method = "dr")`. Four parity tests in `tests/parity_cs_panel.rs`:
ATT and SE to 1e-10, the full influence matrix to 2.9e-9 worst case, both
comparison groups, and a refusal when unit ids are absent.

Three defects were found by the fixtures and fixed:

1. **The two DR routes disagreed about the intercept.**
   `estimate_drdid_repeated_cross_section` always prepends one; `estimate_drdid_panel`
   treats its covariates as a finished design matrix. The same
   `AttGtDrObservation` would have been adjusted for `1 + X` through one route
   and `X` alone through the other. Fixed at the ATT(g,t) layer, where R's
   `xformla = ~ x` semantics belong. The lower-level `estimate_drdid_panel`
   contract is unchanged, since its own parity fixture pins it.

2. **Not-yet-treated eligibility was judged at `time` instead of at
   `max(time, baseline)`.** Under a universal base period a pre-treatment cell
   has baseline later than time, so a unit treated in between was admitted as a
   control while already treated at the baseline it is read against. This
   contaminated exactly the placebo cells the HonestDiD pre-trend surface reads.
   Measured on mpdta cell (g=2006, t=2003, base=2005): 0.008901391 under the old
   rule, 0.012018613 under R's. Fixed in **both** the panel and the RC route.

3. **Influence functions must be rescaled to the full sample.** A cell's psi is
   normalised so that `sqrt(sum(psi^2)) / n_cell` is its SE. Zero-padding to
   `n_total` without rescaling shrinks each cell by a *different* factor, since
   cells differ in size, leaving a covariance matrix whose blocks sit on
   incompatible scales. `did` embeds the same way: measured on cell (2004, 2005),
   R's psi is ours times 500/329.

Note that influence vectors from the panel route are indexed by **unit**, not by
input row. R's SE is exactly `sqrt(sum(psi^2)) / n_units`, and a row-indexed
vector would be wrong by the number of periods.

## Left to do

**Everything on the previous list is done.** `min_e` / `max_e` match `did`
including their effect on the overall; the RC route has an aggregation via
`row_panel`; and the Python surface item was already closed outside this work,
see below.

What is left is one thing, and it is not a code change:

**`register-studies` cannot see any of this yet.** Its `Cargo.toml` takes
`did_methods` as a git dependency on `branch = "main"`, not as a path. Nothing
added here reaches Python until this crate is committed and pushed. That is a
decision for the repository owner, not a task.

Once it can, the change on the other side is small and specific:
`estimate_event_study` in `register-studies/src/rust/did.rs` builds its
`AttGtDrObservation` rows without a `unit_id` and calls
`estimate_att_gt_dr_with_influence`, the RC route. It needs the unit id and
`estimate_att_gt_dr_panel_with_influence`. It already does its own clustering, so
`att_gt_clustered_standard_errors` can replace that local code or leave it alone.

## Item 3 was already closed elsewhere

The note that the Python surface exposed "only per-period scalars and two
restrictions" was true when written and is no longer. Three commits in
`register-studies` closed it: `c5ff59c` added the direction restrictions,
`59e28b8` added the window functionals, and `2c35e37` added the joint path
region. `honest.py` now carries a `post_functionals` helper that bounds every
horizon and every window using the study's own treated-count weights rather than
equal ones.

## The RC route is a different estimator from did's, which nothing recorded

Found while giving the RC route its first reference of any kind.
`did_attgt_dr_ref.json` had sat in `tests/` since before the panel route existed,
consumed by nothing, because it was generated with `att_gt`'s default
`panel = TRUE` while the only Rust path was RC. It could never have passed.

`did` calls `DRDID::drdid_rc` for `est_method = "dr"`, the LOCALLY EFFICIENT
repeated-cross-section estimator. This crate implements the TRADITIONAL one,
`DRDID::drdid_rc1`. On mpdta:

* the point estimates agree to 1e-9;
* the standard errors differ by about 2.8%, 0.0922804 against 0.0897358 for cell
  (2004, 2004);
* and ours tracks `drdid_rc1` to 8.5e-4 relative, close but not exact, which is
  bounded by a test and printed rather than asserted away.

So there is no locally efficient RC estimator in the crate. Adding one would be
the way to make the RC route agree with `did` outright.

Separately, and worth knowing for Study 1: the RC route's standard errors on
mpdta are roughly **four times** the panel route's for identical point estimates.
That is the cost of treating one unit observed twice as two observations, and it
is the direction `estimator_choice.md` predicted without a number.

## Done in the second pass

`aggregate_att_gt` in `src/methods/att_gt/aggte.rs`, matching `did::aggte` for
all four types plus `balance_e`. Five more parity tests, all to 1e-9 on both the
point estimates and the standard errors.

Three more findings, each recorded where the code is:

4. **The weights are estimated and that changes the standard error.** Three of
   the four summaries weight by cohort share `pg`, computed from the same sample.
   Treating it as fixed gives 0.0114607 at event time 0 where the right answer is
   0.0114942. `weight_influence` is the correction, ported from `did:::wif`.

5. **`aggregate_att_gt_event_time_with_influence` computed
   `sum(w^2 * se^2)`**, the variance of a weighted sum of independent terms, on
   cells that share units. It was carrying the influence vectors alongside and
   not using them. Now computed from the combined influence function. Nothing in
   the suite caught this, so a regression test now pins it.

6. **`did` does not use Mammen weights, whatever the documentation says.**
   `BMisc::multiplier_bootstrap` is the C++ kernel it calls; probed with a 1x1
   influence matrix over 200,000 draws it returns exactly `-1` and `1` at 0.4994
   and 0.5006. Implementing the documented Mammen weights puts the simultaneous
   critical value at 2.72 against `did`'s 2.62, twenty Monte Carlo standard
   deviations out. `att_gt_mboot_bands` uses Rademacher, the IQR scale and
   type-1 quantiles, and lands at 2.62.

Also fixed while there: the RC route's influence vectors were zero-padded without
rescaling, the same defect found in the panel route, which mattered as soon as
anything recomputed a standard error from them.

## Clustered per-cell standard errors

Closed the same day, and the earlier note that "`did` has the same split" was
wrong. `did` reports a clustered per-cell standard error under **both** bootstrap
settings: with `bstrap = TRUE` it comes from `mboot`, and with `bstrap = FALSE`
it is analytic. There was a reference all along.

`att_gt_clustered_standard_errors` is that analytic version,
`sqrt(sum_c s_c^2) / n` with `s_c` a cluster's summed influence and no
`G / (G - 1)` correction. Matches `did` 2.5.1 to 1e-10.

Two things worth carrying forward:

* **It moves standard errors in both directions.** On mpdta clustered into
  states, the first cell falls from 0.0221 to 0.0102 and another rises from
  0.0312 to 0.0489. "Clustering inflates standard errors" is a rule of thumb, not
  a fact, and there is no safe direction to round in.
* **Singleton clusters return the pair estimator's own number** to 1e-12, so the
  function can be applied unconditionally. On a design without clustering it is a
  no-op rather than a small unexplained shift.

## Two design questions, decided 19 August 2026

Both were open; both are now settled and both are implemented and R-verified.

**Matching becomes a weight.** Callaway-Sant'Anna with covariate adjustment on
the risk set is an alternative to matching, not a complement, so keeping both
would adjust twice. The match weight enters as `AttGtDrObservation::weight` and
nothing else. It has to reach three places, and
`weights_flow_through_estimation_and_aggregation` checks all three against R's
`weightsname`: the pair estimator, the cohort shares the aggregation weights by,
and the correction for those shares being estimated. The test also asserts that
weighting moves every cell, so a silently dropped weight cannot pass.

Note that cohort shares are scale-invariant: multiplying every weight by a
constant leaves `pg / sum(pg)` and `weight_influence` unchanged, so it does not
matter whether weights arrive normalised.

**The cluster is the family.** Two parents of one child are two units sharing
every household shock. `UnitPanel::clustered_by` attaches the labels;
`att_gt_mboot_bands` takes them as `Option<&[i64]>` and resamples clusters rather
than units, which is what `did` does with `clustervars`. Verified against `did`
on mpdta with counties clustered into 29 states, where the standard errors move
by more than 10% in most cells and one goes from 0.031 to 0.057.

The identity worth knowing about:
`clustering_by_unit_reproduces_the_unclustered_standard_error` pins that one
cluster per unit returns the unclustered answer to 1e-12 on both paths. That is
why `standard_error` in `aggte.rs` centres units before summing within cluster,
and why it omits the `G / (G - 1)` correction that
`inference::clustered_variance_from_index` applies. The two are not
interchangeable, and the one here is the one that agrees with `did`.

Also worth knowing: the bootstrap denominators differ. Draws are over
`n_clusters`, so they carry `sqrt(n_clusters)`, but the reported standard error
is `bSigma * sqrt(n_clusters) / n` with `n` the UNIT count.
