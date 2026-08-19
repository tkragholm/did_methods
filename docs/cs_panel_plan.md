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

## The RC route: two estimators, and now both are here

Found while giving the RC route its first reference of any kind.
`did_attgt_dr_ref.json` had sat in `tests/` since before the panel route existed,
consumed by nothing, because it was generated with `att_gt`'s default
`panel = TRUE` while the only Rust path was RC. It could never have passed.

`did` calls `DRDID::drdid_rc` for `est_method = "dr"`, the LOCALLY EFFICIENT
repeated-cross-section estimator, which fits four outcome regressions (treated
and control, in each period) rather than the control group's two. The crate had
only the TRADITIONAL `DRDID::drdid_rc1`. Point estimates agree; standard errors
differ by about 2.8%, 0.0922804 against 0.0897358 for cell (2004, 2004).

`estimate_drdid_repeated_efficient` is the missing one, and
`estimate_att_gt_dr_efficient_with_influence` routes `ATT(g,t)` through it. The
RC route now reproduces `did`'s `panel = FALSE` output outright: ATT and standard
errors to 1e-9, and the dynamic aggregation with it.

Three details in the reference are easy to miss and each moves the answer:
sampling weights are normalised to mean one before anything else, so every mean
is over the full sample; trimming is asymmetric, keeping treated rows whatever
their propensity score; and the residual inside each regression's asymptotic
linear representation uses that regression's own fitted values rather than the
period-combined control prediction used in the `eta` terms.

**The crate's 1e-8 ridge has to be off for exact parity**, because R fits these
regressions unregularised. It cancels in the point estimate and does not cancel
in the influence function: with the default the worst influence deviation is
2.0e-7, without it 1.5e-8, and the ATT and standard error come to 1.5e-11 and
3.4e-12. The ridge is worth keeping as a default for collinear covariates; it is
just not what R does.

**One gap stays open, and it is in the traditional estimator, not this one.**
`estimate_drdid_repeated_cross_section` tracks `drdid_rc1` to 8.5e-4 relative on
the standard error rather than exactly. The ridge is NOT the cause: disabling it
leaves the worst deviation unchanged. Since the locally efficient implementation
matches its own reference to 1e-11, this is something specific to the
traditional variant. It is bounded by a test rather than chased, because `did`
calls the other one.

Separately, and worth knowing for Study 1: the RC route's standard errors on
mpdta are roughly **four times** the panel route's for identical point estimates.
That is the cost of treating one unit observed twice as two observations, and it
is the direction `estimator_choice.md` predicted without a number.

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
