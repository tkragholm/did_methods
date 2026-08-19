# Callaway-Sant'Anna panel reference fixtures.
#
# Run from the crate root:  Rscript tests/generate_cs_panel_reference.R
#
# WHY THIS FILE EXISTS SEPARATELY from generate_reference_data.R: that script
# writes to a "crates/methods/did_methods/tests/" path from a workspace layout
# this crate no longer lives in, and it pins att_gt only for the RC-shaped
# comparison. This one targets the PANEL DR route and records, in the fixture
# itself, every default that differs between R and what we ask for.
#
# The five did::att_gt defaults that will otherwise produce spurious parity
# failures, all set explicitly below rather than relied upon:
#   base_period   = "varying"      -> we want "universal" to match a fixed t-1 anchor
#   control_group = "nevertreated" -> both are generated here
#   bstrap        = TRUE           -> bootstrap SEs; we compare ANALYTIC, so FALSE
#   cband         = TRUE           -> simultaneous bands; we want pointwise, so FALSE
#   panel         = TRUE           -> the route we are building; stated anyway
#
# mpdta is the did package's county teen-employment panel: 500 counties, 2003-2007,
# first.treat in {2004, 2006, 2007, 0}. lpop is the single covariate.

suppressMessages({
  library(did)
  library(DRDID)
  library(jsonlite)
})

base_path <- "tests/"
data(mpdta)

# att_gt returns inffunc as an n_units x n_cells matrix. Exporting it is the
# point of this fixture: every aggregation, every simultaneous band and the whole
# HonestDiD surface is a function of that matrix, so pinning it pins everything
# downstream.
#
# THE ROW ORDER IS NOT THE SORTED ID. did reorders units by cohort during
# pre-processing, so row 1 of inffunc is the first unit of the earliest treated
# group (17005, first.treat 2004), not the smallest countyreal (8001, 2007).
# Reconstructing that sort from the outside is guesswork, so each fixture carries
# the id vector that its own inffunc rows correspond to, read back out of
# DIDparams. A consumer joins on the id and never has to know how did sorts.
inffunc_unit_order <- function(out) {
  dp <- out$DIDparams
  as.numeric(unique(dp$data[[dp$idname]]))
}

run_cell <- function(control_group, base_period) {
  out <- att_gt(
    yname         = "lemp",
    tname         = "year",
    idname        = "countyreal",
    gname         = "first.treat",
    xformla       = ~ lpop,
    data          = mpdta,
    panel         = TRUE,
    control_group = control_group,
    base_period   = base_period,
    est_method    = "dr",
    bstrap        = FALSE,
    cband         = FALSE,
    alp           = 0.05
  )
  list(
    control_group = control_group,
    base_period   = base_period,
    group         = out$group,
    t             = out$t,
    att           = out$att,
    se            = out$se,
    n_units       = out$n,
    inffunc       = as.matrix(out$inffunc),
    inffunc_unit_order = inffunc_unit_order(out)
  )
}

aggregations <- function(out) {
  grab <- function(a) list(
    overall_att = a$overall.att, overall_se = a$overall.se,
    egt = a$egt, att_egt = a$att.egt, se_egt = a$se.egt,
    crit_val = a$crit.val.egt
  )
  list(
    simple   = grab(aggte(out, type = "simple",   bstrap = FALSE, cband = FALSE)),
    dynamic  = grab(aggte(out, type = "dynamic",  bstrap = FALSE, cband = FALSE)),
    group    = grab(aggte(out, type = "group",    bstrap = FALSE, cband = FALSE)),
    calendar = grab(aggte(out, type = "calendar", bstrap = FALSE, cband = FALSE)),
    # balance_e is the reason we are doing this at all: it fixes the cohort
    # composition across event times, which the fixed-baseline pairwise design
    # cannot do. balance_e = 1 keeps cohorts observed at least 1 period out.
    dynamic_balanced = grab(
      aggte(out, type = "dynamic", balance_e = 1, bstrap = FALSE, cband = FALSE)
    )
  )
}

# --- 1. The core parity target: panel DR, universal base, never-treated ---
tryCatch({
  ref <- run_cell("nevertreated", "universal")
  ref$data_subset <- mpdta[, c("year", "countyreal", "lemp", "first.treat", "lpop")]

  out <- att_gt(yname = "lemp", tname = "year", idname = "countyreal",
                gname = "first.treat", xformla = ~ lpop, data = mpdta, panel = TRUE,
                control_group = "nevertreated", base_period = "universal",
                est_method = "dr", bstrap = FALSE, cband = FALSE)
  ref$aggregations <- aggregations(out)

  write_json(ref, paste0(base_path, "cs_panel_dr_universal_ref.json"),
             digits = 17, auto_unbox = TRUE, matrix = "rowmajor", na = "null", null = "null")
  cat("1. cs_panel_dr_universal_ref.json written\n")
}, error = function(e) cat("1. FAILED:", conditionMessage(e), "\n"))

# --- 2. Not-yet-treated comparison group, same everything else ---
# This is the switch that measures crossover contamination in Study 1.
tryCatch({
  ref <- run_cell("notyettreated", "universal")
  ref$data_subset <- mpdta[, c("year", "countyreal", "lemp", "first.treat", "lpop")]
  write_json(ref, paste0(base_path, "cs_panel_dr_notyet_ref.json"),
             digits = 17, auto_unbox = TRUE, matrix = "rowmajor", na = "null", null = "null")
  cat("2. cs_panel_dr_notyet_ref.json written\n")
}, error = function(e) cat("2. FAILED:", conditionMessage(e), "\n"))

# --- 3. Varying base period, to pin the default we are NOT using ---
# Kept so that a future reader can see, in numbers, that base_period changes
# every coefficient, rather than having to take the comment above on trust.
tryCatch({
  ref <- run_cell("nevertreated", "varying")
  write_json(ref, paste0(base_path, "cs_panel_dr_varying_ref.json"),
             digits = 17, auto_unbox = TRUE, matrix = "rowmajor", na = "null", null = "null")
  cat("3. cs_panel_dr_varying_ref.json written\n")
}, error = function(e) cat("3. FAILED:", conditionMessage(e), "\n"))

# --- 4. One (g,t) cell in isolation, straight through DRDID::drdid_panel ---
# The pair-level target. If this fails, nothing above can pass, and the failure
# is in the pair estimator rather than in the ATT(g,t) loop around it.
tryCatch({
  g <- 2006; t <- 2005; base <- 2005  # universal base for g=2006 is 2005
  tt <- 2006
  wide <- merge(
    subset(mpdta, year == base, select = c("countyreal", "lemp", "first.treat", "lpop")),
    subset(mpdta, year == tt,   select = c("countyreal", "lemp")),
    by = "countyreal", suffixes = c("_base", "_post")
  )
  wide <- subset(wide, first.treat == g | first.treat == 0)
  d <- as.numeric(wide$first.treat == g)
  fit <- drdid_panel(y1 = wide$lemp_post, y0 = wide$lemp_base, D = d,
                     covariates = cbind(1, wide$lpop), inffunc = TRUE)
  write_json(list(
    group = g, time = tt, baseline_time = base,
    att = fit$ATT, se = fit$se, inffunc = as.numeric(fit$att.inf.func),
    countyreal = wide$countyreal, d = d,
    lemp_base = wide$lemp_base, lemp_post = wide$lemp_post, lpop = wide$lpop
  ), paste0(base_path, "drdid_panel_cell_ref.json"),
     digits = 17, auto_unbox = TRUE, na = "null", null = "null")
  cat("4. drdid_panel_cell_ref.json written\n")
}, error = function(e) cat("4. FAILED:", conditionMessage(e), "\n"))


# --- 5. Multiplier bootstrap, for the simultaneous bands ---
# A bootstrap cannot be reproduced across languages: the RNG streams differ. What
# CAN be checked is that the critical value and the bootstrap standard errors
# agree to within Monte Carlo error, so this is generated with a large number of
# iterations to shrink that error and the test is given a tolerance to match.
tryCatch({
  set.seed(20260819)
  out <- att_gt(yname = "lemp", tname = "year", idname = "countyreal",
                gname = "first.treat", xformla = ~ lpop, data = mpdta, panel = TRUE,
                control_group = "nevertreated", base_period = "universal",
                est_method = "dr", bstrap = TRUE, cband = TRUE, biters = 20000)
  write_json(list(
    biters = 20000,
    group = out$group, t = out$t, att = out$att,
    se = out$se, crit_val = out$c
  ), paste0(base_path, "cs_panel_mboot_ref.json"),
     digits = 17, auto_unbox = TRUE, na = "null", null = "null")
  cat("5. cs_panel_mboot_ref.json written\n")
}, error = function(e) cat("5. FAILED:", conditionMessage(e), "\n"))

# --- 6. Clustered multiplier bootstrap ---
# mpdta has no cluster variable, so one is derived: the leading digits of the
# county FIPS are the state, which groups the 500 counties into a handful of
# clusters. That is exactly the shape Study 1 has, where two parents of one child
# are two units in one family, and it is the shape a unit-level bootstrap gets
# wrong.
tryCatch({
  set.seed(20260819)
  d <- mpdta
  d$state <- floor(d$countyreal / 1000)
  out <- att_gt(yname = "lemp", tname = "year", idname = "countyreal",
                gname = "first.treat", xformla = ~ lpop, data = d, panel = TRUE,
                control_group = "nevertreated", base_period = "universal",
                est_method = "dr", bstrap = TRUE, cband = TRUE, biters = 20000,
                clustervars = "state")
  units <- sort(unique(d$countyreal))
  write_json(list(
    biters = 20000,
    n_states = length(unique(d$state)),
    group = out$group, t = out$t, att = out$att, se = out$se, crit_val = out$c,
    unit_order = units,
    cluster_of_unit = floor(units / 1000)
  ), paste0(base_path, "cs_panel_mboot_clustered_ref.json"),
     digits = 17, auto_unbox = TRUE, na = "null", null = "null")
  cat("6. cs_panel_mboot_clustered_ref.json written\n")
}, error = function(e) cat("6. FAILED:", conditionMessage(e), "\n"))

# --- 7. Sampling weights ---
# How the matching enters once it stops being the design: the match weight
# becomes a weight and nothing else. Checks that it reaches the pair estimator,
# the cohort shares and the correction for those shares being estimated.
tryCatch({
  d <- mpdta
  set.seed(11)
  wu <- setNames(
    ifelse(tapply(d$first.treat, d$countyreal, function(z) z[1]) > 0, 1.0,
           sample(c(0.25, 0.5, 1.0), length(unique(d$countyreal)), replace = TRUE)),
    sort(unique(d$countyreal)))
  d$w <- as.numeric(wu[as.character(d$countyreal)])
  out <- att_gt(yname = "lemp", tname = "year", idname = "countyreal",
                gname = "first.treat", xformla = ~ lpop, data = d, panel = TRUE,
                control_group = "nevertreated", base_period = "universal",
                est_method = "dr", bstrap = FALSE, cband = FALSE, weightsname = "w")
  ag <- aggte(out, type = "dynamic", bstrap = FALSE, cband = FALSE)
  units <- sort(unique(d$countyreal))
  write_json(list(
    group = out$group, t = out$t, att = out$att, se = out$se,
    unit_order = units, weight_of_unit = as.numeric(wu[as.character(units)]),
    dynamic = list(overall_att = ag$overall.att, overall_se = ag$overall.se,
                   egt = ag$egt, att_egt = ag$att.egt, se_egt = ag$se.egt)
  ), paste0(base_path, "cs_panel_weighted_ref.json"),
     digits = 17, auto_unbox = TRUE, na = "null", null = "null")
  cat("7. cs_panel_weighted_ref.json written\n")
}, error = function(e) cat("7. FAILED:", conditionMessage(e), "\n"))


# --- 8. Analytic clustered per-cell standard errors ---
# did DOES report these: bstrap = FALSE together with clustervars gives an
# analytic clustered variance per (g,t) cell. So the per-cell standard error has
# a reference and does not have to be left to the bootstrap.
tryCatch({
  d <- mpdta
  d$state <- floor(d$countyreal / 1000)
  out <- att_gt(yname = "lemp", tname = "year", idname = "countyreal",
                gname = "first.treat", xformla = ~ lpop, data = d, panel = TRUE,
                control_group = "nevertreated", base_period = "universal",
                est_method = "dr", bstrap = FALSE, cband = FALSE,
                clustervars = "state")
  units <- sort(unique(d$countyreal))
  write_json(list(
    group = out$group, t = out$t, att = out$att, se = out$se,
    unit_order = units, cluster_of_unit = floor(units / 1000)
  ), paste0(base_path, "cs_panel_clustered_analytic_ref.json"),
     digits = 17, auto_unbox = TRUE, na = "null", null = "null")
  cat("8. cs_panel_clustered_analytic_ref.json written\n")
}, error = function(e) cat("8. FAILED:", conditionMessage(e), "\n"))


# --- 9. min_e / max_e on the dynamic aggregation ---
# Inclusive bounds on event time, applied after balance_e. They narrow the
# OVERALL number too, since the dynamic overall is the mean over the retained
# non-negative event times.
tryCatch({
  out <- att_gt(yname = "lemp", tname = "year", idname = "countyreal",
                gname = "first.treat", xformla = ~ lpop, data = mpdta, panel = TRUE,
                control_group = "nevertreated", base_period = "universal",
                est_method = "dr", bstrap = FALSE, cband = FALSE)
  spec <- list(c(-2, 2), c(0, 3), c(-1, 1))
  trimmed <- lapply(spec, function(s) {
    a <- aggte(out, type = "dynamic", min_e = s[1], max_e = s[2],
               bstrap = FALSE, cband = FALSE)
    list(min_e = s[1], max_e = s[2], overall_att = a$overall.att,
         overall_se = a$overall.se, egt = a$egt, att_egt = a$att.egt,
         se_egt = a$se.egt)
  })
  write_json(trimmed, paste0(base_path, "cs_panel_trimmed_ref.json"),
             digits = 17, auto_unbox = TRUE, na = "null", null = "null")
  cat("9. cs_panel_trimmed_ref.json written\n")
}, error = function(e) cat("9. FAILED:", conditionMessage(e), "\n"))


# --- 10. Repeated cross-section route ---
# panel = FALSE treats each row as its own observation. On mpdta the point
# estimates come out identical to the panel route, but the standard errors are
# about four times larger, which is the cost of discarding the pairing. Recorded
# so the RC route has a reference at all; it previously had none.
tryCatch({
  out <- att_gt(yname = "lemp", tname = "year", idname = "countyreal",
                gname = "first.treat", xformla = ~ lpop, data = mpdta, panel = FALSE,
                control_group = "nevertreated", base_period = "universal",
                est_method = "dr", bstrap = FALSE, cband = FALSE)
  ag <- aggte(out, type = "dynamic", bstrap = FALSE, cband = FALSE)
  write_json(list(
    group = out$group, t = out$t, att = out$att, se = out$se, n_rows = out$n,
    dynamic = list(overall_att = ag$overall.att, overall_se = ag$overall.se,
                   egt = ag$egt, att_egt = ag$att.egt, se_egt = ag$se.egt)
  ), paste0(base_path, "cs_rc_ref.json"),
     digits = 17, auto_unbox = TRUE, na = "null", null = "null")
  cat("10. cs_rc_ref.json written\n")
}, error = function(e) cat("10. FAILED:", conditionMessage(e), "\n"))


# --- 11. The TRADITIONAL repeated-cross-section estimator ---
# did calls DRDID::drdid_rc for est_method = "dr", which is the LOCALLY EFFICIENT
# RC estimator. This crate's RC route implements the traditional one,
# DRDID::drdid_rc1. The point estimates agree; the standard errors do not, by
# about 2.8% on this data. So the RC route's reference is drdid_rc1, cell by
# cell, not did's panel = FALSE output.
tryCatch({
  cells <- list(c(2004,2003,2004), c(2004,2003,2005), c(2004,2003,2006), c(2004,2003,2007),
                c(2006,2005,2003), c(2006,2005,2004), c(2006,2005,2006), c(2006,2005,2007),
                c(2007,2006,2003), c(2007,2006,2004), c(2007,2006,2005), c(2007,2006,2007))
  rows <- lapply(cells, function(c3) {
    g <- c3[1]; base <- c3[2]; tt <- c3[3]
    d <- subset(mpdta, year %in% c(base, tt) & (first.treat == g | first.treat == 0))
    fit1 <- drdid_rc1(y = d$lemp, post = as.numeric(d$year == tt),
                      D = as.numeric(d$first.treat == g), covariates = cbind(1, d$lpop))
    fitE <- drdid_rc(y = d$lemp, post = as.numeric(d$year == tt),
                     D = as.numeric(d$first.treat == g), covariates = cbind(1, d$lpop))
    list(group = g, baseline_time = base, time = tt,
         att = fit1$ATT, se_traditional = fit1$se, se_locally_efficient = fitE$se)
  })
  write_json(rows, paste0(base_path, "drdid_rc1_cells_ref.json"),
             digits = 17, auto_unbox = TRUE, na = "null", null = "null")
  cat("11. drdid_rc1_cells_ref.json written\n")
}, error = function(e) cat("11. FAILED:", conditionMessage(e), "\n"))

cat("done\n")
