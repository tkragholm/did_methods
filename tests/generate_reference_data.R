library(did)
library(contdid)
library(triplediff)
library(HonestDiD)
library(jsonlite)

base_path <- "crates/methods/did_methods/tests/"

# --- 1. did Reference (Staggered ATT(g,t)) ---
tryCatch({
  data(mpdta)
  out_did <- att_gt(yname = "lemp", tname = "year", idname = "countyreal", gname = "first.treat",
                    data = mpdta, control_group = "nevertreated", bstrap = FALSE, cband = FALSE)
  did_ref <- list(
    group = out_did$group, t = out_did$t, att = out_did$att, se = out_did$se,
    data_subset = mpdta[, c("year", "countyreal", "lemp", "first.treat")]
  )
  write_json(did_ref, paste0(base_path, "did_attgt_ref.json"), digits = 15, auto_unbox = TRUE)
  cat("1. did Reference generated.\n")
}, error = function(e) cat("1. did Reference FAILED:", e$message, "\n"))

# --- 2. did DR Reference ---
tryCatch({
  data(mpdta)
  out_did_dr <- att_gt(yname = "lemp", tname = "year", idname = "countyreal", gname = "first.treat",
                       xformla = ~ lpop, data = mpdta, control_group = "nevertreated",
                       est_method = "dr", bstrap = FALSE, cband = FALSE)
  did_dr_ref <- list(
    group = out_did_dr$group, t = out_did_dr$t, att = out_did_dr$att, se = out_did_dr$se,
    data_subset = mpdta[, c("year", "countyreal", "lemp", "first.treat", "lpop")]
  )
  write_json(did_dr_ref, paste0(base_path, "did_attgt_dr_ref.json"), digits = 15, auto_unbox = TRUE)
  cat("2. did DR Reference generated.\n")
}, error = function(e) cat("2. did DR Reference FAILED:", e$message, "\n"))

# --- 3. Continuous DiD Reference (Single 2-period cell) ---
tryCatch({
  set.seed(1234)
  n <- 2000
  df_2p <- data.frame(
    id = rep(1:n, each = 2),
    G = rep(sample(c(2, 0), n, replace = TRUE), each = 2),
    period = rep(1:2, n),
    D = rep(runif(n, 1, 10), each = 2),
    Y = rnorm(2*n)
  )
  df_2p$D[df_2p$G == 0] <- 0
  
  df_pre <- subset(df_2p, period == 1)
  df_post <- subset(df_2p, period == 2)
  
  cont_ref <- list(
    delta_y = df_post$Y - df_pre$Y,
    dose = df_post$D,
    treated = df_post$G == 2
  )
  write_json(cont_ref, paste0(base_path, "contdid_ref.json"), digits = 15, auto_unbox = TRUE)
  cat("3. Continuous DiD data generated.\n")
}, error = function(e) cat("3. Continuous DiD FAILED:", e$message, "\n"))

# --- 4. Triple-Difference ---
tryCatch({
  df_ddd <- triplediff:::generate_test_panel(seed = 123, num_ids = 100)
  df_ddd$G <- ifelse(df_ddd$treat == 1, 2020, 0)
  out_ddd <- ddd(yname = "outcome", tname = "year", idname = "id",
                 gname = "G", pname = "partition",
                 xformla = ~ x1 + x2, data = df_ddd, panel = TRUE)
  ddd_ref <- list(
    att = out_ddd$ATT,
    se = out_ddd$se,
    data = df_ddd
  )
  write_json(ddd_ref, paste0(base_path, "triple_did_ref.json"), digits = 15, auto_unbox = TRUE)
  cat("4. Triple-Difference generated.\n")
}, error = function(e) cat("4. Triple-Difference FAILED:", e$message, "\n"))

# --- 5. Honest DiD Data (Components only) ---
tryCatch({
  library(HonestDiD)
  data(BCdata_EventStudy)
  honest_ref <- list(
    betahat = BCdata_EventStudy$betahat,
    sigma = BCdata_EventStudy$sigma,
    pre_indices = BCdata_EventStudy$prePeriodIndices,
    post_indices = BCdata_EventStudy$postPeriodIndices
  )
  write_json(honest_ref, paste0(base_path, "honest_did_ref.json"), digits = 15, auto_unbox = TRUE)
  cat("5. Honest DiD components generated.\n")
}, error = function(e) cat("5. Honest DiD FAILED:", e$message, "\n"))

# --- 6. Wald Test ---
tryCatch({
  est_pre <- c(-0.01, 0.02, 0.05)
  vcov_pre <- matrix(c(0.01, 0.002, 0.001,
                       0.002, 0.015, 0.003,
                       0.001, 0.003, 0.02), nrow=3)
  stat_wald <- as.numeric(t(est_pre) %*% solve(vcov_pre) %*% est_pre)
  p_wald <- 1 - pchisq(stat_wald, df=3)
  wald_ref <- list(
    estimates = est_pre,
    vcov = vcov_pre,
    statistic = stat_wald,
    p_value = p_wald,
    df = 3
  )
  write_json(wald_ref, paste0(base_path, "wald_test_ref.json"), digits = 15, auto_unbox = TRUE)
  cat("6. Wald Test generated.\n")
}, error = function(e) cat("6. Wald Test FAILED:", e$message, "\n"))


# --- 7. Honest DiD Joint Path (Bonferroni-adjusted C-LF) ---
tryCatch({
  library(HonestDiD)
  data(BCdata_EventStudy)

  num_pre <- length(BCdata_EventStudy$prePeriodIndices)
  num_post <- length(BCdata_EventStudy$postPeriodIndices)
  alpha_joint <- 0.05
  alpha_pointwise <- alpha_joint / num_post
  mbar <- 1.0

  l_vecs <- lapply(seq_len(num_post), function(i) {
    vec <- rep(0, num_post)
    vec[i] <- 1
    vec
  })

  points <- lapply(seq_len(num_post), function(i) {
    result <- createSensitivityResults_relativeMagnitudes(
      betahat = BCdata_EventStudy$betahat,
      sigma = BCdata_EventStudy$sigma,
      numPrePeriods = num_pre,
      numPostPeriods = num_post,
      method = "C-LF",
      Mbarvec = c(mbar),
      l_vec = l_vecs[[i]],
      alpha = alpha_pointwise
    )
    list(
      post_period = i - 1,
      lb = result$lb[[1]],
      ub = result$ub[[1]]
    )
  })

  joint_ref <- list(
    meta = list(
      source = "HonestDiD::createSensitivityResults_relativeMagnitudes",
      method = "C-LF",
      delta = "DeltaRM",
      mbar = mbar,
      alpha_joint = alpha_joint,
      alpha_pointwise = alpha_pointwise,
      pointwise_confidence_level = 1 - alpha_pointwise
    ),
    l_vecs = l_vecs,
    points = points
  )

  write_json(joint_ref, paste0(base_path, "honest_did_joint_path_ref.json"), digits = 15, auto_unbox = TRUE)
  cat("7. Honest DiD joint-path reference generated.\n")
}, error = function(e) cat("7. Honest DiD joint-path reference FAILED:", e$message, "\n"))

# --- 8. Honest DiD Directional (Bonferroni-adjusted C-LF) ---
tryCatch({
  library(HonestDiD)
  data(BCdata_EventStudy)

  num_pre <- length(BCdata_EventStudy$prePeriodIndices)
  num_post <- length(BCdata_EventStudy$postPeriodIndices)
  alpha_joint <- 0.05
  mbar <- 1.0

  basis <- lapply(seq_len(num_post), function(i) {
    vec <- rep(0, num_post)
    vec[i] <- 1
    vec
  })
  diff_01 <- rep(0, num_post)
  diff_01[1] <- 1
  diff_01[2] <- -1

  directions <- c(
    list(list(name = "period_0", l_vec = basis[[1]])),
    list(list(name = "period_1", l_vec = basis[[2]])),
    list(list(name = "period_2", l_vec = basis[[3]])),
    list(list(name = "period_3", l_vec = basis[[4]])),
    list(list(name = "diff_0_1", l_vec = diff_01))
  )

  alpha_pointwise <- alpha_joint / length(directions)
  points <- lapply(directions, function(dir) {
    result <- createSensitivityResults_relativeMagnitudes(
      betahat = BCdata_EventStudy$betahat,
      sigma = BCdata_EventStudy$sigma,
      numPrePeriods = num_pre,
      numPostPeriods = num_post,
      method = "C-LF",
      Mbarvec = c(mbar),
      l_vec = dir$l_vec,
      alpha = alpha_pointwise
    )
    list(
      name = dir$name,
      lb = result$lb[[1]],
      ub = result$ub[[1]]
    )
  })

  directional_ref <- list(
    meta = list(
      source = "HonestDiD::createSensitivityResults_relativeMagnitudes",
      method = "C-LF",
      delta = "DeltaRM",
      mbar = mbar,
      alpha_joint = alpha_joint,
      alpha_pointwise = alpha_pointwise,
      pointwise_confidence_level = 1 - alpha_pointwise
    ),
    directions = directions,
    points = points
  )

  write_json(directional_ref, paste0(base_path, "honest_did_directional_ref.json"), digits = 15, auto_unbox = TRUE)
  cat("8. Honest DiD directional reference generated.\n")
}, error = function(e) cat("8. Honest DiD directional reference FAILED:", e$message, "\n"))

# --- 9. Honest DiD Multi-FLCI Scaffold (input vectors for Rust parity harness) ---
tryCatch({
  library(HonestDiD)
  data(BCdata_EventStudy)

  num_post <- length(BCdata_EventStudy$postPeriodIndices)
  period_0 <- rep(0, num_post)
  period_0[1] <- 1
  avg_0_1 <- rep(0, num_post)
  avg_0_1[1] <- 0.5
  avg_0_1[2] <- 0.5
  diff_0_1 <- rep(0, num_post)
  diff_0_1[1] <- 1
  diff_0_1[2] <- -1

  multi_scaffold <- list(
    meta = list(
      source = "Scaffold for Rust multi-FLCI parity",
      note = "HonestDiD does not expose a direct joint multi-functional DeltaRM FLCI API.",
      confidence_level = 0.95,
      mbar = 1.0
    ),
    l_vecs = list(period_0, avg_0_1, diff_0_1)
  )

  write_json(multi_scaffold, paste0(base_path, "honest_did_multi_flci_scaffold.json"), digits = 15, auto_unbox = TRUE)
  cat("9. Honest DiD multi-FLCI scaffold generated.\n")
}, error = function(e) cat("9. Honest DiD multi-FLCI scaffold FAILED:", e$message, "\n"))

# --- 10. Honest DiD Gaussian Calibration Scaffold (non-Bonferroni) ---
tryCatch({
  library(HonestDiD)
  data(BCdata_EventStudy)
  if (!requireNamespace("MASS", quietly = TRUE)) {
    stop("MASS package is required for Gaussian scaffold generation")
  }

  num_pre <- length(BCdata_EventStudy$prePeriodIndices)
  num_post <- length(BCdata_EventStudy$postPeriodIndices)
  sigma <- BCdata_EventStudy$sigma
  sigma_post <- sigma[(num_pre + 1):(num_pre + num_post), (num_pre + 1):(num_pre + num_post)]
  std_post <- sqrt(diag(sigma_post))
  corr_post <- sigma_post
  for (i in seq_len(num_post)) {
    for (j in seq_len(num_post)) {
      denom <- std_post[i] * std_post[j]
      corr_post[i, j] <- if (i == j) {
        1
      } else if (denom <= 1e-12) {
        0
      } else {
        max(-1, min(1, sigma_post[i, j] / denom))
      }
    }
  }

  directions <- list(
    c(1, 0, 0, 0),
    c(0, 1, 0, 0),
    c(0, 0, 1, 0),
    c(0, 0, 0, 1),
    c(1, -1, 0, 0)
  )
  l_mat <- do.call(rbind, directions)
  cov_dir <- l_mat %*% sigma_post %*% t(l_mat)
  std_dir <- sqrt(diag(cov_dir))
  corr_dir <- cov_dir
  for (i in seq_len(nrow(corr_dir))) {
    for (j in seq_len(ncol(corr_dir))) {
      denom <- std_dir[i] * std_dir[j]
      corr_dir[i, j] <- if (i == j) {
        1
      } else if (denom <= 1e-12) {
        0
      } else {
        max(-1, min(1, cov_dir[i, j] / denom))
      }
    }
  }

  confidence_level <- 0.95
  draws <- 200000
  set.seed(20260310)

  joint_draws <- MASS::mvrnorm(n = draws, mu = rep(0, ncol(corr_post)), Sigma = corr_post)
  joint_max_abs <- apply(abs(joint_draws), 1, max)
  joint_critical <- as.numeric(quantile(joint_max_abs, probs = confidence_level, type = 7))
  joint_pointwise <- 2 * pnorm(joint_critical) - 1

  directional_draws <- MASS::mvrnorm(n = draws, mu = rep(0, ncol(corr_dir)), Sigma = corr_dir)
  directional_max_abs <- apply(abs(directional_draws), 1, max)
  directional_critical <- as.numeric(quantile(directional_max_abs, probs = confidence_level, type = 7))
  directional_pointwise <- 2 * pnorm(directional_critical) - 1

  mbar <- 1.0
  joint_points <- lapply(seq_len(num_post), function(i) {
    l_vec <- rep(0, num_post)
    l_vec[i] <- 1
    result <- createSensitivityResults_relativeMagnitudes(
      betahat = BCdata_EventStudy$betahat,
      sigma = BCdata_EventStudy$sigma,
      numPrePeriods = num_pre,
      numPostPeriods = num_post,
      method = "C-LF",
      Mbarvec = c(mbar),
      l_vec = l_vec,
      alpha = 1.0 - joint_pointwise
    )
    list(
      post_period = i - 1,
      lb = result$lb[[1]],
      ub = result$ub[[1]]
    )
  })

  directional_points <- lapply(seq_along(directions), function(i) {
    result <- createSensitivityResults_relativeMagnitudes(
      betahat = BCdata_EventStudy$betahat,
      sigma = BCdata_EventStudy$sigma,
      numPrePeriods = num_pre,
      numPostPeriods = num_post,
      method = "C-LF",
      Mbarvec = c(mbar),
      l_vec = directions[[i]],
      alpha = 1.0 - directional_pointwise
    )
    list(
      name = if (i <= num_post) paste0("period_", i - 1) else "diff_0_1",
      lb = result$lb[[1]],
      ub = result$ub[[1]]
    )
  })

  gaussian_scaffold <- list(
    meta = list(
      source = "R MASS::mvrnorm Gaussian max-|Z| scaffold",
      confidence_level = confidence_level,
      simulation_draws = draws,
      seed = 20260310
    ),
    joint = list(
      calibrated_max_t_critical_value = joint_critical,
      pointwise_confidence_level = joint_pointwise
    ),
    directional = list(
      calibrated_max_t_critical_value = directional_critical,
      pointwise_confidence_level = directional_pointwise
    ),
    directions = directions,
    mbar = mbar,
    joint_points = joint_points,
    directional_points = directional_points
  )

  write_json(gaussian_scaffold, paste0(base_path, "honest_did_gaussian_scaffold.json"), digits = 15, auto_unbox = TRUE)
  cat("10. Honest DiD Gaussian scaffold generated.\n")
}, error = function(e) cat("10. Honest DiD Gaussian scaffold FAILED:", e$message, "\n"))

# --- 11. DiD_CC external-reference scaffold ---
tryCatch({
  source("crates/methods/did_methods/tests/generate_did_cc_reference_data.R")
  cat("11. DiD_CC scaffold generated.\n")
}, error = function(e) cat("11. DiD_CC scaffold FAILED:", e$message, "\n"))

cat("All reference data generation attempts complete.\n")
