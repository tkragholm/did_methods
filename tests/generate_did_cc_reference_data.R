library(jsonlite)

base_path <- "crates/methods/did_methods/tests/"

build_rows <- function(n_per_cell, composition_shift = FALSE) {
  rows <- vector("list", n_per_cell * 4)
  row_index <- 1L
  x_levels <- c(0, 1, 2)
  balanced_x <- rep(x_levels, length.out = n_per_cell)
  treated_pre_x <- if (composition_shift) {
    rep(c(0, 0, 0, 1, 1, 2), length.out = n_per_cell)
  } else {
    balanced_x
  }
  treated_post_x <- if (composition_shift) {
    rep(c(0, 1, 2, 2, 2, 1), length.out = n_per_cell)
  } else {
    balanced_x
  }
  control_pre_x <- balanced_x
  control_post_x <- balanced_x

  for (i in seq_len(n_per_cell)) {
    cp_x <- control_pre_x[i]
    cpost_x <- control_post_x[i]
    tp_x <- treated_pre_x[i]
    tpost_x <- treated_post_x[i]

    rows[[row_index]] <- list(treated = FALSE, post_period = FALSE, outcome = cp_x, covariates = list(cp_x)); row_index <- row_index + 1L
    rows[[row_index]] <- list(treated = FALSE, post_period = TRUE, outcome = cpost_x + 1, covariates = list(cpost_x)); row_index <- row_index + 1L
    rows[[row_index]] <- list(treated = TRUE, post_period = FALSE, outcome = tp_x, covariates = list(tp_x)); row_index <- row_index + 1L

    effect <- if (composition_shift) {
      1 + tpost_x
    } else {
      1.5
    }
    rows[[row_index]] <- list(
      treated = TRUE,
      post_period = TRUE,
      outcome = tpost_x + 1 + effect,
      covariates = list(tpost_x)
    )
    row_index <- row_index + 1L
  }

  rows
}

rows_to_df <- function(rows) {
  data.frame(
    treated = vapply(rows, function(row) isTRUE(row$treated), logical(1)),
    post_period = vapply(rows, function(row) isTRUE(row$post_period), logical(1)),
    outcome = vapply(rows, function(row) row$outcome, numeric(1)),
    x = vapply(rows, function(row) row$covariates[[1]], numeric(1)),
    stringsAsFactors = FALSE
  )
}

cell_index <- function(treated, post_period) {
  if (treated && post_period) return(1L)
  if (treated && !post_period) return(2L)
  if (!treated && post_period) return(3L)
  4L
}

normalized_weight <- function(numerator) {
  denominator <- mean(numerator)
  if (denominator <= 0) {
    rep(0, length(numerator))
  } else {
    numerator / denominator
  }
}

standard_error_from_influence <- function(influence) {
  n <- length(influence)
  centered <- influence - mean(influence)
  sqrt(mean(centered ^ 2) / n)
}

compute_reference <- function(rows) {
  df <- rows_to_df(rows)
  x_values <- sort(unique(df$x))

  m11 <- numeric(nrow(df))
  m10 <- numeric(nrow(df))
  m01 <- numeric(nrow(df))
  m00 <- numeric(nrow(df))
  p_generalized <- matrix(0, nrow(df), 4)
  p_treated <- numeric(nrow(df))

  for (x_value in x_values) {
    x_mask <- df$x == x_value
    x_rows <- df[x_mask, , drop = FALSE]

    m11[x_mask] <- mean(x_rows$outcome[x_rows$treated & x_rows$post_period])
    m10[x_mask] <- mean(x_rows$outcome[x_rows$treated & !x_rows$post_period])
    m01[x_mask] <- mean(x_rows$outcome[!x_rows$treated & x_rows$post_period])
    m00[x_mask] <- mean(x_rows$outcome[!x_rows$treated & !x_rows$post_period])

    class_counts <- c(
      sum(x_rows$treated & x_rows$post_period),
      sum(x_rows$treated & !x_rows$post_period),
      sum(!x_rows$treated & x_rows$post_period),
      sum(!x_rows$treated & !x_rows$post_period)
    )
    p_generalized[x_mask, ] <- matrix(class_counts / sum(class_counts), nrow = sum(x_mask), ncol = 4, byrow = TRUE)
    p_treated[x_mask] <- mean(x_rows$treated)
  }

  treated_post_indicator <- as.numeric(df$treated & df$post_period)
  treated_pre_indicator <- as.numeric(df$treated & !df$post_period)
  control_post_indicator <- as.numeric(!df$treated & df$post_period)
  control_pre_indicator <- as.numeric(!df$treated & !df$post_period)
  normalized_weights <- rep(1, nrow(df))

  robust_scores <- numeric(nrow(df))
  stationary_scores <- numeric(nrow(df))
  for (i in seq_len(nrow(df))) {
    tau_dr_signal <- df$outcome[i] - (m10[i] + m01[i] - m00[i])
    w11 <- normalized_weight(normalized_weights * treated_post_indicator)[i]
    w10 <- normalized_weight(normalized_weights * treated_pre_indicator * p_generalized[, 1] / pmax(p_generalized[, 2], 1e-12))[i]
    w01 <- normalized_weight(normalized_weights * control_post_indicator * p_generalized[, 1] / pmax(p_generalized[, 3], 1e-12))[i]
    w00 <- normalized_weight(normalized_weights * control_pre_indicator * p_generalized[, 1] / pmax(p_generalized[, 4], 1e-12))[i]
    robust_scores[i] <- w11 * tau_dr_signal - w10 * (df$outcome[i] - m10[i]) - w01 * (df$outcome[i] - m01[i]) + w00 * (df$outcome[i] - m00[i])

    tau_x <- m11[i] - m10[i] - m01[i] + m00[i]
    treated_weight <- normalized_weight(normalized_weights * as.numeric(df$treated))[i]
    sw11 <- normalized_weight(normalized_weights * treated_post_indicator)[i]
    sw10 <- normalized_weight(normalized_weights * treated_pre_indicator)[i]
    sw01 <- normalized_weight(normalized_weights * control_post_indicator * p_treated / pmax(1 - p_treated, 1e-12))[i]
    sw00 <- normalized_weight(normalized_weights * control_pre_indicator * p_treated / pmax(1 - p_treated, 1e-12))[i]
    stationary_scores[i] <- treated_weight * tau_x +
      sw11 * (df$outcome[i] - m11[i]) -
      sw10 * (df$outcome[i] - m10[i]) -
      sw01 * (df$outcome[i] - m01[i]) +
      sw00 * (df$outcome[i] - m00[i])
  }

  robust_att <- mean(robust_scores)
  stationary_att <- mean(stationary_scores)
  robust_if <- robust_scores - robust_att
  stationary_if <- stationary_scores - stationary_att
  diff_if <- stationary_if - robust_if
  diff_se <- standard_error_from_influence(diff_if)
  hausman_stat <- (stationary_att - robust_att) ^ 2 / (diff_se ^ 2)
  hausman_p <- 1 - pchisq(hausman_stat, df = 1)

  list(
    rows = rows,
    robust = list(
      att = robust_att,
      se = standard_error_from_influence(robust_if)
    ),
    stationary = list(
      att = stationary_att,
      se = standard_error_from_influence(stationary_if)
    ),
    hausman = list(
      difference = stationary_att - robust_att,
      difference_se = diff_se,
      statistic = hausman_stat,
      p_value = hausman_p
    )
  )
}

did_cc_ref <- list(
  meta = list(
    source = "Authored R reference formulas on deterministic discrete-covariate fixture",
    note = paste(
      "There is currently no official external DiD_CC R package wired into this parity harness.",
      "These reference values are computed from explicit score formulas on a deterministic fixture."
    )
  ),
  aligned_constant_effect = compute_reference(build_rows(120, FALSE)),
  composition_shift = compute_reference(build_rows(140, TRUE))
)

write_json(
  did_cc_ref,
  paste0(base_path, "did_cc_ref_scaffold.json"),
  digits = 15,
  auto_unbox = TRUE,
  pretty = TRUE
)

cat("DiD_CC reference fixture generated.\n")
