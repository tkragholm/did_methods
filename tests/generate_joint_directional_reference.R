# Regenerate the joint-path and directional C-LF references with explicit grids.
#
# Sections 7 and 8 of generate_reference_data.R call
# createSensitivityResults_relativeMagnitudes without grid.lb / grid.ub, so
# HonestDiD picks a default grid from the data and the returned bounds are
# whatever grid points survive. For the tighter per-period functionals that grid
# is too narrow and the upper bound comes back truncated -- the same defect that
# was proven in honest_did_rm_avg_ref.json, where the C-LF upper bound was
# bit-identical across Mbar and sat below its own identified set.
#
# Each direction gets its own grid here, centred on l'betahat and spanning
# +/- 40 standard errors of that functional, because the scale differs by
# direction: a single-period basis vector and a difference contrast do not share
# a sensible fixed window. Every returned bound is asserted to be interior, so a
# grid that is still too narrow fails loudly instead of quietly truncating.
#
# This script only rewrites the two files it names. Run it from the crate root.

library(HonestDiD)
library(jsonlite)

base_path <- "tests/"
data(BCdata_EventStudy)

num_pre <- length(BCdata_EventStudy$prePeriodIndices)
num_post <- length(BCdata_EventStudy$postPeriodIndices)
alpha_joint <- 0.05
mbar <- 1.0
grid_points <- 10000

post_betahat <- BCdata_EventStudy$betahat[BCdata_EventStudy$postPeriodIndices]
post_sigma <- BCdata_EventStudy$sigma[
  BCdata_EventStudy$postPeriodIndices,
  BCdata_EventStudy$postPeriodIndices
]

clf_bounds <- function(l_vec, alpha, label) {
  centre <- as.numeric(t(l_vec) %*% post_betahat)
  spread <- sqrt(as.numeric(t(l_vec) %*% post_sigma %*% l_vec))
  grid_lb <- centre - 40 * spread
  grid_ub <- centre + 40 * spread

  result <- createSensitivityResults_relativeMagnitudes(
    betahat = BCdata_EventStudy$betahat,
    sigma = BCdata_EventStudy$sigma,
    numPrePeriods = num_pre,
    numPostPeriods = num_post,
    method = "C-LF",
    Mbarvec = c(mbar),
    l_vec = l_vec,
    alpha = alpha,
    gridPoints = grid_points,
    grid.lb = grid_lb,
    grid.ub = grid_ub
  )
  lb <- result$lb[[1]]
  ub <- result$ub[[1]]

  # An interval sitting on the grid edge is a truncation, not an answer.
  stopifnot(ub < grid_ub - 1e-8)
  stopifnot(lb > grid_lb + 1e-8)

  cat(sprintf("%-10s lb %.9f  ub %.9f  (grid +/-40se around %.6f)\n",
              label, lb, ub, centre))
  list(lb = lb, ub = ub)
}

basis <- lapply(seq_len(num_post), function(i) {
  vec <- rep(0, num_post)
  vec[i] <- 1
  vec
})

# --- joint path: one basis direction per post period, Bonferroni over periods ---
alpha_pointwise_joint <- alpha_joint / num_post
cat("joint path (alpha_pointwise =", alpha_pointwise_joint, ")\n")
joint_points <- lapply(seq_len(num_post), function(i) {
  b <- clf_bounds(basis[[i]], alpha_pointwise_joint, paste0("period_", i - 1))
  list(post_period = i - 1, lb = b$lb, ub = b$ub)
})

joint_ref <- list(
  meta = list(
    source = "HonestDiD::createSensitivityResults_relativeMagnitudes",
    method = "C-LF",
    delta = "DeltaRM",
    mbar = mbar,
    alpha_joint = alpha_joint,
    alpha_pointwise = alpha_pointwise_joint,
    pointwise_confidence_level = 1 - alpha_pointwise_joint
  ),
  l_vecs = basis,
  points = joint_points
)
write_json(joint_ref, paste0(base_path, "honest_did_joint_path_ref.json"),
           digits = 15, auto_unbox = TRUE)

# --- directional: the four basis directions plus a difference contrast ---
diff_01 <- rep(0, num_post)
diff_01[1] <- 1
diff_01[2] <- -1

directions <- c(
  lapply(seq_len(num_post), function(i) {
    list(name = paste0("period_", i - 1), l_vec = basis[[i]])
  }),
  list(list(name = "diff_0_1", l_vec = diff_01))
)

alpha_pointwise_dir <- alpha_joint / length(directions)
cat("directional (alpha_pointwise =", alpha_pointwise_dir, ")\n")
directional_points <- lapply(directions, function(dir) {
  b <- clf_bounds(dir$l_vec, alpha_pointwise_dir, dir$name)
  list(name = dir$name, lb = b$lb, ub = b$ub)
})

directional_ref <- list(
  meta = list(
    source = "HonestDiD::createSensitivityResults_relativeMagnitudes",
    method = "C-LF",
    delta = "DeltaRM",
    mbar = mbar,
    alpha_joint = alpha_joint,
    alpha_pointwise = alpha_pointwise_dir,
    pointwise_confidence_level = 1 - alpha_pointwise_dir
  ),
  directions = directions,
  points = directional_points
)
write_json(directional_ref, paste0(base_path, "honest_did_directional_ref.json"),
           digits = 15, auto_unbox = TRUE)

cat("joint-path and directional references regenerated with explicit grids.\n")
