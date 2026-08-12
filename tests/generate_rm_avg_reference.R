# Regenerate honest_did_rm_avg_ref.json with an explicit, wide confidence grid.
#
# Why this script exists
# ----------------------
# The fixture it replaces was produced with HonestDiD's DEFAULT grid, and the
# C-LF upper bound came back truncated at the grid's upper endpoint. Two things
# in the old file give it away: the upper bound is bit-identical across Mbar
# (0.353100805509 at both 0.5 and 1.0) while the lower bound moves, and at
# Mbar = 1.0 that upper bound sits BELOW its own identified-set upper bound
# (0.417122171), which no valid confidence set can do.
#
# Rust was being asked to reproduce a clipped number and failing, correctly.
#
# The grid here is centred on the point estimate and spans +/- 40 standard
# errors of l'betahat, which is far outside any plausible confidence set for
# this data, so the returned bounds are interior and mean what they say.
# gridPoints is raised to keep resolution fine despite the wider span: the
# returned bounds are grid points, so the grid spacing is the resolution floor
# on any comparison against them.
#
# The identified sets are NOT regenerated. HonestDiD 0.2.8 exports no
# identified-set function, and the existing id_lb / id_ub already agree with the
# Rust implementation to 1e-6, so they are carried over from the old fixture
# rather than invented here.

library(HonestDiD)
library(jsonlite)

base_path <- "tests/"
out_path <- paste0(base_path, "honest_did_rm_avg_ref.json")

data(BCdata_EventStudy)
num_pre <- length(BCdata_EventStudy$prePeriodIndices)
num_post <- length(BCdata_EventStudy$postPeriodIndices)

# The average post-period effect: equal weight on every post period.
l_vec <- rep(1 / num_post, num_post)

post_betahat <- BCdata_EventStudy$betahat[BCdata_EventStudy$postPeriodIndices]
post_sigma <- BCdata_EventStudy$sigma[
  BCdata_EventStudy$postPeriodIndices,
  BCdata_EventStudy$postPeriodIndices
]
centre <- as.numeric(t(l_vec) %*% post_betahat)
spread <- sqrt(as.numeric(t(l_vec) %*% post_sigma %*% l_vec))

grid_lb <- centre - 40 * spread
grid_ub <- centre + 40 * spread
grid_points <- 10000

cat(sprintf(
  "l'betahat = %.9f, se = %.9f, grid = [%.9f, %.9f], %d points (spacing %.3e)\n",
  centre, spread, grid_lb, grid_ub, grid_points,
  (grid_ub - grid_lb) / (grid_points - 1)
))

original <- constructOriginalCS(
  betahat = BCdata_EventStudy$betahat,
  sigma = BCdata_EventStudy$sigma,
  numPrePeriods = num_pre,
  numPostPeriods = num_post,
  l_vec = l_vec,
  alpha = 0.05
)

mbars <- c(0.5, 1.0)
previous <- fromJSON(out_path, simplifyVector = FALSE)

rows <- lapply(seq_along(mbars), function(i) {
  mbar <- mbars[[i]]
  result <- createSensitivityResults_relativeMagnitudes(
    betahat = BCdata_EventStudy$betahat,
    sigma = BCdata_EventStudy$sigma,
    numPrePeriods = num_pre,
    numPostPeriods = num_post,
    method = "C-LF",
    Mbarvec = c(mbar),
    l_vec = l_vec,
    alpha = 0.05,
    gridPoints = grid_points,
    grid.lb = grid_lb,
    grid.ub = grid_ub
  )
  lb <- result$lb[[1]]
  ub <- result$ub[[1]]

  # Carry the identified set over from the previous fixture; see the header.
  prev <- previous$relative_magnitude[[i]]
  stopifnot(abs(prev$Mbar - mbar) < 1e-12)

  # A confidence set has to contain its identified set. If this trips, the grid
  # is still too narrow -- widen it rather than accepting the number.
  stopifnot(ub >= prev$id_ub - 1e-8)
  stopifnot(lb <= prev$id_lb + 1e-8)

  # And it must not be sitting on the grid edge.
  stopifnot(ub < grid_ub - 1e-8)
  stopifnot(lb > grid_lb + 1e-8)

  cat(sprintf(
    "Mbar %.1f: old ub %.9f -> new ub %.9f (id_ub %.9f)\n",
    mbar, prev$ub, ub, prev$id_ub
  ))

  list(
    lb = lb,
    ub = ub,
    method = "C-LF",
    Delta = "DeltaRM",
    Mbar = mbar,
    id_lb = prev$id_lb,
    id_ub = prev$id_ub
  )
})

avg_ref <- list(
  l_vec = l_vec,
  # as.numeric, because constructOriginalCS returns 1x1 matrices and write_json's
  # auto_unbox does not reach inside one: they serialise as [[0.1838]] rather than
  # 0.1838, which fails to deserialise on the Rust side. The first attempt at this
  # script shipped that and broke five tests that had been passing.
  original = list(
    lb = as.numeric(original$lb),
    ub = as.numeric(original$ub),
    method = "Original"
  ),
  relative_magnitude = rows
)

write_json(avg_ref, out_path, digits = 15, auto_unbox = TRUE)
cat("honest_did_rm_avg_ref.json regenerated with an explicit grid.\n")
