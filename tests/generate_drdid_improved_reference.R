# Improved doubly robust DiD reference fixtures.
#
# Run from the crate root:  Rscript tests/generate_drdid_improved_reference.R
#
# DRDID::drdid_imp_rc and DRDID::drdid_imp_panel fit the propensity score with
# pscore.cal (the calibrated fit of Sant'Anna and Zhao, falling back to inverse
# probability tilting and then to glm) and weight the control outcome
# regression by ps / (1 - ps). Each case records the ATT, its standard error
# and the influence function, unweighted and with sampling weights.

library(DRDID)
library(jsonlite)

stopifnot(packageVersion("DRDID") == "1.3.0")

case <- function(name, fit, data) {
  list(name = name, att = fit$ATT, se = fit$se,
       inffunc = as.vector(fit$att.inf.func), data = data)
}

# Repeated cross sections: DRDID's own simulated data.
data(sim_rc)
x_rc <- as.matrix(sim_rc[, c("x1", "x2", "x3", "x4")])
set.seed(20261119)
w_rc <- round(runif(nrow(sim_rc), 0.5, 2), 3)
rc_data <- function(w) list(y = sim_rc$y, post = sim_rc$post, d = sim_rc$d,
                            x = unname(x_rc), weights = w)
rc_plain <- drdid_imp_rc(y = sim_rc$y, post = sim_rc$post, D = sim_rc$d,
                         covariates = cbind(1, x_rc), inffunc = TRUE)
rc_weighted <- drdid_imp_rc(y = sim_rc$y, post = sim_rc$post, D = sim_rc$d,
                            covariates = cbind(1, x_rc), i.weights = w_rc,
                            inffunc = TRUE)

# Panel: a simulated panel with selection on two covariates.
set.seed(20261120)
n <- 800
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.4)
d <- rbinom(n, 1, plogis(-0.4 + 0.6 * x1 + 0.5 * x2))
y0 <- 1 + 0.5 * x1 + 0.3 * x2 + rnorm(n)
y1 <- y0 + 0.2 + 0.4 * x1 + 1.0 * d + rnorm(n, sd = 0.5)
x_panel <- cbind(x1, x2)
w_panel <- round(runif(n, 0.5, 2), 3)
panel_data <- function(w) list(y1 = y1, y0 = y0, d = d, x = unname(x_panel),
                               weights = w)
panel_plain <- drdid_imp_panel(y1 = y1, y0 = y0, D = d,
                               covariates = cbind(1, x_panel), inffunc = TRUE)
panel_weighted <- drdid_imp_panel(y1 = y1, y0 = y0, D = d,
                                  covariates = cbind(1, x_panel),
                                  i.weights = w_panel, inffunc = TRUE)

out <- list(
  rc = list(case("rc_unweighted", rc_plain, rc_data(rep(1, nrow(sim_rc)))),
            case("rc_weighted", rc_weighted, rc_data(w_rc))),
  panel = list(case("panel_unweighted", panel_plain, panel_data(rep(1, n))),
               case("panel_weighted", panel_weighted, panel_data(w_panel))))
write_json(out, "tests/drdid_improved_ref.json", digits = NA, auto_unbox = TRUE)
cat("tests/drdid_improved_ref.json written\n")
