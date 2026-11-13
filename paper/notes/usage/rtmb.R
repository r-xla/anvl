# Usage example: RTMB
#
# Part of the notes comparing the packages in the paper's comparison tables (see README.md).
# Every example computes the negative log posterior U(beta) of a Bayesian logistic regression and
# its gradient at `beta0`; setup.R simulates the data and computes the reference values.
#
# - TMB: the objective is a C++ template (`objective_function`), compiled by a C++ compiler.
#   `MakeADFun()` records the AD tape once (CppAD / TMBad), and `obj$fn()`, `obj$gr()`, `obj$he()`
#   evaluate it. Built for (Laplace-approximated) maximum likelihood; `sdreport()` gives standard
#   errors.
# - RTMB: the same machinery, but the objective is an R function. `MakeADFun()` evaluates it once
#   with overloaded operators to record the tape; afterwards R is no longer involved.
# - No GPU. Neither ships an MCMC sampler (tmbstan connects TMB to Stan).

here <- local({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) dirname(sub("^--file=", "", f)) else "."
})
source(file.path(here, "setup.R"))

library(RTMB)

U_rtmb <- function(par) {
  eta <- X %*% par$beta
  -sum(y * eta - log(1 + exp(eta))) + 0.5 * sum(par$beta^2)
}
obj_rtmb <- RTMB::MakeADFun(U_rtmb, list(beta = beta0), silent = TRUE)
obj_rtmb$fn(beta0)
obj_rtmb$gr(beta0)
obj_rtmb$he(beta0) # Hessian
