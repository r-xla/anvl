# Usage example: TMB
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

library(TMB)

cpp <- "
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() () {
  DATA_MATRIX(X);
  DATA_VECTOR(y);
  PARAMETER_VECTOR(beta);
  vector<Type> eta = X * beta;
  Type u = Type(0.5) * (beta * beta).sum();
  for (int i = 0; i < y.size(); i++) u -= y(i) * eta(i) - log(Type(1) + exp(eta(i)));
  return u;
}
"
tmb_dir <- tempfile("tmb")
dir.create(tmb_dir)
cpp_file <- file.path(tmb_dir, "logreg.cpp")
writeLines(cpp, cpp_file)
invisible(compile(cpp_file, flags = "-w"))
dyn.load(dynlib(file.path(tmb_dir, "logreg")))

obj_tmb <- MakeADFun(list(X = X, y = as.double(y)), list(beta = beta0), DLL = "logreg",
  silent = TRUE)
obj_tmb$fn(beta0)
obj_tmb$gr(beta0)
