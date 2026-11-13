# Usage example: quickr
#
# Part of the notes comparing the packages in the paper's comparison tables (see README.md).
# Every example computes the negative log posterior U(beta) of a Bayesian logistic regression and
# its gradient at `beta0`; setup.R simulates the data and computes the reference values.
#
# - Compiles a *typed subset* of R to Fortran by static analysis of the source: `declare(type(...))`
#   gives each argument a type and a shape (symbols such as `n`, `p` relate shapes).
# - No AD and no GPU; the gradient has to be written by hand.
# - Only supported functions can be used (e.g. no `log1p()`), but shapes can vary between calls
#   without recompiling.

here <- local({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) dirname(sub("^--file=", "", f)) else "."
})
source(file.path(here, "setup.R"))

library(quickr)

U_quickr <- quick(function(beta, X, y) {
  declare(type(beta = double(p)), type(X = double(n, p)), type(y = double(n)))
  eta <- X %*% beta
  -sum(y * eta - log(1 + exp(eta))) + 0.5 * sum(beta * beta)
})

grad_quickr <- quick(function(beta, X, y) {
  declare(type(beta = double(p)), type(X = double(n, p)), type(y = double(n)))
  r <- y - 1 / (1 + exp(-(X %*% beta)))
  -(t(X) %*% r) + beta
})

U_quickr(beta0, X, as.double(y))
grad_quickr(beta0, X, as.double(y))
