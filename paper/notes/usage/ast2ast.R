# Usage example: ast2ast
#
# Part of the notes comparing the packages in the paper's comparison tables (see README.md).
# Every example computes the negative log posterior U(beta) of a Bayesian logistic regression and
# its gradient at `beta0`; setup.R simulates the data and computes the reference values.
#
# - `translate()` compiles a typed subset of R to C++ by static analysis; argument types are
#   declared with `argtypes(x |> type(...))` (scalars, vectors, matrices, user-defined structs).
# - AD is explicit: with `derivative = "reverse"`, `deriv(y, x)` inside the function returns the
#   Jacobian of `y` w.r.t. `x`; with `derivative = "forward"`, `seed()` / `get_dot()` /
#   `unseed()` propagate tangents. No higher-order derivatives.
# - CPU only; the result is an R function wrapping the compiled C++ (or an external pointer).

here <- local({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) dirname(sub("^--file=", "", f)) else "."
})
source(file.path(here, "setup.R"))

library(ast2ast)

# `y` is passed as an n x 1 matrix: ast2ast does not combine the n x 1 result of `%*%` with a
# length-n vector ("non-conformable arrays").
U_ast2ast <- translate(function(beta, X, y) {
  argtypes(beta |> type(vec(double)), X |> type(mat(double)), y |> type(mat(double)))
  eta <- X %*% beta
  u <- 0.5 * sum(beta * beta) - sum(y * eta - log(1 + exp(eta)))
  return(u)
})

# Reverse mode: `deriv(out, beta)` returns the Jacobian of `out` (here 1 x p) w.r.t. `beta`.
grad_ast2ast <- translate(function(beta, X, y) {
  argtypes(beta |> type(vec(double)), X |> type(mat(double)), y |> type(mat(double)))
  out <- numeric(1L)
  eta <- X %*% beta
  out[[1L]] <- 0.5 * sum(beta * beta) - sum(y * eta - log(1 + exp(eta)))
  return(deriv(out, beta))
}, derivative = "reverse")

U_ast2ast(beta0, X, matrix(as.double(y)))
grad_ast2ast(beta0, X, matrix(as.double(y)))
