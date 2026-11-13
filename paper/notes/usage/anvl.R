# Usage example: anvl
#
# Part of the notes comparing the packages in the paper's comparison tables (see README.md).
# Every example computes the negative log posterior U(beta) of a Bayesian logistic regression and
# its gradient at `beta0`; setup.R simulates the data and computes the reference values.
#
# - Write the function in R against `AnvlArray`s; most base R generics (`%*%`, `sum()`, `exp()`, ...)
#   have methods.
# - `jit()` traces the function into a graph, lowers it to StableHLO, and compiles it with XLA (once
#   per input shapes and data types). `gradient()` / `value_and_gradient()` transform the graph by
#   reverse-mode AD. They compose.
# - R literals and matrices take the data type of the arrays they meet; here everything is `f64`.
# - GPU: allocate the arrays on `device = "cuda"`.

here <- local({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) dirname(sub("^--file=", "", f)) else "."
})
source(file.path(here, "setup.R"))

library(anvl)

U_anvl <- function(beta, X, y) {
  eta <- X %*% beta
  -sum(y * eta - log(1 + exp(eta))) + 0.5 * sum(beta * beta)
}

vg_anvl <- jit(value_and_gradient(U_anvl, wrt = "beta"))
res <- vg_anvl(
  nv_array(matrix(beta0), dtype = "f64"),
  nv_array(X, dtype = "f64"),
  nv_array(matrix(as.double(y)), dtype = "f64")
)
res

U_anvl(nv_array(matrix(beta0), dtype = "f64"), nv_array(X, dtype = "f64"),
  nv_array(matrix(as.double(y)), dtype = "f64"))
