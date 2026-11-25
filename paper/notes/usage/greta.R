# Usage example: greta
#
# Part of the notes comparing the packages in the paper's comparison tables (see README.md).
# Every example computes the negative log posterior U(beta) of a Bayesian logistic regression and
# its gradient at `beta0`; setup.R simulates the data and computes the reference values.
#
# - `tensorflow` wraps the Python library via `reticulate`: every call goes to Python. Needs a Python
#   environment with TensorFlow.
# - Eager by default; `tf_function()` traces an R function into a TensorFlow graph, and
#   `jit_compile = TRUE` compiles it with XLA (like anvl, but through Python).
# - AD via `tf$GradientTape()`, which records operations inside its block.
# - greta is a probabilistic programming language on top of TensorFlow (Probability): you declare
#   random variables and a likelihood, greta builds the TF graph and samples with HMC.
#
# greta pins old versions (TF 2.15, TFP 0.23, Python 3.10); we request them before Python starts so
# that both packages share one environment.

here <- local({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) dirname(sub("^--file=", "", f)) else "."
})
source(file.path(here, "setup.R"))

reticulate::py_require(
  c("tensorflow==2.15.*", "tensorflow_probability==0.23.*"),
  python_version = "3.10"
)

library(greta)

beta <- normal(0, 1, dim = p)
eta <- X %*% beta
distribution(y) <- bernoulli(ilogit(eta))
m <- model(beta)

draws <- mcmc(m, sampler = hmc(Lmin = 10, Lmax = 10), n_samples = 500, warmup = 500,
  chains = 2, verbose = FALSE)
summary(draws)$statistics[, c("Mean", "SD")]
