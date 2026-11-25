# Usage example: tensorflow
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

library(tensorflow)

Xtf <- tf$constant(X, dtype = tf$float64)
ytf <- tf$constant(as.double(y), dtype = tf$float64)

vg_tf <- tf_function(function(beta) {
  with(tf$GradientTape() %as% tape, {
    tape$watch(beta)
    eta <- tf$linalg$matvec(Xtf, beta)
    u <- -tf$reduce_sum(ytf * eta - tf$math$softplus(eta)) + 0.5 * tf$reduce_sum(beta * beta)
  })
  list(u, tape$gradient(u, beta))
}, jit_compile = TRUE)

res_tf <- vg_tf(tf$constant(beta0, dtype = tf$float64))
as.numeric(res_tf[[1]])
as.numeric(res_tf[[2]])
