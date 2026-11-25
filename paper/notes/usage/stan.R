# Usage example: Stan (cmdstanr)
#
# Part of the notes comparing the packages in the paper's comparison tables (see README.md).
# Every example computes the negative log posterior U(beta) of a Bayesian logistic regression and
# its gradient at `beta0`; setup.R simulates the data and computes the reference values.
#
# - The model is written in the Stan language and transpiled to C++, which a C++ compiler compiles
#   (here via CmdStan, called from R by `cmdstanr`).
# - AD via Stan Math: operator overloading, the expression graph is recorded on every gradient
#   evaluation.
# - Built for inference: NUTS / HMC, variational inference, optimization. Model methods
#   (`log_prob()`, `grad_log_prob()`) expose the density and its gradient on the unconstrained
#   scale.
# - `bernoulli_logit_glm()` is the specialized (and OpenCL-capable) likelihood.

here <- local({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) dirname(sub("^--file=", "", f)) else "."
})
source(file.path(here, "setup.R"))

library(cmdstanr)

code <- "
data {
  int<lower=0> n;
  int<lower=0> p;
  matrix[n, p] X;
  array[n] int<lower=0, upper=1> y;
}
parameters {
  vector[p] beta;
}
model {
  beta ~ std_normal();
  y ~ bernoulli_logit_glm(X, 0, beta);
}
"
mod <- cmdstan_model(write_stan_file(code), compile_model_methods = TRUE, quiet = TRUE)
fit <- mod$sample(data = list(n = n, p = p, X = X, y = y), chains = 2, iter_warmup = 500,
  iter_sampling = 500, refresh = 0, show_messages = FALSE)
fit$summary("beta")[, c("variable", "mean", "sd")]

# Model methods: the log density drops constants, so compare the gradient (= -grad U)
fit$init_model_methods()
-fit$grad_log_prob(beta0)
