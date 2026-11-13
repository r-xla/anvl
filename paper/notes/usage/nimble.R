# Usage example: nimble
#
# Part of the notes comparing the packages in the paper's comparison tables (see README.md).
# Every example computes the negative log posterior U(beta) of a Bayesian logistic regression and
# its gradient at `beta0`; setup.R simulates the data and computes the reference values.
#
# - Models are written in a BUGS-like language (`nimbleCode()`); algorithms and functions in a
#   subset of R (`nimbleFunction()`), with declared argument types. Both are translated to C++ and
#   compiled by `compileNimble()`.
# - AD via CppAD, enabled per function with `buildDerivs`; `nimDerivs()` returns value, gradient
#   and Hessian (orders 0 to 2).
# - Built for MCMC on graphical models (`buildMCMC()`; HMC via the nimbleHMC package).

here <- local({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) dirname(sub("^--file=", "", f)) else "."
})
source(file.path(here, "setup.R"))

library(nimble)

# Derivatives are built for `run`; `derivs()` in another method of the same nimbleFunction
# computes them. `wrt` indexes the flattened arguments, so 1:3 is `beta`.
logreg_nimble <- nimbleFunction(
  setup = function() {},
  run = function(beta = double(1), X = double(2), y = double(1)) {
    eta <- (X %*% beta)[, 1]
    u <- -sum(y * eta - log(1 + exp(eta))) + 0.5 * sum(beta * beta)
    returnType(double(0))
    return(u)
  },
  methods = list(
    grad = function(beta = double(1), X = double(2), y = double(1)) {
      d <- derivs(run(beta, X, y), wrt = 1:3, order = 0:1)
      returnType(ADNimbleList())
      return(d)
    }
  ),
  buildDerivs = "run"
)

c_nimble <- compileNimble(logreg_nimble(), showCompilerOutput = FALSE)
d <- c_nimble$grad(beta0, X, as.double(y))
d$value
drop(d$jacobian)

code_nimble <- nimbleCode({
  for (j in 1:p) beta[j] ~ dnorm(0, sd = 1)
  for (i in 1:n) y[i] ~ dbern(ilogit(inprod(X[i, 1:p], beta[1:p])))
})
model_nimble <- nimbleModel(code_nimble, constants = list(n = n, p = p, X = X),
  data = list(y = y), inits = list(beta = beta0))
samples <- nimbleMCMC(model = model_nimble, niter = 1000, nburnin = 500, progressBar = FALSE)
colMeans(samples)
