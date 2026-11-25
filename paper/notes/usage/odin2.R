# Usage example: odin2 / dust2
#
# Part of the notes comparing the packages in the paper's comparison tables (see README.md).
# odin2 describes dynamical systems (difference and differential equations), so unlike the other
# examples this one does not fit the logistic regression from setup.R: it computes the
# log-likelihood of a deterministic SIR model for incidence data and its gradient.
#
# - Models are written in an R-like domain-specific language (`odin()`), compiled to C++ via dust2.
# - dust2 runs systems and particle filters, in parallel over particles with OpenMP (`n_threads`).
# - Reverse-mode (adjoint) gradients of the log-likelihood of *deterministic* models
#   (`dust_unfilter_create()`) w.r.t. parameters marked `differentiate = TRUE`.
# - monty fits the models (MCMC, including HMC with these gradients).
# - Installed from the mrc-ide R-universe: install.packages(c("odin2", "dust2", "monty"),
#   repos = c("https://mrc-ide.r-universe.dev", "https://cloud.r-project.org")).

library(odin2)
library(dust2)

sir <- odin({
  initial(S) <- N - I0
  initial(I) <- I0
  initial(R) <- 0
  initial(incidence, zero_every = 1) <- 0
  update(S) <- S - n_SI
  update(I) <- I + n_SI - n_IR
  update(R) <- R + n_IR
  update(incidence) <- incidence + n_SI
  n_SI <- S * p_SI
  n_IR <- I * p_IR
  p_SI <- 1 - exp(-beta * I / N * dt)
  p_IR <- 1 - exp(-gamma * dt)
  beta <- parameter(differentiate = TRUE)
  gamma <- parameter(differentiate = TRUE)
  I0 <- parameter()
  N <- 1000
  cases <- data()
  cases ~ Poisson(incidence)
}, quiet = TRUE)

data <- data.frame(time = 1:20, cases = c(3, 2, 2, 2, 1, 3, 2, 5, 5, 6, 9, 6, 8, 10, 12, 15, 17, 14, 20, 25))
unfilter <- dust_unfilter_create(sir, time_start = 0, data = data)
pars <- list(beta = 0.25, gamma = 0.1, I0 = 5)

dust_likelihood_run(unfilter, pars, adjoint = TRUE)
dust_likelihood_last_gradient(unfilter) # d log-likelihood / d (beta, gamma)
