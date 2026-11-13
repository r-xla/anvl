# Usage examples

Personal notes, not part of the paper: how each package compared in the comparison tables
(`../../comparison-tables.md`) is used, one file per package. Every example except `odin2.R`
computes the same quantity, the negative log posterior of a Bayesian logistic regression with a
standard normal prior,

    U(beta) = -sum_i (y_i x_i' beta - log(1 + exp(x_i' beta))) + ||beta||^2 / 2,

and its gradient at `beta0`; `setup.R` simulates the data and prints the reference values
(U = 124.1149, gradient = -26.29, 31.32, -5.80). Where a package is built for it, the example
also samples from the posterior.

| File | Package | What it shows |
|---|---|---|
| `anvl.R` | anvl | `jit(value_and_gradient())`, eager execution |
| `torch.R` | torch | eager autograd |
| `tensorflow.R` | tensorflow | `tf_function(jit_compile = TRUE)` with `GradientTape` |
| `greta.R` | greta | model definition and HMC |
| `quickr.R` | quickr | compiled value and hand-written gradient (no AD) |
| `ast2ast.R` | ast2ast | compiled value and reverse-mode `deriv()` |
| `odin2.R` | odin2, dust2 | SIR model, adjoint gradient of the log-likelihood |
| `stan.R` | Stan (cmdstanr) | sampling, `grad_log_prob()` |
| `tmb.R` | TMB | C++ template, `$fn()` / `$gr()` |
| `rtmb.R` | RTMB | R objective, `$fn()` / `$gr()` / `$he()` |
| `nimble.R` | nimble | `nimbleFunction` with `derivs()`, model and MCMC |

Run a file with `Rscript <file>`. `tensorflow.R` and `greta.R` request TensorFlow 2.15,
TensorFlow Probability 0.23 and Python 3.10 through `reticulate::py_require()` (the versions greta
needs); `stan.R` needs CmdStan; odin2, dust2 and monty come from https://mrc-ide.r-universe.dev.
