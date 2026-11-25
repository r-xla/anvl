# Function coverage of the compared packages

What each package can compile / execute *inside* its compiled, traced or taped program
(for torch: what the eager API offers; for TMB/RTMB: what can be recorded on the AD tape;
for greta: what can be a node of the model graph). Checked 2026-10-21 against the locally
installed versions.

Legend: ✓ = supported (broadly); (✓) = partially, with the limitation in the note;
– = not available (would have to be hand-written from primitives, or done outside the program).

| Category | anvl | torch | tensorflow | greta | quickr | ast2ast | odin2 | Stan | TMB | RTMB | nimble |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Elementwise arithmetic (`+ - * / ^ %%`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Elementwise math (exp, log, trig, sqrt, abs, rounding) | ✓ | ✓ | ✓ | ✓ | (✓) no round, log1p, expm1 | ✓ | ✓ | ✓ | ✓ | (✓) no round | ✓ |
| Special functions (gamma, beta, erf, digamma, Bessel) | (✓) gamma/digamma/erf; no Bessel, beta | (✓) lgamma/digamma/erf/i0; no beta | ✓ | (✓) lgamma, digamma, choose only | – | – | (✓) gamma/beta/erf/choose; no digamma | ✓ | (✓) lgamma/lbeta/Bessel; no digamma | (✓) lgamma/lbeta/Bessel; no digamma | (✓) gamma/lgamma/factorial/besselK |
| Comparison and logical operators | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | (✓) scalar `&&`/`\|\|` only | ✓ | (✓) on AD values via CondExp | (✓) not on AD values | ✓ |
| Reductions (sum, prod, mean, min/max, any/all) | ✓ | ✓ | ✓ | ✓ | (✓) no mean | ✓ | (✓) sum/prod/min/max only | ✓ | ✓ | (✓) no any/all | ✓ |
| Cumulative / scan (cumsum, cumprod, ...) | ✓ | ✓ | ✓ | ✓ | – | (✓) cumsum only | (✓) via self-referencing arrays | (✓) cumulative_sum only | (✓) hand-wriVtten loops | (✓) cumsum/cumprod only | – |
| Subsetting and subassignment (`[`, `[<-`, gather/scatter) | ✓ | ✓ | ✓ | (✓) constant indices | ✓ | ✓ | (✓) index expressions in loops | ✓ | ✓ | ✓ | ✓ |
| Shape manipulation, concatenation, broadcasting | ✓ | ✓ | ✓ | ✓ | (✓) c/cbind/rbind/t; no aperm | (✓) c/cbind/rbind/t only | (✓) fixed `dim()`, no concat | ✓ | ✓ | ✓ | (✓) c/rep/t; no cbind/rbind |
| Sorting, ranking, order statistics | ✓ | ✓ | ✓ | – | (✓) which.max/min only | (✓) sort only | – | ✓ | (✓) taped `order`, O(n²) | (✓) sort only, taped | – |
| Matrix multiplication | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | (✓) via indexed `sum()` | ✓ | ✓ | ✓ | ✓ |
| Linear algebra (Cholesky, QR, LU, SVD, eigen, solve, det) | ✓ | ✓ | ✓ | (✓) chol/solve/eigen; no QR/SVD/det | (✓) chol/solve/svd/qr.solve; no det/eigen | (✓) chol/solve/backsolve only | – | ✓ | ✓ | (✓) no QR | (✓) no QR/LU |
| Distributions: densities / CDFs / quantiles | (✓) norm/unif/exp/binom d/p/q, plogis | (✓) 8 `distr_*`, mostly log_prob | ✓ (TFP) | (✓) 28 distributions, densities only | – | – | (✓) 17, density and sampling only | ✓ ~70 densities, 35 CDFs; few quantiles | ✓ | ✓ | ✓ ~15 d/p/q/r, more d/r |
| Random number generation | ✓ | ✓ | ✓ | (✓) via `calculate()`/`simulate()` | (✓) `runif()` only | – | (✓) discrete-time models only | (✓) generated quantities only | (✓) SIMULATE block, not taped | (✓) `simulate()`, not taped | ✓ |
| Data-dependent control flow (if/while/for) | ✓ `nv_if`/`nv_while` | ✓ eager; traced bakes branches | ✓ `tf$cond`/`tf$while_loop` | – | ✓ | ✓ | (✓) inline `if` expressions only | ✓ | (✓) fixed at taping | (✓) fixed at taping | ✓ |
| ODE / DAE solvers | – | – | ✓ (TFP) | – | – | – | ✓ (+ delays) | ✓ ODE and DAE | (✓) CppAD Runge45/Rosen34 (C++) | – | – |
| Optimizers / root finding in the program | – | (✓) `optim_*`, eager R loop | ✓ (TFP) | – | – | ✓ uniroot, lbfgsb, pso, nnls | – | (✓) root finding, integrate_1d | (✓) Newton (Laplace inner) | (✓) taped `$newton()` | (✓) `optim()`, `integrate()` |
| FFT | – | ✓ | ✓ | – | – | – | – | ✓ | ✓ | ✓ | – |
| Integer / bitwise operations | ✓ | ✓ | ✓ | (✓) `%%`/`%/%` only | (✓) `%%`/`%/%` only | (✓) `%%`/`%/%` only | (✓) `%%`/`%/%` only | (✓) `%`/`%/%` only | (✓) C++ ints, constant at taping | – | (✓) `%%` only |

## Sources and notes per package

- **anvl** (this repo): `NAMESPACE` exports and `R/api*.R`. Special functions: `nv_gamma`, `nv_lgamma`, `nv_digamma`, `nv_psigamma`, `nv_erf`/`nv_erfc`/`nv_erf_inv`. Linear algebra: `nv_chol`, `nv_qr`, `nv_lu`, `nv_svd`, `nv_eigh` (symmetric only), `nv_solve`, `nv_triangular_solve`, `nv_det`/`nv_determinant`. Sorting: `nv_sort`, `nv_order`, `nv_top_k`, `nv_quantile`, `nv_median`. Distributions: d/p/q for norm, unif, exp, binom plus `nv_plogis`; RNG: `nv_rnorm`, `nv_runif`, `nv_rbinom`, `nv_sample`. Scans via `nv_cumsum`/`cumprod`/`cummax`/`cummin` and `nv_scan`. Integers: `nv_and`/`nv_or`/`nv_xor` on ints, shifts, `nv_popcnt` (tested). No FFT, ODE or optimizer.
- **torch** 0.17.0: namespace exports. `distr_*` = bernoulli, categorical, chi2, gamma, mixture_same_family, multivariate_normal, normal, poisson; `distr_normal` has cdf/icdf, `distr_gamma$cdf` returns nothing. Special: `torch_lgamma`, `digamma`, `polygamma`, `mvlgamma`, `erf`/`erfc`/`erfinv`, `i0`. No ODE solver in R torch. `jit_trace()` bakes R control flow; the eager API covers everything.
- **tensorflow** (TF 2.15 + TFP 0.23 via reticulate): checked in Python that `tfp.math.ode.BDF`, `tfp.optimizer.lbfgs_minimize`, `tfp.math.find_root_chandrupatla`, `tf.math.bessel_i0e`, `tf.math.betainc`, `tf.signal.fft`, `tf.bitwise`, `tfp.stats.percentile`, `tf.linalg.qr`/`lu` exist; `tfp.distributions` has ~270 attributes. Distributions, ODEs and optimizers need TFP (the `tfprobability` R package or `reticulate::import`). `tf_function()` does not autograph R code, so control flow must use `tf$cond` / `tf$while_loop`. Not every op lowers under `jit_compile = TRUE`.
- **greta** 0.5.1: `?functions`, `?operators`, `?distributions`, `?overloaded`, `?\`extract-replace-combine\``. A model is a static DAG, so there is no control flow. The optimisers (`opt()`, `adam()` etc.) fit the whole model from outside, and ODEs need the separate greta.dynamics extension.
- **quickr** 0.3.0.9000: `ls(quickr:::r2f_handlers)` (107 handlers), plus `quick()` tests. `cumsum`, `mean`, `sort`, `gamma`, `rnorm`, `dnorm`, `round`, `bitwAnd`, `det` all fail with "Unsupported function"; `runif` and `x[i] <- v` work.
- **ast2ast** 1.0.2: `?translate` "Supported functions" list (31 groups). No special functions, distributions or RNG. It is the only R-syntax package besides torch/TF with in-program optimizers (`uniroot`, `lbfgsb`, `pso`, `nnls`, `jacobian`). It also has Catmull-Rom splines (`cmr`) and functionals (`map`, `Reduce`, `Filter`, `apply`).
- **odin2** 0.3.46: `vignette("functions", "odin2")`, plus `odin_validate()` tests. `prod`/`min`/`max` over arrays and self-referencing arrays (a hand-written cumsum) validate. `qnorm`, `pnorm`, `dnorm`, `cumsum`, `mean`, `digamma`, `besselJ` are "Unsupported function". Distributions come from monty and are used either as `a <- Normal(...)` (draws, discrete time only) or `y ~ Normal(...)` (log-density against data). It also has `interpolate()` (constant/linear/spline) and delays.
- **Stan** (CmdStan 2.39): `stanc --dump-stan-math-signatures` (~2800 signatures), giving 72 `_lpdf`/`_lpmf`, 35 `_lcdf`, 60 `_rng`, `sort_*`, `rank`, `quantile`, `cumulative_sum`, `fft`/`fft2`, `qr*`, `svd*`, `eigendecompose*`, Bessel functions, `inc_beta`, `lambert_w*`. Quantile functions are only `inv_Phi` and `std_normal_qf`. The higher-order functions `ode_*`, `dae`, `solve_newton`/`solve_powell` and `integrate_1d` are from the Stan Functions Reference. `_rng` works only in transformed data and generated quantities.
- **TMB**: headers in `system.file("include", package = "TMB")`, namely `distributions_R.hpp` (d/p/q functions, `rnorm` etc. for SIMULATE), `convenience.hpp`, `atomic_math.hpp` (`matinv`, `logdet`, Bessel, `pbeta`/`qbeta`...), `tmbutils/order.hpp` (taped sort via CondExp), `fft.hpp`, `tmbutils/newton.hpp`, `romberg.hpp`. Any C++ runs, but only what is recorded on the tape is re-evaluated. An ODE via `CppAD::Runge45` compiled, and value and gradient matched `exp(-k)` (scratch test).
- **RTMB** 1.8: `?ADvector`, `?ADmatrix`, `?Distributions`, `?Tape` and the namespace exports (S4 methods for `besselI/J/K/Y`, `lbeta`, d/p/q functions, `fft`, `sort`, `eigen`, `svd`, `solve`, `splinefun`, `findInterval`), plus `MakeTape()` tests. `cumsum`/`cumprod`/`gamma`/`lgamma`/`det`/`chol`/`fft`/`%%` work. `digamma`, `trigamma`, `round`, `cummax`, `order`, `quantile`, `qr`, `qpois`, `pt` fail, and comparisons on AD values error ("Comparison is generally unsafe for AD types").
- **nimble**: the internal size-processing tables `nimble:::sizeCalls`, `eigenizeTranslate` and `specificCallReplacements`, which list every call the compiler accepts. They include d/p/q/r for 15 univariate distributions (beta, binom, chisq, dexp, exp, gamma, invgamma, lnorm, logis, nbinom, norm, pois, t, unif, weibull), d/r only for the multivariate ones (dmnorm, dwish, ddirch, dmulti, dcat, dcar, dlkj...), `nimEigen`, `nimSvd`, `nimOptim`, `nimIntegrate`, `bessel_k`. There is no `cumsum`, `sort`, `cbind`, `qr` or FFT. Control flow means the run code of a `nimbleFunction`; model code (`nimbleCode`) allows only `for` loops and `if` on constants.

## Cells I am unsure about

- **TMB, ODE**: CppAD's `Runge45` / `Rosen34` work in a test, but they are CppAD utilities, not documented TMB API. They may not work under the TMBad framework, and the test build used `-DCPPAD_FRAMEWORK`.
- **TMB, comparisons / control flow**: C++ comparisons on AD types compile but are evaluated once at taping. `CppAD::CondExp*` gives taped branches. Marked (✓) for both.
- **torch, distributions**: whether cdf/icdf exist beyond `distr_normal` was not checked systematically (`distr_gamma$cdf` returned an empty value).
- **torch, optimizers** and **anvl, optimizers**: torch's `optim_*` runs as an eager R loop, so it counts as "in the program" only loosely. In anvl a hand-written optimizer inside `nv_while` + `gradient()` is possible but not provided, so it is marked –.
- **nimble, special functions**: from the compiler tables only (gamma, lgamma, factorial/lfactorial, besselK, logit/probit). I found no `lbeta`/`digamma`/`erf`, but did not compile a test. Not all of these support `nimDerivs`.
- **nimble, shape**: `nimMatrix`/`nimArray` can reshape a vector, but there is no `cbind`/`rbind`/`aperm`. Calling this (✓) is a judgement call.
- **greta, reductions**: `any`/`all` are not documented. ✓ is based on sum/prod/min/max/mean.
- **Stan, RNG**: (✓) because `_rng` is restricted to transformed data / generated quantities; inside those blocks it is fully general.
- **odin2, subsetting / matmul**: rated from the vignette's implicit-loop indexing. Gather with an integer index array was not tested.
- **Integer / bitwise row**: only anvl, torch and tensorflow have bitwise operations. Every other –/(✓) cell rests on the absence of `bitwAnd`-like functions in the docs; only quickr was tested directly.
