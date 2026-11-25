# Random number generation inside compiled programs, across the packages compared in the paper.
#
# Personal notes, not part of the paper. For every package the script checks
#   draw     -- can random numbers be drawn inside the compiled / traced / taped program?
#   fresh    -- is the draw redone on every call of the compiled program (rather than frozen into
#               it at compile / tape time)? For anvl and stateless TF the program is a pure function
#               of an explicit RNG state, so "fresh" means "a new state gives new numbers without
#               recompiling" and the same state gives the same numbers.
#   seed     -- does seeding reproduce the draws, and through what (set.seed(), own seed argument,
#               explicit state)?
#   streams  -- is there a mechanism for independent streams (per chain / particle / device)?
# The generator design (global vs explicit state), the distributions offered, and GPU support are
# taken from the documentation and recorded as fixed strings in the `design`, `dists` and `gpu`
# entries (the machine this ran on is CPU only).
#
# Findings beyond the summary table (see random_numbers.md for details):
# - anvl: nv_rng_state() turns the i32 seed into ui64[2] = (low 16 bits, high 16 bits), and XLA's
#   THREE_FRY reads that as (key, counter). Only 65536 distinct keys exist, and seeds that share
#   their low 16 bits are the *same* stream shifted: seed 65536 starts where seed 0 is after two
#   64-bit blocks. Seeds 0..65535 give distinct keys. There is no split()/fold_in().
# - quickr: only runif() (incl. min/max); its draws are bit-identical to R's runif().
# - TMB / nimble / quickr / greta all hang off R's global RNG (set.seed() reproduces); TMB and
#   nimble draws are bit-identical to R's rnorm().
# - RTMB: rnorm() inside MakeTape() is evaluated once at tape time; obj$simulate()/report() rerun
#   the R function (not the tape) with plain doubles, so simulation happens in interpreted R.
# - Stan: *_rng is rejected in the model block (semantic error); in transformed data it is drawn
#   once per run.
#
# Run with `Rscript random_numbers.R`. Uses the current anvl source via pkgload::load_all().

results <- list()
record <- function(pkg, item, expr) {
  out <- tryCatch(force(expr), error = function(e) paste("error:", strsplit(conditionMessage(e), "\n")[[1L]][1L]))
  results[[pkg]][[item]] <<- out
  cat(sprintf("  %-10s %-8s %s\n", pkg, item, substr(out, 1L, 120L)))
}
yes_if <- function(cond, yes = "yes", no = "NO") if (isTRUE(cond)) yes else no
same <- function(a, b) isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance = 0))
differ <- function(a, b) !isTRUE(all.equal(as.numeric(a), as.numeric(b)))
first_error <- function(expr) {
  tryCatch({
    force(expr)
    "accepted"
  }, error = function(e) {
    lines <- trimws(strsplit(conditionMessage(e), "\n")[[1L]])
    lines <- lines[nzchar(lines)]
    lines[length(lines)]
  })
}

# --------------------------------------------------------------------------------------------
cat("\n== anvl ==\n")
suppressMessages(pkgload::load_all("/Users/sebi/r-xla/anvl-paper", quiet = TRUE))
anvl_vals <- function(x) as.numeric(as_array(x))
anvl_f <- jit(function(state) nv_rnorm(state, shape = 3L))
r1 <- anvl_f(nv_rng_state(1L))
r2 <- anvl_f(r1$state)
r3 <- anvl_f(nv_rng_state(1L))
record("anvl", "draw", "yes (nv_rnorm() inside jit())")
record("anvl", "fresh", yes_if(differ(anvl_vals(r1$values), anvl_vals(r2$values)), "yes (new state -> new draws, same program)"))
record("anvl", "seed", yes_if(same(anvl_vals(r1$values), anvl_vals(r3$values)), "yes: same state -> same draws (pure)"))
# The seed can itself be a traced value: nv_rng_state() inside jit().
anvl_g <- jit(function(seed) nv_rnorm(nv_rng_state(seed), shape = 3L)$values)
traced_seed_ok <- differ(anvl_vals(anvl_g(nv_scalar(1L))), anvl_vals(anvl_g(nv_scalar(2L))))
# Seed collision: seeds that differ only in their high 16 bits share a stream.
anvl_u <- function(seed, n) anvl_vals(nv_runif(nv_rng_state(seed), dtype = "f32", shape = n)$values)
u0 <- anvl_u(0L, 8L)
u65536 <- anvl_u(65536L, 6L)
overlap <- same(u0[3:8], u65536)
record("anvl", "streams", sprintf(
  "one state per stream via nv_rng_state(seed) (traced seed ok: %s); no split(); seed 65536 = seed 0 shifted: %s",
  yes_if(traced_seed_ok), yes_if(overlap, "YES (overlap)", "no")
))
record("anvl", "design", "explicit functional state (ui64[2], XLA THREE_FRY), returned with the draws")
record("anvl", "dists", "runif, rnorm, rbinom, sample_int, sample")
record("anvl", "gpu", "yes (lowers to stablehlo.rng_bit_generator)")

# --------------------------------------------------------------------------------------------
cat("\n== torch ==\n")
suppressMessages(library(torch))
torch_f <- jit_trace(function(x) x + torch_randn(3), torch_zeros(3))
a <- as.numeric(torch_f(torch_zeros(3)))
b <- as.numeric(torch_f(torch_zeros(3)))
torch_manual_seed(1L)
s1 <- as.numeric(torch_f(torch_zeros(3)))
torch_manual_seed(1L)
s2 <- as.numeric(torch_f(torch_zeros(3)))
gen <- torch_generator()
gen$set_current_seed(5L)
gen_ok <- tryCatch(length(as.numeric(torch_normal(0, 1, size = 3, generator = gen))) == 3L, error = function(e) FALSE)
record("torch", "draw", "yes (torch_randn() in jit_trace() / TorchScript)")
record("torch", "fresh", yes_if(differ(a, b), "yes (random op is recorded, not its value)"))
record("torch", "seed", yes_if(same(s1, s2), "yes via torch_manual_seed()"))
record("torch", "streams", sprintf("separate torch_generator() objects (generator= arg works: %s); no split/jump", yes_if(gen_ok)))
record("torch", "design", "global stateful generator per device (+ explicit stateful Generator objects)")
record("torch", "dists", "rand/randn/randint/normal/bernoulli/poisson/multinomial + distr_* (many)")
record("torch", "gpu", "yes (CUDA generator per device)")

# --------------------------------------------------------------------------------------------
cat("\n== tensorflow ==\n")
reticulate::py_require(c("tensorflow==2.15.*", "tensorflow_probability==0.23.*"), python_version = "3.10")
suppressMessages(library(tensorflow))
tf_stateless <- tf_function(function(seed) tf$random$stateless_normal(shape(3L), seed = seed), jit_compile = TRUE)
t1 <- as.numeric(tf_stateless(tf$constant(c(1L, 2L))))
t2 <- as.numeric(tf_stateless(tf$constant(c(1L, 2L))))
t3 <- as.numeric(tf_stateless(tf$constant(c(1L, 3L))))
tf_gen <- tf$random$Generator$from_seed(1L)
tf_stateful <- tf_function(function() tf_gen$normal(shape(3L)), jit_compile = TRUE)
g1 <- as.numeric(tf_stateful())
g2 <- as.numeric(tf_stateful())
tf_gen_b <- tf$random$Generator$from_seed(1L)
tf_stateful_b <- tf_function(function() tf_gen_b$normal(shape(3L)), jit_compile = TRUE)
gb <- as.numeric(tf_stateful_b())
split_gens <- tf_gen$split(2L)
split_differ <- differ(as.numeric(split_gens[[1L]]$normal(shape(2L))), as.numeric(split_gens[[2L]]$normal(shape(2L))))
stateless_split <- tf_function(function(seed) tf$random$experimental$stateless_split(seed, num = 3L))
n_split <- nrow(as.array(stateless_split(tf$constant(c(1L, 2L)))))
record("tensorflow", "draw", "yes (stateless_normal() and Generator$normal() in tf_function(jit_compile = TRUE))")
record("tensorflow", "fresh", yes_if(
  differ(g1, g2) && differ(t1, t3),
  "yes (Generator: each call advances; stateless: new seed -> new draws)"
))
record("tensorflow", "seed", yes_if(same(t1, t2) && same(g1, gb), "yes (stateless seed pair; Generator$from_seed())"))
record("tensorflow", "streams", sprintf(
  "Generator$split() (differ: %s); stateless_split() -> %d seeds",
  yes_if(split_differ), n_split
))
record("tensorflow", "design", "both: stateless ops with explicit seed tensors, or a stateful Generator variable")
record("tensorflow", "dists", "uniform/normal/truncated_normal/poisson/gamma/binomial (+ tfp distributions)")
record("tensorflow", "gpu", "yes")

# --------------------------------------------------------------------------------------------
cat("\n== greta ==\n")
suppressMessages(library(greta))
gx <- normal(0, 1, dim = 2L)
gy <- gx * 2
c1 <- calculate(gy, nsim = 2L)$gy
c2 <- calculate(gy, nsim = 2L)$gy
k1 <- calculate(gy, nsim = 2L, seed = 1L)$gy
k2 <- calculate(gy, nsim = 2L, seed = 1L)$gy
set.seed(3L)
r_a <- calculate(gy, nsim = 2L)$gy
set.seed(3L)
r_b <- calculate(gy, nsim = 2L)$gy
pois_draw <- calculate(poisson(3), nsim = 3L)[[1L]]
record("greta", "draw", "yes (calculate(nsim =) / simulate() sample greta variables in the TF graph)")
record("greta", "fresh", yes_if(differ(c1, c2), "yes (new draws per calculate() call)"))
record("greta", "seed", yes_if(same(k1, k2) && same(r_a, r_b), "yes (seed= arg, or set.seed(): TF seed drawn from R's RNG)"))
record("greta", "streams", "no explicit streams (nsim draws are batched)")
record("greta", "design", "TF seed taken from R's global RNG per call")
record("greta", "dists", sprintf("all greta distributions (normal, poisson, ...; poisson draw ok: %s)", yes_if(length(pois_draw) == 3L)))
record("greta", "gpu", "yes (compute_options = gpu_only(); docs: seed may not be respected on GPU)")

# --------------------------------------------------------------------------------------------
cat("\n== quickr ==\n")
suppressMessages(library(quickr))
q_unif <- quick(function(n) {
  declare(type(n = integer(1)))
  x <- runif(n)
  x
})
qa <- q_unif(3L)
qb <- q_unif(3L)
set.seed(1L)
qs <- q_unif(3L)
set.seed(1L)
qr <- runif(3L)
q_norm <- first_error(quick(function(n) {
  declare(type(n = integer(1)))
  x <- rnorm(n)
  x
}))
record("quickr", "draw", sprintf("runif() only (rnorm(): %s)", q_norm))
record("quickr", "fresh", yes_if(differ(qa, qb)))
record("quickr", "seed", yes_if(same(qs, qr), "yes, set.seed(); identical to R's runif()"))
record("quickr", "streams", "no (single global R stream)")
record("quickr", "design", "R's global RNG (.Random.seed)")
record("quickr", "dists", "uniform only")
record("quickr", "gpu", "no (CPU Fortran)")

# --------------------------------------------------------------------------------------------
cat("\n== ast2ast ==\n")
suppressMessages(library(ast2ast))
a2a <- first_error(translate(function(n) {
  argtypes(n |> type(int))
  return(runif(n))
}))
record("ast2ast", "draw", sprintf("no (runif(): %s)", trimws(gsub("\\s+", " ", a2a))))
record("ast2ast", "fresh", "-")
record("ast2ast", "seed", "-")
record("ast2ast", "streams", "-")
record("ast2ast", "design", "none exposed (pso() uses R's unif_rand() internally)")
record("ast2ast", "dists", "none")
record("ast2ast", "gpu", "no")

# --------------------------------------------------------------------------------------------
cat("\n== odin2/dust2 ==\n")
suppressMessages({
  library(odin2)
  library(dust2)
})
walk <- odin(
  {
    initial(x) <- 0
    update(x) <- x + Normal(0, 1)
  },
  quiet = TRUE
)
run_walk <- function(seed, n_particles) {
  sys <- dust_system_create(walk, list(), n_particles = n_particles, seed = seed)
  dust_system_run_to_time(sys, 1)
  drop(dust_system_state(sys))
}
o1 <- run_walk(1L, 3L)
o1b <- run_walk(1L, 3L)
o5 <- run_walk(1L, 5L)
sys <- dust_system_create(walk, list(), n_particles = 2L, seed = 1L)
dust_system_run_to_time(sys, 1)
first <- dust_system_state(sys)
dust_system_set_state_initial(sys)
dust_system_set_time(sys, 0)
dust_system_run_to_time(sys, 1)
second <- dust_system_state(sys)
rng_bytes <- length(dust_system_rng_state(sys))
record("odin2", "draw", "yes (Normal(), Binomial(), ... in update())")
record("odin2", "fresh", yes_if(differ(first, second), "yes (the RNG state advances across runs)"))
record("odin2", "seed", yes_if(same(o1, o1b), "yes, seed = in dust_system_create() (NULL: from R's RNG)"))
record("odin2", "streams", sprintf(
  "yes: one xoshiro stream per particle (%d bytes for 2), jump-separated; particles differ: %s; prefix stable when n_particles grows: %s",
  rng_bytes, yes_if(length(unique(o1)) == 3L), yes_if(same(o5[1:3], o1))
))
record("odin2", "design", "own per-particle xoshiro256 states held by the system")
record("odin2", "dists", "16: Normal, TruncatedNormal, Uniform, Exponential, Gamma, Beta, Binomial, BetaBinomial, Poisson, NegativeBinomial, Hypergeometric, Cauchy, LogNormal, Weibull, ZIPoisson, ZINegativeBinomial")
record("odin2", "gpu", "no (dust2 dropped dust1's GPU support)")

# --------------------------------------------------------------------------------------------
cat("\n== Stan ==\n")
suppressMessages(library(cmdstanr))
stan_dir <- tempfile("stan")
dir.create(stan_dir)
stan_gq <- cmdstan_model(write_stan_file(
  "generated quantities { real y = normal_rng(0, 1); int k = poisson_rng(3); }",
  dir = stan_dir
), quiet = TRUE)
stan_draws <- function(seed) {
  fit <- stan_gq$sample(
    fixed_param = TRUE, chains = 2, iter_sampling = 3, seed = seed,
    refresh = 0, show_messages = FALSE
  )
  posterior::as_draws_array(fit$draws("y"))
}
sd1 <- stan_draws(1L)
sd1b <- stan_draws(1L)
model_block <- suppressMessages(suppressWarnings(first_error(cmdstan_model(write_stan_file(
  "parameters { real mu; } model { real z = normal_rng(0, 1); mu ~ normal(z, 1); }",
  dir = stan_dir
), quiet = TRUE))))
stan_td <- cmdstan_model(write_stan_file(
  "transformed data { real z = normal_rng(0, 1); } generated quantities { real y = z; }",
  dir = stan_dir
), quiet = TRUE)
td <- as.numeric(stan_td$sample(
  fixed_param = TRUE, chains = 1, iter_sampling = 3, seed = 1,
  refresh = 0, show_messages = FALSE
)$draws("y"))
record("Stan", "draw", sprintf(
  "generated quantities / transformed data only; model block: %s",
  if (identical(model_block, "accepted")) "accepted" else "rejected"
))
record("Stan", "fresh", yes_if(
  differ(sd1[1L, 1L, 1L], sd1[2L, 1L, 1L]) && length(unique(td)) == 1L,
  "yes per iteration in generated quantities; once per run in transformed data"
))
record("Stan", "seed", yes_if(same(sd1, sd1b), "yes, seed = (per run)"))
record("Stan", "streams", yes_if(differ(sd1[, 1L, 1L], sd1[, 2L, 1L]), "yes: per-chain stream derived from seed + chain id"))
record("Stan", "design", "own generator (Boost), seeded per chain; not R's RNG")
record("Stan", "dists", "_rng for nearly every built-in distribution")
record("Stan", "gpu", "no (OpenCL accelerates selected likelihoods, not _rng)")

# --------------------------------------------------------------------------------------------
cat("\n== TMB ==\n")
suppressMessages(library(TMB))
tmb_dir <- tempfile("tmb")
dir.create(tmb_dir)
writeLines("
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() () {
  DATA_VECTOR(y);
  PARAMETER(mu);
  Type nll = -sum(dnorm(y, mu, Type(1), true));
  SIMULATE {
    y = rnorm(y.size(), mu, Type(1));
    Type k = rpois(Type(3));
    REPORT(y);
    REPORT(k);
  }
  return nll;
}", file.path(tmb_dir, "simrng.cpp"))
invisible(capture.output(compile(file.path(tmb_dir, "simrng.cpp"), flags = "-w")))
dyn.load(dynlib(file.path(tmb_dir, "simrng")))
tmb_obj <- MakeADFun(list(y = rep(0, 3L)), list(mu = 0), DLL = "simrng", silent = TRUE)
ta <- tmb_obj$simulate()$y
tb <- tmb_obj$simulate()$y
set.seed(1L)
ts <- tmb_obj$simulate()$y
set.seed(1L)
tr <- rnorm(3L)
record("TMB", "draw", "yes, in SIMULATE{} blocks (evaluated with double, not AD types)")
record("TMB", "fresh", yes_if(differ(ta, tb), "yes per obj$simulate() call"))
record("TMB", "seed", yes_if(same(ts, tr), "yes, set.seed(); identical to R's rnorm()"))
record("TMB", "streams", "no (single global R stream)")
record("TMB", "design", "R's global RNG (.Random.seed)")
record("TMB", "dists", "rnorm, rpois, rbinom, rgamma, rnbinom, rexp, MVNORM/GMRF/AR1 $simulate(), ...")
record("TMB", "gpu", "no")

# --------------------------------------------------------------------------------------------
cat("\n== RTMB ==\n")
suppressMessages(library(RTMB))
rtmb_tape <- MakeTape(function(x) x + rnorm(3), numeric(3))
ra <- rtmb_tape(numeric(3))
rb <- rtmb_tape(numeric(3))
rtmb_dat <- list(y = rep(0, 3L))
rtmb_nll <- function(p) {
  getAll(rtmb_dat, warn = FALSE)
  y <- OBS(y)
  y %~% dnorm(p$mu, 1)
}
rtmb_obj <- MakeADFun(rtmb_nll, list(mu = 0), silent = TRUE)
set.seed(1L)
rs <- rtmb_obj$simulate()$y
set.seed(1L)
rr <- rnorm(3L)
record("RTMB", "draw", "only at tape time: rnorm() in MakeTape() is a constant of the tape")
record("RTMB", "fresh", yes_if(differ(ra, rb), "yes", "NO: tape returns the same draws every call (frozen)"))
record("RTMB", "seed", yes_if(same(rs, rr), "obj$simulate() reruns the R function (not the tape) with set.seed()"))
record("RTMB", "streams", "no (single global R stream)")
record("RTMB", "design", "R's global RNG, evaluated in interpreted R")
record("RTMB", "dists", "R's r* functions in interpreted simulate(); %~% / OBS() simulation for d* densities")
record("RTMB", "gpu", "no")

# --------------------------------------------------------------------------------------------
cat("\n== nimble ==\n")
suppressMessages(library(nimble))
nim_f <- nimbleFunction(run = function(n = integer(0)) {
  x <- numeric(n)
  for (i in 1:n) x[i] <- rnorm(1, 0, 1)
  returnType(double(1))
  return(x)
})
nim_cf <- suppressMessages(compileNimble(nim_f, showCompilerOutput = FALSE))
na <- nim_cf(3L)
nb <- nim_cf(3L)
set.seed(1L)
ns <- nim_cf(3L)
set.seed(1L)
nr <- rnorm(3L)
record("nimble", "draw", "yes (rnorm() etc. in compiled run code)")
record("nimble", "fresh", yes_if(differ(na, nb)))
record("nimble", "seed", yes_if(same(ns, nr), "yes, set.seed(); identical to R's rnorm()"))
record("nimble", "streams", "no (single global R stream; MCMC chains run sequentially)")
record("nimble", "design", "R's global RNG (.Random.seed) via Rmath")
record("nimble", "dists", "R's r* plus rmnorm_chol, rwish, rdirch, rmulti, rcat, ...")
record("nimble", "gpu", "no")

# --------------------------------------------------------------------------------------------
cat("\n\nSummary\n\n")
items <- c("draw", "fresh", "seed", "streams", "design", "gpu")
tab <- do.call(rbind, lapply(results, function(r) vapply(items, function(i) substr(r[[i]], 1L, 45L), character(1L))))
old <- options(width = 300L)
print(noquote(tab[, 1:3]))
cat("\n")
print(noquote(tab[, 4:6]))
options(old)
