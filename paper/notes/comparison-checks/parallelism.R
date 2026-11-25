# Parallelism support of the packages compared in the anvl paper
# ===============================================================
#
# For every package this script checks, and where cheap times:
#
#   1. intra-op   - does one large operation (matrix product, elementwise op) use several cores
#                   automatically, and how is the number of threads controlled?
#   2. devices    - can one R process drive several devices / workers?
#   3. constructs - built-in parallel constructs inside the model or program (Stan reduce_sum,
#                   TMB parallel_accumulator, dust2 n_threads, quickr declare(parallel()), ...)
#   4. shipping   - can compiled objects be sent to another R process (parallel / future / callr)?
#
# Every measurement runs in a fresh child R process (callr) so that thread settings, which most
# runtimes read once at start-up, are not polluted by earlier runs. The child does not read
# ~/.Rprofile (on the author's machine it sets OMP_NUM_THREADS=1); thread variables are set
# explicitly per run.
#
# Reported per run: wall time (`elapsed`), CPU time of the whole process (`cpu`, user + system over
# all threads) and their ratio `cores` = cpu / elapsed, the average number of busy cores. A ratio
# well above 1 shows multithreading even when there is no knob to compare against.
#
# Machine-specific setup (macOS arm64, CRAN R):
# - CRAN R for macOS compiles C/C++ without OpenMP (SHLIB_OPENMP_CXXFLAGS is empty). The OpenMP
#   checks for dust2 and TMB therefore also run with a Makevars that adds Homebrew's libomp
#   (`brew install libomp`), see `omp_makevars` below. Fortran (gfortran, used by quickr) has
#   OpenMP out of the box.
# - /usr/bin is put first on PATH so that Apple clang, not Homebrew LLVM, compiles.
#
# Packages: anvl (current source via pkgload::load_all), torch, tensorflow + greta (Python env
# via reticulate::py_require), quickr, ast2ast, odin2 + dust2 + monty, cmdstanr (CmdStan),
# TMB, RTMB, nimble, plus callr.
#
# Run: Rscript paper/notes/comparison-checks/parallelism.R   (takes ~5 minutes, mostly
# compilation). Timings are noisy when other jobs share the machine; read them as ratios.

ANVL_PATH <- "/Users/sebi/r-xla/anvl-paper"
N_CORES <- 6L # performance cores of the test machine (8 logical cores in total)
TF_PY <- c("tensorflow==2.15.*", "tensorflow_probability==0.23.*")

work_dir <- tempfile("parallelism-")
dir.create(work_dir)

# Makevars that enable OpenMP for C/C++ with Homebrew's libomp, on top of the user's Makevars.
omp_makevars <- file.path(work_dir, "Makevars.omp")
libomp <- "/opt/homebrew/opt/libomp"
has_libomp <- dir.exists(libomp)
writeLines(
  c(
    if (file.exists("~/.R/Makevars")) readLines("~/.R/Makevars"),
    sprintf("SHLIB_OPENMP_CFLAGS = -Xclang -fopenmp -I%s/include", libomp),
    sprintf("SHLIB_OPENMP_CXXFLAGS = -Xclang -fopenmp -I%s/include", libomp),
    sprintf("LDFLAGS += -L%s/lib -lomp", libomp)
  ),
  omp_makevars
)

results <- list()
record <- function(package, aspect, setting, elapsed = NA_real_, cpu = NA_real_, note = "") {
  results[[length(results) + 1L]] <<- data.frame(
    package = package, aspect = aspect, setting = setting,
    elapsed = round(elapsed, 2L), cpu = round(cpu, 2L),
    cores = round(cpu / elapsed, 1L), note = note
  )
  cat(sprintf("  [%s] %-10s %-38s elapsed=%6.2f cpu=%6.2f cores=%4.1f %s\n",
    package, aspect, setting, elapsed, cpu, cpu / elapsed, note))
}

# Run `fun` in a fresh R process with extra environment variables. Returns its value, or the
# error message as a string.
child <- function(fun, args = list(), env = character(), timeout = 600) {
  env <- c(
    callr::rcmd_safe_env(),
    PATH = paste0("/usr/bin:", Sys.getenv("PATH")),
    OMP_NUM_THREADS = "", OMP_THREAD_LIMIT = "",
    env
  )
  env <- env[!duplicated(names(env), fromLast = TRUE)]
  tryCatch(
    callr::r(fun, args = args, env = env, timeout = timeout, user_profile = FALSE),
    error = function(e) paste("ERROR:", conditionMessage(e$parent %||% e))
  )
}
`%||%` <- function(x, y) if (is.null(x)) y else x

# Code shared by the children: timing helper.
timing_code <- quote({
  tm <- function(expr, reps = 5L) {
    e <- substitute(expr)
    eval(e, parent.frame()) # warm-up (compilation, allocation)
    t0 <- proc.time()
    for (i in seq_len(reps)) eval(e, parent.frame())
    t <- proc.time() - t0
    c(elapsed = t[["elapsed"]], cpu = t[["user.self"]] + t[["sys.self"]])
  }
})

# Shipping check: in a fresh process, run `setup` (load the package) and build an object with
# the expression `make`; serialize it; in a second fresh process run `setup`, unserialize it and
# call `use` on it. Returns "OK: <value>" or "ERROR: <message>".
ship <- function(setup, make, use, env = character()) {
  raw <- child(function(setup, make) {
    eval(setup, globalenv())
    serialize(eval(make, globalenv()), NULL)
  }, args = list(setup = setup, make = make), env = env)
  if (is.character(raw)) {
    return(paste("ERROR while building:", raw))
  }
  child(function(setup, raw, use) {
    eval(setup, globalenv())
    out <- tryCatch(use(unserialize(raw)), error = function(e) paste("ERROR:", conditionMessage(e)))
    if (is.character(out)) out else paste("OK:", paste(format(unlist(out), digits = 4L), collapse = " "))
  }, args = list(setup = setup, raw = raw, use = use), env = env)
}

section <- function(x) cat("\n==== ", x, " ====\n", sep = "")

# ================================================================================================
section("anvl")
# ================================================================================================
anvl_setup <- bquote(suppressMessages(pkgload::load_all(.(ANVL_PATH), quiet = TRUE)))

anvl_intra <- function(setup, timing) {
  eval(setup, globalenv())
  eval(timing)
  n <- 2000L
  a <- nv_array(matrix(runif(n * n), n, n), dtype = "f32")
  v <- nv_array(runif(1e7) + 0.5, dtype = "f32")
  mm <- jit(function(a) a %*% a)
  lg <- jit(function(v) lgamma(v))
  list(matmul = tm(await(mm(a))), lgamma = tm(await(lg(v))))
}
r <- child(anvl_intra, list(anvl_setup, timing_code))
record("anvl", "intra-op", "matmul 2000^2 f32 x5 (default)", r$matmul[1], r$matmul[2])
record("anvl", "intra-op", "lgamma 1e7 f32 x5 (default)", r$lgamma[1], r$lgamma[2])
r <- child(anvl_intra, list(anvl_setup, timing_code),
  env = c(XLA_FLAGS = "--xla_cpu_multi_thread_eigen=false"))
record("anvl", "intra-op", "matmul, XLA_FLAGS multi_thread_eigen=false", r$matmul[1], r$matmul[2],
  "no exposed thread knob; this XLA flag is not honoured")

# Several PJRT CPU devices in one process. A sequential loop (nv_scan) is a workload that one
# device runs on a single core, so spreading it over devices is what adds cores.
anvl_devices <- function(setup, timing) {
  eval(setup, globalenv())
  loop <- jit(function(x) {
    nv_scan(init = x, steps = 100000L, body = function(c, x) list(carry = sin(c) * 0.5 + 0.1, out = NULL))$carry
  })
  mm <- jit(function(a) a %*% a)
  xs <- lapply(0:3, function(i) nv_array(runif(1000), dtype = "f32", device = sprintf("cpu:%d", i)))
  as <- lapply(0:3, function(i) nv_array(matrix(runif(2000^2), 2000), dtype = "f32", device = sprintf("cpu:%d", i)))
  invisible(lapply(xs, function(x) await(loop(x))))
  invisible(lapply(as, function(a) await(mm(a))))
  st <- function(expr) {
    t <- system.time(expr)
    c(elapsed = t[["elapsed"]], cpu = t[["user.self"]] + t[["sys.self"]])
  }
  list(
    devices = length(pjrt::devices()),
    out_device = format(device(loop(xs[[3L]]))),
    loop_1dev = st(for (i in 1:4) await(loop(xs[[1L]]))),
    loop_4dev = st({
      out <- lapply(xs, loop) # dispatch returns immediately ...
      lapply(out, await) # ... and the four programs run concurrently
    }),
    dispatch_only = st(out <- lapply(xs, loop)),
    mm_1dev = st(for (i in 1:4) await(mm(as[[1L]]))),
    mm_4dev = st(lapply(lapply(as, mm), await))
  )
}
r <- child(anvl_devices, list(anvl_setup, timing_code), env = c(PJRT_CPU_DEVICE_COUNT = "4"))
if (is.character(r)) {
  record("anvl", "devices", "PJRT_CPU_DEVICE_COUNT=4", note = r)
} else {
  note <- sprintf("%d devices; output of program on cpu:2 lives on %s", r$devices, r$out_device)
  record("anvl", "devices", "scan loop, 4 calls on 1 device", r$loop_1dev[1], r$loop_1dev[2], note)
  record("anvl", "devices", "scan loop, 1 call on each of 4 devices", r$loop_4dev[1], r$loop_4dev[2],
    sprintf("speed-up %.1fx, one R thread", r$loop_1dev[1] / r$loop_4dev[1]))
  record("anvl", "devices", "  dispatch of the 4 calls only", r$dispatch_only[1], r$dispatch_only[2],
    "async: returns before the programs finish")
  record("anvl", "devices", "matmul, 4 calls on 1 device", r$mm_1dev[1], r$mm_1dev[2])
  record("anvl", "devices", "matmul, 1 call on each of 4 devices", r$mm_4dev[1], r$mm_4dev[2],
    sprintf("speed-up %.1fx (one matmul alone does not fill all cores)", r$mm_1dev[1] / r$mm_4dev[1]))
}

anvl_jit <- quote(list(f = jit(function(x) sum(x * 2)), x = nv_array(1:4, dtype = "f32")))
use_anvl <- function(o) as_array(o$f(o$x))
record("anvl", "shipping", "AnvlArray via saveRDS/serialize",
  note = ship(anvl_setup, quote(nv_array(1:4, dtype = "f32")), function(o) as_array(o * 2)))
record("anvl", "shipping", "jit() function, not yet called",
  note = ship(anvl_setup, anvl_jit, use_anvl))
record("anvl", "shipping", "jit() function after a call (compiled)",
  note = ship(anvl_setup, bquote({
    o <- .(anvl_jit)
    o$f(o$x)
    o
  }), use_anvl))

# ================================================================================================
section("torch")
# ================================================================================================
torch_intra <- function(nt, timing) {
  library(torch)
  eval(timing)
  torch_set_num_threads(nt) # must happen before the first parallel op, once per process
  a <- torch_randn(2000, 2000)
  v <- torch_rand(1e7) + 0.5
  list(matmul = tm(torch_mm(a, a)), lgamma = tm(torch_lgamma(v)))
}
for (nt in c(1L, N_CORES)) {
  r <- child(torch_intra, list(nt, timing_code))
  record("torch", "intra-op", sprintf("matmul 2000^2 x5, torch_set_num_threads(%d)", nt), r$matmul[1], r$matmul[2],
    if (nt == 1L) "macOS: matmul goes to Accelerate (AMX), not the intra-op pool" else "")
  record("torch", "intra-op", sprintf("lgamma 1e7 x5, torch_set_num_threads(%d)", nt), r$lgamma[1], r$lgamma[2])
}
r <- child(function() {
  library(torch)
  c(threads = torch_get_num_threads(), interop = torch_get_num_interop_threads(),
    cuda = cuda_is_available(), mps = backends_mps_is_available())
})
record("torch", "devices", "defaults", note = paste(names(r), r, sep = "=", collapse = ", "))
torch_setup <- quote(library(torch))
record("torch", "shipping", "torch_tensor via base serialize",
  note = ship(torch_setup, quote(torch_tensor(c(1, 2, 3, 4))), function(o) as.numeric(o * 2)))
record("torch", "shipping", "torch_tensor via torch_serialize()",
  note = ship(torch_setup, quote(torch_serialize(torch_tensor(c(1, 2, 3, 4)))),
    function(o) as.numeric(torch_load(o) * 2)))

# ================================================================================================
section("tensorflow")
# ================================================================================================
tf_setup <- bquote({
  reticulate::py_require(.(TF_PY), python_version = "3.10")
  suppressMessages(library(tensorflow))
})
tf_intra <- function(nt, setup, timing) {
  eval(setup, globalenv())
  eval(timing)
  if (!is.na(nt)) {
    tf$config$threading$set_intra_op_parallelism_threads(nt)
    tf$config$threading$set_inter_op_parallelism_threads(nt)
  }
  a <- tf$random$normal(shape(2000L, 2000L))
  v <- tf$random$uniform(shape(1e7)) + 0.5
  mm <- tf_function(function(a) tf$matmul(a, a))
  lg <- tf_function(function(v) tf$math$lgamma(v), jit_compile = TRUE)
  list(matmul = tm(mm(a)$numpy()), lgamma = tm(lg(v)$numpy()))
}
for (nt in c(1L, N_CORES)) {
  r <- child(tf_intra, list(nt, tf_setup, timing_code), env = c(TF_CPP_MIN_LOG_LEVEL = "3"))
  if (is.character(r)) {
    record("tensorflow", "intra-op", sprintf("threads=%d", nt), note = r)
    next
  }
  record("tensorflow", "intra-op", sprintf("matmul 2000^2 x5, intra_op_threads=%d", nt), r$matmul[1], r$matmul[2])
  record("tensorflow", "intra-op", sprintf("lgamma 1e7 x5 (XLA), intra_op_threads=%d", nt), r$lgamma[1], r$lgamma[2])
}
r <- child(function(setup) {
  eval(setup, globalenv())
  # One physical CPU can be split into several logical devices before TF initializes.
  cpu <- tf$config$list_physical_devices("CPU")
  tf$config$set_logical_device_configuration(cpu[[1]], list(
    tf$config$LogicalDeviceConfiguration(), tf$config$LogicalDeviceConfiguration()))
  length(tf$config$list_logical_devices("CPU"))
}, list(tf_setup), env = c(TF_CPP_MIN_LOG_LEVEL = "3"))
record("tensorflow", "devices", "logical CPU devices configurable", note = paste("logical CPU devices:", r))
record("tensorflow", "shipping", "tf tensor via base serialize",
  note = ship(tf_setup, quote(tf$constant(c(1, 2, 3, 4))), function(o) as.numeric(o * 2), env = c(TF_CPP_MIN_LOG_LEVEL = "3")))

# ================================================================================================
section("greta")
# ================================================================================================
# greta runs as a script (callr::rscript): passed as a function through callr::r, the same model
# fails while building (`X %*% beta` does not become a greta array there).
greta_script <- function(n_cores, out) {
  file <- tempfile("greta-", tmpdir = work_dir, fileext = ".R")
  writeLines(c(
    deparse(tf_setup),
    "suppressMessages(library(greta))",
    "set.seed(1)",
    "n <- 20000L",
    "p <- 20L",
    "X <- matrix(rnorm(n * p), n, p)",
    "y <- rbinom(n, 1L, plogis(X %*% rnorm(p, sd = 0.3)))",
    "beta <- normal(0, 1, dim = p)",
    "distribution(y) <- bernoulli(ilogit(X %*% beta))",
    "m <- model(beta)",
    "# timed including greta's graph construction (a few seconds, same for both settings)",
    sprintf("t <- system.time(mcmc(m, sampler = hmc(Lmin = 5, Lmax = 5), n_samples = 200, warmup = 100,
      chains = 4, n_cores = %dL, verbose = FALSE))", n_cores),
    sprintf("saveRDS(c(elapsed = t[['elapsed']], cpu = t[['user.self']] + t[['sys.self']]), '%s')", out)
  ), file)
  file
}
for (nc in c(1L, 4L)) {
  out <- tempfile(tmpdir = work_dir, fileext = ".rds")
  res <- tryCatch(
    callr::rscript(greta_script(nc, out), user_profile = FALSE, show = FALSE, stderr = NULL,
      env = c(callr::rcmd_safe_env(), TF_CPP_MIN_LOG_LEVEL = "3")),
    error = function(e) NULL
  )
  if (!file.exists(out)) {
    record("greta", "constructs", sprintf("mcmc n_cores=%d", nc), note = "ERROR: greta script failed")
  } else {
    r <- readRDS(out)
    record("greta", "constructs", sprintf("mcmc, 4 chains, n=20000 p=20, n_cores=%d", nc), r[1], r[2],
      if (nc == 1L) "chains are vectorized in one TF graph" else "")
  }
}

# ================================================================================================
section("quickr")
# ================================================================================================
quickr_run <- function() {
  library(quickr)
  f <- quick(function(x) {
    declare(type(x = double(n)))
    out <- double(length(x))
    declare(parallel())
    for (i in seq_along(x)) {
      s <- 0
      for (k in 1:200) s <- s + sin(x[i] * k)
      out[i] <- s
    }
    out
  })
  x <- runif(2e5)
  invisible(f(x))
  t <- system.time(for (r in 1:5) f(x))
  c(elapsed = t[["elapsed"]], cpu = t[["user.self"]] + t[["sys.self"]])
}
for (nt in c(1L, N_CORES)) {
  r <- child(quickr_run, env = c(OMP_NUM_THREADS = as.character(nt)))
  record("quickr", "constructs", sprintf("declare(parallel()) loop, OMP_NUM_THREADS=%d", nt), r[1], r[2])
}
quickr_setup <- quote(library(quickr))
record("quickr", "shipping", "quick() function to another process",
  note = ship(quickr_setup, quote({
    quick(function(x) {
      declare(type(x = double(n)))
      x * 2
    })
  }), function(f) f(c(1, 2, 3, 4))))

# ================================================================================================
section("ast2ast")
# ================================================================================================
ast_setup <- quote(library(ast2ast))
record("ast2ast", "constructs", "none", note = "translate() has no parallel option; docs do not mention threads")
record("ast2ast", "shipping", "translate() function to another process",
  note = ship(ast_setup, quote({
    translate(function(x) {
      argtypes(x |> type(vec(double)))
      return(x * 2)
    })
  }), function(f) f(c(1, 2, 3, 4))))

# ================================================================================================
section("odin2 / dust2")
# ================================================================================================
dust_run <- function() {
  suppressMessages({
    library(odin2)
    library(dust2)
  })
  sir <- odin({
    update(S) <- S - n_SI
    update(I) <- I + n_SI - n_IR
    update(R) <- R + n_IR
    initial(S) <- N - I0
    initial(I) <- I0
    initial(R) <- 0
    p_SI <- 1 - exp(-beta * I / N * dt)
    p_IR <- 1 - exp(-gamma * dt)
    n_SI <- Binomial(S, p_SI)
    n_IR <- Binomial(I, p_IR)
    N <- parameter(1e6)
    I0 <- parameter(10)
    beta <- parameter(0.2)
    gamma <- parameter(0.1)
  }, quiet = TRUE)
  one <- function(nt) {
    sys <- dust_system_create(sir, list(), n_particles = 4000, dt = 0.25, n_threads = nt, seed = 1L)
    dust_system_set_state_initial(sys)
    t <- system.time(dust_system_run_to_time(sys, 1000))
    c(elapsed = t[["elapsed"]], cpu = t[["user.self"]] + t[["sys.self"]])
  }
  list(openmp = dust_openmp_support(TRUE)$has_openmp_compiler, t1 = one(1L), t4 = one(4L))
}
for (omp in c(FALSE, has_libomp)) {
  if (omp && !has_libomp) next
  r <- child(dust_run, env = if (omp) c(R_MAKEVARS_USER = omp_makevars) else character())
  if (is.character(r)) {
    record("dust2", "constructs", "n_threads", note = r)
    next
  }
  lab <- if (omp) "libomp Makevars" else "default toolchain"
  record("dust2", "constructs", sprintf("4000 particles, n_threads=1 (%s)", lab), r$t1[1], r$t1[2],
    sprintf("OpenMP compiler: %s", r$openmp))
  record("dust2", "constructs", sprintf("4000 particles, n_threads=4 (%s)", lab), r$t4[1], r$t4[2])
}
record("dust2", "constructs", "monty runners",
  note = paste("chains:", paste(grep("runner", getNamespaceExports("monty"), value = TRUE), collapse = ", ")))
dust_setup <- quote(suppressMessages({
  library(odin2)
  library(dust2)
}))
record("dust2", "shipping", "odin() generator (compiled on the fly) to another process",
  note = ship(dust_setup, bquote({
    # a persistent workdir, so the generated package outlives the process that compiled it
    odin({
      initial(x) <- 0
      update(x) <- x + a
      a <- parameter(1)
    }, quiet = TRUE, workdir = .(file.path(work_dir, "odin-ship")))
  }), function(gen) {
    sys <- dust_system_create(gen, list(a = 2))
    dust_system_run_to_time(sys, 3)
    dust_system_state(sys)
  }))

# ================================================================================================
section("Stan (cmdstanr)")
# ================================================================================================
library(cmdstanr)
stan_code <- "
functions {
  real partial_sum_lpmf(array[] int y_slice, int start, int end, matrix X, vector beta) {
    return bernoulli_logit_glm_lupmf(y_slice | X[start:end], 0, beta);
  }
}
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
  target += reduce_sum(partial_sum_lupmf, y, 1, X, beta);
}
"
stan_file <- write_stan_file(stan_code, dir = work_dir)
mod <- cmdstan_model(stan_file, cpp_options = list(stan_threads = TRUE), quiet = TRUE)
set.seed(1)
n <- 20000L
p <- 20L
X <- matrix(rnorm(n * p), n, p)
stan_data <- list(n = n, p = p, X = X, y = rbinom(n, 1L, plogis(X %*% rnorm(p, sd = 0.3))))
stan_time <- function(...) {
  t <- system.time(fit <- mod$sample(data = stan_data, iter_warmup = 200, iter_sampling = 200,
    refresh = 0, show_messages = FALSE, seed = 1, ...))
  # CmdStan runs in child processes: their CPU time is in the *.child columns.
  c(elapsed = t[["elapsed"]], cpu = sum(t[c("user.self", "sys.self", "user.child", "sys.child")], na.rm = TRUE))
}
r <- stan_time(chains = 1, threads_per_chain = 1)
record("Stan", "constructs", "reduce_sum, 1 chain, threads_per_chain=1", r[1], r[2])
r <- stan_time(chains = 1, threads_per_chain = 4)
record("Stan", "constructs", "reduce_sum, 1 chain, threads_per_chain=4", r[1], r[2], "STAN_THREADS + TBB")
r <- stan_time(chains = 4, parallel_chains = 1, threads_per_chain = 1)
record("Stan", "constructs", "4 chains, parallel_chains=1", r[1], r[2])
r <- stan_time(chains = 4, parallel_chains = 4, threads_per_chain = 1)
record("Stan", "constructs", "4 chains, parallel_chains=4", r[1], r[2], "one CmdStan process per chain")
stan_setup <- quote(library(cmdstanr))
record("Stan", "shipping", "CmdStanModel (executable on disk) to another process",
  note = ship(stan_setup, bquote(cmdstan_model(.(stan_file), cpp_options = list(stan_threads = TRUE), quiet = TRUE)), function(m) {
    fit <- m$sample(data = list(n = 2L, p = 1L, X = matrix(c(1, -1)), y = c(1L, 0L)), chains = 1,
      iter_warmup = 50, iter_sampling = 50, refresh = 0, show_messages = FALSE, threads_per_chain = 1)
    nrow(fit$draws("beta", format = "draws_df"))
  }))

# ================================================================================================
section("TMB")
# ================================================================================================
tmb_cpp <- "
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() () {
  DATA_MATRIX(X);
  DATA_VECTOR(y);
  PARAMETER_VECTOR(beta);
  parallel_accumulator<Type> u(this);
  for (int i = 0; i < y.size(); i++) {
    Type eta = (X.row(i).transpose().array() * beta).sum();
    u -= y(i) * eta - logspace_add(Type(0), eta);
  }
  u += Type(0.5) * (beta * beta).sum();
  return u;
}
"
tmb_run <- function(dir, cpp) {
  suppressMessages(library(TMB))
  file <- file.path(dir, "logreg_par.cpp")
  writeLines(cpp, file)
  invisible(compile(file, openmp = TRUE, flags = "-w"))
  dyn.load(dynlib(file.path(dir, "logreg_par")))
  set.seed(1)
  n <- 50000L
  p <- 20L
  X <- matrix(rnorm(n * p), n, p)
  y <- as.double(rbinom(n, 1L, 0.5))
  one <- function(nt) {
    openmp(nt, DLL = "logreg_par")
    obj <- MakeADFun(list(X = X, y = y), list(beta = rep(0.1, p)), DLL = "logreg_par", silent = TRUE)
    obj$gr()
    t <- system.time(for (i in 1:30) obj$gr(rep(0.1, p) + i / 1000))
    c(elapsed = t[["elapsed"]], cpu = t[["user.self"]] + t[["sys.self"]])
  }
  list(t1 = one(1L), t4 = one(4L), max = openmp(max = TRUE, DLL = "logreg_par"))
}
for (omp in c(FALSE, has_libomp)) {
  if (omp && !has_libomp) next
  d <- tempfile("tmb", tmpdir = work_dir)
  dir.create(d)
  r <- child(tmb_run, list(d, tmb_cpp), env = if (omp) c(R_MAKEVARS_USER = omp_makevars) else character())
  if (is.character(r)) {
    record("TMB", "constructs", "parallel_accumulator", note = r)
    next
  }
  lab <- if (omp) "libomp Makevars" else "default toolchain"
  record("TMB", "constructs", sprintf("parallel_accumulator, openmp(1) (%s)", lab), r$t1[1], r$t1[2],
    "parallel_accumulator in the template")
  record("TMB", "constructs", sprintf("parallel_accumulator, openmp(4) (%s)", lab), r$t4[1], r$t4[2])
}
tmb_ship_dir <- file.path(work_dir, "tmbship")
dir.create(tmb_ship_dir)
writeLines("#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() () {
  PARAMETER_VECTOR(x);
  return (x * x).sum();
}", file.path(tmb_ship_dir, "sq.cpp"))
invisible(child(function(f) {
  TMB::compile(f, flags = "-w")
}, list(file.path(tmb_ship_dir, "sq.cpp"))))
record("TMB", "shipping", "MakeADFun() object (DLL loaded in worker)",
  note = ship(bquote({
    library(TMB)
    dyn.load(dynlib(.(file.path(tmb_ship_dir, "sq"))))
  }), quote(MakeADFun(list(), list(x = c(1, 2)), DLL = "sq", silent = TRUE)),
  function(o) c(o$fn(c(1, 2)), o$gr(c(1, 2)))))

# ================================================================================================
section("RTMB")
# ================================================================================================
r <- child(function() {
  suppressMessages(library(RTMB))
  so <- file.path(system.file(package = "RTMB"), "libs", "RTMB.so")
  list(
    threads = TMB::openmp(4, DLL = "RTMB"),
    # undefined OpenMP runtime symbols (omp_* / GOMP_* / __kmpc_*) in the shared library
    omp_symbols = sum(grepl("omp_|GOMP_|__kmpc", system2("nm", c("-u", so), stdout = TRUE)))
  )
})
record("RTMB", "constructs", "TMB::openmp(4, DLL = 'RTMB')",
  note = sprintf("returns %s, but RTMB.so references %d OpenMP symbols (src/Makevars has no OpenMP flags)",
    paste(r$threads, collapse = ""), r$omp_symbols))
record("RTMB", "shipping", "MakeADFun() object to another process",
  note = ship(quote(library(RTMB)), quote(RTMB::MakeADFun(function(p) sum(p$x^2), list(x = c(1, 2)), silent = TRUE)), function(o) c(o$fn(c(1, 2)), o$gr(c(1, 2)))))

# ================================================================================================
section("nimble")
# ================================================================================================
record("nimble", "constructs", "none",
  note = "no threads in compiled code; docs: run chains via parallel::parLapply, building + compiling in each PSOCK worker")
record("nimble", "shipping", "compiled nimbleFunction to another process",
  note = ship(quote(suppressMessages(library(nimble))), quote({
    nf <- nimbleFunction(run = function(x = double(1)) {
      returnType(double(1))
      return(2 * x)
    })
    compileNimble(nf, showCompilerOutput = FALSE)
  }), function(f) f(c(1, 2, 3, 4))))

# ================================================================================================
section("Results")
# ================================================================================================
tab <- do.call(rbind, results)
old <- options(width = 250L)
print(tab, row.names = FALSE, right = FALSE)
options(old)
