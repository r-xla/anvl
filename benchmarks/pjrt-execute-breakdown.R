# Break down pjrt::pjrt_execute() into the real C++ launch vs the R-side wrapper
# overhead, and establish the baseline cost of an R -> C++ (.Call/Rcpp) call.
#
# pjrt_execute() does, in R, before the C++ launch:
#   ...names() check; list(...); lapply(check_buffer); pjrt_execution_options();
#   assert_flag(); then impl_loaded_executable_execute() [the actual C++ launch];
#   then a simplify branch.
#
# We want: how much is the C++ launch (impl_loaded_executable_execute +
# pjrt_execution_options) vs the R bookkeeping around it.

devtools::load_all(quiet = TRUE)
suppressPackageStartupMessages(library(bench))
library(pjrt)

device_name <- Sys.getenv("PJRT_PLATFORM", "cpu")
cat("device:", device_name, " pjrt:", as.character(packageVersion("pjrt")), "\n\n")

# ---- 0. Baseline R->C++ call overhead ---------------------------------------
# A no-op Rcpp function (same .Call/Rcpp machinery as pjrt's impl_* functions)
# and a no-op base-R closure, to bracket the cost of crossing into C++.
Rcpp::cppFunction("SEXP cpp_noop(SEXP x) { return x; }")
r_noop <- function(x) x
z <- 1L

b_r <- bench::mark(r_noop(z), min_iterations = 2000L, check = FALSE, filter_gc = FALSE)
b_cpp <- bench::mark(cpp_noop(z), min_iterations = 2000L, check = FALSE, filter_gc = FALSE)
# raw .Call to the compiled symbol, bypassing the R wrapper closure that
# cppFunction generates (this is what RcppExports.R wrappers ultimately do):
cpp_sym <- attr(cpp_noop, "RcppExport") # may be NULL; fall back below
b_call <- tryCatch(
  bench::mark(.Call("sourceCpp_1_cpp_noop", z), min_iterations = 2000L, check = FALSE, filter_gc = FALSE),
  error = function(e) NULL
)

cat("---- baseline call overhead (per call) ----\n")
cat(sprintf("  R closure no-op        : %6.3f us\n", as.numeric(b_r$median) * 1e6))
cat(sprintf("  Rcpp function no-op    : %6.3f us\n", as.numeric(b_cpp$median) * 1e6))
if (!is.null(b_call)) cat(sprintf("  raw .Call no-op        : %6.3f us\n", as.numeric(b_call$median) * 1e6))

# ---- Build a real execute call for prim_add(x, y) ---------------------------
x <- nv_array(1, device = device_name)
y <- nv_array(1, device = device_name)
await(x)
await(y)
await(prim_add(x, y)) # warm cache

jitted <- environment(prim_add)$jit_fns[["xla"]]
cache <- environment(jitted)$cache
cached <- cache$get(cache$keys_mru_to_lru()[[1L]])
exec <- cached[[1L]]
consts <- cached[[3L]]
pdevice <- cached[[5L]]
phantom_specs <- cached[[6L]]

# Inputs exactly as jit_call_xla assembles them.
make_inputs <- function() {
  phantom_bufs <- lapply(phantom_specs, function(s) {
    pjrt::pjrt_empty(dtype = s$dtype, shape = s$shape, device = pdevice)
  })
  c(consts, list(x$data, y$data), phantom_bufs)
}

exec_opts <- pjrt_execution_options()

# ---- 1. the real C++ launch only --------------------------------------------
# impl_loaded_executable_execute is the Rcpp entry that crosses into PJRT.
b_impl <- bench::mark(
  pjrt:::impl_loaded_executable_execute(exec, make_inputs(), exec_opts),
  min_iterations = 1000L, check = FALSE, filter_gc = FALSE
)

# impl + the per-call options object construction (as pjrt_execute does it).
b_impl_opts <- bench::mark(
  pjrt:::impl_loaded_executable_execute(exec, make_inputs(), pjrt_execution_options()),
  min_iterations = 1000L, check = FALSE, filter_gc = FALSE
)

# just pjrt_execution_options() construction
b_opts <- bench::mark(pjrt_execution_options(), min_iterations = 2000L, check = FALSE, filter_gc = FALSE)

# ---- 2. the full wrapper ----------------------------------------------------
b_full <- bench::mark(
  do.call(pjrt::pjrt_execute, c(list(exec), make_inputs(), list(simplify = FALSE))),
  min_iterations = 1000L, check = FALSE, filter_gc = FALSE
)

# input assembly cost (phantom alloc + list building) alone, for reference
b_inputs <- bench::mark(make_inputs(), min_iterations = 1000L, check = FALSE, filter_gc = FALSE)

us <- function(b) as.numeric(b$median) * 1e6
cat("\n---- pjrt_execute breakdown (n = 1, per call) ----\n")
cat(sprintf("  pjrt_execution_options()                  : %7.2f us\n", us(b_opts)))
cat(sprintf("  make_inputs() [phantom alloc + list]      : %7.2f us\n", us(b_inputs)))
cat(sprintf("  impl_loaded_executable_execute (C++ only) : %7.2f us  <-- real launch\n", us(b_impl)))
cat(sprintf("  impl_..._execute + options ctor           : %7.2f us\n", us(b_impl_opts)))
cat(sprintf("  full pjrt::pjrt_execute() wrapper         : %7.2f us\n", us(b_full)))
cat(sprintf("\n  => R wrapper overhead in pjrt_execute     : %7.2f us (full - impl - inputs)\n",
            us(b_full) - us(b_impl) - us(b_inputs)))
