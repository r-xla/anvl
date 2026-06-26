# Benchmark: launch/dispatch overhead of an anvl primitive (prim_add) vs the
# actual computation time spent inside pjrt (pjrt_execute + device compute).
#
# Strategy
# --------
# Calling `prim_add(x, y)` eagerly dispatches through anvl's jit machinery into
# `pjrt::pjrt_execute`, which returns *asynchronously* (a not-yet-ready buffer).
# `await()` blocks until the device has produced the result.
#
# To respect the async nature we:
#   * await() the inputs before starting any measurement (so input transfer /
#     prior compute is never counted), and
#   * await() the output before stopping the clock (so we capture the full
#     round-trip, not just the enqueue).
#
# We warm up each size once so the compilation cache is hot -- we explicitly do
# NOT want to measure compilation overhead.
#
# We separate the call into layers:
#
#   prim_add(x, y)                     await(.)
#   |------------ anvl launch -------->|--- pjrt compute (wait) -->|
#   | R dispatch | pjrt_execute enqueue|
#
#   1. anvl_launch : time to call prim_add and get back an (async) buffer, NOT
#                    awaited. Pure dispatch/enqueue cost: jit_prepare_call +
#                    to_avals + LRU cache lookup + arg unwrap + pjrt_execute.
#   2. pjrt_launch : same but calling pjrt::pjrt_execute directly on the cached
#                    executable (skipping anvl's R-side dispatch). Isolates the
#                    pjrt_execute launch cost alone.
#   3. roundtrip   : anvl_launch + await(output). Full latency of one op.
#   4. await       : roundtrip - anvl_launch, i.e. time blocked waiting for pjrt
#                    to actually finish the computation (device + sync).
#
# anvl_launch - pjrt_launch  = anvl R-side dispatch overhead.
# await                      = real pjrt compute time.
#
# Sweeping the input size shows the fixed launch overhead staying ~flat while
# the compute portion (await) grows with n.

devtools::load_all(quiet = TRUE)
suppressPackageStartupMessages(library(bench))

cat("pjrt:", as.character(packageVersion("pjrt")), "from", find.package("pjrt"), "\n")

device_name <- Sys.getenv("PJRT_PLATFORM", "cpu")
cat("device:", device_name, "\n")
cat("bench:", as.character(packageVersion("bench")), "\n")
cat("pjrt :", as.character(packageVersion("pjrt")), "\n\n")

# Pull the compiled executable + phantom specs that anvl cached for prim_add(x, y).
# This lets us drive pjrt::pjrt_execute directly, bypassing anvl's R dispatch.
get_cached_exec <- function() {
  jitted <- environment(prim_add)$jit_fns[["xla"]]
  cache <- environment(jitted)$cache
  key <- cache$keys_mru_to_lru()[[1L]]
  cache$get(key) # list(exec, out_tree, const_arrays, ambiguous_out, device, phantom_specs)
}

sizes <- c(1L, 100L, 1e4L, 1e6L)
min_iter <- 500L

results <- lapply(sizes, function(n) {
  # Build inputs and make sure they are fully materialised on device before we
  # start timing anything.
  x <- nv_array(as.double(seq_len(n)), device = device_name)
  y <- nv_array(as.double(seq_len(n)), device = device_name)
  await(x)
  await(y)

  # Warm up: ensures the executable for this (shape, dtype) is cached, so no
  # compilation cost leaks into the measurement.
  await(prim_add(x, y))

  cached <- get_cached_exec()
  exec <- cached[[1L]]
  consts <- cached[[3L]]
  pdevice <- cached[[5L]]
  phantom_specs <- cached[[6L]]
  xb <- x$data
  yb <- y$data

  # Faithful replica of jit_call_xla's pjrt invocation (phantom buffers are
  # freshly allocated each call because they are donated to the outputs).
  raw_execute <- function() {
    phantom_bufs <- lapply(phantom_specs, function(spec) {
      pjrt::pjrt_empty(dtype = spec$dtype, shape = spec$shape, device = pdevice)
    })
    rlang::exec(
      pjrt::pjrt_execute,
      exec,
      !!!consts,
      xb,
      yb,
      !!!phantom_bufs,
      simplify = FALSE
    )
  }

  # 1. anvl launch only: full R dispatch + enqueue, do not wait for the result.
  bm_anvl_launch <- bench::mark(
    prim_add(x, y),
    min_iterations = min_iter, check = FALSE, filter_gc = FALSE
  )

  # 2. raw pjrt_execute launch only (bypassing anvl dispatch).
  bm_pjrt_launch <- bench::mark(
    raw_execute(),
    min_iterations = min_iter, check = FALSE, filter_gc = FALSE
  )

  # 3. full round-trip through anvl: dispatch + wait for pjrt to finish.
  bm_roundtrip <- bench::mark(
    await(prim_add(x, y)),
    min_iterations = min_iter, check = FALSE, filter_gc = FALSE
  )

  anvl_launch <- as.numeric(bm_anvl_launch$median)
  pjrt_launch <- as.numeric(bm_pjrt_launch$median)
  roundtrip <- as.numeric(bm_roundtrip$median)

  data.frame(
    n = n,
    anvl_launch_us = anvl_launch * 1e6,
    pjrt_launch_us = pjrt_launch * 1e6,
    anvl_dispatch_us = (anvl_launch - pjrt_launch) * 1e6,
    await_us = (roundtrip - anvl_launch) * 1e6,
    roundtrip_us = roundtrip * 1e6
  )
})

res <- do.call(rbind, results)
res$overhead_frac <- res$anvl_launch_us / res$roundtrip_us

fmt <- res
num <- setdiff(names(fmt), "n")
fmt[num] <- lapply(fmt[num], round, 2)
fmt$overhead_frac <- round(res$overhead_frac, 3)

cat("==== prim_add launch overhead vs pjrt compute (median per call) ====\n\n")
print(
  fmt[c("n", "anvl_dispatch_us", "pjrt_launch_us", "anvl_launch_us", "await_us", "roundtrip_us", "overhead_frac")],
  row.names = FALSE
)

# `await_us` is roundtrip - anvl_launch, i.e. a difference of two independently
# measured ~600us quantities. For small n the true compute time is far below
# that subtraction's noise floor, so values near zero (or slightly negative)
# just mean "compute is negligible compared to launch overhead".

cat("\nanvl_dispatch_us = anvl R-side dispatch (prepare_call, to_avals, cache lookup, wrap)\n")
cat("pjrt_launch_us   = raw pjrt::pjrt_execute enqueue (phantom alloc + execute, no await)\n")
cat("anvl_launch_us   = anvl_dispatch + pjrt_launch (full prim_add call, not awaited)\n")
cat("await_us         = time blocked in await() waiting for pjrt to finish computing\n")
cat("roundtrip_us     = anvl_launch + await = full latency of one prim_add call\n")
cat("overhead_frac    = anvl_launch / roundtrip (fraction that is pure launch overhead)\n")
