# Benchmark: what one cache-hit call to anvl's jit() costs, and how much of
# that is anvl's native dispatcher.
#
# A jitted function that has already compiled its executable pays, on every
# subsequent call, only *launch* overhead: capture the arguments, key the cache,
# find the executable, hand the input buffers to PJRT, wrap the outputs. None of
# that is device compute. This benchmark peels that cost apart into three nested
# layers, each a strict superset of the one below, all measured on the same
# elementwise `x + y` over f32 arrays:
#
#   0. execute    the floor: impl_loaded_executable_execute(), the very call the
#                 dispatcher makes once it has found its entry, driven directly
#                 from R on a pre-built input list and cached execute options.
#                 Nothing above this line can be removed without removing PJRT.
#   1. dispatch   anvl's native dispatcher on a cache hit: flatten the args,
#                 build and hash the cache key, look it up, assemble the
#                 inputs, execute, wrap the output buffers into AnvlArrays and
#                 re-nest them. This is the whole per-call machinery -- anvl's
#                 jitted closure adds nothing but argument capture on top, so
#                 dispatch(disp, args) and the closure's "jit_run_args" fast
#                 entry are the same layer.
#   2. jit call   the jitted closure itself: layer 1 plus the match.call() +
#                 eval() argument capture. This is what a user pays.
#
# Successive differences therefore attribute the overhead:
#
#   machinery_us = (1) - (0)                   the dispatcher (keying, input
#                                              assembly, native output wrap)
#   capture_us   = (2) - (1)                   anvl's argument capture
#   compute_us   = roundtrip - (2) - await_us  the device work, alone
#
# `pjrt_execute_us` times the R-level pjrt_execute() on the same floor
# executable. It is not a floor but a comparison: a plain R wrapper around the
# same execute costs about what the entire native dispatcher costs.
#
# Layer 0 and `pjrt_execute_us` drive a hand-written add compiled here; layers
# 1-2 drive the equivalent program anvl traced. Both are one stablehlo.add over
# the same shapes, and at n = 1 and n = 100 the two agree to well under a
# microsecond -- but `machinery_us` is a difference across them, so read it as
# "about ten microseconds", not to two decimals.
#
# Reaching impl_loaded_executable_execute() and impl_execution_options_create()
# needs getFromNamespace(): they are Rcpp entry points, deliberately not
# exported. The dispatcher itself is likewise internal to anvl -- it is an
# implementation detail of jit() -- so dispatch()/dispatcher_size() are reached
# the same way.
#
# Async handling. pjrt_execute() returns before the device has computed
# anything, so layers 0-2 measure launch alone. Inputs are await()ed before
# timing. `roundtrip` additionally await()s the output, and so carries both the
# device wait and await()'s own call cost -- which is why `await_us` is measured
# separately, on an output that is already ready, and subtracted out. It is
# around 5 us and does not vary with n, so leaving it in would triple the
# apparent device time of a scalar add.
#
# A launch still *enqueues* work and allocates its output buffer, so a tight
# loop of un-awaited launches would measure the queue and the allocator rather
# than R-level dispatch. Each timed loop is therefore sized to keep the
# outstanding output bytes bounded (`reps_for()`), and every result it produced
# is await()ed once the clock has stopped.
#
# Even so, at the largest size one launch allocates 4 MB and its cost swings by
# more than a whole launch layer, so the layer decomposition is reported only
# for the sizes where every layer clears that noise floor. Launch overhead is a
# fixed cost; the large size is there to show what it takes to amortise it,
# which the second table does.
#
# The Python counterpart (jax.jit on the same add, same sizes) lives in anvl's
# benchmarks/jax-launch-overhead.py and prints a directly comparable launch_us.
#
# Usage:  Rscript benchmarks/jit-launch-overhead.R
# Requires: anvl, built against this pjrt. Set PJRT_NPROC=1 to pin the CPU
# client to one thread (the default fans out and makes `compute_us` noisier).

library(pjrt)
library(anvl)

# The Rcpp entry point the dispatcher itself calls once it has its cache entry,
# and the execute-options constructor it caches.
impl_execute <- getFromNamespace("impl_loaded_executable_execute", "pjrt")
impl_options <- getFromNamespace("impl_execution_options_create", "pjrt")

# anvl's dispatcher: internal, so reached the same way.
dispatch <- getFromNamespace("dispatch", "anvl")
dispatcher_size <- getFromNamespace("dispatcher_size", "anvl")

sizes <- c(1L, 100L, 10000L, 1000000L)
# Sizes at which one launch is cheap enough that a 3-5 us layer is measurable.
decompose_below <- 1e6

# Keep the outstanding (un-awaited) output buffers of one timed loop to roughly
# a megabyte, then spend the same total number of calls on repeated blocks.
reps_for <- function(n) max(1L, min(2000L, as.integer(1e6 %/% n)))
blocks_for <- function(reps) max(7L, min(200L, 2000L %/% reps))
# Measurements that neither allocate a device buffer nor enqueue work are not
# bound by that budget, and must not inherit it: at n = 1e6 reps_for() is 1, and
# timing a sub-10us call one iteration at a time on a GC-cold buffer measures
# the cache, not the call.
reps_pure <- 2000L

# Minimum, over blocks_for(reps) runs, of the mean per-call time of a tight
# `reps`-long loop, in microseconds. The minimum is the right estimator for a
# launch overhead: every disturbance (GC, scheduler, an unlucky eviction) can
# only add time, so the fastest block is the closest look at the true cost.
#
# Every result is kept and, once the clock has stopped, handed to `drain` to
# await the work the loop enqueued -- so the next loop times a quiet device.
# Draining only the last result would be enough on the CPU backend, where
# launches happen to retire in order, but the launches are independent and
# nothing guarantees that in general.
time_us <- function(f, reps, drain = function(x) NULL) {
  best <- Inf
  outs <- vector("list", reps)
  for (b in seq_len(blocks_for(reps))) {
    gc()
    t0 <- Sys.time()
    for (i in seq_len(reps)) {
      outs[[i]] <- f()
    }
    elapsed <- as.numeric(Sys.time() - t0, units = "secs")
    for (out in outs) {
      drain(out)
    }
    best <- min(best, elapsed / reps)
  }
  best * 1e6
}

# The harness's own cost: one loop iteration, one closure call, one list store.
# The baseline must return a non-NULL value: `outs[[i]] <- NULL` would delete
# the element and shrink the list instead of storing into it.
loop_overhead_us <- function(reps) time_us(function() 0L, reps)

# A single-op `add` program over two length-n f32 tensors -- the same
# computation anvl traces from `function(x, y) x + y`, written out directly so
# the pjrt_execute() reference needs no anvl internals.
add_exec <- function(n, device) {
  t <- sprintf("tensor<%dxf32>", n)
  src <- sprintf(
    "func.func @main(%%a: %s, %%b: %s) -> %s {
       %%0 = stablehlo.add %%a, %%b : %s
       return %%0 : %s
     }",
    t,
    t,
    t,
    t,
    t
  )
  pjrt_compile(pjrt_program(src, format = "mlir"), device = device)
}

measure <- function(n, device) {
  reps <- reps_for(n)
  overhead <- loop_overhead_us(reps)
  overhead_pure <- loop_overhead_us(reps_pure)

  jit_add <- jit(function(x, y) x + y)
  x <- nv_array(as.numeric(seq_len(n)), dtype = "f32")
  y <- nv_array(as.numeric(seq_len(n)), dtype = "f32")
  await(x)
  await(y)
  args <- list(x, y)

  await(jit_add(x, y)) # warm anvl's cache: no compile is timed below
  # The Dispatcher the jitted closure holds. Timing it directly, rather than an
  # equivalent one built here, keeps the layers differences over one executable.
  jit_run <- attr(jit_add, "jit_run_args")
  disp <- environment(jit_run)$dispatcher
  stopifnot(inherits(disp, "Dispatcher"), dispatcher_size(disp) == 1L)

  exec <- add_exec(n, device)
  xb <- pjrt_buffer(as.numeric(seq_len(n)), dtype = "f32", device = device)
  yb <- pjrt_buffer(as.numeric(seq_len(n)), dtype = "f32", device = device)
  await(xb)
  await(yb)
  # What the dispatcher hands PJRT: a plain input list and options it built once.
  inputs <- list(xb, yb)
  opts <- impl_options(integer(0), 0L)

  t_exec <- time_us(
    function() impl_execute(exec, inputs, opts),
    reps,
    drain = function(r) await(r[[1L]])
  )
  t_dispatch <- time_us(function() dispatch(disp, args), reps, drain = await)
  t_call <- time_us(function() jit_add(x, y), reps, drain = await)
  t_roundtrip <- time_us(function() await(jit_add(x, y)), reps)
  t_pjrt_execute <- time_us(
    function() pjrt_execute(exec, xb, yb, simplify = FALSE),
    reps,
    drain = function(r) await(r[[1L]])
  )
  # await() is not free even when there is nothing to wait for: it dispatches
  # through anvl's backend table into pjrt. `roundtrip` pays that on top of the
  # device wait, and `launch` does not, so the difference of the two would book
  # it as compute. Measure it on an output that is already ready and take it out.
  ready <- jit_add(x, y)
  await(ready)
  t_await <- time_us(function() await(ready), reps_pure)
  await_cost <- t_await - overhead_pure

  data.frame(
    n = n,
    execute_us = t_exec - overhead,
    machinery_us = t_dispatch - t_exec,
    capture_us = t_call - t_dispatch,
    pjrt_execute_us = t_pjrt_execute - overhead,
    launch_us = t_call - overhead,
    await_us = await_cost,
    compute_us = t_roundtrip - t_call - await_cost,
    roundtrip_us = t_roundtrip - overhead
  )
}

round_cols <- function(d) {
  d[-1L] <- lapply(d[-1L], round, 2L)
  d
}

device <- pjrt_device("cpu:0")
cat("pjrt  :", as.character(packageVersion("pjrt")), "\n")
cat("anvl  :", as.character(packageVersion("anvl")), "\n")
cat("device:", format(device), "\n\n")

invisible(measure(4L, device)) # warm the process

res <- do.call(
  rbind,
  lapply(sizes, function(n) {
    message("n = ", n)
    measure(n, device)
  })
)
res$overhead_frac <- res$launch_us / res$roundtrip_us

cat("==== where a cache-hit jit() call spends its launch, per call (us) ====\n\n")
decomp <- c(
  "n",
  "execute_us",
  "machinery_us",
  "capture_us",
  "launch_us",
  "pjrt_execute_us"
)
print(round_cols(res[res$n < decompose_below, decomp]), row.names = FALSE)

cat("\nexecute_us      = impl_loaded_executable_execute(): the PJRT call itself (floor)\n")
cat("machinery_us    = what anvl's dispatcher adds: flatten, key, hash, lookup,\n")
cat("                  assemble, wrap the outputs into AnvlArrays\n")
cat("capture_us      = anvl's match.call() + eval() argument capture\n")
cat("launch_us       = execute + machinery + capture = one jit() call, not awaited\n")
cat("pjrt_execute_us = comparison: R-level pjrt_execute() on the same floor executable\n")

cat("\n==== launch overhead against the compute it precedes (us) ====\n\n")
amort <- c(
  "n",
  "launch_us",
  "await_us",
  "compute_us",
  "roundtrip_us",
  "overhead_frac"
)
out <- round_cols(res[amort])
out$overhead_frac <- round(res$overhead_frac, 3L)
print(out, row.names = FALSE)

cat("\nawait_us      = await() on an output that is already ready: its own call cost\n")
cat("compute_us    = roundtrip - launch - await: the device wait alone\n")
cat("roundtrip_us  = launch + await + compute = full latency of one jit() call\n")
cat("overhead_frac = launch_us / roundtrip_us\n")
