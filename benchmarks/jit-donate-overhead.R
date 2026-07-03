# Benchmark: pure launch overhead of a jitted call, measured with buffer
# donation so no phantom output buffer is allocated during pjrt_execute().
#
# With `donate = "operand"` the (single) output aliases the donated input, so
# the executable has no unaliased outputs and the dispatcher allocates nothing
# per call -- the measured time is dispatch + enqueue only. We chain
# `x <- g(x)` so every iteration has a valid (fresh) buffer to donate; PJRT
# tracks the dependency, so iterations pipeline and the loop is bounded by
# launch overhead, not compute (scalar f32 log is trivial).
#
# Reference points printed alongside:
#   * raw pjrt_execute() on the same donated executable (the pjrt floor -- what
#     a hypothetical zero-overhead frontend would pay),
#   * the same jit WITHOUT donation (adds the per-call phantom pjrt_empty()
#     allocation that our memory-management model requires for unaliased
#     outputs; JAX does not pay this, which bounds how close we can get).

devtools::load_all(quiet = TRUE)

n_iter <- 2000L
n_reps <- 15L
device_name <- Sys.getenv("PJRT_PLATFORM", "cpu")
cat("device:", device_name, " iters:", n_iter, "x", n_reps, "reps (min)\n\n")

# Minimum over repetitions: the loop is deterministic, so the minimum is the
# run least disturbed by GC / scheduler noise -- the honest overhead estimate.
per_call_us <- function(f, x, n = n_iter, reps = n_reps) {
  best <- Inf
  for (r in seq_len(reps)) {
    gc(FALSE)
    t0 <- Sys.time()
    for (i in seq_len(n)) {
      x <- f(x)
    }
    await(x)
    best <- min(best, as.numeric(Sys.time() - t0))
  }
  round(best / n * 1e6, 1)
}

# --- jitted nv_log with donation (no phantom alloc) --------------------------
g_donate <- jit(function(operand) nv_log(operand), donate = "operand")
x <- nv_array(1.5)
await(x)
x <- g_donate(x) # warm up / compile
us_donate <- per_call_us(g_donate, x)

# --- same jit without donation (phantom pjrt_empty per call) -----------------
g_plain <- jit(function(operand) nv_log(operand))
y <- nv_array(1.5)
await(y)
y <- g_plain(y)
us_plain <- per_call_us(g_plain, y)

# --- raw pjrt floor: execute the donated executable directly -----------------
prep <- jit_prepare_args(list(operand = nv_array(1.5)), character(), device = NULL, backend = "xla")
avals <- to_avals(prep$args_flat, prep$is_static_flat)
compiled <- compile_xla(
  function(operand) nv_log(operand),
  args_flat = avals,
  in_tree = prep$in_tree,
  donate = "operand",
  device = NULL,
  arg_devices = list(tengen::device(prep$args_flat[[1L]]))
)
stopifnot(length(compiled$phantom_specs) == 0L) # donated -> no phantoms
exec <- compiled$exec
buf <- pjrt::pjrt_buffer(1.5)
best <- Inf
for (r in seq_len(n_reps)) {
  gc(FALSE)
  t0 <- Sys.time()
  for (i in seq_len(n_iter)) {
    buf <- pjrt::pjrt_execute(exec, buf, check = FALSE)
  }
  pjrt::await(buf)
  best <- min(best, as.numeric(Sys.time() - t0))
}
us_raw <- round(best / n_iter * 1e6, 1)

cat(sprintf("raw pjrt_execute (donated, floor): %7.1f us\n", us_raw))
cat(sprintf("jit launch, donate = 'operand'   : %7.1f us  (+%.1f over floor)\n", us_donate, us_donate - us_raw))
cat(sprintf("jit launch, no donation          : %7.1f us  (+%.1f phantom alloc)\n", us_plain, us_plain - us_donate))
