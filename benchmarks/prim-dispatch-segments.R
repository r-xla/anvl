# Per-phase timing of the optimized cache-hit dispatch for prim_add(x, y).
#
# Rather than sampling (Rprof), we reconstruct each phase's inputs and time the
# phase in isolation with bench::mark. This attributes the remaining ~380 us of
# dispatch to concrete steps so we can see what is worth attacking next.
#
# The cache-hit path (R/backend-xla.R jit_xla_impl closure) is:
#   match.call()                                  [closure prologue]
#   jit_prepare_call(call, env, static, dev, be)
#     - as.list(call)[-1]; lapply(eval)           [arg capture]
#     - build_tree(mark_some(args, static))       [tree]
#     - flatten(args)                             [flatten]
#     - .mapply(check_jit_input, ...)             [validate]
#   to_avals(args_flat, is_static_flat)
#   cache_key <- list(in_tree, avals, device); cache$get(cache_key)
#   jit_call_xla(...)
#     - lapply(unwrap) -> a$data                  [arg unwrap, no copy now]
#     - lapply(pjrt_empty) phantom
#     - pjrt::pjrt_execute(...)                   [real launch]
#     - jit_wrap_outputs(...)                     [wrap as AnvlArray]

devtools::load_all(quiet = TRUE)
suppressPackageStartupMessages(library(bench))

device_name <- Sys.getenv("PJRT_PLATFORM", "cpu")
cat("device:", device_name, " pjrt:", as.character(packageVersion("pjrt")), "\n\n")

x <- nv_array(1, device = device_name)
y <- nv_array(1, device = device_name)
await(x)
await(y)
await(prim_add(x, y)) # warm cache

# Reach the jitted closure's cache + a cached compile result.
jitted <- environment(prim_add)$jit_fns[["xla"]]
cache <- environment(jitted)$cache
static <- character()

# A representative call object + its evaluation environment.
the_call <- quote(prim_add(x, y))
env <- environment()

mb <- function(label, expr) {
  expr <- substitute(expr)
  b <- bench::mark(eval(expr, env), min_iterations = 500L, check = FALSE, filter_gc = FALSE)
  data.frame(phase = label, us = as.numeric(b$median) * 1e6)
}

# Phase inputs, computed once.
prep <- jit_prepare_call(the_call, env, static, device = NULL, backend = "xla")
avals_in <- to_avals(prep$args_flat, prep$is_static_flat)
cache_key <- cache$keys_mru_to_lru()[[1L]]
cached <- cache$get(cache_key)
exec <- cached[[1L]]
out_tree <- cached[[2L]]
consts <- cached[[3L]]
amb <- cached[[4L]]
pdevice <- cached[[5L]]
phantom_specs <- cached[[6L]]
args_flat <- prep$args_flat
is_static_flat <- prep$is_static_flat
args <- lapply(as.list(the_call)[-1L], eval, envir = env)

rows <- list(
  mb("00 full prim_add(x,y) [not awaited]", prim_add(x, y)),
  mb("01 match.call()+arg eval (prologue)", {
    cl <- the_call
    lapply(as.list(cl)[-1L], eval, envir = env)
  }),
  mb("02 jit_prepare_call (whole)", jit_prepare_call(the_call, env, static, device = NULL, backend = "xla")),
  mb("02a   mark_some+build_tree", build_tree(mark_some(args, static))),
  mb("02b   flatten", flatten(args)),
  mb("02c   .mapply(check_jit_input)", .mapply(
    function(a, s, i) if (s) a else check_jit_input(a, prep$device, prep$in_tree, i, FALSE),
    list(args_flat, is_static_flat, seq_along(args_flat)), NULL
  )),
  mb("03 to_avals", to_avals(args_flat, is_static_flat)),
  mb("04a cache_key list() build", list(prep$in_tree, avals_in, prep$device)),
  mb("04b cache$get (real hit)", cache$get(cache_key)),
  mb("05 jit_call_xla (whole)", jit_call_xla(exec, out_tree, consts, args_flat, is_static_flat, amb, pdevice, phantom_specs, copy_to_device = FALSE)),
  mb("05a   arg unwrap lapply", lapply(args_flat[!is_static_flat], function(a) a$data)),
  mb("05b   phantom pjrt_empty", lapply(phantom_specs, function(s) pjrt::pjrt_empty(dtype = s$dtype, shape = s$shape, device = pdevice))),
  mb("06 jit_wrap_outputs", jit_wrap_outputs(list(x$data), out_tree, amb, "xla"))
)

res <- do.call(rbind, rows)
res$us <- round(res$us, 1)
full <- res$us[res$phase == "00 full prim_add(x,y) [not awaited]"]
res$pct_of_full <- round(100 * res$us / full, 1)

cat("==== per-phase median time (n = 1) ====\n\n")
print(res, row.names = FALSE)
cat(sprintf("\n(full launch = %.1f us; phases are timed in isolation so they\n", full))
cat(" overlap/double-count nested steps, e.g. 02 contains 02a/02b/02c.)\n")
