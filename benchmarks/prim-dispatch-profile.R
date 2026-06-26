# Fine-grained profile of the anvl R-side dispatch overhead for prim_add.
#
# From prim-overhead.R we know one cache-hit prim_add(x, y) call costs ~600 us,
# of which only ~60-75 us is the raw pjrt_execute enqueue. The remaining
# ~500-550 us is anvl's R-side dispatch. Here we find out where that goes.
#
# The cache-hit path in jit_xla_impl (R/backend-xla.R) is:
#   1. jit_prepare_call(match.call(), parent.frame(), static, device, backend)
#        - as.list(call); lapply(eval)         ; resolve args
#        - build_tree(mark_some(args, static))  ; tree structure
#        - flatten(args)                        ; flat arg list
#        - .mapply(check_jit_input, ...)        ; validate inputs
#   2. to_avals(args_flat, is_static_flat)      ; abstract values
#   3. cache_key <- list(in_tree, avals_in, device); cache$get(cache_key)
#        - utils::gethash on a composite key (hashes the whole key)
#   4. jit_call_xla(...)
#        - lapply(copy_buffer)                  ; arg unwrap (no-op copy on CPU)
#        - pjrt_empty(phantom)                  ; phantom output buffers
#        - pjrt::pjrt_execute(...)              ; <-- the real launch
#        - jit_wrap_outputs(...)                ; wrap results as AnvlArray
#
# We profile with Rprof (line.profiling) over a tight loop of small (n = 1)
# calls so compute is negligible and the samples reflect dispatch only. We
# reassign into the same variable each iteration so at most one async buffer is
# live (bounded memory). Outputs are not awaited -- we are measuring launch.

devtools::load_all(quiet = TRUE)

device_name <- Sys.getenv("PJRT_PLATFORM", "cpu")
cat("device:", device_name, "\n\n")

x <- nv_array(1, device = device_name)
y <- nv_array(1, device = device_name)
await(x)
await(y)
await(prim_add(x, y)) # warm up cache

n_iter <- 60000L

prof <- tempfile(fileext = ".out")
Rprof(prof, interval = 0.001, line.profiling = TRUE, memory.profiling = FALSE)
res <- NULL
for (i in seq_len(n_iter)) {
  res <- prim_add(x, y)
}
Rprof(NULL)
await(res)

cat("==== Rprof summary (", n_iter, "calls of prim_add(x, y), n = 1) ====\n\n")
sp <- summaryRprof(prof, lines = "both")

cat("---- by self time, top functions ----\n")
by_self <- sp$by.self
print(head(by_self, 25))

cat("\n---- by total time, top of stack ----\n")
print(head(sp$by.total, 25))

if (!is.null(sp$by.line) && nrow(sp$by.line) > 0) {
  cat("\n---- by line (top self time) ----\n")
  bl <- sp$by.line[order(-sp$by.line$self.time), ]
  print(head(bl, 30))
}

total_s <- sp$sampling.time
cat(sprintf("\ntotal sampled time: %.3f s over %d calls = %.1f us/call\n",
            total_s, n_iter, total_s / n_iter * 1e6))
