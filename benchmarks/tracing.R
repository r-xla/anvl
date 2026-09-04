# Tracing-pipeline benchmarks.
#
# For a few shapes of programs, measure the time spent in each phase of the
# anvl JIT pipeline:
#
#   trace_fn          -- record primitive calls into a graph
#   transform_gradient -- reverse-mode AD transform (only for *_grad rows)
#   stablehlo         -- lower the graph to StableHLO
#   pjrt_compile      -- compile StableHLO to an XLA executable
#
# Run with:
#   Rscript benchmarks/tracing.R
#
# All times are wall-clock medians over a few repetitions, in milliseconds.

suppressMessages(pkgload::load_all(quiet = TRUE))

# ----- Program builders -------------------------------------------------------
# Each builder returns list(f, args). `args` is the list of arguments to pass
# to trace_fn / gradient. We aim for programs that produce a single scalar
# output so gradient transforms are well-defined.

# Chain of N alternating binary ops on a vector.
prog_elementwise_binary <- function(N) {
  body_text <- "function(x) {\n  y <- x\n"
  for (i in seq_len(N)) {
    op <- if (i %% 2L == 0L) "+" else "*"
    body_text <- paste0(body_text, "  y <- y ", op, " x\n")
  }
  body_text <- paste0(body_text, "  sum(y)\n}\n")
  list(f = eval(parse(text = body_text)), args = list(x = nv_array(c(1, 2, 3, 4), dtype = "f32")))
}

# Chain of N unary transcendentals.
prog_elementwise_unary <- function(N) {
  body_text <- "function(x) {\n  y <- x\n"
  fns <- c("nv_exp", "nv_log", "nv_sine", "nv_cosine", "nv_sqrt")
  for (i in seq_len(N)) {
    body_text <- paste0(body_text, "  y <- ", fns[(i %% length(fns)) + 1L], "(y)\n")
  }
  body_text <- paste0(body_text, "  sum(y)\n}\n")
  list(f = eval(parse(text = body_text)), args = list(x = nv_array(c(1.1, 1.2, 1.3, 1.4), dtype = "f32")))
}

# A small MLP with N hidden layers (matmul + bias + tanh) on an input of size d.
prog_mlp <- function(N, d = 8L) {
  Ws <- lapply(seq_len(N), \(i) nv_array(matrix(rnorm(d * d) * 0.1, d, d), dtype = "f32"))
  bs <- lapply(seq_len(N), \(i) nv_array(matrix(rnorm(d) * 0.1, d, 1L), dtype = "f32"))
  body_text <- paste0(
    "function(x) {\n",
    "  y <- x\n",
    paste0(
      vapply(
        seq_len(N),
        \(i) {
          sprintf("  y <- nv_tanh(nv_matmul(Ws[[%d]], y) + bs[[%d]])", i, i)
        },
        character(1L)
      ),
      collapse = "\n"
    ),
    "\n  sum(y * y)\n}\n"
  )
  env <- new.env(parent = globalenv())
  env$Ws <- Ws
  env$bs <- bs
  f <- eval(parse(text = body_text), envir = env)
  list(f = f, args = list(x = nv_array(matrix(rnorm(d), d, 1L), dtype = "f32")))
}

# Reductions over a 2D matrix repeated N times (sum along alternating axes).
prog_reduce <- function(N, d = 8L) {
  body_text <- paste0(
    "function(x) {\n  y <- x\n",
    paste0(
      vapply(
        seq_len(N),
        \(i) {
          if (i %% 2L == 0L) "  y <- y + sum(y)" else "  y <- y * sum(y)"
        },
        character(1L)
      ),
      collapse = "\n"
    ),
    "\n  sum(y)\n}\n"
  )
  list(f = eval(parse(text = body_text)), args = list(x = nv_array(matrix(rnorm(d * d), d, d), dtype = "f32")))
}

# ----- Timing harness ---------------------------------------------------------

ms <- function(times) round(median(times) * 1000, 2)

time_repeats <- function(expr, reps) {
  expr <- substitute(expr)
  envir <- parent.frame()
  replicate(reps, {
    t0 <- Sys.time()
    invisible(eval(expr, envir = envir))
    as.numeric(Sys.time() - t0, units = "secs")
  })
}

run_one <- function(label, prog, with_grad, reps = 5L) {
  f <- prog$f
  args <- prog$args

  fn <- if (with_grad) gradient(f) else f

  # Warmup
  fwd <- trace_fn(fn, args = args)
  if (with_grad) {
    # gradient() returns a fn that, when traced, both traces f and runs
    # transform_gradient internally. We unfold that here so we can time the
    # transform separately.
    raw_fwd <- trace_fn(f, args = args)
    invisible(transform_gradient(raw_fwd, NULL))
    grad_graph <- transform_gradient(raw_fwd, NULL)
  } else {
    raw_fwd <- fwd
    grad_graph <- NULL
  }
  invisible(stablehlo(if (with_grad) grad_graph else raw_fwd))

  # Trace
  t_trace <- time_repeats(trace_fn(f, args = args), reps)

  # Gradient transform (only when applicable)
  t_grad <- if (with_grad) {
    time_repeats(transform_gradient(raw_fwd, NULL), reps)
  } else {
    NA_real_
  }

  graph_to_lower <- if (with_grad) grad_graph else raw_fwd

  # StableHLO
  t_hlo <- time_repeats(stablehlo(graph_to_lower), reps)

  # PJRT compile
  hlo <- stablehlo(graph_to_lower)
  func <- hlo[[1L]]
  src <- stablehlo::repr(func)
  program <- pjrt::pjrt_program(src = src, format = "mlir")
  t_compile <- time_repeats(pjrt::pjrt_compile(program), 3L)

  total_ms <- ms(t_trace) +
    (if (with_grad) ms(t_grad) else 0) +
    ms(t_hlo) +
    ms(t_compile)

  data.frame(
    program = label,
    grad = with_grad,
    fwd_ops = length(raw_fwd$calls),
    lower_ops = length(graph_to_lower$calls),
    trace_ms = ms(t_trace),
    grad_ms = if (with_grad) ms(t_grad) else NA_real_,
    hlo_ms = ms(t_hlo),
    compile_ms = ms(t_compile),
    total_ms = total_ms
  )
}

run_grid <- function(builder, label, sizes) {
  rows <- list()
  for (N in sizes) {
    prog <- builder(N)
    rows[[length(rows) + 1L]] <- run_one(sprintf("%s/N=%d", label, N), prog, FALSE)
    rows[[length(rows) + 1L]] <- run_one(sprintf("%s/N=%d", label, N), prog, TRUE)
  }
  do.call(rbind, rows)
}

# ----- Run --------------------------------------------------------------------

print_table <- function(df) {
  old <- options(width = 200)
  on.exit(options(old))
  print(df, row.names = FALSE)
}

cat("\n=== elementwise binary chain ===\n")
res_bin <- run_grid(prog_elementwise_binary, "binary", c(10L, 100L, 500L))
print_table(res_bin)

cat("\n=== elementwise unary chain ===\n")
res_un <- run_grid(prog_elementwise_unary, "unary", c(10L, 100L, 500L))
print_table(res_un)

cat("\n=== MLP ===\n")
res_mlp <- run_grid(prog_mlp, "mlp", c(2L, 8L, 32L))
print_table(res_mlp)

cat("\n=== reductions ===\n")
res_red <- run_grid(prog_reduce, "reduce", c(10L, 50L, 200L))
print_table(res_red)
