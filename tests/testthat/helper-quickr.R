skip_if_no_quickr <- function() {
  testthat::skip_if_not_installed("quickr")
  if (!("quickr" %in% globals$active_backends)) {
    testthat::skip("quickr backend is not active")
  }
  if (nzchar(Sys.getenv("ANVL_SKIP_QUICKR", ""))) {
    testthat::skip("ANVL_SKIP_QUICKR is set")
  }
}

skip_if_no_quickr_or_pjrt <- function() {
  skip_if_no_quickr()
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")
}

quickr_eval_graph_pjrt <- function(graph, ...) {
  # PJRT graphs accept only the traced array leaves; static args are not inputs.
  args_flat <- flatten(list(...))
  is_static_flat <- graph$is_static_flat
  if (!is.null(is_static_flat)) {
    stopifnot(length(args_flat) == length(is_static_flat))
    args_flat <- args_flat[!is_static_flat]
  }
  # lintr doesn't see helpers across files; use get() to avoid "no visible binding".
  do.call(get("eval_graph_pjrt", mode = "function"), c(list(graph), args_flat))
}

expect_quickr_matches_pjrt <- function(graph, ..., tolerance = 1e-12) {
  f_r <- graph_to_quickr_r_function(graph)
  f_quick <- graph_to_quickr_function(graph, unwrap = TRUE)

  out_r <- f_r(...)
  out_quick <- f_quick(...)
  out_pjrt <- quickr_eval_graph_pjrt(graph, ...)

  testthat::expect_equal(out_r, out_pjrt, tolerance = tolerance)
  testthat::expect_equal(out_quick, out_pjrt, tolerance = tolerance)

  invisible(list(out_r = out_r, out_quick = out_quick, out_pjrt = out_pjrt))
}

expect_quickr_matches_pjrt_fn <- function(
  fn,
  templates,
  runs,
  tolerance = 1e-12
) {
  graph <- trace_fn(fn, templates)
  f_r <- graph_to_quickr_r_function(graph)
  f_quick <- graph_to_quickr_function(graph, unwrap = TRUE)
  run_pjrt <- graph_runner_pjrt(graph) # nolint
  arg_names <- names(templates)

  for (run in runs) {
    args <- run$args
    info <- run$info
    tol <- run$tolerance
    if (is.null(tol)) {
      tol <- tolerance
    }

    args <- args[arg_names]
    out_r <- do.call(f_r, unname(args))
    out_quick <- do.call(f_quick, unname(args))
    out_pjrt <- do.call(run_pjrt, unname(args))

    testthat::expect_equal(out_r, out_pjrt, tolerance = tol, info = info)
    testthat::expect_equal(out_quick, out_pjrt, tolerance = tol, info = info)
  }
}
