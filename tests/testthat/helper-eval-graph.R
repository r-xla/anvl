# Execute a compile_pjrt() result on pjrt AnvlArray inputs (test-only stand-in
# for the retired jit_call_pjrt(); production execution goes through pjrt's
# native dispatcher).
exec_compiled_pjrt_for_test <- function(compiled, args_nv) {
  args_unwrapped <- lapply(args_nv, function(a) {
    pjrt::copy_buffer(a$data, compiled$device)
  })
  phantom_bufs <- lapply(compiled$phantom_specs, function(spec) {
    pjrt::pjrt_empty(dtype = spec$dtype, shape = spec$shape, device = compiled$device)
  })
  out_vals <- rlang::exec(
    pjrt::pjrt_execute,
    compiled$exec,
    !!!compiled$const_arrays,
    !!!args_unwrapped,
    !!!phantom_bufs,
    simplify = FALSE
  )
  unflatten(compiled$out_tree, lapply(out_vals, nv_array, backend = "pjrt"))
}

graph_runner_pjrt <- function(graph) {
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  compiled <- compile_graph_pjrt(graph, device = "cpu")
  input_nodes <- graph$inputs

  flatten_args_for_test <- function(x) {
    if (is.list(x)) {
      if (!length(x)) {
        return(list())
      }
      Reduce(c, lapply(unname(x), flatten_args_for_test))
    } else {
      list(x)
    }
  }

  as_r <- function(x) {
    if (inherits(x, "AnvlArray")) {
      return(as_array(x))
    }
    if (is.list(x)) {
      return(lapply(x, as_r))
    }
    x
  }

  function(...) {
    args <- flatten_args_for_test(list(...))
    if (length(args) != length(input_nodes)) {
      cli::cli_abort("Expected {length(input_nodes)} inputs, got {length(args)}")
    }

    args_nv <- Map(
      function(x, gval) {
        if (inherits(x, "AnvlArray")) {
          return(x)
        }
        expected_shape <- shape(gval$aval)
        expected_dtype <- as.character(gval$aval$dtype)
        if (expected_dtype == "i1") {
          expected_dtype <- "pred"
        }
        if (!length(expected_shape)) {
          if (length(x) != 1L) {
            cli::cli_abort("Expected scalar input")
          }
          nv_scalar(x, dtype = expected_dtype, backend = "pjrt")
        } else {
          nv_array(x, dtype = expected_dtype, shape = expected_shape, backend = "pjrt")
        }
      },
      args,
      input_nodes
    )

    out_nv <- exec_compiled_pjrt_for_test(compiled, args_nv)
    as_r(out_nv)
  }
}

eval_graph_pjrt <- function(graph, ...) {
  run <- graph_runner_pjrt(graph)
  run(...)
}
