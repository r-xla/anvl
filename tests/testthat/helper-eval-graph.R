compile_graph_pjrt <- function(graph) {
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  compiled <- compile_graph_xla(graph, device = "cpu")
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
        expected_shape <- gval$aval$shape$dims
        expected_dtype <- as.character(gval$aval$dtype)
        if (expected_dtype == "i1") {
          expected_dtype <- "pred"
        }
        if (!length(expected_shape)) {
          if (length(x) != 1L) {
            cli::cli_abort("Expected scalar input")
          }
          nv_scalar(x, dtype = expected_dtype, backend = "xla")
        } else {
          nv_array(x, dtype = expected_dtype, shape = expected_shape, backend = "xla")
        }
      },
      args,
      input_nodes
    )

    out_nv <- jit_call_xla(
      compiled$exec,
      compiled$out_tree,
      compiled$const_arrays,
      args_nv,
      rep(FALSE, length(args_nv)),
      compiled$ambiguous_out,
      device = compiled$device,
      phantom_specs = compiled$phantom_specs
    )
    as_r(out_nv)
  }
}

eval_graph_pjrt <- function(graph, ...) {
  run <- compile_graph_pjrt(graph)
  run(...)
}
