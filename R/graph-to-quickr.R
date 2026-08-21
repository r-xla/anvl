#' @include rules-quickr.R
NULL

quickr_restore_leaf <- function(value, info) {
  shape <- as.integer(info$shape)

  if (length(shape) == 1L) {
    value_dims <- dim(value)
    if (is.null(value_dims) || !identical(value_dims, shape)) {
      value <- array(value, dim = shape)
    }
  }

  if (!is.null(info$backend)) {
    value <- nv_array(
      value,
      shape = shape,
      ambiguous = info$ambiguous,
      backend = info$backend
    )
  }

  value
}

quickr_restore_output_flat <- function(value, out_tree, out_infos) {
  if (tree_root_kind(out_tree) == "leaf" && length(out_infos) == 1L) {
    return(list(quickr_restore_leaf(value, out_infos[[1L]])))
  }

  if (!is.list(value) || length(value) != length(out_infos)) {
    cli_abort("Internal error: expected {length(out_infos)} quickr outputs, got {length(value)}")
  }

  .mapply(
    function(x, info) quickr_restore_leaf(x, info),
    list(value, out_infos),
    NULL
  )
}

quickr_restore_output <- function(value, out_tree, out_infos) {
  unflatten(out_tree, quickr_restore_output_flat(value, out_tree, out_infos))
}

graph_to_quickr_prepare <- function(graph) {
  in_tree <- graph$in_tree
  needs_flatten <- tree_root_kind(in_tree) == "list" &&
    any(tree_child_kinds(in_tree) == "list")

  needs_output_wrapper <- !(tree_root_kind(graph$out_tree) == "leaf" && length(graph$outputs) == 1L)
  out_infos <- lapply(graph$outputs, function(node) {
    list(
      dtype = as.character(dtype(node)),
      shape = shape(node),
      ambiguous = node$aval$ambiguous
    )
  })
  is_static_flat <- graph$is_static_flat
  has_static <- !is.null(is_static_flat) && isTRUE(any(is_static_flat))

  needs_wrapper <- isTRUE(needs_output_wrapper) ||
    length(graph$constants) ||
    isTRUE(needs_flatten) ||
    isTRUE(has_static)

  r_fun <- graph_to_quickr_r_fun_impl(graph, include_declare = TRUE)

  list(
    r_fun = r_fun,
    needs_flatten = needs_flatten,
    out_infos = out_infos,
    needs_wrapper = needs_wrapper
  )
}

quickr_assert_static_args <- function(args_flat, is_static_flat, static_args_flat) {
  if (is.null(is_static_flat) || !isTRUE(any(is_static_flat))) {
    return(invisible(NULL))
  }
  if (is.null(static_args_flat)) {
    cli_abort("This graph is missing traced static argument values. Retrace the graph before lowering to quickr.")
  }

  static_runtime <- args_flat[is_static_flat]
  if (length(static_runtime) != length(static_args_flat)) {
    cli_abort(
      "Internal error: expected {length(static_args_flat)} static args, got {length(static_runtime)}"
    )
  }

  mismatch <- vapply(
    seq_along(static_args_flat),
    function(i) !identical(static_runtime[[i]], static_args_flat[[i]]),
    logical(1L)
  )
  if (any(mismatch)) {
    cli_abort(
      "Static arguments must match the values used to trace the graph. Retrace the graph for new static values."
    )
  }

  invisible(NULL)
}

graph_to_quickr_make_wrapper <- function(
  graph,
  r_fun,
  inner_fun,
  out_infos,
  needs_flatten,
  flat = FALSE
) {
  r_arg_names <- formalArgs2(r_fun)
  n_user <- length(graph$inputs)
  leaf_arg_names <- r_arg_names[seq_len(n_user)]

  is_static_flat <- graph$is_static_flat
  has_static <- !is.null(is_static_flat) && isTRUE(any(is_static_flat))
  use_in_tree_formals <- isTRUE(needs_flatten) || isTRUE(has_static)

  const_args <- list()
  if (length(graph$constants)) {
    const_arg_names <- r_arg_names[(n_user + 1L):(n_user + length(graph$constants))]
    const_vals <- lapply(graph$constants, function(node) {
      as_array(node$aval$data)
    })
    const_args <- stats::setNames(const_vals, const_arg_names)
  }

  # Don't retain the full call frame (graph, r_fun, etc.) via parent environments.
  wrapper_env <- new.env(parent = environment(graph_to_quickr_function))
  wrapper_env$inner <- inner_fun
  wrapper_env$out_tree <- graph$out_tree
  wrapper_env$out_infos <- out_infos
  wrapper_env$leaf_arg_names <- leaf_arg_names
  wrapper_env$is_static_flat <- is_static_flat
  wrapper_env$static_args_flat <- graph$static_args_flat
  wrapper_env$const_args <- const_args
  wrapper_env$restore_output <- quickr_restore_output

  if (isTRUE(flat)) {
    # pjrt's dispatcher calls this one, and it passes the call's inputs only:
    # the dynamic leaves, in order. The statics are constants of this very
    # closure -- the dispatcher keyed on them with identical(), so reaching
    # here already proves they match -- and never arrive, so there is nothing
    # to check or drop.
    flat_wrapper <- function(args_flat) {}
    body(flat_wrapper) <- quote({
      args <- stats::setNames(args_flat, leaf_arg_names)
      value <- do.call(inner, c(const_args, args))
      restore_output(value, out_tree, out_infos)
    })
    environment(flat_wrapper) <- wrapper_env
    compiler::cmpfun(flat_wrapper)
    return(flat_wrapper)
  }

  wrapper <- function() {}
  if (isTRUE(use_in_tree_formals)) {
    in_tree <- graph$in_tree
    top_names <- tree_child_names(in_tree) %||%
      rep("", length(tree_child_sizes(in_tree)))
    top_names <- vapply(
      top_names,
      function(x) {
        if (is.null(x) || !nzchar(x)) "x" else x
      },
      character(1L)
    )
    top_names <- make.unique(make.names(top_names))
    formals(wrapper) <- as.pairlist(stats::setNames(rep(list(quote(expr = )), length(top_names)), top_names))
    wrapper_env$top_names <- top_names
  } else {
    formals(wrapper) <- formals(r_fun)[seq_len(n_user)]
    wrapper_env$top_names <- NULL
  }
  wrapper_env$use_in_tree_formals <- use_in_tree_formals

  body(wrapper) <- quote({
    if (isTRUE(use_in_tree_formals)) {
      args_top <- mget(top_names, envir = environment(), inherits = FALSE)
      args <- flatten(args_top)
      if (!is.null(is_static_flat)) {
        if (length(args) != length(is_static_flat)) {
          cli_abort("Expected {length(is_static_flat)} flattened inputs, got {length(args)}")
        }
        quickr_assert_static_args(args, is_static_flat, static_args_flat)
        args <- args[!is_static_flat]
      }
      args <- stats::setNames(args, leaf_arg_names)
    } else if (!length(leaf_arg_names)) {
      args <- list()
    } else {
      args <- mget(leaf_arg_names, envir = environment(), inherits = FALSE)
    }

    value <- do.call(inner, c(const_args, args))
    restore_output(value, out_tree, out_infos)
  })

  environment(wrapper) <- wrapper_env
  # bug in R: changing a function's environment disables byte-code compilation.
  compiler::cmpfun(wrapper)

  wrapper
}

#' Convert an AnvlGraph to a plain R function
#'
#' Lowers a supported subset of `AnvlGraph` objects to a plain R function (no
#' compilation) suitable for `quickr::quick()`. The returned function expects
#' plain R scalars/vectors/arrays and returns plain R values/arrays.
#'
#' Most users will prefer [`jit()`] with `backend = "quickr"`.
#' This function is the lower-level graph API.
#'
#' @param graph ([`AnvlGraph`])\cr
#'   Graph to convert.
#' @return (`function`)
#' @seealso [`jit()`] with `options(anvl.backend = "quickr")` for tracing and compiling a
#'   regular R function in one step.
#' @export
graph_to_quickr_r_function <- function(graph) {
  if (!is_graph(graph)) {
    cli_abort("{.arg graph} must be a {.cls AnvlGraph}")
  }

  prep <- graph_to_quickr_prepare(graph)
  r_fun <- prep$r_fun
  # The lowered function is intended to be runnable as plain R code as well as
  # compilable by {quickr}. When {quickr} is installed, its `declare()` can
  # modify arguments at runtime (e.g. stripping axes), which breaks plain R
  # execution. Keep the `declare(type(...))` call for compilation, but make it
  # a no-op when evaluating the lowered function in R.
  environment(r_fun)$declare <- function(...) invisible(NULL)

  if (!isTRUE(prep$needs_wrapper)) {
    return(r_fun)
  }

  graph_to_quickr_make_wrapper(
    graph = graph,
    r_fun = r_fun,
    inner_fun = r_fun,
    out_infos = prep$out_infos,
    needs_flatten = prep$needs_flatten
  )
}

#' Convert an AnvlGraph to a quickr-compiled function
#'
#' Lowers a supported subset of `AnvlGraph` objects to a plain R function and
#' compiles it with `quickr::quick()`.
#'
#' The returned function expects plain R scalars/vectors/arrays (not
#' [`AnvlArray`]) and returns [`AnvlArray`] outputs by default, or plain R
#' values when `unwrap = TRUE`.
#'
#' If the graph returns multiple outputs (e.g. a nested list), the compiled
#' function returns the same structure by rebuilding the output tree in R.
#'
#' For a list of supported primitives see `vignette("primitives")`.
#'
#' Supported dtypes are `f64`, `i32`, and `pred`.
#' The code generator currently supports arrays up to rank 5. Some primitives
#' are more restricted (e.g. `transpose` currently only handles rank-2 arrays).
#'
#' Most users will prefer [`jit()`] with `backend = "quickr"`. This function is
#' the lower-level graph API.
#'
#' @param graph ([`AnvlGraph`])\cr
#'   Graph to convert.
#' @param unwrap (`logical(1)`)\cr
#'   If `FALSE` (default), each output leaf is wrapped in an [`AnvlArray`]
#'   with the `"quickr"` backend. If `TRUE`, outputs are returned as plain
#'   R values.
#' @param flat (`logical(1)`)\cr
#'   If `FALSE` (default), the returned function takes structured top-level
#'   arguments matching the formals of the traced function. If `TRUE`, it
#'   takes a single flat list of all leaves (including static slots).
#' @return (`function`) that returns [`AnvlArray`] outputs (or a tree of
#'   them), or plain R values when `unwrap = TRUE`.
#' @seealso [`jit()`] with `backend = "quickr"` for tracing and compiling a
#'   regular R function in one step.
#' @keywords internal
graph_to_quickr_function <- function(graph, unwrap = FALSE, flat = FALSE) {
  if (!is_graph(graph)) {
    cli_abort("{.arg graph} must be a {.cls AnvlGraph}")
  }

  assert_quickr_installed("{.fn graph_to_quickr_function}")
  assert_flag(unwrap)
  assert_flag(flat)

  prep <- graph_to_quickr_prepare(graph)
  inner_quick <- quickr_eager_compile(prep$r_fun)

  out_infos <- prep$out_infos
  if (!isTRUE(unwrap)) {
    # Tag out_infos so that quickr_restore_leaf wraps each leaf as an AnvlArray.
    out_infos <- lapply(out_infos, function(info) {
      info$backend <- "quickr"
      info
    })
  }

  graph_to_quickr_make_wrapper(
    graph = graph,
    r_fun = prep$r_fun,
    inner_fun = inner_quick,
    out_infos = out_infos,
    needs_flatten = prep$needs_flatten,
    flat = flat
  )
}
