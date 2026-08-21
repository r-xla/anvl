# We interprete jitting directly as a transformation and not as a higher order primitive
# because this seems simpler for now.

# S3 methods for stablehlo functions to handle AnvlArray

#' @method hlo_scalar AnvlArray
#' @export
hlo_scalar.AnvlArray <- function(value, ..., func = NULL) {
  hlo_scalar(value$data, ..., func = func)
}

#' @method hlo_tensor AnvlArray
#' @export
hlo_tensor.AnvlArray <- function(value, ..., func = NULL) {
  hlo_tensor(value$data, ..., func = func)
}

#' @title HloEnv
#' @description
#' Environment for storing graph value to func value mappings.
#' This is a mutable class.
#' @param parent (`HloEnv` | `NULL`)\cr
#'   Parent environment for lookups.
#' @param gval_to_fval (`hashtab`)\cr
#'   Mapping from graph values to func values.
#' @return (`HloEnv`)
#' @keywords internal
HloEnv <- function(parent = NULL, gval_to_fval = NULL) {
  if (!is.null(parent) && !inherits(parent, "HloEnv")) {
    cli_abort("parent must be an HloEnv or NULL")
  }

  # Use an environment for reference semantics (mutable)
  env <- new.env(parent = emptyenv())
  env$parent <- parent
  env$gval_to_fval <- gval_to_fval %||% hashtab()

  structure(env, class = "HloEnv")
}

env_add <- function(env, gval, fval) {
  env$gval_to_fval[[gval]] <- fval
  invisible(env)
}

env_get <- function(env, gval) {
  fval <- env$gval_to_fval[[gval]]
  if (!is.null(fval)) {
    return(fval)
  }
  parent <- env$parent
  if (!is.null(parent)) {
    return(env_get(parent, gval))
  }
  cli_abort("GraphValue not found in environment")
}

#' @title Lower a graph to StableHLO
#' @description
#' Converts a traced [`AnvlGraph`] into the StableHLO intermediate representation (IR).
#' Each graph operation is translated to its corresponding StableHLO op. The result can
#' be serialized to MLIR text via `stablehlo::repr()` and subsequently compiled to an
#' XLA executable with `pjrt::pjrt_compile()`.
#'
#' The rules for translating to stablehlo are stored in `$rules[["stablehlo"]]` of the primitives.
#'
#' This is a low-level function; most users should use [`jit()`] instead.
#' @param graph ([`AnvlGraph`])\cr
#'   The graph to lower (e.g. produced by [`trace_fn()`]).
#' @param id (`character(1)`)\cr
#'   The id of the resulting StableHLO function. Use `"main"` (the default)
#'   for a top-level lowering (returning from the `main` function finalizes
#'   the module) and `""` for a closure/region lowering (e.g. a while body or
#'   a scatter update computation) that builds an anonymous nested function
#'   inside an enclosing build.
#' @param constants_as_inputs (`logical(1)`)\cr
#'   If `TRUE` (default), constants are registered as inputs to the StableHLO function
#'   so they can be passed in at execution time.
#'   If `FALSE`, they are not added as inputs. Set to `FALSE` for closures.
#'   Note that `GraphLiteral`s are always inlined into the StableHLO function.
#' @param env (`HloEnv` | `NULL`)\cr
#'   Optional environment for reusing variable mappings across nested function lowerings
#'   (e.g. for higher-order primitives like `nv_while`).
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#'   Donated buffers can be aliased with outputs of the same type, enabling in-place
#'   operations.
#' @param donate_unaliased_outputs (`logical(1)`)\cr
#'   If `TRUE` and the current target platform is `"cpu"`, append a
#'   phantom donated input for every output that isn't already aliased
#'   to a user-`donate`d input.
#'   This is needed internally so R keeps track of the CPU buffers memory in order
#'   to know when to garbage collect.
#' @param platform (`NULL` | `character(1)`)\cr
#'   Target platform name (e.g. `"cpu"`, `"cuda"`). Stored on a process-wide
#'   global during the call so that platform-aware lowering rules (queried via
#'   [`current_platform()`]) can branch on it. `NULL` (the default)
#'   leaves the current value untouched — recursive calls from higher-order
#'   primitives inherit the platform of the enclosing call.
#' @return A `list` of length 3:
#'   - the [`stablehlo::Func`]
#'   - The list of [`GraphValue`]s holding [`ConcreteArray`]s.
#'   - A list of phantom-output specs, one per phantom donated input
#'     appended when `donate_unaliased_outputs = TRUE`. Each entry is a
#'     `list(dtype, shape)` describing the buffer the executor must
#'     allocate. Empty when no phantoms were added.
#' @seealso [`trace_fn()`], [`jit()`], [`current_platform()`]
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2))
#' graph <- trace_fn(function(y) y + x, list(y = nv_aval("f32", shape = c())))
#' graph
#' stablehlo(graph)
stablehlo <- function(
  graph,
  id = "main",
  constants_as_inputs = TRUE,
  env = NULL,
  donate = character(),
  donate_unaliased_outputs = FALSE,
  platform = NULL
) {
  assert_string(id)
  if (!is.null(platform)) {
    local_platform(platform)
  }
  # Node -> FuncValue
  env <- HloEnv(parent = env)
  # A top-level lowering builds the module's `main` func (whose hlo_return
  # finalizes the module). A closure/region lowering (id = "", e.g. a scatter
  # update computation or a while body) builds an anonymous nested func
  # inside the enclosing build.
  func <- stablehlo::local_func(id = id)
  inps <- if (constants_as_inputs) c(graph$constants, graph$inputs) else graph$inputs

  gnode_to_fval <- function(gnode) {
    fval <- env_get(env, gnode)
    if (!identical(fval$func, func)) {
      FuncValue(fval$value_id, fval$value_type, func)
    } else {
      fval
    }
  }

  # Compute which inputs are donated (only graph$inputs, not constants)
  donate_flat <- if (length(donate) > 0L && !is.null(graph$in_tree)) {
    # Constants are never donated, inputs may be
    c(
      rep(FALSE, length(graph$constants)),
      tree_leaf_mask(graph$in_tree, donate)
    )
  } else {
    rep(FALSE, length(inps))
  }

  # Get output types for aliasing
  out_types <- lapply(graph$outputs, function(out) {
    at2vt(out$aval)
  })

  # Track which outputs have been aliased (0-based indices)
  aliased_outputs <- integer()

  for (i in seq_along(inps)) {
    node <- inps[[i]]
    vt <- at2vt(node$aval)
    id <- stablehlo::ValueId()

    # Check if this input is donated and find a matching output
    alias <- NULL
    if (donate_flat[[i]]) {
      # Find an output with matching type that hasn't been aliased yet
      for (j in seq_along(out_types)) {
        if ((j - 1L) %in% aliased_outputs) {
          next
        }
        out_vt <- out_types[[j]]
        if (vt == out_vt) {
          alias <- j - 1L # 0-based index for stablehlo
          aliased_outputs <- c(aliased_outputs, alias)
          break
        }
      }
    }

    fi <- stablehlo::FuncInput(id, vt, alias = alias)
    func$inputs <- stablehlo::FuncInputs(c(func$inputs, list(fi)))
    fval <- stablehlo::FuncValue(id, vt, func)
    env_add(env, node, fval)
  }

  # For each output that isn't already aliased to a user-donated input,
  # append a phantom FuncInput that carries tf.aliasing_output = j. XLA
  # reuses the phantom parameter's storage for output j; the executor
  # allocates a fresh pjrt_empty() for each phantom so the output ends up
  # written into R-owned memory (gated on CPU because the trick only
  # gives us a host-visible RAWSXP on CPU). The phantom never appears in
  # the body — it exists purely as donated output storage.
  phantom_specs <- list()
  if (donate_unaliased_outputs && identical(current_platform(), "cpu")) {
    for (j in seq_along(out_types)) {
      if ((j - 1L) %in% aliased_outputs) {
        next
      }
      out_vt <- out_types[[j]]
      id <- stablehlo::ValueId()
      fi <- stablehlo::FuncInput(id, out_vt, alias = j - 1L)
      func$inputs <- stablehlo::FuncInputs(c(func$inputs, list(fi)))
      aliased_outputs <- c(aliased_outputs, j - 1L)
      out_aval <- graph$outputs[[j]]$aval
      phantom_specs[[length(phantom_specs) + 1L]] <- list(
        dtype = out_aval$dtype,
        shape = out_aval$shape$dims
      )
    }
  }

  if (!constants_as_inputs) {
    for (const in graph$constants) {
      if (is.null(env_get(env, const))) {
        cli_abort("Internal error: constant not found in environment")
      }
    }
  }

  do_call <- function(call) {
    prim <- call$primitive
    params <- call$params
    inputs <- lapply(call$inputs, \(x) {
      if (is_graph_literal(x)) {
        # need to add a literal to the program
        fval <- hlo_tensor(
          value = unwrap_if_array(x$aval$data),
          dtype = x$aval$dtype,
          shape = x$aval$shape$dims,
          func = func
        )
        env_add(env, x, fval)
        fval
      } else {
        gnode_to_fval(x)
      }
    })
    rule <- prim[["stablehlo"]]
    if (is_higher_order_primitive(prim)) {
      params <- c(params, list(.env = env))
    }
    # Forward this call's known output types (already inferred at trace time) to
    # rules that opt in by declaring an `output_types` parameter, letting them
    # pass the types to their hlo_* builder and skip stablehlo's re-inference.
    if ("output_types" %in% names(formals(rule))) {
      params <- c(
        params,
        list(output_types = lapply(call$outputs, function(o) at2vt(o$aval)))
      )
    }
    fvals_out <- rlang::exec(rule, !!!c(inputs, params))
    if (length(call$outputs) != length(fvals_out)) {
      cli_abort("Expected {length(call$outputs)} outputs, but got {length(fvals_out)}")
    }
    for (i in seq_along(fvals_out)) {
      env_add(env, call$outputs[[i]], fvals_out[[i]])
    }
  }

  for (call in graph$calls) {
    do_call(call)
  }

  outputs <- lapply(graph$outputs, \(x) {
    if (is_graph_literal(x)) {
      # this only happens when a literal is directly returned
      hlo_tensor(value = x$aval$data, dtype = x$aval$dtype, shape = x$aval$shape$dims, func = func)
    } else {
      gnode_to_fval(x)
    }
  })
  func <- do.call(hlo_return, outputs)

  constants <- graph$constants

  list(func, constants, phantom_specs)
}

#' @title Current Lowering Target Platform
#' @description
#' Returns the target platform currently set by an enclosing [`stablehlo()`]
#' call (e.g. `"cpu"`, `"cuda"`). Platform-aware lowering rules call this to
#' branch on the target — e.g. SVD switches to a layout-flip variant when
#' targeting CUDA with `m < n` because cuSOLVER's `gesvd` requires `m >= n`.
#' Returns `NULL` outside of a lowering call.
#'
#' [`local_platform()`] sets the current platform for the duration of the
#' calling scope, restoring the previous value via [`withr::defer()`] when the
#' scope exits. Useful in tests and for manually exercising platform-aware
#' lowering rules outside of a [`stablehlo()`] call.
#' @param platform (`character(1)` | `NULL`)\cr
#'   Target platform name (e.g. `"cpu"`, `"cuda"`), or `NULL` to clear it.
#' @param envir (`environment`)\cr
#'   Environment whose exit triggers restoration of the previous platform.
#' @return `current_platform()` returns `NULL` or `character(1)`.
#'   `local_platform()` invisibly returns the previous platform.
#' @seealso [`stablehlo()`]
#' @export
current_platform <- function() {
  globals[["LOWERING_PLATFORM"]]
}

#' @rdname current_platform
#' @export
local_platform <- function(platform, envir = parent.frame()) {
  old <- globals[["LOWERING_PLATFORM"]]
  globals[["LOWERING_PLATFORM"]] <- platform
  withr::defer(
    {
      globals[["LOWERING_PLATFORM"]] <- old
    },
    envir = envir
  )
  invisible(old)
}
