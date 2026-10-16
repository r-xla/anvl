# Jitting is interpreted directly as a transformation and not as a higher order
# primitive because this is simpler.

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
#' This is a mutable class. Every graph is lowered against an environment of
#' its own: a sub-graph reads nothing of the graph around it except through
#' its inputs.
#' @param gval_to_fval (`hashtab`)\cr
#'   Mapping from graph values to func values.
#' @return (`HloEnv`)
#' @keywords internal
HloEnv <- function(gval_to_fval = NULL) {
  # Use an environment for reference semantics (mutable)
  env <- new.env(parent = emptyenv())
  env$gval_to_fval <- gval_to_fval %||% hashtab()

  structure(env, class = "HloEnv")
}

env_add <- function(env, gval, fval) {
  env$gval_to_fval[[gval]] <- fval
  invisible(env)
}

env_get <- function(env, gval) {
  env$gval_to_fval[[gval]] %||% cli_abort("GraphValue not found in environment")
}

#' @title Lower a Graph to StableHLO
#' @description
#' Converts a traced [`AnvlGraph`] into the StableHLO intermediate representation (IR),
#' building a [`stablehlo::Func`] with the [stablehlo](https://r-xla.github.io/stablehlo/) package.
#' Each statement of the graph is translated to its corresponding StableHLO op. The result can
#' be serialized to MLIR text via [`stablehlo::repr()`] and subsequently compiled to an
#' XLA executable with [`pjrt::pjrt_compile()`].
#'
#' The rule for translating a primitive to stablehlo is `prim_<name>[["stablehlo"]]`.
#'
#' The arguments of the resulting function are, in this order: the graph's
#' constants, the graph's inputs, the values of the graph's side effects that
#' come from the host (such as the global random state, see [nv_set_seed()]),
#' and the phantom donated inputs (when `donate_unaliased_outputs = TRUE`), one
#' per output that is not already aliased to a donated input, in the order of
#' the outputs. A side effect whose value is created in the program, such as the
#' token that orders prints, is built inside it. Its results are the graph's
#' outputs, then the values of the side effects the host receives; any other
#' side effect's updated value is not returned.
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
#' @param captured (`list(FuncValue)`)\cr
#'   Values of the enclosing function to bind the graph's last `length(captured)`
#'   inputs to. The lowered function reads them the way an MLIR region captures
#'   a value from above, instead of declaring them as inputs. Used for the
#'   sub-graphs of higher-order primitives (e.g. `nv_while`), whose trailing
#'   inputs are the values they close over. The graph's constants always become
#'   inputs, and `GraphLiteral`s are always inlined.
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#'   Donated buffers can be aliased with outputs of the same type, enabling in-place
#'   operations.
#' @param donate_unaliased_outputs (`logical(1)`)\cr
#'   If `TRUE` and the current target platform is `"cpu"`, append a
#'   phantom donated input for every output that isn't already aliased
#'   to a user-`donate`d input. They come after all other arguments.
#'   This is needed internally so R keeps track of the CPU buffers memory in order
#'   to know when to garbage collect.
#' @param platform (`NULL` | `character(1)`)\cr
#'   Target platform name (e.g. `"cpu"`, `"cuda"`). Stored on a process-wide
#'   global during the call so that platform-aware lowering rules (queried via
#'   [`current_platform()`]) can branch on it. `NULL` (the default)
#'   leaves the current value untouched — recursive calls from higher-order
#'   primitives inherit the platform of the enclosing call.
#' @return (`list`)\cr
#'   Of length 4:
#'   - the [`stablehlo::Func`]
#'   - The graph's constants: the [`GraphValue`]s holding [`ConcreteArray`]s,
#'     whose data must be passed as the leading inputs at execution time.
#'   - A list of phantom-output specs, one per phantom donated input
#'     appended when `donate_unaliased_outputs = TRUE`. Each entry is a
#'     `list(dtype, shape)` describing the buffer the executor must
#'     allocate. Empty when no phantoms were added.
#'   - The other [`stablehlo::Func`]s the function calls, which a program
#'     must define next to it: functions the lowering rules build once and
#'     call wherever they need them, e.g. the ThreeFry hash of
#'     [prim_sample_bits()]. Empty for most graphs, and always for a nested
#'     lowering (`id = ""`), whose calls the top-level lowering collects.
#' @seealso [`trace_fn()`], [`jit()`], [`current_platform()`]
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' # the closed-over array `x` becomes the constant input %0, before the
#' # graph's own input `y` (%1)
#' x <- nv_array(c(1, 2))
#' graph <- trace_fn(function(y) y + x, list(y = nv_aval("f32", shape = 2)))
#' out <- stablehlo(graph)
#' out[[1L]]
#' out[[2L]]
#'
#' # a donated input is aliased with an output of the same type
#' graph <- trace_fn(
#'   function(a, b) list(a + b, a * b),
#'   list(a = nv_aval("f32", 3), b = nv_aval("f32", 3))
#' )
#' stablehlo(graph, donate = "a")[[1L]]
#'
#' # on CPU, a phantom input is appended for every output not aliased yet, and
#' # the third element describes the buffers to allocate for them
#' out <- stablehlo(graph, donate_unaliased_outputs = TRUE, platform = "cpu")
#' out[[1L]]
#' out[[3L]]
stablehlo <- function(
  graph,
  id = "main",
  captured = list(),
  donate = character(),
  donate_unaliased_outputs = FALSE,
  platform = NULL
) {
  assert_string(id)
  if (!is.null(platform)) {
    local_platform(platform)
  }
  # GraphNode -> FuncValue
  env <- HloEnv()
  # A top-level lowering builds the module's `main` func (whose hlo_return
  # finalizes the module). A closure/region lowering (id = "", e.g. a scatter
  # update computation or a while body) builds an anonymous nested func
  # inside the enclosing build.
  # A top-level lowering collects the functions its rules build (see
  # `lowering_func()`).
  if (nzchar(id)) {
    local_lowering_funcs()
  }
  func <- stablehlo::local_func(id = id)
  if (any(vapply(graph$side_effects, function(s) is.null(s$input), logical(1L)))) {
    cli_abort("Internal error: a graph that closes over a side effect cannot be lowered on its own.")
  }
  side_effect_inputs <- lapply(side_effects_from_host(graph), function(name) graph$side_effects[[name]]$input)
  n_declared <- length(graph$inputs) - length(captured)
  inps <- c(graph$constants, graph$inputs[seq_len(n_declared)], side_effect_inputs)
  for (i in seq_along(captured)) {
    env_add(env, graph$inputs[[n_declared + i]], captured[[i]])
  }

  # Compute which inputs are donated (only graph$inputs, not constants)
  donate_flat <- if (length(donate) > 0L && !is.null(graph$in_tree)) {
    # Constants and side effects are never donated, inputs may be
    mask <- c(
      rep(FALSE, length(graph$constants)),
      pjrt::tree_leaf_mask(graph$in_tree, donate)
    )
    c(mask, rep(FALSE, max(0L, length(inps) - length(mask))))
  } else {
    rep(FALSE, length(inps))
  }

  out_types <- lapply(graph$outputs, \(out) at2vt(out$aval))
  aliased <- rep(FALSE, length(out_types))

  for (i in seq_along(inps)) {
    node <- inps[[i]]
    vt <- at2vt(node$aval)
    value_id <- stablehlo::ValueId()

    # A donated input is aliased with the first not yet aliased output of its type.
    alias <- NULL
    if (donate_flat[[i]]) {
      j <- match(TRUE, !aliased & vapply(out_types, \(out_vt) vt == out_vt, logical(1L)))
      if (!is.na(j)) {
        alias <- j - 1L # 0-based index for stablehlo
        aliased[[j]] <- TRUE
      }
    }

    fi <- stablehlo::FuncInput(value_id, vt, alias = alias)
    func$inputs <- stablehlo::FuncInputs(c(func$inputs, list(fi)))
    env_add(env, node, stablehlo::FuncValue(value_id, vt, func))
  }

  # For each output that isn't already aliased to a user-donated input,
  # append a phantom FuncInput that carries tf.aliasing_output = j. XLA
  # reuses the phantom parameter's storage for output j; the executor
  # allocates a fresh pjrt_empty() for each phantom so the output ends up
  # written into R-owned memory (gated on CPU because the trick only
  # gives us a host-visible RAWSXP on CPU). The phantom never appears in
  # the body — it exists purely as donated output storage.
  phantom_outputs <- if (donate_unaliased_outputs && identical(current_platform(), "cpu")) {
    which(!aliased)
  } else {
    integer()
  }
  for (j in phantom_outputs) {
    fi <- stablehlo::FuncInput(stablehlo::ValueId(), out_types[[j]], alias = j - 1L)
    func$inputs <- stablehlo::FuncInputs(c(func$inputs, list(fi)))
  }
  phantom_specs <- lapply(phantom_outputs, function(j) {
    out_aval <- graph$outputs[[j]]$aval
    list(dtype = out_aval$dtype, shape = shape(out_aval))
  })

  for (name in side_effects_built(graph)) {
    env_add(env, graph$side_effects[[name]]$input, side_effect(name)$build(func))
  }
  received <- lapply(side_effects_to_host(graph), function(name) graph$side_effects[[name]]$output)
  outputs <- lower_graph_calls(graph, env, func, c(graph$outputs, received))
  func <- do.call(hlo_return, outputs)

  list(func, graph$constants, phantom_specs, if (nzchar(id)) lowering_funcs() else list())
}

# Opens the registry of the functions the rules of a top-level lowering build
# once and call wherever they need them, for the calling scope.
local_lowering_funcs <- function(envir = parent.frame()) {
  old <- globals[["lowering_funcs"]]
  globals[["lowering_funcs"]] <- new.env(parent = emptyenv())
  withr::defer(
    {
      globals[["lowering_funcs"]] <- old
    },
    envir = envir
  )
}

# The functions of the current lowering's registry, sorted by id.
lowering_funcs <- function() {
  funcs <- globals[["lowering_funcs"]]
  lapply(sort(names(funcs)), function(id) funcs[[id]])
}

# The function `id` of the current lowering, which `build(id)` builds (and
# finalizes) on first use.
lowering_func <- function(id, build) {
  funcs <- globals[["lowering_funcs"]]
  if (is.null(funcs)) {
    cli_abort("Internal error: a lowering function is only available in a top-level lowering.")
  }
  if (is.null(funcs[[id]])) {
    funcs[[id]] <- build(id)
  }
  funcs[[id]]
}

# Lower `graph`'s calls into `func`, reading its inputs and constants from `env`
# and returning one FuncValue per node of `outputs`. stablehlo() uses it for a whole
# function; a rule that inlines a sub-graph into a region it builds by hand
# (prim_scan's loop body) first seeds `env` with the region's values for the
# sub-graph's inputs.
lower_graph_calls <- function(graph, env, func, outputs = graph$outputs) {
  gnode_to_fval <- function(gnode) {
    fval <- env_get(env, gnode)
    if (!identical(fval$func, func)) {
      FuncValue(fval$value_id, fval$value_type, func)
    } else {
      fval
    }
  }

  literal_to_fval <- function(literal) {
    hlo_tensor(
      value = unwrap_if_array(literal$aval$data),
      dtype = literal$aval$dtype,
      shape = shape(literal$aval),
      func = func
    )
  }

  do_call <- function(call) {
    prim <- call$primitive
    params <- call$params
    inputs <- lapply(call$inputs, \(x) {
      if (is_graph_literal(x)) {
        fval <- literal_to_fval(x)
        env_add(env, x, fval)
        fval
      } else {
        gnode_to_fval(x)
      }
    })
    rule <- prim[["stablehlo"]]
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

  for (call in graph$statements) {
    do_call(call)
  }

  lapply(outputs, \(x) {
    # an output is a literal only when the function returns one directly
    if (is_graph_literal(x)) literal_to_fval(x) else gnode_to_fval(x)
  })
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
#' @return (`NULL` | `character(1)`)\cr
#'   The current platform.
#'   `local_platform()` invisibly returns the previous platform.
#' @seealso [`stablehlo()`]
#' @examples
#' current_platform()
#' f <- function() {
#'   local_platform("cuda")
#'   current_platform()
#' }
#' f()
#' @export
current_platform <- function() {
  globals[["lowering_platform"]]
}

#' @rdname current_platform
#' @export
local_platform <- function(platform, envir = parent.frame()) {
  old <- globals[["lowering_platform"]]
  globals[["lowering_platform"]] <- platform
  withr::defer(
    {
      globals[["lowering_platform"]] <- old
    },
    envir = envir
  )
  invisible(old)
}
