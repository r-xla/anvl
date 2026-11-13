#' @include aaa.R
NULL

# Side effects: the generic mechanism, which a side effect plugs into with the
# S3 methods on its handle below. The side effects section of the internals
# article describes it. A second kind, a runtime side effect whose value the
# program computes and returns as an implicit output (pjrt's
# `implicit_outputs`), is planned; the mechanism once supported one
# (`git log -S side_effect_receive`).

# ---- What a side effect provides ----------------------------------------------

#' @title Side Effect Protocol
#' @description
#' The generics the handle of a side effect implements (see the side effects
#' section of the internals article).
#' @param effect (handle)\cr
#'   The handle of a side effect, e.g. `global_rng_state_handle`.
#' @param desc (`GraphDescriptor`)\cr
#'   The outermost trace.
#' @param args (`list()` | `NULL`)\cr
#'   The arguments of a call of a jitted function, `NULL` before the run.
#' @param object (any)\cr
#'   The host object, as `side_effect_host_object()` returns it.
#' @param name (`character(1)`)\cr
#'   The name of a value the program takes as an implicit input.
#' @param value (named `list` of `GraphBox`)\cr
#'   A value of the side effect in the current trace.
#' @param n (`numeric(1)`)\cr
#'   A number of uses.
#' @param ids (named `character()`)\cr
#'   The ids of the nodes of a value in a printed graph, e.g. `"%counter1"`.
#' @return
#' - `side_effect_start()`: the `SideEffectView` the outermost trace `desc`
#'   starts from: of fresh `GraphValue`s the program takes as implicit inputs,
#'   which are also its `inputs`; of boxes `desc` already has, such as an
#'   argument's leaves; or of values it creates by adding calls to `desc`.
#' - `side_effect_host_object()`: the object on the host the side effect lives
#'   in, or `NULL`.
#' - `side_effect_input()`: the `PJRTBuffer` the program takes as its implicit
#'   input `name`, which it reads from the host object `object` before every
#'   run.
#' - `side_effect_after()`: the value `n` uses after `value`, boxes of the
#'   current trace.
#' - `side_effect_fork()`: a value of its own, derived from `value` by adding
#'   calls to the current trace, for the body of a loop whose number of
#'   iterations is not known when it is traced.
#' - `side_effect_advance()`: `NULL`; after a run, advances `object` by `n`
#'   uses.
#' - `side_effect_format()`: the line a printed graph shows after its body for
#'   what the host does after the run, given `ids`, the ids of the nodes of the
#'   value the program starts from, by name, e.g. `"%counter1 += 2"`.
#' @noRd
side_effect_start <- function(effect, desc) {
  UseMethod("side_effect_start")
}

side_effect_host_object <- function(effect, args) {
  UseMethod("side_effect_host_object")
}

#' @export
side_effect_host_object.default <- function(effect, args) {
  NULL
}

side_effect_input <- function(effect, object, name) {
  UseMethod("side_effect_input")
}

side_effect_after <- function(effect, value, n) {
  UseMethod("side_effect_after")
}

side_effect_fork <- function(effect, value) {
  UseMethod("side_effect_fork")
}

#' @export
side_effect_fork.default <- function(effect, value) {
  cli_abort(c(
    "This side effect cannot be used in the body of {.fn nv_while}.",
    i = "Its uses are counted, which needs a loop whose number of iterations is known when it is traced."
  ))
}

side_effect_advance <- function(effect, object, n) {
  UseMethod("side_effect_advance")
}

side_effect_format <- function(effect, ids, n) {
  UseMethod("side_effect_format")
}

#' @export
side_effect_format.default <- function(effect, ids, n) {
  sprintf("%s: %s %s", paste(ids, collapse = ", "), format(n), if (n == 1) "use" else "uses")
}

# ---- Views --------------------------------------------------------------------

# The view of the side effect `effect` in one trace:
# - `value`: the value its count starts from in the trace, a named list of
#   GraphBoxes;
# - `n`: its number of uses since `value`;
# - `inputs`: the GraphValues, by name, the trace takes for it beyond its own
#   inputs, or `NULL`;
# - `outputs`: the GraphValues, by name, a loop body returns for it beyond its
#   own outputs, the value its next iteration starts from, which
#   `desc_close_side_effects()` sets; `NULL` in any other trace.
SideEffectView <- function(effect, value, n = 0, inputs = NULL) {
  x <- new.env(parent = emptyenv())
  x$effect <- effect
  x$value <- value
  x$n <- n
  x$inputs <- inputs
  x$outputs <- NULL
  structure(x, class = "SideEffectView")
}

# The view of the side effect `effect` among the views `views`, or `NULL`.
find_view <- function(views, effect) {
  Find(function(x) identical(x$effect, effect), views)
}

# The GraphValues of the field `field` (`"inputs"` or `"outputs"`) of the views
# `views`, one after the other.
views_values <- function(views, field) {
  unname(unlist(lapply(unname(views), `[[`, field), recursive = FALSE))
}

# The value the next use of the view `x` sees.
view_current <- function(x) {
  side_effect_after(x$effect, x$value, x$n)
}

# The view of the side effect `effect` in the trace `desc`, which it adds to
# `desc` on first use where `desc$side_effects_mode` says (see
# `GraphDescriptor()` for the modes).
desc_view <- function(desc, effect) {
  x <- find_view(desc$side_effects, effect)
  if (!is.null(x)) {
    return(x)
  }
  mode <- desc$side_effects_mode %||% ""
  x <- if (mode == "program") {
    side_effect_start(effect, desc)
  } else if (mode == "loop") {
    enclosing <- desc_view(desc$parent, effect)
    inputs <- lapply(enclosing$value, function(box) {
      GraphValue(AbstractArray(dtype = dtype(box$gnode), shape = shape(box$gnode)), label = box$gnode$label)
    })
    SideEffectView(effect, register_gvals(desc, inputs), inputs = inputs)
  } else if (mode == "capture") {
    enclosing <- desc_view(desc$parent, effect)
    SideEffectView(effect, enclosing$value, n = enclosing$n)
  } else {
    cli_abort(c(
      "A side effect cannot be used here.",
      i = paste(
        "A side effect, such as a random draw, can only be used where the trace can hand back what it did,",
        "so not in the condition of {.fn nv_while}, the function of a reduction, a scatter or a sort, or any",
        "other function traced without a {.arg side_effects_mode}."
      )
    ))
  }
  desc$side_effects <- c(desc$side_effects, list(x))
  x
}

# The value a use of the side effect `effect` in the trace `desc` takes, which
# it counts. `desc` must be the current trace.
desc_use_side_effect <- function(desc, effect) {
  x <- desc_view(desc, effect)
  value <- view_current(x)
  x$n <- x$n + 1
  value
}

# At the end of the trace `desc`, a loop body, its views record the value its
# next iteration starts from, which it returns. Any other trace hands back the
# counts its views hold.
desc_close_side_effects <- function(desc) {
  if (identical(desc$side_effects_mode, "loop")) {
    for (x in desc$side_effects) {
      x$outputs <- lapply(view_current(x), `[[`, "gnode")
    }
  }
  invisible(desc)
}

# Whether the tree `x` holds a random state, the one side effect the user holds.
holds_side_effect <- function(x) {
  any(vapply(flatten(x, is_leaf = is_rng_state), is_rng_state, logical(1L)))
}

# Reject trees containing side effects, and return `x` otherwise. `arg` is the
# caller's argument name. A loop carries side effects itself, so they cannot
# also be passed through its state; the default hint explains how to use
# random states in loops. Other callers can supply a more relevant hint.
abort_if_side_effects <- function(
  x,
  arg,
  hint = "To draw from it in the loop body, make it the default before the loop with {.fn local_nv_rng_state}.",
  call = rlang::caller_env()
) {
  if (holds_side_effect(x)) {
    cli_abort(
      c(
        "{.arg {arg}} must not contain a side effect, such as a random state.",
        i = hint
      ),
      call = call
    )
  }
  invisible(x)
}

# `x`, what a traced function returned, unless it holds a side effect: one is no
# value in a trace, which could be returned.
abort_if_side_effect_returned <- function(x) {
  if (holds_side_effect(x)) {
    cli_abort(
      c(
        "A traced function cannot return a side effect, such as a random state.",
        i = "Draw from it inside the function and return what it drew."
      ),
      call = NULL
    )
  }
  x
}

# ---- Loops ----------------------------------------------------------------------

# Makes the body `graph` of a loop call in the trace `desc` carry its side
# effects: their inputs and outputs become inputs and outputs of the body, at
# the positions after `at`. Returns the body's views, `steps` (the number of
# iterations, `NULL` for `nv_while()`) and the operands the call passes for
# them: the enclosing trace's current values, or a fork of a side effect the
# body of an `nv_while()` uses.
desc_begin_loop <- function(desc, graph, steps, at = length(graph$inputs)) {
  views <- graph$side_effects
  graph$inputs <- append(graph$inputs, views_values(views, "inputs"), after = at)
  graph$outputs <- append(graph$outputs, views_values(views, "outputs"), after = at)
  graph$side_effects <- list()
  operands <- lapply(views, function(body) {
    if (is.null(steps) && body$n > 0) {
      side_effect_fork(body$effect, desc_use_side_effect(desc, body$effect))
    } else {
      view_current(desc_view(desc, body$effect))
    }
  })
  list(views = views, steps = steps, operands = unname(unlist(operands, recursive = FALSE)))
}

# Fresh GraphValues like the inputs `loop` carries (see `desc_begin_loop()`),
# for the condition of an `nv_while()`, which takes them without using them.
side_effects_loop_twins <- function(loop) {
  lapply(views_values(loop$views, "inputs"), function(g) GraphValue(g$aval, g$label))
}

# After the loop call `loop` (see `desc_begin_loop()`) in the trace `desc`, which
# returned `out`: the enclosing trace's views continue after `steps` times the
# body's count, and the result is `out` without the side effects' values at the
# positions after `at`, whose nodes it labels like the inputs. The fork of an
# `nv_while()` was counted already.
desc_end_loop <- function(desc, loop, out, at = NULL) {
  inputs <- views_values(loop$views, "inputs")
  if (!length(inputs)) {
    return(out)
  }
  at <- at %||% (length(out) - length(inputs))
  positions <- at + seq_along(inputs)
  for (i in seq_along(inputs)) {
    node <- out[[positions[[i]]]]$gnode
    node$label <- node$label %||% inputs[[i]]$label
  }
  if (!is.null(loop$steps)) {
    for (body in loop$views) {
      x <- desc_view(desc, body$effect)
      x$n <- x$n + loop$steps * body$n
    }
  }
  out[-positions]
}

# ---- Branches and differentiated functions ------------------------------------

# After a construct whose sub-traces `graphs` closed over the enclosing trace's
# views -- the branches of `nv_if()`, the function `gradient()` differentiates
# -- the views of the trace `desc` continue after the larger count of the
# sub-traces, which started at their own. The sub-traces' views are dropped:
# what they counted is the enclosing trace's now.
desc_end_capture <- function(desc, graphs) {
  for (graph in graphs) {
    for (y in graph$side_effects) {
      x <- desc_view(desc, y$effect)
      x$n <- max(x$n, y$n)
    }
    graph$side_effects <- list()
  }
  invisible(desc)
}

# ---- The program and the host -------------------------------------------------

# The GraphValues the side effects of `graph` take beyond its inputs, one after
# the other; a program takes them after the graph's own inputs.
side_effects_inputs <- function(graph) {
  views_values(graph$side_effects, "inputs")
}

# The implicit inputs of the program of `graph` (see `compile_pjrt()`), one per
# value of `side_effects_inputs()`, in that order: before every run, its `get()`
# reads it from the side effect's host object (`side_effect_input()`).
side_effects_implicit_inputs <- function(graph) {
  unlist(
    lapply(unname(graph$side_effects), function(x) {
      Map(
        function(name, g) {
          list(
            get = side_effect_input_getter(x$effect, name),
            dtype = as.character(dtype(g$aval)),
            shape = shape(g$aval)
          )
        },
        names(x$inputs),
        x$inputs
      )
    }),
    recursive = FALSE
  )
}

# The `get()` of the implicit input `name` of the side effect `effect`, which
# keeps nothing of the trace.
side_effect_input_getter <- function(effect, name) {
  force(effect)
  force(name)
  function() side_effect_input(effect, side_effect_host_object(effect, NULL), name)
}

# The function that advances the host objects of the side effects of `graph`
# by their counts after a run of its program with the call's arguments `args`
# (pjrt's `on_success`), `NULL` when the graph uses none.
side_effects_on_success <- function(graph) {
  views <- unname(graph$side_effects)
  if (!length(views)) {
    return(NULL)
  }
  side_effects_settler(lapply(views, `[[`, "effect"), vapply(views, `[[`, numeric(1L), "n"))
}

# The `on_success` of `side_effects_on_success()` for the side effects `effects`
# and their counts `counts`: a function of its own, so that it keeps nothing of
# the trace. It finds all host objects before it advances any, and refuses one
# found twice: two views of it started from the same value.
side_effects_settler <- function(effects, counts) {
  force(effects)
  force(counts)
  function(outputs, args) {
    objects <- lapply(effects, side_effect_host_object, args = args)
    for (i in seq_along(objects)[-1L]) {
      object <- objects[[i]]
      if (!is.null(object) && any(vapply(objects[seq_len(i - 1L)], identical, logical(1L), object))) {
        cli_abort("The same {.cls {class(object)[[1L]]}} was passed to the jitted function twice.")
      }
    }
    for (i in seq_along(effects)) {
      if (counts[[i]] > 0) {
        side_effect_advance(effects[[i]], objects[[i]], counts[[i]])
      }
    }
    invisible(NULL)
  }
}
