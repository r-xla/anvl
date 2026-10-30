#' @include aaa.R
NULL

# Side effects
#
# A side effect is state outside of a traced function that some primitives read
# and update, such as the random state the samplers draw from. It has up to
# three parts:
# - its host object: the state between calls, which lives on the host. A side
#   effect that starts afresh in every program has none.
# - its handle: its identity, which holds no value. A use passes the handle to
#   find its trace's view, and a loop or a branch to match a sub-trace's view
#   to the enclosing trace's. Its class says which side effect it is, and
#   inherits from `TraceTimeSideEffect` or `RuntimeSideEffect`, which says its
#   kind.
# - one view per trace that uses it (`SideEffectView()`), in `desc$side_effects`
#   in the order of first use: its value in that trace.
#
# There are two kinds:
# - A trace-time side effect is changed by its uses in a way the trace knows: n
#   uses take it from `value` to `side_effect_at(effect, value, n)`. A view
#   counts its uses, and what a trace hands back is that count. After a run,
#   the host advances the host object by it (`side_effect_advance()`), so that
#   no call waits for the program of the call before it.
# - A runtime side effect is changed by what the program computes: a use takes
#   its current value and sets the next one. What a trace hands back is that
#   value, which the host receives after a run (`side_effect_receive()`).
#
# Where a view starts is decided by the trace's `side_effects_mode` (see
# `GraphDescriptor()` and `desc_side_effect()`). Inside a loop, both kinds carry
# their value from one iteration to the next. Out of a loop, a branch, a
# differentiated function or the program, a runtime side effect hands back its
# value and a trace-time one its count: `steps` times the body's count for
# `nv_scan()`, the larger of the branches' counts for `nv_if()`. The number of
# iterations of `nv_while()` is not known when it is traced, so the body of one
# starts from a fork of a trace-time side effect (`side_effect_fork()`), which
# is one use of the enclosing trace. The side effects article describes the
# mechanism.
#
# A function that modifies a trace in place starts with `desc_` and takes its
# `GraphDescriptor` as the first argument, so that a call says which trace it
# changes. `side_effect_at()` and `side_effect_fork()` add calls to the current
# trace, so they are only called while the trace of their value is the current
# one.

# ---- What a side effect provides ----------------------------------------------

#' @title Side Effect Protocol
#' @description
#' The generics the handle of a side effect implements (see the comment at the
#' top of R/side-effects.R).
#' @param effect (handle)\cr
#'   The handle of a side effect, e.g. `global_random_state`.
#' @param desc (`GraphDescriptor`)\cr
#'   The outermost trace.
#' @param args (`list()` | `NULL`)\cr
#'   The arguments of a call of a jitted function, `NULL` before the run.
#' @param value (named `list` of `GraphBox`)\cr
#'   A value of the side effect in the current trace.
#' @param n (`numeric(1)`)\cr
#'   A number of uses.
#' @param object (any)\cr
#'   The host object, as `side_effect_host_object()` returns it.
#' @return
#' - `side_effect_start()`: the `SideEffectView` the outermost trace `desc`
#'   starts from: of fresh `GraphValue`s the program takes as implicit inputs,
#'   which are also its `inputs` and are named after the fields of the host
#'   object; of boxes `desc` already has, such as an argument's leaves; or of
#'   values it creates by adding calls to `desc`.
#' - `side_effect_host_object()`: the object on the host the side effect lives
#'   in, or `NULL`. Before a run, its field `name` is read for the implicit
#'   input `name`.
#' - `side_effect_at()`: the value `n` uses after `value`, boxes of the current
#'   trace (trace-time side effects).
#' - `side_effect_fork()`: a value of its own, derived from `value` by adding
#'   calls to the current trace, for the body of a loop whose number of
#'   iterations is not known when it is traced (trace-time side effects).
#' - `side_effect_advance()`: `NULL`; after a run, advances `object` by `n` uses
#'   (trace-time side effects).
#' - `side_effect_receive()`: `NULL`; after a run, takes the program's last
#'   value `value`, a named list of `AnvlArray`s (runtime side effects).
#' - `side_effect_format()`: the line a printed graph shows after its body for
#'   what the host does after the run, given `ids`, the ids of the nodes of the
#'   value the program starts from, by name, e.g. `"%counter1 += 2"`
#'   (trace-time side effects).
#' @param ids (named `character()`)\cr
#'   The ids of the nodes of a value in a printed graph, e.g. `"%counter1"`.
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

side_effect_at <- function(effect, value, n) {
  UseMethod("side_effect_at")
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

side_effect_receive <- function(effect, object, value) {
  UseMethod("side_effect_receive")
}

side_effect_format <- function(effect, ids, n) {
  UseMethod("side_effect_format")
}

#' @export
side_effect_format.default <- function(effect, ids, n) {
  sprintf("%s: %s %s", paste(ids, collapse = ", "), format(n), if (n == 1) "use" else "uses")
}

is_trace_time <- function(effect) {
  inherits(effect, "TraceTimeSideEffect")
}

is_side_effect <- function(x) {
  inherits(x, c("TraceTimeSideEffect", "RuntimeSideEffect"))
}

# ---- Views --------------------------------------------------------------------

# The view of the side effect `effect` in one trace:
# - `value`: its value in the trace, a named list of GraphBoxes: for a runtime
#   side effect the current one, for a trace-time one the one its count starts
#   from;
# - `start`: `value` when the view was created;
# - `n`: a trace-time side effect's number of uses since `value`;
# - `inputs`: the GraphValues, by name, the trace takes for it beyond its own
#   inputs, or `NULL`;
# - `outputs`: the GraphValues, by name, the trace returns for it beyond its own
#   outputs, or `NULL`, which `desc_close_side_effects()` sets.
SideEffectView <- function(effect, value, n = 0, inputs = NULL) {
  x <- new.env(parent = emptyenv())
  x$effect <- effect
  x$value <- value
  x$start <- value
  x$n <- n
  x$inputs <- inputs
  x$outputs <- NULL
  structure(x, class = "SideEffectView")
}

# A copy of the view `x`, of another trace or graph, with the value `value` and
# the outputs `outputs`.
copy_view <- function(x, value = x$value, outputs = x$outputs) {
  copy <- SideEffectView(x$effect, value, n = x$n, inputs = x$inputs)
  copy$outputs <- outputs
  copy
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
side_effect_current <- function(x) {
  if (is_trace_time(x$effect)) side_effect_at(x$effect, x$value, x$n) else x$value
}

# The view of the side effect `effect` in the trace `desc`, which it adds to
# `desc` on first use as `desc$side_effects_mode` says (see `GraphDescriptor()`):
# - "program", the outermost trace: where the side effect says
#   (`side_effect_start()`);
# - "loop", a loop body: fresh inputs like the enclosing trace's value, which
#   the loop carries;
# - "capture", an `nv_if()` branch or the function `gradient()` differentiates:
#   the enclosing trace's value and count, which it continues;
# - anything else: an error, as the trace cannot hand back what it did.
desc_side_effect <- function(desc, effect) {
  x <- find_view(desc$side_effects, effect)
  if (!is.null(x)) {
    return(x)
  }
  mode <- desc$side_effects_mode %||% ""
  x <- if (mode == "program") {
    side_effect_start(effect, desc)
  } else if (mode == "loop") {
    enclosing <- desc_side_effect(desc$parent, effect)
    inputs <- lapply(enclosing$value, function(box) {
      GraphValue(AbstractArray(dtype = dtype(box$gnode), shape = shape(box$gnode)), label = box$gnode$label)
    })
    SideEffectView(effect, register_gvals(desc, inputs), inputs = inputs)
  } else if (mode == "capture") {
    enclosing <- desc_side_effect(desc$parent, effect)
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

# The value a use of the trace-time side effect `effect` in the trace `desc`
# takes, which it counts. `desc` must be the current trace.
desc_use_side_effect <- function(desc, effect) {
  x <- desc_side_effect(desc, effect)
  value <- side_effect_current(x)
  x$n <- x$n + 1
  value
}

# At the end of the trace `desc`, its views record what it hands back for them:
# a loop body the value its next iteration starts from, and any other trace the
# value of a runtime side effect it changed. A trace-time side effect hands back
# its count, which its view holds.
desc_close_side_effects <- function(desc) {
  loop <- identical(desc$side_effects_mode, "loop")
  for (x in desc$side_effects) {
    x$outputs <- if (loop) {
      gnodes_of(side_effect_current(x))
    } else if (!is_trace_time(x$effect) && !identical(gnodes_of(x$value), gnodes_of(x$start))) {
      gnodes_of(x$value)
    }
  }
  invisible(desc)
}

# The nodes of the boxes `boxes`, by name.
gnodes_of <- function(boxes) {
  lapply(boxes, `[[`, "gnode")
}

# `x`, a tree, unless one of its leaves is the handle of a side effect: a loop
# carries side effects itself, so they cannot also be passed through its state.
# `arg` names the state.
abort_if_side_effects <- function(x, arg, call = rlang::caller_env()) {
  if (any(vapply(flatten(x), is_side_effect, logical(1L)))) {
    cli_abort(
      c(
        "{.arg {arg}} must not contain a side effect, such as a random state.",
        i = "To draw from it in the loop body, make it the default before the loop with {.fn local_nv_rng_state}."
      ),
      call = call
    )
  }
  invisible(x)
}

# ---- Loops ----------------------------------------------------------------------

# Makes the body `graph` of a loop call in the trace `desc` carry its side
# effects: their inputs and outputs become inputs and outputs of the body, at
# the positions after `at`. Returns the body's views, their inputs, `steps` (the
# number of iterations, `NULL` for `nv_while()`) and the operands the call passes
# for them: the enclosing trace's current values, or a fork of a trace-time side
# effect the body of an `nv_while()` uses.
desc_begin_loop <- function(desc, graph, steps, at = length(graph$inputs)) {
  views <- graph$side_effects
  inputs <- lapply(views, `[[`, "inputs")
  graph$inputs <- append(graph$inputs, views_values(views, "inputs"), after = at)
  graph$outputs <- append(graph$outputs, views_values(views, "outputs"), after = at)
  graph$side_effects <- list()
  operands <- lapply(views, function(body) {
    if (is_trace_time(body$effect) && is.null(steps) && body$n > 0) {
      side_effect_fork(body$effect, desc_use_side_effect(desc, body$effect))
    } else {
      side_effect_current(desc_side_effect(desc, body$effect))
    }
  })
  list(views = views, inputs = inputs, steps = steps, operands = unname(unlist(operands, recursive = FALSE)))
}

# Fresh GraphValues like the inputs `loop` carries (see `desc_begin_loop()`),
# for the condition of an `nv_while()`, which takes them without using them.
side_effects_loop_twins <- function(loop) {
  lapply(views_values(loop$views, "inputs"), function(g) GraphValue(g$aval, g$label))
}

# After the loop call `loop` (see `desc_begin_loop()`) in the trace `desc`, which
# returned `out`: the enclosing trace's views continue after the loop, and the
# result is `out` without the side effects' values at the positions after `at`,
# whose nodes it labels like the inputs. A runtime side effect continues from
# the value the loop returned for it. A trace-time side effect continues after
# `steps` times the body's count; the fork of an `nv_while()` was counted
# already.
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
  results <- split_side_effect_values(out[positions], lapply(loop$inputs, names))
  for (i in seq_along(loop$views)) {
    body <- loop$views[[i]]
    x <- desc_side_effect(desc, body$effect)
    if (!is_trace_time(body$effect)) {
      x$value <- results[[i]]
    } else if (!is.null(loop$steps)) {
      x$n <- x$n + loop$steps * body$n
    }
  }
  out[-positions]
}

# `values`, the values of several side effects one after the other, split into
# one list per side effect, named by the element of `names` for it.
split_side_effect_values <- function(values, names) {
  ends <- cumsum(lengths(names))
  Map(function(end, nms) stats::setNames(values[end - length(nms) + seq_along(nms)], nms), ends, names)
}

# ---- Branches and differentiated functions ------------------------------------

# Readies the branches `graphs` of an `nv_if()` call in the trace `desc`, which
# closed over the enclosing trace's views. Every branch returns, after its own
# outputs, the value of each runtime side effect any branch changed: its own,
# or the enclosing trace's value, which a branch that did not change it closed
# over. Returns the branches' views, one list per branch, and those runtime side
# effects, with the names of their values.
desc_begin_branches <- function(desc, graphs) {
  views <- lapply(graphs, `[[`, "side_effects")
  changed <- Filter(function(y) !is_trace_time(y$effect) && !is.null(y$outputs), unlist(views, recursive = FALSE))
  effects <- list()
  for (y in changed) {
    if (!any(vapply(effects, identical, logical(1L), y$effect))) {
      effects <- c(effects, list(y$effect))
    }
  }
  for (i in seq_along(graphs)) {
    graph <- graphs[[i]]
    for (effect in effects) {
      y <- find_view(views[[i]], effect)
      outputs <- y$outputs %||% gnodes_of(desc_side_effect(desc, effect)$value)
      # a value the branch did not use is not a node of its graph yet, so it
      # becomes a capture, which `purify_subgraphs()` makes an input
      for (g in outputs) {
        if (!is_graph_literal(g) && !graph_has_node(graph, g)) {
          graph$constants <- c(graph$constants, list(g))
        }
      }
      graph$outputs <- c(graph$outputs, unname(outputs))
    }
    graph$side_effects <- list()
  }
  names <- lapply(effects, function(effect) names(desc_side_effect(desc, effect)$value))
  list(views = views, effects = effects, names = names)
}

# Whether the node `g` is an input, a constant or the output of a call of
# `graph`.
graph_has_node <- function(graph, g) {
  is_g <- function(node) identical(node, g)
  any(vapply(graph$inputs, is_g, logical(1L))) ||
    any(vapply(graph$constants, is_g, logical(1L))) ||
    any(vapply(graph$statements, function(call) any(vapply(call$outputs, is_g, logical(1L))), logical(1L)))
}

# After the `nv_if()` call of the branches `branches` (see
# `desc_begin_branches()`) in the trace `desc`, which returned `out`: the
# enclosing trace's views continue after it (see `desc_end_capture()`), and the
# result is `out` without the side effects' values.
desc_end_branches <- function(desc, branches, out) {
  n <- sum(lengths(branches$names))
  if (!n) {
    desc_end_capture(desc, branches$views, list())
    return(out)
  }
  positions <- length(out) - n + seq_len(n)
  values <- split_side_effect_values(out[positions], branches$names)
  results <- Map(function(effect, value) list(effect = effect, value = value), branches$effects, values)
  desc_end_capture(desc, branches$views, results)
  out[-positions]
}

# After a construct whose sub-traces closed over the enclosing trace's views --
# the branches of `nv_if()`, the function `gradient()` differentiates -- the
# views of the trace `desc` continue: a trace-time side effect after the larger
# count of the sub-traces, which started at its own, and a runtime side effect
# from the value the construct returned for it. `views` holds the views of each
# sub-trace, and `results` one `list(effect, value)` per runtime side effect the
# construct returned a value of.
desc_end_capture <- function(desc, views, results) {
  for (sub in views) {
    for (y in sub) {
      if (is_trace_time(y$effect)) {
        x <- desc_side_effect(desc, y$effect)
        x$n <- max(x$n, y$n)
      }
    }
  }
  for (result in results) {
    x <- desc_side_effect(desc, result$effect)
    x$value <- result$value
  }
  invisible(desc)
}

# ---- The program and the host -------------------------------------------------

# The GraphValues the side effects of `graph` take beyond its inputs, one after
# the other; a program takes them after the graph's own inputs.
side_effects_inputs <- function(graph) {
  views_values(graph$side_effects, "inputs")
}

# The GraphValues the side effects of `graph` return beyond its outputs, one
# after the other; a program returns them after the graph's own outputs.
side_effects_outputs <- function(graph) {
  views_values(graph$side_effects, "outputs")
}

# The implicit outputs of the program of `graph` (see `compile_pjrt()`), one per
# value of `side_effects_outputs()`, in that order, or `NULL` when there are
# none.
side_effects_implicit_outputs <- function(graph) {
  outputs <- lapply(side_effects_outputs(graph), function(g) {
    list(dtype = as.character(dtype(g)), shape = shape(g))
  })
  if (length(outputs)) outputs
}

# The implicit inputs of the program of `graph` (see `compile_pjrt()`), one per
# value of `side_effects_inputs()`, in that order: before every run, its `get()`
# reads the buffer of the field of that name of the side effect's host object.
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
  function() side_effect_host_object(effect, NULL)[[name]]$data
}

# The function that updates the host objects of the side effects of `graph`
# after a run of its program with the call's arguments `args` (pjrt's
# `on_success`), `NULL` when the graph uses none. It finds all objects before it
# updates any, and refuses one found twice: two views of it started from the
# same value.
side_effects_on_success <- function(graph) {
  views <- unname(graph$side_effects)
  if (!length(views)) {
    return(NULL)
  }
  side_effects_settler(
    lapply(views, `[[`, "effect"),
    vapply(views, `[[`, numeric(1L), "n"),
    lapply(views, function(x) names(x$outputs))
  )
}

# The `on_success` of `side_effects_on_success()` for the side effects
# `effects`, the counts `counts` of the trace-time ones, and the names
# `output_names` of the values the program returns for each, which keeps
# nothing of the trace. A trace-time side effect is advanced by its count, and
# a runtime one receives its values among the program's implicit outputs
# `outputs`.
side_effects_settler <- function(effects, counts, output_names) {
  force(effects)
  force(counts)
  force(output_names)
  function(outputs, args) {
    objects <- lapply(effects, side_effect_host_object, args = args)
    for (i in seq_along(objects)[-1L]) {
      object <- objects[[i]]
      if (!is.null(object) && any(vapply(objects[seq_len(i - 1L)], identical, logical(1L), object))) {
        cli_abort("The same {.cls {class(object)[[1L]]}} was passed to the jitted function twice.")
      }
    }
    values <- split_side_effect_values(outputs, output_names)
    for (i in seq_along(effects)) {
      if (is_trace_time(effects[[i]])) {
        if (counts[[i]] > 0) {
          side_effect_advance(effects[[i]], objects[[i]], counts[[i]])
        }
      } else if (length(output_names[[i]])) {
        side_effect_receive(effects[[i]], objects[[i]], values[[i]])
      }
    }
    invisible(NULL)
  }
}
