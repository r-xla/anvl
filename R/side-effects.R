#' @include aaa.R
NULL


# Registers a side effect: a value outside of a traced function that
# primitives read and update (`new_primitive(side_effects = )`), such as the
# global RNG state. Where several side effects are threaded together, they come
# in the order they were first registered in. Registering under a name that is
# already taken replaces that side effect.
#
# Params:
#   - name: the name the side effect is registered under, and which primitives
#     declare.
#   - aval: the AbstractArray (data type and shape) of the value.
#   - getter: a function of no arguments returning the current value, an
#     AnvlArray, creating it when it is not set.
#   - setter: a function of one argument storing the value a call leaves
#     behind.
#   - rules: what each backend needs to run the side effect, named by backend;
#     for pjrt, the state slot `pjrt_state_slot()` builds. A backend without a
#     rule refuses a function that uses the side effect.
register_side_effect <- function(name, aval, getter, setter, rules = list()) {
  globals$side_effects[[name]] <- list(name = name, aval = aval, getter = getter, setter = setter, rules = rules)
  invisible(NULL)
}

# The side effect registered under `name`.
side_effect <- function(name) {
  globals$side_effects[[name]] %||% cli_abort("No side effect is registered under the name {.val {name}}.")
}

# `x`, a list named by side effects, in the order they were registered in.
side_effects_ordered <- function(x) {
  x[intersect(names(globals$side_effects), names(x))]
}

# The current value of the side effect `name` in the trace `desc`, a GraphBox,
# registered on first use as `desc$side_effects_mode` says (see
# `GraphDescriptor()`). Each entry of `desc$side_effects` is a
# `list(input, current)`: `input` is the GraphValue the side effect enters the
# graph through (`NULL` when it is captured), and `current` the value the next
# primitive that uses it reads.
side_effect_get <- function(desc, name) {
  entry <- desc$side_effects[[name]]
  if (!is.null(entry)) {
    return(entry$current)
  }
  mode <- desc$side_effects_mode
  if (identical(mode, "input")) {
    input <- GraphValue(side_effect(name)$aval)
    current <- register_gval(desc, input)
  } else if (identical(mode, "capture")) {
    input <- NULL
    current <- side_effect_get(desc$parent, name)
  } else {
    cli_abort(c(
      "The side effect {.val {name}} cannot be used here.",
      i = paste(
        "A side effect can only be used where its updated value can be handed back,",
        "so not in the condition of {.fn nv_while}, the function of a reduction or a",
        "scatter, or any other function traced without a {.arg side_effects_mode}."
      )
    ))
  }
  desc$side_effects[[name]] <- list(input = input, current = current)
  current
}

# Makes the GraphBox `box` the current value of the side effect `name` in the
# trace `desc`.
side_effect_set <- function(desc, name, box) {
  desc$side_effects[[name]]$current <- box
  invisible(desc)
}

# The side effects of the trace `desc` as the graph built from it records them:
# a `list(input, output)` per side effect, in registration order.
side_effects_of_desc <- function(desc) {
  lapply(side_effects_ordered(desc$side_effects), function(entry) {
    list(input = entry$input, output = entry$current$gnode)
  })
}

# Makes the current values of the side effects `names` in `desc` those that
# `out`, the boxes a higher-order call returns, holds at the positions after
# `at`, and returns `out` without them.
side_effects_take <- function(desc, names, out, at = length(out) - length(names)) {
  if (!length(names)) {
    return(out)
  }
  positions <- at + seq_along(names)
  for (i in seq_along(names)) {
    side_effect_set(desc, names[[i]], out[[positions[[i]]]])
  }
  out[-positions]
}

# Makes the branches `graphs` of a `prim_if()` call in the trace `desc` return
# every side effect one of them uses after their own outputs: the value it
# leaves behind, or, for a branch that does not use it, the current value of
# `desc`, which the branch then closes over. Returns the names of these side
# effects, for `side_effects_take()`.
side_effects_branches <- function(desc, graphs) {
  used <- unlist(lapply(graphs, function(graph) names(graph$side_effects)))
  names <- intersect(names(globals$side_effects), used)
  for (graph in graphs) {
    for (name in names) {
      output <- graph$side_effects[[name]]$output
      if (is.null(output)) {
        output <- side_effect_get(desc, name)$gnode
        graph$constants <- c(graph$constants, list(output))
      }
      graph$outputs <- c(graph$outputs, list(output))
    }
    graph$side_effects <- list()
  }
  names
}

# Makes the body `graph` of a `prim_while()` or `prim_scan()` call in the trace
# `desc` carry the side effects it uses: each becomes an input and an output of
# its own, at the positions after `at`. Returns their names and the operands
# the call passes for them, the current values in `desc`.
side_effects_carry <- function(desc, graph, at = length(graph$inputs)) {
  side_effects <- graph$side_effects
  graph$inputs <- append(graph$inputs, lapply(side_effects, `[[`, "input"), after = at)
  graph$outputs <- append(graph$outputs, lapply(side_effects, `[[`, "output"), after = at)
  graph$side_effects <- list()
  names <- names(side_effects)
  list(
    names = names,
    operands = lapply(names, side_effect_get, desc = desc)
  )
}

# Makes `extra` -- the boxes `graph_value_and_grad()` replayed the values
# `graph` leaves its side effects at into, one per side effect of `graph` (see
# `side_effect_outputs()`) -- the current values of those side effects in the
# current trace.
side_effects_set <- function(graph, extra) {
  desc <- current_descriptor()
  names <- names(graph$side_effects)
  for (i in seq_along(names)) {
    side_effect_set(desc, names[[i]], extra[[i]])
  }
  invisible(NULL)
}

# Errors unless `backend` has a rule for every side effect `graph` uses.
check_side_effects <- function(graph, backend) {
  for (name in names(graph$side_effects)) {
    if (is.null(side_effect(name)$rules[[backend]])) {
      cli_abort("The {backend} backend does not support the side effect {.val {name}}.")
    }
  }
  invisible(graph)
}
