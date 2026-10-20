#' @include aaa.R
NULL


# Registers a side effect: a value outside of a traced function that
# primitives read and update (`new_primitive(side_effects = )`), such as the
# token that orders prints. Its value is one or more values, of the types
# `avals` (a named list of `AbstractValue`s), which a call
# of a primitive takes as arguments named like them and returns updated as its
# last outputs, so that every use takes the values the previous one returned.
# Side effects used together come in the order they were first registered in.
# The internals vignette describes the mechanism.
#
# - Where the values come from: either `get()`, for values that live outside of
#   the program, which returns them as a list of arrays the program takes as
#   inputs; or `build(desc)`, for values the program creates itself, such as a
#   token, which adds the calls that create them to the trace `desc` and
#   returns them as a list of GraphBoxes.
# - `after_run`: `NULL`, or a function the host calls after every run of a
#   program that uses the side effect, as `after_run(values)`, with the values
#   the program leaves behind, which the program returns for that purpose, e.g.
#   to update values that outlive the program. Without it, the program drops
#   the values at its end.
# - `labels`: what a printed graph names the values after (see `GraphValue()`).
register_side_effect <- function(
  name,
  avals,
  get = NULL,
  build = NULL,
  after_run = NULL,
  labels = names(avals)
) {
  if (is.null(get) == is.null(build)) {
    cli_abort("Internal error: a side effect comes from the host ({.arg get}) or from the program ({.arg build}).")
  }
  globals$side_effects[[name]] <- list(
    name = name,
    avals = avals,
    labels = labels,
    get = get,
    build = build,
    after_run = after_run
  )
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

# The number of values of each of the side effects `names`.
side_effects_widths <- function(names) {
  vapply(names, function(name) length(side_effect(name)$avals), integer(1L), USE.NAMES = FALSE)
}

# The current values of the side effect `name` in the trace `desc`, a named
# list of GraphBoxes, registered on first use as `desc$side_effects_mode` says
# (see `GraphDescriptor()`): as fresh inputs, or as the current values of the
# parent trace, which the trace closes over.
side_effect_get <- function(desc, name) {
  entry <- desc$side_effects[[name]]
  if (!is.null(entry)) {
    return(entry$current)
  }
  mode <- desc$side_effects_mode
  if (identical(mode, "input") && !is.null(side_effect(name)$build)) {
    return(side_effect_enter(desc, name, NULL, current = side_effect(name)$build(desc)))
  }
  if (identical(mode, "input") || identical(mode, "loop")) {
    return(side_effect_enter(desc, name, side_effect_graph_values(name)))
  }
  if (!identical(mode, "capture")) {
    cli_abort(c(
      "The side effect {.val {name}} cannot be used here.",
      i = paste(
        "A side effect can only be used where its updated value can be handed back,",
        "so not in the condition of {.fn nv_while}, the function of a reduction or a",
        "scatter, or any other function traced without a {.arg side_effects_mode}."
      )
    ))
  }
  side_effect_enter(desc, name, NULL, current = side_effect_get(desc$parent, name))
}

# Registers the side effect `name` in the trace `desc`, entering it through the
# GraphValues `input` (`NULL` for one the trace closes over or builds), with
# the current values `current`. An entry is a `list(input, current)`.
side_effect_enter <- function(desc, name, input, current = lapply(input, register_gval, desc = desc)) {
  current <- side_effect_label(name, current)
  desc$side_effects[[name]] <- list(input = input, current = current)
  current
}

# Fresh GraphValues for the values of the side effect `name`, labelled.
side_effect_graph_values <- function(name) {
  side_effect <- side_effect(name)
  Map(GraphValue, side_effect$avals, side_effect$labels)
}

# `boxes`, values of the side effect `name`, with their nodes labelled after
# them where they are not yet.
side_effect_label <- function(name, boxes) {
  Map(
    function(box, label) {
      if (!is_graph_literal(box$gnode) && is.null(box$gnode$label)) {
        box$gnode$label <- label
      }
      box
    },
    boxes,
    side_effect(name)$labels
  )
}

# Makes the GraphBoxes `boxes` the current values of the side effect `name` in
# the trace `desc`.
side_effect_set <- function(desc, name, boxes) {
  desc$side_effects[[name]]$current <- side_effect_label(name, stats::setNames(boxes, names(side_effect(name)$avals)))
  invisible(desc)
}

# The side effects of the trace `desc` as the graph built from it records them:
# a `list(input, output)` per side effect, in registration order, the input and
# output a list of GraphValues each, or `NULL`. A top-level trace returns only
# the side effects the host receives (see `side_effect_is_received()`).
side_effects_of_desc <- function(desc) {
  entries <- side_effects_ordered(desc$side_effects)
  Map(
    function(name, entry) {
      output <- lapply(entry$current, `[[`, "gnode")
      if (identical(desc$side_effects_mode, "input") && !side_effect_is_received(name)) {
        output <- NULL
      }
      list(input = entry$input, output = output)
    },
    names(entries),
    entries
  )
}

# Makes the current values of the side effects `names` in `desc` those that
# `out`, the boxes a call returns, holds at the positions after `at`, and
# returns `out` without the side effects' values.
side_effects_take <- function(desc, names, out, at = length(out) - sum(side_effects_widths(names))) {
  if (!length(names)) {
    return(out)
  }
  widths <- side_effects_widths(names)
  positions <- at + seq_len(sum(widths))
  values <- split(out[positions], rep(seq_along(names), widths))
  for (i in seq_along(names)) {
    side_effect_set(desc, names[[i]], unname(values[[i]]))
  }
  out[-positions]
}

# Prepares the branches `graphs` of a `prim_if()` call in the trace `desc`,
# which close over the side effects they use. Both branches return the values
# they leave behind of every side effect either uses as more outputs (a branch
# that does not use it the values it closes over), for `side_effects_take()`.
# Returns the names of the side effects.
side_effects_branches <- function(desc, graphs) {
  used <- unlist(lapply(graphs, function(graph) names(graph$side_effects)))
  names <- intersect(names(globals$side_effects), used)
  for (graph in graphs) {
    for (name in names) {
      output <- graph$side_effects[[name]]$output
      if (is.null(output)) {
        output <- lapply(side_effect_get(desc, name), `[[`, "gnode")
        graph$constants <- c(graph$constants, unname(output))
      }
      graph$outputs <- c(graph$outputs, unname(output))
    }
    graph$side_effects <- list()
  }
  names
}

# Makes the body `graph` of a `prim_while()` or `prim_scan()` call in the trace
# `desc` carry the side effects it uses: their values become inputs and outputs
# of the body, at the positions after `at`. The body starts from the current
# values of `desc`. Returns the names, and the operands the call passes for
# them.
side_effects_carry <- function(desc, graph, at = length(graph$inputs)) {
  side_effects <- graph$side_effects
  flat <- function(field) unname(unlist(lapply(side_effects, `[[`, field), recursive = FALSE))
  graph$inputs <- append(graph$inputs, flat("input"), after = at)
  graph$outputs <- append(graph$outputs, flat("output"), after = at)
  graph$side_effects <- list()
  names <- names(side_effects)
  operands <- lapply(names, side_effect_get, desc = desc)
  list(names = names, operands = unname(unlist(operands, recursive = FALSE)))
}

# Whether a program returns the values of the side effect `name` to the host:
# those of a side effect the host calls `after_run()` for.
side_effect_is_received <- function(name) {
  !is.null(side_effect(name)$after_run)
}

# The side effects of the top-level graph `graph` that its program takes from
# the host, and those it returns to the host.
side_effects_from_host <- function(graph) {
  Filter(function(name) !is.null(side_effect(name)$get), names(graph$side_effects))
}
side_effects_to_host <- function(graph) {
  Filter(side_effect_is_received, names(graph$side_effects))
}

# The GraphValues of the side effects `names` of `graph` in `field`, `"input"`
# or `"output"`, one after the other.
side_effects_values <- function(graph, names, field) {
  unname(unlist(lapply(names, function(name) graph$side_effects[[name]][[field]]), recursive = FALSE))
}
