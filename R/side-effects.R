#' @include aaa.R
NULL


# Registers a side effect: a value outside of a traced function that
# primitives read and update (`new_primitive(side_effects = )`), such as the
# global random state. Where several side effects are threaded together, they
# come in the order they were first registered in. Registering under a name
# that is already taken replaces that side effect.
#
# A program never returns a side effect's value. Every primitive that uses it
# advances it by one step, and every construct around such calls advances it by
# a number of steps known when it is traced (see `side_effect_advance()`), so
# the host applies that advance itself once a call has run.
#
# Params:
#   - name: the name the side effect is registered under, and which primitives
#     declare.
#   - aval: the AbstractArray (data type and shape) of the value.
#   - get: a function of no arguments returning the current value as an R
#     double vector of `aval`'s elements, creating it when it is not set.
#   - advance: a function of one argument `n`, a count, advancing the value on
#     the host by `n` steps.
#   - advance_box: a function of a GraphBox holding the value and a count `n`,
#     returning a GraphBox of the value advanced by `n` steps, traced into the
#     current trace.
#   - fork: a function of a GraphDescriptor, adding to that trace a call that
#     advances the value by one step and returning a GraphBox of a new value
#     derived from it, for a loop body to advance as often as it runs.
#   - backends: the backends that can run a function using the side effect.
register_side_effect <- function(name, aval, get, advance, advance_box, fork, backends) {
  globals$side_effects[[name]] <- list(
    name = name,
    aval = aval,
    get = get,
    advance = advance,
    advance_box = advance_box,
    fork = fork,
    backends = backends
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

# The current value of the side effect `name` in the trace `desc`, a GraphBox,
# registered on first use as `desc$side_effects_mode` says (see
# `GraphDescriptor()`). Each entry of `desc$side_effects` is a
# `list(input, current, advance)`: `input` is the GraphValue the side effect
# enters the graph through (`NULL` when it is captured), `current` the value the
# next primitive that uses it reads, and `advance` the number of steps the trace
# has advanced it by so far.
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
  desc$side_effects[[name]] <- list(input = input, current = current, advance = 0)
  current
}

# Makes the GraphBox `box` the current value of the side effect `name` in the
# trace `desc`.
side_effect_set <- function(desc, name, box) {
  desc$side_effects[[name]]$current <- box
  invisible(desc)
}

# Records that the trace `desc` advanced the side effect `name` by `n` more
# steps.
side_effect_advance <- function(desc, name, n) {
  desc$side_effects[[name]]$advance <- desc$side_effects[[name]]$advance + n
  invisible(desc)
}

# The side effects of the trace `desc` as the graph built from it records them:
# a `list(input, output, advance)` per side effect, in registration order.
side_effects_of_desc <- function(desc) {
  lapply(side_effects_ordered(desc$side_effects), function(entry) {
    list(input = entry$input, output = entry$current$gnode, advance = entry$advance)
  })
}

# Makes the current values of the side effects `names` in `desc` those that
# `out`, the boxes a higher-order call returns, holds at the positions after
# `at`, and returns `out` without them. A side effect whose entry of `keep` is
# `FALSE` is dropped from `out` without becoming the current value.
side_effects_take <- function(desc, names, out, at = length(out) - length(names), keep = rep(TRUE, length(names))) {
  if (!length(names)) {
    return(out)
  }
  positions <- at + seq_along(names)
  for (i in which(keep)) {
    side_effect_set(desc, names[[i]], out[[positions[[i]]]])
  }
  out[-positions]
}

# Clears the side effects of the branches `graphs` of a `prim_if()` call in the
# trace `desc`. A branch reads the value of `desc` it closes over and hands
# nothing back: whichever branch runs, the call advances each side effect by the
# larger of the branches' advances, which `side_effects_branched()` applies
# after the call. Returns the names of these side effects and those advances.
side_effects_branches <- function(desc, graphs) {
  used <- unlist(lapply(graphs, function(graph) names(graph$side_effects)))
  names <- intersect(names(globals$side_effects), used)
  advance <- vapply(
    names,
    function(name) max(vapply(graphs, function(graph) graph$side_effects[[name]]$advance %||% 0, numeric(1L))),
    numeric(1L)
  )
  for (graph in graphs) {
    graph$side_effects <- list()
  }
  list(names = names, advance = unname(advance))
}

# Advances the side effects `branched$names` of the trace `desc` by
# `branched$advance`, as `side_effects_branches()` returned them.
side_effects_branched <- function(desc, branched) {
  for (i in seq_along(branched$names)) {
    name <- branched$names[[i]]
    current <- side_effect_get(desc, name)
    side_effect_set(desc, name, side_effect(name)$advance_box(current, branched$advance[[i]]))
    side_effect_advance(desc, name, branched$advance[[i]])
  }
  invisible(desc)
}

# Makes the body `graph` of a `prim_while()` or `prim_scan()` call in the trace
# `desc` carry the side effects it uses: each becomes an input and an output of
# its own, at the positions after `at`. With `fork = TRUE` (a `prim_while()`
# body, which runs a number of times that is not known when it is traced) the
# body starts from the value each side effect's `fork` derives, which advances
# `desc` by one step, and its final value is dropped (see `side_effects_take()`'s
# `keep`). Otherwise it starts from the current value of `desc` and the call
# advances `desc` by the body's advance per iteration. Returns the names, the
# operands the call passes for them, and the body's advances.
side_effects_carry <- function(desc, graph, at = length(graph$inputs), fork = FALSE) {
  side_effects <- graph$side_effects
  graph$inputs <- append(graph$inputs, lapply(side_effects, `[[`, "input"), after = at)
  graph$outputs <- append(graph$outputs, lapply(side_effects, `[[`, "output"), after = at)
  graph$side_effects <- list()
  names <- names(side_effects)
  operands <- lapply(names, function(name) {
    if (fork) side_effect(name)$fork(desc) else side_effect_get(desc, name)
  })
  list(
    names = names,
    operands = operands,
    advance = vapply(side_effects, `[[`, numeric(1L), "advance", USE.NAMES = FALSE)
  )
}

# Makes `extra` -- the boxes `graph_value_and_grad()` replayed the values
# `graph` leaves its side effects at into, one per side effect of `graph` (see
# `side_effect_outputs()`) -- the current values of those side effects in the
# current trace, which advance by `graph`'s advances.
side_effects_set <- function(graph, extra) {
  desc <- current_descriptor()
  names <- names(graph$side_effects)
  for (i in seq_along(names)) {
    side_effect_set(desc, names[[i]], extra[[i]])
    side_effect_advance(desc, names[[i]], graph$side_effects[[i]]$advance)
  }
  invisible(NULL)
}

# Errors unless `backend` can run every side effect `graph` uses.
check_side_effects <- function(graph, backend) {
  for (name in names(graph$side_effects)) {
    if (!backend %in% side_effect(name)$backends) {
      cli_abort("The {backend} backend does not support the side effect {.val {name}}.")
    }
  }
  invisible(graph)
}
