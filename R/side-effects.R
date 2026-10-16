#' @include aaa.R
NULL


# Registers a side effect: a value outside of a traced function that
# primitives read and update (`new_primitive(side_effects = )`), such as the
# global random state or the token that orders prints. Side effects threaded
# together come in the order they were first registered in. The internals
# vignette describes the mechanism.
#
# - `get` / `build`: where the value comes from. `get()` returns it from the
#   host as an R double or integer vector of `aval`'s elements, which a program
#   takes as an input; `build(func)` builds it in the program's
#   `stablehlo::Func`.
# - `counted`: `NULL` for a value carried as data through loops and branches,
#   or a `list(advance_box, fork, advance)` for one that every use advances by
#   one step: `advance_box(box, n)` advances a GraphBox of it by `n` steps in
#   the current trace, `fork(desc)` adds a call to `desc` that advances it by
#   one step and returns a GraphBox of a value derived from it (for a
#   `nv_while()` body), and `advance(n)` advances it on the host by the `n`
#   steps a program advanced it, so that the program need not return it.
# - `receive`: `NULL`, or a function the value a program leaves behind is handed
#   to, which the program then returns, e.g. to raise an error a check recorded.
register_side_effect <- function(name, aval, get = NULL, build = NULL, counted = NULL, receive = NULL) {
  if (is.null(get) == is.null(build)) {
    cli_abort("Internal error: a side effect comes from the host ({.arg get}) or from the program ({.arg build}).")
  }
  globals$side_effects[[name]] <- list(
    name = name,
    aval = aval,
    get = get,
    build = build,
    counted = counted,
    receive = receive
  )
  invisible(NULL)
}

# The side effect registered under `name`.
side_effect <- function(name) {
  globals$side_effects[[name]] %||% cli_abort("No side effect is registered under the name {.val {name}}.")
}

# Whether the side effect `name` is counted rather than carried.
side_effect_is_counted <- function(name) {
  !is.null(side_effect(name)$counted)
}

# `x`, a list named by side effects, in the order they were registered in.
side_effects_ordered <- function(x) {
  x[intersect(names(globals$side_effects), names(x))]
}

# The current value of the side effect `name` in the trace `desc`, a GraphBox,
# registered on first use as `desc$side_effects_mode` says (see
# `GraphDescriptor()`): as a fresh input, or as the current value of the parent
# trace, which the trace closes over.
side_effect_get <- function(desc, name) {
  entry <- desc$side_effects[[name]]
  if (!is.null(entry)) {
    return(entry$current)
  }
  mode <- desc$side_effects_mode
  if (identical(mode, "input")) {
    return(side_effect_enter(desc, name, GraphValue(side_effect(name)$aval)))
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
# GraphValue `input` (`NULL` for one the trace closes over), with the current
# value `current`. An entry is a `list(input, current, advance)`, `advance` the
# number of steps the trace has advanced a counted side effect by so far.
side_effect_enter <- function(desc, name, input, current = register_gval(desc, input)) {
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
# steps, if it is counted.
side_effect_advance <- function(desc, name, n) {
  if (side_effect_is_counted(name)) {
    desc$side_effects[[name]]$advance <- desc$side_effects[[name]]$advance + n
  }
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
side_effects_take <- function(
  desc,
  names,
  out,
  at = length(out) - length(names),
  keep = rep(TRUE, length(names))
) {
  if (!length(names)) {
    return(out)
  }
  positions <- at + seq_along(names)
  for (i in which(keep)) {
    side_effect_set(desc, names[[i]], out[[positions[[i]]]])
  }
  out[-positions]
}

# Prepares the branches `graphs` of a `prim_if()` call in the trace `desc`,
# which close over the side effects they use. Both branches return the value
# they leave behind of a carried side effect as one more output (a branch that
# does not use it the value it closes over), for `side_effects_take()`. A
# counted side effect is handed back by neither: whichever branch runs, the
# call advances it by the larger of the branches' advances, which
# `side_effects_advance_branches()` applies after the call. Returns the names
# of the carried side effects, and the names and advances of the counted ones.
side_effects_branches <- function(desc, graphs) {
  used <- unlist(lapply(graphs, function(graph) names(graph$side_effects)))
  names <- intersect(names(globals$side_effects), used)
  counted <- names[vapply(names, side_effect_is_counted, logical(1L))]
  carried <- setdiff(names, counted)
  advance <- vapply(
    counted,
    function(name) max(vapply(graphs, function(graph) graph$side_effects[[name]]$advance %||% 0, numeric(1L))),
    numeric(1L)
  )
  for (graph in graphs) {
    for (name in carried) {
      output <- graph$side_effects[[name]]$output
      if (is.null(output)) {
        output <- side_effect_get(desc, name)$gnode
        graph$constants <- c(graph$constants, list(output))
      }
      graph$outputs <- c(graph$outputs, list(output))
    }
    graph$side_effects <- list()
  }
  list(carried = carried, counted = counted, advance = unname(advance))
}

# Advances the counted side effects `branched$counted` of the trace `desc` by
# `branched$advance`, as `side_effects_branches()` returned them.
side_effects_advance_branches <- function(desc, branched) {
  for (i in seq_along(branched$counted)) {
    name <- branched$counted[[i]]
    advanced <- side_effect(name)$counted$advance_box(side_effect_get(desc, name), branched$advance[[i]])
    side_effect_set(desc, name, advanced)
    side_effect_advance(desc, name, branched$advance[[i]])
  }
  invisible(desc)
}

# Makes the body `graph` of a `prim_while()` or `prim_scan()` call in the trace
# `desc` carry the side effects it uses: each becomes an input and an output of
# its own, at the positions after `at`. The body starts from the current value
# of `desc`. With `fork = TRUE` (a `prim_while()` body, which runs a number of
# times that is not known when it is traced) a counted side effect starts from
# the value its `fork` derives instead, which advances `desc` by one step.
# Returns the names, the operands the call passes for them, and the body's
# advances.
side_effects_carry <- function(desc, graph, at = length(graph$inputs), fork = FALSE) {
  side_effects <- graph$side_effects
  graph$inputs <- append(graph$inputs, lapply(side_effects, `[[`, "input"), after = at)
  graph$outputs <- append(graph$outputs, lapply(side_effects, `[[`, "output"), after = at)
  graph$side_effects <- list()
  names <- names(side_effects)
  operands <- lapply(names, function(name) {
    if (fork && side_effect_is_counted(name)) side_effect(name)$counted$fork(desc) else side_effect_get(desc, name)
  })
  list(
    names = names,
    operands = operands,
    advance = vapply(side_effects, `[[`, numeric(1L), "advance", USE.NAMES = FALSE)
  )
}

# The side effects of the top-level graph `graph` that its program takes from
# the host, those it builds, and those it returns to the host.
side_effects_from_host <- function(graph) {
  Filter(function(name) !is.null(side_effect(name)$get), names(graph$side_effects))
}
side_effects_built <- function(graph) {
  Filter(function(name) !is.null(side_effect(name)$build), names(graph$side_effects))
}
side_effects_to_host <- function(graph) {
  Filter(function(name) !is.null(side_effect(name)$receive), names(graph$side_effects))
}
