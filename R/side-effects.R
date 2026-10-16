#' @include aaa.R
NULL


# Registers a side effect: a value outside of a traced function that
# primitives read and update (`new_primitive(side_effects = )`), such as the
# global random state, the token that orders prints, or an error code. Where
# several side effects are threaded together, they come in the order they were
# first registered in. Registering under a name that is already taken replaces
# that side effect.
#
# A side effect says where its value comes from, how it passes through loops
# and branches, and where it goes when a program has run:
#
# - Where it comes from: `get`, a function of no arguments returning the value
#   from the host as an R double or integer vector of `aval`'s elements, which a
#   program takes as an input; or `build`, a function of the program's
#   `stablehlo::Func` returning the value built inside the program (a
#   FuncValue), e.g. a fresh token.
# - How it passes through loops and branches: carried as data (`counted =
#   NULL`), so a loop carries it from one iteration to the next and `nv_if()`
#   returns the value of the branch that ran; or counted, a
#   `list(advance_box, fork)`: every primitive that uses it advances it by one
#   step, and every construct by a number of steps known when it is traced
#   (`nv_scan()`: its steps times the body's, `nv_if()`: the larger branch's,
#   `nv_while()`: one, as it forks). `advance_box(box, n)` returns a GraphBox of
#   the value advanced by `n` steps, traced into the current trace; `fork(desc)`
#   adds to the trace `desc` a call that advances the value by one step and
#   returns a GraphBox of a new value derived from it, for a loop body to
#   advance as often as it runs.
# - Where it goes: nowhere by default, as a program never returns it; or, for a
#   counted side effect, `advance(n)`, which advances the value on the host by
#   the `n` steps the program advanced it, so that the program need not return
#   it; or `receive(value)`, called with the value the program leaves behind,
#   which the program then returns as an `AnvlArray`, e.g. to raise an error a
#   check recorded.
#
# `backends` are the backends that can run a function using the side effect.
register_side_effect <- function(
  name,
  aval,
  get = NULL,
  build = NULL,
  counted = NULL,
  advance = NULL,
  receive = NULL,
  backends
) {
  if (is.null(get) == is.null(build)) {
    cli_abort(
      "Internal error: a side effect comes either from the host ({.arg get}) or from the program ({.arg build})."
    )
  }
  if (!is.null(advance) && is.null(counted)) {
    cli_abort("Internal error: only a counted side effect can be advanced on the host.")
  }
  globals$side_effects[[name]] <- list(
    name = name,
    aval = aval,
    get = get,
    build = build,
    counted = counted,
    advance = advance,
    receive = receive,
    backends = backends
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
# `GraphDescriptor()`). Each entry of `desc$side_effects` is a
# `list(input, current, advance)`: `input` is the GraphValue the side effect
# enters the graph through (`NULL` when it is captured), `current` the value the
# next primitive that uses it reads, and `advance` the number of steps the trace
# has advanced it by so far, which only a counted side effect uses.
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

# Prepares the branches `graphs` of a `prim_if()` call in the trace `desc` for
# the side effects either of them uses. A branch closes over the value of
# `desc`. For a carried side effect, both branches return the value they leave
# behind as one more output (a branch that does not use it the value it closes
# over), which the call returns and `side_effects_take()` makes the current
# value. A counted side effect is handed back by neither: whichever branch runs,
# the call advances it by the larger of the branches' advances, which
# `side_effects_branched()` applies after the call. Returns the names of the
# carried side effects, and the names and advances of the counted ones.
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
side_effects_branched <- function(desc, branched) {
  for (i in seq_along(branched$counted)) {
    name <- branched$counted[[i]]
    current <- side_effect_get(desc, name)
    side_effect_set(desc, name, side_effect(name)$counted$advance_box(current, branched$advance[[i]]))
    side_effect_advance(desc, name, branched$advance[[i]])
  }
  invisible(desc)
}

# Makes the body `graph` of a `prim_while()` or `prim_scan()` call in the trace
# `desc` carry the side effects it uses: each becomes an input and an output of
# its own, at the positions after `at`. The body starts from the current value
# of `desc`, and its final value becomes the current value (see
# `side_effects_take()`). With `fork = TRUE` (a `prim_while()` body, which runs a
# number of times that is not known when it is traced) a counted side effect
# starts from the value its `fork` derives instead, which advances `desc` by one
# step, and its final value is dropped. Returns the names, the operands the call
# passes for them, the body's advances, and which of them were forked.
side_effects_carry <- function(desc, graph, at = length(graph$inputs), fork = FALSE) {
  side_effects <- graph$side_effects
  graph$inputs <- append(graph$inputs, lapply(side_effects, `[[`, "input"), after = at)
  graph$outputs <- append(graph$outputs, lapply(side_effects, `[[`, "output"), after = at)
  graph$side_effects <- list()
  names <- names(side_effects)
  forked <- fork & vapply(names, side_effect_is_counted, logical(1L), USE.NAMES = FALSE)
  operands <- Map(
    function(name, forked) {
      if (forked) side_effect(name)$counted$fork(desc) else side_effect_get(desc, name)
    },
    names,
    forked
  )
  list(
    names = names,
    operands = unname(operands),
    advance = vapply(side_effects, `[[`, numeric(1L), "advance", USE.NAMES = FALSE),
    forked = forked
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

# The side effects of the top-level graph `graph` that its program takes as
# inputs, those it builds itself, and those it returns: their names, in the
# order of `graph$side_effects`.
side_effects_from_host <- function(graph) {
  Filter(function(name) !is.null(side_effect(name)$get), names(graph$side_effects))
}
side_effects_built <- function(graph) {
  Filter(function(name) !is.null(side_effect(name)$build), names(graph$side_effects))
}
side_effects_to_host <- function(graph) {
  Filter(function(name) !is.null(side_effect(name)$receive), names(graph$side_effects))
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
