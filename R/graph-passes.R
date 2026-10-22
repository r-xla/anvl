# Sub-graphs are closed -- what they read of this graph is an operand of their
# call -- so the passes below never have to look into them.
traverse_gnodes <- function(graph, fn) {
  for (call in graph$statements) {
    for (input in call$inputs) {
      fn(input)
    }
  }
  for (output in graph$outputs) {
    fn(output)
  }
}

# A shallow copy of `graph` for a pass to modify: `AnvlGraph` has reference
# semantics, so a pass must not change the graph it was given.
copy_graph <- function(graph) {
  AnvlGraph(
    statements = graph$statements,
    in_tree = graph$in_tree,
    out_tree = graph$out_tree,
    inputs = graph$inputs,
    outputs = graph$outputs,
    constants = graph$constants,
    is_static_flat = graph$is_static_flat,
    static_args_flat = graph$static_args_flat,
    # Positional, one entry per input: a pass may replace an input in place but
    # must not reorder or drop one, so this carries over as it is.
    rdata_types = graph$rdata_types
  )
}

# The part of `graph` that computes `outputs`: the calls they need and no
# others. Inputs and constants stay as they are. A print at the top level is
# never among them, since what reads its operand reads the operand itself (see
# `prim_print()`); one in the sub-graph of a kept call is kept with it.
prune_graph <- function(graph, outputs) {
  new_graph <- copy_graph(graph)
  new_graph$outputs <- outputs
  new_graph$out_tree <- NULL

  needed <- hashtab()
  mark <- function(gvals) {
    for (g in gvals) {
      if (!is_graph_literal(g)) needed[[g]] <- TRUE
    }
  }
  mark(outputs)
  keep <- logical(length(graph$statements))
  for (i in rev(seq_along(graph$statements))) {
    call <- graph$statements[[i]]
    if (any(vapply(call$outputs, \(g) isTRUE(needed[[g]]), logical(1L)))) {
      keep[[i]] <- TRUE
      mark(call$inputs)
    }
  }
  new_graph$statements <- graph$statements[keep]
  new_graph
}

# Whether running `graph` prints, from one of its calls or from the sub-graph
# of one.
graph_prints <- function(graph) {
  any(vapply(graph$statements, call_prints, logical(1L)))
}

call_prints <- function(call) {
  identical(call$primitive$name, "print") ||
    any(vapply(Filter(is_graph, call$params), graph_prints, logical(1L)))
}

# Splits `graph`, the body of a loop, into what can run once, before the loop,
# and what has to run in every step. `varying` are the inputs that change from
# step to step; a call stays in the loop if it reads one of them or what was
# computed from them. So does a call that prints, and everything computed from
# it, so that the print runs once per step.
#
# Returns:
#   - invariant: the values computed before the loop that the loop reads or
#     returns, in the order the loop first needs them.
#   - before: a graph with `graph`'s inputs that computes `invariant`; it reads
#     none of `varying`.
#   - loop: `graph` without the calls `before` makes, taking `invariant` as
#     inputs after `graph`'s own.
#   - in_loop: hashtab(gval -> TRUE) of the values computed in the loop, and
#     `varying`.
split_loop_invariant <- function(graph, varying) {
  in_loop <- hashtab()
  for (g in varying) {
    in_loop[[g]] <- TRUE
  }
  reads_loop <- function(x) !is_graph_literal(x) && isTRUE(in_loop[[x]])
  stays <- vapply(
    graph$statements,
    function(call) {
      stay <- call_prints(call) || any(vapply(call$inputs, reads_loop, logical(1L)))
      if (stay) {
        for (out in call$outputs) in_loop[[out]] <- TRUE
      }
      stay
    },
    logical(1L)
  )

  computed_before <- hashtab()
  for (call in graph$statements[!stays]) {
    for (out in call$outputs) computed_before[[out]] <- TRUE
  }
  read <- c(do.call(c, lapply(graph$statements[stays], \(call) call$inputs)), graph$outputs)
  seen <- hashtab()
  first_read_before <- vapply(
    read,
    function(g) {
      first <- !is_graph_literal(g) && isTRUE(computed_before[[g]]) && is.null(seen[[g]])
      if (first) seen[[g]] <- TRUE
      first
    },
    logical(1L)
  )
  invariant <- read[first_read_before]

  loop <- copy_graph(graph)
  loop$statements <- graph$statements[stays]
  loop$inputs <- c(graph$inputs, invariant)
  loop$rdata_types <- NULL
  list(invariant = invariant, before = prune_graph(graph, invariant), loop = loop, in_loop = in_loop)
}

remove_unused_constants <- function(graph) {
  new_graph <- copy_graph(graph)

  is_used <- hashtab()
  traverse_gnodes(new_graph, function(gval) {
    if (is_graph_value(gval) && is_concrete_array(gval$aval)) {
      is_used[[gval]] <- TRUE
    }
  })
  new_graph$constants <- new_graph$constants[vapply(
    new_graph$constants,
    function(const) isTRUE(is_used[[const]]),
    logical(1L)
  )]
  new_graph
}

inline_scalarish_constants <- function(graph) {
  is_scalarish <- function(gval) {
    is_graph_value(gval) && is_concrete_array(gval$aval) && (nelts(gval$aval) == 1L)
  }

  scalarish_to_lit <- function(gval) {
    GraphLiteral(LiteralArray(
      gval$aval$data,
      shape = shape(gval$aval),
      dtype = dtype(gval$aval)
    ))
  }

  new_graph <- copy_graph(graph)

  map <- hashtab()
  for (const in new_graph$constants) {
    if (is_scalarish(const)) {
      map[[const]] <- scalarish_to_lit(const)
    }
  }
  replace_nodes <- function(nodes) lapply(nodes, \(node) map[[node]] %||% node)
  new_graph$inputs <- replace_nodes(new_graph$inputs)
  new_graph$statements <- lapply(new_graph$statements, function(call) {
    call$inputs <- replace_nodes(call$inputs)
    call
  })
  new_graph$outputs <- replace_nodes(new_graph$outputs)
  new_graph$constants <- new_graph$constants[vapply(
    new_graph$constants,
    function(const) is.null(map[[const]]),
    logical(1L)
  )]
  new_graph
}

graph_optimization_passes <- list(
  inline_scalars = inline_scalarish_constants,
  remove_unused_constants = remove_unused_constants
)

resolve_optimization_passes <- function(optimize) {
  passes <- names(graph_optimization_passes)
  if (is.logical(optimize)) {
    assert_flag(optimize)
    return(if (optimize) passes else character())
  }
  assert_character(optimize, any.missing = FALSE)
  invalid <- setdiff(optimize, passes)
  if (length(invalid)) {
    cli_abort(c(
      "Unknown optimization pass{?es}: {.val {invalid}}.",
      i = "Available passes: {.val {passes}}."
    ))
  }
  intersect(passes, optimize)
}

optimize_graph <- function(graph, optimize = TRUE) {
  for (name in resolve_optimization_passes(optimize)) {
    graph <- graph_optimization_passes[[name]](graph)
  }
  graph
}
