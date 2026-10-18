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
