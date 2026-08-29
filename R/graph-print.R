#' @include graph.R

format_node_id <- function(node, node_ids) {
  if (is_graph_literal(node)) {
    return(format_literal(node))
  }
  id <- node_ids[[node]]
  if (is.null(id)) {
    return("???")
  }
  sprintf("%%%s", id)
}

format_literal <- function(node) {
  val <- node$aval$data
  if (is_anvl_array(val)) {
    val <- as_array(val)
  }
  dt <- repr(dtype(node$aval))
  dt <- if (node$aval$ambiguous) paste0(dt, "?") else dt
  shp <- shape(node$aval)
  sprintf("%s:%s%s", val, dt, if (length(shp)) sprintf("[%s]", shape2string(shp)) else "")
}

format_aval_short <- function(aval) {
  sprintf("%s[%s]", paste0(repr(dtype(aval)), if (aval$ambiguous) "?" else ""), paste(shape(aval), collapse = ", "))
}

build_node_ids <- function(inputs, constants, calls) {
  node_ids <- hashtab()
  for (i in seq_along(inputs)) {
    # don't use i, because i1 looks like a the boolean type
    node_ids[[inputs[[i]]]] <- paste0("x", i)
  }
  for (i in seq_along(constants)) {
    node_ids[[constants[[i]]]] <- paste0("c", i)
  }
  counter <- 1L
  for (call in calls) {
    for (out in call$outputs) {
      node_ids[[out]] <- as.character(counter)
      counter <- counter + 1L
    }
  }
  node_ids
}

format_param <- function(param) {
  if (is.null(param) || (is.list(param) && length(param) == 0L)) {
    return("")
  }
  format_param_value(param)
}

format_param_value <- function(p) {
  if (is.null(p)) {
    return("NULL")
  }
  if (is_graph(p)) {
    return(sprintf("graph[%d -> %d]", length(p$inputs), length(p$outputs)))
  }
  if (is_dtype(p)) {
    return(repr(p))
  }
  if (is.atomic(p)) {
    if (length(p) == 0L) {
      return(sprintf("%s(0)", typeof(p)))
    }
    elts <- if (is.character(p)) sprintf('"%s"', p) else format(p)
    if (length(p) == 1L) elts else sprintf("c(%s)", paste(elts, collapse = ", "))
  } else if (is.list(p)) {
    parts <- vapply(p, format_param_value, character(1))
    if (!is.null(names(p))) {
      parts <- paste0(names(p), " = ", parts)
    }
    sprintf("[%s]", paste(parts, collapse = ", "))
  } else {
    out <- try(deparse(p, nlines = 1L), silent = TRUE)
    if (inherits(out, "try-error") || length(out) != 1L) {
      sprintf("<%s>", class(p)[[1L]])
    } else {
      out
    }
  }
}

format_call <- function(call, node_ids, indent = "  ") {
  input_ids <- vapply(call$inputs, format_node_id, character(1), node_ids = node_ids)
  inputs_str <- paste(input_ids, collapse = ", ")

  output_ids <- vapply(call$outputs, format_node_id, character(1), node_ids = node_ids)
  output_types <- vapply(call$outputs, \(x) format_aval_short(x$aval), character(1))

  outputs_str <- if (length(call$outputs) == 1L) {
    sprintf("%s: %s", output_ids, output_types)
  } else {
    sprintf("(%s): (%s)", paste(output_ids, collapse = ", "), paste(output_types, collapse = ", "))
  }

  params_str <- format_param(call$params)
  if (params_str != "") {
    params_str <- sprintf(" %s ", params_str)
  }
  sprintf("%s%s = %s%s(%s)", indent, outputs_str, call$primitive$name, params_str, inputs_str)
}

format_graph_body <- function(inputs, constants, calls, outputs, title = "Graph") {
  lines <- character()

  # Build node ID mapping
  node_ids <- build_node_ids(inputs, constants, calls)

  # Header
  lines <- c(lines, sprintf("<%s>", title))

  # Inputs section
  if (length(inputs) > 0L) {
    input_strs <- vapply(
      inputs,
      function(node) {
        sprintf("    %s: %s", format_node_id(node, node_ids), format_aval_short(node$aval))
      },
      character(1)
    )
    lines <- c(lines, "  Inputs:", input_strs)
  } else {
    lines <- c(lines, "  Inputs: (none)")
  }

  # Constants section
  if (length(constants) > 0L) {
    const_strs <- vapply(
      constants,
      function(node) {
        sprintf("    %s: %s", format_node_id(node, node_ids), format_aval_short(node$aval))
      },
      character(1)
    )
    lines <- c(lines, "  Constants:", const_strs)
  }

  # Calls section
  if (length(calls) > 0L) {
    lines <- c(lines, "  Body:")
    for (call in calls) {
      lines <- c(lines, format_call(call, node_ids, indent = "    "))
    }
  } else {
    lines <- c(lines, "  Body: (empty)")
  }

  # Outputs section
  if (length(outputs) > 0L) {
    output_strs <- vapply(
      outputs,
      function(node) {
        if (is_graph_literal(node)) {
          sprintf("    %s", format_literal(node))
        } else {
          sprintf("    %s: %s", format_node_id(node, node_ids), format_aval_short(node$aval))
        }
      },
      character(1)
    )
    lines <- c(lines, "  Outputs:", output_strs)
  } else {
    lines <- c(lines, "  Outputs: (none)")
  }

  paste(lines, collapse = "\n")
}

#' @export
format.PrimitiveCall <- function(x, ...) {
  inputs <- paste(
    vapply(
      x$inputs,
      function(inp) {
        if (is_graph_literal(inp)) {
          format_literal(inp)
        } else {
          format_aval_short(inp$aval)
        }
      },
      character(1)
    ),
    collapse = ", "
  )
  outputs <- paste(vapply(x$outputs, \(out) format_aval_short(out$aval), character(1)), collapse = ", ")
  params_str <- if (length(x$params) > 0L) sprintf(" {%d params}", length(x$params)) else ""
  sprintf("%s(%s)%s -> %s", x$primitive$name, inputs, params_str, outputs)
}

#' @export
format.AnvlGraph <- function(x, ...) {
  format_graph_body(
    inputs = x$inputs,
    constants = x$constants,
    calls = x$calls,
    outputs = x$outputs,
    title = "AnvlGraph"
  )
}

#' @export
print.AnvlGraph <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
format.GraphDescriptor <- function(x, ...) {
  # Convert hashtab constants to list
  constants <- x$constants
  format_graph_body(
    inputs = x$inputs,
    constants = constants,
    calls = x$calls$as_list(),
    outputs = x$outputs,
    title = "GraphDescriptor"
  )
}

#' @export
print.GraphDescriptor <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}
