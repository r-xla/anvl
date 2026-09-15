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
  sprintf("%s:%s%s", val, as.character(dtype(node$aval)), format_shape_suffix(shape(node$aval)))
}

# The `[2, 3]` a value repr carries after its data type. A scalar carries none,
# which is what tells `1:f32` apart from the one-element array `1:f32[1, 1]`.
format_shape_suffix <- function(shp) {
  if (length(shp) == 0L) {
    return("")
  }
  sprintf("[%s]", paste(shp, collapse = ", "))
}

# `r_type` is the R storage type this value is uploaded from, out of the graph's
# `rdata_types`. Only the Inputs section has one to pass: the "<- <r type>" note
# says what the caller supplies and what the program uploads it as, which is a
# fact about the input. An output that happens to *be* an input
# (`jit(identity)`) is still just a value of its data type.
format_aval_short <- function(aval, r_type = NA_character_) {
  out <- sprintf("%s[%s]", as.character(dtype(aval)), paste(shape(aval), collapse = ", "))
  if (!is.na(r_type)) {
    # An input the caller supplies as bare R data, which the program uploads at
    # the dtype shown -- worth seeing, since nothing else in the graph says so.
    return(paste0(out, " <- ", r_type))
  }
  out
}

build_node_ids <- function(inputs, constants, calls) {
  node_ids <- hashtab()
  counters <- new.env(parent = emptyenv())
  counters$x <- 0L
  counters$c <- 0L
  counters$v <- 0L
  name_graph_nodes(inputs, constants, calls, node_ids, counters)
  node_ids
}

# Names the nodes of a graph and then, recursively, of its sub-graphs. One table
# covers the whole tree: a sub-graph's constants are nodes it captured from the
# graph around it, so letting them keep the name they already have is what shows
# the capture. Every node is named by the outermost graph that reaches it.
name_graph_nodes <- function(inputs, constants, calls, node_ids, counters) {
  name_node <- function(node, counter, prefix) {
    if (is_graph_literal(node) || !is.null(node_ids[[node]])) {
      return(invisible(NULL))
    }
    counters[[counter]] <- counters[[counter]] + 1L
    node_ids[[node]] <- paste0(prefix, counters[[counter]])
  }
  # don't use "i" for values, because i1 looks like a the boolean type
  for (node in inputs) {
    name_node(node, "x", "x")
  }
  for (node in constants) {
    name_node(node, "c", "c")
  }
  for (call in calls) {
    for (node in call$outputs) {
      name_node(node, "v", "")
    }
  }
  # Sub-graphs come after the whole graph holding them, so that a graph's own
  # values are numbered without a gap where a sub-graph call sits.
  for (call in calls) {
    for (sub in Filter(is_graph, call$params)) {
      name_graph_nodes(sub$inputs, sub$constants, sub$calls, node_ids, counters)
    }
  }
}

# A single param value. Every type that reaches a param has a case here;
# anything else is named by its class rather than deparsed, so that an S3 list
# (`AnvlArray`) reads as `<AnvlArray>` instead of printing its internals,
# pointers and all.
format_param <- function(p, node_ids = NULL, width = getOption("width", 80L)) {
  if (is.null(p)) {
    return("NULL")
  }
  if (is_graph(p)) {
    return(format_graph_param(p, node_ids, width))
  }
  if (is_dtype(p)) {
    return(as.character(p))
  }
  if (is_anvl_array(p)) {
    return(format_array_param(p))
  }
  if (is.atomic(p) && is.null(attr(p, "class"))) {
    if (length(p) == 0L) {
      return(sprintf("%s(0)", typeof(p)))
    }
    elts <- if (is.character(p)) sprintf('"%s"', p) else format(p, trim = TRUE)
    return(if (length(p) == 1L) elts else sprintf("c(%s)", paste(elts, collapse = ", ")))
  }
  if (is.list(p) && is.null(attr(p, "class"))) {
    return(sprintf("list(%s)", paste(format_param_parts(p, node_ids, width), collapse = ", ")))
  }
  sprintf("<%s>", class(p)[[1L]])
}

# The elements of a param list, each prefixed with `name = ` where it has a name.
format_param_parts <- function(params, node_ids = NULL, width = getOption("width", 80L)) {
  parts <- vapply(params, format_param, character(1), node_ids = node_ids, width = width)
  nms <- names(params)
  if (!is.null(nms)) {
    named <- nzchar(nms)
    parts[named] <- paste0(nms[named], " = ", parts[named])
  }
  parts
}

# An array param: a scalar shows its value the way a literal does, a larger
# array only its data type and shape.
format_array_param <- function(x) {
  dt <- as.character(dtype(x))
  if (nelts(x) == 1L) {
    sprintf("%s:%s%s", as_array(x), dt, format_shape_suffix(shape(x)))
  } else {
    sprintf("%s[%s]", dt, paste(shape(x), collapse = ", "))
  }
}

# A sub-graph param, printed in full -- its Inputs and Outputs sections are its
# signature. `node_ids` is the enclosing graph's table, which is what lets a
# captured node keep its outer name; a graph formatted on its own gets a table
# of its own.
format_graph_param <- function(g, node_ids = NULL, width = getOption("width", 80L)) {
  node_ids <- node_ids %||% build_node_ids(g$inputs, g$constants, g$calls)
  sections <- format_graph_sections(
    inputs = g$inputs,
    constants = g$constants,
    calls = g$calls,
    outputs = g$outputs,
    node_ids = node_ids,
    width = width,
    constants_label = "Captures"
  )
  paste(c("graph {", sections, "}"), collapse = "\n")
}

# A call line, wrapped to `width` by putting each param on its own line -- the
# param boundaries are the only place a break does not split a value in half. A
# param that is itself multi-line (a sub-graph) always forces the broken form.
# A line that is one unbreakable unit -- a single long param, a wide output
# type -- overflows `width`, since the only way to shorten it is to split a
# value.
format_call <- function(call, node_ids, indent = "  ", width = getOption("width", 80L)) {
  input_ids <- vapply(call$inputs, format_node_id, character(1), node_ids = node_ids)
  inputs_str <- sprintf("(%s)", paste(input_ids, collapse = ", "))

  output_ids <- vapply(call$outputs, format_node_id, character(1), node_ids = node_ids)
  output_types <- vapply(call$outputs, \(x) format_aval_short(x$aval), character(1))

  outputs_str <- if (length(call$outputs) == 1L) {
    sprintf("%s: %s", output_ids, output_types)
  } else {
    sprintf("(%s): (%s)", paste(output_ids, collapse = ", "), paste(output_types, collapse = ", "))
  }

  header <- sprintf("%s%s = %s", indent, outputs_str, call$primitive$name)
  part_indent <- paste0(indent, "  ")
  parts <- format_param_parts(call$params, node_ids, width = width - nchar(part_indent))
  if (length(parts) == 0L) {
    return(paste0(header, inputs_str))
  }
  one_line <- sprintf("%s [%s] %s", header, paste(parts, collapse = ", "), inputs_str)
  if (!any(grepl("\n", parts, fixed = TRUE)) && nchar(one_line) <= width) {
    return(one_line)
  }
  commas <- c(rep(",", length(parts) - 1L), "")
  paste(
    c(
      paste0(header, " ["),
      paste0(part_indent, gsub("\n", paste0("\n", part_indent), parts, fixed = TRUE), commas),
      sprintf("%s] %s", indent, inputs_str)
    ),
    collapse = "\n"
  )
}

# The Inputs / Constants / Body / Outputs sections of a graph, without the
# header naming it. `constants_label` is "Captures" for a sub-graph, whose
# constants are nodes of the graph around it.
format_graph_sections <- function(
  inputs,
  constants,
  calls,
  outputs,
  node_ids,
  rdata_types = NULL,
  width = getOption("width", 80L),
  constants_label = "Constants"
) {
  lines <- character()

  if (length(inputs) > 0L) {
    r_types <- rdata_types %||% rep(NA_character_, length(inputs))
    input_strs <- vapply(
      seq_along(inputs),
      function(i) {
        node <- inputs[[i]]
        sprintf(
          "    %s: %s",
          format_node_id(node, node_ids),
          format_aval_short(node$aval, r_types[[i]])
        )
      },
      character(1)
    )
    lines <- c(lines, "  Inputs:", input_strs)
  } else {
    lines <- c(lines, "  Inputs: (none)")
  }

  if (length(constants) > 0L) {
    const_strs <- vapply(
      constants,
      function(node) {
        sprintf("    %s: %s", format_node_id(node, node_ids), format_aval_short(node$aval))
      },
      character(1)
    )
    lines <- c(lines, sprintf("  %s:", constants_label), const_strs)
  }

  if (length(calls) > 0L) {
    lines <- c(lines, "  Body:")
    for (call in calls) {
      lines <- c(lines, format_call(call, node_ids, indent = "    ", width = width))
    }
  } else {
    lines <- c(lines, "  Body: (empty)")
  }

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

  lines
}

format_graph_body <- function(
  inputs,
  constants,
  calls,
  outputs,
  title = "Graph",
  rdata_types = NULL,
  width = getOption("width", 80L)
) {
  node_ids <- build_node_ids(inputs, constants, calls)
  sections <- format_graph_sections(
    inputs = inputs,
    constants = constants,
    calls = calls,
    outputs = outputs,
    node_ids = node_ids,
    rdata_types = rdata_types,
    width = width
  )
  paste(c(sprintf("<%s>", title), sections), collapse = "\n")
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
  params_str <- if (length(x$params) > 0L) {
    sprintf(" [%s]", paste(format_param_parts(x$params), collapse = ", "))
  } else {
    ""
  }
  sprintf("%s(%s)%s -> %s", x$primitive$name, inputs, params_str, outputs)
}

#' @export
format.AnvlGraph <- function(x, ..., width = getOption("width", 80L)) {
  format_graph_body(
    inputs = x$inputs,
    constants = x$constants,
    calls = x$calls,
    outputs = x$outputs,
    title = "AnvlGraph",
    rdata_types = x$rdata_types,
    width = width
  )
}

#' @export
print.AnvlGraph <- function(x, ...) {
  cat(format(x, ...), "\n", sep = "")
  invisible(x)
}

#' @export
format.GraphDescriptor <- function(x, ..., width = getOption("width", 80L)) {
  # Convert hashtab constants to list
  constants <- x$constants
  format_graph_body(
    inputs = x$inputs,
    constants = constants,
    calls = x$calls$as_list(),
    outputs = x$outputs,
    title = "GraphDescriptor",
    rdata_types = x$rdata_types,
    width = width
  )
}

#' @export
print.GraphDescriptor <- function(x, ...) {
  cat(format(x, ...), "\n", sep = "")
  invisible(x)
}
