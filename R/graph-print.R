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
  shp <- shape(node$aval)
  sprintf("%s:%s%s", val, dt, if (length(shp)) sprintf("[%s]", shape2string(shp)) else "")
}

# `r_type` is the R storage type this value is uploaded from, out of the graph's
# `rdata_types`. Only the Inputs section has one to pass: the "<- <r type>" note
# says what the caller supplies and what the program uploads it as, which is a
# fact about the input. An output that happens to *be* an input
# (`jit(identity)`) is still just a value of its data type.
format_aval_short <- function(aval, r_type = NA_character_) {
  out <- sprintf("%s[%s]", repr(dtype(aval)), paste(shape(aval), collapse = ", "))
  if (!is.na(r_type)) {
    # An input the caller supplies as bare R data, which the program uploads at
    # the dtype shown -- worth seeing, since nothing else in the graph says so.
    return(paste0(out, " <- ", r_type))
  }
  out
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
  if (identical(param, list())) {
    return("")
  }
  if (test_scalar(param)) {
    as.character(param)
  } else if (is.atomic(param) && length(param) > 1L) {
    sprintf("c(%s)", paste(param, collapse = ", "))
  } else if (is_graph(param)) {
    sprintf("graph[%s -> %s]", length(param$inputs), length(param$outputs))
  } else if (is_dtype(param)) {
    repr(param)
  } else if (is.list(param)) {
    if (!is.null(names(param))) {
      sprintf("[%s]", paste(names(param), "=", sapply(param, format_param), collapse = ", "))
    } else {
      sprintf("[%s]", paste(sapply(param, format_param), collapse = ", "))
    }
  } else {
    x <- try(format(param), silent = TRUE)
    if (length(x) == 1L) {
      x
    } else {
      "<any>"
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

format_graph_body <- function(inputs, constants, calls, outputs, title = "Graph", rdata_types = NULL) {
  lines <- character()

  # Build node ID mapping
  node_ids <- build_node_ids(inputs, constants, calls)

  # Header
  lines <- c(lines, sprintf("<%s>", title))

  # Inputs section
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
    title = "AnvlGraph",
    rdata_types = x$rdata_types
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
    title = "GraphDescriptor",
    rdata_types = x$rdata_types
  )
}

#' @export
print.GraphDescriptor <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}
