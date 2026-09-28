#' @include graph.R

# A node's name, or -- for a literal that no call defines -- its value, which
# is what puts `2:i32` straight into the operand list instead of a name the
# reader would have to look up.
format_node_id <- function(node, node_ids, digits = getOption("digits")) {
  id <- node_ids[[node]]
  if (!is.null(id)) {
    return(sprintf("%%%s", id))
  }
  if (is_graph_literal(node)) {
    return(format_literal(node, digits))
  }
  "???"
}

format_literal <- function(node, digits = getOption("digits")) {
  val <- node$aval$data
  if (is_anvl_array(val)) {
    val <- as_array(val, check = FALSE)
  }
  sprintf(
    "%s:%s%s",
    format(val, trim = TRUE, digits = digits),
    as.character(dtype(node$aval)),
    format_shape_suffix(shape(node$aval))
  )
}

# The `[2,3]` a value repr carries after its data type. A scalar carries none,
# which is what tells `1:f32` apart from the one-element array `1:f32[1,1]`.
format_shape_suffix <- function(shp) {
  if (length(shp) == 0L) {
    return("")
  }
  sprintf("[%s]", shape2string(shp, parenthesize = FALSE))
}

# `r_type` is the R storage type this value is uploaded from, out of the graph's
# `rdata_types`. Only the Inputs section has one to pass: the "<- <r type>" note
# says what the caller supplies and what the program uploads it as, which is a
# fact about the input. An output that happens to *be* an input
# (`jit(identity)`) is still just a value of its data type.
format_aval_short <- function(aval, r_type = NA_character_) {
  out <- sprintf("%s[%s]", as.character(dtype(aval)), shape2string(shape(aval), parenthesize = FALSE))
  if (!is.na(r_type)) {
    # An input the caller supplies as bare R data, which the program uploads at
    # the dtype shown -- worth seeing, since nothing else in the graph says so.
    return(paste0(out, " <- ", r_type))
  }
  out
}

build_node_ids <- function(inputs, constants, statements) {
  node_ids <- hashtab()
  counters <- new.env(parent = emptyenv())
  counters$x <- 0L
  counters$c <- 0L
  counters$v <- 0L
  name_graph_nodes(inputs, constants, statements, node_ids, counters)
  node_ids
}

# Names the nodes of a graph and then, recursively, of its sub-graphs. One table
# covers the whole tree, so that no two nodes share a name; a node is named by
# the outermost graph that reaches it. The one exception is on purpose: a
# sub-graph's captured inputs take the names of the operands its statement
# passes for them, so `true = [%x3] ()` reads as the outer `%x3` it is.
name_graph_nodes <- function(inputs, constants, statements, node_ids, counters) {
  name_node <- function(node, counter, prefix) {
    if (!is.null(node_ids[[node]]) || is_graph_literal(node)) {
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
  for (call in statements) {
    for (node in call$outputs) {
      name_node(node, "v", "")
    }
  }
  # Sub-graphs come after the whole graph holding them, so that a graph's own
  # values are numbered without a gap where a higher-order statement sits.
  for (call in statements) {
    for (sub in Filter(is_graph, call$params)) {
      name_captured_inputs(sub, call, node_ids)
      name_graph_nodes(sub$inputs, sub$constants, sub$statements, node_ids, counters)
    }
  }
}

# The last `n_captures` inputs of `sub` line up with the last operands of
# `call`, which the graph holding it has already named. An operand that is a
# literal has no name to lend, so its input is numbered like any other.
name_captured_inputs <- function(sub, call, node_ids) {
  k <- sub$n_captures %||% 0L
  if (!k) {
    return(invisible(NULL))
  }
  own <- length(sub$inputs) - k
  operands <- utils::tail(call$inputs, k)
  for (j in seq_len(k)) {
    id <- if (!is_graph_literal(operands[[j]])) node_ids[[operands[[j]]]]
    if (!is.null(id)) {
      node_ids[[sub$inputs[[own + j]]]] <- id
    }
  }
}

# A single param value. Every type that reaches a param has a case here;
# anything else is named by its class rather than deparsed, so that an S3 list
# (`AnvlArray`) reads as `<AnvlArray>` instead of printing its internals,
# pointers and all.
format_param <- function(
  p,
  node_ids = NULL,
  width = getOption("width", 80L),
  digits = getOption("digits"),
  expand_graphs = TRUE,
  prefix_width = 0L
) {
  if (is.null(p)) {
    return("NULL")
  }
  if (is_graph(p)) {
    if (!expand_graphs) {
      return(format_graph_signature(p))
    }
    return(format_graph_param(p, node_ids, width, digits, prefix_width = prefix_width))
  }
  if (is_dtype(p)) {
    return(as.character(p))
  }
  if (is_anvl_array(p)) {
    return(format_array_param(p, digits))
  }
  if (is.atomic(p) && is.null(attr(p, "class"))) {
    if (length(p) == 0L) {
      return(sprintf("%s(0)", typeof(p)))
    }
    # Element by element, so that one wide element does not put the whole vector
    # into scientific notation; `digits` is how many of each are shown.
    elts <- if (is.character(p)) {
      sprintf('"%s"', p)
    } else {
      vapply(p, format, character(1L), trim = TRUE, digits = digits)
    }
    return(if (length(p) == 1L) elts else sprintf("c(%s)", paste(elts, collapse = ", ")))
  }
  if (is.list(p) && is.null(attr(p, "class"))) {
    parts <- format_param_parts(p, node_ids, width, digits, expand_graphs = expand_graphs)
    return(sprintf("list(%s)", paste(parts, collapse = ", ")))
  }
  sprintf("<%s>", class(p)[[1L]])
}

# The elements of a param list, each prefixed with `name = ` where it has a name.
format_param_parts <- function(
  params,
  node_ids = NULL,
  width = getOption("width", 80L),
  digits = getOption("digits"),
  expand_graphs = TRUE
) {
  nms <- names(params) %||% rep("", length(params))
  prefixes <- ifelse(nzchar(nms), paste0(nms, " = "), "")
  parts <- vapply(
    seq_along(params),
    function(i) {
      format_param(
        params[[i]],
        node_ids = node_ids,
        width = width,
        digits = digits,
        expand_graphs = expand_graphs,
        prefix_width = nchar(prefixes[[i]])
      )
    },
    character(1L)
  )
  paste0(prefixes, parts)
}

# An array param: a scalar shows its value the way a literal does, a larger
# array only its data type and shape.
format_array_param <- function(x, digits = getOption("digits")) {
  dt <- as.character(dtype(x))
  if (nelts(x) == 1L) {
    sprintf(
      "%s:%s%s",
      format(as_array(x, check = FALSE), trim = TRUE, digits = digits),
      dt,
      format_shape_suffix(shape(x))
    )
  } else {
    sprintf("%s[%s]", dt, shape2string(shape(x), parenthesize = FALSE))
  }
}

# A sub-graph as a single line: the data types it takes and returns. This is
# what a sub-graph comes to where the whole graph cannot go -- outside the graph
# holding it, its node names would refer to a table the reader never sees.
format_graph_signature <- function(g) {
  avals <- function(nodes) {
    paste(vapply(nodes, \(node) format_aval_short(node$aval), character(1L)), collapse = ", ")
  }
  k <- g$n_captures %||% 0L
  n_own <- length(g$inputs) - k
  captured <- if (k) sprintf("[%s] ", avals(g$inputs[n_own + seq_len(k)])) else ""
  sprintf("%s(%s) -> %s", captured, avals(g$inputs[seq_len(n_own)]), avals(g$outputs))
}

# A sub-graph param, printed in full. `node_ids` is the enclosing graph's table,
# so that the sub-graph's names do not clash with the graph around it; a graph
# formatted on its own gets a table of its own.
format_graph_param <- function(
  g,
  node_ids = NULL,
  width = getOption("width", 80L),
  digits = getOption("digits"),
  prefix_width = 0L
) {
  alone <- is.null(node_ids)
  node_ids <- node_ids %||% build_node_ids(g$inputs, g$constants, g$statements)
  lines <- format_graph_lines(
    inputs = g$inputs,
    constants = g$constants,
    statements = g$statements,
    outputs = g$outputs,
    node_ids = node_ids,
    width = width,
    digits = digits,
    typed_captures = alone,
    prefix_width = prefix_width,
    n_captures = g$n_captures
  )
  paste(lines, collapse = "\n")
}

# One comma-separated list inside a call line: its params, its operands, or the
# ids and types of a multi-output call. `open` / `close` are the delimiters that
# surround it, and `parts` the already-formatted elements. A list holding a
# multi-line part (a sub-graph) can never be laid out inline.
call_chunk <- function(open, close, parts) {
  multi <- any(grepl("\n", parts, fixed = TRUE))
  list(open = open, close = close, parts = parts, multi = multi, breakable = length(parts) > 0L)
}

inline_chunk <- function(chunk) {
  paste0(chunk$open, paste(chunk$parts, collapse = ", "), chunk$close)
}

# Packs `parts` into rows no wider than `width`, breaking only at the commas
# between them -- filling the rows rather than giving each part one of its own,
# so a call with thirty operands costs a few rows instead of thirty. A part that
# is itself multi-line (a sub-graph) takes a row alone, since nothing can share
# a row with a last line that is not the row's own.
fill_parts <- function(parts, width) {
  pieces <- paste0(parts, c(rep(",", length(parts) - 1L), ""))
  rows <- character()
  cur <- character()
  for (piece in pieces) {
    if (grepl("\n", piece, fixed = TRUE)) {
      rows <- c(rows, cur, piece)
      cur <- character()
    } else if (!length(cur)) {
      cur <- piece
    } else if (nchar(cur) + 1L + nchar(piece) <= width) {
      cur <- paste(cur, piece)
    } else {
      rows <- c(rows, cur)
      cur <- piece
    }
  }
  c(rows, cur)
}

# Lays out one row from its chunks -- plain strings kept verbatim, lists either
# inline or, where `broken` says so, opened at the end of the running line,
# filled at `indent + 2`, and closed on a line that the chunks after it continue.
render_row <- function(chunks, broken, indent, width) {
  inner <- paste0(indent, "  ")
  lines <- character()
  cur <- indent
  for (i in seq_along(chunks)) {
    chunk <- chunks[[i]]
    if (is.character(chunk)) {
      cur <- paste0(cur, chunk)
    } else if (!broken[[i]]) {
      cur <- paste0(cur, inline_chunk(chunk))
    } else {
      rows <- fill_parts(chunk$parts, width - nchar(inner))
      lines <- c(
        lines,
        paste0(cur, chunk$open),
        paste0(inner, gsub("\n", paste0("\n", inner), rows, fixed = TRUE))
      )
      cur <- paste0(indent, chunk$close)
    }
  }
  c(lines, cur)
}

# The width of the widest line the layout itself is answerable for. A line
# holding a sub-graph is left out: its inner lines were laid out against their
# own budget, and breaking anything here would not shorten them.
layout_width <- function(lines) {
  own <- lines[!grepl("\n", lines, fixed = TRUE)]
  if (!length(own)) 0L else max(nchar(own))
}

# A row of chunks laid out to `width`: everything on one line where it fits,
# otherwise the widest of its comma-separated lists broken into a filled block,
# and the next widest after that, until the row fits. A list holding a sub-graph
# always starts out broken. A row that is one unbreakable unit -- a single long
# param, a wide output type -- overflows `width`, since the only way to shorten
# it is to split a value.
layout_row <- function(chunks, indent, width) {
  flag <- function(name) vapply(chunks, \(ch) is.list(ch) && ch[[name]], logical(1L))
  breakable <- flag("breakable")
  broken <- flag("multi")
  repeat {
    lines <- render_row(chunks, broken, indent, width)
    todo <- which(breakable & !broken)
    if (layout_width(lines) <= width || !length(todo)) {
      break
    }
    widest <- vapply(chunks[todo], \(ch) nchar(inline_chunk(ch)), integer(1L))
    broken[[todo[[which.max(widest)]]]] <- TRUE
  }
  lines
}

# One call of a graph body: `%1: f32[3] = add [params] (operands)`.
format_call <- function(
  call,
  node_ids,
  indent = "  ",
  width = getOption("width", 80L),
  digits = getOption("digits")
) {
  input_ids <- vapply(call$inputs, format_node_id, character(1L), node_ids = node_ids, digits = digits)
  output_ids <- vapply(call$outputs, format_node_id, character(1L), node_ids = node_ids, digits = digits)
  output_types <- vapply(call$outputs, \(x) format_aval_short(x$aval), character(1L))

  chunks <- if (length(call$outputs) == 1L) {
    list(sprintf("%s: %s", output_ids, output_types))
  } else {
    list(call_chunk("(", ")", output_ids), ": ", call_chunk("(", ")", output_types))
  }
  chunks <- c(chunks, sprintf(" = %s", call$primitive$name))
  parts <- format_param_parts(call$params, node_ids, width = width - nchar(indent) - 2L, digits = digits)
  if (length(parts) > 0L) {
    chunks <- c(chunks, list(call_chunk(" [", "] ", parts)))
  }
  chunks <- c(chunks, list(call_chunk("(", ")", input_ids)))
  paste(layout_row(chunks, indent, width), collapse = "\n")
}

# A graph as `[captures] (inputs) { <body> return <outputs> }`, headed by
# `title` where it has one: what it closed over in brackets, its own inputs in
# parens with their data types. For a toplevel graph the captures are its
# constants; for a sub-graph they are its last `n_captures` inputs, named after
# the operands its statement passes for them.
#
# `typed_captures` spells a capture's data type too. Only a graph with nothing
# around it needs that -- a sub-graph's captures are nodes of the graph holding
# it, declared there, so naming them is enough.
format_graph_lines <- function(
  inputs,
  constants,
  statements,
  outputs,
  node_ids,
  title = "",
  rdata_types = NULL,
  width = getOption("width", 80L),
  digits = getOption("digits"),
  typed_captures = FALSE,
  prefix_width = 0L,
  n_captures = 0L
) {
  indent <- "  "
  r_types <- rdata_types %||% rep(NA_character_, length(inputs))
  input_strs <- vapply(
    seq_along(inputs),
    function(i) {
      node <- inputs[[i]]
      sprintf(
        "%s: %s",
        format_node_id(node, node_ids, digits),
        format_aval_short(node$aval, r_types[[i]])
      )
    },
    character(1L)
  )
  n_own <- length(inputs) - (n_captures %||% 0L)
  captured <- c(constants, inputs[n_own + seq_len(length(inputs) - n_own)])
  input_strs <- input_strs[seq_len(n_own)]
  capture_strs <- vapply(
    captured,
    function(node) {
      id <- format_node_id(node, node_ids, digits)
      if (typed_captures) sprintf("%s: %s", id, format_aval_short(node$aval)) else id
    },
    character(1L)
  )

  # Each piece of the signature is separated from the one before it by a space,
  # and the first of them starts the line. A graph that captures nothing has no
  # bracket list at all, and a sub-graph has no title before it.
  add <- function(header, open, close, parts) {
    open <- if (length(header)) paste0(" ", open) else open
    c(header, list(call_chunk(open, close, parts)))
  }
  header <- if (nzchar(title)) list(title) else list()
  if (length(captured) > 0L) {
    header <- add(header, "[", "]", capture_strs)
  }
  header <- c(add(header, "(", ")", input_strs), " {")

  output_ids <- vapply(outputs, format_node_id, character(1L), node_ids = node_ids, digits = digits)
  ret <- if (length(outputs) == 1L) {
    paste0(indent, "return ", output_ids)
  } else {
    layout_row(list("return", call_chunk(" (", ")", output_ids)), indent, width)
  }

  c(
    # Only the signature shares its line with whatever the graph is printed
    # behind -- a `cond = `, say -- so only its budget pays for it.
    layout_row(header, "", width - prefix_width),
    vapply(
      statements,
      format_call,
      character(1L),
      node_ids = node_ids,
      indent = indent,
      width = width,
      digits = digits
    ),
    ret,
    "}"
  )
}

# A whole graph, headed by the class that is printing it. Its captures are
# typed: nothing encloses it, so this is the only place they are declared.
format_graph_body <- function(
  inputs,
  constants,
  statements,
  outputs,
  title = "Graph",
  rdata_types = NULL,
  width = getOption("width", 80L),
  digits = getOption("digits")
) {
  node_ids <- build_node_ids(inputs, constants, statements)
  lines <- format_graph_lines(
    inputs = inputs,
    constants = constants,
    statements = statements,
    outputs = outputs,
    node_ids = node_ids,
    title = sprintf("<%s>", title),
    rdata_types = rdata_types,
    width = width,
    digits = digits,
    typed_captures = TRUE
  )
  paste(lines, collapse = "\n")
}

#' @export
format.GraphStatement <- function(x, ..., digits = getOption("digits")) {
  inputs <- paste(
    vapply(
      x$inputs,
      function(inp) {
        if (is_graph_literal(inp)) {
          format_literal(inp, digits)
        } else {
          format_aval_short(inp$aval)
        }
      },
      character(1L)
    ),
    collapse = ", "
  )
  outputs <- paste(vapply(x$outputs, \(out) format_aval_short(out$aval), character(1L)), collapse = ", ")
  params_str <- if (length(x$params) > 0L) {
    sprintf(
      " [%s]",
      paste(format_param_parts(x$params, digits = digits, expand_graphs = FALSE), collapse = ", ")
    )
  } else {
    ""
  }
  sprintf("%s(%s)%s -> %s", x$primitive$name, inputs, params_str, outputs)
}

#' @export
format.AnvlGraph <- function(x, ..., width = getOption("width", 80L), digits = getOption("digits")) {
  format_graph_body(
    inputs = x$inputs,
    constants = x$constants,
    statements = x$statements,
    outputs = x$outputs,
    title = "AnvlGraph",
    rdata_types = x$rdata_types,
    width = width,
    digits = digits
  )
}

#' @export
print.AnvlGraph <- function(x, ...) {
  cat(format(x, ...), "\n", sep = "")
  invisible(x)
}

#' @export
format.GraphDescriptor <- function(x, ..., width = getOption("width", 80L), digits = getOption("digits")) {
  # Convert hashtab constants to list
  constants <- x$constants
  format_graph_body(
    inputs = x$inputs,
    constants = constants,
    statements = x$statements$as_list(),
    outputs = x$outputs,
    title = "GraphDescriptor",
    rdata_types = x$rdata_types,
    width = width,
    digits = digits
  )
}

#' @export
print.GraphDescriptor <- function(x, ...) {
  cat(format(x, ...), "\n", sep = "")
  invisible(x)
}
