# One graph exercising most of the printer at once: a literal, two constants,
# calls with params, an `if` sub-graph inside a `while` sub-graph, captures of
# an input (`%x1`), a value (`%2`) and a constant (`%c1`), and calls on both
# sides of the sub-graph call.
nested_graph <- function() {
  half <- nv_scalar(0.5, dtype = "f32")
  f <- function(x) {
    step <- x * 2L
    y <- nv_while(
      list(i = nv_scalar(0, dtype = "f32")),
      \(i) i < x,
      \(i) list(i = nv_if(i < step, \() i + step, \() i + half))
    )
    nv_broadcast_to(y$i, c(2, 1))
  }
  trace_fn(f, list(x = nv_scalar(3, dtype = "f32")))
}

# `nested_graph()`'s sub-graph calls carry no params, so this one puts a
# param-bearing call one level down, where wrapping has to account for the
# enclosing indent. Used without a snapshot.
nested_param_graph <- function() {
  f <- function(x) {
    nv_while(
      list(i = nv_scalar(0, dtype = "f32")),
      \(i) i < x,
      \(i) list(i = nv_reduce_max(nv_broadcast_to(i, c(2, 1)), axes = 1, drop = TRUE)[1])
    )
  }
  trace_fn(f, list(x = nv_scalar(3, dtype = "f32")))
}

# A graph whose `gather` call carries more params than fit on a line.
gather_graph <- function() {
  trace_fn(
    function(x, i) x[i, ],
    list(x = nv_array(matrix(1:12, nrow = 3), dtype = "f32"), i = nv_array(c(1L, 3L)))
  )
}

# `n` arrays into one call, so that the operand list -- and, for `sort`, the
# ids and types of the outputs -- outgrow a line on their own.
wide_graph <- function(f, n = 30L) {
  args <- rep_len(list(nv_array(as.numeric(1:3), dtype = "f32")), n)
  trace_fn(f, setNames(args, paste0("a", seq_len(n))))
}

# A `while` whose body captures six values, so that the sub-graph's signature
# line is long enough to feel the `body_graph = ` it is printed behind.
capture_heavy_graph <- function() {
  consts <- lapply(1:6, function(i) nv_scalar(i, dtype = "f32"))
  trace_fn(
    function(x) nv_while(list(i = x), \(i) i < 99, \(i) list(i = Reduce(`+`, consts, i))),
    list(x = nv_scalar(1, dtype = "f32"))
  )
}

# The body of a formatted graph -- everything between its signature line and its
# `return` -- so that a snapshot of one call's layout is not buried under thirty
# lines of signature.
body_section <- function(graph, width) {
  lines <- strsplit(format(graph, width = width), "\n")[[1L]]
  lines[seq(match(") {", lines) + 1L, grep("^  return ", lines) - 1L)]
}

describe("format_node_id()", {
  it("marks a node the table does not name, rather than failing", {
    graph <- trace_fn(function(x) x + 1, list(x = nv_scalar(1, dtype = "f32")))
    expect_equal(format_node_id(graph$inputs[[1L]], hashtab()), "???")
  })
})

describe("format_param()", {
  it("prints NULL and atomic vectors in R syntax", {
    expect_snapshot({
      format_param(NULL)
      format_param(1L)
      format_param(1.5)
      format_param(TRUE)
      format_param("abc")
      format_param(c(1L, 2L, 3L))
      format_param(c(1, 1e6))
      format_param(0.1234567890123)
      format_param(0.1234567890123, digits = 17)
      format_param(c("a", "b"))
      format_param(c(TRUE, FALSE))
      format_param(integer())
      format_param(character())
    })
  })

  it("prints lists as list() calls", {
    expect_snapshot({
      format_param(list())
      format_param(list(1, 2))
      format_param(list(a = 1, b = 2))
      format_param(list(1, b = 2))
      format_param(list(a = NULL, b = 1))
      format_param(list(a = list(b = c(2, 3, 4))))
    })
  })

  it("prints a data type under its anvl name", {
    expect_snapshot({
      format_param(as_dtype("f32"))
      format_param(as_dtype("bool"))
    })
  })

  it("shows an array's value only when it holds one element", {
    expect_snapshot({
      format_param(nv_scalar(2.5, dtype = "f32"))
      format_param(nv_scalar(TRUE))
      format_param(nv_array(array(1.5, dim = c(1, 1)), dtype = "f32"))
      format_param(nv_array(matrix(1:6, nrow = 2), dtype = "i32"))
    })
  })

  it("prints a graph in full", {
    local_registered_default_dtypes()
    g <- trace_fn(function(x, y) x + y, list(x = nv_scalar(0, dtype = "f32"), y = nv_array(1:3)))
    expect_snapshot(cat(format_param(g)))
  })

  it("names an unknown object by its class instead of deparsing it", {
    expect_snapshot({
      format_param(function(x) x)
      format_param(new.env())
      format_param(quote(x + y))
      format_param(structure(list(secret = "hidden"), class = "SomeObject"))
    })
  })
})

describe("format_param_parts()", {
  it("prefixes the elements that have a name", {
    expect_snapshot({
      format_param_parts(list(axis = 1, drop = TRUE))
      format_param_parts(list(1, 2))
      format_param_parts(list(1, drop = TRUE))
    })
  })
})

describe("format.PrimitiveCall()", {
  it("renders its params the way a graph body does", {
    local_registered_default_dtypes()
    graph <- trace_fn(function(x) nv_reduce_max(x, axes = 1, drop = TRUE), list(x = nv_array(1:10)))
    expect_snapshot(cat(format(graph$calls[[1L]])))
  })

  it("leaves out the bracket list of a call that carries no params", {
    graph <- trace_fn(
      function(x, y) x + y,
      list(x = nv_scalar(1, dtype = "f32"), y = nv_scalar(2, dtype = "f32"))
    )
    expect_snapshot(cat(format(graph$calls[[1L]])))
  })

  it("keeps a sub-graph param to its signature, having no graph to name it against", {
    call <- Filter(\(cl) cl$primitive$name == "while", nested_graph()$calls)[[1L]]
    expect_snapshot(cat(format(call)))
  })
})

describe("format.AnvlGraph()", {
  it("shows literals, constants, params, captures and nested sub-graphs", {
    local_registered_default_dtypes()
    expect_snapshot(nested_graph())
  })

  it("gives every node in the tree exactly one name", {
    graph <- nested_graph()
    ids <- unlist(hashvalues(build_node_ids(graph$inputs, graph$constants, graph$calls)))
    expect_equal(anyDuplicated(ids), 0L)
  })

  it("numbers a graph's own values without a gap where a sub-graph call sits", {
    lines <- strsplit(format(nested_graph()), "\n")[[1L]]
    # The outer body is the only one indented by exactly two spaces.
    defs <- grep("^  %[0-9]+: .+ = ", lines, value = TRUE)
    expect_equal(sub("^  %([0-9]+):.*", "\\1", defs), c("1", "2", "3", "4"))
  })

  it("keeps a captured node's outer name inside a sub-graph", {
    lines <- strsplit(format(nested_graph()), "\n")[[1L]]
    # `step` is `%2` in the outer body; the loop body and both `if` branches
    # capture that same node, so it appears in three of the four capture lists.
    heads <- grep("^ +[a-z_]+ = \\[", lines, value = TRUE)
    captures <- sub("\\].*$", "", sub("^[^[]*\\[", "", heads))
    expect_equal(sum(grepl("%2", captures, fixed = TRUE)), 3L)
  })

  it("names a capture without its data type, the enclosing graph having it", {
    lines <- strsplit(format(nested_graph()), "\n")[[1L]]
    expect_true(any(grepl("cond_graph = [%x1] (%x2: f32[]) {", lines, fixed = TRUE)))
    # The graph around it is the one place `%x1` is declared with a type.
    expect_match(lines[[1L]], "(%x1: f32[])", fixed = TRUE)
  })

  it("leaves out the bracket list of a graph that captures nothing", {
    graph <- trace_fn(
      function(x) nv_while(list(i = x), \(i) i < 9, \(i) list(i = i + 1)),
      list(x = nv_scalar(2, dtype = "f32"))
    )
    expect_snapshot(graph)
  })

  it("has nothing between the signature and the return when there are no calls", {
    expect_snapshot(trace_fn(identity, list(x = nv_scalar(1, dtype = "f32"))))
  })

  it("gives a sub-graph param its own rows and fills the short ones around it", {
    graph <- trace_fn(
      function(x) prim_reduce(x, init = 0, axes = 1L, reductor = \(a, b) a + b),
      list(x = nv_array(as.numeric(1:6), dtype = "f32"))
    )
    expect_snapshot(cat(format(graph, width = 80L)))
  })

  it("returns every output of a graph that has more than one", {
    graph <- trace_fn(
      # The out tree's names do not survive into the graph, so the list is bare.
      function(x) list(x + 1, x * 2),
      list(x = nv_array(c(1, 2), dtype = "f32"))
    )
    expect_snapshot(graph)
  })

  it("does not spill the internals of an array an optimization pass inlined", {
    out <- format(inline_scalarish_constants(nested_graph()))
    expect_match(out, "value = 0.5:f32", fixed = TRUE)
    expect_no_match(out, "pointer", fixed = TRUE)
  })

  it("names the fill an optimization pass makes of a constant", {
    seven <- nv_scalar(7, dtype = "f32")
    graph <- trace_fn(
      function(x) x + seven,
      list(x = nv_scalar(2, dtype = "f32")),
      optimize = TRUE
    )
    expect_snapshot(graph)
  })

  it("tells apart two constants of equal value that the pass inlined", {
    # Two nodes, so two `fill` calls; without a name each they print alike.
    # Which of them the pass emits first is not fixed, so only compare the two.
    one <- nv_scalar(1, dtype = "f32")
    other <- nv_scalar(1, dtype = "f32")
    graph <- trace_fn(
      function(x) (x + one) * other,
      list(x = nv_scalar(2, dtype = "f32")),
      optimize = TRUE
    )
    fills <- grep("= fill ", strsplit(format(graph), "\n")[[1L]], value = TRUE)
    expect_length(fills, 2L)
    expect_equal(anyDuplicated(fills), 0L)
  })

  it("names the R type of an input the caller supplies as bare R data", {
    graph <- trace_fn(
      function(x, y, z) x + y + z,
      list(
        x = nv_scalar(1, dtype = "f64"),
        y = nv_aval("double", integer()),
        z = nv_aval("integer", 2L)
      )
    )
    expect_snapshot(graph)
  })

  it("shows a value to `digits` significant digits, seven of them by default", {
    graph <- trace_fn(function(x) x / sqrt(2), list(x = nv_array(1, dtype = "f64")))
    expect_match(format(graph), "1.414214:f64", fixed = TRUE)
    expect_match(format(graph, digits = 17), "1.4142135623730951:f64", fixed = TRUE)
  })

  it("spells a data type the anvl way, not the MLIR way", {
    graph <- trace_fn(function(x) x, list(x = nv_aval("bool", 2L)))
    expect_match(format(graph), "bool[2]", fixed = TRUE)
    expect_no_match(format(graph), "i1", fixed = TRUE)
  })

  it("breaks a call line too long for the width into filled param rows", {
    local_registered_default_dtypes()
    expect_snapshot(cat(format(gather_graph(), width = 80L)))
  })

  it("keeps a call line that fits within the width on one line", {
    expect_match(format(gather_graph(), width = 300L), "= gather [slice_sizes", fixed = TRUE)
  })

  it("breaks a call's operand list, which is as much a list as its params", {
    graph <- wide_graph(\(...) nv_concatenate(..., axis = 1L))
    expect_true(all(nchar(strsplit(format(graph, width = 80L), "\n")[[1L]]) <= 80L))
    expect_snapshot(cat(body_section(graph, width = 80L), sep = "\n"))
  })

  it("breaks the ids and the types of a call with more outputs than fit", {
    graph <- wide_graph(\(...) prim_sort(list(...), axis = 1L), n = 12L)
    expect_snapshot(cat(body_section(graph, width = 80L), sep = "\n"))
  })

  it("breaks a list of one element, which is a list like any other", {
    graph <- trace_fn(
      function(x) nv_reshape(x, c(3, 2)),
      list(x = nv_array(matrix(1:6, nrow = 2), dtype = "f32"))
    )
    expect_snapshot(cat(format(graph, width = 40L)))
  })

  it("shrinks the width budget with nesting, so no line exceeds it", {
    # The `name = ` a sub-graph param carries comes out of its budget too, so
    # the graph here captures enough to overflow a narrow width without it.
    graph <- capture_heavy_graph()
    widths <- c(40L, 60L, 80L, 120L)
    longest <- vapply(
      widths,
      function(width) max(nchar(strsplit(format(graph, width = width), "\n")[[1L]])),
      integer(1)
    )
    expect_equal(widths[longest > widths], integer(0))
  })

  it("spends a nested line's whole width budget, not a conservative part of it", {
    graph <- nested_param_graph()
    target <- "broadcast_axes = integer(0)] (%x3)"
    fits <- function(width) {
      any(grepl(target, strsplit(format(graph, width = width), "\n")[[1L]], fixed = TRUE))
    }
    # The enclosing indent is subtracted exactly once per level, so the line
    # fits at exactly the width it occupies and breaks one character below it.
    n <- nchar(grep(target, strsplit(format(graph, width = 10000L), "\n")[[1L]], fixed = TRUE, value = TRUE))
    expect_length(n, 1L)
    expect_true(fits(n))
    expect_false(fits(n - 1L))
  })

  it("overflows the width rather than splitting a value that cannot be broken", {
    lines <- strsplit(format(gather_graph(), width = 30L), "\n")[[1L]]
    expect_true(any(nchar(lines) > 30L))
    # The param that overruns stays whole on its line.
    expect_true(any(grepl("start_indices_batching_axes = integer(0)", lines, fixed = TRUE)))
  })
})

describe("format.GraphDescriptor()", {
  it("prints the graph a trace has built so far", {
    descriptor <- NULL
    trace_fn(
      function(x) {
        descriptor <<- .current_descriptor()
        x + 1
      },
      list(x = nv_scalar(1, dtype = "f32"))
    )
    expect_snapshot(descriptor)
  })
})
