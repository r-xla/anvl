test_that("trace_fn: simple test", {
  f <- function(x, y) {
    prim_add(x, y)
  }
  graph <- trace_fn(f, list(x = nv_scalar(1), y = nv_scalar(2)))
  expect_true(is_graph(graph))
  expect_list(graph$inputs, len = 2L, types = "GraphValue")
  expect_list(graph$calls, len = 1L, types = "PrimitiveCall")
  expect_list(graph$outputs, len = 1L, types = "GraphValue")
  expect_true(identical(graph$outputs, graph$calls[[1]]$outputs))
})

test_that("trace_fn: in- and outputs are reference identical to the outputs of the calls that produced them", {
  f <- function(x, y) {
    prim_add(x, y)
  }
  graph <- trace_fn(f, list(x = nv_scalar(1), y = nv_scalar(2)))
  expect_true(identical(graph$outputs, graph$calls[[1]]$outputs))
  expect_true(identical(graph$inputs, graph$calls[[1]]$inputs))
})

test_that("trace_fn: nested inputs and outputs", {
  f <- function(lst) {
    list(prim_add(lst[[1]], lst[[2]]))
  }

  graph <- trace_fn(f, list(lst = list(nv_scalar(1), nv_scalar(2))))
  expect_list(graph$inputs, len = 2L, types = "GraphValue")
  expect_list(graph$calls, len = 1L, types = "PrimitiveCall")
  expect_list(graph$outputs, len = 1L, types = "GraphValue")
  expect_equal(
    unflatten(graph$in_tree, list(1, 2)),
    list(lst = list(1, 2))
  )
  expect_equal(
    unflatten(graph$out_tree, 1),
    list(1)
  )
})

test_that("trace_fn: closed-over constants", {
  x <- nv_scalar(1)
  f <- function(y) {
    prim_add(x, y)
  }
  graph <- trace_fn(f, list(y = nv_scalar(2)))
  expect_list(graph$inputs, len = 1L, types = "GraphValue")
  expect_list(graph$calls, len = 1L, types = "PrimitiveCall")
  expect_list(graph$outputs, len = 1L, types = "GraphValue")

  # What do we expect here?
  # We want the resulting graph to have a constant and two inputs

  expect_true(is_graph_value(graph$calls[[1]]$inputs[[1]]))
  expect_true(is_graph_value(graph$calls[[1]]$inputs[[2]]))
  expect_true(identical(x, graph$constants[[1]]$aval$data))
  expect_equal(length(graph$constants), 1L)
})

test_that("trace_fn can deduplicate constants", {
  x <- nv_scalar(1)
  f <- function(y) {
    prim_add(x, x)
  }
  graph <- trace_fn(f, list(y = nv_scalar(2)))
  expect_equal(length(graph$constants), 1L)
  expect_identical(graph$constants[[1]]$aval$data, x)
})

test_that("trace_fn works without arguments", {
  # For this it is necessary to also box outputs in trace_fn()
  x <- nv_scalar(1)
  f <- function() {
    x
  }
  graph <- trace_fn(f, list())
  expect_equal(length(graph$inputs), 0L)
  expect_equal(length(graph$outputs), 1L)
  expect_identical(graph$outputs[[1]]$aval$data, x)
  expect_equal(length(graph$outputs), 1L)
  expect_equal(length(graph$calls), 0L)
})


test_that("local_descriptor creates a graph", {
  globals[["CURRENT_DESCRIPTOR"]] <- NULL
  g <- local_descriptor()
  expect_false(is.null(globals[["CURRENT_DESCRIPTOR"]]))
  expect_true(is_graph_descriptor(globals[["CURRENT_DESCRIPTOR"]]))
})

test_that("local_descriptor restores previous graph", {
  globals[["CURRENT_DESCRIPTOR"]] <- NULL
  g1 <- local_descriptor()
  inner_test <- function() {
    g2 <- local_descriptor()
    (function() local_descriptor())()
    expect_equal(.current_descriptor(), g2)
  }
  inner_test()
  expect_equal(g1, .current_descriptor())
})

test_that(".current_descriptor errors when no graph exists", {
  globals[["CURRENT_DESCRIPTOR"]] <- NULL
  expect_error(.current_descriptor(), "No graph is currently being built")
})

test_that("constants: same array is constant and input at the same time", {
  # Not sure what we want to happen here.
  f <- jit(function(x) {
    h <- function(x) x * y
    y <- nv_scalar(2)
    gradient(h)(y)
  })
  expect_equal(f(nv_scalar(1)), list(x = nv_scalar(2)))
})

test_that("closed-over constant is passed as argument to transformation", {
  x <- nv_scalar(1)
  f <- jit(function() {
    h <- function(y) y * y
    gradient(h)(x)
  })
  expect_equal(f(), list(y = nv_scalar(2)))
})

test_that("can pass constant to nested trace_fn call if it does not exist in the parent graph", {
  f <- jit(function() {
    g <- function(y) y * y
    gradient(g)(nv_scalar(2))
  })
  expect_equal(f(), list(y = nv_scalar(4)))
})

test_that("can pass constant to nested trace_fn call if it is defined in the parent graph", {
  f <- jit(function() {
    nv_add(y, y)
    g <- function(y) y * y
    gradient(g)(y)
  })
  y <- nv_scalar(2)
  expect_equal(f(), list(y = nv_scalar(4)))
})

test_that("GraphLiteral", {
  gl <- GraphLiteral(LiteralArray(1L, integer()))
  expect_equal(dtype(gl), as_dtype("i32"))
  expect_equal(shape(gl), integer())
  expect_snapshot(gl)
})

test_that("trace_fn works with nv_aval inputs", {
  f <- function(x, y) {
    prim_add(x, y)
  }
  in_type <- nv_aval("f32", c(2, 2))
  graph <- trace_fn(f, list(x = in_type, y = in_type))
  expect_true(is_graph(graph))
  expect_equal(graph$inputs[[1L]]$aval, in_type)
  expect_equal(graph$inputs[[2L]]$aval, in_type)
  expect_equal(length(graph$inputs), 2L)
  expect_equal(length(graph$calls), 1L)
  expect_equal(length(graph$outputs), 1L)
  expect_equal(graph$calls[[1L]]$primitive, attr(prim_add, "primitive"))
})

test_that("local_descriptor errors when run in the global environment", {
  expect_error(eval(quote(local_descriptor()), globalenv()), "Don't run local_descriptor in the global environment")
})

test_that("can pass abstract arrays to trace_fn", {
  # Here, its fine because we call into maybe_box_input, which will convert the abstract array
  # into a GraphValue/Box before any infix op can be called
  f <- function(x, y) {
    prim_add(x, y)
  }
  in_type <- nv_aval("f32", c(2, 2))
  graph <- trace_fn(f, list(x = in_type, y = in_type))
  expect_true(is_graph(graph))
  expect_equal(graph$inputs[[1L]]$aval, in_type)
  expect_equal(graph$inputs[[2L]]$aval, in_type)
  expect_equal(length(graph$inputs), 2L)
})

test_that("error handling", {
  expect_snapshot(error = TRUE, jit(prim_ceil)(nv_array(1:4)))
  expect_snapshot(
    error = TRUE,
    jit(prim_transpose, static = "permutation")(nv_array(1:4, shape = c(2, 2)), permutation = c(2, 2))
  )
})

test_that("error handling: stablehlo errors use anvl's terminology", {
  # `cli_abort()` errors from stablehlo store an already formatted message in
  # the condition's fields. Shapes rather than data types, since operands whose
  # data types disagree are refused by `promote_rdata_common()` before inference
  # ever sees them.
  expect_snapshot(error = TRUE, jit(prim_add)(nv_array(1:4), nv_array(1:6)))
  err <- tryCatch(jit(prim_add)(nv_array(1:4), nv_array(1:6)), error = identity)
  expect_false(grepl("tensor", conditionMessage(err), fixed = TRUE))

  # stablehlo's `operand` is anvl's `x`
  err <- tryCatch(jit(prim_ceil)(nv_array(1:4)), error = identity)
  expect_match(conditionMessage(err), "`x` must have dtype float", fixed = TRUE)

  # `ErrorStablehlo` conditions build their message lazily in a
  # `conditionMessage()` method; they keep their class and their 1-based indices.
  # A too-short `permutation` passes anvl's own checks (every entry is a valid,
  # non-duplicated dimension) and is only rejected by stablehlo.
  err <- tryCatch(
    jit(prim_transpose, static = "permutation")(nv_array(1:4, shape = c(2, 2)), permutation = 1L),
    error = identity
  )
  expect_s3_class(err, "ErrorPermuteIndex")
  expect_match(conditionMessage(err), "must be a permutation of c(1, 2)", fixed = TRUE)
})

test_that("user_terminology() rewrites words but not identifiers", {
  expect_equal(
    user_terminology("hlo_tensor() returns a TensorType; rank(operand) and operand_batching_dims are tensors"),
    "hlo_tensor() returns a TensorType; rank(x) and operand_batching_dims are arrays"
  )
})

test_that("can print GraphLiteral if it holds scalar array", {
  expect_snapshot(GraphLiteral(LiteralArray(nv_scalar(1L), dtype = "i32", shape = integer())))
})

test_that("trace_fn(mode = 'toplevel') errors when called inside an existing descriptor", {
  parent <- local_descriptor()
  expect_error(
    trace_fn(function(x) x, list(x = nv_scalar(1)), mode = "toplevel"),
    "must not have a parent descriptor"
  )
})

test_that("trace_fn(mode = 'subgraph') errors without a parent descriptor", {
  expect_error(
    trace_fn(function(x) x, list(x = nv_scalar(1)), mode = "subgraph"),
    "requires a parent descriptor"
  )
})

test_that("trace_fn(mode = 'inline') errors without a parent descriptor", {
  expect_error(
    trace_fn(function(x) x, list(x = nv_scalar(1)), mode = "inline"),
    "requires a parent descriptor"
  )
})

test_that("trace_fn(mode = 'toplevel') passes non-arrayish R values through as static args", {
  f <- function(x, flag) x
  graph <- trace_fn(f, list(x = nv_scalar(1), flag = TRUE))
  expect_equal(length(graph$inputs), 1L)
  expect_equal(graph$is_static_flat, c(FALSE, TRUE))
  expect_equal(graph$static_args_flat, list(TRUE))
})

test_that("trace_fn(mode = 'subgraph') promotes R lits/arrays to AnvlArray inputs", {
  parent <- local_descriptor()
  desc <- local_descriptor()
  graph <- trace_fn(
    function(x, y) list(x, y),
    list(x = 1, y = array(c(2, 3))),
    desc = desc,
    mode = "subgraph"
  )
  expect_equal(length(graph$inputs), 2L)
  expect_equal(shape(graph$inputs[[1L]]), integer())
  expect_equal(shape(graph$inputs[[2L]]), 2L)
})

test_that("trace_fn(mode = 'subgraph') errors on non-arrayish args", {
  parent <- local_descriptor()
  desc <- local_descriptor()
  expect_error(
    trace_fn(
      function(x) x,
      list(x = "string"),
      desc = desc,
      mode = "subgraph"
    ),
    "all args must be arrayish"
  )
})

# An R value written in the body of a traced function, and one passed as an
# argument, are built into the graph at the data type their use site needs.
# These snapshots pin *how* they are built -- an inlined literal for a scalar, a
# constant for an R array, a convert only out of the value's own category --
# which value-level tests cannot see.
describe("how an R value is built into a graph", {
  it("builds an R scalar as an inlined literal and an R array as a constant", {
    m <- matrix(c(1, 2, 3, 4), 2, 2)
    f <- function(x) nv_add(nv_mul(x, 2), nv_add(x, m))
    graph <- trace_fn(f, list(x = nv_aval("f32", c(2L, 2L))))
    expect_snapshot(graph)
  })

  it("builds a closed-over R array used twice as one constant", {
    m <- matrix(c(1, 2, 3, 4), 2, 2)
    f <- function(x) nv_add(nv_add(x, m), m)
    graph <- trace_fn(f, list(x = nv_aval("f32", c(2L, 2L))))
    expect_length(graph$constants, 1L)
    expect_snapshot(graph)
  })

  it("converts inside the program when the value crosses its category", {
    # An R double built at an integer data type is built at f64 -- where it is
    # exact -- and converted by the program, so narrowing follows XLA.
    f <- function(x) nv_add(x, nv_convert(1.5, "i32"))
    # The f64 the staging brings in is what `anvl_staging_widens_warning`
    # reports; here the point is the graph it produces.
    expect_warning(trace_fn(f, list(x = nv_aval("i32", integer()))))
    graph <- suppressWarnings(trace_fn(f, list(x = nv_aval("i32", integer()))))
    expect_snapshot(graph)
  })

  it("uploads an R argument used at two data types once and converts", {
    f <- function(x, y) nv_add(nv_convert(nv_mul(x, y), "f64"), nv_convert(y, "f64"))
    graph <- trace_fn(f, list(x = nv_aval("f32", integer()), y = nv_aval("double", integer())))
    expect_snapshot(graph)
  })

  it("inlines a gradient into the enclosing graph", {
    # The argument has to have a data type: `gradient()` refuses to
    # differentiate with respect to a value that has not decided on one.
    f <- function(t) gradient(function(z) z * z)(t)
    graph <- trace_fn(f, list(t = nv_aval("f64", integer())))
    expect_snapshot(graph)
  })

  it("builds an R literal inside the sub-graph body that uses it", {
    f <- jit(function(x) nv_if(nv_scalar(TRUE), function() nv_add(x, 0.5), function() nv_mul(x, 2)))
    expect_equal(as_array(f(nv_scalar(1, dtype = "f64"))), 1.5)
    g <- jit(function(x) nv_if(nv_scalar(FALSE), function() nv_add(x, 0.5), function() nv_mul(x, 2)))
    expect_equal(as_array(g(nv_scalar(1, dtype = "f64"))), 2)
  })

  it("keeps the loop's data type for an R value in a while body", {
    f <- jit(function(x) {
      nv_while(list(i = nv_scalar(0, dtype = "f64"), acc = x), function(i, acc) i < 3, function(i, acc) {
        list(i = i + 1, acc = acc * 2)
      })
    })
    out <- f(nv_scalar(1, dtype = "f64"))
    expect_equal(as_array(out$acc), 8)
  })
})
