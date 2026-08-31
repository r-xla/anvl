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
  # the condition's fields
  expect_snapshot(error = TRUE, jit(prim_add)(nv_array(1:4), nv_array(c(1, 2, 3, 4))))
  err <- tryCatch(jit(prim_add)(nv_array(1:4), nv_array(c(1, 2, 3, 4))), error = identity)
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


# An R value entering a program has no data type of its own: it is built into the
# program at the data type its use site needs, from the R data itself. These
# tests assert concrete numbers rather than comparing against another framework,
# so a missing optional dependency cannot silently skip a precision check.

describe("RData", {
  it("reports the shape the R value can answer for", {
    x <- RData(integer(), "double")
    expect_equal(shape(x), integer())
    expect_error(dtype(x), "no data type of its own")

    y <- RData(c(2L, 3L), "integer")
    expect_equal(shape(y), c(2L, 3L))
    expect_equal(naxes(y), 2L)
  })

  it("answers the extractors the same way for a bare R value", {
    # Eagerly the R value *is* the uncommitted value, so it has to answer the way
    # the boxed one does under jit().
    expect_equal(shape(1.5), integer())
    expect_equal(naxes(1.5), 0L)
    expect_equal(shape(TRUE), integer())
    expect_equal(shape(array(1:6, c(2, 3))), c(2L, 3L))
    expect_error(dtype(1.5), "no data type of its own")
    expect_error(dtype(1L), "no data type of its own")
    expect_error(dtype(TRUE), "no data type of its own")
    expect_error(dtype(array(1:6, c(2, 3))), "no data type of its own")
    expect_equal(peek_dtype(1.5), as_dtype("f32"))
    # A vector that is not an anvl value at all says so rather than lying.
    expect_error(shape(c(1, 2, 3)), "undefined for a length-3")
  })

  it("has no data type to report", {
    expect_error(jit(function(x) dtype(x))(1), "no data type of its own")
    # The message names the way out.
    expect_error(jit(function(x) dtype(x))(1), "nv_array")
    expect_error(jit(function(x) x + dtype(x))(array(1:4)), "no data type of its own")
    # `nv_*_like` derives the result's dtype from its argument, so it errors too.
    expect_error(jit(function(x) nv_fill_like(x, 3))(1), "no data type of its own")
    # Giving it a dtype answers the question.
    expect_equal(dtype(jit(function(x) nv_fill_like(x, 3))(nv_scalar(1))), as_dtype("f32"))
  })

  it("does have a shape inside a traced function", {
    seen <- list()
    f <- jit(function(x) {
      seen[["shape"]] <<- shape(x)
      seen[["naxes"]] <<- naxes(x)
      x + 1
    })
    invisible(f(array(1:6, c(2, 3))))
    expect_equal(seen$shape, c(2L, 3L))
    expect_equal(seen$naxes, 2L)

    seen <- list()
    invisible(f(1))
    expect_equal(seen$shape, integer())
    expect_equal(seen$naxes, 0L)
  })

  it("does not print the R data it carries", {
    # The data can be a whole array, and what matters about the value is what it
    # is, not what it holds.
    expect_identical(format(RData(integer(), "double")), "RData(double, ())")
    expect_identical(
      format(RData(c(2L, 3L), "integer")),
      "RData(integer, (2,3))"
    )
    # An argument of a jitted function has no data to print in the first place.
    expect_identical(
      format(RData(integer(), "logical")),
      "RData(logical, ())"
    )
    expect_identical(repr(RData(c(2L, 3L), "double")), "double[2x3]")
  })

  it("has no abstract value for an R vector that is not arrayish", {
    expect_error(to_abstract(c(1, 2, 3)), "undefined for a length-3")
    expect_equal(shape(to_abstract(array(1:6, c(2, 3)))), c(2L, 3L))
  })
})

describe("RDataInput", {
  it("names both data types it stands between", {
    x <- RDataInput("f64", c(2L, 3L), "double")
    expect_identical(format(x), "RDataInput(f64, double, (2,3))")
    expect_identical(repr(x), "f64[2x3]<-double")
  })

  it("records on the input itself which data type it is uploaded at", {
    graph <- trace_fn(
      function(x, y) x + y,
      list(x = nv_aval("f64", 2L), y = nv_aval("double", integer()))
    )
    # It lives on the input's own aval, not beside it.
    expect_s3_class(graph$inputs[[1L]]$aval, "AbstractArray")
    expect_false(is_rdata_input(graph$inputs[[1L]]$aval))
    expect_s3_class(graph$inputs[[2L]]$aval, "RDataInput")
    expect_equal(dtype(graph$inputs[[2L]]$aval), as_dtype("f64"))
    expect_equal(graph$inputs[[2L]]$aval$r_type, "double")
    # ... and the vector the backends read is derived from it.
    expect_equal(graph_input_dtypes(graph), c(NA, "f64"))
    # No R input at all means nothing to upload, and the backends skip the step.
    plain <- trace_fn(function(x) x + 1, list(x = nv_aval("f64", 2L)))
    expect_null(graph_input_dtypes(plain))
  })
})

describe("peek_dtype", {
  it("answers for an R value where dtype() does not", {
    # The API needs "what would this commit to" without forcing a commitment.
    seen <- NULL
    invisible(jit(function(x) {
      seen <<- peek_dtype(x)
      x + nv_scalar(1, dtype = "f64")
    })(sqrt(2)))
    expect_equal(seen, as_dtype("f32"))
    expect_equal(peek_dtype(1.5), as_dtype("f32"))
    expect_equal(peek_dtype(1L), as_dtype("i32"))
  })

  it("means the same thing eagerly and under jit()", {
    expect_eager_jit_equal_grid(list(
      "convert without canonicalizing first" = function(x, v) {
        nv_convert(v, peek_dtype(x)) * x
      },
      "peek_dtype() branch" = function(x, v) {
        if (is_dtype_float(peek_dtype(v))) x * v else x + v
      }
    ))
  })
})

describe("resolve_upload_dtype", {
  it("uploads an R argument used at two data types at one that holds both", {
    # The whole-program counterpart of the unit test below, and the only branch of
    # `finalize_rdata_inputs()` where the upload dtype is one no use site asked
    # for: `f16` and `bf16` are both 16 bits and neither holds the other, so the
    # input is uploaded at `f32` and each use site converts down from it.
    # (Asserted on the graph rather than by running it: this backend has no 16-bit
    # float buffers to run it with.)
    graph <- trace_fn(
      function(a, b, v) list(a * v, b * v),
      list(a = nv_aval("f16", integer()), b = nv_aval("bf16", integer()), v = nv_aval("double", integer()))
    )
    expect_equal(graph_input_dtypes(graph), c(NA, NA, "f32"))
    expect_equal(repr(graph$inputs[[3L]]$aval), "f32[]<-double")
    converts <- Filter(function(call) call$primitive$name == "convert", graph$calls)
    expect_length(converts, 2L)
    expect_setequal(
      vapply(converts, function(call) as.character(call$outputs[[1L]]$aval$dtype), character(1L)),
      c("f16", "bf16")
    )
    # Both converts read the input itself, rather than one of them converting the
    # other's result.
    expect_true(all(vapply(converts, function(call) identical(call$inputs[[1L]], graph$inputs[[3L]]), logical(1L))))
  })

  it("picks the narrowest data type that holds every use site", {
    dbl <- RData(integer(), "double")
    # `f16` and `bf16` are both 16 bits and neither holds the other, so the upload
    # has to widen -- to `f32`, which holds both, and not to the value's natural
    # `f64`: a program with no `f64` in it must not acquire one here.
    expect_equal(resolve_upload_dtype(dbl, c("f16", "bf16")), "f32")
    # Where one of them does hold the others, that one is used unchanged.
    expect_equal(resolve_upload_dtype(dbl, c("f32", "f64")), "f64")
    expect_equal(resolve_upload_dtype(dbl, "f16"), "f16")
    # A value the body never used commits to its default.
    expect_equal(resolve_upload_dtype(dbl, character()), "f32")
    expect_equal(resolve_upload_dtype(RData(integer(), "integer"), c("i32", "i64")), "i64")
  })
})

describe("an R value at its use site", {
  it("reaches an f64 use site exactly in the body of a jitted function", {
    x <- nv_scalar(1, dtype = "f64")
    expect_identical(as_array(jit(function(x) x / sqrt(2))(x)), 1 / sqrt(2))
    expect_identical(as_array(jit(function(x) x * pi)(x)), pi)
    expect_identical(as_array(jit(function(x) x + 0.1)(x)), 1.1)
    expect_identical(as_array(jit(function(x) x - 0.1)(x)), 0.9)
  })

  it("reaches an f64 use site exactly in eager mode too", {
    # Every nv_* function is jit-wrapped, so here the literal is an *argument* of
    # the call and is uploaded at the dtype the program decided on.
    x <- nv_scalar(1, dtype = "f64")
    expect_identical(as_array(x / sqrt(2)), 1 / sqrt(2))
    expect_identical(as_array(x * pi), pi)
    expect_identical(as_array(x + 0.1), 1.1)
    expect_identical(as_array(x - 0.1), 0.9)
    # ... and with the R value on the left.
    expect_identical(as_array(sqrt(2) / x), sqrt(2))
  })

  it("reaches f64 exactly when passed as a jit argument", {
    f <- jit(function(t) nv_scalar(-1, dtype = "f64") / t)
    expect_identical(as_array(f(sqrt(2))), -1 / sqrt(2))
  })

  it("is exact for the constants written in an nv_* function's body", {
    # nv_log2()/nv_log10() divide by `log(2)` / `log(10)` written as R literals.
    x <- nv_array(c(1, 2, 4, 8), dtype = "f64")
    expect_equal(as.vector(nv_log2(x)), log2(c(1, 2, 4, 8)), tolerance = 1e-15)
    y <- nv_array(c(1, 10, 100, 1000), dtype = "f64")
    expect_equal(as.vector(nv_log10(y)), log10(c(1, 10, 100, 1000)), tolerance = 1e-15)
  })

  it("reaches an f64 use site exactly as an R array", {
    x <- nv_array(c(1, 1), dtype = "f64")
    expect_identical(as.vector(x * array(c(0.1, sqrt(2)))), c(0.1, sqrt(2)))
  })

  it("is uploaded in its own R category", {
    # A logical can only be handed to the runtime as a logical, an integer as an
    # integer: the upload stays in the value's category and the program converts
    # out of it. Getting this wrong makes the upload itself fail.
    x32 <- nv_array(c(1, 2), dtype = "f32")
    expect_equal(as.vector(nv_mul(x32, TRUE)), c(1, 2))
    expect_equal(as.vector(nv_add(TRUE, nv_array(1:2))), c(2L, 3L))
    expect_equal(as.vector(nv_add(TRUE, nv_scalar(1, dtype = "f64"))), 2)
    # an R logical array as a mask
    expect_equal(as.vector(nv_mul(x32, array(c(TRUE, FALSE)))), c(1, 0))
    # explicit conversions across the category boundary, in both directions
    expect_equal(as.vector(jit(function(x) nv_convert(x, "i32"))(TRUE)), 1L)
    expect_equal(as.vector(jit(function(x) nv_convert(x, "bool"))(1L)), TRUE)
  })

  it("keeps every digit when an R double is converted to an integer data type", {
    # The double is uploaded as f64 -- the one float that holds it exactly -- so
    # the conversion sees the value itself rather than an f32 of it.
    # `i64` comes back as a bit64::integer64; compare its digits, which is the
    # only comparison that stays exact past 2^53.
    f <- function(x) format(as_array(jit(function(v) nv_convert(v, "i64"))(x)))
    expect_equal(f(3e9), "3000000000")
    expect_equal(f(1e18), "1000000000000000000")
    expect_equal(f(-3e9), "-3000000000")
    # ... and still truncates toward zero, like converting a typed array does.
    expect_equal(f(1.9), "1")
    expect_equal(f(-1.9), "-1")
  })

  it("commits to the default data type when nothing claims it", {
    expect_equal(dtype(jit(function() 1)()), as_dtype("f32"))
    expect_equal(dtype(jit(function() 1L)()), as_dtype("i32"))
    expect_equal(dtype(jit(function() TRUE)()), as_dtype("bool"))
    expect_equal(dtype(jit(identity)(1)), as_dtype("f32"))
    expect_equal(dtype(jit(identity)(array(1:4))), as_dtype("i32"))
    # ... including a value that only ever meets other R values
    expect_equal(jit(function() nv_mul(2, 3))(), nv_scalar(6, dtype = "f32"))
    expect_equal(jit(function() nv_mul(2, 3L))(), nv_scalar(6, dtype = "f32"))
    expect_equal(jit(function(x) x + 1)(1), nv_scalar(2, dtype = "f32"))
  })

  it("takes the data type it meets, whatever the width", {
    expect_equal(dtype(nv_array(1L, dtype = "i8") + 1), as_dtype("f32"))
    expect_equal(dtype(nv_array(1L, dtype = "i8") + 1L), as_dtype("i8"))
    expect_equal(dtype(nv_array(1, dtype = "f64") + 1L), as_dtype("f64"))
    expect_equal(dtype(nv_array(TRUE) + 1L), as_dtype("i32"))
    # narrower than the value's own default, and on either side of the operator
    expect_equal(jit(function(x) x * 2L)(nv_scalar(1, dtype = "i16")), nv_scalar(2L, dtype = "i16"))
    expect_equal(jit(function(x) 2 + x)(nv_scalar(1)), nv_scalar(3))
    expect_equal(jit(function(x) nv_mul(2, x))(nv_scalar(3, dtype = "f64")), nv_scalar(6, dtype = "f64")) # nolint
    expect_equal(jit(function(x) x == TRUE)(nv_scalar(FALSE)), nv_scalar(FALSE))
  })

  it("takes the default when it only ever meets other literals", {
    # The f64 arrives on `y`, one step after `x` has already committed. This is
    # the documented limit of committing per operation.
    f <- jit(function(x) {
      y <- x * 2
      y + nv_scalar(1, dtype = "f64")
    })
    out <- f(sqrt(2))
    expect_equal(dtype(out), as_dtype("f64"))
    expect_false(isTRUE(all.equal(as_array(out), 2 * sqrt(2) + 1, tolerance = 1e-15)))
  })

  it("is built in each sub-graph that uses it, out of its category", {
    # The convert is recorded in whatever descriptor is being traced, so a memo
    # entry from one branch must not be handed to the other -- the second branch
    # would reference a value only the first one computes.
    f <- jit(function(a, x) {
      prim_if(nv_scalar(TRUE), function() nv_add(x, a), function() nv_add(x, a))
    })
    expect_identical(as_array(f(3L, nv_scalar(2, dtype = "f64"))), 5)
    g <- jit(function(a, x) {
      nv_while(list(i = x), function(i) i < nv_add(x, a), function(i) list(i = nv_add(i, a)))
    })
    expect_identical(as_array(g(3L, nv_scalar(2, dtype = "f64"))$i), 5)
  })

  it("reaches an unsigned data type through a convert when negative", {
    # An R integer is signed, so it is built at i32/i64 and converted: writing it
    # straight into the IR would not even be valid StableHLO, and the answer has
    # to be the same eagerly as under jit().
    x <- nv_array(c(1L, 2L), dtype = "ui32")
    expect_equal(as_array(nv_add(x, -1L)), as_array(jit(function(x) nv_add(x, -1L))(x)))
    expect_equal(as.character(as_array(nv_add(x, -1L))), c("0", "1"))
    expect_equal(as.character(as_array(nv_add(x, 1L))), c("2", "3"))
    graph <- trace_fn(function(x) nv_add(x, -1L), list(x = nv_aval("ui32", 2L)))
    src <- repr(stablehlo(graph)[[1L]])
    expect_match(src, "dense<-1> : tensor<i32>", fixed = TRUE)
    expect_match(src, "stablehlo.convert", fixed = TRUE)
  })

  it("commits to its default as a sub-graph parameter", {
    # A loop's state is a parameter of its sub-graphs, and those are traced before
    # the state meets anything, so there is nothing for a bare R value there to
    # take a data type from. Documented in `?RData`.
    out <- nv_while(list(i = 1), \(i) i < 10, \(i) list(i = i * 2))
    expect_equal(dtype(out$i), as_dtype("f32"))
    # ... so a body that carries another data type is an error, not an f64 loop.
    expect_error(
      nv_while(list(i = 1), \(i) i < 10, \(i) list(i = i * nv_scalar(2, dtype = "f64"))),
      "same type"
    )
    # Naming the data type is what makes the loop carry it, exactly.
    out <- nv_while(
      list(i = nv_convert(sqrt(2), "f64")),
      \(i) i < nv_scalar(10, dtype = "f64"),
      \(i) list(i = i * nv_scalar(2, dtype = "f64"))
    )
    expect_identical(as_array(out$i), sqrt(2) * 8)
    # A value that merely flows into a sub-graph body is unaffected: it meets a
    # data type at its use site there and is built at it.
    f <- jit(function(v, s) nv_while(list(s = s), \(s) s < nv_scalar(10, dtype = "f64"), \(s) list(s = s + v)))
    expect_identical(as_array(f(sqrt(2), nv_scalar(0, dtype = "f64"))$s), sum(rep(sqrt(2), 8)))
  })

  it("is embedded as f64 with no convert when anchored", {
    graph <- trace_fn(function(x) x / sqrt(2), list(x = nv_aval("f64", integer())))
    src <- repr(stablehlo(graph)[[1L]])
    expect_match(src, "tensor<f64>", fixed = TRUE)
    expect_no_match(src, "convert", fixed = TRUE)
  })

  it("leaves a program that has no f64 in it free of one", {
    graph <- trace_fn(function(x) x / sqrt(2), list(x = nv_aval("f32", integer())))
    src <- repr(stablehlo(graph)[[1L]])
    expect_no_match(src, "f64", fixed = TRUE)
  })
})

describe("nv_array", {
  it("gives a traced R value a data type", {
    # This is what the `dtype()` error tells the user to reach for, so it has to
    # work -- and it has to be exact.
    f <- jit(function(v) nv_array(v, dtype = "f64"))
    expect_identical(as_array(f(sqrt(2))), sqrt(2))
    expect_equal(dtype(jit(function(v) nv_scalar(v, dtype = "i64"))(2L)), as_dtype("i64"))
    # Without a dtype it commits to its default.
    expect_equal(dtype(jit(function(v) nv_array(v))(1)), as_dtype("f32"))
    # A value that already has a dtype is not rebuilt this way.
    expect_error(jit(function(v) nv_array(v + 1, dtype = "f64"))(1), "traced value")
  })
})

describe("nv_convert", {
  it("builds an R value at the target data type directly", {
    expect_identical(as_array(jit(function() nv_convert(sqrt(2), "f64"))()), sqrt(2))
    expect_identical(as_array(nv_convert(sqrt(2), "f64")), sqrt(2))
    # Converting to an integer dtype truncates, as it does for a typed array.
    expect_equal(as_array(nv_convert(1.9, "i32")), 1L)
    expect_equal(as_array(nv_convert(-1.9, "i32")), -1L)
  })
})

describe("prim_convert", {
  it("builds an R value at the target data type directly", {
    # The primitive has no `promote` rule to box a literal written in the body,
    # so without boxing it here the value would commit at its default and be
    # converted from `f32`, giving a different answer than the same call eagerly.
    expect_identical(as_array(jit(function() prim_convert(sqrt(2), "f64"))()), sqrt(2))
    expect_identical(as_array(prim_convert(sqrt(2), "f64")), sqrt(2))
    expect_identical(as_array(jit(function(x) x * prim_convert(sqrt(2), "f64"))(nv_scalar(1, dtype = "f64"))), sqrt(2)) # nolint
    # An R array takes the same route as a scalar.
    expect_identical(as.vector(jit(function() prim_convert(array(c(0.1, sqrt(2))), "f64"))()), c(0.1, sqrt(2))) # nolint
  })
})

describe("an R value in an nv_* function", {
  it("works as a subscript", {
    x <- nv_array(c(1, 2, 3), dtype = "f32")
    expect_equal(as.vector(jit(function(a, i) a[i])(x, 2L)), 2)
    expect_equal(as.vector(jit(function(a) a[2])(x)), 2)
  })

  it("is built at the common data type by binding and cross products", {
    # These combine their arguments downstream (`nv_concatenate()`, `nv_matmul()`),
    # so they have to decide the dtype up front: committing an R value first would
    # round it through `f32` on its way to the `f64` it is combined with.
    x <- nv_matrix(c(1, 1), nrow = 1L, dtype = "f64")
    r <- matrix(c(0.1, sqrt(2)), nrow = 1L)
    expect_identical(as.vector(nv_rbind(x, r)), as.vector(rbind(c(1, 1), c(0.1, sqrt(2)))))
    expect_identical(as.vector(nv_cbind(nv_matrix(1, nrow = 1L, dtype = "f64"), matrix(sqrt(2)))), c(1, sqrt(2))) # nolint
    y <- nv_matrix(1, nrow = 1L, dtype = "f64")
    expect_identical(as.vector(nv_crossprod(y, matrix(sqrt(2)))), sqrt(2))
    expect_identical(as.vector(nv_tcrossprod(y, matrix(sqrt(2)))), sqrt(2))
    # The common dtype is still the one both sides agree on.
    expect_equal(dtype(nv_rbind(nv_matrix(1L, nrow = 1L), matrix(1))), as_dtype("f32"))
  })

  it("uses the array's data type when assigned into one", {
    x <- nv_array(c(1, 2, 3), dtype = "f64")
    x[2] <- 0.1
    expect_identical(as.vector(x), c(1, 0.1, 3))
    # What may be assigned is still decided by the dtype the R value would take:
    # an R double has no place in an integer array, whatever its value.
    y <- nv_array(1:3, dtype = "i32")
    expect_error(y[2] <- 1.5, "R double")
    expect_error(y[2] <- 1, "R double")
    y[2] <- 5L
    expect_identical(as.vector(y), c(1L, 5L, 3L))
  })

  it("is built at the array's data type as nv_pad()'s padding value", {
    for (dt in c("f64", "f32", "i8", "i32")) {
      x <- nv_array(c(1L, 2L), dtype = dt)
      # The padding value yields to `x`, so a literal is built at `x`'s dtype --
      # but only from its own category, so it has to be written in the one `x` is
      # in: a double for a float array, an integer for an integer one.
      is_float <- is_dtype_float(as_dtype(dt))
      expect_equal(dtype(nv_pad(x, if (is_float) 0 else 0L, 1L, 1L)), as_dtype(dt), info = dt)
      expect_error(nv_pad(x, if (is_float) 0L else 0, 1L, 1L), "cannot be used at", info = dt)
    }
    x <- nv_array(c(1, 2), dtype = "f64")
    expect_identical(as.vector(nv_pad(x, sqrt(2), 1L, 0L)), c(sqrt(2), 1, 2))
    # A value that already has a dtype is never moved: one that disagrees with
    # `x` is a mistake to report rather than a conversion to make silently.
    expect_identical(as.vector(nv_pad(x, nv_scalar(0, dtype = "f64"), 1L, 0L)), c(0, 1, 2))
    expect_error(nv_pad(x, nv_scalar(0, dtype = "f32"), 1L, 0L), "must have the same dtype")
  })

  it("yields to the array in nv_solve() and nv_triangular_solve()", {
    a <- nv_array(matrix(c(2, 0, 0, 2), nrow = 2), dtype = "f64")
    out <- nv_solve(a, matrix(c(2, 4), ncol = 1L))
    expect_equal(dtype(out), as_dtype("f64"))
    expect_equal(as.vector(out), c(1, 2))
    # ... and two typed arrays that disagree are still rejected, rather than one
    # of them being widened.
    expect_error(nv_solve(a, nv_array(matrix(c(2, 4), ncol = 1L), dtype = "f32")))
    out <- nv_triangular_solve(a, matrix(c(2, 4), ncol = 1L))
    expect_equal(dtype(out), as_dtype("f64"))
  })

  it("names no data type for nv_rnorm(), which falls back to the default float", {
    state <- nv_rng_state(42L)
    # Bare R values have no data type, so the sample falls back to the default
    # float rather than to whatever R stores its numbers as.
    expect_equal(dtype(nv_rnorm(4L, state, mean = 0, sd = 1)[[2L]]), as_dtype("f32"))
    # A real array names it.
    expect_equal(
      dtype(nv_rnorm(4L, state, mean = nv_scalar(0, dtype = "f64"))[[2L]]),
      as_dtype("f64")
    )
    expect_equal(
      dtype(nv_rnorm(4L, state, sd = nv_scalar(1, dtype = "f64"))[[2L]]),
      as_dtype("f64")
    )
    # An integer array cannot name one, so the default float stands.
    expect_equal(dtype(nv_rnorm(4L, state, mean = nv_scalar(0L))[[2L]]), as_dtype("f32"))
    # ... and an explicit `dtype` still wins.
    expect_equal(
      dtype(nv_rnorm(4L, state, dtype = "f64", mean = 0)[[2L]]),
      as_dtype("f64")
    )
    # An f64 mean cannot be narrowed to an f32 sample.
    expect_error(
      nv_rnorm(4L, state, dtype = "f32", mean = nv_scalar(0, dtype = "f64")),
      "not promotable"
    )
  })
})

describe("resolve_primitive_args", {
  # A primitive brings its arrayish arguments to one data type before it records a
  # call, following the rule it declared (see `new_primitive()`). Without that, an
  # R value would commit to its own default and whether the call worked would
  # depend on whether the array it met happened to be at that default -- which is
  # a fact about `default_dtype_r()`, not about the call.

  it("an R value takes the data type of the operand it meets", {
    for (dt in c("f32", "f64")) {
      x <- nv_scalar(1, dtype = dt)
      expect_equal(dtype(prim_mul(x, 2)), as_dtype(dt), info = dt)
      expect_equal(dtype(prim_add(2, x)), as_dtype(dt), info = dt)
    }
    for (dt in c("i8", "i16", "i32", "i64")) {
      x <- nv_scalar(1L, dtype = dt)
      expect_equal(dtype(prim_add(x, 1L)), as_dtype(dt), info = dt)
      expect_equal(dtype(prim_add(1L, x)), as_dtype(dt), info = dt)
    }
    expect_equal(dtype(prim_and(nv_scalar(TRUE), TRUE)), as_dtype("bool"))
  })

  it("the operand slots that take a scalar accept a literal", {
    x <- nv_array(c(1, 2), dtype = "f64")
    expect_equal(dtype(prim_pad(x, 0, 1L, 0L, 0L)), as_dtype("f64"))
    expect_equal(dtype(prim_clamp(0, x, 1)), as_dtype("f64"))
    expect_equal(dtype(prim_reduce(x, init = 0, axes = 1L, reductor = prim_add)), as_dtype("f64"))
    expect_equal(dtype(prim_ifelse(nv_scalar(TRUE), nv_scalar(1, dtype = "f64"), 0)), as_dtype("f64"))
  })

  it("a literal reaches a primitive with every digit it had", {
    x <- nv_scalar(1, dtype = "f64")
    expect_identical(as_array(prim_mul(x, sqrt(2))), sqrt(2))
    expect_identical(as_array(jit(function(v) prim_mul(v, sqrt(2)))(x)), sqrt(2))
    # ... and the emitted program holds an f64 constant rather than a widened f32
    graph <- trace_fn(function(v) prim_mul(v, sqrt(2)), list(v = nv_aval("f64", integer())))
    src <- repr(stablehlo(graph)[[1L]])
    expect_match(src, "tensor<f64>", fixed = TRUE)
    expect_no_match(src, "convert", fixed = TRUE)
  })

  it("brings a primitive call's operands together without the default data type", {
    # This is the point of the change: `prim_mul(x, 2)` used to work only because
    # `f32` happens to be the default for an R double.
    expect_equal(dtype(prim_mul(nv_scalar(1, dtype = "f32"), 2)), as_dtype("f32"))
    expect_equal(dtype(prim_mul(nv_scalar(1, dtype = "f64"), 2)), as_dtype("f64"))
  })

  it("an all-R group agrees only within one R storage type", {
    expect_equal(dtype(prim_add(1, 2)), as_dtype("f32"))
    expect_equal(dtype(prim_add(1L, 2L)), as_dtype("i32"))
    expect_equal(dtype(prim_and(TRUE, FALSE)), as_dtype("bool"))
    # A mix has no data type to agree on, and says so rather than picking one.
    expect_error(prim_add(1, 2L), "no data type to agree on")
    expect_error(prim_add(TRUE, 1L), "no data type to agree on")
  })

  it("a literal is only ever built within its own category", {
    # A double stays a float, an integer an integer, a logical a bool. Crossing a
    # category is promotion, which is the `nv_*` layer's job.
    expect_error(prim_add(nv_scalar(1L, dtype = "i8"), 1.5), "cannot be used at")
    expect_error(prim_add(nv_scalar(1, dtype = "f64"), 1L), "cannot be used at")
    expect_error(prim_and(nv_scalar(TRUE), 1L), "cannot be used at")
    # The error names the category rule and the way out.
    expect_error(prim_add(nv_scalar(1, dtype = "f64"), 1L), "own category")
    expect_error(prim_add(nv_scalar(1, dtype = "f64"), 1L), "nv_convert")
    # ... where the `nv_*` layer promotes across categories, as it always did.
    expect_equal(dtype(nv_add(nv_scalar(1, dtype = "f64"), 1L)), as_dtype("f64"))
    expect_equal(dtype(nv_add(nv_scalar(1L, dtype = "i8"), 1.5)), as_dtype("f32"))
  })

  it("a group with several data types present is left to type inference", {
    # The rule never converts an operand that has a data type, so a call whose
    # typed operands disagree fails where it always did.
    expect_error(
      prim_add(nv_scalar(1, dtype = "f32"), nv_scalar(1, dtype = "f64")),
      "same array type"
    )
  })

  it("primitives whose operands are meant to differ opt out", {
    # `promote = NULL`: a sort payload and a loop-carried state are deliberately
    # heterogeneous, so there is nothing for an R value to yield to and each
    # commits to its own default, as before.
    expect_equal(dtype(nv_argsort(nv_array(c(3, 1, 2)))), as_dtype("i32"))
    expect_equal(as.vector(nv_sort(nv_array(c(3, 1, 2)))), c(1, 2, 3))
    out <- nv_while(
      list(i = nv_scalar(0L), w = 0.5),
      function(i, w) i < nv_scalar(3L),
      function(i, w) list(i = i + 1L, w = w * nv_scalar(0.5, dtype = "f32"))
    )
    expect_equal(dtype(out[[1L]]), as_dtype("i32"))
    expect_equal(dtype(out[[2L]]), as_dtype("f32"))
  })

  it("an argument a rule leaves out keeps its own data type", {
    # `pred` is a bool however the branches are typed, and indices stay integers.
    out <- prim_ifelse(nv_scalar(TRUE), nv_scalar(1L, dtype = "i8"), 3L)
    expect_equal(dtype(out), as_dtype("i8"))
    x <- nv_array(c(1, 2, 3), dtype = "f64")
    expect_equal(dtype(nv_subset(x, 2)), as_dtype("f64"))
    expect_equal(as.vector(nv_subset(x, nv_array(2L, dtype = "i64"))), 2)
  })

  it("eager and traced primitive calls agree", {
    arrays <- list(f64 = nv_scalar(1, dtype = "f64"), i8 = nv_scalar(1L, dtype = "i8"))
    values <- list(f64 = sqrt(2), i8 = 3L)
    for (dt in names(arrays)) {
      eager <- prim_add(arrays[[dt]], values[[dt]])
      traced <- jit(function(x, v) prim_add(x, v))(arrays[[dt]], values[[dt]])
      expect_equal(dtype(eager), dtype(traced), info = dt)
      expect_identical(format(as_array(eager)), format(as_array(traced)), info = dt)
    }
  })
})

describe("an R argument of a jitted call", {
  it("is converted whatever its R storage type", {
    f <- jit(identity)
    expect_equal(f(1), nv_scalar(1))
    expect_equal(f(1L), nv_scalar(1L))
    # A logical is the one R type that names its dtype: there is nothing for
    # `TRUE` to become other than `bool`.
    expect_equal(f(TRUE), nv_scalar(TRUE))
    expect_equal(f(matrix(1:4, 2, 2)), nv_matrix(1:4, nrow = 2, ncol = 2))
    out <- f(array(1:24, dim = c(2, 3, 4)))
    expect_equal(dtype(out), as_dtype("i32"))
    expect_equal(shape(out), c(2L, 3L, 4L))
  })

  it("is converted in the leaves of a nested argument too", {
    f <- jit(function(pair) pair[[1]] + pair[[2]])
    out <- f(list(1, 2))
    expect_equal(dtype(out), as_dtype("f32"))
    expect_equal(as_array(out), 3)
  })

  it("is left alone when the argument is static", {
    f <- jit(function(x, flag) if (flag) x + 1 else x * 2, static = "flag")
    expect_equal(as_array(f(nv_scalar(3), TRUE)), 4)
    expect_equal(as_array(f(3, FALSE)), 6)
  })

  it("is left alone when a traced value reaches an inner jit", {
    inner <- jit(function(x) x + 1)
    outer <- jit(function(x) inner(x))
    expect_equal(as_array(outer(nv_scalar(1))), 2)
  })

  it("is not baked into the compiled program", {
    f <- jit(function(x, y) x + y)
    x <- nv_scalar(0, dtype = "f64")
    expect_identical(as_array(f(x, sqrt(2))), sqrt(2))
    # Same key, so this is a cache hit -- and it must still return its own value,
    # not the one the program was compiled with.
    expect_identical(as_array(f(x, pi)), pi)
    expect_equal(cache_size(f), 1L)
  })

  it("is exact at the wider data type when used at two", {
    f <- jit(function(t) {
      list(
        wide = nv_scalar(0, dtype = "f64") + t,
        narrow = nv_scalar(0, dtype = "f32") + t
      )
    })
    out <- f(sqrt(2))
    expect_equal(dtype(out$wide), as_dtype("f64"))
    expect_equal(dtype(out$narrow), as_dtype("f32"))
    # Uploaded once as f64 and converted down for the f32 site: one rounding,
    # exactly as an f32 upload would have been.
    expect_identical(as_array(out$wide), sqrt(2))
    expect_identical(as_array(out$narrow), as_array(nv_scalar(sqrt(2), dtype = "f32")))
  })

  it("is still an input the call supplies when the body never uses it", {
    f <- jit(function(x, y) x + 1)
    expect_identical(as_array(f(nv_scalar(1, dtype = "f64"), 99)), 2)
  })

  it("errors for a bare vector with no shape", {
    f <- jit(function(x) x)
    expect_snapshot(f(c(1, 2, 3)), error = TRUE)
  })

  it("errors for a leaf that is not an array or a scalar", {
    f <- jit(function(x) x)
    expect_snapshot(f("hello"), error = TRUE)
  })

  it("names the path to a bad nested list element", {
    f <- jit(function(l) l[[1]])
    expect_snapshot(f(list(list(a = "abc"))), error = TRUE)
  })

  it("names the path to a bad unnamed nested element", {
    f <- jit(function(pair) pair[[1]])
    expect_snapshot(f(list("bad", nv_scalar(1))), error = TRUE)
  })

  it("reaches an f64 use site exactly on the quickr backend", {
    skip_if_no_quickr()
    local_backend("quickr")
    x <- nv_scalar(1, dtype = "f64")
    expect_identical(as_array(jit(function(x) x / sqrt(2))(x)), 1 / sqrt(2))
    expect_identical(as_array(jit(function(x, y) x / y)(x, sqrt(2))), 1 / sqrt(2))
  })

  it("is coerced to the data type the program takes on the quickr backend", {
    skip_if_no_quickr()
    local_backend("quickr")
    # The leaf arrives as an R integer but the program consumes it as f64.
    f <- jit(function(x, y) x + y)
    expect_identical(as_array(f(nv_scalar(1, dtype = "f64"), 2L)), 3)
  })

  it("is converted the same way on the quickr backend", {
    skip_if_no_quickr()
    local_backend("quickr")
    f <- jit(identity)
    expect_equal(f(1), nv_scalar(1))
    expect_equal(f(matrix(1:4, 2, 2)), nv_matrix(1:4, nrow = 2, ncol = 2))
    g <- jit(function(pair) pair[[1]] + pair[[2]])
    expect_equal(g(list(nv_scalar(1L), 2L)), nv_scalar(3L))
  })

  it("errors for a bare vector on the quickr backend", {
    skip_if_no_quickr()
    local_backend("quickr")
    f <- jit(function(x) x)
    expect_snapshot(f(c(1, 2, 3)), error = TRUE)
  })
})

describe("gradient", {
  # An R argument of a jitted call stays open through gradient()'s inline trace:
  # the differentiated body decides which data types the value is used at, exactly
  # as it does under plain jit().

  it("takes a bare R value as an argument", {
    expect_equal(as_array(jit(gradient(function(v) v * v))(2)[[1L]]), 4)
    expect_equal(as_array(jit(gradient(function(v) v * v))(nv_scalar(2))[[1L]]), 4)
    # ... and as a literal in the body of the function being differentiated
    expect_equal(jit(gradient(function(x) x * 2, wrt = "x"))(nv_scalar(1)), list(x = nv_scalar(2)))
  })

  it("reaches an f64 use site exactly through the inline trace", {
    # d/dv of v^2 at f64 is 2v; sqrt(2) is not representable at f32, so this
    # catches a commit at the default dtype.
    f <- function(v) nv_scalar(1, dtype = "f64") * v * v
    r <- jit(gradient(f))(sqrt(2))[[1L]]
    expect_equal(dtype(r), as_dtype("f64"))
    expect_identical(as_array(r), 2 * sqrt(2))
  })

  it("keeps a bare R argument exact in value_and_gradient()", {
    f <- function(v) v * nv_scalar(1, dtype = "f64")
    vg <- jit(value_and_gradient(f))(sqrt(2))
    expect_identical(as_array(vg$value), sqrt(2))
    expect_identical(as_array(vg$grad[[1L]]), 1)
    expect_equal(dtype(vg$value), as_dtype("f64"))
  })

  it("matches plain jit() for an argument used at two data types", {
    # v enters at f64 and, through nv_convert, at f32: the input is supplied at
    # f64 and the f32 use is the program's own convert -- one answer whether the
    # body is differentiated or not.
    f <- function(v) v * nv_scalar(1, dtype = "f64") + nv_convert(nv_convert(v, "f32"), "f64")
    vg <- jit(value_and_gradient(f))(sqrt(2))
    expect_identical(as_array(vg$value), as_array(jit(f)(sqrt(2))))
    # both use sites contribute 1 to the gradient, converted losslessly
    expect_identical(as_array(vg$grad[[1L]]), 2)
  })

  it("widens the input past every use when no used data type holds the others", {
    # f16 and bf16 ask for different halves of f32: the gradient's input widens
    # to f32, like a toplevel trace's upload does.
    f <- function(v) nv_convert(nv_convert(v, "f16"), "f32") * nv_convert(nv_convert(v, "bf16"), "f32")
    vg <- jit(value_and_gradient(f))(sqrt(2))
    expect_identical(as_array(vg$value), as_array(jit(f)(sqrt(2))))
    expect_equal(dtype(vg$grad[[1L]]), as_dtype("f32"))
  })

  it("matches plain jit() for a body that commits the value at its default", {
    # v * v meets no dtype, so v commits at f32 -- under gradient() exactly as
    # under plain jit().
    f <- function(v) v * v * nv_scalar(1, dtype = "f64")
    vg <- jit(value_and_gradient(f))(sqrt(2))
    expect_identical(as_array(vg$value), as_array(jit(f)(sqrt(2))))
  })

  it("shares the argument's value with the enclosing body", {
    q <- function(u) nv_scalar(1, dtype = "f64") * u * u
    # the enclosing trace materializes v at f64 before the gradient call ...
    f <- jit(function(v) {
      w <- v * nv_scalar(1, dtype = "f64")
      w + gradient(q)(v)[[1L]]
    })
    expect_identical(as_array(f(sqrt(2))), sqrt(2) + 2 * sqrt(2))
    # ... and after it, reusing the value the gradient's trace built
    g <- jit(function(v) gradient(q)(v)[[1L]] + v * nv_scalar(1, dtype = "f64"))
    expect_identical(as_array(g(sqrt(2))), 2 * sqrt(2) + sqrt(2))
  })

  it("agrees between two gradient calls on the same bare R argument", {
    q <- function(u) nv_scalar(1, dtype = "f64") * u * u
    f <- jit(function(v) gradient(q)(v)[[1L]] + gradient(q)(v)[[1L]])
    expect_identical(as_array(f(sqrt(2))), 4 * sqrt(2))
  })

  it("keeps a bare R argument exact through a nested gradient", {
    # d/da (2a * a) = 4a, with the inner 2a itself a gradient
    f <- jit(function(v) {
      gradient(function(a) {
        gradient(function(b) nv_scalar(1, dtype = "f64") * b * b)(a)[[1L]] * a
      })(v)[[1L]]
    })
    r <- f(sqrt(2))
    expect_equal(dtype(r), as_dtype("f64"))
    expect_identical(as_array(r), 4 * sqrt(2))
  })

  it("keeps a bare R value outside wrt open too", {
    # n is an R integer used out of its category: it is supplied at i32 and the
    # program converts, like everywhere else.
    f <- jit(gradient(function(x, n) x * nv_convert(n, dtype = "f64"), wrt = "x"))
    expect_identical(as_array(f(nv_scalar(2, dtype = "f64"), 3L)$x), 3)
  })

  it("gives an unused bare R argument an input slot at its default", {
    f <- jit(gradient(function(a, b) nv_scalar(1, dtype = "f64") * a * a))
    r <- f(sqrt(2), pi)
    expect_identical(as_array(r[[1L]]), 2 * sqrt(2))
    expect_identical(as_array(r[[2L]]), 0)
    expect_equal(dtype(r[[2L]]), as_dtype("f32"))
  })

  it("takes an R value written in the enclosing body exactly", {
    # The literal is not an input of any graph: it crosses the gradient boundary
    # and the differentiated body builds it at the data type it meets there.
    f <- jit(function(x) gradient(function(a, b) a * b, wrt = "a")(x, sqrt(2))$a)
    r <- f(nv_scalar(1, dtype = "f64"))
    expect_equal(dtype(r), as_dtype("f64"))
    expect_identical(as_array(r), sqrt(2))
    # ... and asking for its gradient says what a plain R value would.
    g <- jit(function(x) gradient(function(a, b) a * b)(x, sqrt(2)))
    expect_error(g(nv_scalar(1, dtype = "f64")), "passed as a plain R value")
  })

  it("does not bake the R value into the compiled gradient", {
    f <- jit(gradient(function(a, b) nv_scalar(1, dtype = "f64") * a * a))
    expect_identical(as_array(f(sqrt(2), pi)[[1L]]), 2 * sqrt(2))
    # same key, so this is a cache hit -- and it must return its own gradient
    expect_identical(as_array(f(pi, 7)[[1L]]), 2 * pi)
  })
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
    graph <- trace_fn(f, list(x = nv_aval("i32", integer())))
    expect_snapshot(graph)
  })

  it("uploads an R argument used at two data types once and converts", {
    f <- function(x, y) nv_add(nv_convert(nv_mul(x, y), "f64"), nv_convert(y, "f64"))
    graph <- trace_fn(f, list(x = nv_aval("f32", integer()), y = nv_aval("double", integer())))
    expect_snapshot(graph)
  })

  it("keeps an R argument open under an inlined gradient", {
    f <- function(t) gradient(function(z) z * z)(t)
    graph <- trace_fn(f, list(t = nv_aval("double", integer())))
    expect_snapshot(graph)
  })

  it("builds an R literal inside the sub-graph body that uses it", {
    # prim_if's branches are traced as their own graphs, and the printer elides
    # sub-graph bodies, so this is checked by value.
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
