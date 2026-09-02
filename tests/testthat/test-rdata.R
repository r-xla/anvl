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
    expect_error(jit(function(x) dtype(x))(1), "nv_convert")
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

describe("a finished graph's R inputs", {
  it("records which data type each is uploaded at, and from which R type", {
    graph <- trace_fn(
      function(x, y) x + y,
      list(x = nv_aval("f64", 2L), y = nv_aval("double", integer()))
    )
    # Every input's aval is a plain AbstractArray: the data type it is supplied
    # at, and nothing else.
    expect_s3_class(graph$inputs[[1L]]$aval, "AbstractArray")
    expect_s3_class(graph$inputs[[2L]]$aval, "AbstractArray")
    expect_equal(dtype(graph$inputs[[2L]]$aval), as_dtype("f64"))
    # Which of them the caller supplies as bare R data, and as what, is beside
    # them -- one entry per input, NA for an array the caller passes through.
    expect_equal(graph$rdata_types, c(NA_character_, "double"))
    expect_length(graph$rdata_types, length(graph$inputs))
    # ... and the vector the backends read is derived from the two together.
    expect_equal(graph_input_dtypes(graph), c(NA, "f64"))
    # No R input at all means nothing to upload, and the backends skip the step.
    plain <- trace_fn(function(x) x + 1, list(x = nv_aval("f64", 2L)))
    expect_null(graph_input_dtypes(plain))
    expect_true(is.null(plain$rdata_types) || all(is.na(plain$rdata_types)))
  })

  it("survives the graph passes, which replace inputs but never reorder them", {
    graph <- optimize_graph(
      trace_fn(
        function(x, y) x + y,
        list(x = nv_aval("f64", 2L), y = nv_aval("double", integer()))
      )
    )
    expect_length(graph$rdata_types, length(graph$inputs))
    expect_equal(graph_input_dtypes(graph), c(NA, "f64"))
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
    # The first case converts an R double to an integer data type, which stages
    # through f64 and says so.
    suppressWarnings(expect_eager_jit_equal_grid(list(
      "convert without canonicalizing first" = function(x, v) {
        nv_convert(v, peek_dtype(x)) * x
      },
      "peek_dtype() branch" = function(x, v) {
        if (is_dtype_float(peek_dtype(v))) x * v else x + v
      }
    )))
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
    expect_equal(repr(graph$inputs[[3L]]$aval), "f32[]")
    expect_equal(graph$rdata_types, c(NA_character_, NA_character_, "double"))
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
    # the conversion sees the value itself rather than an f32 of it. That is the
    # exactness the staging buys, and the reason it is kept rather than dropped
    # to the value's default; it warns because the f64 is unrequested, which is
    # what the warning above pins.
    # `i64` comes back as a bit64::integer64; compare its digits, which is the
    # only comparison that stays exact past 2^53.
    f <- function(x) suppressWarnings(format(as_array(jit(function(v) nv_convert(v, "i64"))(x))))
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
  it("refuses a traced R value, whatever data type is asked for", {
    # A constructor does not give a traced R value a data type: reaching one
    # outside the value's own category would have to stage through the natural
    # one, bringing in a data type nobody asked for. `nv_convert()` is where a
    # conversion belongs, and it is what the `dtype()` error points at.
    expect_error(jit(function(v) nv_array(v, dtype = "f64"))(sqrt(2)), "traced R value")
    expect_error(jit(function(v) nv_scalar(v, dtype = "i64"))(2L), "traced R value")
    expect_error(jit(function(v) nv_array(v))(1), "traced R value")
    expect_error(jit(function(v) nv_array(v, dtype = "f64"))(sqrt(2)), "nv_convert")
    # A value that already has a data type is refused too, with its own message.
    expect_error(jit(function(v) nv_array(v + 1, dtype = "f64"))(1), "traced value")
  })

  it("points at nv_convert(), which is exact", {
    f <- jit(function(v) nv_convert(v, "f64"))
    expect_identical(as_array(f(sqrt(2))), sqrt(2))
    expect_equal(dtype(jit(function(v) nv_convert(v, "i64"))(2L)), as_dtype("i64"))
  })
})

describe("nv_convert", {
  it("builds an R value at the target data type directly", {
    expect_identical(as_array(jit(function() nv_convert(sqrt(2), "f64"))()), sqrt(2))
    expect_identical(as_array(nv_convert(sqrt(2), "f64")), sqrt(2))
    # Converting to an integer dtype truncates, as it does for a typed array.
    expect_equal(suppressWarnings(as_array(nv_convert(1.9, "i32"))), 1L)
    expect_equal(suppressWarnings(as_array(nv_convert(-1.9, "i32"))), -1L)
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
    expect_error(nv_pad(x, nv_scalar(0, dtype = "f32"), 1L, 0L), "no common data type")
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

describe("staging an R value out of its own category", {
  it("warns when it brings a data type nothing asked for into the program", {
    # An R double cannot be built at an integer or boolean data type, so it is
    # built at f64 and converted -- and a program with no f64 in it acquires
    # one, which some backends cannot run at all.
    expect_warning(
      trace_fn(function(x) prim_convert(x, "i32"), list(x = nv_aval("double", integer()))),
      class = "anvl_staging_widens_warning"
    )
    # Every route into the staging warns, not just the traced one.
    expect_warning(nv_convert(1.9, "i32"), class = "anvl_staging_widens_warning")
    # ... including an in-body literal, which leaves the f64 as a constant
    # rather than an input.
    expect_warning(
      trace_fn(function() prim_convert(2.5, "i32"), list()),
      class = "anvl_staging_widens_warning"
    )
    # ... and the eager path, which has to agree with the traced one.
    expect_warning(nv_convert(1.9, "i32"), class = "anvl_staging_widens_warning")
    # ... and a promotion rule told to let the value cross its own category,
    # which is the one route into the staging that does not name a data type at
    # the call site.
    expect_warning(
      as_anvl_arrays(v = 1.9, .promote = promote_dtype("i32", coerce = TRUE)),
      class = "anvl_staging_widens_warning"
    )
  })

  it("stays quiet where the staging introduces nothing", {
    # An R integer stages at i32 and a logical at bool -- their own defaults, so
    # nothing is brought in that the value would not have committed to anyway.
    quiet <- function(expr) expect_no_warning(expr, class = "anvl_staging_widens_warning")
    quiet(trace_fn(function(x) prim_convert(x, "f32"), list(x = nv_aval("integer", integer()))))
    quiet(trace_fn(function(x) prim_convert(x, "i8"), list(x = nv_aval("integer", integer()))))
    quiet(trace_fn(function(x) prim_convert(x, "f32"), list(x = nv_aval("logical", integer()))))
    # An in-category target is built directly: there is no staging at all.
    quiet(trace_fn(function(x) prim_convert(x, "f64"), list(x = nv_aval("double", integer()))))
  })
})
