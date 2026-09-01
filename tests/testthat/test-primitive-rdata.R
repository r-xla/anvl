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
