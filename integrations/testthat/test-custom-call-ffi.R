# The mechanism the "Custom Calls" article documents, end to end: a handler
# compiled against {pjrt}'s FFI headers, registered under a target name, and
# called from anvl -- eagerly, under jit(), and through a primitive with a
# reverse rule. integrations/run.R has compiled and registered "anvl_hypot"
# before these run.

# the wrapper the article builds around the raw call
nv_hypot <- function(x, y) {
  operands <- nv_promote_to_common(x, y)
  nv_custom_call(
    "anvl_hypot",
    operands[[1L]],
    operands[[2L]],
    output_types = vt(dtype(operands[[1L]]), shape(operands[[1L]]))
  )
}

describe("a compiled custom call", {
  it("computes hypot in f64, where the naive composition underflows", {
    x <- nv_array(c(3, 1e-200), dtype = "f64")
    y <- nv_array(c(4, 1e-200), dtype = "f64")
    # the motivation from the article: x^2 is 0 before the square root runs
    expect_equal(as_array(nv_sqrt(x^2 + y^2)), c(5, 0))
    expect_equal(as_array(nv_hypot(x, y)), c(5, sqrt(2) * 1e-200))
  })

  it("computes hypot in f32 with the same handler", {
    x <- nv_array(c(3, 1e-30), dtype = "f32")
    y <- nv_array(c(4, 1e-30), dtype = "f32")
    out <- nv_hypot(x, y)
    expect_equal(dtype(out), dtype(x))
    expect_equal(as_array(nv_sqrt(x^2 + y^2)), c(5, 0), tolerance = 1e-6)
    expect_equal(as_array(out), c(5, sqrt(2) * 1e-30), tolerance = 1e-6)
  })

  it("promotes mixed dtypes to a common one", {
    out <- nv_hypot(nv_array(c(3, 5), dtype = "f32"), nv_array(c(4, 12), dtype = "f64"))
    expect_equal(dtype(out), as_dtype("f64"))
    expect_equal(as_array(out), c(5, 13))
  })

  it("works inside jit(), for either dtype", {
    f <- jit(function(x, y) nv_reduce_sum(nv_hypot(x, y)))
    expect_equal(as_array(f(nv_array(c(3, 5), dtype = "f64"), nv_array(c(4, 12), dtype = "f64"))), 18)
    expect_equal(
      as_array(f(nv_array(c(3, 5), dtype = "f32"), nv_array(c(4, 12), dtype = "f32"))),
      18,
      tolerance = 1e-6
    )
  })

  it("surfaces a dtype the handler rejects as an R error", {
    x <- nv_array(c(3L, 4L), dtype = "i32")
    expect_error(
      nv_custom_call("anvl_hypot", x, x, output_types = vt("i32", 2)),
      "f32 and f64"
    )
  })

  it("surfaces mismatched operand dtypes as an R error", {
    # nv_custom_call() does not promote, so the handler's own check fires
    expect_error(
      nv_custom_call(
        "anvl_hypot",
        nv_array(c(3, 4), dtype = "f32"),
        nv_array(c(3, 4), dtype = "f64"),
        output_types = vt("f32", 2)
      ),
      "same dtype"
    )
  })
})

describe("a primitive wrapping a custom call", {
  prim_hypot <- new_primitive("hypot", function(x, y) {
    graph_desc_add(self, list(x = x, y = y), infer_fn = function(x, y) {
      list(AbstractArray(dtype = dtype(x), shape = shape(x)))
    })[[1L]]
  })

  prim_hypot[["stablehlo"]] <- function(x, y, output_types) {
    row_major <- rev(seq_along(shape(x)) - 1L)
    list(stablehlo::hlo_custom_call(
      x,
      y,
      call_target_name = "anvl_hypot",
      api_version = 4L,
      has_side_effect = FALSE,
      output_types = output_types,
      operand_layouts = list(row_major, row_major),
      result_layouts = list(row_major)
    ))
  }

  prim_hypot[["reverse"]] <- rule_reverse(function(
    inputs,
    outputs,
    grads,
    params,
    required
  ) {
    x <- inputs[[1L]]
    y <- inputs[[2L]]
    r <- outputs[[1L]]
    grad <- grads[[1L]]
    list(
      if (required[[1L]]) prim_mul(grad, prim_div(x, r)),
      if (required[[2L]]) prim_mul(grad, prim_div(y, r))
    )
  })

  g <- function(x, y) nv_reduce_sum(prim_hypot(x, y))

  it("computes the same values as the raw call", {
    x <- nv_array(c(3, 5), dtype = "f64")
    y <- nv_array(c(4, 12), dtype = "f64")
    expect_equal(as_array(prim_hypot(x, y)), c(5, 13))
    expect_equal(as_array(jit(g)(x, y)), 18)
  })

  it("differentiates, in either dtype", {
    x <- nv_array(c(3, 6), dtype = "f64")
    y <- nv_array(c(4, 8), dtype = "f64")
    grads <- jit(gradient(g))(x, y)
    # d/dx hypot(x, y) = x / hypot(x, y)
    expect_equal(as_array(grads[[1L]]), c(3 / 5, 6 / 10))
    expect_equal(as_array(grads[[2L]]), c(4 / 5, 8 / 10))

    grads32 <- jit(gradient(g))(nv_convert(x, "f32"), nv_convert(y, "f32"))
    expect_equal(dtype(grads32[[1L]]), as_dtype("f32"))
    expect_equal(as_array(grads32[[1L]]), c(3 / 5, 6 / 10), tolerance = 1e-6)
  })
})
