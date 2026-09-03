describe("eager/jit equivalence", {
  it("agrees for the binary operators", {
    ops <- list(
      add = nv_add,
      sub = nv_sub,
      mul = nv_mul,
      div = nv_div,
      pow = nv_pow,
      max = nv_max,
      min = nv_min,
      mod = nv_mod,
      eq = nv_eq,
      lt = nv_lt,
      atan2 = nv_atan2
    )
    for (op_name in names(ops)) {
      op <- ops[[op_name]]
      lhs_first <- function(x, y) op(x, y)
      rhs_first <- function(x, y) op(y, x)
      for (dt in names(typed_arrays())) {
        for (rt in names(r_values)) {
          arr <- typed_arrays()[[dt]]
          expect_eager_jit_equal(lhs_first, list(arr, r_values[[rt]]), sprintf("%s(%s, %s)", op_name, dt, rt))
          expect_eager_jit_equal(rhs_first, list(arr, r_values[[rt]]), sprintf("%s(%s, %s)", op_name, rt, dt))
        }
      }
    }
  })

  it("agrees for the multi-argument API functions", {
    expect_eager_jit_equal_grid(list(
      ifelse = function(x, v) nv_ifelse(nv_array(c(TRUE, FALSE)), x, v),
      clamp = function(x, v) nv_clamp(v, x, 5),
      pad = function(x, v) nv_pad(x, v, 1L, 1L),
      convert = function(x, v) nv_convert(v, "f64") * nv_convert(x, "f64"),
      promote = function(x, v) {
        args <- nv_promote_to_common(x, v)
        args[[1L]] * args[[2L]]
      }
    ))
  })

  it("agrees for an R array, not just a scalar", {
    values <- list(array(c(sqrt(2), sqrt(3))), array(c(3L, 4L)), array(c(TRUE, FALSE)))
    for (dt in names(typed_arrays())) {
      for (i in seq_along(values)) {
        expect_eager_jit_equal(
          function(x, v) nv_mul(x, v),
          list(typed_arrays()[[dt]], values[[i]]),
          sprintf("mul(%s, r array %d)", dt, i)
        )
      }
    }
  })
})

describe("eager/jit equivalence", {
  it("agrees for as_anvl_array() and as_anvl_arrays()", {
    # Both are exported, so user code calls them outside any trace as well as
    # inside one. These helpers are written the way an `nv_*` function is.
    expect_eager_jit_equal_grid(list(
      "dtype() of a canonicalized R value" = function(x, v) {
        nv_fill_like(x, 1) * as.numeric(dtype(as_anvl_array(v)) == as_dtype("f64"))
      },
      "canonicalize, then convert by hand" = function(x, v) {
        args <- as_anvl_arrays(x, v)
        nv_convert(args[[2L]], dtype(args[[1L]])) * args[[1L]]
      },
      "no promotion at all" = function(x, v) {
        args <- as_anvl_arrays(x, v)
        args[[1L]] * args[[2L]]
      },
      "shape of a canonicalized R value" = function(x, v) {
        nv_fill_like(x, length(shape(as_anvl_array(v))))
      }
    ))
  })
})
