# Eager mode and `jit()` have to mean the same thing. They do not share a code
# path: eagerly, every `nv_*` function is its own jitted call and an R value is
# an argument leaf of it, while under `jit()` the same value is written into the
# body of a larger trace. Tracing is an optimization, not a different language,
# so anything a call produces -- its dtype as much as its digits -- has to agree.
#
# The comparison is on every digit: a value that was rounded through `f32` on the
# way to `f64` differs from the exact one only past the seventh, which is exactly
# the class of difference this is here to catch.

# One R storage type of each kind. `sqrt(2)` rather than a round number: a double
# that survives a detour through f32 tells us nothing.
r_values <- list(double = sqrt(2), integer = 3L, logical = TRUE)

typed_arrays <- function() {
  list(
    f64 = nv_array(c(1, 2), dtype = "f64"),
    f32 = nv_array(c(1, 2), dtype = "f32"),
    i8 = nv_array(c(1L, 2L), dtype = "i8"),
    i64 = nv_array(c(1L, 2L), dtype = "i64"),
    bool = nv_array(c(TRUE, FALSE))
  )
}

# Everything an observer can see about a call's result: the dtype, the shape and
# every digit of the data.
observable <- function(value) {
  if (is_anvl_array(value)) {
    data <- as_array(value)
    return(list(
      dtype = as.character(dtype(value)),
      shape = shape(value),
      data = if (is.double(data)) sprintf("%.17g", data) else format(data)
    ))
  }
  if (is.list(value)) {
    return(lapply(value, observable))
  }
  list(value = format(value))
}

# The result of `f`, or the error it raised -- an error is an observable outcome
# too, and one that must not depend on the mode either.
outcome <- function(f, args) {
  tryCatch(observable(do.call(f, args)), error = function(e) list(error = conditionMessage(e)))
}

expect_eager_jit_equal <- function(f, args, label) {
  eager <- outcome(f, args)
  traced <- outcome(jit(f), args)
  testthat::expect_equal(eager, traced, info = label)
}

test_that("binary operators agree between eager mode and jit()", {
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

test_that("multi-argument API functions agree between eager mode and jit()", {
  f64 <- function() nv_array(c(0.25, 0.75), dtype = "f64")
  calls <- list(
    ifelse = function(x, v) nv_ifelse(nv_array(c(TRUE, FALSE)), x, v),
    clamp = function(x, v) nv_clamp(v, x, 5),
    pad = function(x, v) nv_pad(x, v, 1L, 1L),
    convert = function(x, v) nv_convert(v, "f64") * nv_convert(x, "f64"),
    promote = function(x, v) {
      args <- nv_promote_to_common(x, v)
      args[[1L]] * args[[2L]]
    },
    dnorm = function(x, v) nv_dnorm(f64(), mean = v),
    pnorm = function(x, v) nv_pnorm(f64(), sd = v),
    qnorm = function(x, v) nv_qnorm(f64(), mean = v)
  )
  for (call_name in names(calls)) {
    for (dt in names(typed_arrays())) {
      for (rt in names(r_values)) {
        expect_eager_jit_equal(
          calls[[call_name]],
          list(typed_arrays()[[dt]], r_values[[rt]]),
          sprintf("%s(%s, %s)", call_name, dt, rt)
        )
      }
    }
  }
})

test_that("the canonicalization API means the same thing in both modes", {
  # `as_anvl_array()` and friends are exported, so user code calls them outside
  # any trace as well as inside one. These helpers are written the way an `nv_*`
  # function is, and are the level at which the two modes used to differ.
  helpers <- list(
    "dtype() of a canonicalized R value" = function(x, v) {
      nv_fill_like(x, 1) * as.numeric(dtype(as_anvl_array(v)) == as_dtype("f64"))
    },
    "canonicalize, then convert by hand" = function(x, v) {
      args <- as_anvl_arrays(x, v)
      nv_convert(args[[2L]], dtype(args[[1L]])) * args[[1L]]
    },
    "convert without canonicalizing first" = function(x, v) {
      nv_convert(v, peek_dtype(x)) * x
    },
    "anchored promotion" = function(x, v) {
      args <- as_anvl_arrays(x = x, v = v, .promote = promote_like("x"))
      args$x * args$v
    },
    "promotion to the common dtype" = function(x, v) {
      args <- as_anvl_arrays(x, v, .promote = promote_common())
      args[[1L]] * args[[2L]]
    },
    "no promotion at all" = function(x, v) {
      args <- as_anvl_arrays(x, v)
      args[[1L]] * args[[2L]]
    },
    "shape of a canonicalized R value" = function(x, v) {
      nv_fill_like(x, length(shape(as_anvl_array(v))))
    },
    "peek_dtype() branch" = function(x, v) {
      if (is_dtype_float(peek_dtype(v))) x * v else x + v
    }
  )
  for (helper_name in names(helpers)) {
    for (dt in names(typed_arrays())) {
      for (rt in names(r_values)) {
        expect_eager_jit_equal(
          helpers[[helper_name]],
          list(typed_arrays()[[dt]], r_values[[rt]]),
          sprintf("%s [%s, %s]", helper_name, dt, rt)
        )
      }
    }
  }
})

test_that("an R array, not just a scalar, agrees between the modes", {
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
