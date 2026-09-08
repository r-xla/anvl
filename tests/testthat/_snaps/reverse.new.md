# wrt for non-array input: gradient

    Code
      g <- gradient(nv_round, wrt = "method")
      g(nv_scalar(1), method = "nearest_even")
    Condition
      Error in `check_wrt_arrayish()`:
      ! Cannot compute gradient with respect to non-array argument.
      x Got <character>

# wrt for non-array input: value_and_gradient

    Code
      g <- value_and_gradient(nv_round, wrt = "method")
      g(nv_scalar(1), method = "nearest_even")
    Condition
      Error in `check_wrt_arrayish()`:
      ! Cannot compute gradient with respect to non-array argument.
      x Got <character>

# wrt for nested non-array input: gradient

    Code
      g <- gradient(f, wrt = "x")
      g(x = list(nv_scalar(1), 2L))
    Condition
      Error in `check_wrt_arrayish()`:
      ! Can only compute gradient with respect to float arrays.
      x Got i32

# wrt for nested non-array input: value_and_gradient

    Code
      g <- value_and_gradient(f, wrt = "x")
      g(x = list(nv_scalar(1), 2L))
    Condition
      Error in `check_wrt_arrayish()`:
      ! Can only compute gradient with respect to float arrays.
      x Got i32

# can only compute gradient w.r.t. float arrays

    Code
      gradient(nv_floor, wrt = "x")(nv_scalar(1L))
    Condition
      Error in `check_wrt_arrayish()`:
      ! Can only compute gradient with respect to float arrays.
      x Got i32

# wrt arg passed as plain R literal errors clearly

    Code
      jit(function() gradient(nv_log, wrt = "x")(1))()
    Condition
      Error in `pjrt::dispatcher()`:
      ! unused argument (context = default_dtypes_context("pjrt"))

---

    Code
      jit(function() gradient(function(x, y) prim_add(x, y))(1, 2))()
    Condition
      Error in `pjrt::dispatcher()`:
      ! unused argument (context = default_dtypes_context("pjrt"))

