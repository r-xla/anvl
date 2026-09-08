# jit: bare vector without dim errors

    Code
      f(c(1, 2, 3))
    Condition
      Error in `pjrt::dispatcher()`:
      ! unused argument (context = default_dtypes_context("pjrt"))

# jit: non-array/non-scalar leaves (e.g. character) error

    Code
      f("hello")
    Condition
      Error in `pjrt::dispatcher()`:
      ! unused argument (context = default_dtypes_context("pjrt"))

# jit: error shows path for nested list element

    Code
      f(list(list(a = "abc")))
    Condition
      Error in `pjrt::dispatcher()`:
      ! unused argument (context = default_dtypes_context("pjrt"))

# jit: error shows path for unnamed nested element

    Code
      f(list("bad", nv_scalar(1)))
    Condition
      Error in `pjrt::dispatcher()`:
      ! unused argument (context = default_dtypes_context("pjrt"))

