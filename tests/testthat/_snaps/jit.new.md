# rejecting a reference-semantics static names it helpfully

    Code
      f(list(a = 1, opts = list(env = e)), nv_array(1))
    Condition
      Error in `pjrt::dispatcher()`:
      ! unused argument (context = default_dtypes_context("pjrt"))

