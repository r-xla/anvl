# a missing value in an R value / warns where i32 keeps it, naming the argument

    Code
      invisible(add(NA_integer_, nv_scalar(1L, dtype = "i32")))
    Condition
      Warning:
      Input `x` contains at least one "NA", stored on the device as "-2147483648".
      i The value materializes as "NA" again in R, which `as_array()` reports on the way back.
      i Use `suppressWarnings()` to silence this.

