# a missing value in an R value / warns where i32 keeps it, naming the argument

    Code
      invisible(add(NA_integer_, nv_scalar(1L, dtype = "i32")))
    Condition
      Warning:
      Input `x` contains at least one "NA", which "i32" stores as "-2147483648".
      i It materializes as "NA" again in R, which `as_array()` reports on the way back.
      i Set `check = "err"` to make this an error, or `check = FALSE` to silence it; the option `pjrt.upload_check` sets the default.

