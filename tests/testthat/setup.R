old_opts <- options(
  warnPartialMatchArgs = TRUE,
  warnPartialMatchAttr = TRUE,
  warnPartialMatchDollar = TRUE,
  anvl.backend = "pjrt"
)

# The whole suite can be run at a different pair of default data types, so that
# CI can check that nothing depends on the ones the backend registers:
#
#   ANVL_DEFAULT_DTYPES="float=f64,int=i64" devtools::test()
#
# The value has the shape the `anvl.default_dtypes` option takes, written as
# `category=dtype` pairs. A test that asserts a particular default sets it
# itself and is unaffected; one that asserts the *registered* default calls
# `local_registered_default_dtypes()` (see `helper.R`) to clear the override.
default_dtypes_env <- Sys.getenv("ANVL_DEFAULT_DTYPES")
if (nzchar(default_dtypes_env)) {
  pairs <- strsplit(strsplit(default_dtypes_env, ",", fixed = TRUE)[[1L]], "=", fixed = TRUE)
  stopifnot(all(lengths(pairs) == 2L))
  old_opts <- c(
    old_opts,
    options(
      anvl.default_dtypes = setNames(
        trimws(vapply(pairs, `[[`, character(1L), 2L)),
        trimws(vapply(pairs, `[[`, character(1L), 1L))
      )
    )
  )
}

# so we can test multiple devices.
Sys.setenv(PJRT_CPU_DEVICE_COUNT = 2L)
