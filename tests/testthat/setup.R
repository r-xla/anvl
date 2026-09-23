old_opts <- options(
  warnPartialMatchArgs = TRUE,
  warnPartialMatchAttr = TRUE,
  warnPartialMatchDollar = TRUE,
  anvl.backend = "pjrt"
)

# The whole suite can be run at a different pair of default data types, so that
# CI can check that nothing depends on the ones the backend registers:
#
#   ANVL_TEST_DEFAULT_DTYPES="float=f64,int=i64" devtools::test()
#
# The value has the shape the `anvl.default_dtypes` option takes, written as
# `category=dtype` pairs -- naming no backend, so that it applies to every one
# the suite runs on. A test that asserts a particular default sets it itself
# and is unaffected; one that asserts the *registered* default calls
# `local_registered_default_dtypes()` (see `helper.R`) to clear the override.
default_dtypes_env <- Sys.getenv("ANVL_TEST_DEFAULT_DTYPES")
if (nzchar(default_dtypes_env)) {
  pairs <- strsplit(strsplit(default_dtypes_env, ",", fixed = TRUE)[[1L]], "=", fixed = TRUE)
  stopifnot(all(lengths(pairs) == 2L))
  # The override names no backend, so it reaches quickr too -- which supports
  # only `f64`, `i32` and `bool`, so a pair such as `int=i64` is an error there
  # rather than something quickr can honour. Skip those tests for the run
  # instead: what this configuration exists to catch is anvl hardcoding the
  # registered pair, not quickr's data type support.
  Sys.setenv(ANVL_TEST_SKIP_QUICKR = "1")
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

# The whole suite can be run with a default device other than the platform's
# first, so that CI catches anything allocating on the first device where it
# should have followed the trace or its operands:
#
#   ANVL_TEST_DEFAULT_DEVICE=cpu:1 devtools::test()
#
# Everything a call places itself then sits on `cpu:1`, so a buffer that
# reached for the default instead lands on `cpu:0` and jit's device autodetect
# reports the pair -- where at the same default the two would silently agree.
# The value is a device identifier, which the test setup turns into the
# `anvl.default_device` option for the whole run. A test that asserts the
# platform's own default calls `local_platform_default_device()` (see
# `helper.R`) to clear the override.
default_device_env <- Sys.getenv("ANVL_TEST_DEFAULT_DEVICE")
if (nzchar(default_device_env)) {
  # quickr has a single device, so `"cpu:1"` is an error there rather than
  # something it can honour -- and what this configuration exists to catch is
  # anvl placing a pjrt buffer on the wrong device. Skip those tests instead.
  Sys.setenv(ANVL_TEST_SKIP_QUICKR = "1")
  old_opts <- c(old_opts, options(anvl.default_device = default_device_env))
}
