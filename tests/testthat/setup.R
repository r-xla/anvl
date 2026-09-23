old_opts <- options(
  warnPartialMatchArgs = TRUE,
  warnPartialMatchAttr = TRUE,
  warnPartialMatchDollar = TRUE,
  anvl.backend = "pjrt"
)

# The whole suite can be run at other default data types or on another default
# device, so that CI catches anything hardcoding `f32` / `i32` where it should
# read `default_dtypes()`, or allocating on the first CPU device where it should
# have followed the trace or its operands:
#
#   ANVL_DEFAULT_DTYPES="float=f64,int=i64" devtools::test()
#   ANVL_DEFAULT_DEVICE=cpu:1 devtools::test()
#
# anvl reads them when it is loaded and falls back to them when the
# `anvl.default_dtypes` / `anvl.default_device` options are not set. A test that asserts the registered data types or the first
# CPU device clears them with `local_registered_default_dtypes()` /
# `local_unset_default_device()` (see `helper.R`).
#
# Both apply to every backend, but quickr supports only `f64`, `i32` and `bool`
# and has a single CPU device, so they are errors there rather than something
# quickr can honour. What they exist to catch is anvl hardcoding a default, not
# quickr's limits, so skip the quickr tests for such a run.
if (nzchar(Sys.getenv("ANVL_DEFAULT_DTYPES")) || nzchar(Sys.getenv("ANVL_DEFAULT_DEVICE"))) {
  Sys.setenv(ANVL_TEST_SKIP_QUICKR = "1")
}

# so we can test multiple devices.
Sys.setenv(PJRT_CPU_DEVICE_COUNT = 2L)
