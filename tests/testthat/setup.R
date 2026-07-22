old_opts <- options(
  warnPartialMatchArgs = TRUE,
  warnPartialMatchAttr = TRUE,
  warnPartialMatchDollar = TRUE
)

# so we can test multiple devices.
Sys.setenv(PJRT_CPU_DEVICE_COUNT = 2L)
