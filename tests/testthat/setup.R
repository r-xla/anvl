old_opts <- options(
  warnPartialMatchArgs = TRUE,
  warnPartialMatchAttr = TRUE,
  warnPartialMatchDollar = TRUE,
  anvl.default_backend = "pjrt"
)

# The `anvl.backends` option is read when the namespace loads, which has
# already happened by the time this file runs, so the quickr backend the test
# suite exercises is activated directly. `anvl.default_backend` above keeps the
# runtime default at "pjrt" (activating two backends makes it "auto").
activate_backends(c("pjrt", if (requireNamespace("quickr", quietly = TRUE)) "quickr"))

# so we can test multiple devices.
Sys.setenv(PJRT_CPU_DEVICE_COUNT = 2L)
