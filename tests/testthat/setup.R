old_opts <- options(
  warnPartialMatchArgs = TRUE,
  warnPartialMatchAttr = TRUE,
  warnPartialMatchDollar = TRUE
)

# The `anvl.backends` option is read when the namespace loads, which has already
# happened by the time this file runs, so the backends the test suite exercises
# are activated directly -- independently of what the developer has in their
# `.Rprofile`. "pjrt" comes first and is therefore the default backend.
activate_backends(c("pjrt", if (requireNamespace("quickr", quietly = TRUE)) "quickr"))

# so we can test multiple devices.
Sys.setenv(PJRT_CPU_DEVICE_COUNT = 2L)
