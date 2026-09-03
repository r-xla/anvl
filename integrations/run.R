#!/usr/bin/env Rscript

# Launcher for anvl's integration tests.
#
# These are not part of the package's test suite: they compile a custom-call
# handler against {pjrt}'s FFI headers with `Rcpp::sourceCpp()`, which needs a
# C++20 toolchain and takes long enough that `R CMD check` should not pay for
# it. They run in one CI job (Ubuntu x86, CPU), see
# .github/workflows/integration-tests.yaml.
#
# Run them against an installed anvl with
#
#   Rscript integrations/run.R [filter]
#
# where the optional `filter` is a regular expression on the test file names.

args <- commandArgs(trailingOnly = FALSE)
dir <- normalizePath(dirname(sub("^--file=", "", grep("^--file=", args, value = TRUE)[[1L]])))

library(testthat)
library(anvl)

# Step 1 of the "Custom Calls" article: compile the handler. Its exported
# `hypot_handler_ptr()` lands in the global environment, which the test
# environment inherits from.
cat("Compiling", file.path(dir, "hypot.cpp"), "...\n")
Rcpp::sourceCpp(file.path(dir, "hypot.cpp"))

# Step 2: register it. A CUDA handler would go in the same list under `cuda`.
pjrt::pjrt_register_custom_call("anvl_hypot", list(cpu = hypot_handler_ptr()))

filter <- commandArgs(trailingOnly = TRUE)[1L]
if (is.na(filter)) {
  filter <- NULL
}
test_dir(file.path(dir, "testthat"), filter = filter, stop_on_failure = TRUE)
