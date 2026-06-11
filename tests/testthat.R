# This file is part of the standard setup for testthat.
# It is recommended that you do not modify it.
#
# Where should you do additional test configuration?
# Learn more about the roles of various files in:
# * https://r-pkgs.org/testing-design.html#sec-tests-files-overview
# * https://testthat.r-lib.org/articles/special-files.html

library(testthat)
library(anvl)
library(stablehlo)

# Nearly all tests require a PJRT plugin, which has to be downloaded over the
# network. That is not possible on CRAN, so the suite is opt-in: set
# `ANVL_TEST=1` to run it (CI does this). When unset, no tests run.
if (Sys.getenv("ANVL_TEST", unset = "0") == "1") {
  test_check("anvl")
}
