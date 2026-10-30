skip_if_no_quickr <- function() {
  testthat::skip_if_not_installed("quickr")
  if (nzchar(Sys.getenv("ANVL_TEST_SKIP_QUICKR", ""))) {
    testthat::skip("ANVL_TEST_SKIP_QUICKR is set")
  }
}
