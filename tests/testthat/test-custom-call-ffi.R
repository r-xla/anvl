# The end-to-end custom-call tests compile an FFI handler against {pjrt}'s
# headers with `Rcpp::sourceCpp()`, which needs a C++20 toolchain and is too
# slow for `R CMD check` to pay for on every platform. Set ANVL_TEST_FFI to run
# them; CI does so in the one `R-CMD-check` job that also runs the torch tests.
if (nzchar(Sys.getenv("ANVL_TEST_FFI", ""))) {
  source(system.file("extra-tests", "test-custom-call-ffi.R", package = "anvl"), local = TRUE)
}
