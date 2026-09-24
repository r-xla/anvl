# Optional integration with {quickr} (a suggested dependency).

quickr_is_installed <- function() {
  requireNamespace("quickr", quietly = TRUE)
}

assert_quickr_installed <- function(caller) {
  assert_quickr_installed_with(caller, quickr_is_installed())
}

assert_quickr_installed_with <- function(caller, installed) {
  if (isTRUE(installed)) {
    return(invisible(TRUE))
  }
  caller <- if (is.null(caller) || !nzchar(caller)) "this feature" else caller # nocov
  cli_abort("{.pkg quickr} must be installed to use {caller}") # nocov
}

quickr_eager_compile <- function(fun) {
  assert_quickr_installed("{.fn graph_to_quickr_function}")

  # quickr::quick() behaves differently when called from a package namespace
  # (it can create a closure expecting precompiled artifacts). For anvl's use-case
  # we want eager compilation at runtime, so we evaluate quickr::quick() from a
  # non-namespace environment.
  tmp <- new.env(parent = globalenv())
  tmp$fun <- fun
  eval(quote(quickr::quick(fun)), envir = tmp)
}
