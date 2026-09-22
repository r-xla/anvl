# shamelessly copied from: https://github.com/tidyverse/readr/blob/e529cb2775f1b52a0dfa30dabc9f8e0014aa77e6/R/zzz.R
register_s3_method <- function(pkg, generic, class, fun = NULL) {
  if (is.null(fun)) {
    fun <- get(paste0(generic, ".", class), envir = parent.frame())
  } else {
    stopifnot(is.function(fun))
  }

  if (pkg %in% loadedNamespaces()) {
    registerS3method(generic, class, fun, envir = asNamespace(pkg))
  }

  # Always register hook in case package is later unloaded & reloaded
  setHook(
    packageEvent(pkg, "onLoad"),
    function(...) {
      registerS3method(generic, class, fun, envir = asNamespace(pkg))
    },
    action = "append"
  )
}

.onLoad <- function(libname, pkgname) {
  # fmt: skip
  globals$ranges_raw <- list(
    ui8  = minmax_raw(8L, FALSE),
    ui16 = minmax_raw(16L, FALSE),
    ui32 = minmax_raw(32L, FALSE),
    ui64 = minmax_raw(64L, FALSE),
    i8   = minmax_raw(8L, TRUE),
    i16  = minmax_raw(16L, TRUE),
    i32  = minmax_raw(32L, TRUE),
    i64  = minmax_raw(64L, TRUE)
  )

  # Register compare_proxy for waldo/testthat
  register_s3_method("waldo", "compare_proxy", "AnvlArray")
}
