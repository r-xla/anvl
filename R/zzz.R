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
  # The set of active backends is fixed here, once, so that a backend cannot
  # appear or disappear underneath already-jitted functions. Changing
  # `anvl.backends` afterwards has no effect.
  activate_backends(getOption("anvl.backends", c("pjrt", "quickr")))

  # fmt: skip
  globals$ranges_raw <- list(
    ui8  = minmax_raw(8, FALSE),
    ui16 = minmax_raw(16, FALSE),
    ui32 = minmax_raw(32, FALSE),
    ui64 = minmax_raw(64, FALSE),
    i8   = minmax_raw(8, TRUE),
    i16  = minmax_raw(16, TRUE),
    i32  = minmax_raw(32, TRUE),
    i64  = minmax_raw(64, TRUE)
  )

  # Register compare_proxy for waldo/testthat
  register_s3_method("waldo", "compare_proxy", "AnvlArray")

  # Register `jit_roclet()`'s S3 methods on roxygen2's generics only if/when
  # roxygen2 is loaded, so roxygen2 stays a build-time-only dependency and
  # does not need to be in Imports or Suggests.
  register_s3_method("roxygen2", "roxy_tag_parse", "roxy_tag_jit")
  register_s3_method("roxygen2", "roxy_tag_rd", "roxy_tag_jit")
  register_s3_method("roxygen2", "roclet_process", "roclet_jit")
  register_s3_method("roxygen2", "roclet_output", "roclet_jit")
  register_s3_method("roxygen2", "roclet_clean", "roclet_jit")

  # Wrap functions tagged with `@jit` (see `jit_roclet()`). This runs after
  # `activate_backends()` because the wrappers bind a concrete backend when
  # only one is active. The namespace is not sealed until `.onLoad()` returns,
  # so the existing bindings can still be replaced here.
  apply_jit_registry(.jit_registry, asNamespace(pkgname))
}
