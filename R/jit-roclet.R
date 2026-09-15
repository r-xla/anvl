#' @title `@jit` roxygen roclet
#' @description
#' A custom roxygen2 roclet that scans for `@jit` tags on function definitions
#' and emits a registry of functions to wrap in [`jit()`] at package build
#' time.
#'
#' Mark a function with `#' @jit` (optionally `#' @jit static = c(...)`) to
#' request that the function be replaced in the package namespace by
#' `jit(f, static = c(...))`.
#'
#' @section Setting up the roclet in a package:
#' 1. Add `anvl` to your package's `Imports`.
#' 2. Activate the roclet in the `Roxygen` field of `DESCRIPTION`:
#'
#'    ```
#'    Roxygen: list(markdown = TRUE,
#'                  roclets = c("namespace", "rd", "anvl::jit_roclet"))
#'    ```
#'
#' 3. Add `R/jit-registry.R` to the `Collate` field, **before** `R/zzz.R`.
#'    `devtools::document()` will (re)generate this file on every run.
#' 4. In `R/zzz.R`, apply the registry at the top level (right next to
#'    `.onLoad`), so the wrapped functions are byte-compiled during install:
#'
#'    ```r
#'    .onLoad <- function(libname, pkgname) { ... }
#'
#'    anvl::apply_jit_registry(.jit_registry)
#'    ```
#'
#' 5. Tag the functions you want jitted:
#'
#'    ```r
#'    #' @export
#'    #' @jit static = c("flag")
#'    my_fun <- function(x, flag) if (flag) x + 1 else x * 2
#'    ```
#'
#' 6. Run `devtools::document()`. The first run requires that the previous
#'    install of your package already exports the roclet's pieces; on a fresh
#'    setup, omit the custom roclet from `Roxygen`, run `document()` once,
#'    install, and then re-enable it. Subsequent runs work as normal.
#'
#' @section Tag syntax:
#' - `#' @jit` — no static arguments.
#' - `#' @jit static = c("a", "b")` — names of formals to treat as static.
#' - `#' @jit static = c(1L, 2L)` — positions of formals to treat as static.
#' - `#' @jit static = 2:5` — the value is evaluated as R, so ranges and any
#'   other expression yielding a character/integer vector are allowed.
#' - `#' @jit static 2:5` — terse form; the `=` may be omitted.
#'
#' Anything other than a `static = ...` argument is rejected.
#'
#' @seealso [`apply_jit_registry()`], [`jit()`]
#' @return A roxygen2 roclet object.
#' @export
jit_roclet <- function() {
  roclet <- tryCatch(
    roxygen2_fn("roclet"),
    error = function(e) {
      cli_abort("`jit_roclet()` requires the {.pkg roxygen2} package to be installed.")
    }
  )
  roclet("jit")
}

# roxygen2 is only used at package-build time (when this roclet runs), so it is
# deliberately not declared in Imports/Suggests. Reach into its functions via
# getFromNamespace() rather than `::` (or `requireNamespace()`), so R CMD check's
# "checking dependencies in R code" does not flag an undeclared dependency.
roxygen2_fn <- function(name) {
  getFromNamespace(name, "roxygen2")
}

roxy_tag_parse.roxy_tag_jit <- function(x) {
  raw <- trimws(x$raw %||% "")
  if (!nzchar(raw)) {
    x$val <- list(static = character())
    return(x)
  }
  # Allow the terse `static <expr>` form (no `=`) as sugar for
  # `static = <expr>`, e.g. `@jit static 2:5`.
  if (grepl("^static\\s", raw) && !grepl("^static\\s*=", raw)) {
    raw <- sub("^static\\s+", "static = ", raw)
  }
  expr <- tryCatch(
    parse(text = paste0("list(", raw, ")"))[[1L]],
    error = function(e) e
  )
  if (inherits(expr, "error")) {
    return(roxygen2_fn("warn_roxy_tag")(x, "could not parse `@jit` arguments"))
  }
  val <- tryCatch(eval(expr, envir = baseenv()), error = function(e) e)
  if (inherits(val, "error") || !is.list(val)) {
    return(roxygen2_fn("warn_roxy_tag")(x, "could not evaluate `@jit` arguments"))
  }
  known <- "static"
  unknown <- setdiff(names(val), c("", known))
  if (length(unknown)) {
    return(roxygen2_fn("warn_roxy_tag")(
      x,
      sprintf("unknown `@jit` argument(s): %s", paste(unknown, collapse = ", "))
    ))
  }
  static <- val$static %||% character()
  if (is.numeric(static)) {
    static <- as.integer(static)
  } else if (!is.character(static)) {
    return(roxygen2_fn("warn_roxy_tag")(x, "`static` must be character or integer"))
  }
  x$val <- list(static = static)
  x
}

roxy_tag_rd.roxy_tag_jit <- function(x, base_path, env) {
  NULL
}

roclet_process.roclet_jit <- function(x, blocks, env, base_path) {
  entries <- list()
  for (block in blocks) {
    tag <- roxygen2_fn("block_get_tag")(block, "jit")
    if (is.null(tag)) {
      next
    }
    name <- block$object$alias %||% block$object$name %||% block$object$topic
    if (is.null(name)) {
      roxygen2_fn("warn_roxy_tag")(tag, "`@jit` requires a named function")
      next
    }
    obj <- block$object$value
    if (!is.function(obj)) {
      roxygen2_fn("warn_roxy_tag")(tag, "`@jit` may only be applied to functions")
      next
    }
    entries[[length(entries) + 1L]] <- list(
      name = name,
      static = tag$val$static
    )
  }
  entries
}

roclet_output.roclet_jit <- function(x, results, base_path, ...) {
  path <- file.path(base_path, "R", "jit-registry.R")
  lines <- jit_registry_lines(results)
  current <- if (file.exists(path)) readLines(path, warn = FALSE) else character()
  if (!identical(current, lines)) {
    writeLines(lines, path)
  }
  path
}

roclet_clean.roclet_jit <- function(x, base_path) {
  path <- file.path(base_path, "R", "jit-registry.R")
  if (file.exists(path)) {
    unlink(path)
  }
  invisible()
}

# Render the registry list as R source. Kept as a single deparsed list so the
# file is small and trivially diffable.
jit_registry_lines <- function(entries) {
  header <- c(
    "# Generated by anvl::jit_roclet: do not edit by hand",
    "",
    "# Functions tagged with `#' @jit` are rebound at package build time to",
    "# `jit(f, static = <static>)`. See R/jit-roclet.R.",
    ""
  )
  if (!length(entries)) {
    return(c(header, ".jit_registry <- list()"))
  }
  body <- vapply(entries, jit_registry_entry_line, character(1L))
  c(
    header,
    ".jit_registry <- list(",
    paste0("  ", body, c(rep(",", length(body) - 1L), "")),
    ")"
  )
}

jit_registry_entry_line <- function(entry) {
  sprintf(
    "list(name = %s, static = %s)",
    deparse(entry$name),
    paste(deparse(entry$static), collapse = "")
  )
}
