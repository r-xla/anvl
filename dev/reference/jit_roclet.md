# `@jit` roxygen roclet

A custom roxygen2 roclet that scans for `@jit` tags on function
definitions and emits a registry of functions to wrap in
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) at package
build time.

Mark a function with `#' @jit` (optionally `#' @jit static = c(...)`) to
request that the function be replaced in the package namespace by
`jit(f, backend = "auto", static = c(...))`.

## Usage

``` r
jit_roclet()
```

## Value

A roxygen2 roclet object.

## Setting up the roclet in a package

1.  Add `anvl` to your package's `Imports`.

2.  Activate the roclet in the `Roxygen` field of `DESCRIPTION`:

        Roxygen: list(markdown = TRUE,
                      roclets = c("namespace", "rd", "anvl::jit_roclet"))

3.  Add `R/jit-registry.R` to the `Collate` field, **before** `R/zzz.R`.
    `devtools::document()` will (re)generate this file on every run.

4.  In `R/zzz.R`, apply the registry at the top level (right next to
    `.onLoad`), so the wrapped functions are byte-compiled during
    install:

        .onLoad <- function(libname, pkgname) { ... }

        anvl::apply_jit_registry(.jit_registry)

5.  Tag the functions you want jitted:

        #' @export
        #' @jit static = c("flag")
        my_fun <- function(x, flag) if (flag) x + 1 else x * 2

6.  Run `devtools::document()`. The first run requires that the previous
    install of your package already exports the roclet's pieces; on a
    fresh setup, omit the custom roclet from `Roxygen`, run `document()`
    once, install, and then re-enable it. Subsequent runs work as
    normal.

## Tag syntax

- `#' @jit` — no static arguments.

- `#' @jit static = c("a", "b")` — names of formals to treat as static.

- `#' @jit static = c(1L, 2L)` — positions of formals to treat as
  static.

- `#' @jit static = 2:5` — the value is evaluated as R, so ranges and
  any other expression yielding a character/integer vector are allowed.

- `#' @jit static 2:5` — terse form; the `=` may be omitted.

Anything other than a `static = ...` argument is rejected.

## See also

[`apply_jit_registry()`](https://r-xla.github.io/anvl/dev/reference/apply_jit_registry.md),
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)
