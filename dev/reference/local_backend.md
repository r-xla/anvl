# Temporarily Set the Backend

Set the `anvl.backend` option for a scope: `local_backend()` until the
calling frame exits, `with_backend()` for the duration of `code`. Every
array built and every operation run in that scope uses the backend, and
R values materialize at its default data types (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).

## Usage

``` r
local_backend(backend, envir = parent.frame())

with_backend(backend, code)
```

## Arguments

- backend:

  (`character(1)`)  
  Backend to use (`"pjrt"` or `"quickr"`).

- envir:

  (`environment`)  
  The environment to scope the change to.

- code:

  (any)  
  An expression to evaluate with the given backend.

## Value

`local_backend()` returns the previous value of the option, as
`list(anvl.backend = )`, invisibly. `with_backend()` returns the result
of evaluating `code`.

## See also

[`active_backend()`](https://r-xla.github.io/anvl/dev/reference/active_backend.md)

## Examples

``` r
f <- function() {
  local_backend("quickr")
  active_backend()
}
f()
#> [1] "quickr"
active_backend()
#> [1] "pjrt"
with_backend("quickr", active_backend())
#> [1] "quickr"
```
