# Temporarily set the backend

Sets the `anvl.backend` option for the duration of the calling scope.
Every array built and every operation run in that scope uses the
backend, and R values materialize at its default data types (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).

## Usage

``` r
local_backend(backend, envir = parent.frame())
```

## Arguments

- backend:

  (`character(1)`)  
  Backend to use (`"pjrt"` or `"quickr"`).

- envir:

  (`environment`)  
  The environment to scope the change to.

## Value

(named `list`)  
The previous value of the option, as `list(anvl.backend = )`, invisibly.

## See also

[`active_backend()`](https://r-xla.github.io/anvl/dev/reference/active_backend.md),
[`with_backend()`](https://r-xla.github.io/anvl/dev/reference/with_backend.md)

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
```
