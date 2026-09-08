# Temporarily set the backend

Sets the `anvl.backend` option for the duration of the calling scope.
Every array built and every operation run in that scope uses the
backend, and R values commit to its default data types (see
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

  The environment to scope the change to.

## Value

The previous value of the option (invisibly).
