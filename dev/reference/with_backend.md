# Run code with a specific backend

Sets the `anvl.backend` option for the duration of the expression. Every
array built and every operation run in `code` uses the backend.

## Usage

``` r
with_backend(backend, code)
```

## Arguments

- backend:

  (`character(1)`)  
  Backend to use (`"pjrt"` or `"quickr"`).

- code:

  An expression to evaluate with the given backend.

## Value

The result of evaluating `code`.
