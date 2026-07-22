# Quickr backend

Constructs the quickr backend, which stores array data as plain R arrays
and compiles jitted functions to R code via the
[quickr](https://CRAN.R-project.org/package=quickr) package.

## Usage

``` r
AnvlBackendQuickr()
```

## Value

An
[`AnvlBackend`](https://r-xla.github.io/anvl/dev/reference/AnvlBackend.md)
object with subclass `"AnvlBackendQuickr"`.

## Details

To use it, the `"quickr"` package needs to be installed.

Registered automatically under the name `"quickr"` when the package is
loaded; call
[`local_backend("quickr")`](https://r-xla.github.io/anvl/dev/reference/local_backend.md)
or
[`with_backend("quickr", ...)`](https://r-xla.github.io/anvl/dev/reference/with_backend.md)
to use it. Requires the quickr package to be installed.

## Data representation

An
[`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
with `backend = "quickr"` is, under the hood, a plain R vector or array
(`numeric`, `integer`, or `logical`) stored in the `$data` field.
[`as_array()`](https://r-xla.github.io/anvl/dev/reference/as_array.md)
returns the underlying vector/array directly without copying, and
[`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
simply wraps an R vector/array. Data always lives in R's memory and
computation always runs on the CPU, so the only device is
[`quickr_device("cpu")`](https://r-xla.github.io/anvl/dev/reference/quickr_device.md);
every array still carries it in `$device`, as arrays of every backend
do.

## Status

This backend is **experimental** and has a number of limitations:

- Compilation (tracing + quickr lowering) is somewhat slow, so it is
  best suited to long-running or repeatedly-called functions where the
  one-time compilation cost is amortized.

- Only a subset of the primitives that the PJRT backend supports are
  currently lowered to quickr code. See
  [`vignette("primitives")`](https://r-xla.github.io/anvl/dev/articles/primitives.md)
  for an overview.

- Only the data types `f64`, `i32`, and `bool` are supported.

- Only CPU execution is supported.

## Quickr JIT arguments

- `unwrap` (`logical(1)`, default `FALSE`): if `TRUE`, the compiled
  function returns plain R arrays instead of
  [`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)s.
  Useful when the jitted function's output is consumed by non-anvl R
  code and the extra wrapping would only get stripped again.

## See also

[`AnvlBackend()`](https://r-xla.github.io/anvl/dev/reference/AnvlBackend.md),
[`AnvlBackendPjrt()`](https://r-xla.github.io/anvl/dev/reference/AnvlBackendPjrt.md),
[`local_backend()`](https://r-xla.github.io/anvl/dev/reference/local_backend.md),
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md).
