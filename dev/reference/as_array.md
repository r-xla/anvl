# Convert to an R array

Transfers array data to R and returns it as an R
[`array`](https://rdrr.io/r/base/array.html). Only in the case of
scalars is the result a vector of length 1, as R `arrays` cannot have 0
dimensions.

## Usage

``` r
# S3 method for class 'AnvlArray'
as_array(x, check = FALSE, ...)

as_array(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  An array-like object.

- check:

  (`logical(1)`)  
  If `TRUE`, sanity-check the materialized R vector against losing
  information across the device-to-host boundary, and abort if any
  problematic value is detected. Forwarded to the backend; for the
  `pjrt` backend the relevant cases are `i32`/`i64` values colliding
  with the `NA` bit pattern and `ui64` values `>= 2^63` wrapping through
  [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html).
  See
  [`pjrt::as_array.PJRTBuffer()`](https://r-xla.github.io/pjrt/reference/as_array.PJRTBuffer.html)
  for the full list. Defaults to `FALSE`. See the "Gotchas" vignette.

- ...:

  Additional arguments passed to methods (unused).

## Value

An R [`array`](https://rdrr.io/r/base/array.html) or `vector` of length
1.

## Details

This is implemented via the generic
[`tengen::as_array()`](https://r-xla.github.io/tengen/reference/as_array.html).

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
as_array(x)
#> [1] 1 2 3 4
y <- nv_scalar(1L)
# R arrays can't have 0 dimensions:
as_array(y)
#> [1] 1
```
