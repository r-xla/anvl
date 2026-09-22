# Convert to an R array

Transfers array data to R and returns it as an R
[`array`](https://rdrr.io/r/base/array.html). Only in the case of
scalars is the result a vector of length 1, as R `arrays` cannot have 0
axes.

## Usage

``` r
# S3 method for class 'AnvlArray'
as_array(x, check = "warn", ...)

as_array(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  An array-like object.

- check:

  (`character(1)` \| `FALSE`)  
  How to report a materialized value that the R type cannot hold:
  `"warn"` (the default) warns and returns it anyway, `"err"` aborts,
  and `FALSE` skips the scan. `TRUE` is not accepted – with two levels
  of strictness it does not say which one is meant. Forwarded to the
  backend; for the `pjrt` backend the cases scanned for are `i32`/`i64`
  values colliding with the `NA` bit pattern and `ui64` values `>= 2^63`
  wrapping through
  [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html).
  See
  [`pjrt::as_array.PJRTBuffer()`](https://r-xla.github.io/pjrt/reference/as_array.PJRTBuffer.html)
  for the full list, and the "Gotchas" vignette.

- ...:

  Additional arguments passed to methods (unused).

## Value

([`array`](https://rdrr.io/r/base/array.html) \| `vector(1)`)  
An R array with the input's shape, or – for a scalar, which R cannot
represent as an array – a vector of length 1.

## Details

This is implemented via the generic
[`tengen::as_array()`](https://r-xla.github.io/tengen/reference/as_array.html).

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
as_array(x)
#> [1] 1 2 3 4
y <- nv_scalar(1L)
# R arrays can't have 0 axes:
as_array(y)
#> [1] 1
```
