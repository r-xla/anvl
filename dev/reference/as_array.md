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
  for the full list, and the
  [Gotchas](https://r-xla.github.io/anvl/articles/gotchas.html) article.

- ...:

  Additional arguments passed to methods (unused).

## Value

([`array`](https://rdrr.io/r/base/array.html) \| `vector(1)`)  
An R array with the input's shape, or – for a scalar, which R cannot
represent as an array – a vector of length 1.

## Details

This is implemented via the generic
[`tengen::as_array()`](https://r-xla.github.io/tengen/reference/as_array.html).

## Data types

R has fewer data types than anvl, so the values are converted to the R
type that can represent them:

|  |  |
|----|----|
| Data type | R type |
| `f32`, `f64` | `double` |
| `i8`, `i16`, `i32`, `ui8`, `ui16` | `integer` |
| `i64`, `ui32`, `ui64` | [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html) |
| `bool` | `logical` |

This has two consequences:

- An `f32` value is widened to a `double` and keeps the rounding error
  of the 32-bit float, e.g. `as_array(nv_scalar(0.1))` is not exactly
  `0.1`.

- Some integer values cannot be represented in R: an `i32` or `i64`
  value equal to the smallest representable integer is read as `NA`, and
  a `ui64` value `>= 2^63` wraps to a negative number. The `check`
  argument decides whether this is reported.

To obtain a different R type, convert the array with
[`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md)
first, or use the coercion functions described in
[`as.double()`](https://r-xla.github.io/anvl/dev/reference/as-AnvlArray.md).

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
