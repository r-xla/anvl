# Convert an array to a raw vector

Returns the underlying bytes of an array as a
[raw](https://rdrr.io/r/base/raw.html) vector.

## Usage

``` r
as_raw(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  An array-like object.

- ...:

  Additional arguments passed to method:

  - `row_major` (`logical(1)`)  
    Whether to write the bytes in row-major order.

## Value

([`raw`](https://rdrr.io/r/base/raw.html))

## Details

This is implemented via the generic
[`tengen::as_raw()`](https://r-xla.github.io/tengen/reference/as_raw.html).

## Examples

``` r
x <- nv_array(1:4, shape = c(2, 2), dtype = "f32")
as_raw(x, row_major = TRUE)
#>  [1] 00 00 80 3f 00 00 40 40 00 00 00 40 00 00 80 40
as_raw(x, row_major = FALSE)
#>  [1] 00 00 80 3f 00 00 00 40 00 00 40 40 00 00 80 40
```
