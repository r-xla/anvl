# Coerce AnvlArray to an R Vector

Convert an
[`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md) to a
bare R vector. The array's shape is discarded; the result is always a
flat vector. Each method requires a compatible dtype:

- [`as.double()`](https://rdrr.io/r/base/double.html) /
  [`as.numeric()`](https://rdrr.io/r/base/numeric.html): float or
  (signed/unsigned) integer dtypes.

- [`as.integer()`](https://rdrr.io/r/base/integer.html): signed or
  unsigned integer dtypes.

- [`as.logical()`](https://rdrr.io/r/base/logical.html): `bool`.

- [`as.vector()`](https://rdrr.io/r/base/vector.html): any dtype; the R
  type is chosen by the dtype, or forced via `mode` (e.g. `"integer"`,
  `"double"`, `"logical"`, `"list"`).

Use [`as_array()`](https://r-xla.github.io/anvl/reference/as_array.md)
to obtain an R array that preserves the shape, or
[`nv_convert()`](https://r-xla.github.io/anvl/reference/nv_convert.md)
to change the dtype of an
[`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md)
before coercing.

## Usage

``` r
# S3 method for class 'AnvlArray'
as.double(x, ...)

# S3 method for class 'AnvlArray'
as.integer(x, ...)

# S3 method for class 'AnvlArray'
as.logical(x, ...)

# S3 method for class 'AnvlArray'
as.vector(x, mode = "any")
```

## Arguments

- x:

  ([`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md))  
  Array to coerce.

- ...:

  Unused.

- mode:

  (`character(1)`)  
  For [`as.vector()`](https://rdrr.io/r/base/vector.html) only. See
  [`base::as.vector()`](https://rdrr.io/r/base/vector.html). Defaults to
  `"any"`, meaning the natural R type for the array's dtype.

## Value

An R vector of the corresponding type (`double`, `integer`, or
`logical`).

## Examples

``` r
x <- nv_array(c(1.5, 2.5, 3.5, 4.5), shape = c(2L, 2L))
as.numeric(x)
#> [1] 1.5 2.5 3.5 4.5
as.integer(nv_array(1:6, shape = c(2L, 3L)))
#> [1] 1 2 3 4 5 6
as.logical(nv_array(c(TRUE, FALSE), dtype = "bool"))
#> [1]  TRUE FALSE
as.vector(x)
#> [1] 1.5 2.5 3.5 4.5
```
