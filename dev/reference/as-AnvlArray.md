# Coerce AnvlArray to an R Vector

Convert an
[`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
to a flat R vector, discarding the array's shape. Each method requires a
compatible dtype:

- [`as.double()`](https://rdrr.io/r/base/double.html) /
  [`as.numeric()`](https://rdrr.io/r/base/numeric.html): float or
  (signed/unsigned) integer dtypes.

- [`as.integer()`](https://rdrr.io/r/base/integer.html): signed or
  unsigned integer dtypes.

- [`bit64::as.integer64()`](https://bit64.r-lib.org/reference/as.integer64.character.html):
  signed or unsigned integer dtypes. This is how to read `i64`, `ui64`
  and `ui32` values that an R `integer` cannot hold. It is lossless for
  `i64` and `ui32`, but
  [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html)
  is itself signed, so a `ui64` value `>= 2^63` wraps to a negative one
  (exactly `2^63` becomes `NA`); that is warned about, and
  `check = "err"` makes it an error.

- [`as.logical()`](https://rdrr.io/r/base/logical.html): `bool`.

- [`as.vector()`](https://rdrr.io/r/base/vector.html): any dtype; the R
  type is chosen by the dtype. For the dtypes R has no native type for
  (`i64`, `ui64`, `ui32`) that is the
  [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html)
  [`as_array()`](https://r-xla.github.io/anvl/dev/reference/as_array.md)
  returns, since a bare double could not hold the values – and it keeps
  its class, so [`is.vector()`](https://rdrr.io/r/base/vector.html) is
  `FALSE` for it.

Use
[`as_array()`](https://r-xla.github.io/anvl/dev/reference/as_array.md)
to obtain an R array that preserves the shape, or
[`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md)
to change the dtype of an
[`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
before coercing. [`as.vector()`](https://rdrr.io/r/base/vector.html)'s
signature is fixed by the generic, so it takes no `check` argument and
always reports at the default level; call
[`as_array()`](https://r-xla.github.io/anvl/dev/reference/as_array.md)
directly to pick another one.

## Usage

``` r
# S3 method for class 'AnvlArray'
as.double(x, check = "warn", ...)

# S3 method for class 'AnvlArray'
as.integer(x, check = "warn", ...)

# S3 method for class 'AnvlArray'
as.integer64(x, check = "warn", ...)

# S3 method for class 'AnvlArray'
as.logical(x, check = "warn", ...)

# S3 method for class 'AnvlArray'
as.vector(x, mode = "any")
```

## Arguments

- x:

  ([`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))  
  Array to coerce.

- check:

  (`character(1)` \| `FALSE`)  
  Forwarded to
  [`as_array()`](https://r-xla.github.io/anvl/dev/reference/as_array.md);
  see there for details.

- ...:

  Unused.

- mode:

  (`character(1)`)  
  Must be `"any"` (the default), meaning the natural R type for the
  array's dtype. Only present because
  [`base::as.vector()`](https://rdrr.io/r/base/vector.html)'s signature
  requires it; pick an R type with one of the other methods instead.

## Value

(`vector`)

## Examples

``` r
x <- nv_array(c(1.5, 2.5, 3.5, 4.5), shape = c(2L, 2L))
as.numeric(x)
#> [1] 1.5 2.5 3.5 4.5
as.integer(nv_array(1:6, shape = c(2L, 3L)))
#> [1] 1 2 3 4 5 6
bit64::as.integer64(nv_array(1:6, shape = c(2L, 3L), dtype = "i64"))
#> integer64
#> [1] 1 2 3 4 5 6
as.logical(nv_array(c(TRUE, FALSE), dtype = "bool"))
#> [1]  TRUE FALSE
as.vector(x)
#> [1] 1.5 2.5 3.5 4.5
```
