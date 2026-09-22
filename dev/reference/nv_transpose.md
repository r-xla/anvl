# Transpose

Permutes the axes of an array. You can also use
[`t()`](https://rdrr.io/r/base/t.html) for matrices.

## Usage

``` r
nv_transpose(x, permutation = NULL)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- permutation:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  New ordering of axes. If `NULL` (default), reverses the axes. Negative
  values count from the end, i.e. `-1` refers to the last axis.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has `x`'s data type and shape `shape(x)[permutation]`, or
`rev(shape(x))` when `permutation` is `NULL`.

## The [`t()`](https://rdrr.io/r/base/t.html) generic

[`t()`](https://rdrr.io/r/base/t.html) requires a matrix, whereas
[`base::t()`](https://rdrr.io/r/base/t.html) also transposes a vector
(into a one-row matrix) and reverses the axes of a higher-rank array.

## See also

[`prim_transpose()`](https://r-xla.github.io/anvl/dev/reference/prim_transpose.md)
for the underlying primitive.

## Examples

``` r
# the 2x3 becomes a 3x2, keeping its data type
x <- nv_matrix(1:6, nrow = 2)
t(x)
#> AnvlArray
#>  1 2
#>  3 4
#>  5 6
#> [ CPUi32{3,2} ] 
```
