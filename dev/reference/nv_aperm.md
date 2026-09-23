# Transpose

Permutes the axes of an array, like
[`base::aperm()`](https://rdrr.io/r/base/aperm.html). `nv_transpose()`
is another spelling of the same function. You can also use
[`aperm()`](https://rdrr.io/r/base/aperm.html), or
[`t()`](https://rdrr.io/r/base/t.html) for matrices.

## Usage

``` r
nv_aperm(x, perm = NULL)

nv_transpose(x, perm = NULL)

# S3 method for class 'AnvlArray'
aperm(a, perm = NULL, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- perm:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  New ordering of axes. If `NULL` (default), reverses the axes. Negative
  values count from the end, i.e. `-1` refers to the last axis.

- a:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The array whose axes to permute.

- ...:

  No additional arguments.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has `x`'s data type and shape `shape(x)[perm]`, or `rev(shape(x))` when
`perm` is `NULL`.

## The [`t()`](https://rdrr.io/r/base/t.html) generic

[`t()`](https://rdrr.io/r/base/t.html) requires a matrix, whereas
[`base::t()`](https://rdrr.io/r/base/t.html) also transposes a vector
(into a one-row matrix) and reverses the axes of a higher-rank array.

## The [`aperm()`](https://rdrr.io/r/base/aperm.html) generic

[`aperm()`](https://rdrr.io/r/base/aperm.html) permutes the axes exactly
as `nv_aperm()` does; like
[`base::aperm()`](https://rdrr.io/r/base/aperm.html), `perm = NULL`
reverses them.

## See also

[`prim_transpose()`](https://r-xla.github.io/anvl/dev/reference/prim_transpose.md)
for the underlying primitive.

## Examples

``` r
# the 2x3 becomes a 3x2, keeping its data type
x <- nv_matrix(1:6, nrow = 2)
nv_aperm(x)
#> AnvlArray
#>  1 2
#>  3 4
#>  5 6
#> [ CPUi32{3,2} ] 
t(x)
#> AnvlArray
#>  1 2
#>  3 4
#>  5 6
#> [ CPUi32{3,2} ] 
nv_aperm(nv_array(1:24, shape = c(2, 3, 4)), c(3, 1, 2))
#> AnvlArray
#> (1,.,.) =
#>  1 3 5
#>  2 4 6
#> 
#> (2,.,.) =
#>   7  9 11
#>   8 10 12
#> 
#> (3,.,.) =
#>  13 15 17
#>  14 16 18
#> 
#> (4,.,.) =
#>  19 21 23
#>  20 22 24
#> [ CPUi32{4,2,3} ] 
```
