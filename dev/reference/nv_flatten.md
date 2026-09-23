# Flatten

Flattens an array with one or more axes into a 1-D array, using
col-major semantics.

## Usage

``` r
nv_flatten(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's data type, and one axis holding all of its elements – so
a scalar, which has none, becomes a length-1 vector.

## Examples

``` r
# the 2x2 matrix becomes a length-4 vector
x <- matrix(1:4, nrow = 2)
nv_flatten(x)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#> [ CPUi32{4} ] 
# the same column-major order base R uses
c(x)
#> [1] 1 2 3 4
```
