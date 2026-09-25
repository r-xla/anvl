# Truncate

Element-wise truncation (round toward zero). You can also use
[`trunc()`](https://rdrr.io/r/base/Round.html).

## Usage

``` r
nv_trunc(x)

# S3 method for class 'AnvlArray'
trunc(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any numeric data type: a float is rounded and keeps
  its own, and an integer one is already whole and is returned
  unchanged. An R value materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and is treated in the same way.

- ...:

  Not used; must be empty.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape and data type.

## See also

[`nv_floor()`](https://r-xla.github.io/anvl/dev/reference/nv_floor.md),
[`nv_ceiling()`](https://r-xla.github.io/anvl/dev/reference/nv_ceiling.md),
[`nv_round()`](https://r-xla.github.io/anvl/dev/reference/nv_round.md).

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1.2, 2.7, -1.5))
trunc(x)
#> AnvlArray
#>   1
#>   2
#>  -1
#> [ CPUf32{3} ] 
trunc(nv_array(1:3)) # an integer array is already whole
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUi32{3} ] 
```
