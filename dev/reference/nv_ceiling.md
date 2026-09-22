# Ceiling

Element-wise ceiling (round toward positive infinity). You can also use
[`ceiling()`](https://rdrr.io/r/base/Round.html).

## Usage

``` r
nv_ceiling(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any numeric data type: a float is rounded and keeps
  its own, and an integer one is already whole and is returned
  unchanged. An R value materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and is treated in the same way.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape and data type.

## See also

[`prim_ceil()`](https://r-xla.github.io/anvl/dev/reference/prim_ceil.md)
for the underlying primitive.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1.2, 2.7, -1.5))
ceiling(x)
#> AnvlArray
#>   2
#>   3
#>  -1
#> [ CPUf32{3} ] 
ceiling(nv_array(1:3)) # an integer array is already whole
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUi32{3} ] 
```
