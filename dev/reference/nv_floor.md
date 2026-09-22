# Floor

Element-wise floor (round toward negative infinity). You can also use
[`floor()`](https://rdrr.io/r/base/Round.html).

## Usage

``` r
nv_floor(x)
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

[`prim_floor()`](https://r-xla.github.io/anvl/dev/reference/prim_floor.md)
for the underlying primitive.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1.2, 2.7, -1.5))
floor(x)
#> AnvlArray
#>   1
#>   2
#>  -2
#> [ CPUf32{3} ] 
floor(nv_array(1:3)) # an integer array is already whole
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUi32{3} ] 
```
