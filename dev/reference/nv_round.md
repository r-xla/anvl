# Round

Element-wise rounding to a whole number.

## Usage

``` r
nv_round(x, method = "nearest_even")
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any numeric data type: a float is rounded and keeps
  its own, and an integer one is already whole and is returned
  unchanged. An R value materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and is treated in the same way.

- method:

  (`character(1)`)  
  Rounding method. Either `"nearest_even"` (default) or `"afz"` (away
  from zero).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape and data type.

## See also

[`prim_round()`](https://r-xla.github.io/anvl/dev/reference/prim_round.md)
for the underlying primitive.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1.4, 2.5, 3.6))
nv_round(x)
#> AnvlArray
#>  1
#>  2
#>  4
#> [ CPUf32{3} ] 
nv_round(nv_array(1:3)) # an integer array is already whole
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUi32{3} ] 
```
