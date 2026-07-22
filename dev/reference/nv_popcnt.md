# Population Count

Element-wise population count (number of set bits).

## Usage

``` r
nv_popcnt(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_popcnt()`](https://r-xla.github.io/anvl/dev/reference/prim_popcnt.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(7L, 3L, 15L))
nv_popcnt(x)
#> AnvlArray
#>  3
#>  2
#>  4
#> [ CPUi32{3} ] 
```
