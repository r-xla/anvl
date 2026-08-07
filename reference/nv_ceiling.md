# Ceiling

Element-wise ceiling (round toward positive infinity). You can also use
[`ceiling()`](https://rdrr.io/r/base/Round.html).

## Usage

``` r
nv_ceiling(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_ceil()`](https://r-xla.github.io/anvl/reference/prim_ceil.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(1.2, 2.7, -1.5))
ceiling(x)
#> AnvlArray
#>   2
#>   3
#>  -1
#> [ CPUf32{3} ] 
```
