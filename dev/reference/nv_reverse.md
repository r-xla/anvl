# Reverse

Reverses the order of elements along specified dimensions.

## Usage

``` r
nv_reverse(x, dims)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reverse.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as `x`.

## See also

[`prim_reverse()`](https://r-xla.github.io/anvl/dev/reference/prim_reverse.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 2, 3, 4, 5))
nv_reverse(x, dims = 1L)
#> AnvlArray
#>  5
#>  4
#>  3
#>  2
#>  1
#> [ CPUf32{5} ] 
```
