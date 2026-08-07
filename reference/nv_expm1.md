# Exponential Minus One

Element-wise `exp(x) - 1`, more accurate for small `x`.

## Usage

``` r
nv_expm1(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_expm1()`](https://r-xla.github.io/anvl/reference/prim_expm1.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(0, 0.001, 1))
nv_expm1(x)
#> AnvlArray
#>  0.0000
#>  0.0010
#>  1.7183
#> [ CPUf32{3} ] 
```
