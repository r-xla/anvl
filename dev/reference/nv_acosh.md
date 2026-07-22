# Inverse Hyperbolic Cosine

Element-wise inverse hyperbolic cosine. You can also use
[`acosh()`](https://rdrr.io/r/base/Hyperbolic.html).

## Usage

``` r
nv_acosh(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_acosh()`](https://r-xla.github.io/anvl/dev/reference/prim_acosh.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 2, 10))
acosh(x)
#> AnvlArray
#>  0.0000
#>  1.3170
#>  2.9932
#> [ CPUf32{3} ] 
```
