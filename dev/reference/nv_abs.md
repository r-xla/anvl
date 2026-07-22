# Absolute Value

Element-wise absolute value. You can also use
[`abs()`](https://rdrr.io/r/base/MathFun.html).

## Usage

``` r
nv_abs(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_abs()`](https://r-xla.github.io/anvl/dev/reference/prim_abs.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(-1, 2, -3))
abs(x)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
