# Square Root

Element-wise square root. You can also use
[`sqrt()`](https://rdrr.io/r/base/MathFun.html).

## Usage

``` r
nv_sqrt(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_sqrt()`](https://r-xla.github.io/anvl/dev/reference/prim_sqrt.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 4, 9))
sqrt(x)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
