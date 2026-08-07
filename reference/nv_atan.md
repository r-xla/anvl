# Arc Tangent

Element-wise inverse tangent. You can also use
[`atan()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_atan(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_atan()`](https://r-xla.github.io/anvl/reference/prim_atan.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
atan(x)
#> AnvlArray
#>  -0.7854
#>   0.0000
#>   0.7854
#> [ CPUf32{3} ] 
```
