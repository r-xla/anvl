# Arc Cosine

Element-wise inverse cosine. You can also use
[`acos()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_acos(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_acos()`](https://r-xla.github.io/anvl/reference/prim_acos.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
acos(x)
#> AnvlArray
#>  3.1416
#>  1.5708
#>  0.0000
#> [ CPUf32{3} ] 
```
