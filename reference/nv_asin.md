# Arc Sine

Element-wise inverse sine. You can also use
[`asin()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_asin(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_asin()`](https://r-xla.github.io/anvl/reference/prim_asin.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
asin(x)
#> AnvlArray
#>  -1.5708
#>   0.0000
#>   1.5708
#> [ CPUf32{3} ] 
```
