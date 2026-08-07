# Sine

Element-wise sine. You can also use
[`sin()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_sin(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_sin()`](https://r-xla.github.io/anvl/reference/prim_sin.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(0, pi / 2, pi))
sin(x)
#> AnvlArray
#>   0.0000e+00
#>   1.0000e+00
#>  -8.7423e-08
#> [ CPUf32{3} ] 
```
