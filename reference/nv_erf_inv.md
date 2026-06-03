# Inverse Error Function

Element-wise inverse error function (the inverse of `erf` on `(-1, 1)`).

## Usage

``` r
nv_erf_inv(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_erf_inv()`](https://r-xla.github.io/anvl/reference/prim_erf_inv.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(-0.5, 0, 0.5))
nv_erf_inv(x)
#> AnvlArray
#>  -0.4769
#>   0.0000
#>   0.4769
#> [ CPUf32{3} ] 
```
