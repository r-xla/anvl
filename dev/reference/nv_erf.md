# Error Function

Element-wise error function
`erf(x) = (2 / sqrt(pi)) * integral_0^x exp(-t^2) dt`.

## Usage

``` r
nv_erf(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_erf()`](https://r-xla.github.io/anvl/dev/reference/prim_erf.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
nv_erf(x)
#> AnvlArray
#>  -0.8427
#>   0.0000
#>   0.8427
#> [ CPUf32{3} ] 
```
