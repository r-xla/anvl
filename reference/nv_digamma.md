# Digamma

Element-wise digamma function (logarithmic derivative of the gamma
function). You can also use
[`digamma()`](https://rdrr.io/r/base/Special.html).

## Usage

``` r
nv_digamma(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_digamma()`](https://r-xla.github.io/anvl/reference/prim_digamma.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(0.5, 1, 2, 5))
digamma(x)
#> AnvlArray
#>  -1.9635
#>  -0.5772
#>   0.4228
#>   1.5061
#> [ CPUf32{4} ] 
```
