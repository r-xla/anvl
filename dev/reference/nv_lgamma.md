# Log-Gamma

Element-wise natural logarithm of the absolute value of the gamma
function. You can also use
[`lgamma()`](https://rdrr.io/r/base/Special.html).

## Usage

``` r
nv_lgamma(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any numeric data type: a float keeps its own, and an
  integer one is converted to the default float data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
  An R value materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and is converted in the same way.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape, and its data type – or the default float data
type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
where the input was an integer one.

## See also

[`prim_lgamma()`](https://r-xla.github.io/anvl/dev/reference/prim_lgamma.md)
for the underlying primitive.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(0.5, 1, 2, 5))
lgamma(x)
#> AnvlArray
#>  5.7236e-01
#>  4.7684e-07
#>  0.0000e+00
#>  3.1781e+00
#> [ CPUf32{4} ] 

# an R value materializes at its default data type
nv_lgamma(2)
#> AnvlArray
#>  0
#> [ CPUf32{} ] 
```
