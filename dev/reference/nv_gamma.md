# Gamma Function

Element-wise gamma function. You can also use
[`gamma()`](https://rdrr.io/r/base/Special.html).

## Usage

``` r
nv_gamma(x)
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

[`nv_lgamma()`](https://r-xla.github.io/anvl/dev/reference/nv_lgamma.md),
which is what the hardware computes.

## Examples

``` r
gamma(nv_array(c(0.5, 1, 5, -1.5)))
#> AnvlArray
#>   1.7725
#>   1.0000
#>  24.0000
#>   2.3633
#> [ CPUf32{4} ] 
```
