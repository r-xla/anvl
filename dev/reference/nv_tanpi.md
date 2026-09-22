# Tangent of a Multiple of Pi

Element-wise `tan(pi * x)`. You can also use
[`tanpi()`](https://rdrr.io/r/base/Trig.html). Like base R's
[`base::tanpi()`](https://rdrr.io/r/base/Trig.html), it is exact for a
whole argument and `NaN` at the half integers, where the tangent has its
poles.

## Usage

``` r
nv_tanpi(x)
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

[`nv_sinpi()`](https://r-xla.github.io/anvl/dev/reference/nv_sinpi.md),
[`nv_cospi()`](https://r-xla.github.io/anvl/dev/reference/nv_cospi.md),
[`nv_tan()`](https://r-xla.github.io/anvl/dev/reference/nv_tan.md)

## Examples

``` r
tanpi(nv_array(c(0, 0.25, 0.5, 1)))
#> AnvlArray
#>    0
#>    1
#>  nan
#>    0
#> [ CPUf32{4} ] 
```
