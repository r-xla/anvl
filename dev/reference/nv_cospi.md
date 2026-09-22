# Cosine of a Multiple of Pi

Element-wise `cos(pi * x)`. You can also use
[`cospi()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_cospi(x)
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
[`nv_tanpi()`](https://r-xla.github.io/anvl/dev/reference/nv_tanpi.md),
[`nv_cos()`](https://r-xla.github.io/anvl/dev/reference/nv_cos.md)

## Examples

``` r
cospi(nv_array(c(0, 0.5, 1, 1.5)))
#> AnvlArray
#>   1
#>   0
#>  -1
#>   0
#> [ CPUf32{4} ] 
```
