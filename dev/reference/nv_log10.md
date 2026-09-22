# Base-10 Logarithm

Element-wise base-10 logarithm. You can also use
[`log10()`](https://rdrr.io/r/base/Log.html).

## Usage

``` r
nv_log10(x)
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

[`nv_log()`](https://r-xla.github.io/anvl/dev/reference/nv_log.md),
[`nv_log2()`](https://r-xla.github.io/anvl/dev/reference/nv_log2.md)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1, 10, 100, 1000))
nv_log10(x)
#> AnvlArray
#>  0.0000
#>  1.0000
#>  2.0000
#>  3.0000
#> [ CPUf32{4} ] 

# an R value materializes at its default data type
nv_log10(100)
#> AnvlArray
#>  2.0000
#> [ CPUf32{} ] 
```
