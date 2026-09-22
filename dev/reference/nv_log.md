# Natural Logarithm

Element-wise natural logarithm. You can also use
[`log()`](https://rdrr.io/r/base/Log.html).

## Usage

``` r
nv_log(x)
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

[`prim_log()`](https://r-xla.github.io/anvl/dev/reference/prim_log.md)
for the underlying primitive.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1, 2.718, 7.389))
log(x)
#> AnvlArray
#>  0.0000
#>  0.9999
#>  2.0000
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
nv_log(2)
#> AnvlArray
#>  0.6931
#> [ CPUf32{} ] 
```
