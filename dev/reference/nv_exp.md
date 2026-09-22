# Exponential

Element-wise exponential. You can also use
[`exp()`](https://rdrr.io/r/base/Log.html).

## Usage

``` r
nv_exp(x)
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

[`prim_exp()`](https://r-xla.github.io/anvl/dev/reference/prim_exp.md)
for the underlying primitive.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(0, 1, 2))
exp(x)
#> AnvlArray
#>  1.0000
#>  2.7183
#>  7.3891
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
nv_exp(1)
#> AnvlArray
#>  2.7183
#> [ CPUf32{} ] 
```
