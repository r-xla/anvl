# Logistic (Sigmoid)

Element-wise logistic sigmoid: `1 / (1 + exp(-x))`. This is like
[`stats::plogis()`](https://rdrr.io/r/stats/Logistic.html) with the
default location and scale.

## Usage

``` r
nv_plogis(x)
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

[`prim_plogis()`](https://r-xla.github.io/anvl/dev/reference/prim_plogis.md)
for the underlying primitive.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-2, 0, 2))
nv_plogis(x)
#> AnvlArray
#>  0.1192
#>  0.5000
#>  0.8808
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
nv_plogis(2)
#> AnvlArray
#>  0.8808
#> [ CPUf32{} ] 
```
