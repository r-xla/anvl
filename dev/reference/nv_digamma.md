# Digamma

Element-wise digamma function (logarithmic derivative of the gamma
function). You can also use
[`digamma()`](https://rdrr.io/r/base/Special.html).

## Usage

``` r
nv_digamma(x)
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

[`prim_digamma()`](https://r-xla.github.io/anvl/dev/reference/prim_digamma.md)
for the underlying primitive.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(0.5, 1, 2, 5))
digamma(x)
#> AnvlArray
#>  -1.9635
#>  -0.5772
#>   0.4228
#>   1.5061
#> [ CPUf32{4} ] 

# an R value materializes at its default data type
nv_digamma(2)
#> AnvlArray
#>  0.4228
#> [ CPUf32{} ] 
```
