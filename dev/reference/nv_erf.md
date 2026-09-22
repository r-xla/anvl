# Error Function

Element-wise error function
`erf(x) = (2 / sqrt(pi)) * integral_0^x exp(-t^2) dt`.

## Usage

``` r
nv_erf(x)
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

[`prim_erf()`](https://r-xla.github.io/anvl/dev/reference/prim_erf.md)
for the underlying primitive.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-1, 0, 1))
nv_erf(x)
#> AnvlArray
#>  -0.8427
#>   0.0000
#>   0.8427
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
nv_erf(1)
#> AnvlArray
#>  0.8427
#> [ CPUf32{} ] 
```
