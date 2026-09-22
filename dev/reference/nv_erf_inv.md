# Inverse Error Function

Element-wise inverse error function (the inverse of `erf` on `(-1, 1)`).

## Usage

``` r
nv_erf_inv(x)
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

[`prim_erf_inv()`](https://r-xla.github.io/anvl/dev/reference/prim_erf_inv.md)
for the underlying primitive.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-0.5, 0, 0.5))
nv_erf_inv(x)
#> AnvlArray
#>  -0.4769
#>   0.0000
#>   0.4769
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
nv_erf_inv(0.5)
#> AnvlArray
#>  0.4769
#> [ CPUf32{} ] 
```
