# Cosine

Element-wise cosine. You can also use
[`cos()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_cos(x)
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

[`prim_cos()`](https://r-xla.github.io/anvl/dev/reference/prim_cos.md)
for the underlying primitive.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(0, pi / 2, pi))
cos(x)
#> AnvlArray
#>   1.0000e+00
#>  -4.3711e-08
#>  -1.0000e+00
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
nv_cos(0)
#> AnvlArray
#>  1
#> [ CPUf32{} ] 
```
