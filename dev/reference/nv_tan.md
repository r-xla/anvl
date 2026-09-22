# Tangent

Element-wise tangent. You can also use
[`tan()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_tan(x)
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

[`prim_tan()`](https://r-xla.github.io/anvl/dev/reference/prim_tan.md)
for the underlying primitive.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(0, 0.5, 1))
tan(x)
#> AnvlArray
#>  0.0000
#>  0.5463
#>  1.5574
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
nv_tan(0.5)
#> AnvlArray
#>  0.5463
#> [ CPUf32{} ] 
```
