# Exponential Minus One

Element-wise `exp(x) - 1`, more accurate for small `x`. You can also use
[`expm1()`](https://rdrr.io/r/base/Log.html).

## Usage

``` r
nv_expm1(x)
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

[`prim_expm1()`](https://r-xla.github.io/anvl/dev/reference/prim_expm1.md)
for the underlying primitive.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(0, 0.001, 1))
nv_expm1(x)
#> AnvlArray
#>  0.0000
#>  0.0010
#>  1.7183
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
nv_expm1(0.001)
#> AnvlArray
#>  0.0010
#> [ CPUf32{} ] 
```
