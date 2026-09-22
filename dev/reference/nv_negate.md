# Negation

Negates an array element-wise. You can also use the unary `-` operator.

## Usage

``` r
nv_negate(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any numeric data type. An R value materializes at
  its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape and data type.

## See also

[`prim_negate()`](https://r-xla.github.io/anvl/dev/reference/prim_negate.md)
for the underlying primitive.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1, -2, 3))
-x
#> AnvlArray
#>  -1
#>   2
#>  -3
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
nv_negate(1)
#> AnvlArray
#>  -1
#> [ CPUf32{} ] 
```
