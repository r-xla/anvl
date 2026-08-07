# Negation

Negates an array element-wise. You can also use the unary `-` operator.

## Usage

``` r
nv_negate(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_negate()`](https://r-xla.github.io/anvl/reference/prim_negate.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, -2, 3))
-x
#> AnvlArray
#>  -1
#>   2
#>  -3
#> [ CPUf32{3} ] 
```
