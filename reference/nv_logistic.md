# Logistic (Sigmoid)

Element-wise logistic sigmoid: `1 / (1 + exp(-x))`.

## Usage

``` r
nv_logistic(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_logistic()`](https://r-xla.github.io/anvl/reference/prim_logistic.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(-2, 0, 2))
nv_logistic(x)
#> AnvlArray
#>  0.1192
#>  0.5000
#>  0.8808
#> [ CPUf32{3} ] 
```
