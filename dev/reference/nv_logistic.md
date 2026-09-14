# Logistic (Sigmoid)

Element-wise logistic sigmoid: `1 / (1 + exp(-x))`.

## Usage

``` r
nv_logistic(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array. An integer array is converted to the default floating
  point type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape as the input, and its data type – or the default
float data type if the input was an integer array.

## See also

[`prim_logistic()`](https://r-xla.github.io/anvl/dev/reference/prim_logistic.md)
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
