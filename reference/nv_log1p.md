# Log Plus One

Element-wise `log(1 + x)`, more accurate for small `x`.

## Usage

``` r
nv_log1p(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_log1p()`](https://r-xla.github.io/anvl/reference/prim_log1p.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(0, 0.001, 1))
nv_log1p(x)
#> AnvlArray
#>  0.0000
#>  0.0010
#>  0.6931
#> [ CPUf32{3} ] 
```
