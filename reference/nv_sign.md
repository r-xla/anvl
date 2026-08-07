# Sign

Element-wise sign function. You can also use
[`sign()`](https://rdrr.io/r/base/sign.html).

## Usage

``` r
nv_sign(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_sign()`](https://r-xla.github.io/anvl/reference/prim_sign.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(-3, 0, 5))
sign(x)
#> AnvlArray
#>  -1
#>   0
#>   1
#> [ CPUf32{3} ] 
```
