# Complementary Error Function

Element-wise complementary error function `erfc(x) = 1 - erf(x)`.

## Usage

``` r
nv_erfc(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_erfc()`](https://r-xla.github.io/anvl/dev/reference/prim_erfc.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
nv_erfc(x)
#> AnvlArray
#>  1.8427
#>  1.0000
#>  0.1573
#> [ CPUf32{3} ] 
```
