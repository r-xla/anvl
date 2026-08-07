# Base-2 Logarithm

Element-wise base-2 logarithm. You can also use
[`log2()`](https://rdrr.io/r/base/Log.html).

## Usage

``` r
nv_log2(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nv_log()`](https://r-xla.github.io/anvl/reference/nv_log.md),
[`nv_log10()`](https://r-xla.github.io/anvl/reference/nv_log10.md)

## Examples

``` r
x <- nv_array(c(1, 2, 4, 8))
nv_log2(x)
#> AnvlArray
#>  0
#>  1
#>  2
#>  3
#> [ CPUf32{4} ] 
```
