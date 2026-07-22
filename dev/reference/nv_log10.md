# Base-10 Logarithm

Element-wise base-10 logarithm. You can also use
[`log10()`](https://rdrr.io/r/base/Log.html).

## Usage

``` r
nv_log10(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nv_log()`](https://r-xla.github.io/anvl/dev/reference/nv_log.md),
[`nv_log2()`](https://r-xla.github.io/anvl/dev/reference/nv_log2.md)

## Examples

``` r
x <- nv_array(c(1, 10, 100, 1000))
nv_log10(x)
#> AnvlArray
#>  0.0000
#>  1.0000
#>  2.0000
#>  3.0000
#> [ CPUf32{4} ] 
```
