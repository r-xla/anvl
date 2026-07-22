# Natural Logarithm

Element-wise natural logarithm. You can also use
[`log()`](https://rdrr.io/r/base/Log.html).

## Usage

``` r
nv_log(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_log()`](https://r-xla.github.io/anvl/dev/reference/prim_log.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 2.718, 7.389))
log(x)
#> AnvlArray
#>  0.0000
#>  0.9999
#>  2.0000
#> [ CPUf32{3} ] 
```
