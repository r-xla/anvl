# Matrix Trace

Computes the trace (sum of diagonal elements) of a 2-D array.

## Usage

``` r
nv_trace(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
A scalar with the same data type as `x`.

## See also

[`nv_extract_diag()`](https://r-xla.github.io/anvl/reference/nv_extract_diag.md),
[`nv_diag()`](https://r-xla.github.io/anvl/reference/nv_diag.md)

## Examples

``` r
x <- nv_array(c(1, 0, 0, 0, 2, 0, 0, 0, 3), shape = c(3, 3))
nv_trace(x)
#> AnvlArray
#>  6
#> [ CPUf32{} ] 
```
