# Matrix Trace

Computes the trace (sum of diagonal elements) of a 2-D array.

## Usage

``` r
nv_trace(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input, with exactly 2 axes. Can be any data type. An R value
  materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
A scalar with `x`'s data type, except a boolean input, which is counted
at the default integer data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).

## See also

[`nv_extract_diag()`](https://r-xla.github.io/anvl/dev/reference/nv_extract_diag.md),
[`nv_diag()`](https://r-xla.github.io/anvl/dev/reference/nv_diag.md)

## Examples

``` r
# the diagonal is summed to a scalar
x <- nv_array(c(1, 0, 0, 0, 2, 0, 0, 0, 3), shape = c(3, 3))
nv_trace(x)
#> AnvlArray
#>  6
#> [ CPUf32{} ] 
```
