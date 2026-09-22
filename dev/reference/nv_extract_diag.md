# Extract Diagonal

Extracts the diagonal elements from a 2-D array.

## Usage

``` r
nv_extract_diag(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input, with exactly 2 axes. Can be any data type. An R value
  materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's data type, and one axis of length `min(nrow, ncol)`
holding the diagonal elements.

## See also

[`nv_diag()`](https://r-xla.github.io/anvl/dev/reference/nv_diag.md) for
creating a diagonal matrix,
[`nv_trace()`](https://r-xla.github.io/anvl/dev/reference/nv_trace.md)

## Examples

``` r
# the diagonal of a 3x3 matrix, keeping its data type
x <- nv_array(1:9, shape = c(3, 3))
nv_extract_diag(x)
#> AnvlArray
#>  1
#>  5
#>  9
#> [ CPUi32{3} ] 
```
