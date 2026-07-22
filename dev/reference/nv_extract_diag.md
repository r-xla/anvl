# Extract Diagonal

Extracts the diagonal elements from a 2-D array.

## Usage

``` r
nv_extract_diag(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
A 1-D array of length `min(nrow, ncol)` containing the diagonal
elements.

## See also

[`nv_diag()`](https://r-xla.github.io/anvl/dev/reference/nv_diag.md) for
creating a diagonal matrix,
[`nv_trace()`](https://r-xla.github.io/anvl/dev/reference/nv_trace.md)

## Examples

``` r
x <- nv_array(1:9, shape = c(3, 3))
nv_extract_diag(x)
#> AnvlArray
#>  1
#>  5
#>  9
#> [ CPUi32{3} ] 
```
