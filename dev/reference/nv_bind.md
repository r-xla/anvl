# Combine arrays by rows or columns

Combine arrays along the row (`nv_rbind`) or column (`nv_cbind`) axis.
Arguments are first promoted to a common data type (see
[`nv_promote_to_common()`](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md)).

Each input is then handled according to its rank:

- 0-D: broadcast to match the non-stacked axes of the other inputs.

- 1-D: treated as a single row/column.

- Other: used as-is.

## Usage

``` r
nv_rbind(...)

nv_cbind(...)

# S3 method for class 'AnvlArray'
rbind(..., deparse.level = 1)

# S3 method for class 'AnvlArray'
cbind(..., deparse.level = 1)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrays to combine. Inputs are promoted to a common data type.

- deparse.level:

  Ignored. Kept for compatibility with
  [`base::rbind()`](https://rdrr.io/r/base/cbind.html) and
  [`base::cbind()`](https://rdrr.io/r/base/cbind.html).

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  

## Differences from base R

[`base::rbind()`](https://rdrr.io/r/base/cbind.html) and
[`base::cbind()`](https://rdrr.io/r/base/cbind.html) applied to an
[`array()`](https://rdrr.io/r/base/array.html) of rank \> 2 flatten the
trailing axes into the column axis (so a `c(2, 3, 4)` array becomes a
`2 x 12` matrix). `nv_rbind` and `nv_cbind` instead preserve all
non-stacked axes: combining two `c(2, 3, 4)` arrays with `nv_rbind`
produces a `c(4, 3, 4)` array, and with `nv_cbind` a `c(2, 6, 4)` array.

## See also

[`nv_concatenate()`](https://r-xla.github.io/anvl/dev/reference/nv_concatenate.md)

## Examples

``` r
# Vectors as rows / columns
nv_rbind(nv_array(1:3), nv_array(4:6))
#> AnvlArray
#>  1 2 3
#>  4 5 6
#> [ CPUi32{2,3} ] 
nv_cbind(nv_array(1:3), nv_array(4:6))
#> AnvlArray
#>  1 4
#>  2 5
#>  3 6
#> [ CPUi32{3,2} ] 

# Scalar broadcasting
nv_rbind(nv_matrix(1:6, nrow = 2), nv_scalar(0))
#> AnvlArray
#>  1 3 5
#>  2 4 6
#>  0 0 0
#> [ CPUf32{3,3} ] 

# Rank-3 arrays preserve trailing axes
a <- nv_array(1:24, shape = c(2, 3, 4))
shape(nv_rbind(a, a)) # c(4, 3, 4)
#> [1] 4 3 4
```
