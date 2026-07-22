# Squeeze

Removes dimensions of size 1 from an array.

## Usage

``` r
nv_squeeze(x, dims = NULL)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Dimensions to squeeze. Negative values count from the end, i.e. `-1`
  refers to the last dimension. If `NULL` (default), all dimensions of
  size 1 are removed.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as `x` with the specified dimensions removed.

## See also

[`nv_unsqueeze()`](https://r-xla.github.io/anvl/dev/reference/nv_unsqueeze.md),
[`nv_reshape()`](https://r-xla.github.io/anvl/dev/reference/nv_reshape.md)

## Examples

``` r
x <- nv_array(1:6, shape = c(1, 6, 1))
nv_squeeze(x)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPUi32{6} ] 
```
