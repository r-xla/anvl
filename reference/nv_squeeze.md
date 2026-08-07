# Squeeze

Removes axes of size 1 from an array.

## Usage

``` r
nv_squeeze(x, axes = NULL)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Axes to squeeze. Negative values count from the end, i.e. `-1` refers
  to the last axis. If `NULL` (default), all axes of size 1 are removed.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same data type as `x` with the specified axes removed.

## See also

[`nv_unsqueeze()`](https://r-xla.github.io/anvl/reference/nv_unsqueeze.md),
[`nv_reshape()`](https://r-xla.github.io/anvl/reference/nv_reshape.md)

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
