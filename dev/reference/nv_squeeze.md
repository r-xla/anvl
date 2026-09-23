# Squeeze

Removes axes of size 1 from an array. `nv_drop()` is another spelling of
the same function; with the default `axes = NULL` it drops every size-1
axis, like [`base::drop()`](https://rdrr.io/r/base/drop.html).

## Usage

``` r
nv_squeeze(x, axes = NULL)

nv_drop(x, axes = NULL)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Axes to squeeze. Negative values count from the end, i.e. `-1` refers
  to the last axis. If `NULL` (default), all axes of size 1 are removed.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has `x`'s data type, with the specified axes removed from its shape.

## See also

[`nv_unsqueeze()`](https://r-xla.github.io/anvl/dev/reference/nv_unsqueeze.md),
[`nv_reshape()`](https://r-xla.github.io/anvl/dev/reference/nv_reshape.md)

## Examples

``` r
# the two size-1 axes are dropped
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
nv_drop(x)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPUi32{6} ] 
```
