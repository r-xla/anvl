# Static Slice

Extracts a slice from an array using static (compile-time) indices. For
dynamic indexing, use
[`nv_subset()`](https://r-xla.github.io/anvl/dev/reference/nv_subset.md)
instead.

## Usage

``` r
nv_static_slice(x, start_indices, limit_indices, strides)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- start_indices:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Start indices (inclusive), one per axis.

- limit_indices:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  End indices (inclusive), one per axis.

- strides:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Step sizes, one per axis. A stride of 1 selects every element.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has `x`'s data type and shape
`ceiling((limit_indices - start_indices + 1) / strides)` per axis.

## See also

[`nv_subset()`](https://r-xla.github.io/anvl/dev/reference/nv_subset.md),
[`prim_static_slice()`](https://r-xla.github.io/anvl/dev/reference/prim_static_slice.md)
for the underlying primitive.

## Examples

``` r
# elements 2 through 5, the limit being inclusive
x <- nv_array(1:10)
nv_static_slice(x, start_indices = 2L, limit_indices = 5L, strides = 1L)
#> AnvlArray
#>  2
#>  3
#>  4
#>  5
#> [ CPUi32{4} ] 
```
