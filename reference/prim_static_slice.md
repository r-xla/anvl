# Primitive Static Slice

Extracts a slice from an array using static (compile-time) indices. All
indices, limits, and strides are fixed R integers.

Use
[`prim_dynamic_slice()`](https://r-xla.github.io/anvl/reference/prim_dynamic_slice.md)
instead when the start position must be computed at runtime (e.g.
depends on array values).

## Usage

``` r
prim_static_slice(operand, start_indices, limit_indices, strides)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish value of any data type.

- start_indices:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Start indices (inclusive), one per dimension. Must satisfy
  `1 <= start_indices <= limit_indices` per dimension.

- limit_indices:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  End indices (inclusive), one per dimension. Must satisfy
  `limit_indices <= nv_shape(operand)` per dimension.

- strides:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Step sizes, one per dimension. Must be `>= 1`. A stride of `1` selects
  every element; a stride of `2` selects every other element, etc.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same data type as the input and shape
`ceiling((limit_indices - start_indices + 1) / strides)`. It is
ambiguous if the input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_slice()`](https://r-xla.github.io/stablehlo/reference/hlo_slice.html).

## See also

[`prim_dynamic_slice()`](https://r-xla.github.io/anvl/reference/prim_dynamic_slice.md),
[`prim_scatter()`](https://r-xla.github.io/anvl/reference/prim_scatter.md),
[`prim_gather()`](https://r-xla.github.io/anvl/reference/prim_gather.md),
[`nv_subset()`](https://r-xla.github.io/anvl/reference/nv_subset.md),
`[`

## Examples

``` r
# 1-D: extract elements 2 through 4 (limit is exclusive)
x <- nv_array(1:10)
prim_static_slice(x, start_indices = 2L, limit_indices = 5L, strides = 1L)
#> AnvlArray
#>  2
#>  3
#>  4
#>  5
#> [ CPUi32{4} ] 

# 1-D: every other element using strides
x <- nv_array(1:10)
prim_static_slice(x, start_indices = 1L, limit_indices = 10L, strides = 2L)
#> AnvlArray
#>  1
#>  3
#>  5
#>  7
#>  9
#> [ CPUi32{5} ] 

# 2-D: extract a submatrix (rows 1-2, columns 2-3)
x <- nv_matrix(1:12, nrow = 3, ncol = 4)
prim_static_slice(x,
  start_indices = c(1L, 2L),
  limit_indices = c(3L, 4L),
  strides       = c(1L, 1L)
)
#> AnvlArray
#>   4  7 10
#>   5  8 11
#>   6  9 12
#> [ CPUi32{3,3} ] 
```
