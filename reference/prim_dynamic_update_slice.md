# Primitive Dynamic Update Slice

Returns a copy of `x` with a slice replaced by `update` at a
runtime-determined position. This is the write counterpart of
[`prim_dynamic_slice()`](https://r-xla.github.io/anvl/reference/prim_dynamic_slice.md):
dynamic slice reads a block from an array, while dynamic update slice
writes a block into an array.

## Usage

``` r
prim_dynamic_update_slice(x, update, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish value of any data type.

- update:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  The values to write at the specified position. Must have the same data
  type and number of axes as `x`, with `nv_shape(update) <= nv_shape(x)`
  per axis.

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md) of
  integer type)  
  Scalar start indices, one per axis of `x`. Each must be a scalar
  array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same data type and shape as `x`. It is ambiguous if the input is
ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_dynamic_update_slice()`](https://r-xla.github.io/stablehlo/reference/hlo_dynamic_update_slice.html).

## Out Of Bounds Behavior

Start indices are clamped before the slice is extracted:
`adjusted_start_indices = clamp(1, start_indices, nv_shape(x) - slice_sizes + 1)`.
This means that out-of-bounds indices will not cause an error, but the
effective start position may differ from the requested one.

## See also

[`prim_dynamic_slice()`](https://r-xla.github.io/anvl/reference/prim_dynamic_slice.md),
[`prim_scatter()`](https://r-xla.github.io/anvl/reference/prim_scatter.md),
[`prim_gather()`](https://r-xla.github.io/anvl/reference/prim_gather.md),
[`nv_subset_assign()`](https://r-xla.github.io/anvl/reference/nv_subset_assign.md),
`[<-`

## Examples

``` r
# 1-D: overwrite two elements starting at position 2
x <- nv_array(1:5)
update <- nv_array(c(10L, 20L))
start <- nv_scalar(2L)
prim_dynamic_update_slice(x, update, start)
#> AnvlArray
#>   1
#>  10
#>  20
#>   4
#>   5
#> [ CPUi32{5} ] 

# 2-D: write a 2x2 block into a 3x4 matrix
x <- nv_fill(0L, shape = c(3, 4))
update <- nv_matrix(c(1L, 2L, 3L, 4L), nrow = 2, ncol = 2)
row_start <- nv_scalar(2L)
col_start <- nv_scalar(3L)
prim_dynamic_update_slice(x, update, row_start, col_start)
#> AnvlArray
#>  0 0 0 0
#>  0 0 1 3
#>  0 0 2 4
#> [ CPUi32{3,4} ] 
```
