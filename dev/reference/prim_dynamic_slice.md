# Primitive Dynamic Slice

Extracts a slice from an array whose start position is determined at
runtime via array-valued indices. The slice shape (`slice_sizes`) is a
fixed R integer vector.

Use
[`prim_static_slice()`](https://r-xla.github.io/anvl/dev/reference/prim_static_slice.md)
instead when all indices are known at compile time and you need stride
support.

## Usage

``` r
prim_dynamic_slice(operand, ..., slice_sizes)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
  of integer type)  
  Scalar start indices, one per dimension. Each must be a scalar array.
  Pass one scalar per dimension of `operand`.

- slice_sizes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Size of the slice in each dimension. Must have length equal to
  `ndims(operand)` and satisfy `1 <= slice_sizes <= nv_shape(operand)`
  per dimension.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as the input and shape `slice_sizes`. It is
ambiguous if the input is ambiguous.

## Out Of Bounds Behavior

Start indices are clamped before the slice is extracted:
`adjusted_start_indices = clamp(1, start_indices, nv_shape(operand) - slice_sizes + 1)`.
This means that out-of-bounds indices will not cause an error, but the
effective start position may differ from the requested one.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_dynamic_slice()`](https://r-xla.github.io/stablehlo/reference/hlo_dynamic_slice.html).

## See also

[`prim_static_slice()`](https://r-xla.github.io/anvl/dev/reference/prim_static_slice.md),
[`prim_dynamic_update_slice()`](https://r-xla.github.io/anvl/dev/reference/prim_dynamic_update_slice.md),
[`prim_scatter()`](https://r-xla.github.io/anvl/dev/reference/prim_scatter.md),
[`prim_gather()`](https://r-xla.github.io/anvl/dev/reference/prim_gather.md),
[`nv_subset()`](https://r-xla.github.io/anvl/dev/reference/nv_subset.md),
`[`

## Examples

``` r
# 1-D: extract 3 elements starting at position 3
x <- nv_array(1:10)
start <- nv_scalar(3L)
prim_dynamic_slice(x, start, slice_sizes = 3L)
#> AnvlArray
#>  3
#>  4
#>  5
#> [ CPUi32{3} ] 

# 2-D: extract a 2x2 block from a matrix
x <- nv_matrix(1:12, nrow = 3, ncol = 4)
row_start <- nv_scalar(2L)
col_start <- nv_scalar(1L)
prim_dynamic_slice(x, row_start, col_start, slice_sizes = c(2L, 2L))
#> AnvlArray
#>  2 5
#>  3 6
#> [ CPUi32{2,2} ] 
```
