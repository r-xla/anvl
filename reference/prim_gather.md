# Primitive Gather

Gathers slices from the `x` array at positions specified by
`start_indices`. Each index vector in `start_indices` identifies a
starting position in `x`, and a slice of size `slice_sizes` is extracted
from that position. The gathered slices are assembled into the output
array.

This is the inverse of
[`prim_scatter()`](https://r-xla.github.io/anvl/reference/prim_scatter.md):
gather reads slices from a array at given indices, while scatter writes
slices into an array at given indices.

## Usage

``` r
prim_gather(
  x,
  start_indices,
  slice_sizes,
  offset_axes,
  collapsed_slice_axes,
  x_batching_axes,
  start_indices_batching_axes,
  start_index_map,
  index_vector_axis,
  indices_are_sorted = FALSE,
  unique_indices = FALSE
)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish value of any data type.

- start_indices:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md) of
  integer type)  
  Array of starting indices. Contains index vectors that map to
  positions in `x` via `start_index_map`. The axis specified by
  `index_vector_axis` holds the index vectors.

- slice_sizes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Size of the slice to gather from `x` in each axis. Must have length
  equal to `naxes(x)`.

- offset_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Axes in the output that correspond to the non-collapsed slice axes of
  `x`.

- collapsed_slice_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Axes of `x` that are collapsed (removed) from the slice. The
  corresponding entries in `slice_sizes` must be `1`. Together with
  `offset_axes` and `x_batching_axes`, these must account for all axes
  of `x`.

- x_batching_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Axes of `x` that are batch axes. Use `integer(0)` when there are no
  batch axes.

- start_indices_batching_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Axes of `start_indices` that correspond to batch axes. Must have the
  same length as `x_batching_axes`.

- start_index_map:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Maps each component of the index vector to an `x` axis. For example,
  `start_index_map = c(1L)` means each index vector indexes into the
  first axis of `x`.

- index_vector_axis:

  (`integer(1)`)  
  Axis of `start_indices` that contains the index vectors. If set to
  `naxes(start_indices) + 1`, each scalar element of `start_indices` is
  treated as a length-1 index vector.

- indices_are_sorted:

  (`logical(1)`)  
  Whether indices are guaranteed to be sorted. Setting to `TRUE` may
  improve performance but produces undefined behavior if the indices are
  not actually sorted. Default `FALSE`.

- unique_indices:

  (`logical(1)`)  
  Whether indices are guaranteed to be unique (no duplicates). Setting
  to `TRUE` may improve performance but produces undefined behavior if
  the indices are not actually unique. Default `FALSE`.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same data type as `x`. The output shape is composed of the
offset axes (from the slice) and the remaining axes from
`start_indices`. See the underluing stableHLO function for more details.

## Out Of Bounds Behavior

Start indices are clamped before the slice is extracted:
`clamp(1, start_index, nv_shape(x) - slice_sizes + 1)`. This means that
out-of-bounds indices will not cause an error, but the effective start
position may differ from the requested one.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_gather()`](https://r-xla.github.io/stablehlo/reference/hlo_gather.html).

## See also

[`prim_scatter()`](https://r-xla.github.io/anvl/reference/prim_scatter.md),
[`nv_subset()`](https://r-xla.github.io/anvl/reference/nv_subset.md),
[`nv_subset_assign()`](https://r-xla.github.io/anvl/reference/nv_subset_assign.md),
`[`, `[<-`

## Examples

``` r
# Gather rows 1 and 3 from a 3x3 matrix
x <- nv_matrix(1:9, nrow = 3)
indices <- nv_matrix(c(1L, 3L), ncol = 1)
prim_gather(
  x, indices,
  slice_sizes = c(1L, 3L),
  offset_axes = 2L,
  collapsed_slice_axes = 1L,
  x_batching_axes = integer(0),
  start_indices_batching_axes = integer(0),
  start_index_map = 1L,
  index_vector_axis = 2L
)
#> AnvlArray
#>  1 4 7
#>  3 6 9
#> [ CPUi32{2,3} ] 
```
