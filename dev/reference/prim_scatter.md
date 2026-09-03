# Primitive Scatter

Produces a result array identical to `x` except that slices at positions
specified by `scatter_indices` are updated with values from the `update`
array. When multiple indices point to the same location, the
`update_computation` function determines how to combine the values (by
default the new value replaces the old one).

This is the inverse of
[`prim_gather()`](https://r-xla.github.io/anvl/dev/reference/prim_gather.md):
gather reads slices from an array at given indices, while scatter writes
slices into an array at given indices.

## Usage

``` r
prim_scatter(
  x,
  scatter_indices,
  update,
  update_window_axes,
  inserted_window_axes,
  x_batching_axes,
  scatter_indices_batching_axes,
  scatter_axes_to_x_axes,
  index_vector_axis,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  update_computation = NULL
)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type. The base array to scatter into.

- scatter_indices:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
  of integer type)  
  Array of indices. Contains index vectors that map to positions in `x`
  via `scatter_axes_to_x_axes`. The axis specified by
  `index_vector_axis` holds the index vectors.

- update:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Update values array. Must have the same data type as `x`.

- update_window_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Axes of `update` that are window axes, i.e. they correspond to the
  slice being written into `x`.

- inserted_window_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Axes of `x` whose slices have size 1 and are inserted (not present) in
  the `update` window. Together with `update_window_axes` and
  `x_batching_axes`, these must account for all axes of `x`.

- x_batching_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Axes of `x` that are batch axes. Use `integer(0)` when there are no
  batch axes.

- scatter_indices_batching_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Axes of `scatter_indices` that correspond to batch axes. Must have the
  same length as `x_batching_axes`.

- scatter_axes_to_x_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Maps each component of the index vector to an `x` axis. For example,
  `scatter_axes_to_x_axes = c(1L)` means each index vector indexes into
  the first axis of `x`.

- index_vector_axis:

  (`integer(1)`)  
  Axis of `scatter_indices` that contains the index vectors. If set to
  `naxes(scatter_indices) + 1`, each scalar element of `scatter_indices`
  is treated as a length-1 index vector.

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

- update_computation:

  (`function`)  
  Binary function `f(old, new)` that combines the existing value in `x`
  with the value from `update`. The default (`NULL`) uses
  `function(old, new) new`, which replaces the old value.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type and shape as `x`.

## Out Of Bounds Behavior

If a computed result index falls outside the bounds of `x`, the update
for that index is silently ignored.

## Update Order

When multiple indices in `scatter_indices` map to the same element of
`x`, the order in which `update_computation` is applied is
implementation-defined and may vary between plugins ("cpu", "cuda").

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_scatter()`](https://r-xla.github.io/stablehlo/reference/hlo_scatter.html).

## See also

[`prim_gather()`](https://r-xla.github.io/anvl/dev/reference/prim_gather.md),
[`nv_subset()`](https://r-xla.github.io/anvl/dev/reference/nv_subset.md),
[`nv_subset_assign()`](https://r-xla.github.io/anvl/dev/reference/nv_subset_assign.md),
`[`, `[<-`

## Examples

``` r
# Scatter values 10 and 30 into positions 1 and 3 of a zero vector
x <- nv_array(c(0, 0, 0, 0, 0))
indices <- nv_matrix(c(1L, 3L), ncol = 1)
updates <- nv_array(c(10, 30))
prim_scatter(
  x, indices, updates,
  update_window_axes = integer(0),
  inserted_window_axes = 1L,
  x_batching_axes = integer(0),
  scatter_indices_batching_axes = integer(0),
  scatter_axes_to_x_axes = 1L,
  index_vector_axis = 2L
)
#> AnvlArray
#>  10
#>   0
#>  30
#>   0
#>   0
#> [ CPUf32{5} ] 
```
