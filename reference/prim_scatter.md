# Primitive Scatter

Produces a result array identical to `x` except that slices at positions
specified by `scatter_indices` are updated with values from the `update`
array. When multiple indices point to the same location, the `update_fn`
function determines how to combine the values (by default the new value
replaces the old one).

This is the inverse of
[`prim_gather()`](https://r-xla.github.io/anvl/reference/prim_gather.md):
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
  update_fn = "replace"
)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  The base array to scatter into. Can be any data type. `x` and `update`
  must have the same data type. An R value among them assumes the data
  type of the others when it is in its [data type
  category](https://r-xla.github.io/anvl/reference/dtype_categories.md),
  and its [default data
  type](https://r-xla.github.io/anvl/reference/default_dtypes.md) when
  none of them has one.

- scatter_indices:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Array of indices, of an integer data type, which it keeps – the
  indices take no part in `x`'s. Contains index vectors that map to
  positions in `x` via `scatter_axes_to_x_axes`. The axis specified by
  `index_vector_axis` holds the index vectors.

- update:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Update values array. Shares `x`'s data type – see `x`.

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
  Maps each component of an index vector in `scatter_indices` to an axis
  of `x`. For example, `scatter_axes_to_x_axes = 1L` means each index
  vector indexes into the first axis of `x`. This is
  [`prim_gather()`](https://r-xla.github.io/anvl/reference/prim_gather.md)'s
  `start_index_map`.

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

- update_fn:

  (`character(1)` \| `function`)  
  How an update combines with the value in `x` it lands on. One of

  - `"replace"` (the default): the update replaces the value,

  - `"add"`, `"sub"`, `"mul"`: the value plus, minus or times the
    update,

  - `"max"`, `"min"`: the larger or the smaller of the two,

  or a binary function `f(old, new)` of two scalars at `x`'s data type,
  which returns one. Only a named `update_fn` can be differentiated, see
  section Gradients.

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
Has the same data type and shape as `x`.

## Out of Bounds Behavior

If a computed result index falls outside the bounds of `x`, the update
for that index is silently ignored.

## Update Order

When multiple indices in `scatter_indices` map to the same element of
`x`, `update_fn` combines each of their updates with it in turn, so
`"add"` adds them all up. The order is implementation-defined and may
vary between plugins ("cpu", "cuda"), which matters for `"replace"`,
where one of them is the one that ends up written.

## Gradients

[`gradient()`](https://r-xla.github.io/anvl/reference/gradient.md)
differentiates a scatter whose `update_fn` is one of the names, but not
a function, even one that computes the same.

- With `"replace"`, an update gets the gradient only where it is the one
  that was written.

- With `"mul"`, the gradient with respect to `update` needs
  `unique_indices = TRUE`.

- With `"max"` and `"min"`, `x` and the updates that tie for the result
  share its gradient equally.

An update that falls outside of `x`, and is therefore ignored, gets a
gradient of zero.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_scatter()`](https://r-xla.github.io/stablehlo/reference/hlo_scatter.html),
specified under [scatter](https://openxla.org/stablehlo/spec#scatter).
The axis and index arguments are intricate; anvl states them in its own
terms and converts on the way down.

## See also

[`prim_gather()`](https://r-xla.github.io/anvl/reference/prim_gather.md),
[`nv_subset()`](https://r-xla.github.io/anvl/reference/nv_subset.md),
[`nv_subset_assign()`](https://r-xla.github.io/anvl/reference/nv_subset_assign.md),
`[`, `[<-`

## Examples

``` r
# scatter values 10 and 30 into positions 1 and 3 of a zero vector
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
