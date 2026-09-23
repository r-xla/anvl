# Primitive Top-K

Returns the `k` largest values along the last axis, sorted in decreasing
order, and with `indices = TRUE` their indices into that axis as well.

For other axes, transpose so the target axis is last, call
`prim_top_k()`, then transpose back.
[`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md)
does this.

## Usage

``` r
prim_top_k(x, k, indices)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input, with at least 1 axis. Can be any numeric data type. An R
  value materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- k:

  (`integer(1)`)  
  Number of top elements. Must satisfy `1 <= k <= shape(x)[naxes(x)]`.

- indices:

  (`logical(1)`)  
  Whether to also return the indices of the top elements. Without them
  the order among tied values is unspecified, which lets the lowering
  pick the cheapest selection for the platform.

## Value

(named `list` of one or two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Element `values`, the top-`k` values at the input's data type, and, when
`indices` is `TRUE`, `indices`, their indices along the last axis at the
default integer data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
Both have the input's shape with the last axis replaced by `k`. With
indices, ties are broken by lower index first.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_top_k()`](https://r-xla.github.io/stablehlo/reference/hlo_top_k.html).
Without `indices` on CUDA it lowers to an unstable descending
[`hlo_sort()`](https://r-xla.github.io/stablehlo/reference/hlo_sort.html)
of the values followed by an
[`hlo_slice()`](https://r-xla.github.io/stablehlo/reference/hlo_slice.html),
which is what the CHLO op expands to there minus the index operand and
the stability the ties no longer need; XLA's CPU backend has a dedicated
top-k kernel, so it keeps
[`hlo_top_k()`](https://r-xla.github.io/stablehlo/reference/hlo_top_k.html).

## See also

[`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md),
[`prim_sort()`](https://r-xla.github.io/anvl/dev/reference/prim_sort.md)

## Examples

``` r
# `values` keeps the input's data type, `indices` is the default integer
x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
prim_top_k(x, k = 3L, indices = FALSE)
#> $values
#> AnvlArray
#>  9
#>  6
#>  5
#> [ CPUf32{3} ] 
#> 
prim_top_k(x, k = 3L, indices = TRUE)
#> $values
#> AnvlArray
#>  9
#>  6
#>  5
#> [ CPUf32{3} ] 
#> 
#> $indices
#> AnvlArray
#>  6
#>  8
#>  5
#> [ CPUi32{3} ] 
#> 
```
