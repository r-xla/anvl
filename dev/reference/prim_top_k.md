# Primitive Top-K

Returns the `k` largest values along the last axis, sorted in descending
order, together with their indices into that axis.

For other axes, transpose so the target axis is last, call
`prim_top_k()`, then transpose back.
[`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md)
does this.

## Usage

``` r
prim_top_k(x, k, indices = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Array of integer, unsigned integer, or floating-point dtype with rank
  \>= 1.

- k:

  (`integer(1)`)  
  Number of top elements. Must satisfy `1 <= k <= shape(x)[naxes(x)]`.

- indices:

  (`logical(1)`)  
  Whether to also return the indices of the top elements. Without them
  the order among tied values is unspecified, which lets the lowering
  pick the cheapest selection for the platform.

## Value

`list` of one or two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
values:  
The top-`k` values (same dtype as `x`) and, if `indices` is `TRUE`,
their indices along the last axis, of the default integer data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
Both have the same shape as `x` with the last axis replaced by `k`. With
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
x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
prim_top_k(x, k = 3L)
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
