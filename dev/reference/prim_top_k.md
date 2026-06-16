# Primitive Top-K

Returns the `k` largest values along the last dimension, sorted in
descending order, together with their indices into that dimension.

For other dimensions, transpose so the target dimension is last, call
`prim_top_k()`, then transpose back.
[`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md)
does this.

## Usage

``` r
prim_top_k(operand, k)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Tensor of integer, unsigned integer, or floating-point dtype with rank
  \>= 1.

- k:

  (`integer(1)`)  
  Number of top elements. Must satisfy
  `1 <= k <= shape(operand)[ndims(operand)]`.

## Value

`list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
values:  
The top-`k` values (same dtype as `operand`) and their indices along the
last dimension (dtype `i32`, matching JAX). Both have the same shape as
`operand` with the last dimension replaced by `k`. Ties are broken by
lower index first.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_top_k()`](https://r-xla.github.io/stablehlo/reference/hlo_top_k.html).

## See also

[`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md),
[`prim_sort()`](https://r-xla.github.io/anvl/dev/reference/prim_sort.md)

## Examples

``` r
x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
prim_top_k(x, k = 3L)
#> [[1]]
#> AnvlArray
#>  9
#>  6
#>  5
#> [ CPUf32{3} ] 
#> 
#> [[2]]
#> AnvlArray
#>  6
#>  8
#>  5
#> [ CPUi32{3} ] 
#> 
```
