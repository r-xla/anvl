# Primitive Dot General

General dot product of two arrays, supporting contraction over arbitrary
axes and batching.

## Usage

``` r
prim_dot_general(
  lhs,
  rhs,
  contracting_axes,
  batching_axes,
  precision = "highest"
)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Left and right operand. Operands are [promoted to a common data
  type](https://r-xla.github.io/anvl/reference/nv_promote_to_common.md).
  Scalars are
  [broadcast](https://r-xla.github.io/anvl/reference/nv_broadcast_scalars.md)
  to the shape of the other operand.

- contracting_axes:

  (`list(integer(), integer())`)  
  A list of two integer vectors specifying which axes of `lhs` and `rhs`
  to contract over. The contracted axes must have matching sizes.

- batching_axes:

  (`list(integer(), integer())`)  
  A list of two integer vectors specifying which axes of `lhs` and `rhs`
  are batch axes. These must have matching sizes.

- precision:

  (`character(1)`)  
  Controls the trade-off between speed and numerical accuracy of the
  operation. One of `"highest"` (default), `"high"` or `"default"`. Only
  the StableHLO backend honors this; it is ignored by the quickr
  backend.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
The output shape is the batch axes followed by the remaining
(non-contracted, non-batched) axes of `lhs`, then `rhs`.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_dot_general()`](https://r-xla.github.io/stablehlo/reference/hlo_dot_general.html).

## See also

[`nv_matmul()`](https://r-xla.github.io/anvl/reference/nv_matmul.md),
`%*%`

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
y <- nv_matrix(1:6, nrow = 3)
prim_dot_general(x, y,
  contracting_axes = list(2L, 1L),
  batching_axes = list(integer(0), integer(0))
)
#> AnvlArray
#>  22 49
#>  28 64
#> [ CPUi32{2,2} ] 
```
