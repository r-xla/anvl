# Primitive Dot General

General dot product of two arrays, supporting contraction over arbitrary
dimensions and batching.

## Usage

``` r
prim_dot_general(
  lhs,
  rhs,
  contracting_dims,
  batching_dims,
  precision = "highest"
)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Left and right operand. Operands are [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md).
  Scalars are
  [broadcast](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_scalars.md)
  to the shape of the other operand.

- contracting_dims:

  (`list(integer(), integer())`)  
  A list of two integer vectors specifying which dimensions of `lhs` and
  `rhs` to contract over. The contracted dimensions must have matching
  sizes.

- batching_dims:

  (`list(integer(), integer())`)  
  A list of two integer vectors specifying which dimensions of `lhs` and
  `rhs` are batch dimensions. These must have matching sizes.

- precision:

  (`character(1)`)  
  Controls the trade-off between speed and numerical accuracy of the
  operation. One of `"highest"` (default), `"high"` or `"default"`. Only
  the StableHLO backend honors this; it is ignored by the quickr
  backend.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
The output shape is the batch dimensions followed by the remaining
(non-contracted, non-batched) dimensions of `lhs`, then `rhs`.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_dot_general()`](https://r-xla.github.io/stablehlo/reference/hlo_dot_general.html).

## See also

[`nv_matmul()`](https://r-xla.github.io/anvl/dev/reference/nv_matmul.md),
`%*%`

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
y <- nv_matrix(1:6, nrow = 3)
prim_dot_general(x, y,
  contracting_dims = list(2L, 1L),
  batching_dims = list(integer(0), integer(0))
)
#> AnvlArray
#>  22 49
#>  28 64
#> [ CPUi32{2,2} ] 
```
