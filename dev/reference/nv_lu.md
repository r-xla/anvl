# LU Decomposition

Computes the partial-pivoted LU decomposition of a matrix `x`: \$\$P A =
L U,\$\$ where \\P\\ is a permutation matrix, \\L\\ is unit lower
triangular, and \\U\\ is upper triangular.

This function returns `L` and `U` as separate matrices. Use
[`prim_lu()`](https://r-xla.github.io/anvl/dev/reference/prim_lu.md) to
get them in packed `LU` form.

## Usage

``` r
nv_lu(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input, with exactly 2 axes. Can be any numeric data type: a float
  keeps its own, and an integer one is converted to the default float
  data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
  An R value materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and is converted in the same way.

## Value

(named `list` of
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
`L` and `U` have the input's data type – or the default float data type
(see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
where the input was an integer one; `pivots` and `permutation` are
indices at the default integer data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).

- `L` – unit lower-triangular factor of shape `(m, k)`, where
  `(m, n) = shape(x)` and `k = min(m, n)`.

- `U` – upper-triangular factor of shape `(k, n)`.

- `pivots` – length `k`, at the default integer data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
  LAPACK-style sequential row swaps as returned by `getrf`.

- `permutation` – length `m`, at that same data type. A permutation
  vector representing \\P\\.

## See also

[`prim_lu()`](https://r-xla.github.io/anvl/dev/reference/prim_lu.md)

## Examples

``` r
# `L` and `U` keep the input's data type; the pivots are the default integer
x <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
nv_lu(x)
#> $L
#> AnvlArray
#>  1.0000 0.0000
#>  0.7500 1.0000
#> [ CPUf64{2,2} ] 
#> 
#> $U
#> AnvlArray
#>   4.0000  6.0000
#>   0.0000 -1.5000
#> [ CPUf64{2,2} ] 
#> 
#> $pivots
#> AnvlArray
#>  1
#>  2
#> [ CPUi32{2} ] 
#> 
#> $permutation
#> AnvlArray
#>  1
#>  2
#> [ CPUi32{2} ] 
#> 
```
