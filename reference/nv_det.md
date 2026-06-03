# Determinant

Computes the determinant of a square matrix via
[`nv_determinant()`](https://r-xla.github.io/anvl/reference/nv_determinant.md).

## Usage

``` r
nv_det(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Square matrix of floating-point data type.

## Value

Scalar [`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)
with the same dtype as `operand`.

## See also

[`nv_determinant()`](https://r-xla.github.io/anvl/reference/nv_determinant.md),
[`nv_solve()`](https://r-xla.github.io/anvl/reference/nv_solve.md),
[`prim_lu()`](https://r-xla.github.io/anvl/reference/prim_lu.md)

## Examples

``` r
a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
nv_det(a)
#> AnvlArray
#>  -6
#> [ CPUf64{} ] 
```
