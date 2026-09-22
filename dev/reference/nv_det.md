# Determinant

Computes the determinant of a square matrix via
[`nv_determinant()`](https://r-xla.github.io/anvl/dev/reference/nv_determinant.md).

## Usage

``` r
nv_det(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input, a square matrix with exactly 2 axes. Can be any numeric
  data type: a float keeps its own, and an integer one is converted to
  the default float data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
  An R value materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and is converted in the same way.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
A scalar with the input's data type – or the default float data type
(see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
where the input was an integer one.

## See also

[`nv_determinant()`](https://r-xla.github.io/anvl/dev/reference/nv_determinant.md),
[`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md),
[`prim_lu()`](https://r-xla.github.io/anvl/dev/reference/prim_lu.md)

## Examples

``` r
# a scalar with the matrix's data type
a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
nv_det(a)
#> AnvlArray
#>  -6
#> [ CPUf64{} ] 
```
