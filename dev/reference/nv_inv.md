# Matrix Inverse

Computes `x^-1`, the inverse of a square non-singular matrix `x`, by
solving `x %*% y = I` for `y`.

For most use cases prefer
[`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md)
directly: forming the explicit inverse is both slower and less
numerically stable than solving against a right-hand side.

## Usage

``` r
nv_inv(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input, a square non-singular matrix with exactly 2 axes. Can be
  any numeric data type: a float keeps its own, and an integer one is
  converted to the default float data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
  An R value materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and is converted in the same way.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
The inverse, with the input's shape, and its data type – or the default
float data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
where the input was an integer one.

## See also

[`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md)

## Examples

``` r
# the inverse has the matrix's shape and data type
a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
nv_inv(a)
#> AnvlArray
#>  -0.5000  1.0000
#>   0.5000 -0.6667
#> [ CPUf64{2,2} ] 
```
