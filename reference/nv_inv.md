# Matrix Inverse

Computes `operand^-1`, the inverse of a square non-singular matrix, by
solving `operand %*% x = I`.

For most use cases prefer
[`nv_solve()`](https://r-xla.github.io/anvl/reference/nv_solve.md)
directly: forming the explicit inverse is both slower and less
numerically stable than solving against a right-hand side.

## Usage

``` r
nv_inv(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Square non-singular matrix.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
The inverse, same shape and dtype as `operand`.

## See also

[`nv_solve()`](https://r-xla.github.io/anvl/reference/nv_solve.md)

## Examples

``` r
a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
nv_inv(a)
#> AnvlArray
#>  -0.5000  1.0000
#>   0.5000 -0.6667
#> [ CPUf64{2,2} ] 
```
