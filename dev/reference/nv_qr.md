# QR Decomposition

Computes the reduced QR decomposition of a matrix `x`: \$\$A = Q R,\$\$
where \\Q\\ has orthonormal columns (\\Q^\top Q = I\\) and \\R\\ is
upper triangular. For an \\m \times n\\ input with \\k = \min(m, n)\\,
\\Q\\ has shape \\m \times k\\ and \\R\\ has shape \\k \times n\\.

## Usage

``` r
nv_qr(x)

# S3 method for class 'AnvlArray'
qr(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Matrix of data type floating-point with exactly 2 dimensions.

- ...:

  No additional arguments.

## Value

Named `list` with elements `Q` (shape `(m, k)`) and `R` (shape
`(k, n)`), where `(m, n) = shape(x)` and `k = min(m, n)`. Both have the
same data type as `x`.

## See also

[`prim_qr()`](https://r-xla.github.io/anvl/dev/reference/prim_qr.md)

## Examples

``` r
x <- nv_matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, dtype = "f32")
nv_qr(x)
#> $Q
#> AnvlArray
#>  -0.2673  0.8729
#>  -0.5345  0.2182
#>  -0.8018 -0.4364
#> [ CPUf32{3,2} ] 
#> 
#> $R
#> AnvlArray
#>  -3.7417 -8.5524
#>   0.0000  1.9640
#> [ CPUf32{2,2} ] 
#> 
```
