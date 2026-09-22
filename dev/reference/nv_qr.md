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
  One input, with exactly 2 axes. Can be any numeric data type: a float
  keeps its own, and an integer one is converted to the default float
  data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
  An R value materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and is converted in the same way.

- ...:

  No additional arguments.

## Value

(named `list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Elements `Q` (shape `(m, k)`) and `R` (shape `(k, n)`), where
`(m, n) = shape(x)` and `k = min(m, n)`. Both have the input's data type
– or the default float data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
where the input was an integer one.

## See also

[`prim_qr()`](https://r-xla.github.io/anvl/dev/reference/prim_qr.md)

## Examples

``` r
# `Q` is 3x2 and `R` 2x2, both at the input's data type
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

# an integer matrix is decomposed at the default float data type
nv_qr(nv_matrix(1:6, nrow = 3))
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
