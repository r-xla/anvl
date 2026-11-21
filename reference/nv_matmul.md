# Matrix Multiplication

Matrix multiplication of two arrays. You can also use the `%*%`
operator. Supports batched matrix multiplication: above rank 2, the
first two axes are the matrix and the trailing ones are batch axes,
which both operands must share.

## Usage

``` r
nv_matmul(x, y, precision = "highest")

# S3 method for class 'AnvlArray'
x %*% y
```

## Arguments

- x, y:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Numeric arrays with at least 2 axes. Can be any numeric data type; the
  two are [promoted to a common data
  type](https://r-xla.github.io/anvl/reference/nv_promote_to_common.md).
  An R value assumes the data type of the other operand, and
  materializes at its [default data
  type](https://r-xla.github.io/anvl/reference/default_dtypes.md) when
  that has none either.

- precision:

  (`character(1)`)  
  Controls the trade-off between speed and numerical accuracy of the
  operation. One of `"highest"` (default), `"high"` or `"default"`. See
  [`prim_dot_general()`](https://r-xla.github.io/anvl/reference/prim_dot_general.md)
  for details.

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
Has the operands' common data type and the shape given under Shapes.

## Shapes

- `x`: `(m, n, b1, ..., bk)`

- `y`: `(n, p, b1, ..., bk)`

- output: `(m, p, b1, ..., bk)`

## See also

[`prim_dot_general()`](https://r-xla.github.io/anvl/reference/prim_dot_general.md)
for the underlying primitive.

## Examples

``` r
# a 2x3 times a 3x2 gives a 2x2 at the operands' common data type
x <- nv_matrix(1:6, nrow = 2)
y <- nv_matrix(1:6, nrow = 3)
nv_matmul(x, y)
#> AnvlArray
#>  22 49
#>  28 64
#> [ CPUi32{2,2} ] 
x %*% y
#> AnvlArray
#>  22 49
#>  28 64
#> [ CPUi32{2,2} ] 
# a batch of three such products along the trailing axis
x <- nv_array(1:18, shape = c(2, 3, 3))
y <- nv_array(1:18, shape = c(3, 2, 3))
x %*% y
#> AnvlArray
#> (1,.,.) =
#>   22 220 634
#>   49 301 769
#> 
#> (2,.,.) =
#>   28 244 676
#>   64 334 820
#> [ CPUi32{2,2,3} ] 
```
