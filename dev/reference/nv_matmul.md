# Matrix Multiplication

Matrix multiplication of two arrays. You can also use the `%*%`
operator. Supports batched matrix multiplication when inputs have more
than 2 axes.

## Usage

``` r
nv_matmul(lhs, rhs, precision = "highest")
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Numeric arrays with at least 2 axes. Can be any numeric data type; the
  two are [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md).
  An R value assumes the data type of the other operand, and
  materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when that has none either.

- precision:

  (`character(1)`)  
  Controls the trade-off between speed and numerical accuracy of the
  operation. One of `"highest"` (default), `"high"` or `"default"`. See
  [`prim_dot_general()`](https://r-xla.github.io/anvl/dev/reference/prim_dot_general.md)
  for details.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the operands' common data type and the shape given under Shapes.

## Shapes

- `lhs`: `(b1, ..., bk, m, n)`

- `rhs`: `(b1, ..., bk, n, p)`

- output: `(b1, ..., bk, m, p)`

## See also

[`prim_dot_general()`](https://r-xla.github.io/anvl/dev/reference/prim_dot_general.md)
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
```
