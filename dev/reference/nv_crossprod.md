# Cross Product (Matrix)

Computes `t(lhs) %*% rhs`. If `rhs` is missing, computes
`t(lhs) %*% lhs`.

## Usage

``` r
nv_crossprod(lhs, rhs = NULL)

# S3 method for class 'AnvlArray'
crossprod(x, y = NULL, ...)
```

## Arguments

- lhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  An array with at least 2 axes.

- rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
  \| `NULL`)  
  Optional second array. If `NULL`, uses `lhs`.

- x, y:

  Same as `lhs` and `rhs`; the names used by the base R S3 generic.

- ...:

  No additional arguments.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)

## See also

[`nv_tcrossprod()`](https://r-xla.github.io/anvl/dev/reference/nv_tcrossprod.md),
[`nv_matmul()`](https://r-xla.github.io/anvl/dev/reference/nv_matmul.md)

## Examples

``` r
x <- nv_matrix(1:6, nrow = 3, dtype = "f32")
nv_crossprod(x)
#> AnvlArray
#>  14 32
#>  32 77
#> [ CPUf32{2,2} ] 
```
