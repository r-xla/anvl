# Reshape

Reshapes an array to a new shape using col-major semantics.

## Usage

``` r
nv_reshape(x, shape)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Target shape. Must have the same number of elements as `x`. At most
  one entry may be `-1`, in which case its extent is inferred from the
  remaining entries and the number of elements of `x`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the given `shape` and the same data type as `x`.

## See also

[`prim_reshape()`](https://r-xla.github.io/anvl/dev/reference/prim_reshape.md)
for the underlying primitive.

## Examples

``` r
# the elements keep their column-major order; the data type is untouched
x <- array(1:6, dim = c(3, 2))
nv_reshape(x, 6L)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPUi32{6} ] 
# the order base R reads them in, too
c(x)
#> [1] 1 2 3 4 5 6

# infer the size of the second axis
nv_reshape(x, c(2, -1))
#> AnvlArray
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2,3} ] 
# flatten
nv_reshape(x, -1)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPUi32{6} ] 
```
