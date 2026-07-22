# Reshape

Reshapes an array to a new shape without changing the underlying data.
Returns the input unchanged if it already has the target shape.

## Usage

``` r
nv_reshape(x, shape)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Target shape. Must have the same number of elements as `x`. At most
  one entry may be `-1`, in which case its extent is inferred from the
  remaining entries and the number of elements of `x`.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the given `shape` and the same data type as `x`.

## Details

Note that row-major order is used, which differs from R's column-major
order.

## See also

[`prim_reshape()`](https://r-xla.github.io/anvl/dev/reference/prim_reshape.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(1:6)
nv_reshape(x, c(2, 3))
#> AnvlArray
#>  1 2 3
#>  4 5 6
#> [ CPUi32{2,3} ] 
nv_reshape(x, c(2, -1)) # infer the second dimension
#> AnvlArray
#>  1 2 3
#>  4 5 6
#> [ CPUi32{2,3} ] 
nv_reshape(x, -1) # flatten
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPUi32{6} ] 
```
