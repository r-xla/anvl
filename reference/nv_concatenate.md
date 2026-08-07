# Concatenate

Concatenates arrays along an axis. Operands are promoted to a common
data type and scalars are broadcast before concatenation.

## Usage

``` r
nv_concatenate(..., axis = NULL)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrays to concatenate. Must have the same shape except along `axis`.

- axis:

  (`integer(1)` \| `NULL`)  
  Axis along which to concatenate. Negative values count from the end,
  i.e. `-1` refers to the last axis. If `NULL` (default), assumes all
  inputs are at most 1-D and concatenates along axis 1.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the common data type and a shape matching the inputs in all axes
except `axis`, which is the sum of input sizes.

## See also

[`prim_concatenate()`](https://r-xla.github.io/anvl/reference/prim_concatenate.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(4, 5, 6))
nv_concatenate(x, y)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPUf32{6} ] 
```
