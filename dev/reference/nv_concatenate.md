# Concatenate

Concatenates arrays along an axis. Operands are promoted to a common
data type and scalars are broadcast before concatenation.

You can also use [`c()`](https://rdrr.io/r/base/c.html), which flattens
its arguments first, like base R.

## Usage

``` r
nv_concatenate(..., axis = NULL)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrays to concatenate. Must have the same shape except along `axis`.

- axis:

  (`integer(1)` \| `NULL`)  
  Axis along which to concatenate. Negative values count from the end,
  i.e. `-1` refers to the last axis. If `NULL` (default), assumes all
  inputs are at most 1-D and concatenates along axis 1.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the common data type and a shape matching the inputs in all axes
except `axis`, which is the sum of input sizes.

## The [`c()`](https://rdrr.io/r/base/c.html) generic

[`c()`](https://rdrr.io/r/base/c.html) concatenates scalars and 1-D
arrays into a 1-D array, like
[`base::c()`](https://rdrr.io/r/base/c.html) does for vectors. An array
with more than one axis is an error: base R would flatten it in
column-major order, whereas an anvl array flattens in row-major order
(see the "Gotchas" vignette), so concatenate those along an explicit
`axis` instead.

## See also

[`prim_concatenate()`](https://r-xla.github.io/anvl/dev/reference/prim_concatenate.md)
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
