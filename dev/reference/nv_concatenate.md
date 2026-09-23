# Concatenate

Concatenates arrays along an axis. Operands are promoted to a common
data type and scalars are broadcast before concatenation.

You can also use [`c()`](https://rdrr.io/r/base/c.html) on scalars and
1-D arrays, like base R.

## Usage

``` r
nv_concatenate(..., axis = NULL)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrays to concatenate. Can be of any data type; they are [promoted to
  a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md)
  and scalars are
  [broadcast](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_scalars.md).
  Must have the same shape except along `axis`.

- axis:

  (`integer(1)` \| `NULL`)  
  Axis along which to concatenate. Negative values count from the end,
  i.e. `-1` refers to the last axis. If `NULL` (default), concatenates
  along axis 1, which requires every input to have at most one axis; for
  anything else `axis` must be given, since there is no neutral axis to
  join two matrices along.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the common data type and a shape matching the inputs in all axes
except `axis`, which is the sum of input sizes.

## The [`c()`](https://rdrr.io/r/base/c.html) generic

[`c()`](https://rdrr.io/r/base/c.html) concatenates scalars and 1-D
arrays into a 1-D array, like
[`base::c()`](https://rdrr.io/r/base/c.html) does for vectors. An array
with more than one axis is an error rather than being flattened the way
base R does; concatenate those along an explicit `axis`, or
[`nv_flatten()`](https://r-xla.github.io/anvl/dev/reference/nv_flatten.md)
them first.

## See also

[`prim_concatenate()`](https://r-xla.github.io/anvl/dev/reference/prim_concatenate.md)
for the underlying primitive.

## Examples

``` r
# the operands are promoted to a common data type; axis 1 grows to 6
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

m <- nv_matrix(1:4, nrow = 2)
nv_concatenate(m, m, axis = 1L) # required: `m` has two axes
#> AnvlArray
#>  1 3
#>  2 4
#>  1 3
#>  2 4
#> [ CPUi32{4,2} ] 
```
