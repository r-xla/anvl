# Subset an Array

Extracts a subset from an array. You can also use the `[` operator.
Supports R-style indexing including scalar indices (which drop axes),
ranges (`a:b`), `array(c(...))` for selecting multiple elements along an
axis, and boolean masks.

## Usage

``` r
# S3 method for class 'AnvlArray'
x[...]

nv_subset(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/reference/default_dtypes.md).

- ...:

  Subset specifications, one per axis. Omitted trailing axes select all
  elements.

  A boolean mask (an R logical array such as `arr(TRUE, FALSE)`, or an
  arrayish value of dtype `bool`) selects the elements at the `TRUE`
  positions. A mask for one axis must have as many elements as the size
  of that axis. A mask that is the only subscript and has the same shape
  as `x` selects across the whole array, yielding a 1-D result. Under
  [`jit()`](https://r-xla.github.io/anvl/reference/jit.md), the values
  of a mask must be known at compile time, because the number of
  selected elements determines the output shape: R logical arrays and
  arrays created in or closed over by the function work, a mask computed
  from the function's inputs does not.

  See the
  [Subsetting](https://r-xla.github.io/anvl/articles/subsetting.html)
  article for details.

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
Has the input's data type, and the shape the specifications select – a
scalar index drops its axis, a range, an index array or an axis mask
keeps it, and a whole-array mask yields a 1-D result.

## See also

[`nv_subset_assign()`](https://r-xla.github.io/anvl/reference/nv_subset_assign.md)
for updating subsets, the
[Subsetting](https://r-xla.github.io/anvl/articles/subsetting.html)
article for a comprehensive guide.

## Examples

``` r
x <- nv_matrix(1:12, nrow = 3)
x
#> AnvlArray
#>   1  4  7 10
#>   2  5  8 11
#>   3  6  9 12
#> [ CPUi32{3,4} ] 
# select row 2
nv_subset(x, 2)
#> AnvlArray
#>   2
#>   5
#>   8
#>  11
#> [ CPUi32{4} ] 
x[2, ]
#> AnvlArray
#>   2
#>   5
#>   8
#>  11
#> [ CPUi32{4} ] 

# select rows 1 to 2, all columns
nv_subset(x, 1:2)
#> AnvlArray
#>   1  4  7 10
#>   2  5  8 11
#> [ CPUi32{2,4} ] 
x[1:2, ]
#> AnvlArray
#>   1  4  7 10
#>   2  5  8 11
#> [ CPUi32{2,4} ] 

# Select rows 1 and 3 with a mask
x[arr(TRUE, FALSE, TRUE), ]
#> AnvlArray
#>   1  4  7 10
#>   3  6  9 12
#> [ CPUi32{2,4} ] 

# Select all elements greater than 6 (not in `jit()`, see above)
x[x > 6]
#> AnvlArray
#>   7
#>   8
#>   9
#>  10
#>  11
#>  12
#> [ CPUi32{6} ] 
```
