# Subset an Array

Extracts a subset from an array. You can also use the `[` operator.
Supports R-style indexing including scalar indices (which drop axes),
ranges (`a:b`), and `array(c(...))` for selecting multiple elements
along a axis.

## Usage

``` r
nv_subset(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- ...:

  Subset specifications, one per axis. Omitted trailing axes select all
  elements. See
  [`vignette("subsetting")`](https://r-xla.github.io/anvl/dev/articles/subsetting.md)
  for details.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's data type, and the shape the specifications select – a
scalar index drops its axis, a range or an index array keeps it.

## See also

[`nv_subset_assign()`](https://r-xla.github.io/anvl/dev/reference/nv_subset_assign.md)
for updating subsets,
[`vignette("subsetting")`](https://r-xla.github.io/anvl/dev/articles/subsetting.md)
for a comprehensive guide.

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
```
