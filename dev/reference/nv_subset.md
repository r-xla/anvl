# Subset an Array

Extracts a subset from an array. You can also use the `[` operator.
Supports R-style indexing including scalar indices (which drop
dimensions), ranges (`a:b`), and `array(c(...))` for selecting multiple
elements along a dimension.

## Usage

``` r
# S3 method for class 'AnvlArray'
x[...]

nv_subset(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- ...:

  Subset specifications, one per dimension. Omitted trailing dimensions
  select all elements. See
  [`vignette("subsetting")`](https://r-xla.github.io/anvl/dev/articles/subsetting.md)
  for details.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)

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
# Select row 2
x[2, ]
#> AnvlArray
#>   2
#>   5
#>   8
#>  11
#> [ CPUi32{4} ] 

# Select rows 1 to 2, all columns
x[1:2, ]
#> AnvlArray
#>   1  4  7 10
#>   2  5  8 11
#> [ CPUi32{2,4} ] 
```
