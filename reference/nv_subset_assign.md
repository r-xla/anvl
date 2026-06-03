# Update Subset

Updates elements of an array at specified positions, returning a new
array. You can also use the `[<-` operator.

## Usage

``` r
# S3 method for class 'AnvlArray'
x[...] <- value

nv_subset_assign(operand, ..., value)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Same as `operand`; this is the name used by the base R S3 generic.

- ...:

  Subset specifications, one per dimension. See
  [`vignette("subsetting")`](https://r-xla.github.io/anvl/articles/subsetting.md)
  for details.

- value:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Replacement values. Scalars are broadcast to the subset shape.
  Non-scalar values must match the subset shape.

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
A new array with the same shape as `operand` and the subset replaced.

## See also

[`nv_subset()`](https://r-xla.github.io/anvl/reference/nv_subset.md),
[`vignette("subsetting")`](https://r-xla.github.io/anvl/articles/subsetting.md)
for a comprehensive guide.

## Examples

``` r
x <- nv_matrix(1:12, nrow = 3)
# Set row 1 to zeros
x[1, ] <- nv_scalar(0L)
x
#> AnvlArray
#>   0  0  0  0
#>   2  5  8 11
#>   3  6  9 12
#> [ CPUi32{3,4} ] 
```
