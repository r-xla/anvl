# Update Subset

Updates elements of an array at specified positions, returning a new
array. You can also use the `[<-` operator.

## Usage

``` r
nv_subset_assign(x, ..., value, inplace = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The array to update. Can be any data type. An R object is materialized
  at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- ...:

  Subset specifications, one per axis. See
  [`vignette("subsetting")`](https://r-xla.github.io/anvl/dev/articles/subsetting.md)
  for details.

- value:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Replacement values. Scalars are broadcast to the subset shape and
  non-scalar values must match it. The value is converted to the data
  type of `x`.

- inplace:

  (`logical(1)`)  
  Whether to write into the memory of `x` instead of allocating a new
  array. This avoids the copy of `x` that an eager subset assignment
  otherwise makes, but it consumes `x`: its buffer is
  [donated](https://r-xla.github.io/anvl/dev/reference/jit.md), so `x` –
  and any other R variable referring to the same array – can no longer
  be used afterwards. With `[<-`, pass it among the subscripts, e.g.
  `x[1, inplace = TRUE] <- 0`, which rebinds `x` to the result. It is an
  error inside a jitted function: there, other references to `x` would
  stay valid, so the same code would behave differently in eager and in
  jit mode. Inside
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md), the
  compiler already avoids the copy anyway. Default is `FALSE`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has `x`'s data type and shape, with the subset replaced.

## See also

[`nv_subset()`](https://r-xla.github.io/anvl/dev/reference/nv_subset.md),
[`vignette("subsetting")`](https://r-xla.github.io/anvl/dev/articles/subsetting.md)
for a comprehensive guide.

## Examples

``` r
x <- nv_matrix(1:12, nrow = 3)
# set row 1 to zeros
nv_subset_assign(x, 1, value = nv_scalar(0L))
#> AnvlArray
#>   0  0  0  0
#>   2  5  8 11
#>   3  6  9 12
#> [ CPUi32{3,4} ] 
x[1, ] <- nv_scalar(0L)
x
#> AnvlArray
#>   0  0  0  0
#>   2  5  8 11
#>   3  6  9 12
#> [ CPUi32{3,4} ] 

# write into the memory of `x` instead of copying it
x[2, inplace = TRUE] <- nv_scalar(1L)
x
#> AnvlArray
#>   0  0  0  0
#>   1  1  1  1
#>   3  6  9 12
#> [ CPUi32{3,4} ] 
```
