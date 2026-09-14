# Number of Elements

The number of elements of an array – the generic
[`base::length()`](https://rdrr.io/r/base/length.html) on an anvl array,
i.e. the product of its axis sizes.

## Usage

``` r
# S3 method for class 'AnvlArray'
length(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

(`integer(1)`)

## See also

[dim()](https://r-xla.github.io/anvl/dev/reference/dim.AnvlArray.md),
[nelts()](https://r-xla.github.io/tengen/reference/nelts.html)

## Examples

``` r
length(nv_matrix(1:6, nrow = 2))
#> [1] 6
```
