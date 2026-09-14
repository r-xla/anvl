# Shape of an Array

The shape of an array, i.e. its axis sizes – the generic
[`base::dim()`](https://rdrr.io/r/base/dim.html) on an anvl array, and
the same thing as
[shape()](https://r-xla.github.io/tengen/reference/shape.html).

Unlike base R, it also has a value for an array with a single axis,
where [`dim()`](https://rdrr.io/r/base/dim.html) on an R vector is
`NULL`.

## Usage

``` r
# S3 method for class 'AnvlArray'
dim(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

([`integer()`](https://rdrr.io/r/base/integer.html))

## See also

[length()](https://r-xla.github.io/anvl/dev/reference/length.AnvlArray.md),
[shape()](https://r-xla.github.io/tengen/reference/shape.html)

## Examples

``` r
dim(nv_matrix(1:6, nrow = 2))
#> [1] 2 3
dim(nv_array(1:3))
#> [1] 3
```
