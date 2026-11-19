# Create an R Array

Create an R array without having to wrap data in
[`c()`](https://rdrr.io/r/base/c.html)

## Usage

``` r
arr(..., shape = "auto")
```

## Arguments

- ...:

  (atomic vectors)  
  Values of the new array. They are combined with
  [`c()`](https://rdrr.io/r/base/c.html), so vectors are spliced in and
  the result takes the common R type of all values, and fill the array
  in column-major order. A single value is recycled to fill all of
  `shape`.

- shape:

  (`"auto"` \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of the new array, with at least one axis. The default `"auto"`
  creates a 1-D array holding all values.

## Value

(`array`)

## See also

[`shp()`](https://r-xla.github.io/anvl/reference/shp.md) to create a
shape.

## Examples

``` r
arr(1, 2, 3)
#> [1] 1 2 3
arr(1, 2, 3, 4, shape = shp(2, 2))
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
# vectors are spliced in
arr(1:3, 4L)
#> [1] 1 2 3 4
# a single value fills the whole shape
arr(0, shape = shp(2, 3))
#>      [,1] [,2] [,3]
#> [1,]    0    0    0
#> [2,]    0    0    0
```
