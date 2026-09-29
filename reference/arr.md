# Create an R Array

Create an R array without having to wrap data in
[`c()`](https://rdrr.io/r/base/c.html)

## Usage

``` r
arr(..., shape = NULL)
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

  (`NULL` \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of new array. If `NULL` (default), uses length of elements to
  create a 1D array.

## Value

(`array`)

## Examples

``` r
arr(1, 2, 3)
#> [1] 1 2 3
arr(1, 2, 3, 4, shape = c(2, 2))
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
# vectors are spliced in
arr(1:3, 4L)
#> [1] 1 2 3 4
# a single value fills the whole shape
arr(0, shape = c(2, 3))
#>      [,1] [,2] [,3]
#> [1,]    0    0    0
#> [2,]    0    0    0
```
