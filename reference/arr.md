# Create an R array

Create an R array without having to wrap data in
[`c()`](https://rdrr.io/r/base/c.html)

## Usage

``` r
arr(..., shape = NULL)
```

## Arguments

- ...:

  (any)  
  Values of new array.

- shape:

  (`NULL` \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of new array. If `NULL` (default), uses length of elements to
  create a 1D array.

## Examples

``` r
arr(1, 2, 3)
#> [1] 1 2 3
arr(1, 2, 3, 4, shape = c(2, 2))
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
```
