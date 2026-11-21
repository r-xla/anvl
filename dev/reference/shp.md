# Create a Shape

Creates the shape of an array from its axis sizes: `shp(2, 3)` is the
shape of a 2x3 matrix, and `shp()` is the shape of a scalar, which has
no axes.

The result is the integer vector that
[`shape()`](https://r-xla.github.io/xlamisc/reference/shape.html)
returns, so it can be passed to any `shape` argument and compared with
`identical(shape(x), shp(2, 3))`. Unlike
[`c()`](https://rdrr.io/r/base/c.html), which returns `NULL` when called
without arguments, `shp()` always returns an integer vector. A `shape`
argument refuses `NULL`, so a `NULL` that slips in by accident – such as
the [`dim()`](https://rdrr.io/r/base/dim.html) of a vector – is an error
rather than the shape of a scalar.

## Usage

``` r
shp(...)
```

## Arguments

- ...:

  ([`numeric()`](https://rdrr.io/r/base/numeric.html))  
  The axis sizes: non-negative whole numbers. Vectors are spliced in, so
  `shp(shape(x), 3)` appends an axis of size 3 to the shape of `x`.
  `NULL`, a logical and a factor are refused.

## Value

([`integer()`](https://rdrr.io/r/base/integer.html))  
The axis sizes.

## See also

[shape()](https://r-xla.github.io/xlamisc/reference/shape.html) for the
shape of an array,
[`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md).

## Examples

``` r
shp(2, 3)
#> [1] 2 3
# the shape of a scalar
shp()
#> integer(0)
# vectors are spliced in
shp(c(2, 3), 4)
#> [1] 2 3 4
nv_array(1:6, shape = shp(2, 3))
#> AnvlArray
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2,3} ] 
nv_array(1L, shape = shp())
#> AnvlArray
#>  1
#> [ CPUi32{} ] 
```
