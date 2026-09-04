# Create a Shape object

Constructs a `Shape`, the axis sizes of an array. A `Shape` *is* its
integer vector, with a class attached, so
[`length()`](https://rdrr.io/r/base/length.html) is the number of axes
and `shape[i]` is the size of axis `i`.

## Usage

``` r
Shape(dims = integer())
```

## Arguments

- dims:

  An [`integer()`](https://rdrr.io/r/base/integer.html) vector of axis
  sizes (\>= 0). `NA` marks an axis whose size is only known at run
  time.

## Value

A `Shape` object.

## See also

[`shape()`](https://r-xla.github.io/anvl/dev/reference/shape.md),
[`stablehlo::Shape()`](https://r-xla.github.io/stablehlo/reference/Shape.html)

## Examples

``` r
Shape(c(2L, 3L))
#> (2x3)
```
