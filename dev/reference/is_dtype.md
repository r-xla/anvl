# Check if an object is a DataType

Tests whether `x` is a `DataType` object.

## Usage

``` r
is_dtype(x)
```

## Arguments

- x:

  An object to test.

## Value

(`logical(1)`)

## See also

[`as_dtype()`](https://r-xla.github.io/anvl/dev/reference/as_dtype.md),
[`tengen::is_dtype()`](https://r-xla.github.io/tengen/reference/is_dtype.html)

## Examples

``` r
is_dtype("f32")
#> [1] FALSE
is_dtype(as_dtype("f32"))
#> [1] TRUE
```
