# Convert to a DataType

Coerces a value to a `DataType`. Accepts data type strings (e.g.
`"f32"`, `"i64"`, `"bool"`) or existing `DataType` objects (they are
returned unchanged).

## Usage

``` r
as_dtype(x)
```

## Arguments

- x:

  A character string or `DataType` to convert.

## Value

(`DataType`)

## Details

This is implemented via the generic
[`tengen::as_dtype()`](https://r-xla.github.io/tengen/reference/as_dtype.html).

## See also

[`is_dtype()`](https://r-xla.github.io/anvl/dev/reference/is_dtype.md),
[`tengen::as_dtype()`](https://r-xla.github.io/tengen/reference/as_dtype.html),
[`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html)

## Examples

``` r
as_dtype("f32")
#> <f32>
as_dtype("i32")
#> <i32>
```
