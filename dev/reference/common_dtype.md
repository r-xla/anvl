# Type Promotion Rules

Compute the common data type.

## Usage

``` r
common_dtype(lhs_dtype, rhs_dtype)
```

## Arguments

- lhs_dtype:

  ([`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  The left-hand side type.

- rhs_dtype:

  ([`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  The right-hand side type.

## Value

([`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))

## Examples

``` r
common_dtype("i32", "f32")
#> <f32>
common_dtype("i32", "i64")
#> <i64>
```
