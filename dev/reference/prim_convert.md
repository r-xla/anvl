# Primitive Convert

Converts the elements of an array to a different data type.

## Usage

``` r
prim_convert(operand, dtype, ambiguous = FALSE)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Target data type.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvl/dev/articles/type-promotion.md)
  for more details.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the given `dtype` and the same shape as `operand`. Ambiguity is
controlled by the `ambiguous` parameter.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_convert()`](https://r-xla.github.io/stablehlo/reference/hlo_convert.html).

## See also

[`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md)

## Examples

``` r
x <- nv_array(c(1L, 2L, 3L))
prim_convert(x, dtype = "f32")
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
