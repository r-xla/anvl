# Construct a stablehlo ValueType

Shorthand for building a tensor
[`stablehlo::ValueType`](https://r-xla.github.io/stablehlo/reference/ValueType.html)
from a dtype and shape — convenient inside stablehlo lowering rules that
need to declare custom-call output types or similar.

## Usage

``` r
vt(dtype, shape)
```

## Arguments

- dtype:

  (`character(1)` \|
  [`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  The data type.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \|
  [`stablehlo::Shape`](https://r-xla.github.io/stablehlo/reference/Shape.html))  
  The shape.

## Value

([`stablehlo::ValueType`](https://r-xla.github.io/stablehlo/reference/ValueType.html))  
A tensor type of the given data type and shape.

## Examples

``` r
vt("f32", c(2L, 3L))
#> <ValueType: tensor<2x3xf32>>
```
