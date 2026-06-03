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

  A dtype (string or
  [`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html)).

- shape:

  An integer vector or
  [`stablehlo::Shape`](https://r-xla.github.io/stablehlo/reference/Shape.html).

## Value

(`ValueType`)
