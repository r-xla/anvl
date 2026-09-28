# Convert AbstractArray to ValueType

Convert an
[`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md)
to a
[`stablehlo::ValueType`](https://r-xla.github.io/stablehlo/reference/ValueType.html).

## Usage

``` r
at2vt(x)
```

## Arguments

- x:

  ([`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md))  
  The abstract array. Must not be an
  [`RData`](https://r-xla.github.io/anvl/dev/reference/RData.md).

## Value

([`stablehlo::ValueType`](https://r-xla.github.io/stablehlo/reference/ValueType.html))  
A tensor type of `x`'s data type and shape.

## Examples

``` r
at2vt(nv_aval("f32", c(2L, 3L)))
#> <ValueType: tensor<2x3xf32>>
```
