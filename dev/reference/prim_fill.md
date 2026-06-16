# Primitive Fill

Creates an array of a given shape and data type, filled with a scalar
value. The advantage of using this function instead of e.g. doing
`nv_array(1, shape = c(100, 100))` is that lowering of `prim_fill()` is
efficiently represented in the compiled program, while the latter uses
100 \* 100 \* 4 bytes of memory.

## Usage

``` r
prim_fill(value, shape, dtype, ambiguous = FALSE, device = NULL)
```

## Arguments

- value:

  (`numeric(1)`)  
  Scalar value to fill the array with.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of the output array.

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvl/dev/articles/type-promotion.md)
  for more details.

- device:

  ( `character(1)` \| `PJRTDevice` \|
  [`quickr_device`](https://r-xla.github.io/anvl/dev/reference/quickr_device.md)
  \| `NULL`)  
  Device for data to live on.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the given `shape` and `dtype`.

## Implemented Rules

- `stablehlo`

- `quickr`

## StableHLO

Lowers to
[`hlo_tensor()`](https://r-xla.github.io/stablehlo/reference/hlo_constant.html).

## See also

[`nv_fill()`](https://r-xla.github.io/anvl/dev/reference/nv_fill.md)

## Examples

``` r
prim_fill(3.14, shape = c(2, 3), dtype = "f32")
#> AnvlArray
#>  3.1400 3.1400 3.1400
#>  3.1400 3.1400 3.1400
#> [ CPUf32{2,3} ] 
```
