# Primitive Iota

Creates an array with values increasing along the specified axis.

## Usage

``` r
prim_iota(axis, dtype, shape, start = 1L, ambiguous = FALSE, device = NULL)
```

## Arguments

- axis:

  (`integer(1)`)  
  Axis along which values increase. Negative values count from the end
  of `shape`, i.e. `-1` refers to the last axis.

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of the output array.

- start:

  (`integer(1)`)  
  Starting value.

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
Has the given `dtype` and `shape`.

## Implemented Rules

- `stablehlo`

- `quickr`

## StableHLO

Lowers to
[`hlo_iota()`](https://r-xla.github.io/stablehlo/reference/hlo_iota.html).

## See also

[`nv_iota()`](https://r-xla.github.io/anvl/dev/reference/nv_iota.md)

## Examples

``` r
prim_iota(axis = 1L, dtype = "i32", shape = 5L)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#>  5
#> [ CPUi32{5} ] 
```
