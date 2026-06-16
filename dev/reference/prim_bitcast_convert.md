# Primitive Bitcast Convert

Reinterprets the bits of an array as a different data type without
modifying the underlying data.

## Usage

``` r
prim_bitcast_convert(operand, dtype)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Target data type. If it has the same bit width as the input, the
  output shape is unchanged. If narrower, an extra trailing dimension is
  added. If wider, the last dimension is consumed.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the given `dtype`.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_bitcast_convert()`](https://r-xla.github.io/stablehlo/reference/hlo_bitcast_convert.html).

## See also

[`nv_bitcast_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_bitcast_convert.md)

## Examples

``` r
x <- nv_array(1L)
prim_bitcast_convert(x, dtype = "i8")
#> AnvlArray
#>  1 0 0 0
#> [ CPUi8{1,4} ] 
x <- nv_array(rep(1L, 4), dtype = "i8")
prim_bitcast_convert(x, dtype = "i32")
#> AnvlArray
#>  1.6843e+07
#> [ CPUi32{} ] 
```
