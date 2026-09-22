# Primitive Bitcast Conversion

Reinterprets the bits of an array as a different data type without
modifying the underlying data.

## Usage

``` r
prim_bitcast_convert(x, dtype)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type except `bool`. An R value materializes
  at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Any target data type except `bool`. One of the same bit width as the
  input's leaves the shape unchanged; a narrower one adds a trailing
  axis holding the pieces; a wider one consumes the last axis, whose
  size must equal the ratio of the two widths.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the given `dtype`, and the shape described under `dtype`.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_bitcast_convert()`](https://r-xla.github.io/stablehlo/reference/hlo_bitcast_convert.html),
specified under
[bitcast_convert](https://openxla.org/stablehlo/spec#bitcast_convert).

## See also

[`nv_bitcast_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_bitcast_convert.md)

## Examples

``` r
# same width: the bits are reread, the shape stays
prim_bitcast_convert(nv_array(1L, dtype = "i32"), dtype = "f32")
#> AnvlArray
#>  1.4013e-45
#> [ CPUf32{1} ] 

# narrower: a trailing axis holds the four bytes of each i32
prim_bitcast_convert(nv_array(1L, dtype = "i32"), dtype = "i8")
#> AnvlArray
#>  1 0 0 0
#> [ CPUi8{1,4} ] 

# wider: the last axis is consumed, and its size must be the width ratio
prim_bitcast_convert(nv_array(rep(1L, 4), dtype = "i8"), dtype = "i32")
#> AnvlArray
#>  1.6843e+07
#> [ CPUi32{} ] 
```
