# Primitive Fill

Creates an array of a given shape and data type, filled with a scalar
value. The advantage of using this function instead of e.g. doing
`nv_array(1, shape = c(100, 100))` is that lowering of `prim_fill()` is
efficiently represented in the compiled program, while the latter uses
100 \* 100 \* 4 bytes of memory.

## Usage

``` r
prim_fill(value, shape, dtype, device = NULL)
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

- device:

  (`NULL` \| `character(1)` \|
  [device](https://r-xla.github.io/anvl/dev/reference/nv_device.md))  
  The device the data lives on, given either as:

  - a *device string* naming the platform (e.g. `"cpu"`, `"cuda"`,
    `"cuda:<n>"`), which is resolved against the backend in use, or

  - a *device object* as returned by
    [`nv_device()`](https://r-xla.github.io/anvl/dev/reference/nv_device.md):
    a
    [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html)
    for the `"pjrt"` backend or a
    [`quickr_device`](https://r-xla.github.io/anvl/dev/reference/quickr_device.md)
    for the `"quickr"` backend. Because a device object is
    backend-specific, it also determines the backend.

  The default (`NULL`) uses
  [`default_device()`](https://r-xla.github.io/anvl/dev/reference/default_device.md):
  the CPU, or the platform named by the `PJRT_PLATFORM` environment
  variable on the `"pjrt"` backend.

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
