# Iota

Creates an array with values increasing along the specified axis,
starting from `start`.

`nv_iota_like()` is a variant where `dtype`, `shape`, `ambiguous`, and
`device` default to those of `like`.

## Usage

``` r
nv_iota(axis, dtype, shape, start = 1L, ambiguous = FALSE, device = NULL)

nv_iota_like(
  like,
  axis,
  shape = NULL,
  start = 1L,
  dtype = NULL,
  ambiguous = NULL,
  device = NULL
)
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
  Shape.

- start:

  (`integer(1)`)  
  Starting value (default 1).

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvl/articles/type-promotion.md)
  for more details.

- device:

  (`NULL` \| `character(1)` \|
  [device](https://r-xla.github.io/anvl/reference/nv_device.md))  
  The device the data lives on, given either as:

  - a *device string* naming the platform (e.g. `"cpu"`, `"cuda"`,
    `"cuda:<n>"`), which is resolved against the backend in use, or

  - a *device object* as returned by
    [`nv_device()`](https://r-xla.github.io/anvl/reference/nv_device.md):
    a
    [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html)
    for the `"pjrt"` backend or a
    [`quickr_device`](https://r-xla.github.io/anvl/reference/quickr_device.md)
    for the `"quickr"` backend. Because a device object is
    backend-specific, it also determines the backend.

  The default (`NULL`) uses
  [`default_device()`](https://r-xla.github.io/anvl/reference/default_device.md):
  the CPU, or the platform named by the `PJRT_PLATFORM` environment
  variable on the `"pjrt"` backend.

- like:

  ([`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md))  
  Existing array whose attributes are used as defaults (only for
  `nv_iota_like()`).

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the given `dtype` and `shape`.

## See also

[`nv_seq()`](https://r-xla.github.io/anvl/reference/nv_seq.md) for a
simpler 1-D sequence,
[`prim_iota()`](https://r-xla.github.io/anvl/reference/prim_iota.md) for
the underlying primitive.

## Examples

``` r
nv_iota(axis = 1L, dtype = "i32", shape = 5L)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#>  5
#> [ CPUi32{5} ] 
x <- nv_fill(0L, shape = c(2, 3))
nv_iota_like(x, axis = 1L)
#> AnvlArray
#>  1 1 1
#>  2 2 2
#> [ CPUi32{2,3} ] 
```
