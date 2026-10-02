# Create an Array of a Given Data Type

Shorthands for
[`nv_array()`](https://r-xla.github.io/anvl/reference/AnvlArray.md) with
the data type fixed by the function name: `nv_float32(data)` is
`nv_array(data, dtype = "f32")`.

|                |           |
|----------------|-----------|
| Function       | Data type |
| `nv_bool()`    | `bool`    |
| `nv_int8()`    | `i8`      |
| `nv_int16()`   | `i16`     |
| `nv_int32()`   | `i32`     |
| `nv_int64()`   | `i64`     |
| `nv_uint8()`   | `ui8`     |
| `nv_uint16()`  | `ui16`    |
| `nv_uint32()`  | `ui32`    |
| `nv_uint64()`  | `ui64`    |
| `nv_float32()` | `f32`     |
| `nv_float64()` | `f64`     |

## Usage

``` r
nv_bool(data, shape = NULL, device = NULL, byrow = FALSE)

nv_int8(data, shape = NULL, device = NULL, byrow = FALSE)

nv_int16(data, shape = NULL, device = NULL, byrow = FALSE)

nv_int32(data, shape = NULL, device = NULL, byrow = FALSE)

nv_int64(data, shape = NULL, device = NULL, byrow = FALSE)

nv_uint8(data, shape = NULL, device = NULL, byrow = FALSE)

nv_uint16(data, shape = NULL, device = NULL, byrow = FALSE)

nv_uint32(data, shape = NULL, device = NULL, byrow = FALSE)

nv_uint64(data, shape = NULL, device = NULL, byrow = FALSE)

nv_float32(data, shape = NULL, device = NULL, byrow = FALSE)

nv_float64(data, shape = NULL, device = NULL, byrow = FALSE)
```

## Arguments

- data:

  (any)  
  [`integer()`](https://rdrr.io/r/base/integer.html),
  [`double()`](https://rdrr.io/r/base/double.html), or
  [`logical()`](https://rdrr.io/r/base/logical.html) scalar, vector, or
  array. Alternatively a [`raw()`](https://rdrr.io/r/base/raw.html)
  vector holding the native little-endian byte payload of `prod(shape)`
  elements of `dtype`; both `dtype` and `shape` are then required (only
  supported on the `"pjrt"` backend). Raw payloads are read in
  column-major element order, or row-major with `byrow = TRUE`. An
  existing
  [`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md) is
  returned unchanged if the `shape`, `dtype` and `device` given agree
  with it, and is an error otherwise.

- shape:

  (`NULL` \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The output shape of the array. The default (`NULL`) is to infer it
  from the data if possible. Note that
  [`nv_array`](https://r-xla.github.io/anvl/reference/AnvlArray.md)
  interprets length 1 vectors as having shape `(1)`. Empty data has no
  shape to infer – `0`, `c(2, 0)` and `c(0, 3)` all hold no elements –
  so `shape` is required there. To create a "scalar" with no axes (shape
  `()`), use
  [`nv_scalar`](https://r-xla.github.io/anvl/reference/AnvlArray.md) or
  explicitly specify `shape = integer()`.

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
    for the `"quickr"` backend. It must belong to the active backend
    ([`active_backend()`](https://r-xla.github.io/anvl/reference/active_backend.md));
    a device of another backend is an error.

  The default (`NULL`) uses
  [`default_device()`](https://r-xla.github.io/anvl/reference/default_device.md).

- byrow:

  (`logical(1)`)  
  When constructing from an R object and the result has at least two
  axes, fill the array in row-major order rather than the default
  column-major order. Only allowed when `data` is an R object — passing
  an existing `AnvlArray` together with `byrow = TRUE` is an error.

## Value

([`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md))

## Out of Range values

Because base R has fewer data types than anvl, creating `AnvlArray`s
from R often involves type conversions. When such conversions are
performed, anvl performs a scan of the inputs to ensure that the
requested data type can actually hold the input data. For example,
trying to create an unsigned integer from a negative R
[`integer()`](https://rdrr.io/r/base/integer.html) fails. The same holds
where an R value takes its data type from the array it meets rather than
from an argument: `nv_scalar(1L, "ui8") + (-2L)` is refused, where
converting an array with
[`nv_convert()`](https://r-xla.github.io/anvl/reference/nv_convert.md)
wraps around.

## See also

[`nv_array()`](https://r-xla.github.io/anvl/reference/AnvlArray.md),
[`nv_convert()`](https://r-xla.github.io/anvl/reference/nv_convert.md)
to change the data type of an existing array.

## Examples

``` r
nv_float32(1:4)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#> [ CPUf32{4} ] 
nv_uint8(c(0, 255))
#> AnvlArray
#>    0
#>  255
#> [ CPUui8{2} ] 
nv_int64(1:6, shape = c(2, 3))
#> AnvlArray
#>  1 3 5
#>  2 4 6
#> [ CPUi64{2,3} ] 
nv_bool(c(TRUE, FALSE))
#> AnvlArray
#>  1
#>  0
#> [ CPUbool{2} ] 
```
