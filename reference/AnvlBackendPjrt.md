# PJRT Backend

Constructs the PJRT backend, which stores array data in PJRT buffers
(via
[`pjrt::pjrt_buffer()`](https://r-xla.github.io/pjrt/reference/pjrt_buffer.html))
and compiles jitted functions to XLA executables via
[`stablehlo()`](https://r-xla.github.io/anvl/reference/stablehlo.md) and
[`pjrt::pjrt_compile()`](https://r-xla.github.io/pjrt/reference/pjrt_compile.html).
This is the default backend.

## Usage

``` r
AnvlBackendPjrt()
```

## Value

([`AnvlBackend`](https://r-xla.github.io/anvl/reference/AnvlBackend.md))  
With subclass `"AnvlBackendPjrt"`.

## Data representation

An [`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md)
with `backend = "pjrt"` wraps a
[`pjrt::pjrt_buffer()`](https://r-xla.github.io/pjrt/reference/pjrt_buffer.html)
stored in the `$data` field. The buffer owns the memory holding the
array values and may live on any device supported by PJRT (CPU, CUDA,
...). Calling
[`as_array()`](https://r-xla.github.io/anvl/reference/as_array.md)
transfers the buffer contents back to an R array; calling
[`nv_array()`](https://r-xla.github.io/anvl/reference/AnvlArray.md) on
an R object uploads it to the requested device.

Each `AnvlArray` therefore has an associated device, queryable via
[`device()`](https://r-xla.github.io/anvl/reference/device.md). A device
is a
[`pjrt::as_pjrt_device()`](https://r-xla.github.io/pjrt/reference/as_pjrt_device.html)
object (e.g. the platform `"cpu"` or `"cuda"`, optionally with an index
such as `"cuda:1"`). When `device` is `NULL` in
[`nv_array()`](https://r-xla.github.io/anvl/reference/AnvlArray.md) or
the [`jit()`](https://r-xla.github.io/anvl/reference/jit.md) wrapper,
the device defaults to
[`default_device()`](https://r-xla.github.io/anvl/reference/default_device.md),
or is inferred from the existing inputs of a jitted call. Operations
require all inputs to live on the same device.

## Supported data types

`bool`; the signed integers `i8`, `i16`, `i32` and `i64`; the unsigned
integers `ui8`, `ui16`, `ui32` and `ui64`; and the floats `f32` and
`f64`. An R double materializes at `f32` on this backend and an R
integer at `i32` unless the defaults say otherwise (see
[`default_dtypes()`](https://r-xla.github.io/anvl/reference/default_dtypes.md)).

## Floating-point behavior

Subnormal floating-point values may be preserved when stored in an array
and read back into R, yet treated as zero in calculations. On CPUs, XLA
enables a mode that replaces subnormal inputs and results with zero. The
exact behavior depends on the platform, backend, and operation.

See the [Gotchas](https://r-xla.github.io/anvl/articles/gotchas.html)
article for an explanation and examples.

## See also

[`AnvlBackend()`](https://r-xla.github.io/anvl/reference/AnvlBackend.md),
[`AnvlBackendQuickr()`](https://r-xla.github.io/anvl/reference/AnvlBackendQuickr.md),
[`local_backend()`](https://r-xla.github.io/anvl/reference/local_backend.md),
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md).
