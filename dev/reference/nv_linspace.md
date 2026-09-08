# Evenly Spaced Sequence

Creates a 1-D array with `steps` evenly spaced values from `start` to
`end` (both inclusive), like R's `seq(start, end, length.out = steps)`.

The spacing `(end - start) / (steps - 1)` is generally not a whole
number, so the result is floating-point and `dtype` must name a float
data type. Convert the result with
[`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md)
to obtain integers, which leaves the rounding yours to choose.

`nv_linspace_like()` is a variant where `dtype` and `device` default to
those of `like`.

## Usage

``` r
nv_linspace(start, end, steps, dtype = NULL, device = NULL)

nv_linspace_like(like, start, end, steps, dtype = NULL, device = NULL)
```

## Arguments

- start, end:

  (`numeric(1)`)  
  First and last value of the sequence. `end` may lie below `start`, in
  which case the values decrease.

- steps:

  (`integer(1)`)  
  Number of values to generate. Must be at least 1; for `steps = 1` the
  result is `start`.

- dtype:

  (`NULL` \| `character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Floating-point data type. `NULL` (default) uses the backend's default
  float data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
  For `nv_linspace_like()`, `NULL` uses `dtype(like)`, which must then
  be a floating-point data type.

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

- like:

  ([`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))  
  Existing array whose attributes are used as defaults (only for
  `nv_linspace_like()`).

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
1-D array of length `steps`.

## See also

[`nv_seq()`](https://r-xla.github.io/anvl/dev/reference/nv_seq.md) for
consecutive integers.

## Examples

``` r
nv_linspace(0, 1, steps = 5L)
#> AnvlArray
#>  0.0000
#>  0.2500
#>  0.5000
#>  0.7500
#>  1.0000
#> [ CPUf32{5} ] 
x <- nv_array(c(1, 2, 3), dtype = "f64")
nv_linspace_like(x, 0, 1, steps = 3L)
#> AnvlArray
#>  0.0000
#>  0.5000
#>  1.0000
#> [ CPUf64{3} ] 
```
