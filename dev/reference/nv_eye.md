# Identity Matrix

Creates an `n x n` identity matrix.

`nv_eye_like()` is a variant where `dtype` and `device` default to those
of `like`.

## Usage

``` r
nv_eye(n, dtype = "f32", device = NULL)

nv_eye_like(like, n, dtype = NULL, device = NULL)
```

## Arguments

- n:

  (`integer(1)`)  
  Size of the identity matrix.

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

- like:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Existing array whose attributes are used as defaults (only for
  `nv_eye_like()`).

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
An `n x n` identity matrix.

## See also

[`nv_diag()`](https://r-xla.github.io/anvl/dev/reference/nv_diag.md) for
general diagonal matrices.

## Examples

``` r
nv_eye(3L)
#> AnvlArray
#>  1 0 0
#>  0 1 0
#>  0 0 1
#> [ CPUf32{3,3} ] 
x <- nv_fill(0, shape = c(3, 3), dtype = "f64")
nv_eye_like(x, 3L)
#> AnvlArray
#>  1 0 0
#>  0 1 0
#>  0 0 1
#> [ CPUf64{3,3} ] 
```
