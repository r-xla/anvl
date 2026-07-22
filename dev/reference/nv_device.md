# Create a Device

Constructs a backend-specific device object.

A device identifies a compute resources, such as CPU, or a specific GPU.
It is relevant for data allocation (e.g. via
[`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))
but also compilation
([jit](https://r-xla.github.io/anvl/dev/reference/jit.md)).

## Usage

``` r
nv_device(x, backend = NULL)
```

## Arguments

- x:

  (`character(1)` \| device object)  
  Identifier for the device (e.g. `"cpu"`, `"cuda"`, `"cuda:<n>"`), or
  an existing device object (returned as-is).

- backend:

  (`NULL` \| `character(1)`)  
  The backend for which to create the device. Defaults to
  [`default_backend()`](https://r-xla.github.io/anvl/dev/reference/default_backend.md)
  when `NULL`.

## Value

A backend-specific device object (e.g. `PJRTDevice` for `"pjrt"`,
[`quickr_device`](https://r-xla.github.io/anvl/dev/reference/quickr_device.md)
for `"quickr"`).

## See also

[`backend()`](https://r-xla.github.io/anvl/dev/reference/backend.md),
[`AnvlBackend()`](https://r-xla.github.io/anvl/dev/reference/AnvlBackend.md).

## Examples

``` r
# Create CPU device for pjrt backend:
nv_device("cpu", "pjrt")
#> <CpuDevice(id=0)>
# Create CPU device for quickr backend:
nv_device("cpu", "quickr")
#> QuickrDevice(cpu) 
# Pass through an existing device:
dev <- nv_device("cpu")
identical(nv_device(dev), dev)
#> [1] TRUE
```
