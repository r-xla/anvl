# Create a Device

Constructs a backend-specific device object for the active backend
([`active_backend()`](https://r-xla.github.io/anvl/dev/reference/active_backend.md)).

A device identifies a compute resources, such as CPU, or a specific GPU.
It is relevant for data allocation (e.g. via
[`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))
but also compilation
([jit](https://r-xla.github.io/anvl/dev/reference/jit.md)).

## Usage

``` r
nv_device(x)
```

## Arguments

- x:

  (`character(1)` \| device object)  
  Identifier for the device (e.g. `"cpu"`, `"cuda"`, `"cuda:<n>"`), or
  an existing device object of the active backend (returned as-is).

## Value

A backend-specific device object (e.g. `PJRTDevice` for `"pjrt"`,
[`quickr_device`](https://r-xla.github.io/anvl/dev/reference/quickr_device.md)
for `"quickr"`).

## See also

[`backend()`](https://r-xla.github.io/anvl/dev/reference/backend.md),
[`AnvlBackend()`](https://r-xla.github.io/anvl/dev/reference/AnvlBackend.md),
[`active_backend()`](https://r-xla.github.io/anvl/dev/reference/active_backend.md).

## Examples

``` r
# Create CPU device for the active backend
nv_device("cpu")
#> <CpuDevice(id=0)>
# Create CPU device for the quickr backend:
with_backend("quickr", nv_device("cpu"))
#> QuickrDevice(cpu) 
# Pass through an existing device:
dev <- nv_device("cpu")
identical(nv_device(dev), dev)
#> [1] TRUE
```
