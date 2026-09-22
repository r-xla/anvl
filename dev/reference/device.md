# Get the device of an array

Returns the device on which an array is allocated.

## Usage

``` r
device(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  An array-like object.

- ...:

  Additional arguments passed to methods (unused).

## Value

(device object)  
Backend-dependent device object. One of:

- [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html)

- [`quickr_device`](https://r-xla.github.io/anvl/dev/reference/quickr_device.md)

## Details

This is implemented via the generic
[`tengen::device()`](https://r-xla.github.io/tengen/reference/device.html).

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
device(x)
#> <CpuDevice(id=0)>
```
