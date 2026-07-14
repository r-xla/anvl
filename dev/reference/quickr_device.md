# Quickr device

Device descriptor for the quickr backend. The only supported `type` is
`"cpu"`.

Each call returns a fresh object; the quickr backend does not intern its
devices. pjrt's dispatcher canonicalizes devices with
[`identical()`](https://rdrr.io/r/base/identical.html) as a fallback to
object identity, so equal-but-distinct `QuickrDevice` objects still
collapse to one device – and quickr has a single device, so interning
would buy nothing.

## Usage

``` r
quickr_device(x = "cpu")
```

## Arguments

- x:

  (`character(1)`)  
  Device type. Currently only supports `"cpu"`.

## Value

A `QuickrDevice` object.

## See also

[`nv_device()`](https://r-xla.github.io/anvl/dev/reference/nv_device.md),
[`AnvlBackendQuickr()`](https://r-xla.github.io/anvl/dev/reference/AnvlBackendQuickr.md).
