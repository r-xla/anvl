# Temporarily Set the Default Device

Sets the `anvl.default_device` option, which
[`default_device()`](https://r-xla.github.io/anvl/dev/reference/default_device.md)
returns in place of the first CPU device: `local_default_device()` for
the calling scope, `with_default_device()` for one expression.

This is what a call that names no device allocates on, and what a jitted
function whose graph pins no device of its own compiles for.

## Usage

``` r
local_default_device(device, envir = parent.frame())

with_default_device(device, code)
```

## Arguments

- device:

  (`character(1)` \| device object)  
  The device to make the default, e.g. `"cpu:1"`. A string is looked up
  on whichever backend asks for the default, so an identifier only one
  backend knows (`"cpu:1"` is beyond quickr's single device) makes the
  default an error on the others.

- envir:

  (`environment`)  
  Scope the option is reset at the end of. Defaults to the caller.

- code:

  (`any`)  
  Expression to evaluate with the default device set.

## Value

`local_default_device()` returns the previous option value invisibly,
`with_default_device()` the value of `code`.

## See also

[`default_device()`](https://r-xla.github.io/anvl/dev/reference/default_device.md),
[`local_backend()`](https://r-xla.github.io/anvl/dev/reference/local_backend.md)

## Examples

``` r
with_default_device("cpu:0", device(nv_array(1:3)))
#> <CpuDevice(id=0)>
```
