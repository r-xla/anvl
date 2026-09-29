# Temporarily Set the Default Device

Sets the `anvl.default_device` option, which
[`default_device()`](https://r-xla.github.io/anvl/reference/default_device.md)
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

  (`NULL` \| `character(1)` \| device object)  
  The device to make the default, e.g. `"cuda:0"`. A string is looked up
  on whichever backend asks for the default, so an identifier only one
  backend knows (quickr has no `"cuda"`) makes the default an error on
  the others. `NULL` clears the option.

- envir:

  (`environment`)  
  Scope the option is reset at the end of. Defaults to the caller.

- code:

  (`any`)  
  Expression to evaluate with the default device set.

## Value

`local_default_device()`: (named `list`)  
The previous value of the option, as `list(anvl.default_device = )`,
invisibly.

`with_default_device()`: (any)  
The value of `code`.

## See also

[`default_device()`](https://r-xla.github.io/anvl/reference/default_device.md),
[`local_backend()`](https://r-xla.github.io/anvl/reference/local_backend.md)

## Examples

``` r
getOption("anvl.default_device")
#> NULL
# the option holds the identifier; it is resolved to a device only when
# `default_device()` is called
with_default_device("cuda:0", getOption("anvl.default_device"))
#> [1] "cuda:0"
```
