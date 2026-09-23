# Get the default device

Returns the default device of the active backend: the device the
`anvl.default_device` option names (see
[`local_default_device()`](https://r-xla.github.io/anvl/dev/reference/local_default_device.md)),
else the one the `ANVL_DEFAULT_DEVICE` environment variable names (e.g.
`ANVL_DEFAULT_DEVICE=cuda`, read once when anvl is loaded), else the
first CPU device.

## Usage

``` r
default_device(backend = NULL)
```

## Arguments

- backend:

  (`NULL` \| `character(1)`)  
  Backend. Defaults to
  [`active_backend()`](https://r-xla.github.io/anvl/dev/reference/active_backend.md)
  when `NULL`.

## Value

(device object)  
Backend-specific.

## See also

[`nv_device()`](https://r-xla.github.io/anvl/dev/reference/nv_device.md),
[`active_backend()`](https://r-xla.github.io/anvl/dev/reference/active_backend.md),
[`local_default_device()`](https://r-xla.github.io/anvl/dev/reference/local_default_device.md)
