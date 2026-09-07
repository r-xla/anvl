# Get the default device

Returns the default device of the active backend. For the `"pjrt"`
backend, the default device is configured by the `PJRT_PLATFORM`
environment variable (defaulting to `"cpu"`). Other backends (e.g.
`"quickr"`) only support CPU.

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

A backend-specific device object.

## See also

[`nv_device()`](https://r-xla.github.io/anvl/dev/reference/nv_device.md),
[`active_backend()`](https://r-xla.github.io/anvl/dev/reference/active_backend.md)
