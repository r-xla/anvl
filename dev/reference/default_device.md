# Get the default device

Returns a device object for the default backend and platform. For the
`"pjrt"` backend, the platform is determined by the `PJRT_PLATFORM`
environment variable (defaulting to `"cpu"`). Other backends (e.g.
`"quickr"`) only support CPU. The backend defaults to
[`default_backend()`](https://r-xla.github.io/anvl/dev/reference/default_backend.md).

## Usage

``` r
default_device(backend = NULL)
```

## Arguments

- backend:

  (`NULL` \| `character(1)`)  
  Backend. Defaults to
  [`default_backend()`](https://r-xla.github.io/anvl/dev/reference/default_backend.md)
  when `NULL`.

## Value

A backend-specific device object.

## See also

[`nv_device()`](https://r-xla.github.io/anvl/dev/reference/nv_device.md),
[`default_backend()`](https://r-xla.github.io/anvl/dev/reference/default_backend.md)
