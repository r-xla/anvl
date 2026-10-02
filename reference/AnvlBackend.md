# Create a Backend

Create a Backend

## Usage

``` r
AnvlBackend(
  new_data,
  new_empty,
  dtype,
  shape,
  as_array,
  as_raw,
  platform,
  device,
  new_device,
  print_data,
  jit,
  await_data,
  default_dtypes
)
```

## Arguments

- new_data:

  (`function(data, dtype, shape, device, row_major = FALSE)`)  
  Constructs an AnvlArray from R data. Must return
  `structure(list(data = , backend = <name>, ...), class = "AnvlArray")`,
  where `data` holds the underlying data (a `PJRTBuffer` for the
  `"pjrt"` backend, an R [`array()`](https://rdrr.io/r/base/array.html)
  for the `"quickr"` backend) and `backend` is the name the backend is
  registered under. `row_major` gives the element order of raw byte
  payloads; a backend that does not support raw `data` should abort on
  it.

- new_empty:

  (`function(dtype, shape, device)`)  
  Constructs an AnvlArray of the given `dtype` and `shape` with
  unspecified contents. Called by
  [`nv_empty()`](https://r-xla.github.io/anvl/reference/AnvlArray.md).

- dtype:

  (`function(x)`)  
  Extracts the dtype from an AnvlArray.

- shape:

  (`function(x)`)  
  Extracts the shape from an AnvlArray.

- as_array:

  (`function(x, check)`)  
  Converts an AnvlArray to an R array. The `check` level is forwarded
  from
  [`as_array()`](https://r-xla.github.io/anvl/reference/as_array.md);
  backends may use it to abort when materialization would lose
  information (e.g. ui64 values wrapping through
  [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html)).
  See
  [`pjrt::as_array.PJRTBuffer()`](https://r-xla.github.io/pjrt/reference/as_array.PJRTBuffer.html).

- as_raw:

  (`function(x, row_major)`)  
  Converts an AnvlArray to raw bytes.

- platform:

  (`function(x)`)  
  Returns the platform name (e.g. `"cpu"`).

- device:

  (`function(x)`)  
  Returns the device object for an AnvlArray.

- new_device:

  (`function(x)`)  
  Constructs a backend-specific device object from a device identifier
  (e.g. `"cpu"` or `"cuda:1"`). Called by
  [`nv_device()`](https://r-xla.github.io/anvl/reference/nv_device.md).

- print_data:

  (`function(x, footer, ...)`)  
  Prints the array data with a footer, passing `...` (e.g. `max_rows`
  for the pjrt backend) on to the backend's printer.

- jit:

  (`function(f, static, cache_size, <options>, device = NULL)`)  
  Creates the backend's implementation of a JIT-compiled function and
  returns it as a `function`. The formals in place of `<options>` are
  the backend-specific options
  [`jit()`](https://r-xla.github.io/anvl/reference/jit.md) accepts
  through `...`.

- await_data:

  (`function(x)`)  
  Blocks until the array's underlying data is ready. Called by
  [`await()`](https://r-xla.github.io/anvl/reference/await.md) for
  `AnvlArray`s; a no-op for backends without async execution.

- default_dtypes:

  (`NULL` \| `list(float, int)`)  
  The default data types for this backend. Can be overwritten, see
  [`default_dtypes()`](https://r-xla.github.io/anvl/reference/default_dtypes.md).

## Value

(`AnvlBackend`)
