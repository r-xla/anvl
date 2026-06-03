# Create a backend

Create a backend

## Usage

``` r
AnvlBackend(
  new_data,
  dtype,
  shape,
  ambiguous,
  as_array,
  as_raw,
  platform,
  device,
  new_device,
  print_data,
  jit,
  await_data
)
```

## Arguments

- new_data:

  (`function`)  
  Constructs an AnvlArray from R data. This should be a
  [`structure()`](https://rdrr.io/r/base/structure.html) with at least a
  `$data` field that contains the actual underlying data (`PJRTBuffer`
  for `"xla"` backend, [`array()`](https://rdrr.io/r/base/array.html)
  for `"quickr"` backend).

- dtype:

  (`function`)  
  Extracts the dtype from an AnvlArray.

- shape:

  (`function`)  
  Extracts the shape from an AnvlArray.

- ambiguous:

  (`function`)  
  Extracts the ambiguous flag from an AnvlArray.

- as_array:

  (`function`)  
  Converts an AnvlArray to an R array.

- as_raw:

  (`function`)  
  Converts an AnvlArray to raw bytes.

- platform:

  (`function`)  
  Returns the platform name (e.g. `"cpu"`).

- device:

  (`function`)  
  Returns the device object for an AnvlArray.

- new_device:

  (`function`)  
  Constructs a backend-specific device object from a device type string
  (e.g. `"cpu"`). Called by
  [`nv_device()`](https://r-xla.github.io/anvl/reference/nv_device.md).

- print_data:

  (`function`)  
  Prints the array data with a footer.

- jit:

  (`function`)  
  Creates a JIT-compiled function implementation.

- await_data:

  (`function`)  
  Blocks until the array's underlying data is ready. Called by
  [`await()`](https://r-xla.github.io/anvl/reference/await.md) for
  `AnvlArray`s; a no-op for backends without async execution.

## Value

An `AnvlBackend` object.
