# Create a Backend

**\[experimental\]**

A backend defines how arrays are stored, converted to R and placed on
devices, and how
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) runs a
function. `AnvlBackend()` bundles the methods that do this, and
[`register_backend()`](https://r-xla.github.io/anvl/dev/reference/register_backend.md)
makes the result selectable by name. anvl registers two backends itself,
[`AnvlBackendPjrt()`](https://r-xla.github.io/anvl/dev/reference/AnvlBackendPjrt.md)
and
[`AnvlBackendQuickr()`](https://r-xla.github.io/anvl/dev/reference/AnvlBackendQuickr.md),
and a package can add more.

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
  Constructs an AnvlArray of the given `dtype` and `shape` (an
  [`integer()`](https://rdrr.io/r/base/integer.html) vector, which
  [`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  has already worked out) from R data. Must return
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
  [`nv_empty()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md).

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
  [`as_array()`](https://r-xla.github.io/anvl/dev/reference/as_array.md);
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
  [`nv_device()`](https://r-xla.github.io/anvl/dev/reference/nv_device.md).

- print_data:

  (`function(x, footer, ...)`)  
  Prints the array data with a footer, passing `...` (e.g. `max_rows`
  for the pjrt backend) on to the backend's printer.

- jit:

  (`function(f, static, cache_size, <options>, device = NULL)`)  
  Creates the backend's implementation of a JIT-compiled function and
  returns it as a `function`. The formals in place of `<options>` are
  the backend-specific options
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) accepts
  through `...`. The returned function is given the formals of `f`, so
  it reads its arguments by name, e.g. with
  [`match.call()`](https://rdrr.io/r/base/match.call.html), rather than
  through `...`. It is only called outside a trace: inside one,
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) calls `f`
  itself.

- await_data:

  (`function(x)`)  
  Blocks until the array's underlying data is ready. Called by
  [`await()`](https://r-xla.github.io/anvl/dev/reference/await.md) for
  `AnvlArray`s; a no-op for backends without async execution.

- default_dtypes:

  (`NULL` \| `list(float, int)`)  
  The default data types for this backend. Can be overwritten, see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).
  Only the internal backend that holds the constants of a trace has
  none; a backend that can be selected needs them.

## Value

(`AnvlBackend`)

## Details

An array of the backend is an
[`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
that `new_data` and `new_empty` construct and the other methods take
apart; see `new_data` for its structure.

A device of the backend is an object whose class inherits from
`"AnvlDevice"`, such as
`structure("cpu", class = c("MyDevice", "AnvlDevice"))`. Its class needs
a [`backend()`](https://r-xla.github.io/anvl/dev/reference/backend.md)
method that returns the name the backend is registered under, and two
devices must be comparable with `==`.

The `jit` method should trace the function into an
[`AnvlGraph`](https://r-xla.github.io/anvl/dev/reference/AnvlGraph.md)
with
[`trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md),
so that every backend runs the same graph, and then lower or evaluate
each statement of the graph with a rule of the primitive it applies: a
rule of a type registered with
[`register_rule_type()`](https://r-xla.github.io/anvl/dev/reference/register_rule_type.md),
which `statement$primitive[[<type>]]` returns.

## See also

[`register_backend()`](https://r-xla.github.io/anvl/dev/reference/register_backend.md),
[`register_rule_type()`](https://r-xla.github.io/anvl/dev/reference/register_rule_type.md)
