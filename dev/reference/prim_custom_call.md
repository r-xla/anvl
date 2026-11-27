# Primitive FFI Call

**\[experimental\]**

Invokes an XLA FFI handler registered with
[`pjrt::pjrt_register_custom_call()`](https://r-xla.github.io/pjrt/reference/pjrt_register_custom_call.html).
XLA documents custom calls, and the FFI API a handler is written
against, in its [custom call
guide](https://openxla.org/xla/custom_call).

This is the escape hatch for operations StableHLO cannot express, or
cannot express efficiently. Unlike every other primitive, its output
types cannot be derived from its inputs – the handler decides them – so
they are declared up front.

Custom calls exist only on the `"pjrt"` backend, which is the one that
compiles StableHLO and runs the handler. There is no `quickr` rule and
there cannot be one, so a function containing a custom call fails to
compile under
[`local_backend("quickr")`](https://r-xla.github.io/anvl/dev/reference/local_backend.md).

Within pjrt the call is platform-agnostic: it names a target, and XLA
invokes whichever handler was registered for the platform the program is
compiled for. Registering a `cuda` handler alongside the `cpu` one is
therefore all it takes to run the same call on a GPU.

Use
[`nv_custom_call()`](https://r-xla.github.io/anvl/dev/reference/nv_custom_call.md)
unless you need the exact primitive signature.

## Usage

``` r
prim_custom_call(
  operands,
  target_name,
  output_types = NULL,
  attrs = list(),
  has_side_effect = is.null(output_types),
  operand_layouts = NULL,
  result_layouts = NULL,
  aliases = NULL,
  device = NULL
)
```

## Arguments

- operands:

  (`list` of
  [`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The values passed to the handler, in the order it binds them.

- target_name:

  (`character(1)`)  
  The name the handler was registered under.

- output_types:

  ([`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md)
  \| `list` of them \| `NULL`)  
  Data type and shape of each result. The handler, not the inputs,
  decides these, so anvl cannot infer them. `NULL` declares a call that
  only runs for its side effect; `operands` are then returned unchanged,
  so at least one is required.

- attrs:

  (`list`)  
  Named attributes the handler reads via `.Attr<T>("name")` or
  `.Attrs<Dictionary>()`. A string is converted for you, and so is a
  number or a logical, at a data type fixed by its R type, whatever
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  says: a double is `f64`, an integer `i64` and a logical `bool`, so the
  handler reads them as `double`, `int64_t` and `bool`. A vector of
  several becomes an array attribute of that data type. Pass a
  `stablehlo` attribute object such as
  [`stablehlo::ScalarAttr()`](https://r-xla.github.io/stablehlo/reference/ScalarAttr.html)
  for another data type.

- has_side_effect:

  (`logical(1)`)  
  Whether XLA must keep the call even when its results are unused.
  Defaults to `TRUE` exactly when `output_types` is `NULL`. Under
  [`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md),
  a side-effect call in the body of an
  [`nv_scan()`](https://r-xla.github.io/anvl/dev/reference/nv_scan.md)
  may still be dropped.

- operand_layouts, result_layouts:

  (`list` \| `NULL`)  
  The memory layout of each operand and result: its axes in
  minor-to-major order, the axis whose index varies fastest in memory
  first. `NULL` (the default) is R's own column-major order,
  `seq_len(n)` for an `n`-axis array, which is how a LAPACK routine
  wants a matrix. A handler written for row-major (C order) buffers
  wants `rev(seq_len(n))` instead. Keep the default for a call that may
  see an array with an axis of size zero: XLA's CPU compiler crashes
  compiling any other layout for one.

  A handler whose platforms disagree – a LAPACK CPU kernel wanting
  column-major next to a row-major CUDA one – can instead pass a list of
  such lists named by platform, e.g.
  `list(cpu = list(1:2), cuda = list(2:1))`. The entry is picked when
  the program is lowered, which is the first point at which the target
  platform is known.

- aliases:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  One entry per result, giving the index of the operand whose buffer
  that result is written into, or `NA` for no aliasing. XLA then hands
  the handler the same pointer for both, so an in-place kernel avoids a
  copy.

- device:

  (`NULL` \| `character(1)` \|
  [device](https://r-xla.github.io/anvl/dev/reference/nv_device.md))  
  Where to run. Only needed when there are no operands to take it from.

## Value

`list` of
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
One element per result, or the operands unchanged for a side-effect only
call.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`hlo_custom_call()`](https://r-xla.github.io/stablehlo/reference/hlo_custom_call.html).

## See also

[`nv_custom_call()`](https://r-xla.github.io/anvl/dev/reference/nv_custom_call.md)
