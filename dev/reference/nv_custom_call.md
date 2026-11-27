# Call an XLA FFI Handler

**\[experimental\]**

Calls a handler registered with
[`pjrt::pjrt_register_custom_call()`](https://r-xla.github.io/pjrt/reference/pjrt_register_custom_call.html),
the way
[`prim_custom_call()`](https://r-xla.github.io/anvl/dev/reference/prim_custom_call.md)
does, but taking the operands through `...` and unwrapping a single
result.

This is anvl's escape hatch for operations StableHLO cannot express, or
cannot express efficiently. How to write, register and call a handler is
shown in the [Custom
Calls](https://r-xla.github.io/anvl/articles/custom_calls.html) article;
XLA documents custom calls, and the FFI API a handler is written
against, in its [custom call
guide](https://openxla.org/xla/custom_call).

Custom calls are a `"pjrt"` backend feature: the handler is registered
with the PJRT runtime and invoked by the XLA-compiled program, so there
is nothing for the `quickr` backend to call. Under
[`local_backend("quickr")`](https://r-xla.github.io/anvl/dev/reference/local_backend.md)
a function containing a custom call fails to compile.

Within pjrt the call is platform-agnostic. It names a target, and XLA
invokes whichever handler
[`pjrt::pjrt_register_custom_call()`](https://r-xla.github.io/pjrt/reference/pjrt_register_custom_call.html)
was given for the platform the program is compiled for, so the same
`nv_custom_call()` runs on CPU or CUDA depending only on where its
operands live.

The call is also opaque to XLA and therefore to
[`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md):
to differentiate through a handler, wrap it in a primitive of your own
with a pullback rule – see the [Adding a
Primitive](https://r-xla.github.io/anvl/articles/extending_primitive.html)
article.

## Usage

``` r
nv_custom_call(
  target_name,
  ...,
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

- target_name:

  (`character(1)`)  
  The name the handler was registered under.

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The operands, in the order the handler binds them.

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

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md) or
`list` of
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
A single value when `output_types` is a single
[`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md)
(or when a side-effect only call has a single operand), otherwise a
list.

## See also

[`prim_custom_call()`](https://r-xla.github.io/anvl/dev/reference/prim_custom_call.md)

## Examples

``` r
# "eigh" is one of the handlers {pjrt} registers on load. It returns
# (eigenvectors, eigenvalues) and, being LAPACK, wants column-major
# buffers -- R's own order, which is the default layout.
x <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
nv_custom_call(
  "eigh",
  x,
  output_types = list(AbstractArray("f64", c(2, 2)), AbstractArray("f64", 2))
)
#> [[1]]
#> AnvlArray
#>  -0.7071  0.7071
#>   0.7071  0.7071
#> [ CPUf64{2,2} ] 
#> 
#> [[2]]
#> AnvlArray
#>  1
#>  3
#> [ CPUf64{2} ] 
#> 
```
