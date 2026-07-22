# NA

@../claude-config/CLAUDE.md

## Package Overview

`anvl` is a code transformation framework for R, similar to JAX. It
provides JIT compilation
([`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)) and
automatic differentiation
([`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md),
[`value_and_gradient()`](https://r-xla.github.io/anvl/dev/reference/value_and_gradient.md)).

## Two-Layer API

- **`nv_*` functions**
  (e.g. [`nv_fill()`](https://r-xla.github.io/anvl/dev/reference/nv_fill.md),
  [`nv_matmul()`](https://r-xla.github.io/anvl/dev/reference/nv_matmul.md))
  – user-facing API in `R/api.R` and `R/api-*.R`. These handle
  broadcasting, type promotion, default arguments, and then delegate to
  `prim_*` primitives.
- **`prim_*` functions**
  (e.g. [`prim_fill()`](https://r-xla.github.io/anvl/dev/reference/prim_fill.md),
  [`prim_mul()`](https://r-xla.github.io/anvl/dev/reference/prim_mul.md))
  – low-level primitives in `R/primitives.R`, exported directly under
  their `prim_<name>` R symbols. Calling a primitive records an
  operation into the computation graph during tracing (or executes it
  eagerly).

When adding new functionality, decide which layer it belongs to. Most
new operations need both: a `prim_*` primitive with rules, and an `nv_*`
wrapper with R-idiomatic semantics.

Inside `nv_*` API functions, pass plain R literals (e.g. `0`, `1`,
`NaN`) directly to primitives instead of wrapping them in
[`nv_scalar()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
/
[`nv_scalar_like()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md).

## Supported dtypes

- there is currently no support for complex numbers.

## Primitive System

Primitives are `JitPrimitive` callables constructed by
[`new_primitive()`](https://r-xla.github.io/anvl/dev/reference/new_primitive.md)
(defined in `R/primitive.R`). The returned object is both callable (it
wraps `fn` with
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)) and
carries an `AnvlPrimitive` metadata object via `attr(., "primitive")`.
Primitives are stored as `prim_<name>` variables.
[`new_primitive()`](https://r-xla.github.io/anvl/dev/reference/new_primitive.md)
lexically binds `self` (the `AnvlPrimitive`) into the body’s enclosing
environment, so inside a primitive body you write
`graph_desc_add(self, ...)` — never the primitive name as a string.
Interpretation rules are accessed via `prim_<name>[["<rule_type>"]]`:

- **`stablehlo`** – JIT lowering rules in `R/rules-stablehlo.R`. These
  convert traced operations into StableHLO IR. Since stablehlo uses
  0-based indexing, convert indices by subtracting 1.
- **`reverse`** – Autodiff rules in `R/rules-reverse.R`, built with
  [`rule_reverse()`](https://r-xla.github.io/anvl/dev/reference/rule_reverse.md).
- **`quickr`** – R-native lowering rules in `R/rules-quickr.R` for the
  quickr backend.

## Broadcasting

Anvl’s elementwise binary operators (`+`, `-`, `*`, `/`, `nv_add`,
`nv_mul`, …) only **auto-broadcast scalars** — i.e. operands with
`shape = integer()`. They do **not** do general numpy-style
broadcasting; mixing two non-scalar arrays of different (but
broadcastable) shapes raises
[`nv_broadcast_scalars()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_scalars.md)
errors like *“All non-scalar arrays must have the same shape, … Use
[`nv_broadcast_arrays()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_arrays.md)
for general broadcasting.”*

When two non-scalar arrays need to be combined and only differ by size-1
dimensions (e.g. `[2, 3] * [1, 3]`), explicitly broadcast first via
`nv_broadcast_arrays(a, b)` (or `nv_broadcast_to(x, target_shape)` /
[`prim_broadcast_in_dim()`](https://r-xla.github.io/anvl/dev/reference/prim_broadcast_in_dim.md)
for a one-sided broadcast).

## Graph Tracing

When a function is JIT-compiled, anvl traces it by executing with
`GraphBox` objects instead of real data. Operations record themselves
into an `AnvlGraph` (see `R/graph.R`). The graph is then lowered to
StableHLO IR or quickr code for compilation.

Key types: `GraphValue` (traced variable), `GraphLiteral` (embedded
constant), `AbstractArray` (shape + dtype metadata), `AnvlGraph`.

## NSE and Tracing

[`force()`](https://rdrr.io/r/base/force.html) is only needed in
higher-order primitives that trace R functions internally
(e.g. `prim_sort` traces a comparator, `prim_scatter` traces an update
computation). In those cases, force all arrayish inputs first so they
aren’t accidentally captured as unevaluated promises in the sub-graph
descriptor — R’s lazy evaluation otherwise causes hard-to-debug errors.
Plain primitives that don’t open a sub-descriptor don’t need
[`force()`](https://rdrr.io/r/base/force.html).

## Testing

Each rule of each primitive should be tested. Tests are organized as:

- `tests/testthat/test-primitives-stablehlo.R` – sources
  `inst/extra-tests/test-primitives-stablehlo-torch.R`
- `tests/testthat/test-primitives-reverse.R` – sources
  `inst/extra-tests/test-primitives-reverse-torch.R`

Prefer testing by comparing with the corresponding torch function. If
the test is trivial or the functionality is not covered by torch, test
manually instead. Write one or the other, not both.

Tests that use the quickr backend must call `skip_if_no_quickr()` at the
top of the test body. This helper skips when quickr is not installed,
and also when the `ANVL_SKIP_QUICKR` environment variable is set (quickr
tests can be slow and are often skipped locally). To test a different
backend, use
[`local_backend()`](https://r-xla.github.io/anvl/dev/reference/local_backend.md)
(not
[`withr::local_options()`](https://withr.r-lib.org/reference/with_options.html)
directly).

## Documentation

When writing roxygen2 documentation for primitives or API functions:

- Do not mention “1-based” indexing. Since this is an R package, 1-based
  indexing is the default.
- Use `@templateVar primitive_id <name>` with `@template section_rules`
  to auto-generate the “Implemented Rules” section.
- Use `@rdname` or `@inheritParams` to share documentation between
  `prim_*` and `nv_*` variants.
- Where a `man-roxygen/` template is too generic for a specific
  primitive (e.g. the input has specific dtype constraints), write the
  `@param` inline instead.
