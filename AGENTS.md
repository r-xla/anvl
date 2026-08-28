@../claude-config/CLAUDE.md

## Package Overview

`anvl` is a code transformation framework for R, similar to JAX.
It provides JIT compilation (`jit()`) and automatic differentiation (`gradient()`, `value_and_gradient()`).

## Two-Layer API

- **`nv_*` functions** (e.g. `nv_fill()`, `nv_matmul()`) -- user-facing API in `R/api.R` and `R/api-*.R`. These handle broadcasting, type promotion, default arguments, and then delegate to `prim_*` primitives.
- **`prim_*` functions** (e.g. `prim_fill()`, `prim_mul()`) -- low-level primitives in `R/primitives.R`, exported directly under their `prim_<name>` R symbols. Calling a primitive records an operation into the computation graph during tracing (or executes it eagerly).

When adding new functionality, decide which layer it belongs to. Most new operations need both: a `prim_*` primitive with rules, and an `nv_*` wrapper with R-idiomatic semantics.

Inside `nv_*` API functions, pass plain R literals (e.g. `0`, `1`, `NaN`) directly to primitives instead of wrapping them in `nv_scalar()` / `nv_scalar_like()`. The literal takes the dtype of the operands it meets, so write it in the *category* the operand is in -- `0` for a float array, `0L` for an integer one. Shape is a separate matter: primitives do not broadcast, so a literal only works in a slot that takes a scalar (a padding value, a clamp bound, a reduction's `init`). For an elementwise primitive, broadcast first with `nv_broadcast_scalars()`.

## Terminology

- **Axis, axis size, shape.** An *axis* is an index that identifies a direction of an array; the *size* of that axis (its *axis size*) is the extent along it; the *shape* is the vector of all axis sizes. For a `20x5x3` array the axes are `1`, `2`, `3` and the shape is `c(20, 5, 3)`, so the size of axis `1` is `20`. Name identifiers accordingly: use `axis`/`axes` when the value is an index (or vector of indices) and `shape` for the vector of sizes; for a single size use an *axis size* name (e.g. `axis_size`, `n`). Helpers reflect this: `naxes(x)` is the number of axes (the rank), so `seq_len(naxes(x))` is the axis indices, and `shape(x)` returns the axis sizes.
- **Don't use "dim"/"dimension" for anvl concepts.** We don't speak of an array's "dimensions" or name size-valued identifiers `dim`/`dims` — say *axis size* (a single size) or *shape* (the vector) instead. `dim`/`dimensions` is reserved for foreign call boundaries only (next point).
- Speak of the **size of an axis**, never the "length of an axis" (reserve "length" for vectors and 1-D arrays).
- Keep the foreign spelling at call boundaries: stablehlo, torch, and base R speak of "dimensions", so calls into them keep those argument names (e.g. `hlo_reduce(dimensions = axes - 1L)`, `array(dim = ...)`) with the anvl-side axis variable on the right.
- **Arrays, not tensors.** In anvl-facing docs, messages, and identifiers, say *array* rather than *tensor*. The primary array argument of `nv_*` / `prim_*` functions is called `x`.

## Supported dtypes

- there is currently no support for complex numbers.

## R Values Have No Data Type

An R value entering a program -- a length-1 vector or an `array()`, written in
the body of a traced function or passed as an argument to a jitted one -- is
*not* converted at the boundary. It is carried as an `RDataArray` (an
`AbstractArray` with no dtype, boxed in a `GraphRData` node) and built into the
program at the dtype its use site needs, from the R data itself. That is what
makes `x_f64 / sqrt(2)` exact. See `vignette("type-promotion")`.

- **Canonicalize once, with a rule.** Call `as_anvl_arrays()` over the whole
  argument set at the top of an `nv_*` function: it aligns devices and backends,
  applies the `.promote` rule, and leaves every argument something `dtype()` and
  `device()` answer for. It always converts, and means the same thing eagerly
  and under `jit()`. *Without* a rule an R value converts at its default, so a
  function whose result dtype depends on its arguments must say so with one --
  `promote_common()`, `promote_like(arg)`, `promote_dtype(dtype)` or
  `promote_yield()`, each restrictable with `only =` and combinable with
  `promote_grouped()` -- rather than canonicalize first and `nv_convert()`
  afterwards. A rule *realizes* an input at the target (`realize_at()`), which
  is what keeps the R values exact. The two rules that name a target
  (`promote_like()`, `promote_dtype()`) refuse an input it cannot hold; say
  `force = TRUE` only where the narrowing is the function's contract.
- **Never call `dtype()` on an argument that may be a bare R value** -- there is
  nothing to report yet, so it errors. Use `peek_dtype()` to ask what it *would*
  commit to (a category test, a `nan_rm` branch), and commit it only where that
  dtype becomes the operation's own. `shape()` and `naxes()` answer as usual.
- **A primitive** promotes nothing unless it says so: `new_primitive()`'s
  `promote` defaults to `NULL`. A primitive whose arrayish arguments must agree
  declares `promote = promote_yield()` (restricted with `only =` where some
  operands are meant to differ), which is what makes `prim_mul(x_f64, 2)` work.
  The rule is applied *before the body runs*, by the wrapper `new_primitive()`
  puts around it, so the body may read a data type or trace a sub-graph without
  meeting an unresolved operand.
  An R value then takes the dtype the other operands have, but only **within its
  own category**: a double becomes a float, an integer an integer, a logical a
  `bool`. Crossing a category is promotion and belongs to the `nv_*` layer, so
  `prim_add(x_f64, 1L)` is an error where `nv_add(x_f64, 1L)` is `f64`. A trace
  output commits whatever is left, at `default_dtype_r()`.
- **Built, not converted.** A value is built directly only at a dtype that holds
  it faithfully; any other target is built at its natural dtype (`f64` / `i32` /
  `bool`) and converted by the program, because R's coercion and XLA's `convert`
  disagree on overflow and `NaN`.
- **To hold a value open** until the dtype is known, do not canonicalize it:
  `nv_convert()` and the primitives take it as it is.
- **As a jitted function's argument** the value is unknown at trace time (the
  cache keys it by R type and shape only, so the program must not depend on it).
  Its input aval is an `RDataInput` carrying the dtype the trace decided it is
  uploaded at; `graph_input_dtypes()` reads those off for pjrt's dispatcher and
  the quickr wrapper.

## Primitive System

Primitives are `JitPrimitive` callables constructed by `new_primitive()` (defined in `R/primitive.R`). The returned object is both callable (it wraps `fn` with `jit()`) and carries an `AnvlPrimitive` metadata object via `attr(., "primitive")`. Primitives are stored as `prim_<name>` variables. `new_primitive()` lexically binds `self` (the `AnvlPrimitive`) into the body's enclosing environment, so inside a primitive body you write `graph_desc_add(self, ...)` — never the primitive name as a string. Interpretation rules are accessed via `prim_<name>[["<rule_type>"]]`:

- **`stablehlo`** -- JIT lowering rules in `R/rules-stablehlo.R`. These convert traced operations into StableHLO IR. Since stablehlo uses 0-based indexing, convert indices by subtracting 1.
- **`reverse`** -- Autodiff rules in `R/rules-reverse.R`, built with `rule_reverse()`.
- **`quickr`** -- R-native lowering rules in `R/rules-quickr.R` for the quickr backend.

## Broadcasting

Anvl's elementwise binary operators (`+`, `-`, `*`, `/`, `nv_add`, `nv_mul`, …) only **auto-broadcast scalars** — i.e. operands with `shape = integer()`. They do **not** do general numpy-style broadcasting; mixing two non-scalar arrays of different (but broadcastable) shapes raises `nv_broadcast_scalars()` errors like *"All non-scalar arrays must have the same shape, ... Use `nv_broadcast_arrays()` for general broadcasting."*

When two non-scalar arrays need to be combined and only differ by size-1 axes (e.g. `[2, 3] * [1, 3]`), explicitly broadcast first via `nv_broadcast_arrays(a, b)` (or `nv_broadcast_to(x, target_shape)` / `prim_broadcast_in_axes()` for a one-sided broadcast).

## Graph Tracing

When a function is JIT-compiled, anvl traces it by executing with `GraphBox` objects instead of real data. Operations record themselves into an `AnvlGraph` (see `R/graph.R`). The graph is then lowered to StableHLO IR or quickr code for compilation.

Key types: `GraphValue` (traced variable), `GraphLiteral` (embedded constant), `AbstractArray` (shape + dtype metadata), `AnvlGraph`.

## NSE and Tracing

`force()` is only needed in higher-order primitives that trace R functions internally (e.g. `prim_sort` traces a comparator, `prim_scatter` traces an update computation). In those cases, force all arrayish inputs first so they aren't accidentally captured as unevaluated promises in the sub-graph descriptor — R's lazy evaluation otherwise causes hard-to-debug errors. Plain primitives that don't open a sub-descriptor don't need `force()`.

## Testing

Each rule of each primitive should be tested. Tests are organized as:

- `tests/testthat/test-primitives-stablehlo.R` -- sources `inst/extra-tests/test-primitives-stablehlo-torch.R`
- `tests/testthat/test-primitives-reverse.R` -- sources `inst/extra-tests/test-primitives-reverse-torch.R`

Prefer testing by comparing with the corresponding torch function. If the test is trivial or the functionality is not covered by torch, test manually instead. Write one or the other, not both.

Tests that use the quickr backend must call `skip_if_no_quickr()` at the top of the test body.
This helper skips when quickr is not installed, and also when the `ANVL_SKIP_QUICKR` environment variable is set (quickr tests can be slow and are often skipped locally).
To test a different backend, use `local_backend()` (not `withr::local_options()` directly).

## Documentation

When writing roxygen2 documentation for primitives or API functions:

- Do not mention "1-based" indexing. Since this is an R package, 1-based indexing is the default.
- Use `@templateVar primitive_id <name>` with `@template section_rules` to auto-generate the "Implemented Rules" section.
- Use `@rdname` or `@inheritParams` to share documentation between `prim_*` and `nv_*` variants.
- Where a `man-roxygen/` template is too generic for a specific primitive (e.g. the input has specific dtype constraints), write the `@param` inline instead.
