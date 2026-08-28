@../claude-config/CLAUDE.md

## Package Overview

`anvl` is a code transformation framework for R, similar to JAX.
It provides JIT compilation (`jit()`) and automatic differentiation (`gradient()`, `value_and_gradient()`).

## Two-Layer API

- **`nv_*` functions** (e.g. `nv_fill()`, `nv_matmul()`) -- user-facing API in `R/api.R` and `R/api-*.R`. These handle broadcasting, type promotion, default arguments, and then delegate to `prim_*` primitives.
- **`prim_*` functions** (e.g. `prim_fill()`, `prim_mul()`) -- low-level primitives in `R/primitives.R`, exported directly under their `prim_<name>` R symbols. Calling a primitive records an operation into the computation graph during tracing (or executes it eagerly).

When adding new functionality, decide which layer it belongs to. Most new operations need both: a `prim_*` primitive with rules, and an `nv_*` wrapper with R-idiomatic semantics.

Inside `nv_*` API functions, pass plain R literals (e.g. `0`, `1`, `NaN`) directly to primitives instead of wrapping them in `nv_scalar()` / `nv_scalar_like()`.

## Terminology

- **Axis, axis size, shape.** An *axis* is an index that identifies a direction of an array; the *size* of that axis (its *axis size*) is the extent along it; the *shape* is the vector of all axis sizes. For a `20x5x3` array the axes are `1`, `2`, `3` and the shape is `c(20, 5, 3)`, so the size of axis `1` is `20`. Name identifiers accordingly: use `axis`/`axes` when the value is an index (or vector of indices) and `shape` for the vector of sizes; for a single size use an *axis size* name (e.g. `axis_size`, `n`). Helpers reflect this: `naxes(x)` is the number of axes (the rank), so `seq_len(naxes(x))` is the axis indices, and `shape(x)` returns the axis sizes.
- **Don't use "dim"/"dimension" for anvl concepts.** We don't speak of an array's "dimensions" or name size-valued identifiers `dim`/`dims` — say *axis size* (a single size) or *shape* (the vector) instead. `dim`/`dimensions` is reserved for foreign call boundaries only (next point).
- Speak of the **size of an axis**, never the "length of an axis" (reserve "length" for vectors and 1-D arrays).
- Keep the foreign spelling at call boundaries: stablehlo, torch, and base R speak of "dimensions", so calls into them keep those argument names (e.g. `hlo_reduce(dimensions = axes - 1L)`, `array(dim = ...)`) with the anvl-side axis variable on the right.
- **Arrays, not tensors.** In anvl-facing docs, messages, and identifiers, say *array* rather than *tensor*. The primary array argument of `nv_*` / `prim_*` functions is called `x`.

## Supported dtypes

- there is currently no support for complex numbers.

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

`integrations/` holds tests that `R CMD check` must not run -- currently the custom-call path from the "Custom Calls" article, which compiles an FFI handler against {pjrt}'s headers with `Rcpp::sourceCpp()`.
They have their own launcher (`Rscript integrations/run.R`) and run in a single CI job (Ubuntu x86, CPU).

Tests that use the quickr backend must call `skip_if_no_quickr()` at the top of the test body.
This helper skips when quickr is not installed, and also when the `ANVL_SKIP_QUICKR` environment variable is set (quickr tests can be slow and are often skipped locally).
To test a different backend, use `local_backend()` (not `withr::local_options()` directly).

## Documentation

When writing roxygen2 documentation for primitives or API functions:

- Do not mention "1-based" indexing. Since this is an R package, 1-based indexing is the default.
- Use `@templateVar primitive_id <name>` with `@template section_rules` to auto-generate the "Implemented Rules" section.
- Use `@rdname` or `@inheritParams` to share documentation between `prim_*` and `nv_*` variants.
- Where a `man-roxygen/` template is too generic for a specific primitive (e.g. the input has specific dtype constraints), write the `@param` inline instead.
