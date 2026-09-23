@../claude-config/CLAUDE.md

## Package Overview

`anvl` is a code transformation framework for R, similar to JAX.
It provides JIT compilation (`jit()`) and automatic differentiation (`gradient()`, `value_and_gradient()`).

## Commands

The generic R workflow (`devtools::test()`, `make format`, `jarl check .`, ...) is in the shared
config above. anvl-specific:

- **Tests are gated behind `ANVL_TEST=1`** -- `tests/testthat.R` only calls `test_check()` when it
  is set, so `R CMD check` in a shell without it runs *no* tests. `.Renviron` sets it (together with
  `PJRT_INSTALL=1`) for work inside the package.
- Single file: `testthat::test_active_file("tests/testthat/test-reverse.R")`, or
  `devtools::test(filter = "reverse")`.
- `ANVL_TEST_SKIP_QUICKR=1` skips the (slow) quickr tests. `setup.R` sets `PJRT_CPU_DEVICE_COUNT=2`
  so multi-device tests have something to spread over.
- `ANVL_DEFAULT_DEVICE` / `ANVL_DEFAULT_DTYPES` are package-level: anvl reads them once when it is
  loaded and falls back to them when the `anvl.default_device` / `anvl.default_dtypes` options are
  not set. The suite uses them to run on another configuration, and `setup.R` skips quickr for
  such a run:
  - `ANVL_DEFAULT_DEVICE=cuda` runs the suite on the CUDA plugin (`is_cpu()` / `is_cuda()` in
    `helper.R` branch on it). `ANVL_DEFAULT_DEVICE=cpu:1` runs it on the second CPU device, so
    anything allocating on the first CPU device where it should have followed the trace or its
    operands lands on a device of its own, which jit's autodetect reports; the `default-device`
    workflow runs it on the `full-test` PR label. A test that asserts the unset default calls
    `local_unset_default_device()` (`helper.R`) to clear both.
  - `ANVL_DEFAULT_DTYPES="float=f64,int=i64"` runs the suite at another pair of default data types;
    the `default-dtypes` workflow runs it so that anything hardcoding `f32` / `i32` where it should
    read `default_dtypes()` fails in CI. A test that asserts the *registered* pair calls
    `local_registered_default_dtypes()` (`helper.R`) to clear the override.
- anvl tracks the **dev** versions of its r-xla dependencies:
  `pak::pkg_install(c("r-xla/xlamisc", "r-xla/pjrt", "r-xla/stablehlo", "r-xla/tengen"))`.

## Two-Layer API

- **`nv_*` functions** (e.g. `nv_fill()`, `nv_matmul()`) -- user-facing API in `R/api.R` and `R/api-*.R`. These handle broadcasting, type promotion, default arguments, and then delegate to `prim_*` primitives.
- **`prim_*` functions** (e.g. `prim_fill()`, `prim_mul()`) -- low-level primitives in `R/primitives.R`, exported directly under their `prim_<name>` R symbols. Calling a primitive records an operation into the computation graph during tracing (or executes it eagerly).

When adding new functionality, decide which layer it belongs to. Most new operations need both: a `prim_*` primitive with rules, and an `nv_*` wrapper with R-idiomatic semantics.

Inside `nv_*` API functions, pass plain R literals (e.g. `0`, `1`, `NaN`) directly to primitives instead of wrapping them in `nv_scalar()` / `nv_scalar_like()`. The literal takes the dtype of the operands it meets. Shape is a separate matter: primitives do not broadcast, so a literal only works in a slot that takes a scalar (a padding value, a clamp bound, a reduction's `init`). For an elementwise primitive, broadcast first with `nv_broadcast_scalars()`.

The two spellings of a whole number are not symmetric here. An R integer widens into whatever
category it meets, while a plain `1` is an R *double* and pulls an integer array into the float
category -- `x_i32 - 1` is `f32`, where `x_i32 - 1L` is `i32`. So a whole number keeps its `L`
even when it meets a float array: `nv_ifelse(mask, 0L, x)`, `prim_fill(1L, dtype = dtype(x),
...)`, `hlo_scalar(0L, dtype = dtype(x), ...)`, `U - 1L`.

Drop the `L` only where the value is genuinely a real number that happens to be whole -- a
distribution parameter, a probability bound, a threshold, a coefficient: `sd = 1`,
`lower = 0, upper = 1`, `nv_pmax(-d, 1)`, `2 / sqrt(pi)`, `base::log(2 * pi)`.

That distinction bites hardest on a literal that meets *nothing*, where it decides a data type
outright by settling on the default of its own category. `nv_rnorm(mean = 0, sd = 1)` has to stay
plain: written `0L` / `1L`, a call that names no dtype returns the sample at the default
*integer*.

## Terminology

- **Axis, axis size, shape.** An *axis* is an index that identifies a direction of an array; the *size* of that axis (its *axis size*) is the extent along it; the *shape* is the vector of all axis sizes. For a `20x5x3` array the axes are `1`, `2`, `3` and the shape is `c(20, 5, 3)`, so the size of axis `1` is `20`. Name identifiers accordingly: use `axis`/`axes` when the value is an index (or vector of indices) and `shape` for the vector of sizes; for a single size use an *axis size* name (e.g. `axis_size`, `n`). Helpers reflect this: `naxes(x)` is the number of axes (the rank), so `seq_len(naxes(x))` is the axis indices, and `shape(x)` returns the axis sizes.
- **Don't use "dim"/"dimension" for anvl concepts.** We don't speak of an array's "dimensions" or name size-valued identifiers `dim`/`dims` — say *axis size* (a single size) or *shape* (the vector) instead. `dim`/`dimensions` is reserved for foreign call boundaries only (next point).
- Speak of the **size of an axis**, never the "length of an axis" (reserve "length" for vectors and 1-D arrays).
- Keep the foreign spelling at call boundaries: stablehlo, torch, and base R speak of "dimensions", so calls into them keep those argument names (e.g. `hlo_reduce(dimensions = axes - 1L)`, `array(dim = ...)`) with the anvl-side axis variable on the right.
- **Arrays, not tensors.** In anvl-facing docs, messages, and identifiers, say *array* rather than *tensor*. The primary array argument of `nv_*` / `prim_*` functions is called `x`.
- **Materialize, take, canonicalize.** Three words for the R value -> `AnvlArray` story, one each:
  - *materialize* is the **event** -- an R value becoming an array at a data type. An R value
    *materializes at* a data type and *materializes as* an array; before that it is
    *unmaterialized*. `materialize_at()` and `materialize_rdata()` are the functions that do it.
  - *takes* / *settles on* is **which** data type it gets: a value *takes* the data type of the
    array it meets, and *settles on* the default when it meets nothing. `peek_dtype()` reports
    the data type a value *would take*.
  - *canonicalize* is the **code discipline** of calling `as_anvl_array()` / `as_anvl_arrays()`
    at the top of an `nv_*` function so it works eagerly and under `jit()`.

  Don't reach for a synonym (*commit*, *realize*, *standardize*) for any of the three.

## Supported dtypes

The data types and the words the docs use for groups of them are in `?dtypes`
(`R/promotion.R`) and `man-roxygen/section_dtype_words.R`: *any* / *numeric* / *integer* /
*integerish* / *signed numeric* / *float* / *boolean*. Two things to keep in mind:

- There is currently no support for complex numbers.
- We currently do not worry about any float type other than `f32` and `f64`.

## Type Promotion

An R value entering a program is not converted at the boundary -- it is built into the program at the
dtype its use site needs, which is what makes `x_f64 / sqrt(2)` exact. `vignette("type-promotion")`
is the reference for how this works and for the `.promote` rules (`promotion_common()`,
`promotion_like()`, `promotion_dtype()`, `promotion_rdata_common()`) that `nv_*` functions pass to
`as_anvl_arrays()`. Two rules that bite while writing code:

- Never call `dtype()` on an argument that may still be a bare R value -- it errors. Use
  `peek_dtype()` to ask what it *would* materialize at.
- A primitive promotes nothing unless its body says so: one whose operands must agree calls
  `apply_promotion()` on them before anything else reads them.
- A trace output that met nothing materializes at the default float / integer of the active backend,
  which `default_dtypes()` reports (`default_float()` / `default_int()` for one category) and the
  option `anvl.default_dtypes` overrides. A trace is pinned to the pair the dispatcher keyed its
  program on (`GraphDescriptor$default_dtypes`); name a category with `default_float()` /
  `default_int()`, never hardcode `"f32"` / `"i32"` as a default. `default_dtype_r()` is for the
  few places that map an R storage type chosen at run time, and `current_default_dtypes()` for
  the whole pair.
- **One backend at a time.** The backend is the option `anvl.backend` (`active_backend()`,
  `local_backend()`, `with_backend()`). Every jitted function runs on it, reading it at call time;
  nothing infers a backend from an argument, no array operation or `jit()` takes a `backend`
  argument, and an array or device of another backend is an error. This is what makes the default
  dtypes unambiguous in eager code. (A handful of helpers about the backend itself do name one:
  `install_anvl()`, `default_device()`, `local_default_dtypes()` / `with_default_dtypes()`.)

## One Backend at a Time

The backend is the option `anvl.backend` (`active_backend()`, `local_backend()`, `with_backend()`).
Every jitted function runs on it, reading it at call time; nothing infers a backend from an
argument, no array operation or `jit()` takes a `backend` argument, and an array or device of
another backend is an error. Only helpers *about* the backend name one (`install_anvl()`,
`default_device()`, `local_default_dtypes()` / `with_default_dtypes()`).

## Primitive System

Primitives are `JitPrimitive` callables constructed by `new_primitive()` (defined in `R/primitive.R`). The returned object is both callable (it wraps `fn` with `jit()`) and carries an `AnvlPrimitive` metadata object via `attr(., "primitive")`. Primitives are stored as `prim_<name>` variables, and the string passed to `new_primitive()` is that same `<name>` -- not the StableHLO op it lowers to -- so printed graphs and error messages name a function the reader can look up. `test-primitives-meta.R` enforces this. `new_primitive()` lexically binds `self` (the `AnvlPrimitive`) into the body's enclosing environment, so inside a primitive body you write `graph_desc_add(self, ...)` — never the primitive name as a string. Interpretation rules are accessed via `prim_<name>[["<rule_type>"]]`:

- **`stablehlo`** -- JIT lowering rules in `R/rules-stablehlo.R`. These convert traced operations into StableHLO IR. Since stablehlo uses 0-based indexing, convert indices by subtracting 1.
- **`reverse`** -- Autodiff rules in `R/rules-reverse.R`, built with `rule_reverse()`.
- **`quickr`** -- R-native lowering rules in `R/rules-quickr.R` for the quickr backend.

## Jit-wrapping

API functions are wrapped in `jit()` at the definition itself, with `static`
after the function so the signature reads on its own line:

```r
nv_foo <- jit(function(x, axis) {
  ...
}, static = "axis")
```

Wrap every function whose body issues **more than one operation**.

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
This helper skips when quickr is not installed, and also when the `ANVL_TEST_SKIP_QUICKR` environment variable is set (quickr tests can be slow and are often skipped locally).
To test a different backend, use `local_backend()` (not `withr::local_options()` directly).

## Documentation

When writing roxygen2 documentation for primitives or API functions:

- Do not mention "1-based" indexing. Since this is an R package, 1-based indexing is the default.
- Use `@templateVar primitive_id <name>` with `@template section_rules` to auto-generate the "Implemented Rules" section.
- Use `@rdname` or `@inheritParams` to share documentation between `prim_*` and `nv_*` variants.
- Where a `man-roxygen/` template is too generic for a specific primitive (e.g. the input has specific dtype constraints), write the `@param` inline instead.
