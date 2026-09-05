# Configurable default dtypes, per backend

- **Date:** 2026-09-04
- **Status:** Superseded by [One backend at a time, one default dtype config](2026-09-04-ambient-backend-default-dtypes-design.md); its per-backend options and the argument-based backend inference it relied on are replaced there. The pjrt `context` key and most of `R/default-dtypes.R` carry over.
- **Packages touched:** anvl, and pjrt (a `context` key component on its
  dispatcher)
- **Builds on:** [RData: R values keep their dtype open during
  tracing](2026-08-20-rdata-type-design.md) and [Primitives resolve R data with
  promotion rules](2026-08-27-primitive-rdata-design.md). This proposal does
  not touch the yielding rule; it only makes the *fallback* those designs call
  "the default" a per-backend setting.

## Problem

An R value that meets nothing typed commits to a default dtype: `f32` for a
double, `i32` for an integer, `bool` for a logical. That default is hardcoded in
one place, `default_dtype_r()`, and is the same for every backend. Users cannot
change it, and the backends cannot disagree about it -- even though they
already do, in an inconsistent way:

- **quickr has no single precision.** Its `new_data` hardcodes `f64` for an R
  double (`R/backend-quickr.R:160`), so `nv_array(1.5)` on quickr is `f64`. But
  `default_dtype_r()` still says `f32`, so a literal that commits *inside a
  trace* on quickr is labelled `f32`: `jit(\() 1.5)()` on quickr yields an `f32`
  array whose storage is a double. The same value has two defaults depending
  on whether it enters eagerly or in a trace.
- **pjrt's default is not anvl's.** anvl's pjrt `new_data` passes `dtype = NULL`
  through to `pjrt_buffer()`, which applies *pjrt's* default (`"f32"`,
  `../pjrt/R/buffer.R:53`). It happens to agree with `default_dtype_r()`, but
  nothing ties the two together.
- **Several `nv_*` functions bake the default into their signature**:
  `nv_seq(dtype = NULL)` falls back to `"i32"` / `"f32"` inline,
  `nv_eye(dtype = "f32")`, `nv_rnorm(dtype = "f32")`,
  `nv_runif(dtype = "f32")`, `nv_rbinom(dtype = "i32")`,
  `nv_sample_int(dtype = "i32")`.

Users have a legitimate reason to want a different default: scientific code
on CPU that wants `f64` throughout without annotating every literal, or index
arithmetic that wants `i64`. JAX has `jax_enable_x64` and torch has
`torch_set_default_dtype()` for exactly this. The setting is naturally *per
backend*: quickr's natural float is `f64`, pjrt's is `f32`, and a user who runs
both in one session wants each to keep its own.

## Design

The default an R double or R integer commits to becomes a property of the
backend -- a registered built-in default that the user can override with an
option -- and every place that used to reach for `default_dtype_r()` asks the
backend the current computation belongs to. Nothing else about how R values
take a dtype changes.

### What is configurable

Two settings per backend, `float` and `int`. `bool` is not configurable: it is
the only boolean dtype.

- `float` is what an R double commits to. Only `f32` and `f64` are accepted
  for now. A double would build directly at `f16` / `bf16` too, but neither is
  in the dtype vocabulary pjrt's dispatcher keys and wraps outputs with
  (`AnvlDtype` in `../pjrt/src/dispatch_key.h`), and `gradient()` supports only
  `f32` / `f64`. Widening the set is a separate change.
- `int` is what an R integer commits to. Only `i32` and `i64` are accepted --
  the signed integer dtypes an R integer builds at directly. A narrower default
  would make the *upload* of an R argument go through R's coercion, which
  wraps where the program's `convert` clamps (see
  `rdata_builds_directly()`); an unsigned default cannot hold a negative R
  integer at all.

The settings decide only what a value becomes when *nothing else does*. The
yielding rule is untouched: `x_f32 + 1.5` is `f32` under a `f64` default, and
`x_i8 * 2L` is `i8` under an `i64` default. What changes is the fallback: `nv_array(1.5)`,
`jit(\() 1.5)()`, the all-literal branch of `common_dtype_of()`, and the
category-crossing case of `promote_dt_rdata()` (`x_i32 + 1.5` is a float *at
the default*, so `f64` under a `f64` default).

### Where the values live

`AnvlBackend()` gains a field:

```r
AnvlBackend(
  ...,
  default_dtypes = list(float = "f32", int = "i32")
)
```

This is the backend's *built-in* default. pjrt registers `f32` / `i32`; quickr
registers `f64` / `i32`, and the hardcoded `f64` in its `new_data` goes away
with it. The plain backend registers nothing: it exists only to hold constants
captured during a trace, and a constant created in a trace takes the default of
the backend being traced *for* (next section), never a default of its own.

The user override is an option per backend and category, read on every use:

```r
options(anvl.default_float.pjrt = "f64")
options(anvl.default_int.quickr = "i64")
```

Flat options rather than one nested list so that a single setting can be
written in an `.Rprofile` without knowing the others, and so that
`getOption()` answers directly. A set option is validated when read
(`as_dtype()`, then the category and width rules above); an invalid one is an
error at the first use, naming the option.

The resolution is one internal function:

```r
# The dtype an R value of this storage type commits to when nothing in the
# program tells it what it is. The single place that decision is made;
# `defaults` is the pair the current computation is pinned to.
default_dtype_r <- function(r_type, defaults = current_default_dtypes()) {
  switch(
    r_type,
    double = defaults$float,
    integer = defaults$int,
    logical = as_dtype("bool"),
    cli_abort("No default type for R type {.val {r_type}}")
  )
}
```

where `defaults` is what `default_dtypes(backend)` returns: for each category,
`getOption("anvl.default_<cat>.<backend>")` layered over
`globals$backends[[backend]]$default_dtypes[[cat]]`. Eager construction with an
explicit backend passes `default_dtypes(backend)`; everything inside a trace
gets the pair the trace was pinned to (below). All of it lives in
`R/default-dtypes.R`, with `tests/testthat/test-default-dtypes.R` beside it.

### The user-facing surface

Three exported functions, parallel to `default_backend()` / `local_backend()` /
`with_backend()`:

```r
default_dtypes(backend = NULL)
#> list(float = <DataType>, int = <DataType>)   for `backend %||% default_backend()`

local_default_dtypes(dtypes, backend = NULL, envir = parent.frame())
with_default_dtypes(dtypes, code, backend = NULL)
```

`dtypes` is a named character vector or list with elements `float` and/or
`int` (`c(float = "f64")`), so that `code` can be passed positionally as in
`with_backend(backend, code)`. The setters set only the categories named, for
the given backend (default: the default backend), and validate eagerly so a
typo fails at the call rather than at the first array. They go under *Backend*
in `_pkgdown.yml`, and the option families are documented on
`default_dtypes()`.

### The trace carries the defaults it was keyed on

Today nothing during a trace knows which backend it is being traced for: the
trace runs inside `compile_pjrt()` / `compile_quickr()`, but the descriptor does
not record it, and `nv_array()` inside a trace goes to the plain backend. The
default must be the *compiling* backend's -- `jit(f, backend = "quickr")`
under `default_backend() == "pjrt"` has to commit `f`'s literals at quickr's
default -- and it must be the very pair the dispatcher built the cache key from
(next section), not a second reading of the options. So the resolved pair is
pinned on the trace:

- `GraphDescriptor()` gains a `default_dtypes` field, `list(float, int)`.
  `compile_pjrt()` and `compile_quickr()` take the pair as an argument and pass
  it to `local_descriptor()`; the compile callbacks read it off
  `info$context`, which is what the dispatcher keyed the entry on. A
  sub-descriptor (`mode = "subgraph"` / `"inline"`) inherits it from the
  descriptor it nests in.
- `current_default_dtypes()` returns the current descriptor's pair while
  tracing, and `default_dtypes(default_backend())` otherwise. A bare
  `trace_fn()` (tests, `nv_aval` users) therefore traces under the default
  backend's current defaults, and `default_dtype_r()`'s second argument becomes
  that pair rather than a backend name.

`jit(backend = "auto")` needs nothing extra: the backend is picked at call time
and the trace happens inside that backend's compile callback.

### The default is computed, not stored

`RDataArray` currently stores `default_dtype` at construction. Construction
happens where the backend is not always known (`to_abstract()` of an R value,
`avals_from_dispatch()`, tests), and a stored value would go stale when the
option changes. The field is removed; the three readers compute it:

- `peek_dtype()`: `default_dtype_r(aval$r_type)` for an `RDataArray`.
- `resolve_upload_dtype()`: the same, when the body never used the value.
- `dtype.RDataArray()` / `abort_no_dtype()`: the same, for the message.

`rdata_in_category()` stops going through the default at all -- the category of
an R type is fixed (`double` is a float, `integer` an integer, `logical` a
`bool`) whatever the default's width, so it maps `r_type` to a category
directly.

### Eager construction

`nv_array()` / `nv_scalar()` with `dtype = NULL` resolve the
dtype in anvl, with `default_dtype(data)` for the backend they are about to
build on (`backend %||% backend(device) %||% default_backend()`), *before*
calling the backend's `new_data`. `new_data` then always receives a dtype, so
pjrt's own default in `pjrt_buffer()` is never reached from anvl, and quickr's
special case disappears. pjrt itself is not changed: its default is its own
business for its own users.

The `nv_*` functions that bake in a default switch to `dtype = NULL` and
resolve it as `default_dtypes()$float` / `$int` of the backend in play:
`nv_seq()`, `nv_eye()`, `nv_rnorm()`, `nv_runif()`, `nv_rbinom()`,
`nv_sample_int()`. `nv_fill()` already calls `default_dtype(value)` and picks
up the change for free.

### Compiled programs and the cache: a `context` key in pjrt

A compiled program bakes the default in: `jit(\() 1.5)` compiled under `f32`
returns `f32` forever. pjrt's dispatcher keys a cache entry on the leaves'
kind, dtype and shape, the statics and the default device -- nothing anvl-side
-- so changing `anvl.default_float.pjrt` between two calls would serve the
stale program. anvl already refuses this for the default device: the
dispatcher's `default_device` resolver is re-read per call and its result is
part of the key. The default dtypes get the same treatment, generalized: the
dispatcher learns a **context**, an opaque piece of key material the caller
resolves per call.

**pjrt side.** `pjrt::dispatcher()` gains

```r
dispatcher(capacity, compile, ..., context = NULL)
```

- `context` (`function` | `NULL`): called with no arguments on *every*
  dispatch, before the cache is probed. It must return a `character()` (anvl
  returns `c(float = "f32", int = "i32")`); anything else is an error naming
  the argument. Its result is part of the cache key, so an entry compiled
  under one context is never served under another. Unlike `default_device`,
  which is consulted only when no array names a device, the context is
  resolved for every call: any program can contain a literal, so every entry
  depends on it.
- `info$context` carries the resolved vector to the `compile` callback, as
  `info$default_device` does for the device, so the callback compiles under
  exactly what the key was built from rather than resolving a default of its
  own.
- In C++: `CacheKey` gains `std::vector<std::string> context`; `CacheKeyHash`
  folds its length and each string, and `CacheKeyEq` compares element-wise;
  `Dispatcher` holds a `std::optional<Rcpp::Function> context_fn_` and
  `impl_dispatcher_create()` takes it as one more argument. The key is
  otherwise untouched, and a dispatcher created without `context` keys exactly
  as today.
- Tests in `../pjrt/tests/testthat/test-dispatch.R`, mirroring the
  `default_device` ones: the callback sees `info$context`; the same context is
  a hit; a changed context is a miss that leaves both entries cached
  (`dispatcher_size()` is 2); a `context` returning a non-character is
  rejected.

**anvl side.** `jit_pjrt_impl()` and `jit_quickr_impl()` pass

```r
context = default_dtypes_context(<backend>)
```

a closure that resolves everything constant per backend (option names, the
registered fallbacks) once and per call returns the current defaults as
`c(float = , int = )`. The compile callbacks hand `info$context` on to
`compile_pjrt()` / `compile_quickr()` (through `default_dtypes_from_key()`),
which pin it on the descriptor (previous section). The per-call cost is two
`getOption()` lookups and a two-string hash, of the same order as the
leaf-aval hashing the dispatcher already does.

### What is not in scope

- A per-*device* default (e.g. `bf16` on TPU, `f32` on CPU, both under pjrt).
  The option scheme extends to it (`anvl.default_float.pjrt.tpu`), but nothing
  asks for it yet.
- `f16` / `bf16` as a float default. Both need pjrt's dispatcher to key and
  wrap them (`AnvlDtype`) and `gradient()` to differentiate them
  (`R/asserts.R:117`, `R/reverse.R:152`) first.
- quickr lowers only `f64`, `i32` and `bool`. Setting `int = "i64"` on quickr is
  accepted by the option and fails at lowering with the existing "Unsupported
  dtype for quickr lowering" error; restricting the option per backend would
  duplicate knowledge the lowering already has.

## Guarantees

1. Without any option set, pjrt behaves exactly as today (`f32` / `i32`).
2. quickr commits an R double to `f64` everywhere -- eagerly and in a trace.
   The `f32`-labelled double storage that `jit(\() 1.5)()` produces today is
   gone.
3. The yielding rule is unchanged: an R value that meets a typed array of its
   own (or a higher) category takes that dtype, whatever the default.
4. A default applies to the backend it names and to no other.
5. A compiled program is never served under a default other than the one it
   was compiled with: the pair is part of the dispatcher's cache key, and the
   trace is pinned to the pair the key was built from.
6. `dtype()` on a bare R value still errors; `peek_dtype()` answers the current
   default for the current backend.
7. Every default is resolved on the anvl side. No backend runtime (pjrt,
   quickr) chooses a dtype for anvl.

## Testing plan

- **pjrt, float:** under `local_default_dtypes(c(float = "f64"))`, each of
  `nv_array(1.5)`, `nv_scalar(1.5)`, `nv_fill(0, 3)`, `nv_seq(0, 1, steps = 3)`,
  `nv_eye(2)`, `nv_rnorm(...)`, `jit(\() 1.5)()`, `jit(\(x) x)(1.5)` (the upload
  dtype via `graph_input_dtypes()`), and `peek_dtype(1.5)` is `f64`;
  `nv_array(1, dtype = "f32") + 1.5` stays `f32`; `nv_array(1L) + 1.5` is `f64`.
- **pjrt, int:** under `local_default_dtypes(c(int = "i64"))`, `nv_array(1L)`,
  `nv_seq(1, 3)`, `jit(\() 1L)()`, `jit(\(x) x)(1L)` are `i64`; `nv_array(1L,
  dtype = "i32") + 1L` stays `i32`.
- **Trace follows the compiling backend:** with `anvl.default_float.pjrt =
  "f64"` and the default backend `pjrt`, `jit(\() 1.5, backend = "quickr")()`
  is quickr's `f64` and `jit(\() 1.5, backend = "pjrt")()` is `f64`; with pjrt
  left at `f32`, the pjrt one is `f32` (`skip_if_no_quickr()`).
- **Cache:** a jitted function called under `f32`, then under `f64`, then under
  `f32` again returns `f32`, `f64`, `f32`, and the second `f32` call is a cache
  hit (no recompilation). Also for a function with an array input and no
  literal, to pin down that the context keys every entry.
- **Pinned trace:** the pair the trace commits at is `info$context`, not a
  fresh option read -- covered by the pjrt dispatcher tests above plus one
  anvl test that a literal committed under `local_default_dtypes(c(float =
  "f64"))` inside `jit()` is `f64` for both a literal-only and a mixed program.
- **Isolation:** setting pjrt's float leaves `default_dtypes("quickr")` at
  `f64` / `i32`, and vice versa.
- **quickr:** the existing "default floating dtype is f64 for quickr" test is
  joined by `jit(\() 1.5)()` being `f64` on quickr, and by the traced-literal
  case `jit(\(x) x + 1.5)(nv_array(1L))` being `f64`.
- **Validation:** `local_default_dtypes(c(float = "i32"))`, `(c(float = "f16"))`,
  `(c(int = "f32"))`, `(c(int = "i8"))`, `(c(int = "ui32"))` and `options(anvl.default_float.pjrt =
  "nope")` followed by `nv_array(1.5)` all error naming the setting.
- **Getter:** `default_dtypes()` with nothing set returns the registered
  defaults for each backend.
- **Vignette:** `vignette("type-promotion")`'s "commits to the default" section
  gains a paragraph and the `f32` / `i32` bullets are rephrased as "the
  backend's default float / integer, `f32` / `i32` for pjrt".

## Rejected alternatives

- **One global default** (`jax_enable_x64` style). It cannot express what the
  package already needs -- quickr at `f64` and pjrt at `f32` in one session --
  and a backend's natural dtype is a property of the backend, not of the user.
- **Keeping quickr's hardcoded `f64` in `new_data`.** It is the eager half of a
  default the trace does not know about; that is the inconsistency this
  proposal removes.
- **Passing the default to `jit()`.** The package's own `nv_*` wrappers are
  jitted once at build time (`@jit` / `apply_jit_registry()`), so a
  construction-time setting could never reach them. The default has to be read
  when the program is traced.
- **Storing the default on `RDataArray`.** Construction does not always know
  the backend, and a stored value cannot follow the option. Three readers
  computing it is less machinery than threading a backend into every
  constructor.
- **Changing `pjrt_buffer()`'s default.** pjrt is a lower layer with users of
  its own. anvl resolving every dtype before the call is the right boundary,
  and it makes the pjrt default irrelevant to anvl rather than coordinated with
  it.
- **Not keying the cache on the default.** Silent reuse of a program compiled
  under another default is exactly the class of bug the `default_device`
  callback exists to prevent; the same standard applies.
- **One dispatcher per default pair on the anvl side**, instead of a key
  component in pjrt. It avoids touching pjrt, but it splits `cache_size` per
  pair, hides the dependency from `dispatcher_size()`, and leaves the compile
  callback reading the options a second time instead of receiving what the key
  was built from. A key component is what the cache is for.
- **A setter with a generation counter instead of options** (cheaper hot path,
  eager validation). It cannot be set from an `.Rprofile` before anvl loads
  without a second mechanism, and it breaks the `options()` /
  `local_*()` / `with_*()` pattern every other anvl setting follows. Options
  are the recommendation; the counter is the fallback if the per-call
  `getOption()` cost ever registers.
