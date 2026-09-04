# One backend at a time, one default dtype config

- **Date:** 2026-09-04
- **Status:** Implemented
- **Supersedes:** [Configurable default dtypes, per
  backend](2026-09-04-default-dtypes-design.md). That design is implemented in
  the current working tree; this one replaces it. The parts it keeps are named
  in *What is kept from the current tree*.
- **Packages touched:** anvl; pjrt only through the `dispatcher(context = )`
  key component the superseded design already added.

## Problem

An R value entering a program has no dtype (see the [RData
design](2026-08-20-rdata-type-design.md)). When nothing it meets decides one,
it commits to a *default*: today `f32` for a double, `i32` for an integer. Two
things are wrong with how that default is decided.

**It is hardcoded, and the backends disagree with it.** quickr has no single
precision, so its `new_data` builds an R double as `f64`, while a literal that
commits inside a trace on quickr is labelled `f32` with double storage. pjrt's
own `pjrt_buffer()` default (`f32`) is what anvl's arrays get, by accident of
agreement. Users cannot ask for `f64` throughout without annotating every
literal.

**The backend an operation runs on is decided per call, from its arguments.**
`jit(backend = "auto")`, which every package `nv_*` function is wrapped with,
picks the backend from the array arguments, else from a device argument, else
from `default_backend()`. As long as the default dtype was one constant this
was harmless. The moment the default depends on the backend it is not, because
the default is read in two kinds of places:

- *inside a trace*, where the dispatcher has already chosen the backend and
  the trace can be pinned to that backend's defaults (sound);
- *eagerly, between dispatches*, in plain R code that has no backend to ask.
  `as_anvl_arrays(x, 1.5, .promote = promote_common())` in a non-jitted
  wrapper decides the common dtype eagerly: with `x` an `i32` quickr array and
  pjrt the default backend, the category-crossing rule takes the *ambient*
  float default (`f32`) and realizes `1.5` at `f32` on quickr, whose default is
  `f64`. Wrong dtype, no error. `peek_dtype()`-driven branches and `nv_fill()`
  with a foreign `device` have the same shape.

The superseded design tried to make this work with per-backend options and a
convention ("resolve defaults inside the trace, never before dispatch"). The
convention is auditable but not enforceable: it breaks silently the first time
someone writes a plain R helper over arrays of a non-default backend. The
per-backend options, and the `backend` argument they force onto every setter,
are also more configuration surface than the feature deserves.

## Design

Two decisions, which only work together.

### 1. The backend is ambient

There is exactly one backend in force at any time: `default_backend()`, set by
`options(anvl.backend = )`, `local_backend()` or `with_backend()`. The option
is renamed from `anvl.default_backend`: it is no longer a default that
arguments may override, it is *the* backend. Every operation runs on it.
Nothing infers a backend from an argument any more:

- `jit()` loses its `backend` argument. A jitted function resolves the backend
  **at call time**, so a function created under pjrt and called inside
  `with_backend("quickr", ...)` runs on quickr. (Package `nv_*` functions,
  wrapped at build time, need call-time resolution anyway; `jit_auto()` keeps
  its lazily created per-backend dispatchers, keyed by `default_backend()`,
  and loses only the argument scan.)
- `device_arg()` stays, with a smaller job. It used to pick the *backend* per
  device argument; now it only reads the compilation device of a function
  without array inputs (`prim_fill()`, `prim_iota()`) from a static argument,
  and that device has to belong to the ambient backend.
- The constructors lose their `backend` arguments: `nv_array()`, `nv_scalar()`,
  `nv_empty()`, `nv_array_like()`, `nv_scalar_like()`, `nv_device()`,
  `nv_read()`, `nv_unserialize()`. They build on the ambient backend. A
  `device` of another backend is an error.
- An array of another backend handed to an operation is rejected. pjrt's
  dispatcher already does this with a clear message ("expected an AnvlArray of
  backend pjrt, got quickr"), and `check_single_backend()` covers closed-over
  constants; both stay as the enforcement.

Arrays still carry their backend, and they may leave the scope they were made
in: `x <- with_backend("quickr", nv_array(1))` is fine. Using `x` is only valid
where quickr is the ambient backend; anywhere else it errors, it never computes
at a wrong dtype. That is the trade this design makes deliberately: an error
where the superseded design had a convention.

### 2. One default dtype config, with a per-backend fallback

Each backend registers the defaults an R double and an R integer commit to on
it, in `AnvlBackend(default_dtypes = list(float = , int = ))`: pjrt `f32` /
`i32`, quickr `f64` / `i32`, none for the plain backend (it only holds
constants captured while tracing for another backend). On top of that there is
**one global override pair**, the options `anvl.default_float` and
`anvl.default_int`, unset by default. The effective default is

```r
getOption("anvl.default_float") %||% globals$backends[[default_backend()]]$default_dtypes$float
```

and likewise for `int`. So `with_backend("quickr", ...)` gives quickr's `f64`
without writing anything, setting `anvl.backend` directly in an `.Rprofile`
behaves the same as the context manager, and a user who sets
`anvl.default_float = "f64"` gets it on every backend, which is what a global
option should mean.

| Situation | Effective float |
|---|---|
| pjrt, nothing set | `f32` |
| `with_backend("quickr", ...)`, nothing set | `f64` |
| `options(anvl.backend = "quickr")`, nothing else | `f64` |
| `options(anvl.default_float = "f64")`, any backend | `f64` |
| `with_backend("quickr", with_default_dtypes(c(float = "f32"), ...))` | `f32` |

Allowed values: `float` is `"f32"` or `"f64"` (pjrt's dispatcher keys and
wraps neither `f16` nor `bf16`, and `gradient()` supports only these two);
`int` is `"i32"` or `"i64"`, the signed dtypes an R integer builds at directly
(a narrower one would make the upload of an R argument go through R's
coercion, which wraps where the program's `convert` clamps; an unsigned one
cannot hold a negative). A default that a backend cannot lower -- `i64` on
quickr -- fails at lowering with the existing "Unsupported dtype for quickr
lowering" error; the option does not second-guess the backend.

The defaults decide only what a value becomes when *nothing else does*. The
yielding rule is untouched: `x_f32 + 1.5` is `f32` under an `f64` default,
`x_i8 * 2L` is `i8` under an `i64` default. What changes is the fallback:
`nv_array(1.5)`, `jit(\() 1.5)()`, the all-R-values branch of
`common_dtype_of()`, and the category-crossing case of `promote_dt_rdata()`
(`x_i32 + 1.5` is a float *at the default*).

### The user-facing surface

```r
default_dtypes()
#> list(float = <DataType>, int = <DataType>)   the effective pair, now

local_default_dtypes(dtypes, envir = parent.frame())
with_default_dtypes(dtypes, code)
```

`dtypes` is a named character vector or list with elements `float` and/or
`int` (`c(float = "f64")`), so that `code` is positional as in
`with_backend(backend, code)`. The setters set only the categories named and
validate eagerly. No `backend` argument anywhere: the backend is ambient. The
API-level constructors that used to hardcode a default take `dtype = NULL` for
"the effective default": `nv_seq()`, `nv_eye()`, `nv_runif()`, `nv_rnorm()`,
`nv_rbinom()`, `nv_sample_int()`; `nv_fill()` already did.

### Why eager and traced now agree

Every default read resolves through one function:

```r
current_default_dtypes <- function() {
  desc <- globals[["CURRENT_DESCRIPTOR"]]
  if (!is.null(desc)) desc$default_dtypes else effective_default_dtypes(default_backend())
}
```

- **Eagerly**, the pair is the ambient backend's effective pair. Since every
  operation runs on the ambient backend, this is the pair the next dispatch
  is keyed on. `as_anvl_arrays()` in a plain wrapper, `peek_dtype(1.5)`,
  `nv_fill(0, 3)`, the `dtype(1.5)` error message: all one answer.
- **In a trace**, the *baseline* is pinned on the `GraphDescriptor` by the
  compile callback, from `info$context` -- the very vector the dispatcher built
  the cache key from -- so the compiled program provably matches its key.
  Sub-descriptors inherit it; a bare `trace_fn()` takes the ambient pair.
  Switching the backend inside a traced body therefore changes nothing.
- **A scoped override inside a traced body is honoured**, over that baseline,
  so one program can use different precisions in different parts of itself:
  `with_default_dtypes(c(float = "f64"), <part of the body>)`. This is sound
  because the override is written in the body and so belongs to the program --
  it traces the same way every time the key does, and the key still carries
  only the baseline. A scope covers the values *built* inside it; a bare R
  value handed back out of one has not committed yet and takes the default
  where it is eventually used, by the same per-operation rule that lets it take
  the data type of an array it meets.

There is no longer a code path that can read a default for a backend other
than the one the operation runs on.

### Compiled programs and the cache

A compiled program bakes the default in. pjrt's dispatcher keys every entry on
a `context` vector its resolver returns per call (added by the superseded
design and unchanged here): anvl passes `c(float = , int = )`, the effective
pair for the dispatcher's backend, read from the options on every call with a
fast path for the common shape of a set option. Changing an option or the
backend therefore never serves a stale program, and switching back hits the
old entry.

## What is kept from the current tree

- pjrt: `dispatcher(context = )`, `CacheKey::context`, `info$context`, and
  their tests.
- anvl: `R/default-dtypes.R` as the single home of the mechanism;
  `AnvlBackend(default_dtypes = )` (required); `GraphDescriptor$default_dtypes`
  and the pinning in `compile_pjrt()` / `compile_quickr()`;
  `current_default_dtypes()`; `default_dtype_r(r_type, defaults)`;
  `RDataArray` without a stored default; `rdata_category()`;
  `resolve_upload_dtype(aval, requested, defaults)`; `nv_array()` resolving
  the dtype before `new_data`; the `dtype = NULL` API defaults; the
  `f32`/`f64`, `i32`/`i64` validation; the `context` closure per backend.

## What changes relative to the current tree

- The option `anvl.default_backend` becomes `anvl.backend`. `default_backend()`,
  `local_backend()` and `with_backend()` keep their names.
- Options `anvl.default_float.<backend>` / `anvl.default_int.<backend>` become
  `anvl.default_float` / `anvl.default_int`. `default_dtypes()`,
  `local_default_dtypes()`, `with_default_dtypes()` lose `backend`.
- `jit()` loses `backend`; `jit_auto()` is folded into `jit()` and loses
  `jit_auto_detect_backend()`; `resolve_device()` and `backend.JitFunction()`
  go. `device_arg()` stays (device only).
- `nv_array()`, `nv_scalar()`, `nv_empty()`, `nv_array_like()`,
  `nv_scalar_like()`, `nv_device()`, `nv_read()`, `nv_unserialize()` lose
  `backend`. `default_device()` keeps its internal `backend` parameter.
- The `nv_fill()` device-based resolution is reverted: a foreign device is an
  error, so the ambient pair is right.
- `new_primitive()` wraps with `jit(fn, static = static, device = device)`.
- Tests that construct arrays with `backend = "quickr"` and operate on them
  move inside `local_backend("quickr")`; tests of backend inference are
  deleted; `test-default-dtypes.R` drops the
  per-backend cases and gains the eager-agreement cases below.
- Docs: `jit()`'s *Device and Backend selection* section, the
  `add-api-function` skill ("work with any backend" now means: never name a
  backend; use `_like` for constants), `vignette("type-promotion")`, NEWS.

## Guarantees

1. With nothing set, pjrt behaves exactly as today (`f32` / `i32`).
2. quickr commits an R double to `f64` everywhere, eagerly and in a trace.
3. The yielding rule is unchanged.
4. **Eager and traced defaults agree**: for any operation, the default an R
   value commits to is the effective pair of the backend the operation runs
   on, and there is exactly one such backend at any time.
5. An array of another backend is an error at the dispatcher, never a wrong
   dtype.
6. A compiled program is never served under a default other than the one it
   was compiled with.
7. Every default is resolved on the anvl side; no backend runtime chooses a
   dtype for anvl.

## Testing plan

- **Effective pair:** `default_dtypes()` is `f32`/`i32` under pjrt, `f64`/`i32`
  under `local_backend("quickr")`; with `anvl.default_float = "f64"` it is
  `f64` under both; `with_backend("quickr", with_default_dtypes(c(float =
  "f32"), default_dtypes()))` is `f32`.
- **Eager construction and traces** follow the effective pair (the existing
  `test-default-dtypes.R` cases, minus `backend =`): `nv_array()`,
  `nv_scalar()`, `nv_fill()`, `nv_seq()`, `nv_eye()`, the samplers,
  `peek_dtype()`, `jit(\() 1.5)()`, `jit(\(x) x)(1.5)`, constants inside a
  trace, the all-R-values promotion branch.
- **Eager agreement, the motivating case:** under `local_backend("quickr")`,
  a plain (non-jitted) function that calls `as_anvl_arrays(x, 1.5, .promote =
  promote_common())` on an `i32` quickr array realizes `1.5` at `f64`; the same
  function under pjrt realizes at `f32`.
- **Yielding untouched**, **exactness** (`x / sqrt(2)` at an `f64` default),
  **validation** (`f16`, `i8`, `ui32`, unnamed or misnamed `dtypes`), as
  today.
- **Cache:** a jitted function called under `f32`, `f64`, `f32` returns
  `f32`, `f64`, `f32` with exactly two traces; the same across a backend switch
  (`f` called under pjrt, then inside `with_backend("quickr", ...)`, then under
  pjrt again).
- **Ambient backend:** `jit(f)` created under pjrt runs on quickr inside
  `with_backend("quickr", ...)`; a quickr array used under pjrt errors naming
  the backends; a device of another backend passed to a constructor errors;
  `jit(f, backend = )` and `nv_array(backend = )` no longer exist.

## Rejected alternatives

- **Per-backend options with argument-based backend inference** (the
  superseded design). Sound inside traces, unsound in eager code between
  dispatches; correctness rested on a convention. See *Problem*.
- **Keep inference, make the constructors ambient-only.** Arrays leave scopes
  regardless, so `f(with_backend("quickr", nv_array(1)))` still has eager code
  reading the wrong default. Closing the constructors closes nothing.
- **`with_backend()` writes the dtype options as a side effect.** Then setting
  `anvl.default_backend` directly gives a different result from the context
  manager, and a backend switch has to decide whether to clobber a user's
  override. The fallback lookup gives the same behaviour with no state.
- **A global default with no per-backend fallback** (JAX's model). Every JAX
  backend supports `f32`; quickr does not, so a global `f32` keeps today's
  `f32`-labelled doubles on quickr and a global `f64` penalizes pjrt users.
- **Keeping explicit `jit(f, backend = )`.** Its outputs are arrays of a
  non-ambient backend, unusable until the scope changes; it is `with_backend()`
  with an extra way to spell it. One mechanism.
