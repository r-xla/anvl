# RData: R values keep their dtype open during tracing

- **Date:** 2026-08-20
- **Status:** Implemented
- **Issue:** [r-xla/anvl#373](https://github.com/r-xla/anvl/issues/373) — `f64`
  arithmetic silently losing precision when combined with R literals
- **Supersedes:** the `weak`/`undetermined` design on `feat/literal-dtype-commitment`
  and `worktree-ambiguous-float` (origin sets, anchors, end-of-trace commitment,
  graph fixup). None of that machinery is needed here.

## Problem

An R value entering an anvl program is stamped with a dtype at its entry point,
before anything is known about what it will be combined with. R doubles are
stamped `f32`, so `sqrt(2)` is rounded to 24 bits of mantissa; when it later
meets an `f64` array, promotion inserts an `f32 -> f64` convert, which widens the
*already rounded* value. `x_f64 / sqrt(2)` is therefore off by ~1e-8 instead of
~1e-16.

There are two entry points, and both matter:

- **In-body**: `jit(\(x) x / sqrt(2))`. The literal becomes a `GraphLiteral`
  whose `LiteralArray$data` still holds the exact R double; only the dtype stamp
  (and the convert it forces) loses the precision.
- **Arguments**: eager `x_f64 / sqrt(2)`. Every `nv_*` function is jit-wrapped,
  so the literal is an *argument leaf* of `nv_div`'s jit call. pjrt's dispatcher
  keys it as a bare R double and uploads it to the device with a hardcoded
  `"f32"` (`dispatch_engine.cpp`), before anvl's trace has any say.

## Design

An R value (a length-1 atomic, or an `is.array()` value) that enters a trace is
**not converted**. It is represented as a new abstract array kind, `RDataArray`,
which carries the R data (or, for an argument leaf, just its storage type and
shape) and **no dtype**. It is *materialized* — built into a real graph value —
at each use site, at the dtype that site resolved to, directly from the R data.

The value is what yields in promotion: an `RDataArray` takes the dtype of the
array it meets -- within its own category where it can -- and contributes
`f32`/`i32`/`bool` (`default_dtype_r()`) when nothing else does. What changes
next to the old behaviour is *how* the value is realized: from the exact R
double at the resolved dtype, instead of from an `f32` stamp plus a widening
convert.

### The `ambiguous` flag goes away

The flag existed to model exactly this yielding, by marking a *result* as
literal-derived so it could yield again later. With the yielding attached to the
uncommitted value itself, the flag has nothing left to do: the yielding *rule*
is unchanged (`promote_dt_rdata()` is `promote_dt_ambiguous_to_known()` under a
new name, identical over every dtype pair), and only its *reach* changes -- it
now applies to the operand that has not committed, rather than to every value
descended from one. `nv_add(1, 1)` is a plain `f32`.

What that costs is real, and larger than one example: a result that used to keep
yielding now promotes normally. `(x_bool * 1L) + y_i16` is `i32` rather than
`i16`, and the same goes for the other five `weak i32` x `{i8, i16, ui8, ui16,
ui32, ui64}` cells -- including two where the old result was not simply the
narrower dtype (`weak i32` x `ui32` was `ui32`, plain promotion is `i64`). The
trade is deliberate: one mechanism instead of two, and a value's dtype no longer
depends on how far back its ancestry reaches.

This is PyTorch's model for Python scalars (a "wrapped number" yields for the one
operation it takes part in, and results never carry the mark), rather than JAX's
(`weak_type` propagates through results).

So the flag is deleted throughout: the `ambiguous()` generic, the `ambiguous`
argument of the constructors and of `prim_fill()` / `prim_iota()` /
`prim_convert()`, the `ambiguity` argument of `eq_type()`, the `?` suffix in
printed dtypes, and pjrt's `Aval::ambiguous`. `common_dtype()` becomes the
promotion of two known dtypes; `promote_dt_rdata()` is the yielding rule.

One consequence for pjrt: its cache key kept an array leaf and a bare R leaf
apart only by that bit. They now trace to different programs -- an R value can
commit to a dtype the array never had -- so the key carries the leaf *kind*
instead.

### Commitment rule

An R value commits when it first meets a typed array — it takes the dtype
promotion computes for that operation. If it never meets one (it only ever meets
other R values), it commits to the **default dtype** (`f32` for doubles, `i32`
for integers, `bool` for logicals; the float default is intended to become
configurable later).

Commitment is local and per use site; there is no whole-trace analysis. The
consequence is that transitivity through an intermediate is not caught:

```r
jit(function(x) {
  y <- x * 2                  # x's only use: meets the literal 2 -> f32
  y + nv_scalar(1, "f64")     # the f64 arrives on y, not on x
})(sqrt(2))
```

`x` commits to `f32` at `x * 2`. This is accepted and documented; the workaround
is to give the value a dtype (`nv_scalar(sqrt(2), "f64")`). Every case where an R
value *directly* meets a typed array is exact.

### The generics

An `RDataArray` answers the extractors the way the underlying R value can:

| generic | on `RDataArray` |
| --- | --- |
| `shape()`, `naxes()` | works — `()` for a length-1 vector, `dim()` for an R array |
| `dtype()` | **errors**: there is no dtype yet |

So `jit(function(x) nv_fill_like(x, 3))(1)` errors, where it previously produced
an `f32` array: `nv_*_like` derives its result dtype from its argument, and an
uncommitted value has none to give. The error names the fix (pass an explicit
`dtype`, or `nv_array(x, dtype = )`).

Internally, promotion reads the aval's `default_dtype` field rather than the
generic, so working out what a value *would* commit to never commits it.

### Materialization

`materialize_rdata(box, dtype)` returns a `GraphBox` for the value realized at
`dtype`, memoized per dtype on the node:

- **in-body scalar** -> `GraphLiteral(LiteralArray(data, dtype = dtype))`, inlined
  into the program by the existing lowering (`hlo_tensor(value = <raw R data>)`).
- **in-body array** -> a plain-backend `AnvlArray` of that dtype, registered as a
  graph constant.
- **argument leaf** -> a `GraphValue` that becomes a program input of that dtype.

### Only within the value's own category

Building the R value at the dtype directly is only sound while R's coercion and
the program's `convert` agree, and they do not once the value leaves its own
category: XLA clamps a float that overflows an integer dtype and maps `NaN` to
zero, while R wraps or produces `NA` — and a literal that is out of range for the
target cannot be written into the IR at all (`nv_convert(-1, "ui8")` would emit
an unsigned constant of `-1`).

So a value is built directly only at a dtype that holds it faithfully: a double
at any float, an R integer at any float or any integer of at least 32 bits, a
logical at `bool`. Every other target is reached by building at the value's
**natural** dtype — `f64`, `i32`, `bool`, the one that holds the R value exactly
— and emitting a `convert`. Narrowing is then the program's own, identical for a
literal, an argument and an eager call, and identical to what a typed array of
the same value would give.

Within the category nothing is lost: float-to-float rounds exactly once whether
it is built at the target or converted from `f64`, and an R integer reaches any
32-bit-or-wider integer dtype unchanged.

### Argument leaves and the upload dtype

An argument leaf has no value at trace time, and must not have one: the jit cache
keys a raw R leaf on its storage type and shape only, so the compiled program
must not depend on the value. A leaf therefore materializes into a *program
input*, and the trace records which dtype that input needs.

A leaf can be used at more than one dtype. Every dtype it is *built* at is in
its own category (the rule above), so the widest of them holds every use site's
value, and that is the **upload dtype**. A narrower site converts down from an
exact upload, which rounds exactly once — the same result it would have had on
its own, so no use site's value depends on which other sites exist. A leaf that
is only ever converted out of its category uploads at its natural dtype, and the
conversion happens in the program.

In the overwhelmingly common case a leaf is used at exactly one dtype, and the
graph contains no convert at all.

### pjrt: `input_dtypes`

pjrt's dispatcher currently uploads a bare R leaf with a hardcoded
`"f32"`/`"i32"`/`"pred"`. The compile callback now returns `input_dtypes`, one
entry per execute-time input (`NA` for an `AnvlArray` input, which is passed
through unchanged). The cache entry stores them and `PjrtEngine::run()` uploads
with them. The key classifies the leaf by its kind, storage type and shape — what the
trace depends on — and never by its value.

`ClosureEngine` (the quickr backend) hands the raw leaves to the compiled R
closure, which coerces them itself, so it needs nothing from this.

## Guarantees

- **Exactness**: an R value reaching an `f64` use site arrives at full `f64`
  precision. No double rounding anywhere; a single host-side rounding when it
  commits below `f64`.
- **Value-independent caching**: the trace reads a leaf's storage type and shape,
  never its value. `f(1.5)` and `f(2.5)` share an executable.
- **No new f64 in f64-free programs**: an upload only widens to `f64` when an
  `f64` use site asked for it, i.e. when the program already contains `f64`.
- **No surprise widening**: an R value never drags a program to a wider dtype;
  it takes the one it meets, or the default. The one exception is a value that
  is *only* converted out of its category, which uploads at its natural dtype
  so the conversion sees the value itself.
- **One answer per program**: a use site's value does not depend on what the
  other use sites of the same value are, and the eager, in-body and argument
  spellings of the same conversion all agree.

## Testing plan

- Issue reproductions at ~1e-15 relative error, in both entry modes: in-body
  (`jit(\(x) x / sqrt(2))(f64)`) and argument (eager `x_f64 / sqrt(2)`,
  `jit(\(t) nv_scalar(-1, "f64") / t)(sqrt(2))`), for `+ - * /`, and for
  `nv_log2()` / `nv_log10()`, whose constants live in their bodies.
- R arrays, not just scalars: `x_f64 * array(c(0.1, 0.2))`.
- Value-independent caching: `f(1.5)` and `f(2.5)` share one cache entry, and the
  second call returns the right value (i.e. nothing baked the first one in).
- Multi-dtype leaf: one argument used at `f32` and at `f64` in one program —
  uploaded once at `f64`, converted for the `f32` site, both results exact.
- Commitment: a value that meets nothing takes the default (`jit(identity)(1)`
  is an `f32` scalar); one that meets a typed array takes its dtype, whatever
  the width; the documented transitive case still commits to the default.
- The generics: `dtype()` on an uncommitted value errors with the guidance
  message; `shape()` / `naxes()` work; `nv_fill_like(x, 3)` on one errors.
- Emitted IR: an anchored literal appears as a direct `f64` constant with no
  `f32 -> f64` convert; an `f32`-only program contains no `f64`.
- quickr equivalents of the core cases, `skip_if_no_quickr()`-guarded.

## Rejected alternatives

- **Origin sets + anchors + end-of-trace commitment** (the previous branches):
  fixes the transitive case, at the cost of whole-trace bookkeeping, a graph
  fixup pass, and a rule that a later statement can change an earlier one's
  dtype. Not worth it for the case it buys.
- **Always uploading raw doubles as `f64` and converting in-program**: needs no
  per-input dtype, but puts `f64` into programs that have none (Metal cannot run
  them) and wastes bandwidth on large R arrays.
- **Re-tracing** once the dtypes are known: tracing is expensive.
- **Keeping the `ambiguous` flag alongside the new type**: two mechanisms for
  one job, and the flag's only distinguishing case (a weak value wider than the
  known one it meets) can no longer arise.
- **Making raw scalar arguments static**: bakes the value into the program and
  recompiles for every distinct literal.
