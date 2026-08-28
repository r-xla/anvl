# Primitives resolve R data with promotion rules

- **Date:** 2026-08-27
- **Status:** Implemented (revised after an adversarial review; see *History*)
- **Builds on:** `specs/2026-08-20-rdata-type-design.md`, which gave R values no
  dtype and made the `nv_*` layer decide one with a `.promote` rule. This is the
  same mechanism one layer down, with one rule kind added for it.
- **Prerequisite:** closing the category leak in `rdata_builds_directly()`
  (below), which is a live bug in its own right.

## Problem

A primitive commits every R value it receives to that value's *default* dtype
(`f32` / `i32` / `bool`), and does nothing else.

So whether a primitive call works depends on whether the array it is called with
happens to be at the default dtype:

```r
prim_mul(nv_scalar(1, dtype = "f32"), 2)   # f32
prim_mul(nv_scalar(1, dtype = "f64"), 2)   # Error: Got array<f64> and array<f32>
prim_pad(x_f64, 0, 1L, 0L, 0L)             # Error: `x` and `padding_value` must have the same dtype
prim_clamp(0, x_f64, 1)                    # Error: same
prim_add(1, 2L)                            # Error: Got array<f32> and array<i32>
```

**This is a coincidence, not a rule.** `prim_mul(x, 2)` working is a fact about
`f32` being the default, not about the call. The float default is intended to
become configurable; the day it does, a set of primitive calls silently changes
from working to failing.

**The failure is also in the wrong vocabulary** -- stablehlo's inference,
several frames down, phrased as `array<2xf64> and array<f32>`, naming a dtype
the caller never wrote and never mentioning that an R value was involved.

The `nv_*` layer has neither problem: it decides a dtype with a `.promote` rule
and *realizes* every R value at it. The primitive layer is where the old
behaviour still lives.

## Design

**A primitive declares a `.promote` rule over its arrayish arguments, the same
way a caller of `as_anvl_arrays()` does.** The rules, their `only =` and their
lists already say precisely "which of these arguments share a data type, and how
is it chosen".

What the primitive layer needs that the API layer does not is a rule that
*never moves an operand which already has a dtype*. That is one new rule kind.

### `promote_yield(only = NULL)`

> The target is the data type the already-typed operands have. Operands with no
> data type yet take it. Operands that have one are never converted.

Over the arguments the rule covers:

1. **No typed member** -- every member is an R value. They must all be of the
   same R storage type, and they commit to its default: `prim_add(1, 2)` is
   `f32`, `prim_add(1L, 2L)` is `i32`, `prim_add(TRUE, FALSE)` is `bool`. A
   group that mixes R storage types has no data type to agree on **and is an
   error** -- `prim_add(1, 2L)` does not silently become `f32`.
2. **Exactly one typed dtype** -- that is the target. Every uncommitted member is
   *realized* at it, built from the R data so it keeps every digit, provided the
   target is in the value's own category (below).
3. **More than one typed dtype** -- the operands cannot be brought together
   without converting one of them, which this rule does not do. **Resolve
   nothing** and let type inference report it, exactly as today.

Case 3 is deliberately passive. An earlier draft aborted here, to produce a
better message than stablehlo's; that turned out to break working code (see
*History*) and to contradict the fast path, since a call with two typed operands
has nothing uncommitted to trigger the rule in the first place. Improving that
message is a separate change to type inference, not this one.

### Within the value's own category, and no further

**A primitive only ever brings a literal to a data type of its own category.**

| R storage type | may become |
| --- | --- |
| `double` | any float |
| `integer` | any signed or unsigned integer |
| `logical` | `bool` |

Nothing else. An R integer does **not** become a float here, even though it
would be exact -- `prim_add(x_f64, 1L)` is an error, where `prim_add(x_f64, 1)`
is `f64`. Crossing a category is promotion, and promotion is the `nv_*` layer's
job: `nv_add(x_f64, 1L)` is `f64` as it always was.

That is the whole rule, and it is what keeps the primitive layer predictable:
the literal you write is built at a data type you could have written yourself,
or the call is rejected. Both rejections say so in anvl's own vocabulary rather
than stablehlo's:

```
Error in `prim_add()`:
! `rhs` is an R double, which cannot be used at the `i8` data type of `lhs`.
i A primitive builds a literal only at a data type of its own category: a
  double becomes a float, an integer an integer, a logical a `bool`.
i Use `nv_add()`, which promotes across categories, or convert explicitly with
  `nv_convert()`.
```

```
Error in `prim_add()`:
! `lhs` is an R double and `rhs` an R integer, so there is no data type for them
  to agree on.
i Give one of them a data type with `nv_scalar()`, or use `nv_add()`, which
  promotes across categories.
```

The test is a new predicate -- `rdata_in_category(r_type, dtype)` -- rather than
the existing `promote_dt_rdata()`, which is the *promotion* rule and does let an
R integer become a float. It is close to `rdata_builds_directly()`
(`graph.R:112`) without that function's float clause and width condition, since
those describe how to build a value faithfully, not whether it may.

**Category, not range.** `prim_add(x_i8, 200L)` yields `-55` and
`prim_add(x_ui8, -1L)` yields `0`, because narrowing inside the integer category
still wraps. This is accepted rather than fixed: R has no unsigned type and no
fixed-width integers below 32 bits, so no R-side type could carry the range, and
a range check would have to inspect a value the compiled program must not depend
on. It is what the `nv_*` layer already does, and it should be documented at
both layers.

The name is the design's own verb: the `RData` spec says an R value *yields* to
the array it meets. `promote_yield()` is that sentence as a rule.

### Why not `promote_common()` at the primitive layer

The two agree on the common cases and differ where a typed operand would have to
move:

| call | `promote_yield()` | `promote_common()` |
| --- | --- | --- |
| `prim_add(x_f64, 1)` | `f64`, R value built at `f64` | same |
| `prim_add(x_i8, 1L)` | `i8` | same |
| `prim_add(1, 2)` | `f32` | same |
| `prim_add(x_f64, 1L)` | error: an integer literal stays an integer | `f64` -- crosses the category |
| `prim_add(1, 2L)` | error: no data type to agree on | `f32` -- crosses the category |
| `prim_add(x_i8, 1.5)` | error: a float literal stays a float | `f32` -- **converts the `i8` operand** |
| `prim_add(x_f32, y_f64)` | today's inference error | `f64` -- **converts `x_f32`** |

Converting a typed operand is what makes `nv_add()` an API function: it
negotiates between its arguments. A primitive that did the same would stop being
a faithful record of one IR operation -- you could no longer tell from
`prim_add(a, b)` what the graph gets.

Nothing technical stands in the way of either. Resolution emits real
`prim_convert` nodes ahead of the call, exactly as `realize_at()` already does
for `nv_*`, so the graph stays honest and the interpretation rules are
untouched: `prim_convert` has its own reverse rule (`rules-reverse.R:559`), and
converts only arise out-of-category, i.e. for non-differentiable dtypes.

### Where the resolution happens

**At the primitive's entry**, in the callable `new_primitive()` builds, before
the body runs. The arrayish arguments are the non-`static` formals, which
`new_primitive()` already knows, taken in formal order with `...` spliced in
where it sits -- the same values, in the same order and under the same names,
that the body goes on to hand `graph_desc_add()`.

Not in `graph_desc_add()`, which was the first draft's choice and is too late:
several primitives read their operands' dtypes *before* recording a call.
`prim_reduce()` reads `dtype(x)` and `dtype(init)` to trace its reductor
(`primitives.R:973`), `prim_scatter()` pre-checks with `peek_dtype()`, and the
higher-order primitives trace their subgraphs first -- `maybe_box_input()`
commits every R value to its default to make a subgraph parameter slot, so a
later resolution would leave the parent operand disagreeing with the parameter
it was traced against.

Resolving at entry fixes all of those at once and is a single place, so
`graph_desc_add()` drops its own resolution and keeps only
`trace_commit_rdata_box()` for the arguments no rule covers. It also means the
whole primitive body sees settled operands, which is easier to reason about than
"settled by the time they are recorded".

The resolution itself is factored out of `as_anvl_arrays()` into a shared
`resolve_promote(args, promote)`, so the two layers run the same code.
Primitives do not go through `as_anvl_arrays()`: they need no device alignment,
since `jit()` places values during tracing.

Eager mode needs nothing of its own: every `prim_*` is `jit()`-wrapped, so
`prim_mul(x_f64, 2)` eagerly is a trace whose `2` is an R argument leaf. It is
resolved here, and `finalize_rdata_inputs()` records `f64` as its upload dtype.

### What each primitive declares

`new_primitive()` gains `promote`, defaulting to `promote_yield()` -- one group
over all arrayish arguments. `promote = NULL` means *no rule*: every R value
commits to its own default, which is today's behaviour.

| primitive | declaration | why |
| --- | --- | --- |
| `prim_sort` | `promote = NULL` | key and payload are **meant** to differ (`nv_argsort()` is `prim_sort(list(x_f32, idx_i32))`) |
| `prim_while` | `promote = NULL` | the loop-carried state is meant to be heterogeneous, and `nv_while` *is* `prim_while` |
| `prim_ifelse` | `promote_yield(only = c("true_value", "false_value"))` | `pred` is a `bool` |
| `prim_gather`, `prim_scatter` | `promote_yield(only = <value operands>)` | the indices are left **uncovered**, never pinned with `promote_dtype()`, which would convert a user's `i64` indices |
| `prim_dynamic_slice`, `prim_dynamic_update_slice` | `promote_yield(only = "x")` | likewise; the start indices are variadic and unnamed |

`prim_polygamma(n, x)` and `prim_triangular_solve(a, b)` want the default:
confirmed by running them with mismatched dtypes, both of which reject.
`prim_convolution(x, kernel)` is expected to want the default (XLA requires
matching element types) but has **not** been run.

Two cautions on the audit. The six `lhs`/`rhs` call sites are shared factories
(`make_binary_op` covers 12 primitives, `make_compare_op` 6), so the number of
two-operand primitives is around 24, not 6. And `prim_fill()` / `prim_iota()`
call `graph_desc_add()` with no arrayish arguments at all, so the resolver needs
an empty-group case.

### `promote_yield()` is not only for primitives

It is an ordinary rule, so `as_anvl_arrays(.promote = )` takes it, and it says
something `nv_*` functions want too: *these arguments must agree, and I will not
widen the array you gave me*. `nv_solve(a, b)` and `nv_triangular_solve(a, b)`
are the case -- they reject `a` and `b` of different dtypes today, R values
included:

```r
nv_solve(a_f64, matrix_of_R_doubles)   # Error: `a` and `b` must have the same dtype
```

With `promote_yield(only = c("a", "b"))` the R matrix yields to `a`, and two
typed arrays that disagree are still rejected -- which is the behaviour those
functions were reaching for. The category rule applies there too, so a matrix of
R *integers* is still rejected; a function that wants those to cross should use
`promote_like("a")`, which promotes.

## Prerequisite: an R integer must not request a float

`resolve_upload_dtype()` (`graph.R:745`) picks the *widest* dtype a leaf was
built at, breaking ties alphabetically, justified by "the widest of them holds
every use site's value". Today that justification is false, and the reason is
the same category leak this design removes from the primitive layer:

```r
rdata_builds_directly("integer", <any float>)   # TRUE
```

So an R **integer** leaf can be built at both `i32` and `f32`. They tie at width
32, `"f32"` sorts first, and the leaf uploads as a float carrying 24 bits of
mantissa:

```r
f <- jit(function(a, xi, xf) list(nv_add(xi, a), nv_add(xf, a)))
f(1073741825L, nv_scalar(0L), nv_scalar(0))
#> i32 site: 1073741824      # want 1073741825
```

(A double leaf builds only at floats and a logical only at `bool`, so an R
integer is the only one that can straddle. The width-64 tie is harmless: an R
integer is at most 2^31, which `f64` holds exactly.)

**Fix: an R integer is built only at an integer dtype.** A float use site is
served by converting the integer in the program, exactly as a narrow integer
site already is. That is one clause removed from `rdata_builds_directly()`:

```r
integer = is_dtype_intish(dtype) && dtype_width(dtype) >= 32L
```

`resolve_upload_dtype()` then needs no change at all -- with the leak closed, its
candidates really are all in the value's own category, and the widest of them
really does hold every use site's value. It also puts the two predicates in the
same shape: `rdata_in_category()` is the category test, and
`rdata_builds_directly()` is that test plus "wide enough to build directly
rather than via a convert".

No value changes. An R integer at `f64` is exact whether it is built there or
converted from `i32`; at `f32` it rounds once either way. What changes is the
IR: an R integer used at a float dtype now appears as an integer constant and a
`convert` rather than a float constant. For an in-body literal the compiler
folds that away; for an argument leaf it is the point of the fix.

This is a live bug today, independent of everything else here. It becomes more
reachable under this design, because case 2 is what lets primitive call sites
ask for a leaf at a second dtype.

## What this does and does not fix

It fixes the primitives whose operand is *meant* to be a scalar:

```r
prim_pad(x_f64, 0, 1L, 0L, 0L)     # 0 built at f64
prim_clamp(0, x_f64, 1)            # both bounds built at f64
prim_reduce(x_f64, init = 0, ...)  # init built at f64
prim_mul(x_f64_scalar, sqrt(2))    # exact, no f32 rounding
prim_add(1L, 2L)                   # i32
```

The calls it does *not* make work, it rejects for a stated reason instead of a
stablehlo type mismatch -- `prim_pad(x_f64, 0L)` and `prim_add(1, 2L)` both name
the category rule and the way out. That is a smaller win than making them work,
and it is the intended trade: a literal is built at a data type the author could
have written, or not at all.

It does **not** make `prim_mul(x_f64_array, 2)` work. That fails on *shape*
(`array<2xf64>` vs `array<f64>`), and shape is deliberately left alone:
primitives do not broadcast, and an R value's shape -- unlike its dtype -- is not
undetermined. `shape(1.5)` is `()`, and answering that is part of the existing
design. Broadcasting stays `nv_broadcast_scalars()`'s job.

**The benefit is narrower than it looks, and worth stating plainly.** `nv_pad()`
and `nv_clamp()` already do this one layer up with `promote_like("x")`, and
*more* strongly, since they also convert a typed bound. A grep over `R/` finds no
`nv_*` body passing a bare literal into an arrayish primitive slot, so the change
has no internal consumers today; its value is to the exported primitives, which
are user-facing and callable eagerly, and to removing the dependence on the
default dtype. The reverse rules that do want a literal write
`prim_fill(1, dtype = dtype(x), shape = shape(x))` because **shape**, not dtype,
is what binds them; none of them gets shorter.

## Guarantees

- **No dependence on the default**, for every primitive that declares a rule.
  `prim_sort` and `prim_while` opt out and keep it, which is correct for them:
  their operands are meant to differ, so there is nothing to yield to.
- **Exactness reaches the primitive layer.** An R value used by a primitive at
  `f64` arrives at full `f64` precision -- given the prerequisite fix, without
  which it does not.
- **Primitives stay faithful.** No typed operand is ever converted, so a
  `prim_*` call remains a record of one IR operation.
- **One answer in both modes**, since eager and traced both resolve at entry.
- **One vocabulary.** The layer difference is which rule each layer declares,
  not a second mechanism.
- **A literal is built only at a data type its author could have written.** An
  R double never becomes an integer, an R integer never a float, an R logical
  never anything but `bool`. Everything else is promotion, and stays upstairs.
- **Nothing that works today stops working.** Case 3 is passive, the two
  heterogeneous primitives opt out, and the category rule is a superset of
  today's behaviour: a literal succeeds today only when the other operand is at
  its own default, which is always within its category.

## Testing plan

- **Regression first.** `nv_argsort()`, `nv_sort()` with a payload, and a
  heterogeneous `nv_while()` loop (`i32` counter, `f32` state) keep working.
  These are what the first draft broke.
- **Per primitive, per slot.** For every primitive with two or more arrayish
  arguments, call it with an R value in each slot against a non-default dtype
  (`f64`, `i8`, `i64`, `bool`) and assert the result's dtype and value.
- **The declarations.** One test per declared exception: `prim_ifelse(pred,
  x_f64, 0)` is `f64` with `pred` still `bool`; a gather with a bare R index
  keeps the index integral; a gather with an `i64` index keeps it `i64`.
- **All-R groups.** `prim_add(1, 2)` is `f32`, `prim_add(1L, 2L)` is `i32`,
  `prim_add(TRUE, FALSE)` is `bool`; every mixed group (`prim_add(1, 2L)`,
  `prim_add(TRUE, 1L)`) aborts naming both arguments.
- **The category rule.** `prim_add(x_f64, 1L)`, `prim_add(x_i8, 1.5)` and
  `prim_mul(x_bool, 1L)` all abort naming the argument; `prim_add(x_f64, 1)`,
  `prim_add(x_i8, 1L)` and `prim_mul(x_bool, TRUE)` all succeed. The accepted
  wrap-around cases (`prim_add(x_i8, 200L)`) are pinned so the behaviour is
  deliberate rather than incidental.
- **Case 3 is passive.** `prim_add(x_f32, y_f64)` still errors from inference.
- **Upload dtype.** A leaf used at `i32` and `f32` in one program keeps every
  bit at the `i32` site, and an R integer leaf never uploads as a float.
- **The IR shift.** An R integer used at a float dtype appears as an integer
  constant plus a `convert`; the values it produces are unchanged at `f64` and
  at `f32`.
- **Exactness.** `prim_mul(x_f64, sqrt(2))` is exact in both modes; the IR holds
  an `f64` constant and no `convert`.
- **Eager equals traced**: extend `test-eager-jit-equivalence.R` with a
  primitive-level sweep.
- **Cost.** Measured: a traced `prim_add` costs ~640 us (each `prim_*` is a
  `jit()` call), a `vapply()` over two arguments ~8.5 us -- about 1.3%. The fast
  path is not needed for cost.

## Rejected alternatives

- **A separate `dtype_groups` field on the primitive** (the first draft). It is
  `only =` under a second name -- two vocabularies for one job.
- **`promote_common()` at the primitive layer.** Prims promote, and `nv_*`'s
  promotion becomes largely redundant. Rejected for the last two rows of the
  table above: it converts a typed operand, so `prim_add(a, b)` would no longer
  tell you what the graph gets.
- **Aborting when a group holds several typed dtypes** (the first draft's step
  4). It breaks `nv_argsort()` and `nv_while()`, and contradicts the fast path.
- **Reject R values at the primitive layer.** No surprise, because nothing
  works; unpleasant for a layer that is exported and callable eagerly.
- **Infer the group dynamically from whichever operands are typed.** Ambiguous
  exactly where it matters: in `prim_ifelse(pred_bool, 1, x_f64)` nothing says
  which typed operand the `1` should follow.
- **Treat shape as undetermined too.** Contradicts the existing design: an R
  value *has* a shape and `shape()` reports it.

## History

Revised after an adversarial review that ran the cases rather than reading them.
It sank the first draft's step 4 (abort on several typed dtypes) by finding
`nv_argsort()` and `nv_while()`, showed `graph_desc_add()` is not the single
funnel by finding `prim_reduce()`, caught the `promote_dtype()` suggestion for
gather indices contradicting the design's own guarantee, found the
upload-dtype bug, and established that the benefit accounting overstated the
case. The entry-point resolution, the gather declaration and the
honest accounting above are all consequences.

The category rule replaced an earlier draft in which an all-R group took
`common_dtype_of()` of its members, making `prim_add(1, 2L)` an `f32`. A literal
crossing its category is promotion, and promotion belongs to the `nv_*` layer;
the primitive layer builds a literal only where the author could have written the
data type themselves.

Applying that same rule to `rdata_builds_directly()` turned the upload-dtype fix
from a special case in `resolve_upload_dtype()` into the removal of one clause,
and made that function's original justification true rather than patched.
