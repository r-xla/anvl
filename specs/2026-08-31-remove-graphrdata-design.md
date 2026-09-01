# Removing `GraphRData` and `RDataInput` from the trace

- **Date:** 2026-08-31
- **Status:** Implemented
- **Builds on:** [`2026-08-20-rdata-type-design.md`](2026-08-20-rdata-type-design.md),
  which introduced `RData` / `GraphRData` / `RDataInput`. *Which* data type an R
  argument is uploaded at is settled there and is not revisited; this is only
  about where the deferral and its result are stored.
- **Reverses:** that spec's "the finished graph carries it where the input is
  rather than beside it" — see *Why beside it, after all*.

## Problem

Two classes exist only to hold data type information that has nowhere else to
live, and both make the graph's value types less uniform than they should be.

**`GraphRData`** is a node that appears in `desc$inputs` during a trace and is
gone by the time the trace finishes. It is not a value: no data type, never an
operand of a `PrimitiveCall`, swapped out for a real `GraphValue` by
`finalize_rdata_inputs()`. It costs a mutable environment class with four
methods and two predicates, an input list whose elements are only *usually*
`GraphValue`s, a per-node memo (`node$mat`) carrying a descriptor-reachability
rule so that a value built in `prim_if()`'s first branch is not referenced from
its second, and `finalize_inline_rdata_inputs()`, whose job is to reconcile two
such memos in three cases.

**`RDataInput`** is an `AbstractArray` subclass whose extra field is the R
storage type. After finalize, that field is read in exactly one place —
`graph-print.R:29`, for the `<- integer` annotation — and the class's other
consumer, `graph_input_dtypes()`, uses it only as a yes/no flag before reading
the `dtype` every `AbstractArray` already has. So an aval subclass exists to
serve one printer, and every function that walks avals (`at2vt()`, the
lowering rules, the passes) has to tolerate a subclass it gains nothing from.
`repr.RDataInput()` appears to be dead: nothing calls `repr()` on an aval.

A finished `AnvlGraph` never contains a `GraphRData`. That is what makes this
worth doing: the class buys nothing once the trace ends, so it should not be a
node at all.

## What the deferral actually has to do

Any replacement owes the same five things:

1. **Reserve the input slot in argument order.** The caller supplies inputs
   positionally, so the slot exists from the moment the argument is seen even
   though its data type is not known.
2. **Record every data type the body used the argument at**, so
   `resolve_upload_dtype()` can pick one that holds them all.
3. **Deduplicate.** An argument used twice at one data type takes one input.
4. **Link an inline trace's use to the enclosing trace's argument**, so
   `jit(gradient(f))(sqrt(2))` resolves against the union of both bodies' uses.
5. **Answer `shape()` / `naxes()` while open, and error on `dtype()`.**

Only (1) forces anything into the graph. (2)–(4) are trace bookkeeping that
happens to be stored on a node today.

## Design

### During the trace: `GraphBox(GraphValue(RData(...)))`

The argument is a `GraphValue` from the first moment, and its aval is an
`RData` — already an `AbstractArray` with no data type. The openness moves from
the *node's class* to the *value's aval*, where the type system already
expresses it.

- `register_rdata_input()` registers an ordinary input. `desc$inputs` is a list
  of `GraphValue`s, always.
- The per-data-type memo and the inline trace's `outer` link move to two
  hashtabs on the descriptor (`rdata_mat`, `rdata_outer`), keyed by that input
  `GraphValue`. Both are trace bookkeeping and neither outlives the trace, which
  is the whole reason they should not be fields of a graph node.
- `materialize_rdata()` is unchanged in behaviour: a use site needing data type
  `D` gets a value built at `D`, memoized. Out-of-category uses still build at
  `rdata_natural_dtype(r_type)` and convert.
- `dtype()` on the input errors while the trace is open because its aval is an
  `RData` — the same error, from the same place the rest of the system gets it.
  `is_rdata_box()` becomes a question about the aval, not the node's class.

### As built: no fold was needed

The design below anticipated a fold step — one convert per argument removed once
resolution proves it redundant. The implementation does not need it, by taking
the spec's own refinement to its conclusion: rather than emitting a convert at
every use site and folding one away, `materialize_rdata()` keeps building a
fresh value per data type asked for (as it does today) and finalize *chooses
which of them becomes the input*. Nothing is ever emitted that has to be
removed, so there is no rewrite pass and no transient converts.

That leaves the memo in place — relocated from the node to two hashtabs on the
descriptor (`rdata_mat`, `rdata_outer`), keyed by the input `GraphValue`. So the
memo's descriptor-reachability rule and the three-case inline reconciliation
survive; what goes is the node class, the aval subclass, and any need for the
graph's value types to know about either. Removing the memo itself would want
the fold, and can be revisited on its own.

### At finalize: `inputs` + `rdata_types`

`resolve_upload_dtype()` runs unchanged. Then, instead of stamping an
`RDataInput` onto the aval:

```r
graph$inputs      # list<GraphValue>, avals all plain AbstractArray
graph$rdata_types # character(), one per input: NA, or the R storage type
                  # e.g. c(NA, "double", "integer")
```

The input's aval becomes `AbstractArray(resolved, shape)` like any other, and
the R storage type moves to the parallel vector. Together they carry exactly
what `RDataInput` did: `dtype` and `shape` from the aval, `r_type` and the
is-it-R-data flag from the vector.

Which of the values the body built becomes the input is decided here, as before:
finalize picks the one at `resolved` and adds a convert from it for each other
data type the body asked for. Nothing has to be un-emitted.

The vector is **derived at finalize**, in the one pass that is already replacing
those avals — not maintained in lockstep during tracing. There is no window in
which it can drift, and nothing to keep in step at `register_input()`. Its only
invariant is positional, and every existing mutation of an input list is a
positional replacement (`finalize_rdata_inputs()` at `graph.R:1043`,
`inline_scalarish_constants()` at `graph-passes.R:81`); nothing reorders or
filters inputs.

`graph_input_dtypes()` then reads the flag from `rdata_types` and the data type
from the aval, which is what it already does minus the subclass test.

### Why beside it, after all

The 2026-08-20 spec put the R storage type on the aval so the finished graph
would carry it "where the input is rather than beside it". That was the right
instinct for a field the graph's consumers would reach for — but a year of
consumers later there is one, the printer, and it reads the field only to
annotate. The cost of the placement is that `AbstractArray` has a subclass, so
no pass over avals can assume the plain shape. Paying a subclass for one
annotation is the wrong trade; a parallel vector whose only invariant is
positional, in a graph where inputs are only ever replaced positionally, is
cheaper than a type distinction every aval consumer must tolerate.

## What this deletes

- `GraphRData`, its four methods and two predicates.
- `RDataInput`, its constructor, `format`/`print`/`repr` methods and
  `is_rdata_input()`.

*Not* deleted, per *As built* above: `node$mat` (relocated to the descriptor as
`rdata_mat`) and the three-case split in `finalize_inline_rdata_inputs()`. Both
would go with the fold variant; neither is a graph node, so neither blocks this.

## What it cost, as built

- **Two hashtabs on the descriptor** in place of two fields on a node.
- **`rdata_types` must survive the graph passes.** `remove_unused_constants()`
  and `inline_scalarish_constants()` rebuild the graph with `AnvlGraph(...)` and
  silently dropped the new field on the first run, which the pjrt dispatcher
  caught immediately (`input_dtypes is required, because input 1 is bare R
  data`). Both now carry it. This is the parallel vector's one real hazard and
  it has a test.
- **The printer takes the graph's vector**, since the annotation no longer rides
  on the aval. `format_aval_short()` gained an `r_type` argument that only the
  Inputs section passes.

### What the fold variant would have cost

- **A fold step in finalize.** One convert per R argument is removed and its
  output's references retargeted — a pass over the descriptor's calls and
  outputs, the same order finalize already is.

  This is not the "graph fixup pass" the 2026-08-20 spec rejected. That objection
  was to a pass that *re-decides* data types, letting a later statement change an
  earlier one's. This fold removes a convert that resolution has just proved
  redundant; it decides nothing.

  A refinement if the fold is unwelcome: let the first use claim the input
  provisionally and emit no convert for it. Then the common case (one data type)
  folds nothing, and only a resolution that widens past the first use pays.

- **Transient converts.** Between a use site and finalize the graph holds one
  convert per data type asked for, including the one that will be folded away.
  A trace inspected mid-flight looks different from today's.

- **Snapshot churn.** An in-flight input prints as its `RData` aval rather than
  `GraphRData(double[])`. The printer's `<- integer` annotation now comes from
  `graph$rdata_types` rather than the aval, so `format_aval_short()` needs the
  graph in hand — it is called from `format_graph_body()`, which has it, but the
  argument has to be threaded. Finished graphs print the same.

## Risk

Retained, since the fold variant is still open. The inline case is the part I am
least sure of. The claim that forwarding a
requested set replaces the three-way reconciliation follows from removing the
memos, but `gradient()`'s trace nesting has corners — a `prim_while` body inside
a forward trace, an argument used only in the backward pass — that this proposal
has not walked end to end. Do that before committing to the deletion.

The fallback, if it does not hold, is to keep the deferral as it is and take only
the `RDataInput` half: that part is independent, since it concerns what finalize
*writes* rather than how the trace defers.

## What does not change

`RData`, `resolve_upload_dtype()`, `rdata_build_candidates()`, `dtype_holds()`,
`rdata_builds_directly()` and the exactness guarantees they encode. Which data
type an argument uploads at, and why, is settled by the 2026-08-20 design. The
pjrt and quickr dispatch paths get the same `input_dtypes` vector they get today.

## How to verify

The existing suite is the specification: `test-rdata.R`, `test-rdata-gradient.R`,
`test-primitive-rdata.R`, `test-jit-dispatch.R` and
`test-eager-jit-equivalence.R` between them pin the exactness assertions, the
two-data-type upload, the widening case where no requested type holds the others,
the sibling sub-graph case and the eager/traced agreement. All must pass
unchanged. `_snaps/graph.md` needs re-recording only where it shows a trace in
flight.

Worth adding: that `length(graph$rdata_types) == length(graph$inputs)` for every
finished graph, and that no input aval is data-type-less — both currently true by
construction and untested.

## Rejected alternatives

- **Move the bookkeeping to the descriptor but keep the per-data-type memo**
  (`desc$rdata_slots`, `desc$inputs` holding a hole until finalize). Deletes the
  node class without deleting the memo or the inline reconciliation, so it pays
  the disruption and keeps the complexity.
- **Upload every R argument at its natural data type and narrow afterwards.**
  Uniform during tracing, but needs a pass that rewrites input data types and
  drops the converts that become identities — and reintroduces `f64` in programs
  that have none for the whole window before the narrowing runs.
- **Re-trace once the data types are known.** Rejected in the 2026-08-20 spec:
  tracing is expensive.
- **Keep `RDataInput` but drop its `r_type`**, taking the annotation from
  elsewhere. Leaves an `AbstractArray` subclass whose only content is its own
  identity — the flag — which the parallel vector supplies for free.
- **Make R arguments static.** Bakes the value into the program and recompiles
  per literal; rejected in the 2026-08-20 spec.
