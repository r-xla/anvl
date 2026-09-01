# Removing `GraphRData` from the trace

- **Date:** 2026-08-31
- **Status:** Proposed
- **Builds on:** [`2026-08-20-rdata-type-design.md`](2026-08-20-rdata-type-design.md),
  which introduced `RData` / `GraphRData` / `RDataInput`. The decision *which*
  data type an R argument is uploaded at is not revisited here; only where the
  bookkeeping for that decision lives.

## Problem

`GraphRData` is a node class that appears in `desc$inputs` during a trace and is
gone by the time the trace finishes. It is not a value: it has no data type, is
never an operand of a `PrimitiveCall`, and `finalize_rdata_inputs()` swaps it out
for a real `GraphValue` at the end.

So the node union has a member that only exists mid-flight, and everything that
walks a trace has to know it. Concretely, today it costs:

- a mutable environment class with `format`/`print`/`shape`/`dtype` methods, plus
  `is_graph_rdata()` and `is_rdata_box()` predicates;
- an input list whose elements are *usually* `GraphValue`s, so nothing may assume
  they are;
- a per-node memo (`node$mat`, data type name → `GraphBox`) carrying a
  descriptor-reachability rule, because a memoized value built in `prim_if()`'s
  first branch must not be referenced from its second;
- `finalize_inline_rdata_inputs()`, whose job is to reconcile *two* such memos —
  an inline trace's and its enclosing trace's — in three cases.

A finished `AnvlGraph` never contains a `GraphRData`; the objection is entirely
to the intermediate state. That is what makes this worth doing: the class buys
nothing after the trace ends, so it should not be a graph node at all.

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

## Option 1 — move the bookkeeping off the graph (minimal)

`desc$inputs[[i]]` holds a hole (`NULL`) until finalize fills it. The aval, the
memo and the `outer` link move to `desc$rdata_slots[[i]]`, a plain record. The
body's handle becomes a box over a lightweight slot reference rather than over a
node.

Deletes the class and its methods from the node union; changes no behaviour, no
graph output, no snapshots. Does not simplify the memo or the inline
reconciliation — those move house unchanged.

This is the safe half-day version. It is worth doing only if Option 2 is judged
too large; it is not a stepping stone to it, since Option 2 deletes the memo
rather than relocating it.

## Option 2 — one open input, converts at the use sites (recommended)

**The argument is a `GraphValue` from the first moment. Its aval is an `RData`,
which is already an `AbstractArray` with no data type.** The openness moves from
the *node's class* to the *value's aval*, where the rest of the type system
already expresses it.

- `register_rdata_input()` becomes an ordinary `register_input()` of a
  `GraphValue(aval = RData(shape, r_type))`. `desc$inputs` is a list of
  `GraphValue`s, always.
- A use site needing data type `D` gets `prim_convert(input, D)`, memoized per
  `(input, D)` **in the current descriptor**. That memo is a local CSE table, not
  a rule about which descriptors can see which values.
- Out-of-category uses keep today's semantics explicitly rather than by
  recursion: a request for a data type the R value cannot be uploaded at
  contributes `rdata_natural_dtype(r_type)` to the requested set, and the convert
  runs from there. (An R double used only at `i32` still uploads `f64` and
  converts, as it does now — see *Only within the value's own category* in the
  2026-08-20 spec.)
- `finalize_rdata_inputs()` resolves as it does today, stamps the input's aval as
  `RDataInput(resolved, shape, r_type)`, and **folds** the one convert whose
  target is `resolved`: drop the call, and point its output's references at the
  input.

`dtype()` on the input errors while the trace is open because its aval is an
`RData` — the same error, from the same place the rest of the system gets it.
`is_rdata_box()` becomes a question about the aval, not about the node's class.

### What this deletes

- The `GraphRData` class, its four methods and two predicates.
- `node$mat` and its descriptor-reachability rule. Sub-graphs capture the input
  like any other outer value, and each makes its own convert in its own
  descriptor, so the sibling-branch question does not arise.
- Most of `finalize_inline_rdata_inputs()`. An inline trace's input for the
  argument is an input like any other; the trace forwards its requested set to
  the enclosing slot, and one resolution stamps both. The three-way case split
  exists today only because there are two per-data-type memos to reconcile — with
  no memos there is nothing to reconcile.

### What it costs

- **A fold step in finalize.** One convert per R argument is removed and its
  output's references retargeted — a pass over the descriptor's calls and
  outputs, the same order as finalize already is.

  This is not the "graph fixup pass" the 2026-08-20 spec rejected. That objection
  was to a pass that *re-decides* data types, letting a later statement change an
  earlier one's. This fold removes a convert that resolution has just proved
  redundant; it decides nothing.

  A cheap refinement, if the fold is unwelcome: let the first use claim the input
  provisionally and emit no convert for it. Then the common case (one data type)
  folds nothing, and only a resolution that widens past the first use pays.

- **Transient converts.** Between a use site and finalize the graph holds one
  convert per data type asked for, including the one that will be folded away.
  A trace inspected mid-flight looks different from today's.

- **Snapshot churn.** An in-flight input prints as its `RData` aval rather than
  `GraphRData(double[])`. Finished graphs are unchanged.

### Risk

The inline case is the part I am least sure of. The claim that forwarding a
requested set replaces the three-way reconciliation follows from removing the
memos, but `gradient()`'s trace nesting has corners (`prim_while` bodies inside a
forward trace, an argument used only in the backward pass) that this proposal has
not walked end to end. Do that before committing to the deletion; if it does not
hold, Option 1 still stands on its own.

## What does not change

`RData`, `RDataInput`, `resolve_upload_dtype()`, `rdata_build_candidates()`,
`dtype_holds()`, `rdata_builds_directly()` and the exactness guarantees they
encode. Which data type an argument uploads at, and why, is settled by the
2026-08-20 design and is untouched here. `graph_input_dtypes()` and the pjrt and
quickr dispatch paths read the same `RDataInput` avals off the same finished
graph.

## How to verify

The existing suite is the specification: `test-rdata.R`, `test-rdata-gradient.R`,
`test-primitive-rdata.R`, `test-jit-dispatch.R` and
`test-eager-jit-equivalence.R` between them pin the exactness assertions, the
two-data-type upload, the widening case where no requested type holds the others,
the sibling sub-graph case and the eager/traced agreement. All of them must pass
unchanged. `_snaps/graph.md` will need re-recording only where it shows a trace
in flight.

Worth adding: a test that no `GraphRData` — or, after this, no aval-less input —
survives into a finished graph, which is currently true by construction and
untested.

## Rejected alternatives

- **Upload every R argument at its natural data type and narrow afterwards.**
  Uniform during tracing, but needs a pass that rewrites input data types and
  drops the converts that become identities — and the 2026-08-20 spec already
  rejected putting `f64` into programs that have none, which this reintroduces
  for the whole window before the narrowing runs.
- **Re-trace once the data types are known.** Rejected in the 2026-08-20 spec:
  tracing is expensive.
- **Keep the node but hide it behind an accessor.** Does not address the
  complaint: the input list still holds a non-value.
- **Make R arguments static.** Bakes the value into the program and recompiles
  per literal; rejected in the 2026-08-20 spec.
