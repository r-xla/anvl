# TODOs

## Reverse mode: optimization potential

`prim_if()`'s reverse rule splits each branch into a forward graph that also
returns residuals and a backward graph that reads them (`split_vjp()`), so the
backward pass does not rerun the forward branch. Inspecting the reverse graphs
showed correct wiring throughout; what follows is avoidable work, deliberately
left out for now to keep the implementation simple.

### Cleaner structure first

JAX has the same special cases for `cond`, but spread over general passes, each
with a small per-primitive rule: JVP with symbolic zeros (`ad.Zero`), partial
evaluation (`pe.partial_eval_jaxpr_nounits`), dead code elimination
(`pe.dce_jaxpr`, `_cond_dce_rule`) and transposition (`_cond_transpose`), see
`jax/_src/lax/control_flow/conditionals.py`. The optimizations below are best
added along those lines rather than as more flags on the one reverse pass:

- **Symbolic zeros:** an absent cotangent stays `NULL` throughout, and rules
  handle `NULL` grads, instead of `run_backward_pass()` filling missing
  cotangents with `zeros()`.
- **Liveness:** narrow the values that require a gradient to those a cotangent
  can reach (walking backward from the loss), as a pass of its own.

### 1. Calls whose outputs never reach the loss are differentiated with zeros

Requirements only propagate *forward* from `wrt` (`propagate_requirements()`,
`R/reverse.R`). `run_backward_pass()` then runs every call with a required
output, filling missing cotangents with `zeros()`. This is live code -- the
zero cotangents are added into the real gradient -- so XLA's own DCE cannot
remove it, and it does not fold `mul(x, 0)` for floats (`inf * 0` is `NaN`).
The fill for plain primitives predates `prim_if()`'s rule, but for an `if` it
is much costlier: the forward `if` saves residuals and a whole backward `if`
runs on zero cotangents.

Examples:

- a predicate computed from `x`: `nv_if(sum(exp(x)) > 1, \() sum(x), \() sum(x * 2))`
  -- `exp(x)` gets a zero cotangent (`mul(broadcast(0), exp(x))`);
- an `if` whose output only feeds another `if`'s predicate:
  `s <- nv_if(p, \() sum(sin(x) * x), \() sum(x)); nv_if(s > 0, ...)`;
- a float output of an `if` that is never used:
  `nv_if(p, \() list(a = sum(x * 2), b = sum(sin(x) * x)), ...)$a`;
- the discarded value of a `gradient()` called inside a branch.

Fix: restrict the requirements to what can reach the loss before the forward
replay. The last example needs per-output information in the `if` rule --
which of its outputs are live -- so that it saves residuals and builds a
backward only for those: either a `required_out` argument for forward rules,
or a DCE pass that prunes the `if`'s dead outputs (and the branch code
computing them) before it is differentiated.

### 2. Residual slots are not shared between the branches of an `if`

`prim_if()`'s reverse rule (`R/rules-reverse.R`) gives each branch its own
residual slots and zero-pads the other branch's. For
`nv_if(p, \() sum(exp(x) * x), \() sum(log(x) * x))` with `x` of length 1000,
the forward `if` returns two `f64[1000]` residuals and fills 1000 zeros in each
branch, where one shared slot would do. JAX merges residuals of equal aval
(`_merge_branch_residuals`, then `_join_cond_outputs` fills the other
branches' slots).

### 3. A residual that is also a branch output is returned twice

`split_vjp()` (`R/reverse.R`) does not check residuals against the branch's
outputs, e.g. `s <- exp(x); list(a = s, b = sum(s * x))` gives
`return (%8, %10, %8)`, and the other branch zero-fills the duplicate. The
backward pass could read the `if`'s own output instead, as it already reads the
branch inputs from the `if`'s operands.

### Minor (XLA removes it, or it costs nothing)

- Zero cotangents are built for integer and RNG-state outputs and passed to
  the backward `if`.
- The backward `if` receives every branch input, including ones neither
  backward branch reads.
- `optimize_graph()` has no dead-code pass; unneeded forward work stays in the
  graph for XLA to remove.
- Duplicated work XLA's CSE merges: two `broadcast(1)` seeds, `mul(ct, x)`
  twice from the rule of `x * x`.
- `cos(x)` is recomputed in the backward branch from `x` instead of saved: the
  residuals are whatever the reverse rules read.
