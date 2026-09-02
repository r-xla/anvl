# Review: "R values have no data type" (branch `pr-434`)

## Summary

This is a well-executed implementation of a genuinely hard design. The core machinery
(`RData`/`GraphRData`/`RDataInput`, `materialize_rdata()` with its per-descriptor memo
reachability check, `finalize_rdata_inputs()` with pre-call converts, `resolve_upload_dtype()`)
is correct in every scenario I traced, including the tricky ones: sibling sub-graphs,
multi-dtype argument leaves, out-of-category converts, the f16/bf16 widening case, and
value-independent caching. The promotion-rule layer is a clean single mechanism shared by
`as_anvl_arrays()` and the primitives, and the test suite is unusually strong — bit-exact
assertions, eager/traced equivalence sweeps, and pinned wrap-around semantics.

Three themes need attention: (1) `prim_convert` breaks the "one answer in both modes"
guarantee for an in-body literal — the one primitive whose whole point is naming a dtype;
(2) a handful of `nv_*` functions still canonicalize without a rule and promote downstream,
the exact anti-pattern the PR's own AGENTS.md forbids; (3) merge hygiene — unfinished
REVIEW/RESPONSE markers, a hand-edited NAMESPACE, and a stale pjrt version floor.

## Status

Addressed on this branch, except where noted:

| # | Finding | Status |
|---|---|---|
| 1 | `prim_convert` on an in-body R literal | fixed + test |
| 2 | REVIEW/RESPONSE markers | **left to the author** |
| 3 | `pjrt` version floor | bumped to `0.5.0.9000` |
| 4 | bind / cross products promote downstream | fixed + test |
| 5 | hand-edited `NAMESPACE` | **not a defect** -- `document()` reproduces the grouped `importFrom()` blocks unchanged |
| 6 | `padding_value` docs | `nv_pad()` now yields instead of promoting; docs and NEWS updated |
| 7 | errors do not name the argument | fixed + test; the unactionable `force = TRUE` hint dropped |
| 8 | widening-upload path untested | tested at the graph level (this backend has no 16-bit float buffers to run one) |
| 9 | quickr NaN/Inf literals untested | tested against quickr |
| 10 | quickr's `f64` eager default | **ignored, by the author's decision** |
| 11 | `jit(gradient(f))` commits at the inline boundary | fixed + tests (`R/graph.R`, `tests/testthat/test-rdata-gradient.R`) |

## Blocking

1. **`prim_convert` on an in-body R literal rounds through the default dtype (eager ≠ traced).**
   `R/primitives.R:2333-2339`. The body handles `is_rdata_box(x)`, but inside a trace a bare
   literal arrives *unboxed* (`promote = NULL`, so no wrapper; jit's tracing passthrough calls
   the body with raw arguments). It falls through to `graph_desc_add()`, whose last resort
   (`R/graph.R:1181`) commits it at the default and then records a convert — so
   `jit(\(x) x * prim_convert(sqrt(2), "f64"))` builds `(f64)(f32)sqrt(2)`, while the same call
   eagerly is exact (the literal is an argument leaf, boxed as `RData`). This contradicts the
   body's own comment ("the result holds every digit the R value had"), AGENTS.md's "To hold a
   value open ... `nv_convert()` and the primitives take it as it is", and the spec's
   "one answer in both modes" guarantee. Fix: box a bare R value before the check, e.g.
   ```r
   if (currently_tracing() && has_no_dtype(x)) x <- maybe_box_arrayish(x)
   ```
   (or route the no-dtype case through `realize_at(x, dtype)` as `nv_convert()` does), and add
   the missing test: `jit(\() prim_convert(sqrt(2), "f64"))()` must equal `sqrt(2)` exactly.
   No other primitive special-cases `is_rdata_box`, so this is a one-primitive fix.

2. **Unfinished review-dialogue markers shipped in source and vignette.**
   `R/array.R:982-991`, `R/array.R:1033-1043`, `R/array.R:1062-1066` (three
   `# REVIEW:` / `# RESPONSE:` exchanges), and `vignettes/extending_api.Rmd:144-145, 210-211`
   (`<!-- REVIEW: ... I should write this myself --> <!-- RESPONSE: left for you to write -->`).
   The vignette ones explicitly mark prose the author still intends to rewrite. Before merge,
   distill the useful content of the code RESPONSEs into ordinary comments (the RDataInput
   rationale at 985-991 is worth keeping in some form — much of it is already in the roxygen
   above it) and delete the markers; finish or accept the two vignette paragraphs.

## Should fix

3. **`DESCRIPTION` declares `pjrt (>= 0.5.0)`, but the PR requires the unreleased dispatcher contract.**
   `DESCRIPTION:32`. `avals_from_dispatch()` reads `av$kind` (`R/jit.R:330`) and
   `jit_pjrt_compile_cb()` returns `input_dtypes` that the dispatcher must honor
   (`R/backend-pjrt.R:36`, `R/backend-pjrt.R:248`); pjrt 0.5.0 supplies neither, so every
   jitted call fails with released pjrt. The sibling pjrt is at 0.5.0.9000 with the `kind`
   support on a feature branch. Bump the floor to the version that ships the new dispatcher
   (and check `stablehlo (>= 0.4.0)` still suffices) as part of landing the cross-package change.

4. **`nv_rbind()`, `nv_cbind()`, `nv_crossprod()`, `nv_tcrossprod()` canonicalize without a rule, then promote downstream.**
   `R/api.R:433`, `R/api.R:443`, `R/api.R:2779`, `R/api.R:2805`. Each calls
   `as_anvl_arrays(...)` with no `.promote`, committing any R value at its default, and then
   the downstream `nv_concatenate()` / `nv_matmul()` promotes the *committed* array with a
   convert. `nv_rbind(x_f64, matrix(c(0.1, 0.2), 1))` therefore rounds the R doubles through
   `f32` on the way to `f64` — exactly the "canonicalize first and convert afterwards" pattern
   AGENTS.md forbids for functions whose result dtype depends on their arguments. Fix: pass
   `.promote = promote_common()` in these four (matching R's own `rbind`/`crossprod`
   promotion), or drop the early canonicalization and let the promoting callee handle it
   (`shape()` already answers for bare R values, so the shape logic does not need arrays).
   Worth a quick audit for any other multi-array `nv_*` that combines its canonicalized
   arguments downstream.

5. **`NAMESPACE` appears hand-edited and will churn on the next `document()`.**
   `NAMESPACE:472-618`: the new grouped multi-line `importFrom(pjrt, await, build_tree, ...)`
   entries are not a format roxygen2 emits (it writes one directive per line, as the untouched
   `importFrom(stablehlo,hlo_abs)` block still shows, and the roxygen sources still use plain
   `@importFrom` tags, e.g. `R/aaa.R:25`). The project rule is that NAMESPACE is generated;
   the next `devtools::document()` will rewrite all of this. Regenerate and commit the
   roxygen-produced form.

6. **`padding_value` docs contradict the new behavior.**
   `R/api.R:1440` and `R/primitives.R:2221` both still say "Must have the same dtype as `x`."
   `nv_pad()` now uses `promote_like("x")` — a promotable typed value is converted, an R value
   is built at `x`'s dtype (within its category) — and `prim_pad()`'s `promote_rdata_common()` accepts
   an R value of `x`'s category. Reword both `@param` entries (the nv-level one should describe
   the promote-like contract, the prim-level one the rdata-common/category contract) and re-document.

## Nice to have

7. **Category-rule errors do not name the offending argument, and the `force = TRUE` hint is unactionable for end users.**
   `R/promotion.R:384-391` and `R/promotion.R:400-407`: the spec's error examples name the
   argument ("`rhs` is an R double...", "`lhs` is an R double and `rhs` an R integer"); the
   implemented messages say "An R double cannot be used at..." / "The R values here have no
   data type to agree on" with no argument name, which hurts exactly the multi-operand calls
   (`prim_pad(x, 0)`, `prim_clamp(0, x, 1)`) the design set out to diagnose well. The
   positions/names are available in `resolve_rdata_common()`. Separately, `assert_promotes_to()`
   (`R/promotion.R:302`, `R/promotion.R:314`) tells the user to "ask for the conversion with
   `force = TRUE`", but `force` is an argument of the *rule*, not of `nv_clamp()` /
   `[<-` — a user hitting `x_i32[2] <- 1.5` cannot pass it. Drop that bullet from the
   user-facing path or phrase it for function authors.

8. **No end-to-end test for the widening-upload path.**
   `R/graph.R:751-763`: when `resolve_upload_dtype()` returns a dtype *not* in the memo
   (double leaf used at `f16` and `bf16` → upload `f32`), `finalize_rdata_inputs()` creates a
   fresh input gval and feeds *both* use sites via pre-calls. Only the unit test of
   `resolve_upload_dtype()` covers this (`tests/testthat/test-rdata.R:482`); a whole-program
   test (one argument used at `f16` and `bf16`, assert both values and that the graph's input
   is `f32` with two converts) would pin the only finalize branch that currently runs untested.

9. **The new quickr NaN/Inf literal paths are untested.**
   `R/rules-quickr.R:898-906` (bound-literal lookup in `quickr_expr_of_node()`),
   `R/rules-quickr.R:1490-1505` (binding special-float call inputs to temps), and the
   `prim_fill` special case (`R/rules-quickr.R:1575-1586`) were added in this PR but no quickr
   test exercises a NaN/Inf literal (e.g. `nan_rm = TRUE` reductions, or `nv_fill(Inf, ...)`
   under `local_backend("quickr")` with `skip_if_no_quickr()`).

## Low confidence / needs the author's judgement

10. **The quickr backend's rule-less default is `f64`, contradicting the new mode-equivalence prose.**
    `R/backend-quickr.R:160` converts a bare R double at `f64`, while `default_dtype_r()`
    (`R/promotion.R:548`) — what a trace commits at — is fixed `f32`. So eager
    `as_anvl_array(1.5)` under `local_backend("quickr")` is `f64` where the traced commitment
    is `f32`, which sits awkwardly next to the new `as_anvl_array()` docs ("the same conversion
    while tracing and in eager mode ... its default (`f32` for a double)", `R/array.R:204-213`).
    In practice quickr re-derives output dtypes from R storage, which masks it, and this
    predates the PR — but the PR's docs now state a guarantee the quickr backend does not meet.
    Either scope the doc claim, or fold the backend float default into `default_dtype_r()`
    (the spec's "intended to become configurable later").

11. **`jit(gradient(f))` commits a bare R argument at the default at the inline boundary.**
    `R/graph.R:655-657` (`mode = "inline"` calls `trace_commit_rdata_box()`): the value
    commits to `f32` before `f`'s body can name a dtype, so
    `jit(gradient(\(v) v * nv_scalar(1, dtype = "f64")))(sqrt(2))` rounds `v` through `f32`,
    where the same body without `gradient()` is exact. The existing test
    (`tests/testthat/test-rdata.R:73`) uses `2`, which is exact at `f32`, so the rounding is
    invisible. This may be an accepted boundary (like the documented transitive case), but it
    is neither documented nor tested as such — worth a deliberate decision and, if accepted,
    a sentence in the `RData` docs.
