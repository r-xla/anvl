# Review: "R values have no data type" (branch `pr-434`)

Second pass, over `main..pr-434`. The first round's findings are all resolved; this file
tracks the second round.

## Summary

Every finding from the first review round is resolved or closed, re-verified rather than
taken from the status table: `prim_convert()` builds a bare literal at the target dtype in
both modes, the four bind/cross-product functions carry `promote_common()`, the `pjrt`
floor is `0.5.0.9000`, the REVIEW/RESPONSE markers are gone, the category errors name the
argument, `finalize_rdata_inputs()`'s widening branch has a whole-program test, quickr's
NaN/Inf paths are tested, `gradient()` refuses a value with no data type as its
differentiated argument, and `devtools::document()` reproduces `NAMESPACE` byte-for-byte,
so the grouped `importFrom()` blocks really are roxygen's own output.

The RData machinery held up under adversarial probing: `prim_if` branches at two data
types, `prim_while` condition-and-body use, R array arguments, an argument used at `f32`
and `f64` at once (uploaded at `f64`, converted down), an unused R argument, nested
`jit()`, `gradient()` with an open R value outside `wrt`, cache hits on the same R type and
shape with different values, and the `f16`/`bf16` → `f32` widening upload all behave as the
design says.

## Addressed in this round

| # | Finding | Status |
|---|---------|--------|
| 1 | `vignettes/type-promotion.Rmd` chunks error | fixed |
| 2 | two `?promotion_rule` examples error | fixed |
| 3 | `test-rdata.R:337` bit-exact `sum()` expectation | **left as is** -- see below |
| 4 | PJRT CPU plugin miscompiles 4-lane `f32` | **not addressed** -- environment |
| 5 | `nv_pad()` used the primitive-level rule | fixed |
| 6 | `NEWS.md` contradicted itself on `new_primitive()` | fixed by the author |
| 7 | `NEWS.md` omitted `jit_eval()` / `ambiguous` removals | fixed by the author |
| 8 | `add-api-function` skill taught the removed argument | fixed |
| 9 | `param_while_init` edit not re-documented | fixed |
| 10 | `test-rdata.R` had no `R/rdata.R` counterpart | fixed |
| 11 | uncommitted working tree | committed by the author |

### 1. Vignette

`vignettes/type-promotion.Rmd:101` passed a data type to `promote_like()`, which takes an
argument *name or position* — now `promote_dtype("f64")`. Line 118 passed a `list()`
positionally and never named `.promote` — now
`do.call(as_anvl_arrays, c(args, list(.promote = promote_fn)))`. The "as_anvl_arrys" typo
above it is fixed too. Both chunks lacked `error = TRUE`, so `R CMD build` and pkgdown
halted there; a bare `knitr::knit()` hides this because its standalone default *is*
`error = TRUE`. All chunks of the vignette now evaluate cleanly.

### 2. Examples

`R/promotion.R:81` lost its wrapping `try()`, leaving `silent = TRUE` as an unused
argument to a rule whose formals are `(args)`; restored, with a line saying what it
demonstrates. The `promotion_rule()` "widest float" example called `is_dtype_float()` and
`dtype_width()`, which anvl imports from tengen but does not re-export, so it died under
`library(anvl)` alone; both are now `tengen::`-qualified. `devtools::run_examples()` is
clean.

### 5. `nv_pad()`

`R/api.R` now uses `promote_like("x")` instead of `promote_rdata_common()`, so the padding
value is *promoted* to `x`'s data type the way `nv_clamp()`'s bounds are —
`nv_pad(x_f32, 0L)` works, and an `f32` padding value for an `f64` array is converted. A
double still does not become an integer, and narrowing (`f64` value into an `f32` array) is
still an error rather than silent. `prim_pad()` keeps `promote_rdata_common()`: agreeing,
not promoting, is the primitive's contract. `@param padding_value` spells the contract out,
and the tests in `test-rdata.R` and `test-promotion-rules.R` were updated to the new
behaviour (the latter's diagnostic example moved to `nv_pad(x_i32, 0)`, which still fails
and still names `padding_value`).

### 10. `R/rdata.R`

The RData layer moved out of `R/graph.R` into `R/rdata.R` (540 lines): the `RData` aval and
its extractors, the `dtype()`/`shape()` methods for bare R values, the per-descriptor
materialization memo, `build_r_staged()` / `build_r_at()` / `materialize_rdata()`,
`peek_dtype()`, `commit_rdata_box()`, the input-slot machinery
(`register_rdata_input()` … `finalize_inline_rdata_inputs()`), `resolve_upload_dtype()` and
its capacity helpers, and `graph_input_dtypes()`. `test-primitive-rdata.R` merged into
`test-rdata.R` as the `describe("a primitive's operands")` block, so the feature now has one
source file and one test file named after it. Note that the collate roclet is not enabled
(`Roxygen: list(markdown = TRUE, roclets = c("namespace", "rd", "anvl::jit_roclet"))`), so
`DESCRIPTION`'s `Collate:` needed the new file added by hand.

## Still open

### 3. `test-rdata.R:337` is architecture-dependent, not container-dependent

```r
expect_identical(as_array(f(sqrt(2), nv_scalar(0, dtype = "f64"))$s), sum(rep(sqrt(2), 8)))
```

Left as the author asked, but the cause is worth recording, because it is not the sandbox:
R's `sum()` accumulates in `LDOUBLE`. Where `long double` is wider than `double` — x86-64
Linux and Intel macOS, 80-bit — it rounds once at the end and gives
`11.313708498984761164`, while the `f64` while-loop adds sequentially and gives
`11.313708498984762940`. Where `long double` *is* `double` — arm64 macOS — the two agree,
which is why it passes locally. So this will fail on any x86-64 CI runner.
``Reduce(`+`, rep(sqrt(2), 8), 0)`` is the same value on both and keeps the bit-exactness
the test is there for.

### 4. The PJRT CPU plugin miscompiles some 4-lane `f32` programs

Six failures (`test-api-distributions.R:208,217,228,321,326` for `nv_qnorm`,
`test-api-generics.R:209` for `lgamma`). Not this branch: it reproduces with no anvl in the
picture —

```r
src <- "func.func @main (%0: tensor<4xf32>) -> tensor<4xf32> {
%1 = \"chlo.lgamma\" (%0) : (tensor<4xf32>) -> (tensor<4xf32>)
return %1 : tensor<4xf32>
}"
# lane 3 comes back -1.872209 where lgamma(2) is 0; length 5 and f64 are fine
```

— and the `nv_qnorm` graph this branch emits gives the same wrong lanes when compiled from
`main`'s source tree. Recorded so the failures are not mistaken for a regression if they
show up on another machine.

## Verification

Run after the changes above:

- `devtools::test()` — the seven failures above and nothing else; no regressions from the
  `R/rdata.R` split. Quickr tests pass with `ANVL_SKIP_QUICKR` unset.
- `devtools::run_examples()` — clean.
- All 17 vignettes' chunks evaluate; `type-promotion.Rmd` has no failing chunk.
- `devtools::document()`, `pkgdown::check_pkgdown()`, `jarl check .`, `air format --check` —
  all clean.
