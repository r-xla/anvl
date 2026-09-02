# Review: "R values have no data type" (branch `pr-434`)

Second pass, over `main..pr-434` plus the uncommitted working tree.

## Summary

Every finding from the first review round is resolved or closed, and I re-verified each
one rather than taking the status table at its word: `prim_convert()` now builds a bare
literal at the target dtype in both modes, the four bind/cross-product functions carry
`promote_common()`, the `pjrt` floor is `0.5.0.9000`, the REVIEW/RESPONSE markers are
gone, the category errors name the argument, `finalize_rdata_inputs()`'s widening branch
has a whole-program test, quickr's NaN/Inf paths are tested, `gradient()` now *refuses* a
value with no data type as its differentiated argument (with tests, moved into
`test-reverse.R`), and `devtools::document()` reproduces `NAMESPACE` byte-for-byte, so the
grouped `importFrom()` blocks really are roxygen's own output.

The RData machinery itself held up under adversarial probing: `prim_if` branches at two
data types, `prim_while` condition-and-body use, R array arguments, an argument used at
`f32` and `f64` at once (uploaded at `f64`, converted down), an unused R argument, nested
`jit()`, `gradient()` with an open R value outside `wrt`, cache hits on the same R type and
shape with different values, and the `f16`/`bf16` → `f32` widening upload all behave as the
design says.

What stands in the way of merging is a ring of things around the code rather than in it:
the type-promotion vignette and two `?promotion_rule` examples do not run, one new test
asserts the wrong number, and the suite is red for a sixth reason that is not this
branch's fault but has to be settled before CI can be green.

## Blocking

1. **`vignettes/type-promotion.Rmd` does not build: two chunks error.**
   - `vignettes/type-promotion.Rmd:101` — `as_anvl_arrays(1, 2, .promote = promote_like("f64"))`.
     `promote_like(arg)` takes an argument *name or position*, not a data type, so this
     aborts with ``` `arg` names an argument this call does not have: "f64" ```. It wants
     `promote_dtype("f64")` (or a named call, `as_anvl_arrays(x = ..., .promote = promote_like("x"))`).
   - `vignettes/type-promotion.Rmd:118` — `as_anvl_arrays(args, promote_fn)`. `args` is a
     `list()` and `.promote` is never named, so `align_arrayish()` aborts with
     "Expected arrayish input, but got `<list>`". It wants
     `rlang::exec(as_anvl_arrays, !!!args, .promote = promote_fn)`.
     The prose above it (line 115) also says "as_anvl_arrys()".

   Neither chunk sets `error = TRUE`, so `R CMD build` and pkgdown halt there. This is easy
   to miss because a bare `knitr::knit()` defaults to `error = TRUE` and renders the
   failures inline — `rmarkdown::render("vignettes/type-promotion.Rmd")` is the check that
   catches it. I evaluated the chunks of all 17 vignettes; these two are the only failures.

2. **Two `@examples` in `?promotion_rule` error, so `R CMD check` fails.**
   - `R/promotion.R:81` (`man/promotion_rule.Rd:97`) —
     `print(promote_like("x")(list(...), silent = TRUE))`. A rule's formals are `(args)`, so
     this is `unused argument (silent = TRUE)`; the wrapping `try(` was lost. Restore it:
     `print(try(promote_like("x")(list(x = nv_scalar(1, "f32"), nv_scalar(1, "f64"))), silent = TRUE))`.
   - `R/promotion.R:219` (the `promotion_rule()` "widest float" example) — it calls
     `is_dtype_float()` and `dtype_width()`, which anvl imports from tengen but neither
     exports nor re-exports. An example runs with only anvl attached, so it dies with
     "could not find function \"is_dtype_float\"". Qualify them (`tengen::`) or re-export
     the two predicates the vignettes and this example lean on.

   `devtools::run_examples()` is clean once these two are fixed; nothing else in `man/`
   fails.

3. **`tests/testthat/test-rdata.R:337` asserts a number R does not compute.**
   ```r
   expect_identical(as_array(f(sqrt(2), nv_scalar(0, dtype = "f64"))$s), sum(rep(sqrt(2), 8)))
   ```
   The loop adds `sqrt(2)` to an `f64` accumulator eight times in sequence
   (`11.313708498984762940`); R's `sum()` accumulates in long double and rounds once
   (`11.313708498984761164`). They differ in the last bit, so `expect_identical()` fails
   deterministically. Use the sequential value — ``Reduce(`+`, rep(sqrt(2), 8), 0)`` — which
   is what the program actually does and keeps the bit-exactness the test is there for.

4. **The suite is also red for a reason outside this branch: the installed PJRT CPU plugin
   miscompiles some 4-lane `f32` programs.** Six of the seven failures
   (`test-api-distributions.R:208,217,228,321,326` for `nv_qnorm`, `test-api-generics.R:209`
   for `lgamma`) are this. It reproduces with no anvl in the picture:
   ```r
   src <- "func.func @main (%0: tensor<4xf32>) -> tensor<4xf32> {
   %1 = \"chlo.lgamma\" (%0) : (tensor<4xf32>) -> (tensor<4xf32>)
   return %1 : tensor<4xf32>
   }"
   # lane 3 comes back -1.872209 where lgamma(2) is 0; length 5 and f64 are fine
   ```
   and the `nv_qnorm` graph this branch emits gives the same wrong lanes when compiled from
   `main`'s source tree. So the branch does not cause it — but the `nv_qnorm` refactor
   (`promote_like("p")` in place of canonicalize-then-convert) changed the emitted program
   enough to unmask it, and `devtools::test()` cannot be green until the plugin is fixed or
   pinned. Confirm on CI's plugin before merging, and decide whether `nv_qnorm`'s `f32`
   tests need a tolerance or a skip in the meantime.

## Should fix

5. **`nv_pad()` applies the primitive-level rule at the API layer, and says nothing about it.**
   `R/api.R:1459` uses `promote_rdata_common()`, so `nv_pad(x_f32, 0L)` is an error
   ("`padding_value` is an R integer, which cannot be used at the \"f32\" data type here"),
   while its neighbours promote across categories — `nv_clamp(0L, x_f32, 1L)` and
   `nv_ifelse(pred, 1L, x_f32)` both give `f32`. AGENTS.md puts category crossing at the
   `nv_*` layer, and `@param padding_value` (`R/api.R:1442`) now reads only "Scalar value to
   use for padding.", so nothing warns the user. Either use `promote_like("x")` at the
   `nv_*` level and leave `prim_pad()` strict, or state the restriction the way
   `nv_solve()` does ("Must have the same data type as `a`.").

6. **`NEWS.md` contradicts itself about `new_primitive()`.**
   `NEWS.md:27-32` says "the `promote` argument of `new_primitive()` defaults to `NULL`, and
   the primitives whose arrayish arguments must agree declare
   `promote = promote_rdata_common()`"; `NEWS.md:51-57` says "`new_primitive()` and
   `AnvlPrimitive()` lost their `promote` argument". The second is what the code does — the
   first describes an intermediate design that `apply_promotion()` replaced. Drop it.

7. **`NEWS.md` does not mention two removed exports.**
   `jit_eval()` is gone (`R/jit.R`, `man/jit_eval.Rd` deleted) and appears nowhere in NEWS.
   The `ambiguous()` generic is gone too, along with the `ambiguous` argument of
   `nv_array()`, `nv_scalar()`, `nv_matrix()`, `nv_empty()`, `nv_aval()`, `AbstractArray()`,
   `nv_fill()`, `nv_iota()`, `nv_seq()` and every `_like()` variant; the only trace in NEWS
   is "The `ambiguity` concept no longer exists in the type system" inside a type-system
   bullet. Both are signature-breaking for anyone with existing code, and NEWS is where they
   would look.

8. **`.claude/skills/add-api-function/SKILL.md` still teaches the removed argument.**
   Line 78 tells authors the `_like` variants "default `dtype`, `shape`, `ambiguous`, and
   `device` from `x`", and line 132 lists `param_ambiguous` among the templates to use —
   but `man-roxygen/param_ambiguous.R` was deleted in this branch. A new API function
   written from this skill will not document.

## Nice to have

9. **The working tree's `param_while_init` edit has not been re-documented.**
   `man-roxygen/param_while_init.R` was shortened (uncommitted), but `man/nv_while.Rd:11-17`
   and `man/prim_while.Rd:11-17` still carry the old paragraph. Run `devtools::document()`
   before committing. The new text also has a typo: "commited" → "committed".

10. **`test-rdata.R` and `test-primitive-rdata.R` are feature-named with no `R/rdata.R`.**
    The project rule is to either move such tests into the files matching the sources they
    exercise (here `test-graph.R`, `test-array.R`, `test-promotion.R`, `test-primitives.R`)
    or give the feature its own `R/<name>.R` and name the test file after it. The gradient
    half of this already made that move (`test-rdata-gradient.R` → `test-reverse.R`), so
    it is the same call applied to the two files that remain. Given how much of the concept
    lives in `graph.R` (materialize/finalize/resolve-upload) and `promotion.R`, extracting
    `R/rdata.R` is probably the cheaper of the two.

11. **Housekeeping before the PR.** The branch carries uncommitted work — the AGENTS.md
    rewrite, `CLAUDE.md` reduced to `@AGENTS.md` (the `@../claude-config/CLAUDE.md` include
    correctly moved to AGENTS.md's first line), the `param_while_init` edit and the
    `test-rdata-gradient.R` → `test-reverse.R` move — plus an untracked `assets/`
    directory. `jarl check .`, `air format --check`, `pkgdown::check_pkgdown()` and
    `devtools::document()` are all clean on the committed tree.
