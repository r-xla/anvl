# Control flow and its differentiability: verification results

Script: `control_flow.R` (same directory). Run on 2026-10-22: `Rscript control_flow.R`, all
sections ran (exit 0). Versions: anvl dev source (`pkgload::load_all()`), torch 0.17.0,
tensorflow 2.20.0 (R) with TF 2.15 / Python 3.10 and tfautograph, greta 0.5.1, quickr 0.3.0.9000,
ast2ast 1.0.2, odin2 0.3.46 / dust2 0.3.28, cmdstanr 0.9.0 / CmdStan 2.39.0, TMB 1.9.19, RTMB 1.8,
nimble 1.3.0.

## Method

Each construct is compiled / traced / taped once and evaluated at two inputs that take different
paths, so freezing the path at tracing or taping time shows up as a wrong result at the second
input:

| construct | code | inputs | values | d/dx |
|---|---|---|---|---|
| (a) if | `if (x > 0) x^2 else -x` | 2, -2 | 4, 2 | 4, -1 |
| (b) while | `while (x < 10) x <- 2 * x` | 2, 3 | 16, 12 | 8, 4 |
| (c) loop | `for (i in 1:3) x <- x * 1.1` | 2, -1 | 2.662, -1.331 | 1.331, 1.331 |
| all | if(while(loop(x))) | 2, 0.5 | 113.36, 113.38 | 113.38, 453.52 |

The reference derivatives are finite differences of the plain R functions. `ok` means correct at
both inputs; `ok*` means correct but with a caveat; `--` means an error or a wrong result.

## Results table (best variant per cell; the script prints every attempt)

```
 pkg                      if  while loop grad if grad while grad loop grad all
 anvl                     ok  ok    ok   ok      --         ok*       --
 torch (eager)            ok  ok    ok   ok      ok         ok        ok
 torch (TorchScript)      ok  ok    ok   ok      ok         ok        ok
 torch (jit_trace)        --  --
 tensorflow (tf ops)      ok  ok    ok   ok      ok         ok        ok
 tensorflow (tf ops, XLA) ok  ok    ok   ok      ok*        ok        ok*
 tensorflow (autograph)   ok  ok    ok   ok      ok         ok        ok
 greta                    ok* --    ok*  --      --         ok*       --
 quickr                   ok  ok    ok   n/a     n/a        n/a       n/a
 ast2ast                  ok  ok    ok   ok      ok         ok        ok
 odin2/dust2              ok  --    ok*  ok      --         ok*       --
 Stan                     ok  ok    ok   ok      ok         ok        ok
 TMB                      ok* --    ok*  ok*     --         ok*       --
 RTMB                     ok* --    ok*  ok*     --         ok*       --
 nimble                   ok  ok    ok   ok*     ok*        ok        ok*
```

## Suggested comparison-table rows

**Data-dependent control flow** (branches and loops whose path depends on run-time values, inside
the compiled or taped program):

| package | cell |
|---|---|
| anvl | ✓ `nv_if()`, `nv_while()`, `nv_scan()` (functional; R `if`/`while` on traced values error, R `for` is unrolled) |
| torch | ✓ R control flow (eager); TorchScript via `jit_compile()` (`jit_trace()` freezes the path) |
| tensorflow | ✓ `tf$cond()`, `tf$while_loop()`, `tf$scan()`; R `if`/`while`/`for` via autograph (tfautograph) |
| greta | – (0/1 masks only; R loops are unrolled when the model is built) |
| quickr | ✓ |
| ast2ast | ✓ |
| odin2/dust2 | (✓) `if` expressions only; no loop statements (array equations, time steps) |
| Stan | ✓ |
| TMB | (✓) branches only with `CondExp*()`; C++ `if`/`while` are frozen at taping |
| RTMB | (✓) branches only as taped 0/1 masks; R `if`/`while` on AD values error; loops are unrolled |
| nimble | ✓ (in `nimbleFunction`s; not in model code) |

**Differentiable control flow:**

| package | cell |
|---|---|
| anvl | (✓) `nv_if()` and unrolled loops; not `nv_while()` / `nv_scan()` |
| torch | ✓ |
| tensorflow | ✓ (with XLA, `while_loop` needs `maximum_iterations`) |
| greta | (✓) unrolled loops only; gradients internal to the samplers |
| quickr | – (no AD) |
| ast2ast | ✓ (forward and reverse) |
| odin2/dust2 | (✓) `if` expressions and time steps; likelihood gradient only |
| Stan | ✓ |
| TMB | (✓) `CondExp*()` and fixed loops; other control flow gives silently wrong gradients |
| RTMB | (✓) masks with `TapeConfig(comparison = "tape")` and fixed loops; no while |
| nimble | (✓) data-dependent paths need `derivs(..., reset = TRUE)`; otherwise the gradients are silently wrong |

If only one row fits, merge them: "✓ / ✓ grad" style, e.g. anvl "✓ (grad: if only)", TMB "frozen
at taping (CondExp)", nimble "✓ (grad: reset = TRUE)".

## Evidence per package

### anvl
- `jit(function(x) nv_if(x > 0, \() x * x, \() -x))` is correct at 2 and -2. `nv_while()` gives
  16 and 12. `nv_scan(steps = 3L)` is correct and lowers to a stablehlo `while` with a static
  trip count, so it is not unrolled.
- `gradient()` through `nv_if()` is correct (4, -1).
- Through `nv_while()`: `No reverse rule for primitive while. ℹ Cannot compute gradient through
  this primitive.`
- Through `nv_scan()`: `No reverse rule for primitive scan.` (The `nv_while` roxygen section
  "Gradients" says this is by design for while.)
- A plain R `for` inside `jit()` is unrolled at trace time and differentiates fine. So does
  `nv_if()` composed with an unrolled loop: (7.086, -1.331).
- Plain R `if` / `while` on a traced value inside `jit()` errors with `the condition has length >
  1`. The same happens eagerly on an `AnvlArray`. Eagerly, `if (as_array(x > 0))` works, but that
  is not compiled control flow.

### torch
- Eager: R `if`/`while`/`for` with the condition read back to R via `$item()`. Autograd records the
  path actually taken, so every gradient is correct. `if (x > 0)` straight on a tensor errors with
  `argument is not interpretable as logical`, so `$item()` (or `as.logical()`) is needed.
- `jit_compile()` (TorchScript source, a Python subset, not R): `if bool(x > 0)`, `while`,
  `for i in range(3)` are all correct, and so are their gradients.
- `jit_trace()` of the eager R function, traced at x = 2, freezes the path: `if` gives 4, 4
  (expected 4, 2) and `while` gives 16, 24 (expected 16, 12).

### tensorflow
- `tf$cond()`, `tf$while_loop()`, `tf$scan()` in `tf_function()`, with `GradientTape` inside the
  function: all values and gradients are correct.
- With `jit_compile = TRUE` (XLA), forward is fine and grad through `if`/`scan` is fine. Grad
  through `while_loop` errors with `InvalidArgumentError: XLA compilation requires a fixed tensor
  list size. Set the max number of elements. ... setting maximum_iteration to a suitable value.`
  With `maximum_iterations = 100L` it is correct.
- **Autograph is available from R.** `tf_function(autograph = TRUE)` (the default) uses the R
  package tfautograph and translates plain R `if`, `while`, and `for` over `tf$range()` into graph
  control flow. All values and gradients are correct. With `autograph = FALSE`, R `if` on a
  symbolic tensor errors with `RuntimeError: the condition has length > 1`.

### greta
- R `if` / `while` on a greta array error with `argument is not interpretable as logical`.
  `base::ifelse()` errors with `argument to 'which' is not logical`.
- Comparisons return 0/1 greta arrays, so `(x > 0) * x^2 + (x <= 0) * (-x)` works (`calculate()` at
  2, -2 gives 4, 2). Both sides are evaluated.
- R `for` is unrolled while the graph is built. There is no user-level gradient API (see
  `automatic_differentiation.R`), so "grad" for the unrolled loop holds only in the sense that HMC
  gradients flow.

### quickr
- `if`, `while`, `for` in `quick()` compile to Fortran and are all correct. quickr has no AD.

### ast2ast
- `if`, `while`, `for` in `translate()` are correct.
- Reverse mode (`deriv(y, x)`, `derivative = "reverse"`) and forward mode (`seed`/`get_dot`) give
  correct gradients through all three constructs and their composition, at both inputs.
  ast2ast's AD runs at run time, so nothing is frozen.

### odin2 / dust2
- `update(x) <- if (a > 0) a * a else -a` is correct at both inputs. The adjoint likelihood
  gradient (`dust_likelihood_run(adjoint = TRUE)`) through it is correct.
- No loop statements. `while`/`for` at top level fail with `Unclassifiable expression ℹ Expected an
  assignment (with '<-') or a relationship (with '~')`. The `errors` vignette lists `while`/`for`
  as reserved words (E1045). Loops exist only as implicit array equations (`y[] <- ...`) and as
  time steps.
- A 3-step `update(x) <- x * 1.1` gives correct values and adjoint gradients.
- Test-design note: `obs ~ Normal(x, 1)` with an integer literal standard deviation failed C++
  compilation (`no matching function for call to 'normal'`, deduced `double` vs `int`). The
  workaround is `sd_obs <- parameter(1)`. This looks like an odin2 code-generation bug, not a
  control-flow limit.

### Stan
- Stan `if`, `while` (on a parameter), and `for` are correct via `log_prob()`. `grad_log_prob()` is
  correct for all constructs and the composition. Stan Math re-records the expression graph on
  every evaluation.

### TMB
- `MakeADFun()` tapes once at the initial parameter value, and `obj$fn`/`obj$gr` replay that tape.
  C++ `if (y > Type(0))` compiles but is frozen: 4, 4 (expected 4, 2), gradient 4, -4 (expected
  4, -1).
- `while` is frozen at 3 doublings: 16, 24 and gradient 8, 8. The composition gives gradient
  28.345 instead of 453.52. None of these errors or warns; the results are silently wrong.
- `CppAD::CondExpGt(y, 0, y*y, -y)` is correct for both value and gradient. A fixed-count `for` is
  unrolled onto the tape and is fine.

### RTMB
- By default (`TapeConfig(comparison = "forbid")`), any comparison of AD values errors with
  `Comparison is generally unsafe for AD types`. That covers R `if`, `while`, and `ifelse()`.
- `ifelse()`'s AD method has signature `(test = "num", yes = "ad", no = "ad")`, so it only handles
  a plain (data) test. With an AD test it errors with `argument "e2" is missing`.
- `TapeConfig(comparison = "tape")` is the idiom documented in `?TapeConfig`: indicators on the
  tape, `f(x)*(x<0) + g(x)*(x>=0)`. It gives correct values and gradients. But R `if` on a taped
  comparison always takes the first branch (the advector is a complex number that coerces to
  TRUE): 2, -2 for `if (x <= 0)`, where 4, 2 is expected. And `while` never terminates; the script
  guards it at 60 iterations.
- `comparison = "allow"` ("Do not use") behaves like TMB: the path is frozen at taping, giving
  16, 24 and gradient 8, 8.
- R `for` is unrolled at taping and is fine.

### nimble
- `if`, `while`, `for` in a compiled `nimbleFunction` are correct.
- `derivs(..., reset = FALSE)` (the default) reuses the tape recorded at the first call (x = 2):
  `if` gradient 4, -4 and `while` gradient 8, 8. This is silently wrong. With `reset = TRUE`
  (re-record each call) everything is correct.
- Test-design pitfall: in a `buildDerivs` method an undeclared `for` index is typed as an AD scalar,
  and C++ fails (`cannot increment value of type 'CppAD::AD<double>'`). The fix is
  `declare(i, integer())` (or `i <- 1L`).
- The BUGS model language (`nimbleCode`) is different: `if` on a stochastic node errors with
  `argument is not interpretable as logical` (`if` there is build-time only; `step()` is the
  indicator idiom).

## Uncertain points

- **greta "if":** it could count as "(✓) via 0/1 masks" or as "–". There is no actual branching,
  and both sides are evaluated (so NaN in the unused side would poison the result). The suggested
  cell uses "–".
- **anvl "loop":** `nv_scan()` is a real compiled loop but not differentiable yet. Unrolled R loops
  are differentiable. Whether "Differentiable control flow" should say "(✓)" depends on the
  wording.
- **TMB / RTMB:** fixed-count loops work, but by unrolling onto the tape (tape size grows with the
  trip count). TMB does have real runtime branching through `CondExp*`, and RTMB through taped
  indicators. "(✓)" is generous for "data-dependent"; "frozen at taping" is the honest summary
  for statement-level `if`/`while`.
- **nimble:** I did not check whether nimble's HMC (nimbleHMC) resets tapes for models. Model code
  cannot branch on node values anyway, so it is likely irrelevant there.
- **odin2:** dust2 also supports stochastic models (particle filter), where the adjoint is not
  available (see `automatic_differentiation.R`). The `if` gradient here is for deterministic
  models only.
- **torch eager:** each `$item()` is a host sync. That is fine on CPU but a performance caveat on
  GPU; it is not a correctness issue.

## Summary

- Full runtime control flow with correct gradients through `if`, `while`, and loops: torch (eager
  and TorchScript), tensorflow (`tf$cond`/`while_loop`/`scan` or R syntax via tfautograph), Stan,
  ast2ast.
- tensorflow + XLA needs `maximum_iterations` to differentiate a `while_loop`.
- quickr: full control flow, no AD.
- anvl: compiled `nv_if`/`nv_while`/`nv_scan`. Gradients only through `nv_if()` (and unrolled R
  loops); `nv_while`/`nv_scan` have no reverse rule.
- nimble: full control flow in nimbleFunctions. Gradients are correct only with
  `derivs(reset = TRUE)`; the default replays a stale tape and silently returns wrong gradients.
- TMB: C++ `if`/`while` on AD types are frozen at `MakeADFun()` and silently wrong elsewhere.
  `CppAD::CondExpGt` is the branch idiom; fixed loops are unrolled.
- RTMB: comparisons of AD values error by default. `TapeConfig(comparison = "tape")` plus mask
  arithmetic gives differentiable branches. There is no data-dependent while; R `if` on a taped
  comparison is silently wrong. Fixed loops are unrolled.
- odin2/dust2: `if` expressions (differentiable through the adjoint); no loop statements beyond
  array equations and time steps.
- greta: no control flow on greta arrays (masks only); R loops are unrolled at model build.
- Correction to the brief: autograph is available from R. `tf_function(autograph = TRUE)`, via the
  tfautograph package, translates R `if`/`while`/`for`.
