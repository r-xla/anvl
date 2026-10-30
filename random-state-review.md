# Review: random state handling on `feat/rng-split-review`

2026-10-23. Reviewed the working tree of `feat/rng-split-review` (`eee426d8d` plus
the staged and unstaged changes), trying to break the random state handling with
experiments against the installed dependencies (see [setup](#setup)).

## Summary

The core of the design holds. Counting the draws at trace time, forking for
`nv_while()`, taking the larger branch for `nv_if()` and continuing the count
through `gradient()` all held: every nesting I tried, up to
`nv_scan()` in `nv_while()` in `nv_scan()` with an `nv_if()`, advanced the random
state by exactly the hand-computed amount, and drew what the eager code draws (see
[what held up](#what-held-up)).

What breaks is at the edges:

- where device buffers are shared or donated;
- where R evaluates lazily;
- where anvl re-derives pjrt's argument tree;
- where the traced stand-in for a random state differs from the real one.

Two of the findings are serious: a reachable segfault, and compiled code that
silently draws different numbers than the same code run eagerly.

| # | Severity | Finding |
| --- | --- | --- |
| [1](#1-a-donated-random-state-buffer-crashes-r) | High | A donated random-state buffer crashes R (SIGSEGV) at the next draw |
| [2](#2-a-jitted-function-called-in-a-trace-draws-in-another-order) | High | A jitted function called inside a trace draws in another order than eagerly, or not at all |
| [3](#3-a-registered-tree-node-next-to-a-random-state-breaks-on_success) | Medium | A user-registered tree node next to a random state breaks `on_success()`: error after the run, partial advance |
| [4](#4-class-checks-see-another-class-under-jit) | Medium | `inherits(rng, "AnvlRandomState")` is `FALSE` under `jit()`, so class-checking code silently draws from the global random state |
| [5](#5-with_nv_rng_state-inside-nv_if-or-nv_scan-fails-in-eager-code) | Low–Medium | `with_nv_rng_state()` inside `nv_if()` / `nv_scan()` fails in eager code, with a message about `jit()` |
| [6](#6-with_nv_rng_stater-fr-says-r-was-passed-twice) | Low–Medium | `with_nv_rng_state(r, f(r))` fails with "passed to the jitted function twice", after the run |
| [7](#7-assigning-rngrandom_counter-fails-under-jit) | Low | Assigning `rng$random_counter` works eagerly, fails under `jit()` with a cryptic error |
| [8](#8-nv_scan-never-reaches-its-side-effect-check-and-a-test-fails) | Low | `nv_scan()` never reaches its side-effect check; the branch's own test fails |
| [9](#9-a-random-state-argument-must-share-the-device-of-the-other-inputs) | Low | A random-state argument must share the device of the other inputs |
| [10](#10-random-state-arguments-are-not-validated) | Low | Random-state arguments are not validated |
| [11](#11-the-traced-default-can-outlive-its-trace) | Low | The traced default random state can outlive its trace |
| [12](#12-reads-in-nv_while-see-the-fork-or-are-refused) | Low (docs) | Reads in an `nv_while()` body see the fork; reads in its condition are refused |
| [13](#13-each-call-that-draws-pays-about-150-µs) | Performance | Each call that draws pays about 150 µs of host overhead |
| [14](#14-smaller-things) | Low | Smaller things: the old positional API, a closed-over state in a loop's `init`, docs drift |

Your suggestion to keep the global random state as an R object is evaluated, with
a prototype, in [the global random state as an R object](#the-global-random-state-as-an-r-object).

## Findings

### 1. A donated random-state buffer crashes R

`nv_get_random_state()` is documented to return a copy, but `copy_random_state()`
(`R/rng.R:150`) only puts the same `AnvlArray`s into a new environment. The copy
shares its buffers with the global random state, so donating one of them deletes
the global's, and the next draw crashes R:

```r
nv_set_seed(1L)
s <- nv_get_random_state()
jit(function(k) k + 1L, donate = "k")(s$random_key)
nv_runif(1L)
#>  *** caught segfault ***
#> address 0x0, cause 'invalid permissions'
```

A random state passed with `donate` crashes the same way once an output aliases its
key:

```r
f <- jit(function(rng) with_nv_rng_state(rng, prim_sample_bits(2L, "ui32")), donate = "rng")
r <- nv_random_state(1L)
f(r)                                # the ui32[2] output aliases the donated key
with_nv_rng_state(r, nv_runif(1L))  # segfault
```

It crashes rather than erroring because of pjrt. Its execute path
(`impl_loaded_executable_execute()`, pjrt `src/pjrt.cpp:964`) and
`read_implicit_input()` (`src/dispatch_engine.cpp:812`) check their inputs only
through cached metadata: `device_ptr()`, `element_type()` and `dimensions()`. They
never call `checked_buffer()`.

A buffer whose metadata an earlier run cached passes those checks after it has
been donated, and its null handle reaches `PJRT_LoadedExecutable_Execute`. Every
buffer of the global random state is such a buffer after its first draw.

This is a pjrt bug that plain arrays hit too:

```r
x <- nv_array(c(1, 2))
f <- jit(function(a) a * 2)
f(x)
y <- x
jit(function(a) a + 1, donate = "a")(y)
f(x)  # segfault
```

The random state makes it easy to hit, for three reasons:

- the copy shares buffers;
- the global state's metadata is always cached;
- nothing stops a random state from being donated.

Fix:

- **pjrt:** check every input with `checked_buffer()` / `is_deleted()` in
  `impl_loaded_executable_execute()`. That covers both arguments and implicit
  inputs.
- **anvl:** give `nv_get_random_state()` buffers of its own, or a host-side
  representation ([below](#the-global-random-state-as-an-r-object)).
- **anvl:** refuse `donate` for an argument that holds a random state, or drop its
  leaves from the donation mask.

### 2. A jitted function called in a trace draws in another order

Under tracing, a jitted function called from another jitted function is inlined,
and gets its arguments as promises (`R/jit.R:169-176`). The comment there says so
on purpose: "one it never uses is never evaluated".

An eager call instead evaluates every supplied argument first, in the order of the
formals (`R/jit.R:177`). Once argument expressions draw, the two differ:

```r
noisy <- jit(function(x, s) nv_rnorm(1L) * s + x)
outer <- function() noisy(nv_scalar(0), nv_runif(1L))
with_nv_seed(1L, outer())       #> -0.2845  (nv_runif() draws, then nv_rnorm())
with_nv_seed(1L, jit(outer)())  #> -1.9241  (nv_rnorm() draws, then nv_runif())

ignore <- jit(function(x, unused) x + 1)
outer2 <- function() {
  ignore(nv_array(0), nv_runif(1L))
  nv_runif(1L)
}
with_nv_seed(1L, outer2())       #> 0.7630, counter 2
with_nv_seed(1L, jit(outer2)())  #> 0.8626, counter 1: the unused argument never draws
```

This is silent, and it breaks the guarantee both articles lead with:

> "a compiled function draws exactly what the same calls made one after the other
> without `jit()` would draw" (`side-effects.Rmd:164`, `random-numbers.Rmd:167`).

It affects any jitted function, anvl's own `nv_*` functions and primitives
included, whose body forces an argument after one of its own draws, or never
forces it. Before this branch, evaluation order was not observable, because the
random state was passed and returned explicitly.

Fix: in the tracing branch of the `jit()` wrapper, force the supplied arguments in
the order of the formals, then `...`, as the eager branch does. I tried this on a
copy of the package with these two lines before `.jit_fwd` is built:

```r
for (.jit_nm in .jit_given) get(.jit_nm, envir = .jit_env)
if (.jit_dots) eval(quote(list(...)), .jit_env)
```

Both cases above then agree with eager code, and the [full suite](#test-suite) gives
the same result with the change as without it. The cost: an argument the function
never uses is now evaluated in a trace, as it already is eagerly.

### 3. A registered tree node next to a random state breaks `on_success()`

`random_state_argument()` (`R/rng.R:461`) finds a random state among the call's
arguments by counting leaves "as `pjrt::flatten()` does". But it treats every
classed object as one leaf, while pjrt recurses into the children of any class
registered with `pjrt::register_tree_node()`. NEWS (`NEWS.md:22`) advertises that
function for users' own classes.

So a registered object before or between random states shifts the count, and one
that holds a random state hides it:

```r
pjrt::register_tree_node(
  "Empty",
  flatten = function(x) list(list(), NULL),
  unflatten = function(children, aux) structure(list(), class = "Empty"),
  repr = function(children, aux) "Empty"
)
f <- jit(function(a, e, b) list(with_nv_rng_state(a, nv_runif(1L)), with_nv_rng_state(b, nv_rnorm(1L))))
ra <- nv_random_state(1L)
rb <- nv_random_state(2L)
f(ra, structure(list(), class = "Empty"), rb)
#> Error: no applicable method for 'device' applied to an object of class "NULL"
# `ra` was advanced by 1, `rb` not at all (a successful call gives 1 and 2), and the outputs are lost
```

A random state inside a registered `Chain` object (`list(pos = , rng = )`) fails
with the same error.

The error comes from `side_effect_advance()` on `NULL`, after the program ran.
`side_effects_settler()` (`R/side-effects.R:507`) finds all host objects before it
advances any, but it never checks that it found them. So the first random state is
advanced, and the second is not.

Fix:

- Find the random states the way pjrt does. pjrt exports nothing that returns the
  registered objects of a tree, so there are two options:
  - add one to pjrt, e.g. `flatten(is_leaf = )` as in JAX's `tree_flatten()`;
  - in anvl, have the registered `flatten` of `AnvlRandomState` record the objects
    it is called on while the settler runs `pjrt::flatten(args)`. That yields them
    in preorder, the order in which the trace's `unflatten` created the traced
    handles.

  Either way, identify a random state by its ordinal among the random states
  rather than by its leaf index.
- Independently, have the settler abort before advancing anything when it does not
  find an object.

### 4. Class checks see another class under `jit()`

Under `jit()`, a random-state argument arrives as an `AnvlTracedRandomState`, of
class `c("AnvlTracedRandomState", "TraceTimeSideEffect")`, which does not inherit
from `AnvlRandomState`:

```r
draw <- function(rng = NULL) {
  if (inherits(rng, "AnvlRandomState")) with_nv_rng_state(rng, nv_runif(1L)) else nv_runif(1L)
}
r <- nv_random_state(1L); nv_set_seed(5L)
draw(r)       #> 0.8626: from `r`; `r` at counter 1, the global random state at 0
r <- nv_random_state(1L); nv_set_seed(5L)
jit(draw)(r)  #> 0.3439: from the global random state; `r` at 0, the global at 1
```

S3 methods for `AnvlRandomState` and `checkmate::assert_class()` are affected the
same way. `is_random_state()` accepts both classes, but it is internal.

Fix: export a predicate that accepts both classes, and point to it from
`?nv_random_state`. Adding `"AnvlRandomState"` to the traced class needs care: pjrt
would then flatten the handle as a registered node wherever it ends up in a tree,
such as a loop's state or a branch's outputs, and `abort_if_side_effects()` would no
longer see it.

### 5. `with_nv_rng_state()` inside `nv_if()` or `nv_scan()` fails in eager code

`nv_if()`, `nv_while()` and `nv_scan()` trace their functions even when they are
called eagerly. A random state used inside a branch or a body is therefore one that
the traced function closes over:

```r
rng <- nv_random_state(1L)
nv_if(TRUE, function() with_nv_rng_state(rng, nv_runif(1L)), function() nv_runif(1L))
#> Error: `rng_state` must be a random state the jitted function takes as an argument.
#> ℹ A random state the function closes over cannot be drawn from in it.
nv_scan(nv_scalar(0), body = function(c, x) {
  list(carry = c + with_nv_rng_state(rng, nv_runif(integer())), out = NULL)
}, steps = 2L)
#> (the same error)
```

The message speaks of a jitted function the user never wrote. Setting the random
state around the construct works: `with_nv_rng_state(rng, nv_if(...))`.

Fix: say in the message, and in `?local_nv_rng_state`, that the control-flow
functions trace their functions, and that the random state should be set around
them. A fuller fix: a closed-over random state could be taken the way the global
one is, as implicit inputs read from the object and advanced in `on_success`. The
reason given for refusing it ("the program could neither take its current key and
counter nor advance it") applies to the global random state too, and implicit
inputs solve exactly that. The usual caveat applies: an object a function closes
over is fixed when the function is traced.

### 6. `with_nv_rng_state(r, f(r))` says `r` was passed twice

```r
g <- function(r) list(with_nv_rng_state(r, nv_runif(1L)), nv_runif(1L))
r <- nv_random_state(1L)
with_nv_rng_state(r, g(r))       # works: `r` at counter 2
r <- nv_random_state(1L)
with_nv_rng_state(r, jit(g)(r))
#> Error: The same <AnvlRandomState> was passed to the jitted function twice.
```

Inside `with_nv_rng_state(r, ...)`, `r` is the host object of the global random
state. The program's view of the global random state and its view of the argument
therefore resolve to the same object in `side_effects_settler()`.

Refusing is right, since both views would split at the same counter. But:

- the message is wrong for this case;
- it is raised in `on_success`, after the program ran (pjrt
  `src/dispatch_engine.cpp:762-770`), so the outputs are lost, and so is any
  donated input;
- a function that only reads `r$random_counter` while it draws from the global
  random state is refused too.

Fix: name this case in the message, e.g. "is also the global random state, made
so by `with_nv_rng_state()`".

### 7. Assigning `rng$random_counter` fails under `jit()`

```r
skip <- function(rng) {
  rng$random_counter <- rng$random_counter + nv_scalar(10L, dtype = "ui64")
  with_nv_rng_state(rng, nv_runif(1L))
}
skip(nv_random_state(1L))       # works: counter 11
jit(skip)(nv_random_state(1L))
#> Error: unused argument (base::quote(list(<environment>, <environment>)))
```

The traced handle's `random_key` and `random_counter` are active bindings without
a setter (`R/rng.R:340-341`).

Fix: give them a setter that aborts with a clear message (a random state cannot be
changed in a jitted function), or document that.

### 8. `nv_scan()` never reaches its side-effect check, and a test fails

`nv_scan()` (`R/api.R:3518`) runs `map_tree(init, as_anvl_array)` before
`prim_scan()` reaches `abort_if_side_effects()` (`R/primitives.R:2711-2712`). A
random state in `init` or `xs` therefore gives:

```text
Error applying `.f` to leaf at `rng`.
Caused by error in `.f()`:
! Expected arrayish input, but got <AnvlTracedRandomState/TraceTimeSideEffect>
```

The test at `tests/testthat/test-rng.R:917-920` expects the intended message, and
fails. It is the one error of the [full suite](#test-suite). `nv_while()` is
`prim_while()` itself, so its check is reached.

Fix: call `abort_if_side_effects()` for `init` and `xs` at the top of `nv_scan()`.

### 9. A random-state argument must share the device of the other inputs

```r
r1 <- with_default_device("cpu:1", nv_random_state(1L))
x0 <- nv_array(0, device = "cpu:0")
with_nv_rng_state(r1, x0 + nv_runif(1L))  # works
jit(function(x, rng) with_nv_rng_state(rng, x + nv_runif(1L)))(x0, r1)
#> Error: invalid input `rng$random_key`: it lives on a different device than an earlier input;
#> all inputs must share a device unless a target device is fixed
```

The global random state is copied to the program's device, as implicit inputs
are, but a random-state argument is not. With a host-side representation
([below](#the-global-random-state-as-an-r-object)), this goes away.

### 10. Random-state arguments are not validated

Eagerly, `local_nv_rng_state()` checks the random state with
`assert_random_state()`. A random state passed to a jitted function is never
checked, so a broken one fails deep in the lowering:

```r
bad <- nv_random_state(1L)
bad$random_key <- nv_array(c(1, 2, 3), dtype = "ui32")
with_nv_rng_state(bad, nv_runif(1L))
#> Error: `rng_state` must be a random state, as `nv_random_state()` or `nv_get_random_state()` return.
jit(function(rng) with_nv_rng_state(rng, nv_runif(1L)))(bad)
#> Error: Argument 1 type must match the callee's input type.
#> ✖ Expected tensor<2xui32>, got tensor<3xui32>.
```

A counter of `i64` fails the same way. A key that is an R vector fails earlier, in
the dispatcher, with an error about `rng$random_key`.

Fix: check the avals of `effect$key` and `effect$counter` in
`side_effect_start.AnvlTracedRandomState()` (`R/rng.R:409`).

### 11. The traced default can outlive its trace

`local_nv_rng_state()` keeps the traced default in a process-wide global,
`globals$default_traced_random_state` (`R/rng.R:272-274`), which a `withr::defer()`
on `envir` restores. With an `envir` that outlives the trace, the handle stays the
default after the trace has ended. Every later trace that draws then fails,
unrelated eager calls included:

```r
keep <- new.env()
f <- jit(function(rng) {
  local_nv_rng_state(rng, envir = keep)
  nv_runif(1L)
})
f(nv_random_state(1L))
nv_runif(7L)
#> Error: A random state can only be drawn from by a jitted function it is passed to.
```

This needs an explicit `envir`, so it is unlikely, but the cause is that the
default is not tied to the trace.

Fix: keep the default on the descriptor, where sub-traces read it through
`parent`, or reset it when the outermost trace ends.

### 12. Reads in `nv_while()` see the fork, or are refused

`?nv_random_state` says that `random_key` and `random_counter` "read in the jitted
function what they read eagerly". In an `nv_while()` body, they read the fork
instead:

```r
f <- function(rng) {
  local_nv_rng_state(rng)
  nv_while(
    list(i = 0L, k = nv_array(c(0, 0), dtype = "ui32")),
    function(i, k) i < 2L,
    function(i, k) {
      nv_runif(1L)
      list(i = i + 1L, k = rng$random_key)
    }
  )$k
}
r <- nv_random_state(1L)
jit(f)(r)     #> 507451445 1853169794: the child's key
r$random_key  #> 0 1
```

In the condition of `nv_while()`, reading `rng$random_counter` is refused ("A side
effect cannot be used here"), although a read is not a use.

Fix:

- Document that reads in a `while` body see the fork.
- Possibly allow reads in a condition. The condition already takes twins of the
  values the loop carries (`side_effects_loop_twins()`), so a read there could
  resolve to them.

### 13. Each call that draws pays about 150 µs

Per call on the CPU, the median of three runs of 3000 calls each (the runs vary by
about ±20 %):

| call | per call |
| --- | --- |
| `jit(function(x) x + 1)` | 31 µs |
| `jit(function(x) x + nv_runif(1L))` | 184 µs |
| eager `nv_runif(1L)` | 246 µs |

The difference is `on_success`, which takes about 150 µs on its own. Of that, the
eager `counter + nv_scalar(n, ...)` in `side_effect_advance()` (`R/rng.R:447-451`)
takes 136 µs, and `nv_scalar()` alone about 100 µs. The prototype in the next
section roughly halves the overhead.

### 14. Smaller things

- **The old positional API.** `nv_runif(3L, state)` now passes the state as `min`,
  and fails with
  `Expected arrayish input, but got <AnvlTracedRandomState/TraceTimeSideEffect>`.
  As this is the breaking change, a targeted message would help migration: a random
  state where a parameter was expected, so use `with_nv_rng_state()`.
- **A closed-over random state in a loop's `init`** fails with
  `Expected arrayish value, but got <AnvlTracedRandomState>`. `abort_if_side_effects()` runs before
  `init` is materialized (`R/primitives.R:2611`), and the `AnvlRandomState`
  `unflatten` (`R/rng.R:301-307`) turns the materialized key and counter into a
  traced handle.
- **Docs drift.** `man/nv_while.Rd` and `man/prim_while.Rd` say that `init` "must
  not contain a random state", but `man-roxygen/param_while_init.R` does not, so
  the next `devtools::document()` drops the sentence. I checked by regenerating in
  a copy; nothing else differs.
- **`stablehlo()` output.** `stablehlo()` now returns the helper functions as its
  fourth element. The internals article (`vignettes/articles/internals.Rmd:129-137`)
  still compiles `stablehlo(graph)[[1L]]` alone, which is no longer a complete
  module for a graph that draws.

## The global random state as an R object

I read your suggestion as: keep the global random state's key and counter on the
host, as plain R values (its four 32-bit words), instead of as `AnvlArray`s.

### Prototype

I monkey-patched it into a loaded copy, without changing the repository:

- A host object holds `words`. Its `random_key` and `random_counter` are uploaded
  afresh whenever pjrt's `get()` reads them.
- `side_effect_advance()` adds `n` to the words in R, carrying into the high word.
- `nv_get_random_state()` builds a random state of fresh arrays from the words.
- `nv_set_seed()` and `nv_set_random_state()` store words.

### Results

- **Same draws.** The draws are identical to the branch's: eager and jitted,
  `nv_rnorm()`, `nv_while()` and `nv_sample_int()`, the state derived from base R,
  the carry into the counter's high word, and `with_nv_rng_state()` of a random
  state of its own.
- **No crash.** The crash of finding 1 through `nv_get_random_state()` is gone.
- **Faster,** once the upload goes through `pjrt::pjrt_buffer()` rather than
  `nv_array()`:

| per call, CPU (median of three runs) | branch | words, uploaded with `nv_array()` | words, uploaded with `pjrt_buffer()` |
| --- | --- | --- | --- |
| jitted function that draws | 184 µs | 249 µs | 115 µs |
| eager `nv_runif(1L)` | 246 µs | 289 µs | 146 µs |
| jitted function that does not draw | 31 µs | 32 µs | 31 µs |

The advance in R takes 2 µs, against 136 µs for the eager device add, and the two
raw uploads take about 2 × 21 µs.

### What else it gives

- **No device for the global random state.** It is not copied to each program's
  device, and programs on several devices draw without the state ever moving.
- **No device operation between calls.** No call waits on anything of the call
  before it, not even a tiny add, which is the property the design is after.
- **No device sync** to print or save a random state.
- **A true copy** from `nv_get_random_state()`.

### What it does not fix on its own

A random state of its own stays device arrays, and `with_nv_rng_state(r, ...)`
makes it the host object of the global random state. So the second crash of
finding 1 (a donated random-state argument) remains, and so do findings 9 and 10.

### Recommendation: make `AnvlRandomState` itself the host object

- **As an argument,** its tree node's children become two R-data leaves, the key's
  words and the counter's words (`array(.., 2L)`). pjrt uploads them to the
  program's device at the declared data type. That means no device conflicts,
  nothing to donate, and still one program for every random state: R arrays of
  the same type and shape share a cache entry (checked).
- **The counter travels as two `ui32` words,** exact over its full 64 bits; an R
  double is exact only below 2^53. `random_state_call()` (`R/rules-stablehlo.R`)
  already splits the counter into words, so taking the words directly simplifies
  it.
- **A closed-over random state** could then be taken exactly like the global one,
  which fixes finding 5.
- **In pjrt,** an implicit input's `get()` could return R data, which pjrt would
  upload straight to the entry's device at the declared data type, as it does for
  R-data leaves. That saves the R-level buffer construction and the copy between
  devices.

Two questions remain open:

- what `rng$random_key` and `rng$random_counter` should return eagerly (arrays
  built on demand would keep today's behaviour);
- the cost on CUDA, to measure: an upload of 16 bytes per random state per call,
  against today's scalar upload plus a kernel launch for the add.

Taken as far as `AnvlRandomState`, this addresses findings 1, 9 and 13, and makes
finding 5 straightforward to fix. The other findings are independent of it.

## What held up

All of these matched eager code, a hand count, or the documented behaviour:

- **Counting through nesting.** `nv_scan()` of 5 steps over `nv_while()` over
  `nv_scan()` of 4 steps, plus an `nv_if()`, plus a draw: the counter advanced by
  exactly 10, one fork and one draw per outer step. A printed graph that mixes a
  random-state argument with `nv_while()`, `nv_if()` and `nv_scan()` shows the
  right `then { %counter1 += 5 }`.
- **Edge cases of the counter.** `nv_scan(steps = 0L)` with a body that draws
  advances nothing, and the draw after it equals the eager one. Zero-size draws
  (`nv_runif(0L)`, `prim_sample_bits(c(2, 0), "ui8")`, `nv_rnorm(0L)`,
  `nv_sample_int(0L, 3L)`) work, and advance by one. A counter wrapping around
  2^64, and the carry into its high word, give the same result eagerly and jitted.
- **Gradients.** `gradient()` through `nv_if()` whose branches draw, for both
  predicates, advances by the larger branch. A second-order gradient of a function
  that draws draws once. `prim_fork_random_state()` works inside a differentiated
  function, and `gradient(wrt = "rng")` gives a clear error.
- **Arguments.** Named arguments out of order, static arguments before a random
  state, random states in `...` and in nested lists all work. A random state as a
  static argument, or returned from a jitted function, gives a clear error.
  Donating a random state that no output aliases is harmless.
- **Devices.** The global random state on `cpu:1` with programs on `cpu:0`, and
  `jit(device = "cpu:1")` with a random state on `cpu:0`, both work.
- **Seeds.** Seeds outside the 32-bit range, and `NA`, are rejected.
- **The generator.** The ThreeFry lowering is pinned by known-answer vectors and by
  JAX's values in the suite. I did not re-derive it.

## Test suite

- **The branch's suite,** with `ANVL_TEST_SKIP_QUICKR=1`: 6299 passed, 120 skipped,
  0 failed, and 1 error, `test-rng.R:917-920` ([finding 8](#8-nv_scan-never-reaches-its-side-effect-check-and-a-test-fails)).
- **The same suite with the change of [finding 2](#2-a-jitted-function-called-in-a-trace-draws-in-another-order)**,
  on a copy: the same result, 6299 passed, 120 skipped, 0 failed, and the same
  1 error.

## Setup

- **Code:** anvl `feat/rng-split-review` at `eee426d8d`, with the staged and
  unstaged changes of 2026-10-23.
- **R and dependencies:** R 4.5.3 on macOS 26.5 (arm64). pjrt 0.6.0.9001 as
  installed; it matches the `feat/tree-node-types` worktree at `e4cbbc43`, whose
  sources are cited above. stablehlo 0.5.0, xlamisc 0.5.0.9000.
- **Devices:** the CPU plugin, with `PJRT_CPU_DEVICE_COUNT=2` for the device
  cases.

Each repro above is self-contained after `devtools::load_all()`. The scripts, and
the prototype of the previous section, are in a temporary session directory, which
will not last:
`/private/tmp/claude-501/-Users-sebi-r-xla-anvl/00f8128c-9c3a-4124-821d-2a03316437b2/scratchpad/repros/`
and
`/private/tmp/claude-501/-Users-sebi-r-xla-anvl/00f8128c-9c3a-4124-821d-2a03316437b2/scratchpad/proto_host_state_raw.R`.
