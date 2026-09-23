---
name: add-api-function
description: Add a user-facing nv_* API function to anvl, wrapping primitives with R-idiomatic semantics
user_invocable: true
---

# Add an API Function (`nv_*`) to anvl

You are adding a user-facing API function to the anvl package.
API functions (`nv_*`) call into or more primitives (`prim_*`) to provide a convenient, R-idiomatic interface or add higher level functionality.

See `vignettes/extending_api.Rmd` for the in-depth explanation of the patterns below; this skill is the short operational checklist.

## Design Principles

### Work with any backend

API functions shipped with {anvl} must work with **both** the pjrt and quickr backends. There is one active backend at a time (`active_backend()`), every jitted function runs on it, and an array of another backend is an error. In practice this means:

- Never name a backend in an API function: no `backend =` arguments, no `with_backend()` calls. The caller chooses the backend.
- If the function creates a constant inside its body, use the `nv_<op>_like()` variant (see below) so the constant inherits the input's device. Do **not** call `device()` on a traced input -- it fails under `jit()`.
- Never hardcode `"f32"` / `"i32"` as a default data type; take `dtype = NULL` and resolve it with `default_float()` / `default_int()`, which read the active backend's defaults (see `default_dtypes()`).

### Follow R semantics

- **Naming:** Use R naming conventions. If base R or a common R package already has a function for this operation, match its name. For example: `nv_abs` (not `nv_absolute`), `nv_aperm` (matching `aperm()`), `nv_seq` (matching `seq()`). Only deviate from R names when there is a good reason (e.g. no R equivalent, or the R name would be ambiguous in the array context).
- **Semantics:** Match R behavior where it makes sense. For example, `nv_seq(from, to)` mirrors R's `seq()`, reductions like `nv_sum()` map to `sum()`. When R semantics conflict with array programming conventions (e.g. recycling rules vs. explicit broadcasting), prefer the array convention but document the difference.
- **R generics:** If a base R generic exists for this operation, implement an S3 method. For example:
  - `t()` → `t.AnvlBox` / `t.AnvlArray` dispatching to `nv_aperm()`
  - `abs()` → handled via `Math.AnvlBox` group generic
  - `+`, `-`, `*`, `/` → handled via `Ops.AnvlBox` group generic
  - `sum()`, `prod()`, `min()`, `max()` → handled via `Summary.AnvlBox` group generic
  - `[` → `[.AnvlBox` dispatching to `nv_subset()`

  Check `R/api-generics.R` for the existing group generics (`Ops`, `Math`, `Summary`) and individual method registrations. If your operation fits an existing group generic, add it there. Otherwise, create a standalone S3 method.

### Propose, then confirm

The exact convenience a wrapper should add varies by operation. **Propose a wrapper to the user but ask them to confirm** the semantic differences before implementing. Common patterns include:

- **Type promotion:** bring the inputs to one dtype with a rule, `as_anvl_arrays(..., .promote = promotion_common())`
- **Broadcasting:** broadcast scalars to match array shapes via `nv_broadcast_scalars()`
- **Default arguments:** infer `dtype` from the input when not provided
- **Idempotency:** skip no-op cases (return the input unchanged if already correct dtype/shape)
- **Input coercion:** bring auxiliary arguments to the input's dtype with `promotion_like("x")`

## Implementation

### Where to put the code

- Most API functions go in `R/api.R`.
- RNG functions go in `R/api-rng.R`.
- Subsetting goes in `R/api-subset.R`.
- S3 method registrations go in `R/api-generics.R`.

For simple binary ops, use the factory:

```r
nv_<name> <- make_do_binary(prim_<name>)
```

This automatically adds type promotion and scalar broadcasting.

For simple unary ops that need no extra convenience, alias the primitive directly:

```r
nv_<name> <- prim_<name>
```

For ops needing custom logic, write a function that normalizes its array inputs at the top:

- `as_anvl_array(x)` for a single array input.
- `as_anvl_arrays(...)` for multiple array inputs (infers a common device, errors on mismatched backends/devices).

A function whose _result_ dtype depends on its arguments must canonicalize with a rule -- for example `as_anvl_arrays(x = x, y = y, .promote = promotion_common())` -- rather than canonicalize first and `nv_convert()` afterwards. Without a rule an R value materializes at its default (the active backend's float default, `f32` for a double on pjrt) and any later conversion rounds through it. See `?promotion_rule` and `vignette("type-promotion")`; name the arguments so a rule can point at one.

After conversion, use `shape()`, `naxes()`, and `dtype()` directly -- they work on both concrete `AnvlArray`s and the `GraphBox` tracers that appear under `jit()`. Before conversion, `shape()` and `naxes()` still answer, but `dtype()` does not: a bare R value has none yet, so ask `peek_dtype()` what it _would_ materialize at.

### Constants and the `_like` pattern

If the function creates a constant inside its body (via `nv_fill`, `nv_iota`, `nv_seq`, `nv_scalar`, `nv_eye`, ...), the constant must be placed on the same backend/device as the input.
Under `jit()` this happens automatically, but in **eager mode** you are responsible:

- Use the `nv_<op>_like(x, ...)` variants, which default `dtype`, `shape`, and `device` from `x`.
- Example: `nv_fill_like(x, 0)` gives a zeros array matching `x`'s backend/device/dtype.

If you are adding a new array-creator function (`nv_foo` that allocates data rather than transforming an input), also add a `nv_foo_like(like, ...)` variant next to it.
Any dispatch-on-input constants inside other API functions should go through `_like`, not the bare creator.

### Integer literals

Write a whole number with the `L` suffix -- axis numbers, shape entries, indices, counts, and the
arithmetic and comparisons around them (`naxes(x) == 0L`, `axis + 1L`, `rep(1L, rank)`).

This holds for a literal that meets an array too. It takes that array's dtype, and an R integer
widens into any category, while a plain `1` is an R *double* that pulls an integer array into the
float category (`x_i32 - 1` is `f32`). So write `nv_ifelse(mask, 0L, x)` and `nv_fill_like(x, 0L)`
even when `x` is a float.

Keep the plain spelling only where the value is genuinely a real number that happens to be whole:
a distribution parameter, a probability bound, a threshold, a coefficient -- `sd = 1`,
`lower = 0, upper = 1`, `nv_pmax(-d, 1)`. This matters most for an argument default, which may meet
nothing at all and then settles on the default of its own category: `nv_rnorm(mean = 0, sd = 1)`
written with `0L` / `1L` returns the sample at the default *integer*.

### Binary element-wise ops

For element-wise binary primitives, use the `make_do_binary()` factory -- it already composes `nv_promote_to_common()` + `nv_broadcast_scalars()` before calling the primitive:

```r
nv_<name> <- make_do_binary(prim_<name>)
```

For full NumPy-style broadcasting (not just scalar-against-array), use `nv_broadcast_arrays()` after promotion (see `nv_outer()` for an example).

### Bringing auxiliary arguments to the input's dtype

If the underlying primitive requires all its inputs to share a dtype (e.g. `prim_clamp`, `prim_pad`), say so with a rule at the top rather than converting afterwards:

```r
args <- as_anvl_arrays(x = x, min = min, max = max, .promote = promotion_like("x"))
```

`promotion_like("x")` _builds_ an R bound at `x`'s dtype -- so `nv_clamp(x_f64, 0, 1)` keeps every digit, where `nv_convert(0, dtype(x))` would have materialized the literal at `f32` first -- and refuses a typed bound `x`'s dtype cannot hold instead of narrowing it silently. `dtype(x)` is not available here anyway: `x` may still be a bare R value.

### Static arguments

An API function is wrapped in `jit()` at the definition itself, with `static`
after the function so the signature reads on its own line:

```r
#' @export
nv_foo <- jit(function(x, axis) {
  ...
}, static = "axis")   # or static = 2:4
```

Omit `static` entirely when there are none: `nv_foo <- jit(function(x) { ... })`.

Any argument the function body _inspects_ -- branches on, validates with
`assert_*`, uses to compute shape/axes -- must be named (or positioned) in that
`static` list.
Typical candidates: `axes`, `shape`, `axis`, flags, mode strings, dtype specifiers.
Arrayish inputs (the actual data) should never be static.

## Roxygen2 Documentation

API functions use a consistent documentation pattern. Use templates from `man-roxygen/` where applicable.
If no proper template for a parameter or the return value exist, write the documentation inline.

### Structure

```r
#' @title <Short Title>
#' @description
#' <One-sentence description.> You can also use `<R operator or generic>()`.
#' @templateVar dtypes any data type    # the phrase the operand accepts
#' @template param_unary_x              # or @template params_lhs_rhs, etc.
#' @param <custom_param> (<type>)\cr    # for params not covered by templates
#'   <Description.>
#' @template return_unary               # or return_binary, return_reduce, etc.
#' @seealso [prim_<name>()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' <example code>
#' @export
```

### Key conventions

- **`@title`**: short, e.g. "Absolute Value", "Addition", "Transpose"
- **`@description`**: one sentence describing what the function does. If an R operator or generic dispatches to this function, mention it: "You can also use `abs()`.", "You can also use the `+` operator."
- **`@template`**: use templates for common parameter/return patterns. See the man-roxygen/ folder
  for available templates.
  - `params_reduce` — axes + drop params for reductions
- **`@param`**: write inline for parameters not covered by templates
- **`@seealso`**: always link to the underlying `prim_*` primitive. Optionally link to related `nv_*` functions.
- **`@examplesIf pjrt::plugins_downloaded()`**: wrap examples in this guard. Since all `prim_*` functions are auto-jitted and `nv_*` functions call into `prim_*` functions, examples can call them directly.
- **`@family`**: use for groups of related functions (e.g. `@family rng` for all RNG functions)

### S3 methods for R generics

When implementing an R generic, unify documentation using `@name` and `@rdname`:

```r
#' @rdname nv_<name>
#' @export
<generic>.<class> <- function(x, ...) {
  nv_<name>(x, ...)
}
```

The main documentation lives on the `nv_*` function; S3 methods use `@rdname` to point there.

## Add to `_pkgdown.yml`

The `nv_*` function must be added to the appropriate semantic section in `_pkgdown.yml` (e.g. "Arithmetic operations", "Mathematical functions", "Reduction operations", "Linear algebra", etc.). Check the existing sections and pick the best fit.

## Write Tests (in `tests/testthat/test-api.R`)

Add a **forward-pass-only** test for the `nv_*` wrapper. **Only test functionality not already covered by the primitive tests** — the convenience the wrapper adds on top of the primitive (e.g. type promotion, scalar broadcasting, default-arg behavior, R-operator dispatch). Do not re-test core correctness of the operation, edge cases like empty axes, dtype handling, or gradients — those belong with the primitive. If the wrapper is a thin alias (`nv_foo <- prim_foo`), a single sanity test is enough; often a default-argument check is the only thing worth asserting.

```r
describe("nv_foo", {
  it("promotes dtypes automatically", {
    out <- jit(nv_foo)(nv_array(1L, dtype = "i32"), nv_array(1.5, dtype = "f32"))
    expect_equal(dtype(out), "f32")
  })

  it("broadcasts scalar to array", {
    out <- jit(nv_foo)(nv_scalar(2), nv_array(c(1, 2, 3)))
    expect_equal(shape(out), 3L)
  })

  it("works via the + operator", {
    out <- nv_array(c(1, 2)) + nv_array(c(3, 4))
    expect_equal(as_array(out), array(c(4, 6), dim = 2L))
  })
})
```

## Verify

```r
devtools::document()
devtools::load_all()
devtools::test()
```

## Checklist

- [ ] Design confirmed with user (naming, semantics, convenience features)
- [ ] R generic / S3 method added if applicable (in `R/api-generics.R`)
- [ ] `nv_<name>` implemented with roxygen docs and `@export`
- [ ] Array inputs normalized at the top via `as_anvl_array()` / `as_anvl_arrays()`
- [ ] Binary element-wise ops built with `make_do_binary()` (or equivalent `nv_promote_to_common()` + `nv_broadcast_scalars()` pipeline)
- [ ] Auxiliary arguments converted to the input dtype via `nv_convert()` where the primitive requires it
- [ ] No-op shortcuts return the input unchanged (e.g. identity reshape / convert / broadcast)
- [ ] Constants created inside the function use `nv_<op>_like()` so they live on the right backend/device
- [ ] If the function is an array creator, a matching `nv_<name>_like()` variant is provided
- [ ] Arguments that the body inspects (shape, axes, flags, mode strings, dtype specifiers) are listed in the function's `jit(static = ...)` call
- [ ] `_pkgdown.yml`: added to appropriate semantic section
- [ ] Forward-pass test in `tests/testthat/test-api.R` covers the wrapper's convenience behavior
- [ ] `devtools::document()` run
- [ ] `devtools::test()` passes
