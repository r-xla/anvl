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

API functions shipped with {anvl} must work with **both** the pjrt and quickr backends. In practice this means:

- If the function internally `jit()`s a helper, set `backend = "auto"` on that `jit()` call so it adapts to the caller's backend.
- If the function creates a constant inside its body, use the `nv_<op>_like()` variant (see below) so the constant inherits the input's backend/device. Do **not** call `device()` on a traced input -- it fails under `jit()`.

### Follow R semantics

- **Naming:** Use R naming conventions. If base R or a common R package already has a function for this operation, match its name. For example: `nv_abs` (not `nv_absolute`), `nv_transpose` (matching `t()`), `nv_seq` (matching `seq()`). Only deviate from R names when there is a good reason (e.g. no R equivalent, or the R name would be ambiguous in the array context).
- **Semantics:** Match R behavior where it makes sense. For example, `nv_seq(start, end)` mirrors R's `seq()`, reductions like `nv_reduce_sum()` map to `sum()`. When R semantics conflict with array programming conventions (e.g. recycling rules vs. explicit broadcasting), prefer the array convention but document the difference.
- **R generics:** If a base R generic exists for this operation, implement an S3 method. For example:
  - `t()` → `t.AnvlBox` / `t.AnvlArray` dispatching to `nv_transpose()`
  - `abs()` → handled via `Math.AnvlBox` group generic
  - `+`, `-`, `*`, `/` → handled via `Ops.AnvlBox` group generic
  - `sum()`, `prod()`, `min()`, `max()` → handled via `Summary.AnvlBox` group generic
  - `[` → `[.AnvlBox` dispatching to `nv_subset()`

  Check `R/api-generics.R` for the existing group generics (`Ops`, `Math`, `Summary`) and individual method registrations. If your operation fits an existing group generic, add it there. Otherwise, create a standalone S3 method.

### Propose, then confirm

The exact convenience a wrapper should add varies by operation. **Propose a wrapper to the user but ask them to confirm** the semantic differences before implementing. Common patterns include:

- **Type promotion:** auto-promote inputs to a common dtype via `nv_promote_to_common()`
- **Broadcasting:** broadcast scalars to match array shapes via `nv_broadcast_scalars()`
- **Default arguments:** infer `dtype` from the input when not provided
- **Idempotency:** skip no-op cases (return operand unchanged if already correct dtype/shape)
- **Input coercion:** convert auxiliary arguments to match the operand's dtype

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

After conversion, use `shape()`, `ndims()`, and `dtype()` directly -- they work on both concrete `AnvlArray`s and the `GraphBox` tracers that appear under `jit()`.

### Constants and the `_like` pattern

If the function creates a constant inside its body (via `nv_fill`, `nv_iota`, `nv_seq`, `nv_scalar`, `nv_eye`, ...), the constant must be placed on the same backend/device as the input.
Under `jit()` this happens automatically (if `backend = "auto"` is set on the outer `jit()` call), but in **eager mode** you are responsible:

- Use the `nv_<op>_like(x, ...)` variants, which default `dtype`, `shape`, `ambiguous`, and `device` from `x`.
- Example: `nv_fill_like(x, 0)` gives a zeros array matching `x`'s backend/device/dtype.

If you are adding a new array-creator function (`nv_foo` that allocates data rather than transforming an input), also add a `nv_foo_like(like, ...)` variant next to it.
Any dispatch-on-input constants inside other API functions should go through `_like`, not the bare creator.

### Binary element-wise ops

For element-wise binary primitives, use the `make_do_binary()` factory -- it already composes `nv_promote_to_common()` + `nv_broadcast_scalars()` before calling the primitive:

```r
nv_<name> <- make_do_binary(nvl_<name>)
```

For full NumPy-style broadcasting (not just scalar-against-tensor), use `nv_broadcast_arrays()` after promotion (see `nv_outer()` for an example).

### Converting auxiliary arguments

If the underlying primitive requires all its inputs to share a dtype (e.g. `nvl_clamp`, `nvl_pad`), convert the helper arguments to the operand dtype via `nv_convert(aux, dtype(operand))`. `nv_convert()` is a no-op when the dtype already matches, so the extra calls are free.

### Static arguments

Any argument the function body *inspects* -- branches on, validates with `assert_*`, uses to compute shape/dims -- must be declared `static =` on the outer `jit()` call (and forwarded via `static =` to `check_eager()` in tests).
Typical candidates: `dims`, `shape`, `dim`, flags, mode strings, dtype specifiers.
Arrayish inputs (the actual data) should never be static.

## Roxygen2 Documentation

API functions use a consistent documentation pattern. Use templates from `man-roxygen/` where applicable.
If no proper template for a parameter or the return value exist, write the documentation inline.

### Structure

```r
#' @title <Short Title>
#' @description
#' <One-sentence description.> You can also use `<R operator or generic>()`.
#' @template param_operand              # or @template params_lhs_rhs, etc.
#' @param <custom_param> (<type>)\cr    # for params not covered by templates
#'   <Description.>
#' @template return_unary               # or return_binary, return_reduce, etc.
#' @seealso [nvl_<name>()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' <example code>
#' @export
```

### Key conventions

- **`@title`**: short, e.g. "Absolute Value", "Addition", "Transpose"
- **`@description`**: one sentence describing what the function does. If an R operator or generic dispatches to this function, mention it: "You can also use `abs()`.", "You can also use the `+` operator."
- **`@template`**: use templates for common parameter/return patterns:
  - `param_operand` — single operand
  - `params_lhs_rhs` — binary operands (includes promotion/broadcasting note)
  - `param_dtype`, `param_shape`, `param_ambiguous` — common params
  - `return_unary`, `return_binary`, `return_reduce`, `return_reduce_boolean`
  - `params_reduce` — dims + drop params for reductions
- **`@param`**: write inline for parameters not covered by templates
- **`@seealso`**: always link to the underlying `nvl_*` primitive. Optionally link to related `nv_*` functions.
- **`@examplesIf pjrt::plugins_downloaded()`**: wrap examples in this guard. Since all `nvl_*` functions are auto-jitted and `nv_*` functions call into `nvl_*` functions, examples can call them directly.
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

Every API function also needs a `check_eager()` entry in the "cross-device eager (check_eager)" `describe` block at the bottom of `test-api.R`. `check_eager()` (defined in `tests/testthat/helper.R`) runs the function both in eager mode on `cpu:1` and jit-compiled on `cpu:0`, and asserts:

1. The eager output lives on `cpu:1`.
2. The jitted output lives on `cpu:0`.
3. The two outputs agree value-wise (tolerance defaults to `1e-6`).

This is what catches bugs where constants end up on the wrong device, or where eager vs jit diverge.

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

# In the "cross-device eager (check_eager)" describe block:
it("nv_foo", {
  check_eager(nv_foo, vec_f, vec_f2)
})

# For a function with a static argument, forward it via `static =`:
it("nv_reduce_foo", {
  check_eager(nv_reduce_foo, vec_f, dims = 1L, static = "dims")
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
- [ ] Auxiliary arguments converted to the operand dtype via `nv_convert()` where the primitive requires it
- [ ] No-op shortcuts return the input unchanged (e.g. identity reshape / convert / broadcast)
- [ ] Constants created inside the function use `nv_<op>_like()` so they live on the right backend/device
- [ ] If the function is an array creator, a matching `nv_<name>_like()` variant is provided
- [ ] Arguments that the body inspects (shape, dims, flags, mode strings, dtype specifiers) are declared `static =` on every `jit()` / `check_eager()` call
- [ ] `_pkgdown.yml`: added to appropriate semantic section
- [ ] Forward-pass test in `tests/testthat/test-api.R` covers the wrapper's convenience behavior
- [ ] `check_eager()` entry in the "cross-device eager (check_eager)" `describe` block, with any `static =` arguments forwarded
- [ ] `devtools::document()` run
- [ ] `devtools::test()` passes
