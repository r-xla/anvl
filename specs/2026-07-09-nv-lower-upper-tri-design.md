# Design: `nv_lower_tri` / `nv_upper_tri`

Date: 2026-07-09
Issue: [r-xla/anvl#195](https://github.com/r-xla/anvl/issues/195) — "Export functions for triangular matrices"

## Motivation

anvl exports `nv_tril` and `nv_triu`, which return an array with one triangle
zeroed, plus `nv_chol` and `nv_triangular_solve`. It has no way to obtain the
triangular *mask* itself.

The two operations are distinct and neither replaces the other:

| | returns | built from |
|---|---|---|
| `nv_tril(x)` | `x`'s values, other triangle zeroed | an array |
| `nv_lower_tri(shape)` | a boolean array | a shape |

Precedent splits the same way in both R and Python. Base R's `lower.tri(x)` /
`upper.tri(x)` return a logical mask; base R has *no* zeroing extractor (the
idiom is the assignment `x[upper.tri(x)] <- 0`). The extractor's R name comes
from the recommended `Matrix` package: `Matrix::tril(x, k = 0)`. numpy mirrors
this exactly with `np.tri` (mask) versus `np.tril` (extractor).

anvl currently has the extractor and lacks the mask. This spec adds it.

The mask logic is presently written out three times — inline in `nv_tril`
(`R/api.R:2635`), inline in `nv_triu` (`R/api.R:2663`), and as
`tril_mask`/`triu_mask` (`R/rules-reverse.R:1247`, `:1254`). The copies have
already drifted: both helpers take a `dt` argument neither uses, and `tril_mask`
compares with `prim_ge(rows, cols)` while `triu_mask` uses `rows <= cols`. This
change collapses all three onto one definition.

## Public API

```r
nv_lower_tri(shape, diagonal = -1L, device = NULL)
nv_upper_tri(shape, diagonal =  1L, device = NULL)
nv_lower_tri_like(like, diagonal = -1L, shape = NULL, device = NULL)
nv_upper_tri_like(like, diagonal =  1L, shape = NULL, device = NULL)
```

Each returns a `bool` array of the given `shape`. `shape` must have length 2;
rectangular shapes are permitted, matching `lower.tri` and `np.tri`.

Element `(i, j)` is `TRUE` when:

- `nv_lower_tri`: `i >= j - diagonal`
- `nv_upper_tri`: `i <= j - diagonal`

### Naming

`nv_lower_tri` / `nv_upper_tri` take base R's names, and like base R they return
a logical mask. `nv_tril` / `nv_triu` keep the `Matrix` and numpy names for the
extractor. Both anvl names therefore mirror a library the user already knows,
and the two families stay visually distinct at a call site — a property that
`nv_lower_triangle` next to `nv_lower_tri` would lose.

### The `diagonal` default

`diagonal` carries exactly the meaning it already has in `nv_tril` / `nv_triu`,
which is numpy's `k`. Only the *default* differs, and it differs deliberately:
base R's `lower.tri()` excludes the diagonal by default, whereas
`nv_tril(x, diagonal = 0L)` includes it. Defaulting to `-1L` (lower) and `1L`
(upper) means the borrowed name keeps its promise:

```r
nv_lower_tri(c(n, n))          # == lower.tri(x)
nv_lower_tri(c(n, n), 0L)      # == lower.tri(x, diag = TRUE)
nv_upper_tri(c(n, n))          # == upper.tri(x)
nv_upper_tri(c(n, n), 0L)      # == upper.tri(x, diag = TRUE)
```

The asymmetric defaults are the cost of that guarantee. The parameter itself
stays fully general, so any offset is expressible — something base R's logical
`diag` argument cannot do.

### dtype

The return is always `bool`, as base R's logical mask is. There is no `dtype`
argument; a numeric mask is `nv_convert(m, "f32")`.

## Layer and implementation

This is a pure `nv_*` API addition. **No new primitive.** The functions compose
`prim_iota` with a comparison. Because the output is `bool` it is not
differentiable, so there is no reverse rule and nothing to register.

A single unexported helper holds the logic:

```r
# R/api.R
# `diagonal` must already be a scalar integer; callers validate and coerce.
tri_mask <- function(shape, diagonal, lower, device = NULL) {
  rows <- prim_iota(dim = 1L, dtype = "i32", shape = shape, start = 1L, device = device)
  cols <- prim_iota(dim = 2L, dtype = "i32", shape = shape, start = 1L, device = device)
  if (lower) rows >= cols - diagonal else rows <= cols - diagonal
}
```

Everything routes through it:

- `nv_lower_tri` / `nv_upper_tri` validate their arguments (`assert_int(diagonal)`)
  and coerce with `as.integer(diagonal)` before calling `tri_mask`, exactly as
  `nv_tril` does today at `R/api.R:2634-2637`.
- `nv_lower_tri_like` / `nv_upper_tri_like` follow the `nv_iota_like` pattern,
  using `like_defaults(like, shape = shape, device = device)`. `dtype` is not
  passed through, so it stays `bool` regardless of `like`'s dtype.
- `nv_tril` / `nv_triu` become thin wrappers:
  ```r
  nv_tril <- function(operand, diagonal = 0L) {
    operand <- as_anvl_array(operand)
    nv_ifelse(nv_lower_tri_like(operand, diagonal), operand, 0)
  }
  ```
- `R/rules-reverse.R` deletes `tril_mask` and `triu_mask`; `triangular_mask()`
  and the `prim_chol` reverse rule call `tri_mask(c(n, n), 0L, lower)` instead.

`tri_mask` exists so the reverse rules never have to call `nv_lower_tri`, which
carries a `@jit static 1:3` annotation and with it a static-shape contract that
does not belong inside gradient tracing. The unexported helper gives one
definition of the mask without that coupling.

Note that this is narrower than "the helper avoids `nv_*` entirely". `tri_mask`'s
`>=`, `<=`, and `-` dispatch through `Ops.AnvlArray` to `nv_ge`, `nv_le`, and
`nv_sub`, which are themselves jit-registered. That is pre-existing and harmless:
the `triu_mask` it replaces already wrote `rows <= cols`, and the surrounding
reverse rules already call `nv_matmul`.

Public behaviour of `nv_tril` / `nv_triu` is unchanged. `nv_lu` (`R/api.R:1882`,
`:1895`), their only internal caller, is untouched.

## Jit

`nv_lower_tri`'s arguments are all static R values — a shape vector, an integer,
a device — so it registers as `static = 1:3`, following `nv_eye`'s `static = 1:3`.
The `_like` variants take a traced array first, so `static = 2:4`.

Four entries are added to `R/jit-registry.R`.

## Errors

Via `cli_abort`, in the style of `nv_tril`'s existing `"operand must be a 2-D array"`:

- `shape` not length 2
- `shape` containing negative extents
- `diagonal` not a scalar integer (`assert_int`, as `nv_tril` does)

## Testing

anvl's CLAUDE.md prefers comparing against torch, but directs that trivial
functionality or functionality torch does not cover be tested manually instead.
Base R's `lower.tri` / `upper.tri` are the exact reference here and the
functions are trivial, so tests compare against base R only — not both.

In `tests/testthat/test-api.R`:

- defaults reproduce `lower.tri(x)` and `upper.tri(x)`
- `diagonal = 0L` reproduces `lower.tri(x, diag = TRUE)` / `upper.tri(x, diag = TRUE)`
- several other `diagonal` offsets, including magnitudes `>= 2`, which base R's
  logical `diag` cannot express — compared against an explicit `outer()` mask
- rectangular shapes, both `m > n` and `m < n`
- returned dtype is `bool`
- `_like` variants inherit shape and device from `like`, and return `bool` even
  when `like` is a float array
- quickr backend, guarded by `skip_if_no_quickr()`

Regression coverage for the refactor:

- the existing `nv_tril` / `nv_triu` tests must pass unchanged
- the existing `chol` and `triangular_solve` reverse-rule tests must pass
  unchanged, since `triangular_mask()` is rewritten

## Files touched

| File | Change |
|---|---|
| `R/api.R` | add `tri_mask`, `nv_lower_tri`, `nv_upper_tri`; rewrite `nv_tril`, `nv_triu` |
| `R/api-like.R` | add `nv_lower_tri_like`, `nv_upper_tri_like` |
| `R/jit-registry.R` | 4 new entries |
| `R/rules-reverse.R` | delete `tril_mask` / `triu_mask`; rewrite `triangular_mask` |
| `pkgdown/_pkgdown.yml` | list the 4 new functions beside `nv_tril` / `nv_triu` (around line 262) |
| `NEWS.md` | changelog entry |
| `tests/testthat/test-api.R` | tests above |
| `NAMESPACE`, `man/` | regenerated by `devtools::document()` |

Closes #195.

## Out of scope

- `tril_indices` / `triu_indices` equivalents. They return variable-length index
  vectors whose length depends on `n` and `k`, which does not fit XLA's static
  shape requirement without making both arguments compile-time constants.
- Batched (`>2`-D) masks. `nv_tril` / `nv_triu` are 2-D only today; widening
  both families is separate work.
- A `dtype` argument, per the decision above.
