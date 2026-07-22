# Boolean Mask Subsetting

Add boolean-mask support to `nv_subset()` and `nv_subset_assign()` (and the `[` / `[<-`
operators). Masks come in two forms: per-dimension masks that shrink one dimension, and
whole-array masks that flatten to the selected elements.

Masks whose values come from an anvl array (e.g. `x > 6`) are eager-only, because the
number of selected elements -- and therefore the output shape -- depends on runtime data.
Masks that are R logical arrays are compile-time constants and work under `jit()` as well.

## Semantics

### Mask forms

| Form | Example | Result |
|------|---------|--------|
| Per-dimension | `x[arr(TRUE, FALSE, TRUE), ]` | Keeps rank; that dimension shrinks to `sum(mask)` |
| Whole-array (flat) | `x[x > 6]` | 1-D array of the selected elements |

Flat-mask output follows R's column-major order, so `as_array(x[m])` equals
`as.array(x)[as_array(m)]`.

### Accepted mask values

A mask is either an R logical **array** (`arr(TRUE, FALSE, TRUE)`, or any `array()` with a
`dim` attribute) or an anvl array of dtype `bool`.

Bare R logical vectors are rejected, including length-1 ones. This mirrors the existing
rule for numeric subscripts (`R/api-subset.R`, "Vectors of length > 1 are not allowed as
subset indices"), and avoids inventing a meaning for `x[TRUE, ]`: base R recycles it to
"select all", while an anvl-style reading would make it a length-1 mask valid only on a
size-1 dimension. Rather than pick a surprising winner, both are rejected with a hint:

```r
x[arr(TRUE, FALSE, TRUE), ]   # ok
x[c(TRUE, FALSE, TRUE), ]     # Error: use `arr()` to select multiple elements
x[TRUE, ]                     # Error (same hint)
```

Non-`bool` anvl arrays keep their current meaning: indices, not masks.

### Disambiguation

Every mask carries a shape, which makes the single-subscript case unambiguous:

> A subscript is a **flat** mask when its shape equals the operand's full shape.
> Otherwise it is a **per-dimension** mask on dimension 1, with trailing dimensions full.

```r
x <- nv_matrix(1:12, nrow = 3)     # 3 x 4
x[x > 6]                            # flat            -> [6]
x[arr(TRUE, FALSE, TRUE)]           # per-dim on rows -> [2, 4]
```

The per-dimension reading of a single subscript deliberately differs from base R, which
would flatten. It is consistent with anvl's existing convention that `x[2]` on a matrix
means "row 2", not "element 2".

For a rank-1 operand the two forms coincide, and both produce the same result.

### Validation

- Length must match exactly: `length(mask) == dim size` per dimension, or
  `shape(mask) == shape(x)` for a flat mask. No recycling.
- `NA` in a mask is an error; there is no representation for it.
- An anvl mask is pulled to the host with `as_array()` (a device sync) to compute
  `which()`. Under tracing this is impossible, so it errors; see below.

### Assignment

`x[mask] <- value` and `x[mask, ] <- value` return an array with the **same shape as `x`**.
A scalar `value` broadcasts to the selected shape; a non-scalar `value` must match it
exactly (`[k]` for a flat mask, where `k = sum(mask)`). Masks cannot address the same
element twice, so the scatter is emitted with `unique_indices = TRUE`.

### Tracing

| Mask source | Eager | Inside `jit()` |
|-------------|-------|----------------|
| R logical array | Yes | Yes -- resolved to `which()` at trace time |
| anvl `bool` array | Yes | Error: output shape is data-dependent |

### Empty selection

An all-`FALSE` mask yields a genuine zero-sized array (`[0]`, or `[0, 4]` for a
per-dimension mask on the rows of a 3 x 4 array), rather than an error.

## Implementation

### Per-dimension masks

No new machinery. `parse_subset_spec()` gains a logical branch: validate the mask, then
return `SubsetIndices(which(mask))`. Everything downstream -- gather parameters, scatter
parameters, the `prim_gather` call, `subset_scatter_core()` -- is untouched. An anvl mask
is converted to an R logical array first, guarded by `currently_tracing()`.

New helpers in `R/api-subset.R`:

- `is_mask_subscript(e)` -- recognises an R logical array or an anvl `bool` array.
- `as_r_mask(e, expected_shape)` -- validates shape/`NA`, rejects bare vectors with the
  `arr()` hint, errors when an anvl mask is used under tracing, and returns an R logical
  array.

### Flat masks

A new path in `nv_subset()` / `nv_subset_assign()`, taken before `parse_subset_specs()`
when there is exactly one subscript and it is a mask whose shape equals the operand shape.
`which(mask, arr.ind = TRUE)` produces a `[k, rank]` integer matrix of index tuples in R's
column-major order, which is what makes the output order match base R.

Two helpers build the *same* parameter lists the existing code already consumes, so the
`prim_gather` call in `nv_subset()` and `subset_scatter_core()` in `nv_subset_assign()`
are reused verbatim:

- `flat_mask_to_gather()`: `slice_sizes = rep(1L, rank)`, `offset_dims = integer()`,
  `collapsed_slice_dims = seq_len(rank)`, `start_index_map = seq_len(rank)`,
  `index_vector_dim = 2L`, `unique_indices = TRUE`.
- `flat_mask_to_scatter()`: `update_window_dims = integer()`,
  `inserted_window_dims = seq_len(rank)`, `scatter_dims_to_operand_dims = seq_len(rank)`,
  `index_vector_dim = 2L`, `update_shape = k`, `unique_indices = TRUE`.

`indices_are_sorted` is `FALSE` on both: column-major tuple order is not the lexicographic
order the StableHLO attribute refers to.

### Zero-size fix

Zero-sized selections currently fail because a size-0 dimension is misclassified as
scalar. `which(sizes > 1L)` skips it, so the "all starts are scalar" branch builds a
start-index array of the wrong rank and `prim_gather()` reports
"length(start_index_map) must equal the index vector size".

Change `> 1L` to `!= 1L` in four places:

- `dynamic_start_indices()` and `static_start_indices()` -- `multi_index_dims`
- `subset_specs_to_gather()` and `subset_specs_to_scatter()` -- the
  `is_subset_indices(s) && s$size > 1L` predicate

Only size-0 behaviour changes; size-1 is unaffected either way. Both `prim_gather()` and
`prim_scatter()` were verified to handle zero-sized index arrays correctly already, so no
primitive changes are needed. This also fixes the pre-existing failures of
`x[array(integer(0)), ]` and `x[1:0, ]`.

## Testing

In `tests/testthat/test-api-subset.R`, comparing against base R subsetting wherever the
semantics coincide (`as_array(x[m])` vs `as.array(x)[as_array(m)]`):

- Per-dimension masks on each dimension of a 2-D array, and on a 1-D array.
- Flat masks: `x[x > 6]`, mixed dtypes, rank 1 and rank 2.
- Assignment: both forms, scalar `value` (broadcast) and exact-shape `value`; result keeps
  the operand's shape.
- All-`TRUE` and all-`FALSE` masks, including the zero-sized results.
- `jit()`: a static R logical mask compiles and runs; an anvl `bool` mask raises the
  eager-only error.
- Errors: `NA` in a mask, wrong-length mask, bare logical vector (hint mentions `arr()`),
  wrong-shape flat mask.
- Regression: `x[array(integer(0)), ]` and `x[1:0, ]` return zero-sized arrays.

## Documentation

- `vignettes/subsetting.Rmd`: the "Mask" row of the support table becomes Dynamic = "eager
  only", Static = "Yes"; replace the "not supported" paragraph with a mask section covering
  both forms, the `arr()` requirement, and the disambiguation rule.
- Roxygen on `nv_subset()` and `nv_subset_assign()`: document masks in `@param ...` and add
  an example.
- `NEWS.md`: one entry for mask support, one for the zero-sized subsetting fix.

## Out of scope

- Negative indexing (`x[-1]`), still unsupported.
- Masks under `jit()` sourced from anvl arrays -- fundamentally impossible with static
  shapes; the existing masking-pattern workaround in `vignettes/static_shapes.Rmd` remains
  the answer.
