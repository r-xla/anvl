# Todos

What is left from the hardcoded-dtype sweep (`"f32"` / `"i32"` literals that
should consult the configured default dtypes).

Nothing open.

## Done: two `nan_rm` branches commit at the active default float

Both were the same shape — an `i32` count met a bare R **double**, which
crosses categories and so realized at `default_dtype_r("double")` instead of
yielding to the operand, so the `nan_rm` flag silently changed the result's
data type. Both now count at the operand's own data type, which keeps the
arithmetic that follows inside its category.

- **`nv_var()` / `nv_sd()`, `R/api.R`** — the valid-value count is built at
  `dtype(ssum)`, so the `0` and `correction` in
  `nv_max(0, count - correction)` yield to it. An R *integer* meeting a float
  array yields to the array, so only the R double `0` was ever the problem.
- **`nv_quantile()` / `nv_median()`, `R/api.R`** — the `nan_rm` branch of
  `n_valid_kd` is built at `dtype(x)`, so both branches agree and the `- 1` in
  `(n_valid_b - 1) * probs_b` yields to it rather than committing `h` → `lo_f`
  → `frac` → `out` at the default float.

Each is pinned by a `"does not let nan_rm change the data type"` test in
`tests/testthat/test-api.R`, asserting that `nan_rm = TRUE` and
`nan_rm = FALSE` agree and that an `f32` operand stays `f32` under
`with_default_dtypes(c(float = "f64", int = "i64"))`.

## Done: index data types

Every index anvl hands back follows `default_dtypes()$int` rather than always
being `i32`: `prim_argmax` / `prim_argmin`, `prim_cummax` / `prim_cummin`,
`nv_argsort`, `prim_top_k`'s indices, and `prim_lu`'s `pivots` /
`permutation`. Where the backend fixes the width — `hlo_top_k`'s `i32`
indices, LAPACK's 32-bit `ipiv` — the operation is built as before and its
index outputs converted afterwards.

## Done: the rest of the sweep

- `gather_clamp_indices()` (`R/utils.R`) builds its clamp bound at the default
  integer instead of a hardcoded `i64` (which quickr cannot lower).
- `nv_unif_rand()` (`R/api-rng.R`) takes a required `dtype` instead of
  defaulting to `"f64"`.
- `nv_pnorm()` / `nv_qnorm()` assert `f32` or `f64` rather than letting a
  half-precision operand fall into the `f64` coefficient branch.
- `nv_var()` / `nv_sd()` coerce `correction` with
  `assert_int(correction, coerce = TRUE)`.
