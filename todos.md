# Todos

What is left from the hardcoded-dtype sweep (`"f32"` / `"i32"` literals that
should consult the configured default dtypes).

## Open: two `nan_rm` branches commit at the active default float

Both are the same shape — an `i32` count meets a bare R **double**, which
crosses categories and so realizes at `default_dtype_r("double")` instead of
yielding to the operand. The `nan_rm` flag then silently changes the result's
data type.

- **`nv_var()` / `nv_sd()`, `R/api.R`** — `nv_max(0, count - correction)`.
  With an `f64` default, `nv_var(x_f32, nan_rm = TRUE)` returns `f64` while
  `nan_rm = FALSE` returns `f32`; `f16` / `bf16` inputs widen with nothing set
  at all. Fix: convert the count to `dtype(ssum)`, after which the `0` and
  `correction` yield within their category — which is exactly why the
  `nan_rm = FALSE` branch below is already right.
- **`nv_quantile()` / `nv_median()`, `R/api.R`** — the two branches of
  `n_valid_kd` disagree (`i32` vs `dtype(x)`), and `(n_valid_b - 1) * probs_b`
  then commits the `i32` branch at the default float, which propagates through
  `h` → `lo_f` → `frac` → `out`. Fix: convert the `nan_rm` branch's count to
  `dtype(x)` so both branches agree and `- 1` yields to it.

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
