# Data type and shape documentation: coverage

Tracks which help pages state their data type and shape behavior in the
parameter and return descriptions, following the conventions below. Update this
file in the same change that documents a function.

## Conventions

- The accepted data types are named with the vocabulary from `?dtypes`: *any*,
  *numeric*, *integer*, *integerish*, *signed numeric*, *float*, *boolean* —
  supplied through the `dtypes` template variable where a template applies.
- What an R value does is stated once per page: it assumes the data type of the
  operands it meets, or commits to its
  [default data type][default_dtypes] when nothing claims it. Never name the
  default's concrete value, since defaults will become configurable.
- Templates: `param_unary_x` (single arrayish operand), `params_prim_lhs_rhs` /
  `params_lhs_rhs` (binary), `params_reduce` + `return_reduce` (reductions).
  `roxy_agree("x", "update")` is the inline helper for a primitive whose
  operands must reach one data type.
- Return values say the resulting data type and shape, not "the same as the
  input" where an input may be a bare R value.
- Examples show working calls only, each with a one-line comment, and never
  state what a default data type is.

Every claim is checked against the running package before it is written.

## Primitives

### Covered (99 of 100)

Elementwise binary: `prim_add` `prim_sub` `prim_mul` `prim_div` `prim_pow`
`prim_remainder` `prim_max` `prim_min` `prim_atan2` `prim_and` `prim_or`
`prim_xor` `prim_shift_left` `prim_shift_right_logical`
`prim_shift_right_arithmetic` `prim_eq` `prim_ne` `prim_gt` `prim_ge`
`prim_lt` `prim_le`

Elementwise unary: `prim_abs` `prim_sign` `prim_negate` `prim_not`
`prim_popcnt` `prim_sqrt` `prim_rsqrt` `prim_cbrt` `prim_exp` `prim_expm1`
`prim_log` `prim_log1p` `prim_logistic` `prim_sin` `prim_cos` `prim_tan`
`prim_sinh` `prim_cosh` `prim_tanh` `prim_asin` `prim_acos` `prim_atan`
`prim_asinh` `prim_acosh` `prim_atanh` `prim_floor` `prim_ceil` `prim_round`
`prim_erf` `prim_erfc` `prim_erf_inv` `prim_digamma` `prim_lgamma`
`prim_is_finite`

Reductions and scans: `prim_reduce_sum` `prim_reduce_prod` `prim_reduce_max`
`prim_reduce_min` `prim_reduce_any` `prim_reduce_all` `prim_reduce`
`prim_cumsum` `prim_cumprod`

Operands that must agree: `prim_clamp` `prim_pad` `prim_ifelse`
`prim_dynamic_update_slice` `prim_scatter` `prim_polygamma`
`prim_triangular_solve` `prim_convolution` `prim_dot_general`

Shape and data type changing, single operand: `prim_transpose`
`prim_reshape` `prim_broadcast_in_axes` `prim_reverse` `prim_convert`
`prim_bitcast_convert` `prim_print`

Slicing, gathering, indices and constructors: `prim_static_slice`
`prim_dynamic_slice` `prim_gather` `prim_top_k` `prim_argmax` `prim_argmin`
`prim_cummax` `prim_cummin` `prim_fill` `prim_iota` `prim_concatenate`

Higher order, promoting nothing on purpose: `prim_sort` `prim_if` `prim_while`

Matrix decompositions: `prim_chol` `prim_qr` `prim_lu` `prim_svd` `prim_eigh`

### To do

None. Every primitive except `prim_rng_bit_generator` is covered.

## API functions

### Covered (74)

Elementwise binary: `nv_add` `nv_sub` `nv_mul` `nv_div` `nv_pow`
`nv_remainder` `nv_mod` `nv_max` `nv_min` `nv_atan2` `nv_and` `nv_or` `nv_xor`
`nv_shift_left` `nv_shift_right_logical` `nv_shift_right_arithmetic` `nv_eq`
`nv_ne` `nv_gt` `nv_ge` `nv_lt` `nv_le`

Elementwise unary: `nv_abs` `nv_sign` `nv_negate` `nv_not` `nv_popcnt`
`nv_sqrt` `nv_rsqrt` `nv_cbrt` `nv_exp` `nv_expm1` `nv_log` `nv_log1p`
`nv_log2` `nv_log10` `nv_logistic` `nv_sin` `nv_cos` `nv_tan` `nv_sinh`
`nv_cosh` `nv_tanh` `nv_asin` `nv_acos` `nv_atan` `nv_asinh` `nv_acosh`
`nv_atanh` `nv_floor` `nv_ceiling` `nv_trunc` `nv_round` `nv_erf` `nv_erfc`
`nv_erf_inv` `nv_digamma` `nv_lgamma` `nv_is_finite` `nv_is_nan`
`nv_is_infinite`

Reductions and scans: `nv_reduce_sum` `nv_reduce_prod` `nv_reduce_max`
`nv_reduce_min` `nv_reduce_any` `nv_reduce_all` `nv_mean` `nv_var` `nv_sd`
`nv_cumsum` `nv_cumprod` `nv_cummax` `nv_cummin`

Sequences: `nv_seq` `nv_seq_like` `nv_linspace` `nv_linspace_like`

### To do (77)

- Shape and layout: `nv_transpose` `nv_reshape` `nv_flatten` `nv_squeeze`
  `nv_unsqueeze` `nv_reverse` `nv_broadcast_to` `nv_broadcast_arrays`
  `nv_broadcast_scalars` `nv_concatenate` `nv_rbind` `nv_cbind` `nv_matrix`
- Subsetting: `nv_subset` `nv_static_slice` `nv_select` `nv_diag`
  `nv_extract_diag` `nv_tril` `nv_triu` `nv_lower_tri` `nv_lower_tri_like`
  `nv_upper_tri` `nv_upper_tri_like`
- Data types: `nv_convert` `nv_bitcast_convert` `nv_promote_to_common`
  `nv_aval`
- Constructors: `nv_array` `nv_array_like` `nv_scalar` `nv_scalar_like`
  `nv_fill` `nv_fill_like` `nv_empty` `nv_empty_like` `nv_iota` `nv_iota_like`
  `nv_eye` `nv_eye_like`
- Operands that must agree: `nv_clamp` `nv_pad` `nv_ifelse` `nv_polygamma`
- Order statistics and indices: `nv_sort` `nv_argsort` `nv_top_k` `nv_argmax`
  `nv_argmin` `nv_median` `nv_quantile`
- Linear algebra: `nv_matmul` `nv_outer` `nv_crossprod` `nv_tcrossprod`
  `nv_trace` `nv_det` `nv_determinant` `nv_inv` `nv_solve`
  `nv_triangular_solve` `nv_chol` `nv_qr` `nv_lu` `nv_svd` `nv_eigh`
- Convolutions: `nv_conv1d` `nv_conv2d` `nv_conv3d`
- Control flow: `nv_if` `nv_while`
- Device and IO: `nv_device` `nv_print` `nv_save` `nv_read` `nv_serialize`
  `nv_unserialize`

## Noticed while documenting

Behavior worth a second look, found by checking claims against the running
package. None of it is fixed here — this is documentation work.

- `prim_bitcast_convert()`'s error for a too-small last axis says "The last
  dimension of `x` must be 2", using *dimension* where the project's
  terminology is *axis* / *axis size*.
- `nv_seq()` rejects `bool` but takes every other data type, while
  `nv_linspace()` is restricted to `f32` and `f64` by `assert_float_dtype()`,
  which is deliberately narrower than `is_dtype_float()`. Two different notions
  of "floating-point" are in play across the API.
- `nv_linspace()` defaults to `f32` regardless of backend, while `nv_array()`
  documents a backend-dependent default (`f64` on `"quickr"`). The sequence
  constructors do not follow the backend.
- List returns are inconsistently named: `prim_qr()` and `prim_svd()` return
  named lists (`Q`/`R`, `d`/`u`/`v`), while `prim_top_k()`, `prim_cummax()` and
  `prim_cummin()` return unnamed two-element lists, so callers index them
  positionally.
- `prim_dynamic_slice()` does not check the data type of its start indices at
  the anvl level. A float index reaches the backend and fails with a raw
  StableHLO message ("operand #1 must be variadic of 0D tensor of ... integer
  values"), where `prim_top_k()` and friends give a `cli` error.
- `prim_fill()` with a negative value at an unsigned data type also fails in
  the backend ("expected unsigned integer elements, but parsed negative
  value") rather than in an anvl check.
- `prim_iota()` rejects boolean, as `nv_seq()` does, but `prim_fill()` accepts
  it — the constructors do not agree on which data types they cover.

## Cannot be verified in this container

The CPU plugin baked into the dev image registers no FFI handlers for these, so
the calls fail with "No FFI handler registered". Their data type and shape
constraints are checked through the assertions that fire before the call;
anything about the returned values rests on the package's own tests. Downloading
a matching plugin was attempted and made things worse (the fresh plugin
crashes R), so the image's plugin is left in place.

- `prim_print` (`print_tensor`)
- `prim_qr` (`geqrf`), `prim_lu`, `prim_svd`, `prim_eigh`, and the `nv_*`
  functions built on them

## Out of scope

Owned by the maintainer, not to be touched by this effort: the random number
generators and the distribution and density functions —
`prim_rng_bit_generator` `nv_rng_state` `nv_rnorm` `nv_runif` `nv_rbinom`
`nv_sample` `nv_sample_int` `nv_dnorm` `nv_pnorm` `nv_qnorm`
