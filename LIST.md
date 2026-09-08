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
- Templates: `param_unary_x` for a single arrayish operand, or
  `param_unary_x_must` where the phrase names the R side and the default
  sentence would be redundant; `params_prim_lhs_rhs` / `params_lhs_rhs`
  (binary); `params_reduce` + `return_reduce` (reductions).
  `roxy_agree("x", "update")` is the inline helper for a primitive whose
  operands must reach one data type; it goes on the primary operand, and the
  other parameters point at it ("Shares `x`'s data type -- see `x`."). An
  `nv_*` function that promotes with `promote_like("x")` instead
  (`nv_clamp()`, `nv_pad()`, `nv_subset_assign()`) writes `x` inline, names
  the accepted data types there, and puts the promotion sentence on the sibling.
- Return values say the resulting data type and shape, not "the same as the
  input" where an input may be a bare R value, and state their type in
  parentheses the way parameters do: `@return ([`arrayish`])\cr`.
- The vocabulary itself is defined once, in `man-roxygen/section_dtype_words.R`,
  and shown on both `?arrayish` (where every parameter's type links) and
  `?dtypes`.
- Examples show working calls only and never state what a default data type is.
  Where a call demonstrates something about data types or shapes -- a
  promotion, an R value taking a data type, an index output, a shape change --
  a one-line comment says what comes out; a plain `prim_exp(x)` needs none.
- A primitive modelled on a StableHLO op says so through
  `r roxy_spec("<op>")`, which names the `hlo_*` function and links the op's
  entry in the specification, and one modelled on a CHLO op through
  `r roxy_spec_chlo("<op>")`, which links the generated CHLO reference instead
  -- CHLO ops (`acos`, `erf`, `top_k`, ...) are not in the StableHLO
  specification. Anything a block adds beyond that -- a reducer, a comparison
  direction, a comparator -- follows as its own sentence. The primitives backed
  by a custom call (`prim_qr()`, `prim_lu()`, `prim_svd()`, `prim_eigh()`)
  keep their hand-written section, since no spec op describes them.
- The `@description` says what the function computes, not how its inputs are
  converted: promotion, broadcasting and data type constraints belong in the
  parameters and the return value. The exceptions are functions whose *purpose*
  is a conversion (`nv_convert()`, `nv_bitcast_convert()`,
  `nv_promote_to_common()`, the `nv_broadcast_*()` family).

Every claim is checked against the running package before it is written.

## Primitives

### Covered (100 of 100)

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

None. `prim_rng_bit_generator` is covered too: the algorithm names, the state
whose length depends on the choice, and the data types it can draw.
`prim_convolution` was the last one missing and now has its data types, its
return, a rules section and a worked example.

## API functions

### Covered (all exported pages except the maintainer's, below)

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

Operands that must agree: `nv_clamp` `nv_pad` `nv_ifelse` `nv_polygamma`

Order statistics and indices: `nv_sort` `nv_argsort` `nv_top_k` `nv_argmax`
`nv_argmin` `nv_median` `nv_quantile`

Constructors: the `AnvlArray` page (`nv_array`, `nv_scalar`, `nv_matrix`,
`nv_empty`), `nv_fill`, `nv_iota`, `nv_eye` and their `_like` variants
(`nv_array_like`, `nv_scalar_like`, `nv_empty_like`, `nv_fill_like`,
`nv_iota_like`, `nv_eye_like`, `nv_lower_tri_like`, `nv_upper_tri_like`),
which share their parent's page

Shape and layout: `nv_flatten` `nv_bind` `nv_rbind` `nv_cbind` `nv_diag`
`nv_extract_diag` `nv_subset` `nv_subset_assign` `nv_reshape`
`nv_transpose` `nv_reverse`
`nv_select` `nv_squeeze` `nv_unsqueeze` `nv_static_slice` `nv_concatenate`
`nv_lower_tri` `nv_upper_tri` `nv_tril` `nv_triu`

Data types and broadcasting: `nv_convert` `nv_bitcast_convert`
`nv_promote_to_common` `nv_broadcast_scalars` `nv_broadcast_arrays`
`nv_broadcast_to` `nv_aval`

Linear algebra: `nv_matmul` `nv_outer` `nv_crossprod` `nv_tcrossprod`
`nv_trace` `nv_inv` `nv_det` `nv_determinant` `nv_chol` `nv_solve`
`nv_triangular_solve` `nv_lu`, and `nv_qr` / `nv_svd` / `nv_eigh`, which
inherit their primitives'

Convolution: `nv_conv1d` `nv_conv2d` `nv_conv3d`

Control flow: `nv_if` `nv_while`

IO: `nv_save` `nv_read` `nv_serialize` `nv_unserialize` `nv_print`

### To do

None. `nv_device` has no array data type to state, so it was left alone, and
the distribution and RNG pages are the maintainer's (see Out of scope).

## Review

A review agent checked the pass for consistency; its report is `FINDINGS.md`.
Everything it found that was *wrong* is fixed here:

- False statements: `prim_static_slice`'s example comments claimed an exclusive
  limit, `prim_argmin` inherited an `axis` describing the maximum, the `nv_conv*`
  returns named a `kernel` argument that is called `weight`, `nv_polygamma`
  claimed integer inputs are converted where an all-integer call is refused,
  `nv_trace` claimed the input's data type for a boolean input (it counts at
  `i32`), and `prim_dynamic_update_slice` inherited a clamp formula with a
  `slice_sizes` argument it does not have.
- Dead links: the fifteen CHLO ops (`acos`, `erf`, `top_k`, ...) pointed at
  StableHLO spec anchors that do not exist, so they go through the new
  `roxy_spec_chlo()`; `prim_fill` linked `spec#tensor` where the op is
  `constant`; and seven references called a nonexistent `nv_shape()`.
- Coverage the sweep had missed: `prim_convolution` (the whole page), thirteen
  `nv_*` pages still on a bare "Input array.", `nv_matmul`'s and `nv_fill`'s
  empty return and `dtype`, and `nv_chol`'s accepted data types.
- Consistency: `nv_array` no longer spells out the concrete defaults (`?dtypes`
  carries them, including the backend caveat), `nv_and`/`nv_or`/`nv_xor`/`nv_not`
  are described as bitwise rather than logical, the boolean and integer
  constraints moved out of the type slot into prose, "dtype" and
  "floating-point" as prose are gone, and the unused templates (`param_x`,
  `param_dtype`, `param_prim_x_any`, the three `section_shapes_*`) were removed
  along with the skill instructions that recommended them.
- Grammar and typos from the rewrites: `prim_while`'s dangling clause,
  `prim_bitcast_convert`'s subject-less sentence, `as_array`'s orphaned "Of
  length 1.", "whatever the input's is", `nv_flatten`'s title, and the
  remaining "1-based" and "dimension" wording.

Every example on every page this pass changed was executed; the only ones that
could not run are the twelve listed under "Cannot be verified in this
container".

Two of its observations were deliberately left alone: the `try()` on
`?promotion_rule` demonstrates the narrowing error that page is about, and
"NumPy-style broadcasting" / "Torch-style NCW layout" name a layout convention
rather than claiming a framework match.

## Noticed while documenting

Behavior worth a second look, found by checking claims against the running
package. This is documentation work, so it is left alone unless a doc claim
would otherwise be false; the items marked **Fixed here** are the exceptions.

- `prim_bitcast_convert()`'s error for a too-small last axis says "The last
  dimension of `x` must be 2", using *dimension* where the project's
  terminology is *axis* / *axis size*.
- **Fixed here.** Two notions of "floating-point" were in play:
  `assert_float_dtype()` accepted only `f32` and `f64`, where `is_dtype_float()`
  covers the whole float category. It now asks `is_dtype_float()`, so `f16` and
  `bf16` count as floats everywhere, and the RNG -- which assembles a float from
  random bits and so really does need 32 or 64 of them -- got its own
  `assert_rng_float_dtype()`.
- **Fixed here.** List returns were named inconsistently: `prim_qr()` and
  `prim_svd()` returned named lists (`Q`/`R`, `d`/`u`/`v`) while `prim_top_k()`,
  `prim_cummax()`, `prim_cummin()`, `prim_rng_bit_generator()` and the `nv_*`
  samplers returned unnamed two-element ones. They are all named now
  (`values`/`indices`, `state`/`values`); positional indexing still works.
- On the `"quickr"` backend, a result's data type does not come from the graph
  at all: `quickr_restore_leaf()` (`R/graph-to-quickr.R:42`) calls the backend's
  `new_data()` with `dtype = NULL`, so the label is re-derived from the R storage
  type of the returned vector -- a double becomes `f64`, an integer `i32`. Since
  the code generator represents `f32` as a double, everything narrower than
  `f64` silently comes back as `f64`: `nv_fill(1.5, 2L, dtype = "f32")`,
  `nv_convert(x_f64, "f32")`, `nv_add(x_f32, y_f32)`, even `jit(identity)` of an
  `f32` input. Data types quickr does not represent at all (`i8`, `ui8`, ...)
  are properly rejected at lowering; `f32` is the one that passes and is
  relabelled. `nv_array()` and `nv_scalar()` keep an explicit `f32` because they
  call `new_data()` directly and compile nothing.
- Two lowerings on `"quickr"` ignore their output data type outright:
  `nv_iota(axis = 1L, shape = 3L, dtype = "f32")` returns `i32 [1 2 3]`, and
  `nv_linspace(0, 1, 3L)` returns `i32 [0 0 1]` -- the wrong data type *and* the
  wrong values (integer division), with `dtype = "f64"` making no difference.
  Real bugs, not documentation gaps.
- **Fixed here.** `nv_runif(shape, state, min = a, max = a)` returned a bare
  array where every other sampler returns `list(state, values)`, so a caller
  doing `out$values` broke on the degenerate interval. It now returns the pair,
  with the state unchanged since no draw is made.
- **Fixed here.** `nv_rbinom(..., dtype = "bool")` returned `bool` for
  `size = 1` but silently `i32` for `size > 1`, since counting the successes
  goes through `nv_reduce_sum()`, and `nv_sample_int(..., dtype = "bool")`
  collapsed every drawn index to `TRUE`. Both reject a boolean data type now,
  through the new `assert_numeric_dtype()` -- which is what their docs already
  claimed, since *numeric* excludes boolean.
- `prim_dynamic_slice()` does not check the data type of its start indices at
  the anvl level. A float index reaches the backend and fails with a raw
  StableHLO message ("operand #1 must be variadic of 0D tensor of ... integer
  values"), where `prim_top_k()` and friends give a `cli` error.
- `prim_fill()` with a negative value at an unsigned data type also fails in
  the backend ("expected unsigned integer elements, but parsed negative
  value") rather than in an anvl check.
- `nv_polygamma()`'s `n` used to be a static argument, so it rejected an
  `AnvlArray` while `prim_polygamma()`'s `n` accepted one. Nothing in the body
  needed the static value, and JAX treats `n` as a traced array too, so the
  annotation was dropped and both layers now agree. Note that JAX additionally
  requires `n` to arrive with an integer dtype; anvl cannot copy that check
  without rejecting `nv_polygamma(1, x)`, since R's `1` is a double.
- `nv_polygamma()` promotes an integer or boolean `x` to a float, where
  `prim_polygamma()` refuses anything but a float. Same for `nv_median()` /
  `nv_quantile()` against a float-only reading of their primitives.
- `nv_pad()` and `nv_clamp()` refuse an R double for an integer `x`
  (`nv_pad(x_i32, 0)` errors, `0L` works), because `promote_like("x")` keeps
  the literal in its own category. Easy to trip over, and the old `nv_pad()`
  text implied either literal would do.

## Cannot be verified in this container

The CPU plugin baked into the dev image registers no FFI handlers for these, so
the calls fail with "No FFI handler registered". Their data type and shape
constraints are checked through the assertions that fire before the call;
anything about the returned values rests on the package's own tests. Downloading
a matching plugin was attempted and made things worse (the fresh plugin
crashes R), so the image's plugin is left in place.

- `prim_print` (`print_tensor`), `nv_print`
- `prim_qr` (`geqrf`), `prim_lu`, `prim_svd`, `prim_eigh`, and the `nv_*`
  functions built on them (`nv_qr`, `nv_lu`, `nv_svd`, `nv_eigh`, `nv_det`,
  `nv_determinant`, `nv_inv`, `nv_solve`)

These are exactly the twelve help pages whose examples cannot run here; every
other page changed by this pass had its examples executed.

## Out of scope

Owned by the maintainer, not to be touched by this effort: the random number
generators and the distribution and density functions —
`nv_rng_state` `nv_rnorm` `nv_runif` `nv_rbinom` `nv_sample` `nv_sample_int`
`nv_dnorm` `nv_pnorm` `nv_qnorm`. `prim_rng_bit_generator` was documented on
request; the `param_initial_state` template it used to share with those
`nv_*` functions was left alone, since they always drive `"THREE_FRY"` and so
always take a two-element state.

On request, those pages did get the promotion and broadcasting behavior they
were missing, and nothing else: which data types the operands accept, that
`mean` and `sd` are brought to `x`/`q`/`p` (widening only) and broadcast when
scalar, that `min`/`max`/`size`/`prob` are plain R numbers and so promote
nothing, and the named `state`/`values` return. `man-roxygen/param_dtype.R`
rendered a bare "Data type." for two of them and went away: `dtype` differs too
much per function (a 32- or 64-bit float for `nv_runif()`, any numeric data type
for `nv_rbinom()`) to share one sentence.
