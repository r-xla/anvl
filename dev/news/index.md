# Changelog

## anvl (development version)

### Breaking changes

- The element-wise
  [`nv_max()`](https://r-xla.github.io/anvl/dev/reference/nv_max.md) /
  [`nv_min()`](https://r-xla.github.io/anvl/dev/reference/nv_min.md) and
  [`prim_max()`](https://r-xla.github.io/anvl/dev/reference/prim_max.md)
  /
  [`prim_min()`](https://r-xla.github.io/anvl/dev/reference/prim_min.md)
  are now
  [`nv_pmax()`](https://r-xla.github.io/anvl/dev/reference/nv_pmax.md) /
  [`nv_pmin()`](https://r-xla.github.io/anvl/dev/reference/nv_pmin.md)
  and
  [`prim_pmax()`](https://r-xla.github.io/anvl/dev/reference/prim_pmax.md)
  /
  [`prim_pmin()`](https://r-xla.github.io/anvl/dev/reference/prim_pmin.md),
  following [`base::pmax()`](https://rdrr.io/r/base/Extremes.html) /
  [`base::pmin()`](https://rdrr.io/r/base/Extremes.html).
- The reductions lost their `reduce_` prefix: write
  [`nv_sum()`](https://r-xla.github.io/anvl/dev/reference/nv_sum.md),
  [`nv_prod()`](https://r-xla.github.io/anvl/dev/reference/nv_prod.md),
  [`nv_max()`](https://r-xla.github.io/anvl/dev/reference/nv_max.md),
  [`nv_min()`](https://r-xla.github.io/anvl/dev/reference/nv_min.md),
  [`nv_any()`](https://r-xla.github.io/anvl/dev/reference/nv_any.md),
  [`nv_all()`](https://r-xla.github.io/anvl/dev/reference/nv_all.md) and
  the matching `prim_*()` instead of `nv_reduce_sum()` and friends.
  [`prim_reduce()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce.md)
  keeps its name.
- `nv_argsort()` is now
  [`nv_order()`](https://r-xla.github.io/anvl/dev/reference/nv_order.md),
  `nv_reverse()` / `prim_reverse()` are
  [`nv_rev()`](https://r-xla.github.io/anvl/dev/reference/nv_rev.md) /
  [`prim_rev()`](https://r-xla.github.io/anvl/dev/reference/prim_rev.md),
  `nv_argmax()` / `nv_argmin()` and their primitives are
  [`nv_which_max()`](https://r-xla.github.io/anvl/dev/reference/nv_which_max.md)
  /
  [`nv_which_min()`](https://r-xla.github.io/anvl/dev/reference/nv_which_min.md)
  and
  [`prim_which_max()`](https://r-xla.github.io/anvl/dev/reference/prim_which_max.md)
  /
  [`prim_which_min()`](https://r-xla.github.io/anvl/dev/reference/prim_which_min.md),
  and `prim_ceil()` is
  [`prim_ceiling()`](https://r-xla.github.io/anvl/dev/reference/prim_ceiling.md).
- `nv_polygamma()` / `prim_polygamma()` are now
  [`nv_psigamma()`](https://r-xla.github.io/anvl/dev/reference/nv_psigamma.md)
  /
  [`prim_psigamma()`](https://r-xla.github.io/anvl/dev/reference/prim_psigamma.md)
  and take `(x, deriv)` like
  [`base::psigamma()`](https://rdrr.io/r/base/Special.html) instead of
  `(n, x)`; `deriv` defaults to `0`.
- `nv_logistic()` / `prim_logistic()` are now
  [`nv_plogis()`](https://r-xla.github.io/anvl/dev/reference/nv_plogis.md)
  /
  [`prim_plogis()`](https://r-xla.github.io/anvl/dev/reference/prim_plogis.md).
- The general transpose is now `nv_aperm(x, perm)`, matching
  [`base::aperm()`](https://rdrr.io/r/base/aperm.html);
  [`nv_transpose()`](https://r-xla.github.io/anvl/dev/reference/nv_aperm.md)
  stays as another spelling of it. It and
  [`prim_transpose()`](https://r-xla.github.io/anvl/dev/reference/prim_transpose.md)
  call their second argument `perm` instead of `permutation`.
- [`default_device()`](https://r-xla.github.io/anvl/dev/reference/default_device.md)
  no longer follows `PJRT_PLATFORM`; set `ANVL_DEFAULT_DEVICE` or the
  `anvl.default_device` option instead.
- [`prim_reshape()`](https://r-xla.github.io/anvl/dev/reference/prim_reshape.md)
  and
  [`nv_reshape()`](https://r-xla.github.io/anvl/dev/reference/nv_reshape.md),
  and with them
  [`nv_flatten()`](https://r-xla.github.io/anvl/dev/reference/nv_flatten.md)
  and every `axis = NULL` flattening default, are now column-major like
  base R.
- [`prim_bitcast_convert()`](https://r-xla.github.io/anvl/dev/reference/prim_bitcast_convert.md)
  puts the axis holding an element’s pieces first rather than last when
  the two data types differ in width, so the pieces of one element are
  adjacent in the order
  [`nv_flatten()`](https://r-xla.github.io/anvl/dev/reference/nv_flatten.md)
  reads and a narrowing conversion lays the bytes out the way
  [`as_raw()`](https://r-xla.github.io/anvl/dev/reference/as_raw.md)
  writes them.
- [`nv_broadcast_to()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_to.md)
  and
  [`nv_broadcast_arrays()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_arrays.md)
  align axes from the first instead of the last: a shorter shape gets
  size-1 axes appended, so a length-`nrow` vector broadcasts against a
  matrix where a length-`ncol` one no longer does. Write
  [`prim_broadcast_in_axes()`](https://r-xla.github.io/anvl/dev/reference/prim_broadcast_in_axes.md)
  with an explicit axis mapping for the previous right-aligned behavior.
- The `@jit` roxygen tag was removed; wrap functions in
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) at the
  definition instead.
- A primitive is now named after the `prim_*()` function that exports it
  rather than the StableHLO op it lowers to.
- [`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md),
  [`nv_cummax()`](https://r-xla.github.io/anvl/dev/reference/nv_cummax.md)
  and
  [`nv_cummin()`](https://r-xla.github.io/anvl/dev/reference/nv_cummin.md)
  take `indices` instead of `with_indices`, spelling it the way
  [`prim_top_k()`](https://r-xla.github.io/anvl/dev/reference/prim_top_k.md)
  does.
- [`nv_clamp()`](https://r-xla.github.io/anvl/dev/reference/nv_clamp.md)
  and
  [`prim_clamp()`](https://r-xla.github.io/anvl/dev/reference/prim_clamp.md)
  take `(x, min, max)` instead of `(min_val, x, max_val)`.
- [`nv_seq()`](https://r-xla.github.io/anvl/dev/reference/nv_seq.md) /
  [`nv_seq_like()`](https://r-xla.github.io/anvl/dev/reference/nv_seq.md)
  call their bounds `from` and `to`, like
  [`base::seq()`](https://rdrr.io/r/base/seq.html).
- [`nv_ifelse()`](https://r-xla.github.io/anvl/dev/reference/nv_ifelse.md)
  and
  [`prim_ifelse()`](https://r-xla.github.io/anvl/dev/reference/prim_ifelse.md)
  take `(test, yes, no)`, like
  [`base::ifelse()`](https://rdrr.io/r/base/ifelse.html).
  [`prim_if()`](https://r-xla.github.io/anvl/dev/reference/prim_if.md) /
  [`nv_if()`](https://r-xla.github.io/anvl/dev/reference/nv_if.md) keep
  `(pred, true, false)`: they mirror the `if` construct, not
  [`ifelse()`](https://rdrr.io/r/base/ifelse.html).
- [`nv_scan()`](https://r-xla.github.io/anvl/dev/reference/nv_scan.md)
  takes `(init, xs, body)`, the order
  [`prim_scan()`](https://r-xla.github.io/anvl/dev/reference/prim_scan.md)
  uses, and both call the trip count `steps` instead of `length`.
- [`prim_sort()`](https://r-xla.github.io/anvl/dev/reference/prim_sort.md)’s
  `descending` / `is_stable` are now `decreasing` / `stable`, as in
  [`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md).
- [`prim_top_k()`](https://r-xla.github.io/anvl/dev/reference/prim_top_k.md)’s
  `indices` no longer has a default; pass it explicitly.
- [`prim_reduce()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce.md)
  takes `(x, init, axes, reducer, drop)`: `reductor` is now `reducer`,
  and it comes before `drop`.
- [`prim_static_slice()`](https://r-xla.github.io/anvl/dev/reference/prim_static_slice.md)
  /
  [`nv_static_slice()`](https://r-xla.github.io/anvl/dev/reference/nv_static_slice.md)
  call their (inclusive) upper bound `end_indices` instead of
  `limit_indices`.
- [`nv_crossprod()`](https://r-xla.github.io/anvl/dev/reference/nv_crossprod.md),
  [`nv_tcrossprod()`](https://r-xla.github.io/anvl/dev/reference/nv_tcrossprod.md),
  [`nv_outer()`](https://r-xla.github.io/anvl/dev/reference/nv_outer.md)
  and
  [`nv_matmul()`](https://r-xla.github.io/anvl/dev/reference/nv_matmul.md)
  call their operands `x` and `y`, as base R does. So do
  [`nv_pow()`](https://r-xla.github.io/anvl/dev/reference/nv_pow.md),
  [`nv_remainder()`](https://r-xla.github.io/anvl/dev/reference/nv_remainder.md),
  [`nv_xor()`](https://r-xla.github.io/anvl/dev/reference/nv_xor.md) and
  their primitives, instead of `lhs` / `rhs`.
- [`nv_linspace()`](https://r-xla.github.io/anvl/dev/reference/nv_linspace.md)
  /
  [`nv_linspace_like()`](https://r-xla.github.io/anvl/dev/reference/nv_linspace.md)
  take `(from, to, length_out)` instead of `(start, end, steps)`, like
  [`base::seq()`](https://rdrr.io/r/base/seq.html).
- [`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md)
  and
  [`nv_rnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md)
  take `dtype` after the distribution parameters, as
  [`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md)
  and
  [`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md)
  do.
- [`prim_convolution()`](https://r-xla.github.io/anvl/dev/reference/prim_convolution.md)
  calls its input axes `x_batch_axis`, `x_feature_axis` and
  `x_spatial_axes` instead of `input_*`, and
  [`nv_conv1d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv1d.md)
  /
  [`nv_conv2d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv2d.md)
  /
  [`nv_conv3d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv3d.md)
  call their second operand `kernel` instead of `weight`, as
  [`prim_convolution()`](https://r-xla.github.io/anvl/dev/reference/prim_convolution.md)
  does.
- [`nv_pad()`](https://r-xla.github.io/anvl/dev/reference/nv_pad.md)
  takes `(x, value, low, high, interior)` instead of
  `(x, padding_value, edge_padding_low, edge_padding_high, interior_padding)`;
  [`prim_pad()`](https://r-xla.github.io/anvl/dev/reference/prim_pad.md)
  keeps the StableHLO names.
- [`nv_triangular_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_triangular_solve.md)
  takes `left`, `unit_diag` and `transpose` instead of `left_side`,
  `unit_diagonal` and `transpose_a`;
  [`prim_triangular_solve()`](https://r-xla.github.io/anvl/dev/reference/prim_triangular_solve.md)
  keeps the StableHLO names.
- [`prim_scatter()`](https://r-xla.github.io/anvl/dev/reference/prim_scatter.md)’s
  `update_computation` is now `update_fn`.
- [`nv_lower_tri_like()`](https://r-xla.github.io/anvl/dev/reference/nv_lower_tri.md)
  /
  [`nv_upper_tri_like()`](https://r-xla.github.io/anvl/dev/reference/nv_upper_tri.md)
  take `shape` before `diagonal`, as
  [`nv_lower_tri()`](https://r-xla.github.io/anvl/dev/reference/nv_lower_tri.md)
  /
  [`nv_upper_tri()`](https://r-xla.github.io/anvl/dev/reference/nv_upper_tri.md)
  do.
- [`nv_quantile()`](https://r-xla.github.io/anvl/dev/reference/nv_quantile.md),
  [`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md)
  and their [`quantile()`](https://rdrr.io/r/stats/quantile.html) /
  [`median()`](https://rdrr.io/r/stats/median.html) methods call
  `interpolation` `method`.
- [`nv_unsqueeze()`](https://r-xla.github.io/anvl/dev/reference/nv_unsqueeze.md)
  takes `axes`, inserting several axes at once.
- [`nv_atan2()`](https://r-xla.github.io/anvl/dev/reference/nv_atan2.md)
  and
  [`prim_atan2()`](https://r-xla.github.io/anvl/dev/reference/prim_atan2.md)
  take `(y, x)`, like
  [`base::atan2()`](https://rdrr.io/r/base/Trig.html).
- The array constructors spell their trailing arguments `shape`,
  `dtype`, `device` in that order:
  `nv_array(data, shape, dtype, device, byrow)`,
  `nv_empty(shape, dtype, device)`,
  [`nv_iota()`](https://r-xla.github.io/anvl/dev/reference/nv_iota.md) /
  [`prim_iota()`](https://r-xla.github.io/anvl/dev/reference/prim_iota.md)
  `(axis, shape, dtype, start, device)`, and likewise
  [`nv_array_like()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md),
  [`nv_empty_like()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  and
  [`nv_iota_like()`](https://r-xla.github.io/anvl/dev/reference/nv_iota.md).
- [`nv_shift_left()`](https://r-xla.github.io/anvl/dev/reference/nv_shift_left.md),
  [`nv_shift_right_logical()`](https://r-xla.github.io/anvl/dev/reference/nv_shift_right_logical.md),
  [`nv_shift_right_arithmetic()`](https://r-xla.github.io/anvl/dev/reference/nv_shift_right_arithmetic.md)
  and their primitives take `(x, shift)` instead of `(lhs, rhs)`. The
  result keeps `x`’s data type, which `shift` is brought to, instead of
  promoting both.
- The RNG functions
  ([`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md),
  [`nv_rnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md),
  [`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md),
  [`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md),
  [`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md),
  [`prim_rng_bit_generator()`](https://r-xla.github.io/anvl/dev/reference/prim_rng_bit_generator.md))
  call their state argument `state` instead of `initial_state`, matching
  the `state` element they return.
- [`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md)
  takes `axes` instead of `axis`, ranking the elements of several axes
  together, and `axes = NULL` (the default) now ranks over every axis
  where it used to take the last one. Write `axes = -1` for the old
  default.
- The type system of {anvl} was changed to avoid the problems reported
  in issue [\#373](https://github.com/r-xla/anvl/issues/373).
  Specifically, the ambiguity system was replaced with the `RData`
  system and a new system of rules for type promotions. With it, also
  the promotion behavior of various primitives and API functions was
  improved.
- An R value is now built directly at every data type of its own
  category, narrow and unsigned integers included, so one the data type
  cannot hold is an error instead of wrapping around: `x_ui8 + (-2L)`
  and `x_i8 + 300L` are refused. Write
  [`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md)
  on an array where the wraparound is what you want.
- [`as_array()`](https://r-xla.github.io/anvl/dev/reference/as_array.md)
  and the [`as.double()`](https://rdrr.io/r/base/double.html) /
  [`as.integer()`](https://rdrr.io/r/base/integer.html) /
  [`bit64::as.integer64()`](https://bit64.r-lib.org/reference/as.integer64.character.html)
  / [`as.logical()`](https://rdrr.io/r/base/logical.html) methods take
  `check = "warn"`, `"err"` or `FALSE` instead of a flag, following
  {pjrt}, and warn by default about a value R’s type cannot hold. Write
  `check = "err"` where you wrote `check = TRUE`, and `check = FALSE` to
  materialize silently.
- [`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  and
  [`nv_scalar()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  no longer take a `check` argument, following {pjrt}: what happens to
  an `NA` is fixed by the dtype it is built at, and the input is always
  scanned for values the requested dtype cannot hold. Call
  [`anyNA()`](https://rdrr.io/r/base/NA.html) on the data yourself if
  you want to hear about a missing value the dtype accepts.
- [`common_dtype()`](https://r-xla.github.io/anvl/dev/reference/common_dtype.md)
  now errors for `ui64` and a signed integer instead of returning `i64`,
  which could not hold every `ui64` value. Convert one of them with
  [`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md).
- `jit_eval()` was removed as it is no longer needed.
- [`nv_sum()`](https://r-xla.github.io/anvl/dev/reference/nv_sum.md),
  [`nv_prod()`](https://r-xla.github.io/anvl/dev/reference/nv_prod.md),
  [`nv_cumsum()`](https://r-xla.github.io/anvl/dev/reference/nv_cumsum.md)
  and
  [`nv_cumprod()`](https://r-xla.github.io/anvl/dev/reference/nv_cumprod.md)
  now accumulate a boolean array at the default integer data type
  instead of returning a boolean.
- [`as.vector()`](https://rdrr.io/r/base/vector.html) on an `AnvlArray`
  now only accepts `mode = "any"` (the default) and errors for any other
  `mode`.
- The `steps` argument of
  [`nv_seq()`](https://r-xla.github.io/anvl/dev/reference/nv_seq.md) /
  [`nv_seq_like()`](https://r-xla.github.io/anvl/dev/reference/nv_seq.md)
  was removed.
- `default_backend()` is now called
  [`active_backend()`](https://r-xla.github.io/anvl/dev/reference/active_backend.md).
- There is now exactly one backend used at a time and it is configured
  via the `anvl.backend` option. With this change the `device_arg`
  parameter was removed from
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) as it is
  no longer needed.
- A `Shape` is now represented as an integer vector.
- The operators `&`, `|`, `!`, as well as the generics
  [`sum()`](https://rdrr.io/r/base/sum.html) and
  [`all()`](https://rdrr.io/r/base/all.html) now require a boolean input
  array, improving consistency with base R.
- The method for `round` was removed, as `digits` is currently not
  supported.
- [`nv_quantile()`](https://r-xla.github.io/anvl/dev/reference/nv_quantile.md)
  and
  [`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md)
  now reduce over `axes` (plural) instead of a single `axis`, defaulting
  to every axis like
  [`nv_mean()`](https://r-xla.github.io/anvl/dev/reference/nv_mean.md)
  and base R’s [`quantile()`](https://rdrr.io/r/stats/quantile.html) /
  [`median()`](https://rdrr.io/r/stats/median.html), and gained a `drop`
  argument. Write `nv_median(x, axes = -1L)` for the previous default.
- [`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md)
  and
  [`nv_order()`](https://r-xla.github.io/anvl/dev/reference/nv_order.md)
  now flatten a multi-axis array when `axis = NULL`, instead of working
  along the last axis, so [`sort()`](https://rdrr.io/r/base/sort.html)
  on an anvl array agrees with base R. Write `axis = -1L` for the
  previous default.
- [`prim_sort()`](https://r-xla.github.io/anvl/dev/reference/prim_sort.md)
  no longer defaults `axis` to `1L`; pass it explicitly, as with every
  other primitive.
- [`nv_which_max()`](https://r-xla.github.io/anvl/dev/reference/nv_which_max.md)
  and
  [`nv_which_min()`](https://r-xla.github.io/anvl/dev/reference/nv_which_min.md)
  now reduce over `axes` (plural) instead of a single `axis`, defaulting
  to every axis so that they pair with
  [`nv_max()`](https://r-xla.github.io/anvl/dev/reference/nv_max.md) /
  [`nv_min()`](https://r-xla.github.io/anvl/dev/reference/nv_min.md).
  Reducing several axes indexes their column-major flattening. Write
  `nv_which_max(x, axes = -1L)` for the previous default.
- The `tensor_to_gval` argument of
  [`GraphDescriptor()`](https://r-xla.github.io/anvl/dev/reference/GraphDescriptor.md)
  is now called `array_to_gval`.
- `vt2at()` was removed.

### Features

- [`nv_subset_assign()`](https://r-xla.github.io/anvl/dev/reference/nv_subset_assign.md)
  and `[<-` gain `inplace`, which writes into the memory of `x` instead
  of copying it, e.g. `x[1, inplace = TRUE] <- 0`; `x` is donated.
- New
  [`local_default_device()`](https://r-xla.github.io/anvl/dev/reference/local_default_device.md)
  and
  [`with_default_device()`](https://r-xla.github.io/anvl/dev/reference/local_default_device.md)
  set the `anvl.default_device` option, which names the device a call
  that names none allocates on in place of the first CPU device.
- The environment variables `ANVL_DEFAULT_DEVICE` and
  `ANVL_DEFAULT_DTYPES`, read when anvl is loaded, are used when the
  `anvl.default_device` and `anvl.default_dtypes` options are not set.
- [`nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/nv_rng_state.md)
  accepts a seed of any signed or unsigned integer data type, bringing
  it to `i32`, where it took an `i32` only. The state stays `ui64[2]`
  whatever the seed and the default integer data type are.
- [`nv_flatten()`](https://r-xla.github.io/anvl/dev/reference/nv_flatten.md)
  accepts a scalar, returning a length-1 array, instead of erroring.
- [`as_anvl_array()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md)
  gained a `.promote` argument, naming the data type the input is
  brought to, as
  [`as_anvl_arrays()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md)
  already had.
- [`nv_is_nan()`](https://r-xla.github.io/anvl/dev/reference/nv_is_nan.md),
  [`nv_is_finite()`](https://r-xla.github.io/anvl/dev/reference/nv_is_finite.md)
  and
  [`nv_is_infinite()`](https://r-xla.github.io/anvl/dev/reference/nv_is_infinite.md)
  accept any data type and answer a constant (all `FALSE` / all `TRUE` /
  all `FALSE`) for one that holds no NaN or infinity, instead of
  comparing – or, for
  [`nv_is_finite()`](https://r-xla.github.io/anvl/dev/reference/nv_is_finite.md)
  and
  [`nv_is_infinite()`](https://r-xla.github.io/anvl/dev/reference/nv_is_infinite.md),
  erroring.
- The linear algebra functions
  ([`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md),
  [`nv_triangular_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_triangular_solve.md),
  [`nv_chol()`](https://r-xla.github.io/anvl/dev/reference/nv_chol.md),
  [`nv_inv()`](https://r-xla.github.io/anvl/dev/reference/nv_inv.md),
  [`nv_det()`](https://r-xla.github.io/anvl/dev/reference/nv_det.md),
  [`nv_determinant()`](https://r-xla.github.io/anvl/dev/reference/nv_determinant.md),
  [`nv_lu()`](https://r-xla.github.io/anvl/dev/reference/nv_lu.md),
  [`nv_qr()`](https://r-xla.github.io/anvl/dev/reference/nv_qr.md),
  [`nv_svd()`](https://r-xla.github.io/anvl/dev/reference/nv_svd.md),
  [`nv_eigh()`](https://r-xla.github.io/anvl/dev/reference/nv_eigh.md))
  accept integer input, computing at the default float data type where
  the input is not a float already, instead of erroring. The `prim_*`
  ones still take a float only.
- [`nv_sign()`](https://r-xla.github.io/anvl/dev/reference/nv_sign.md)
  accepts an unsigned integer array, returning `0` or `1` like base R’s
  [`sign()`](https://rdrr.io/r/base/sign.html) on a non-negative number;
  [`prim_sign()`](https://r-xla.github.io/anvl/dev/reference/prim_sign.md)
  still takes a signed input only.
- Error messages of primitives should now be greatly improved and
  mention the right argument names. This was achieved by porting the
  stablehlo inference functions to anvl’s terminology. A message about a
  parameter also reports the value it was given, e.g. \``x` Got c(1,
  2)\`.
- [`aperm()`](https://rdrr.io/r/base/aperm.html) and
  [`quantile()`](https://rdrr.io/r/stats/quantile.html) now work on an
  `AnvlArray` / `AnvlBox`, forwarding to
  [`nv_aperm()`](https://r-xla.github.io/anvl/dev/reference/nv_aperm.md)
  and
  [`nv_quantile()`](https://r-xla.github.io/anvl/dev/reference/nv_quantile.md).
- New
  [`nv_drop()`](https://r-xla.github.io/anvl/dev/reference/nv_squeeze.md),
  another spelling of
  [`nv_squeeze()`](https://r-xla.github.io/anvl/dev/reference/nv_squeeze.md);
  with the default `axes = NULL` it drops every size-1 axis like
  [`base::drop()`](https://rdrr.io/r/base/drop.html).
- [`nv_rev()`](https://r-xla.github.io/anvl/dev/reference/nv_rev.md)
  gained an `axes = NULL` default that reverses every axis, matching
  [`rev()`](https://rdrr.io/r/base/rev.html) and `numpy.flip()`, and
  returns `x` unchanged for an empty `axes` instead of erroring.
- [`nv_seq()`](https://r-xla.github.io/anvl/dev/reference/nv_seq.md) /
  [`nv_seq_like()`](https://r-xla.github.io/anvl/dev/reference/nv_seq.md)
  gained a `by` argument and now count down when `from > to`, like
  [`seq()`](https://rdrr.io/r/base/seq.html).
- New
  [`jit_cache_size()`](https://r-xla.github.io/anvl/dev/reference/jit_cache_size.md)
  reports how many compiled programs a jitted function currently holds
  for a backend.
- The random number generators
  ([`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md),
  [`nv_rnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md),
  [`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md),
  [`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md),
  [`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md))
  and
  [`prim_rng_bit_generator()`](https://r-xla.github.io/anvl/dev/reference/prim_rng_bit_generator.md)
  return a named list with elements `state` and `values` instead of an
  unnamed pair, and
  [`prim_top_k()`](https://r-xla.github.io/anvl/dev/reference/prim_top_k.md),
  [`prim_cummax()`](https://r-xla.github.io/anvl/dev/reference/prim_cummax.md)
  and
  [`prim_cummin()`](https://r-xla.github.io/anvl/dev/reference/prim_cummin.md)
  name theirs `values` and `indices`.
- [`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  accepts a [`raw()`](https://rdrr.io/r/base/raw.html) vector holding
  the native byte payload of `prod(shape)` elements of `dtype` (both
  then required); `byrow` selects row-major element order for the
  payload. Only supported on the `"pjrt"` backend; the inverse direction
  is the existing
  [`as_raw()`](https://r-xla.github.io/anvl/dev/reference/as_raw.md).
- New
  [`nv_scan()`](https://r-xla.github.io/anvl/dev/reference/nv_scan.md):
  a fixed-length loop in the style of JAX’s `lax.scan` that threads a
  carry through a body function and stacks each step’s outputs along a
  new leading axis. Supports nested carries, multiple `xs` and `out`
  leaves, reverse scans, `xs = NULL` counted loops, carry-only loops and
  zero-length scans. Backed by the new
  [`prim_scan()`](https://r-xla.github.io/anvl/dev/reference/prim_scan.md)
  primitive, which lowers to a `while` loop on the pjrt backend.
- The reductions ([`sum()`](https://rdrr.io/r/base/sum.html),
  [`prod()`](https://rdrr.io/r/base/prod.html),
  [`max()`](https://rdrr.io/r/base/Extremes.html),
  [`min()`](https://rdrr.io/r/base/Extremes.html),
  [`range()`](https://rdrr.io/r/base/range.html),
  [`any()`](https://rdrr.io/r/base/any.html),
  [`all()`](https://rdrr.io/r/base/all.html)) now work with multiple
  data inputs.
- The default data types for floating point numbers and integers can now
  be configured via the `anvl.default_dtypes` field. You can configure
  this for a specific scope via
  [`local_default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/local_default_dtypes.md)
  and
  [`with_default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/local_default_dtypes.md).
  To convert a function to one running at a specified precision, use
  [`with_dtypes()`](https://r-xla.github.io/anvl/dev/reference/with_dtypes.md).
- New
  [`nv_linspace()`](https://r-xla.github.io/anvl/dev/reference/nv_linspace.md)
  and
  [`nv_linspace_like()`](https://r-xla.github.io/anvl/dev/reference/nv_linspace.md),
  replacing
  [`nv_seq()`](https://r-xla.github.io/anvl/dev/reference/nv_seq.md)
  with a provided `steps` argument.
- [`as.vector()`](https://rdrr.io/r/base/vector.html) now returns a
  [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html)
  for integer data types that do not fit into R’s 32 bit integers.
- New
  [`nv_floor_div()`](https://r-xla.github.io/anvl/dev/reference/nv_floor_div.md)
  for flooring (integer) division, and the `%/%` operator now works on
  arrays.
- Added support for more generics:
  - Reversing an array via `rev`.
  - Concatenating vectors via [`c()`](https://rdrr.io/r/base/c.html).
  - Floor division via `nv_floor_div`/`%/%`.
  - Trigonometric functions `sinpi`, `cospi` and `tanpi` and their
    corresponding `nv_*` functions.
  - The `gamma` generic.
- `log(x, base)` now accepts its second argument like in base R.
- New
  [`nv_range()`](https://r-xla.github.io/anvl/dev/reference/nv_range.md)
  returns the minimum and the maximum of an array, stacked along a new
  first axis, and is what the
  [`range()`](https://rdrr.io/r/base/range.html) uses.
- The `nv_*` functions that compute in floating point
  ([`nv_sqrt()`](https://r-xla.github.io/anvl/dev/reference/nv_sqrt.md),
  [`nv_log()`](https://r-xla.github.io/anvl/dev/reference/nv_log.md),
  [`nv_atan2()`](https://r-xla.github.io/anvl/dev/reference/nv_atan2.md),
  …) now compute an integer array at the default float data type.
- [`nv_floor()`](https://r-xla.github.io/anvl/dev/reference/nv_floor.md),
  [`nv_ceiling()`](https://r-xla.github.io/anvl/dev/reference/nv_ceiling.md),
  [`nv_trunc()`](https://r-xla.github.io/anvl/dev/reference/nv_trunc.md)
  and
  [`nv_round()`](https://r-xla.github.io/anvl/dev/reference/nv_round.md)
  return an integer array unchanged, like base R does.
- Improved documentation of API functions and primitives.
- Printed graphs read as `[captures] (inputs) { ... return ... }`, show
  sub-graphs in full, and wrap long lines to the console width;
  [`format()`](https://rdrr.io/r/base/format.html) takes `width` and
  `digits` arguments.
- New functions for the uniform distribution:
  [`nv_dunif()`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md),
  [`nv_punif()`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md),
  and
  [`nv_qunif()`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md),
  documented together with
  [`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md)
  on
  [`?nv_uniform`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md).
- [`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md)’s
  `min` and `max` now accept arrayish inputs, scalar or of the sample’s
  shape. Like base R’s
  [`runif()`](https://rdrr.io/r/stats/Uniform.html), an invalid interval
  (`max < min`, or a bound that is not finite) now gives `NaN` instead
  of an error. Note that the RNG state now advances even on samples
  where `min == max`.

### Performance

- [`nv_quantile()`](https://r-xla.github.io/anvl/dev/reference/nv_quantile.md)
  and
  [`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md)
  select the needed order statistics with `top_k` instead of a full sort
  when every requested quantile lies in the same half of the axis.
  Results are unchanged: the interpolation index is computed at `f64`,
  so it agrees with the window the host sizes.
- [`prim_top_k()`](https://r-xla.github.io/anvl/dev/reference/prim_top_k.md)
  gained `indices`; without them the CUDA lowering uses an unstable sort
  of the values and a slice instead of the CHLO op, which costs no more
  than a full sort there. `nv_top_k(indices = FALSE)` and the quantile
  fast path use it.

### Bug fixes

- A bare R integer start index of
  [`prim_dynamic_slice()`](https://r-xla.github.io/anvl/dev/reference/prim_dynamic_slice.md)
  /
  [`prim_dynamic_update_slice()`](https://r-xla.github.io/anvl/dev/reference/prim_dynamic_update_slice.md)
  takes the data type of the other start indices, so
  `prim_dynamic_slice(x, nv_scalar(1L, "i64"), 1L, ...)` no longer
  fails.
- [`nv_rnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md)
  with a scalar `shape` and a non-scalar `mean` or `sd` returned one
  draw shifted/scaled to the shape of `mean`/`sd`; it is now an error,
  as any shape other than a scalar or `shape` already was.
- Whatever a `prim_*()` refuses now reports that primitive as the call,
  rather than the helper that checked the argument or the anonymous
  function [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)
  wraps.
- [`prim_convolution()`](https://r-xla.github.io/anvl/dev/reference/prim_convolution.md)
  (and so
  [`nv_conv1d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv1d.md)
  /
  [`nv_conv2d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv2d.md)
  /
  [`nv_conv3d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv3d.md))
  refuses a negative `padding` that takes away more than a spatial axis
  holds, which made XLA’s own inference abort the R process, and a
  zero-sized kernel spatial axis. Negative padding that only empties an
  axis stays legal.
- [`prim_convolution()`](https://r-xla.github.io/anvl/dev/reference/prim_convolution.md)
  and
  [`prim_static_slice()`](https://r-xla.github.io/anvl/dev/reference/prim_static_slice.md)
  no longer overflow on a padding, dilation or index that is large but
  inside the integer range, which surfaced as R’s
  `missing value where TRUE/FALSE needed`.
- [`prim_if()`](https://r-xla.github.io/anvl/dev/reference/prim_if.md)
  refuses a `true` or `false` that is not a function, as
  [`prim_while()`](https://r-xla.github.io/anvl/dev/reference/prim_while.md)
  already did for `cond` and `body`.
- The `_like` constructors
  ([`nv_scalar_like()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md),
  [`nv_array_like()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md),
  [`nv_fill_like()`](https://r-xla.github.io/anvl/dev/reference/nv_fill.md),
  [`nv_iota_like()`](https://r-xla.github.io/anvl/dev/reference/nv_iota.md),
  [`nv_empty_like()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))
  no longer allocate on the first CPU device when `like` is an array
  built inside a trace. The stray device made
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) abort
  with “found more than one device” wherever the operands were
  elsewhere, which took out every
  [`nv_qnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md)
  call on CUDA.
- [`nv_unserialize()`](https://r-xla.github.io/anvl/dev/reference/nv_unserialize.md)
  / [`nv_read()`](https://r-xla.github.io/anvl/dev/reference/nv_read.md)
  place the loaded arrays on
  \[[`default_device()`](https://r-xla.github.io/anvl/dev/reference/default_device.md)\],
  where they always used pjrt’s first device.
- [`nv_chol()`](https://r-xla.github.io/anvl/dev/reference/nv_chol.md) /
  [`prim_chol()`](https://r-xla.github.io/anvl/dev/reference/prim_chol.md)
  and
  [`prim_triangular_solve()`](https://r-xla.github.io/anvl/dev/reference/prim_triangular_solve.md)
  accept batched inputs again: axes before the last two are batch axes.
- A function returned by
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) no longer
  evaluates its arguments a second time. It used to rebuild the call
  with [`match.call()`](https://rdrr.io/r/base/match.call.html) and
  evaluate the argument expressions again in the caller’s frame, which
  computed them twice whenever something had evaluated them already –
  most visibly under S3 dispatch, which evaluates the first argument to
  choose a method.
- [`nv_conv1d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv1d.md)
  /
  [`nv_conv2d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv2d.md)
  /
  [`nv_conv3d()`](https://r-xla.github.io/anvl/dev/reference/nv_conv3d.md)
  now promote `x` and `weight` to a common data type.
- The floating-point `nv_*` functions refuse a boolean,
  [`nv_matmul()`](https://r-xla.github.io/anvl/dev/reference/nv_matmul.md),
  [`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md)
  and
  [`nv_triangular_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_triangular_solve.md)
  included – a boolean used to meet a numeric operand at that operand’s
  data type and pass their data type check.
- [`nv_crossprod()`](https://r-xla.github.io/anvl/dev/reference/nv_crossprod.md)
  and
  [`nv_tcrossprod()`](https://r-xla.github.io/anvl/dev/reference/nv_tcrossprod.md)
  transpose only the last two axes, so they work on batched arrays.
- [`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md)
  checks `k` before coercing it, so a fractional or logical `k` is
  refused rather than silently truncated.
- A range that counts down (`x[3:1]`) now selects in reverse instead of
  failing.
- Coercing a traced array to R inside
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) –
  [`as_array()`](https://r-xla.github.io/anvl/dev/reference/as_array.md),
  [`as.vector()`](https://rdrr.io/r/base/vector.html),
  [`as.numeric()`](https://rdrr.io/r/base/numeric.html),
  [`as.character()`](https://rdrr.io/r/base/character.html) and friends
  – now aborts with an explanation instead of falling through to the
  base R generic. Some of those used to fail with a message about lists
  or dimensions, and
  [`as.vector()`](https://rdrr.io/r/base/vector.html),
  [`as.list()`](https://rdrr.io/r/base/list.html) and
  [`as.character()`](https://rdrr.io/r/base/character.html) silently
  returned the traced box itself.
- [`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md)
  and
  [`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md)
  reject a boolean `dtype`, which cannot hold a count or an index.
- Subsetting with `drop` (e.g. `x[1, , drop = FALSE]`) now gives a
  better error message, as `drop` is not supported.
- [`nv_quantile()`](https://r-xla.github.io/anvl/dev/reference/nv_quantile.md)
  and
  [`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md)
  now compute at the default float data type for a non-float input.
- [`as.vector()`](https://rdrr.io/r/base/vector.html) now works
  correctly for `AnvlArray`s that are converted to
  [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html).
  It used to drop that class along with the shape, exposing the raw
  64-bit pattern as a double.
- The gradient of a conversion into a non-float data type is now zero
  instead of one.
  [`prim_convert()`](https://r-xla.github.io/anvl/dev/reference/prim_convert.md)
  /
  [`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md)
  passed the cotangent through whatever the data types were, so
  `nv_convert(nv_convert(x, "i32"), "f64")` reported a gradient of 1
  where
  [`nv_floor()`](https://r-xla.github.io/anvl/dev/reference/nv_floor.md)
  – the same function on the reals – correctly reported 0. Conversions
  between floats still pass the gradient through.
- [`prim_scatter()`](https://r-xla.github.io/anvl/dev/reference/prim_scatter.md)
  now checks that `update_computation` returns one value of `x`’s data
  type, as
  [`prim_reduce()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce.md)
  already did for its `reducer`. A combiner returning something else
  made type inference declare a data type the call could not produce,
  and failed in the backend.
- [`prim_reduce()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce.md)’s
  `reducer` no longer has to name its arguments `lhs` and `rhs`. They
  were passed by name, so `function(a, b)` failed with
  `unused arguments (lhs = ..., rhs = ...)`; they are now matched
  positionally, as
  [`prim_scatter()`](https://r-xla.github.io/anvl/dev/reference/prim_scatter.md)
  already matched its `update_computation`.
- Improved the numerics for
  [`nv_mod()`](https://r-xla.github.io/anvl/dev/reference/nv_mod.md).
- The variadic array functions
  ([`nv_concatenate()`](https://r-xla.github.io/anvl/dev/reference/nv_concatenate.md),
  [`nv_rbind()`](https://r-xla.github.io/anvl/dev/reference/nv_bind.md),
  [`nv_cbind()`](https://r-xla.github.io/anvl/dev/reference/nv_bind.md),
  [`nv_broadcast_scalars()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_scalars.md),
  [`nv_broadcast_arrays()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_arrays.md),
  [`nv_promote_to_common()`](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md))
  say so when given no array, instead of warning or failing internally.
- [`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  of a zero-length vector asks for a `shape` instead of failing inside
  the backend; which axis is empty cannot be inferred from the data.
- [`nv_save()`](https://r-xla.github.io/anvl/dev/reference/nv_save.md)
  and
  [`nv_serialize()`](https://r-xla.github.io/anvl/dev/reference/nv_serialize.md)
  given a single array say so, instead of failing inside
  [`nv_subset()`](https://r-xla.github.io/anvl/dev/reference/nv_subset.md).
  [`nv_serialize()`](https://r-xla.github.io/anvl/dev/reference/nv_serialize.md)
  to a connection returns invisibly.
- The `_like()` functions name `like` when it is an R value with no data
  type.
- [`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md)
  and
  [`nv_triangular_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_triangular_solve.md)
  promote their operands, as
  [`nv_matmul()`](https://r-xla.github.io/anvl/dev/reference/nv_matmul.md)
  does, instead of refusing two arrays that disagree.
- On the `"quickr"` backend a call whose outputs are all empty emits the
  empty arrays directly, instead of an elementwise operation quickr
  rejects.
- [`nv_any()`](https://r-xla.github.io/anvl/dev/reference/nv_any.md),
  [`nv_all()`](https://r-xla.github.io/anvl/dev/reference/nv_all.md) and
  [`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md)
  are jitted, and
  [`nv_psigamma()`](https://r-xla.github.io/anvl/dev/reference/nv_psigamma.md)’s
  `deriv` is no longer static, so it accepts an array as
  [`prim_psigamma()`](https://r-xla.github.io/anvl/dev/reference/prim_psigamma.md)
  does.
- [`nv_qnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md)
  is accurate to its operand’s data type rather than to the default
  float; its coefficients used to be materialized at the default.
- [`nv_dnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md),
  [`nv_pnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md)
  and
  [`nv_qnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md)
  name their own operand when it is not a float.
- [`prim_fill()`](https://r-xla.github.io/anvl/dev/reference/prim_fill.md)
  / [`nv_fill()`](https://r-xla.github.io/anvl/dev/reference/nv_fill.md)
  check that `value` is something `dtype` can hold: a whole number for
  an integer data type, a non-negative one for an unsigned one, a
  logical or `0` / `1` for `bool`.
- Every data type of the float category counts as a float, so `f16` and
  `bf16` pass the checks that used to accept only `f32` and `f64`.
  [`nv_pnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md)
  and
  [`nv_qnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md)
  keep the narrower requirement, as they carry one coefficient set per
  width.
- The gradient of
  [`nv_gamma()`](https://r-xla.github.io/anvl/dev/reference/nv_gamma.md)
  is now correct for positive whole numbers.
- [`prim_any()`](https://r-xla.github.io/anvl/dev/reference/prim_any.md)
  /
  [`prim_all()`](https://r-xla.github.io/anvl/dev/reference/prim_all.md)
  (and
  [`nv_any()`](https://r-xla.github.io/anvl/dev/reference/nv_any.md) /
  [`nv_all()`](https://r-xla.github.io/anvl/dev/reference/nv_all.md))
  now reject a non-boolean input when the call is traced. Type inference
  declared a `bool` output whatever the input was, so an integer operand
  reached the lowering and failed with
  `Data types of inputs and init_values must match`.
- Printed graphs, arrays and error messages now spell a data type the
  way anvl does, so `bool` no longer shows up as its MLIR spelling `i1`.
- Improved the documentation and various error messages.
- [`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md)
  with `min == max` returns the `state` / `values` pair every other
  sampler returns, instead of the filled array on its own.
- [`nv_concatenate()`](https://r-xla.github.io/anvl/dev/reference/nv_concatenate.md)
  broadcasts a scalar against arrays with two or more axes instead of
  failing.
- [`nv_mod()`](https://r-xla.github.io/anvl/dev/reference/nv_mod.md)
  matches base R’s `%%` for an infinite divisor: `-5 %% Inf` is `Inf`,
  not `0`.
- [`local_default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/local_default_dtypes.md)
  /
  [`with_default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/local_default_dtypes.md)
  reject a category other than `float` and `int`.
- [`value_and_gradient()`](https://r-xla.github.io/anvl/dev/reference/value_and_gradient.md)
  rejects an `f` that is not a function, as
  [`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md)
  does.
- [`local_backend()`](https://r-xla.github.io/anvl/dev/reference/local_backend.md)
  /
  [`with_backend()`](https://r-xla.github.io/anvl/dev/reference/with_backend.md)
  reject the internal `"plain"` backend.
- [`trunc()`](https://rdrr.io/r/base/Round.html) on an array rejects
  further arguments with a clear error.

### Tests

- The environment variables that configure only the test suite are now
  spelled with an `ANVL_TEST` prefix: `ANVL_TEST_SKIP_QUICKR`.
  `ANVL_TEST` itself is unchanged.
- The suite can be run with `ANVL_DEFAULT_DEVICE=cpu:1`, which makes
  anything allocating on the first CPU device rather than following the
  trace land on a device of its own instead of agreeing with everything
  else by accident. The `default-device` workflow runs it that way on
  the `full-test` label.
- Moved some of pjrt’s dispatcher tests into anvl.

## anvl 0.4.0

### Breaking changes

- Renamed `dim`/`dims` to `axis`/`axes` throughout the package (an axis
  is an index, a dimension is a size); `ndims()` is now
  [`naxes()`](https://r-xla.github.io/anvl/dev/reference/naxes.md).
- Renamed the primary array argument of `prim_*` / `nv_*` functions from
  `operand` to `x`.
- Renamed the `"xla"` backend to `"pjrt"`.
- `xla()` has been removed; use
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) instead.
- `nv_rdunif()` has been renamed to
  [`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md),
  mirroring R’s [`sample.int()`](https://rdrr.io/r/base/sample.html).
- [`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md)’s
  `lower`/`upper` arguments are now `min`/`max`,
  [`nv_rnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md)’s
  `mu`/`sigma` are now `mean`/`sd`, and
  [`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md)’s
  `n` is now `size`, matching the corresponding R functions.

### Bug fixes

- [`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md)
  (formerly `nv_rdunif()`) was off by one: the first integer was drawn
  twice as often as it should have been, and the last integer was never
  drawn at all.

### Features

- New
  [`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md)
  samples from an arbitrary population.
- New
  [`nv_lower_tri()`](https://r-xla.github.io/anvl/dev/reference/nv_lower_tri.md)
  and
  [`nv_upper_tri()`](https://r-xla.github.io/anvl/dev/reference/nv_upper_tri.md)
  (with
  [`nv_lower_tri_like()`](https://r-xla.github.io/anvl/dev/reference/nv_lower_tri.md)
  /
  [`nv_upper_tri_like()`](https://r-xla.github.io/anvl/dev/reference/nv_upper_tri.md))
  return a boolean triangular mask for a given shape, mirroring base R’s
  [`lower.tri()`](https://rdrr.io/r/base/lower.tri.html) /
  [`upper.tri()`](https://rdrr.io/r/base/lower.tri.html).
- New functions for the normal distribution:
  [`nv_dnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md),
  [`nv_qnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md),
  and
  [`nv_pnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md)
  thanks to Louis Aslett. They are implemented to be accurate far into
  either tail.
- [`nv_rnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md)’s
  `mean` and `sd` now accept arrayish inputs.
- Dimension arguments (`dim`, `dims`, `dimension`, `permutation`) now
  accept negative values that count from the end, so `-1` refers to the
  last dimension.
- Reshaping functions accept a single `-1` in shape indicating a
  dimension to be inferred.
- Added support for 1-3 dimensional convolutions, thanks to Troy
  Hernandez.
- `AnvlArray` constructors and converters have gained a `check` argument
  that opts into scanning for `NA` values, see the “Gotchas” vignette
  for more information.
- [`nv_var()`](https://r-xla.github.io/anvl/dev/reference/nv_var.md) and
  [`nv_sd()`](https://r-xla.github.io/anvl/dev/reference/nv_sd.md) now
  default to `dims = NULL`, which reduces over all dimensions and
  returns a scalar, consistent with the other reductions.
- [`trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md)
  gained an `optimize` argument controlling which graph optimization
  passes run on the traced graph. `TRUE` runs all passes, `FALSE`
  (default) runs none, and a character vector (e.g.
  `c("inline_scalars", "remove_unused_constants")`) selects a subset.
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) always
  traces with all passes enabled.
- Improved the installation vignette

### Performance

- Most `nv_*()` API functions are now JIT-compiled internally (via a new
  `@jit` roxygen roclet), speeding up eager-mode execution.
- Tracing
  ([`trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md))
  performance has been improved.
- StableHLO lowering has been sped up.
- Calling
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)ted
  functions is now significantly faster.

### Bug fixes

- Reductions now reject dimensions that are out of range for the operand
  instead of silently ignoring them.
- `NULL` is now treated as an empty node when flattening and
  unflattening trees.
- `nv_argmax()` / `nv_argmin()` and
  [`nv_cummax()`](https://r-xla.github.io/anvl/dev/reference/nv_cummax.md)
  /
  [`nv_cummin()`](https://r-xla.github.io/anvl/dev/reference/nv_cummin.md)
  now break ties order-independently, so they return the same result on
  GPU as on CPU ([\#368](https://github.com/r-xla/anvl/issues/368)).
  `nv_argmax()` / `nv_argmin()` prefer the smallest index;
  [`nv_cummax()`](https://r-xla.github.io/anvl/dev/reference/nv_cummax.md)
  /
  [`nv_cummin()`](https://r-xla.github.io/anvl/dev/reference/nv_cummin.md)
  prefer the last occurrence.
- [`nv_diag()`](https://r-xla.github.io/anvl/dev/reference/nv_diag.md)
  now errors on non-1-D input instead of silently producing an incorrect
  result.
- [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) now
  rejects static arguments with reference semantics.
- Error messages now speak of arrays instead of tensors.

## anvl 0.3.0

### Breaking Changes

- [`nv_empty()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  /
  [`nv_empty_like()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  return arrays with unspecified contents (no longer zero-initialized).

### New Features

- On CPU, jitted XLA functions now back every non-aliased output with an
  R-owned RAWSXP. anvl appends a phantom donated input per unaliased
  output during lowering, allocates
  [`pjrt::pjrt_empty()`](https://r-xla.github.io/pjrt/reference/pjrt_buffer.html)
  buffers at execute time, and `pjrt` migrates the keepalive onto the
  output XPtr. The output’s host bytes are then managed by R’s GC.
- Renamed user-facing API functions to match base R names: `nv_sine()`
  -\>
  [`nv_sin()`](https://r-xla.github.io/anvl/dev/reference/nv_sin.md),
  `nv_cosine()` -\>
  [`nv_cos()`](https://r-xla.github.io/anvl/dev/reference/nv_cos.md),
  `nv_ceil()` -\>
  [`nv_ceiling()`](https://r-xla.github.io/anvl/dev/reference/nv_ceiling.md),
  `nv_cholesky()` -\>
  [`nv_chol()`](https://r-xla.github.io/anvl/dev/reference/nv_chol.md).
  The corresponding primitives were renamed in step: `prim_sine()` -\>
  [`prim_sin()`](https://r-xla.github.io/anvl/dev/reference/prim_sin.md),
  `prim_cosine()` -\>
  [`prim_cos()`](https://r-xla.github.io/anvl/dev/reference/prim_cos.md),
  `prim_cholesky()` -\>
  [`prim_chol()`](https://r-xla.github.io/anvl/dev/reference/prim_chol.md).
- `nv_reduce_mean()` was renamed to
  [`nv_mean()`](https://r-xla.github.io/anvl/dev/reference/nv_mean.md).
- [`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md)
  no longer requires `a` to be symmetric positive-definite as it uses LU
  instead of Cholesky decomposition. Because of this, it is no longer
  differentiable, as the reverse rule for LU is not implemented yet.
- [`nv_chol()`](https://r-xla.github.io/anvl/dev/reference/nv_chol.md) /
  [`prim_chol()`](https://r-xla.github.io/anvl/dev/reference/prim_chol.md)
  now default to `lower = FALSE` (upper-triangular factor), matching
  base R’s [`chol()`](https://rdrr.io/r/base/chol.html). Previously
  defaulted to `lower = TRUE`.

### New Features

#### Linear algebra

- New matrix-decomposition primitives and corresponding `nv_*()`
  functions: `qr`, `lu`, `svd`, `eigh`. None of them implement a reverse
  rule yet.
- New API functions:
  - [`nv_triangular_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_triangular_solve.md)
    (wraps the already-existing
    [`prim_triangular_solve()`](https://r-xla.github.io/anvl/dev/reference/prim_triangular_solve.md)).
  - [`nv_det()`](https://r-xla.github.io/anvl/dev/reference/nv_det.md)
    and
    [`nv_determinant()`](https://r-xla.github.io/anvl/dev/reference/nv_determinant.md).
    The latter can also be called via the
    [`determinant()`](https://rdrr.io/r/base/det.html) generic.
  - [`nv_inv()`](https://r-xla.github.io/anvl/dev/reference/nv_inv.md),
    which can also be called via `solve(operand)` (missing second
    argument).
- `qr`, `chol`, and `solve` from base R now dispatch to
  [`nv_qr()`](https://r-xla.github.io/anvl/dev/reference/nv_qr.md),
  [`nv_chol()`](https://r-xla.github.io/anvl/dev/reference/nv_chol.md),
  and
  [`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md)
  on `AnvlArray` / `AnvlBox` inputs.

#### Element-wise math

- New unary primitives and corresponding `nv_*()` functions: `acos`,
  `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cosh`, `sinh`, `digamma`,
  `lgamma`, `polygamma`, `erf`, `erf_inv`, `erfc`.
- New API functions
  [`nv_mod()`](https://r-xla.github.io/anvl/dev/reference/nv_mod.md)
  (flooring remainder) and
  [`nv_trunc()`](https://r-xla.github.io/anvl/dev/reference/nv_trunc.md)
  (truncation toward zero).

#### Cumulative reductions

- New primitives and corresponding `nv_*()` functions: `cumsum`,
  `cumprod`, `cummax`, `cummin`.
  [`prim_cumprod()`](https://r-xla.github.io/anvl/dev/reference/prim_cumprod.md)
  does not yet have a reverse rule.

#### Sorting and searching

- New primitives
  [`prim_sort()`](https://r-xla.github.io/anvl/dev/reference/prim_sort.md),
  [`prim_top_k()`](https://r-xla.github.io/anvl/dev/reference/prim_top_k.md),
  [`prim_reduce()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce.md),
  `prim_argmax()`, `prim_argmin()`.
- New API functions:
  - [`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md)
    / `nv_argsort()` – sort along a dimension, or return the permutation
    that does.
  - [`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md)
    – the `k` largest values along a dimension.
  - [`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md)
    /
    [`nv_quantile()`](https://r-xla.github.io/anvl/dev/reference/nv_quantile.md)
    – median / quantiles along a dimension.
    [`median()`](https://rdrr.io/r/stats/median.html) dispatches to
    [`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md).
  - `nv_argmax()` / `nv_argmin()` – index of the maximum / minimum along
    a dimension (ties broken by smallest index).
  - [`nv_select()`](https://r-xla.github.io/anvl/dev/reference/nv_select.md)
    – select a slice along a dimension by index.

#### Array construction / shape

- [`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  gained a `byrow` argument that fills the array from an R object in
  row-major order, mirroring `matrix(byrow = TRUE)`
  ([\#165](https://github.com/r-xla/anvl/issues/165)).
- New `nv_matrix(data, nrow, ncol, ...)` which works like R’s
  [`matrix()`](https://rdrr.io/r/base/matrix.html).
- New API functions
  [`nv_rbind()`](https://r-xla.github.io/anvl/dev/reference/nv_bind.md)
  and
  [`nv_cbind()`](https://r-xla.github.io/anvl/dev/reference/nv_bind.md)
  and corresponding [`rbind()`](https://rdrr.io/r/base/cbind.html) /
  [`cbind()`](https://rdrr.io/r/base/cbind.html) generics.
- New API function
  [`nv_flatten()`](https://r-xla.github.io/anvl/dev/reference/nv_flatten.md)
  for flattening to 1-D.

#### Misc

- New `AnvlArray` -\> R `vector` converters:
  [`as.numeric()`](https://rdrr.io/r/base/numeric.html),
  [`as.double()`](https://rdrr.io/r/base/double.html),
  [`as.integer()`](https://rdrr.io/r/base/integer.html),
  [`as.logical()`](https://rdrr.io/r/base/logical.html),
  [`as.vector()`](https://rdrr.io/r/base/vector.html).
- New function
  [`await()`](https://r-xla.github.io/anvl/dev/reference/await.md) that
  blocks until the underlying computation has finished.
- New tree utilities
  [`map_tree()`](https://r-xla.github.io/pjrt/reference/map_tree.html)
  and
  [`pmap_tree()`](https://r-xla.github.io/pjrt/reference/pmap_tree.html)
  for applying functions leaf-wise over (possibly nested) lists.
- Added support for `range` generic.
- Improved NaN handling across various primitives and API functions.

### Other

- `nv_reduce_sum()`, `nv_reduce_prod()`, `nv_reduce_max()`,
  `nv_reduce_min()`, `nv_reduce_any()`, `nv_reduce_all()` and
  [`nv_mean()`](https://r-xla.github.io/anvl/dev/reference/nv_mean.md)
  now default `dims = NULL`, which reduces over all dimensions and
  returns a scalar. Previously, `dims` was required.

### Bug Fixes

- The overloaded `%%` operator now calls the new
  [`nv_mod()`](https://r-xla.github.io/anvl/dev/reference/nv_mod.md) to
  be consistent with base R.
- The reverse rule for `prim_reduce_prod()` no longer produces `NaN` /
  `Inf` gradients when the input contains zeros.
- The CI now actually runs the torch-comparison tests.
- [`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md)
  not properly respects the `lower` argument.

## anvl 0.2.0

### Breaking Changes

- The package was renamed from `anvil` to `anvl` to avoid a conflict
  with the Bioconductor package `AnVIL`.
- `AnvilTensor`/`nv_tensor` were renamed to `AnvlArray` and `nv_array`
  to be more in line with R’s
  [`array()`](https://rdrr.io/r/base/array.html). Also, `nv_aten()` was
  renamed to
  [`nv_aval()`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md).
- Subsetting with [`list()`](https://rdrr.io/r/base/list.html)
  (e.g. `x[list(1, 3)]`) is no longer supported. Use
  [`array()`](https://rdrr.io/r/base/array.html) to wrap the indices
  instead, e.g. `x[array(c(1L, 3L))]`. This mirrors the input convention
  used everywhere else in the package.
- Removed *debug mode*.
- Remove NSE support for `nvl_if`. It now requires passing 0-argument
  closures as `true` and `false` arguments.
- Primitives renamed from `nvl_*` to `prim_*`. The underlying primitive
  object containing the rules and metadata is now part of the
  `JitPrimitive` function via the `primitive` attribute.

### New Features

- Better composability:
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)ted
  functions can now be used in other
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)-calls.
  This is the mechanism underlying the new *eager mode*.
- *Eager mode* was added: This means, you can now do
  `nv_add(1, nv_array(1:2))` and it will actually perform the
  computation and not only do type inference.
- An experimental [{quickr}](https://github.com/t-kalinowski/quickr)
  backend was added It only runs on CPU for now and supports a subset of
  available operations. You can enable it via the `backend` argument in
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) and
  [`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  or via the `anvl.default_backend` option.
- New primitives:
  - `nvl_cholesky()` to compute the Cholesky decomposition of a matrix.
  - `nvl_triangular_solve()` to solve a system of linear equations with
    a triangular matrix.
- New API functions (+ corresponding R generic implementations):
  - [`nv_diag()`](https://r-xla.github.io/anvl/dev/reference/nv_diag.md)
    to create a diagonal matrix from a 1-D tensor.
  - [`nv_eye()`](https://r-xla.github.io/anvl/dev/reference/nv_eye.md)
    to create an identity matrix.
  - [`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md)
    to solve a system of linear equations.
  - `nv_cholesky()` to compute the Cholesky decomposition of a matrix.
  - [`nv_device()`](https://r-xla.github.io/anvl/dev/reference/nv_device.md)
    constructs a backend-specific device object
    (e.g. `nv_device("cpu")`) that can be passed as `device` to array
    constructors like
    [`nv_fill()`](https://r-xla.github.io/anvl/dev/reference/nv_fill.md)
    or
    [`nv_iota()`](https://r-xla.github.io/anvl/dev/reference/nv_iota.md).
  - [`nv_crossprod()`](https://r-xla.github.io/anvl/dev/reference/nv_crossprod.md)
    and
    [`nv_tcrossprod()`](https://r-xla.github.io/anvl/dev/reference/nv_tcrossprod.md)
    for matrix cross-products.
  - [`nv_outer()`](https://r-xla.github.io/anvl/dev/reference/nv_outer.md)
    for the outer product.
  - [`nv_extract_diag()`](https://r-xla.github.io/anvl/dev/reference/nv_extract_diag.md)
    to extract the diagonal of a matrix.
  - [`nv_trace()`](https://r-xla.github.io/anvl/dev/reference/nv_trace.md)
    to compute the trace of a matrix.
  - [`nv_tril()`](https://r-xla.github.io/anvl/dev/reference/nv_tril.md)
    and
    [`nv_triu()`](https://r-xla.github.io/anvl/dev/reference/nv_triu.md)
    to extract lower/upper triangular parts.
  - [`nv_squeeze()`](https://r-xla.github.io/anvl/dev/reference/nv_squeeze.md)
    and
    [`nv_unsqueeze()`](https://r-xla.github.io/anvl/dev/reference/nv_unsqueeze.md)
    to drop or add length-1 dimensions.
  - [`nv_log2()`](https://r-xla.github.io/anvl/dev/reference/nv_log2.md)
    and
    [`nv_log10()`](https://r-xla.github.io/anvl/dev/reference/nv_log10.md).
  - [`nv_is_infinite()`](https://r-xla.github.io/anvl/dev/reference/nv_is_infinite.md)
    and
    [`nv_is_nan()`](https://r-xla.github.io/anvl/dev/reference/nv_is_nan.md).
  - [`nv_sd()`](https://r-xla.github.io/anvl/dev/reference/nv_sd.md) and
    [`nv_var()`](https://r-xla.github.io/anvl/dev/reference/nv_var.md)
    for standard deviation and variance.
- [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) now
  accepts integer positions for the `static` argument.
- New S3 methods [`dim()`](https://rdrr.io/r/base/dim.html),
  [`nrow()`](https://rdrr.io/r/base/nrow.html),
  [`ncol()`](https://rdrr.io/r/base/nrow.html), and
  [`length()`](https://rdrr.io/r/base/length.html) for anvl arrays.
- Printing tensors via
  [`nv_print()`](https://r-xla.github.io/anvl/dev/reference/nv_print.md)
  now also works on GPUs.
- R vectors of length 1 and arrays are now auto-converted when being
  passed to `jit`ted functions.
- Improved device handling in
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)

### Performance

- Many operations are now done asynchronously, which improves
  performance, especially on GPUs.

### Bug Fixes

- +-Inf/NaN are correctly created for `f64` when inlined into the XLA
  exectuable ([\#182](https://github.com/r-xla/anvl/issues/182)). This
  caused wrong results with e.g. `nv_reduce_max()` when working with
  `f64`.
- Corrected argument checks in
  [`nv_iota()`](https://r-xla.github.io/anvl/dev/reference/nv_iota.md).
- Fix check that `wrt` arguments in
  [`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md)
  must be floats.
- [`nv_subset()`](https://r-xla.github.io/anvl/dev/reference/nv_subset.md)
  and
  [`nv_subset_assign()`](https://r-xla.github.io/anvl/dev/reference/nv_subset_assign.md)
  now error on trailing-comma subscripts
  ([\#273](https://github.com/r-xla/anvl/issues/273)).

### Documentation

- New vignette on implementing Gaussian Processes.
- New vignette on implementing Metropolis-Hastings sampling.

### Platform support and installation

- An installation guide was added.
- Linux on ARM is now supported (CPU only).
- To use the CUDA backend, it is now possible to install the `cuda12.8`
  package (see installation guide), which only requires a compatible
  CUDA driver.

## anvl 0.1.0

Initial release
