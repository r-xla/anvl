# anvl (development version)

## Breaking changes

* The element-wise `nv_max()` / `nv_min()` and `prim_max()` / `prim_min()` are
  now `nv_pmax()` / `nv_pmin()` and `prim_pmax()` / `prim_pmin()`, following
  `base::pmax()` / `base::pmin()`.
* The reductions lost their `reduce_` prefix: write `nv_sum()`, `nv_prod()`,
  `nv_max()`, `nv_min()`, `nv_any()`, `nv_all()` and the matching `prim_*()`
  instead of `nv_reduce_sum()` and friends. `prim_reduce()` keeps its name.
* `nv_argsort()` is now `nv_order()`, `nv_reverse()` / `prim_reverse()` are
  `nv_rev()` / `prim_rev()`, `nv_argmax()` / `nv_argmin()` and their primitives
  are `nv_which_max()` / `nv_which_min()` and `prim_which_max()` /
  `prim_which_min()`, and `prim_ceil()` is `prim_ceiling()`.
* `nv_polygamma()` / `prim_polygamma()` are now `nv_psigamma()` /
  `prim_psigamma()` and take `(x, deriv)` like `base::psigamma()` instead of
  `(n, x)`; `deriv` defaults to `0`.
* `nv_logistic()` / `prim_logistic()` are now `nv_plogis()` /
  `prim_plogis()`.
* The general transpose is now `nv_aperm(x, perm)`, matching `base::aperm()`;
  `nv_transpose()` stays as another spelling of it. It and `prim_transpose()`
  call their second argument `perm` instead of `permutation`.
* `default_device()` no longer follows `PJRT_PLATFORM`; set `ANVL_DEFAULT_DEVICE`
  or the `anvl.default_device` option instead.
* `prim_reshape()` and `nv_reshape()`, and with them `nv_flatten()` and every
  `axis = NULL` flattening default, are now column-major like base R.
* `prim_bitcast_convert()` puts the axis holding an element's pieces first
  rather than last when the two data types differ in width, so the pieces of
  one element are adjacent in the order `nv_flatten()` reads and a narrowing
  conversion lays the bytes out the way `as_raw()` writes them.
* `nv_broadcast_to()` and `nv_broadcast_arrays()` align axes from the first
  instead of the last: a shorter shape gets size-1 axes appended, so a
  length-`nrow` vector broadcasts against a matrix where a length-`ncol` one
  no longer does. Write `prim_broadcast_in_axes()` with an explicit axis
  mapping for the previous right-aligned behavior.
* The `@jit` roxygen tag was removed; wrap functions in `jit()` at the
  definition instead.
* `nv_top_k()`, `nv_cummax()` and `nv_cummin()` take `indices` instead of
  `with_indices`, spelling it the way `prim_top_k()` does.
* `nv_clamp()` and `prim_clamp()` take `(x, min, max)` instead of
  `(min_val, x, max_val)`.
* `nv_seq()` / `nv_seq_like()` call their bounds `from` and `to`, like
  `base::seq()`.
* `nv_ifelse()` and `prim_ifelse()` take `(test, yes, no)`, like
  `base::ifelse()`. `prim_if()` / `nv_if()` keep `(pred, true, false)`: they
  mirror the `if` construct, not `ifelse()`.
* `nv_scan()` takes `(init, xs, body)`, the order `prim_scan()` uses, and both
  call the trip count `steps` instead of `length`.
* `prim_sort()`'s `descending` / `is_stable` are now `decreasing` / `stable`,
  as in `nv_sort()`.
* `prim_top_k()`'s `indices` no longer has a default; pass it explicitly.
* `prim_reduce()` takes `(x, init, axes, reducer, drop)`: `reductor` is now
  `reducer`, and it comes before `drop`.
* `prim_static_slice()` / `nv_static_slice()` call their (inclusive) upper
  bound `end_indices` instead of `limit_indices`.
* `nv_crossprod()`, `nv_tcrossprod()`, `nv_outer()` and `nv_matmul()` call
  their operands `x` and `y`, as base R does. So do `nv_pow()`,
  `nv_remainder()`, `nv_xor()` and their primitives, instead of `lhs` / `rhs`.
* `nv_linspace()` / `nv_linspace_like()` take `(from, to, length_out)` instead
  of `(start, end, steps)`, like `base::seq()`.
* `nv_runif()` and `nv_rnorm()` take `dtype` after the distribution
  parameters, as `nv_rbinom()` and `nv_sample_int()` do.
* `prim_convolution()` calls its input axes `x_batch_axis`, `x_feature_axis`
  and `x_spatial_axes` instead of `input_*`, and `nv_conv1d()` /
  `nv_conv2d()` / `nv_conv3d()` call their second operand `kernel` instead of
  `weight`, as `prim_convolution()` does.
* `nv_pad()` takes `(x, value, low, high, interior)` instead of
  `(x, padding_value, edge_padding_low, edge_padding_high, interior_padding)`;
  `prim_pad()` keeps the StableHLO names.
* `nv_triangular_solve()` takes `left`, `unit_diag` and `transpose` instead of
  `left_side`, `unit_diagonal` and `transpose_a`; `prim_triangular_solve()`
  keeps the StableHLO names.
* `prim_scatter()`'s `update_computation` is now `update_fn`.
* `nv_lower_tri_like()` / `nv_upper_tri_like()` take `shape` before
  `diagonal`, as `nv_lower_tri()` / `nv_upper_tri()` do.
* `nv_quantile()`, `nv_median()` and their `quantile()` / `median()` methods
  call `interpolation` `method`.
* `nv_unsqueeze()` takes `axes`, inserting several axes at once.
* `nv_atan2()` and `prim_atan2()` take `(y, x)`, like `base::atan2()`.
* The array constructors spell their trailing arguments `shape`, `dtype`,
  `device` in that order: `nv_array(data, shape, dtype, device, byrow)`,
  `nv_empty(shape, dtype, device)`, `nv_iota()` / `prim_iota()`
  `(axis, shape, dtype, start, device)`, and likewise `nv_array_like()`,
  `nv_empty_like()` and `nv_iota_like()`.
* `nv_shift_left()`, `nv_shift_right_logical()`, `nv_shift_right_arithmetic()`
  and their primitives take `(x, shift)` instead of `(lhs, rhs)`. The result
  keeps `x`'s data type, which `shift` is brought to, instead of promoting both.
* The RNG functions (`nv_runif()`, `nv_rnorm()`, `nv_rbinom()`,
  `nv_sample()`, `nv_sample_int()`, `prim_rng_bit_generator()`) call their
  state argument `state` instead of `initial_state`, matching the `state`
  element they return.
* `nv_top_k()` takes `axes` instead of `axis`, ranking the elements of several
  axes together, and `axes = NULL` (the default) now ranks over every axis
  where it used to take the last one. Write `axes = -1` for the old default.
* The type system of {anvl} was changed to avoid the problems reported in issue #373.
  Specifically, the ambiguity system was replaced with the `RData` system and a new system of rules for type promotions.
  With it, also the promotion behavior of various primitives and API
  functions was improved.
* An R value is now built directly at every data type of its own category,
  narrow and unsigned integers included, so one the data type cannot hold is an
  error instead of wrapping around: `x_ui8 + (-2L)` and `x_i8 + 300L` are
  refused. Write `nv_convert()` on an array where the wraparound is what you
  want.
* `as_array()` and the `as.double()` / `as.integer()` /
  `bit64::as.integer64()` / `as.logical()` methods take `check = "warn"`,
  `"err"` or `FALSE` instead of a flag, following {pjrt}, and warn by
  default about a value R's type cannot hold. Write `check = "err"` where
  you wrote `check = TRUE`, and `check = FALSE` to materialize silently.
* `nv_array()` and `nv_scalar()` no longer take a `check` argument, following
  {pjrt}: what happens to an `NA` is fixed by the dtype it is built at, and the
  input is always scanned for values the requested dtype cannot hold. Call
  `anyNA()` on the data yourself if you want to hear about a missing value the
  dtype accepts.
* `common_dtype()` now errors for `ui64` and a signed integer instead of
  returning `i64`, which could not hold every `ui64` value. Convert one of them
  with `nv_convert()`.
* `jit_eval()` was removed as it is no longer needed.
* `nv_sum()`, `nv_prod()`, `nv_cumsum()` and `nv_cumprod()` now
  accumulate a boolean array at the default integer data type instead of returning a boolean.
* `as.vector()` on an `AnvlArray` now only accepts `mode = "any"` (the
  default) and errors for any other `mode`.
* The `steps` argument of `nv_seq()` / `nv_seq_like()` was removed.
* `default_backend()` is now called `active_backend()`.
* There is now exactly one backend used at a time and it is configured via the
  `anvl.backend` option.
  With this change the `device_arg` parameter was removed from `jit()` as it is no longer needed.
* A `Shape` is now represented as an integer vector.
* The operators `&`, `|`,  `!`, as well as the generics `sum()` and `all()`
  now require a boolean input array, improving consistency with base R.
* The method for `round` was removed, as `digits` is currently not supported.
* `nv_quantile()` and `nv_median()` now reduce over `axes` (plural) instead of a
  single `axis`, defaulting to every axis like `nv_mean()` and base R's
  `quantile()` / `median()`, and gained a `drop` argument. Write
  `nv_median(x, axes = -1L)` for the previous default.
* `nv_sort()` and `nv_order()` now flatten a multi-axis array when
  `axis = NULL`, instead of working along the last axis, so `sort()` on an
  anvl array agrees with base R. Write `axis = -1L` for the previous default.
* `prim_sort()` no longer defaults `axis` to `1L`; pass it explicitly, as with
  every other primitive.
* `nv_which_max()` and `nv_which_min()` now reduce over `axes` (plural) instead of a
  single `axis`, defaulting to every axis so that they pair with
  `nv_max()` / `nv_min()`. Reducing several axes indexes their
  column-major flattening. Write `nv_which_max(x, axes = -1L)` for the previous
  default.
* The `tensor_to_gval` argument of `GraphDescriptor()` is now called
  `array_to_gval`.

## Features

* New `local_default_device()` and `with_default_device()` set the
  `anvl.default_device` option, which names the device a call that names none
  allocates on in place of the first CPU device.
* The environment variables `ANVL_DEFAULT_DEVICE` and `ANVL_DEFAULT_DTYPES`, read
  when anvl is loaded, are used when the `anvl.default_device` and
  `anvl.default_dtypes` options are not set.
* `nv_rng_state()` accepts a seed of any signed or unsigned integer data type,
  bringing it to `i32`, where it took an `i32` only. The state stays `ui64[2]`
  whatever the seed and the default integer data type are.
* `nv_flatten()` accepts a scalar, returning a length-1 array, instead of
  erroring.
* `as_anvl_array()` gained a `.promote` argument, naming the data type the
  input is brought to, as `as_anvl_arrays()` already had.
* `nv_is_nan()`, `nv_is_finite()` and `nv_is_infinite()` accept any data type
  and answer a constant (all `FALSE` / all `TRUE` / all `FALSE`) for one that
  holds no NaN or infinity, instead of comparing -- or, for `nv_is_finite()`
  and `nv_is_infinite()`, erroring.
* The linear algebra functions (`nv_solve()`, `nv_triangular_solve()`,
  `nv_chol()`, `nv_inv()`, `nv_det()`, `nv_determinant()`, `nv_lu()`,
  `nv_qr()`, `nv_svd()`, `nv_eigh()`) accept integer input, computing at the
  default float data type where the input is not a float already, instead of
  erroring. The `prim_*` ones still take a float only.
* `nv_sign()` accepts an unsigned integer array, returning `0` or `1` like
  base R's `sign()` on a non-negative number; `prim_sign()` still takes a
  signed input only.
* `aperm()` and `quantile()` now work on an `AnvlArray` / `AnvlBox`,
  forwarding to `nv_aperm()` and `nv_quantile()`.
* New `nv_drop()`, another spelling of `nv_squeeze()`; with the default
  `axes = NULL` it drops every size-1 axis like `base::drop()`.
* `nv_rev()` gained an `axes = NULL` default that reverses every axis,
  matching `rev()` and `numpy.flip()`, and returns `x` unchanged for an empty
  `axes` instead of erroring.
* `nv_seq()` / `nv_seq_like()` gained a `by` argument and now count down
  when `from > to`, like `seq()`.
* New `jit_cache_size()` reports how many compiled programs a jitted function
  currently holds for a backend.
* The random number generators (`nv_runif()`, `nv_rnorm()`, `nv_rbinom()`,
  `nv_sample_int()`, `nv_sample()`) and `prim_rng_bit_generator()` return a
  named list with elements `state` and `values` instead of an unnamed pair,
  and `prim_top_k()`, `prim_cummax()` and `prim_cummin()` name theirs
  `values` and `indices`.
* `nv_array()` accepts a `raw()` vector holding the native byte payload of
  `prod(shape)` elements of `dtype` (both then required); `byrow` selects
  row-major element order for the payload. Only supported on the `"pjrt"`
  backend; the inverse direction is the existing `as_raw()`.
* New `nv_scan()`: a fixed-length loop in the style of JAX's `lax.scan` that
  threads a carry through a body function and stacks each step's outputs
  along a new leading axis. Supports nested carries, multiple `xs` and
  `out` leaves, reverse scans, `xs = NULL` counted loops, carry-only loops
  and zero-length scans. Backed by the new `prim_scan()` primitive, which
  lowers to a `while` loop on the pjrt backend.
* The reductions (`sum()`, `prod()`, `max()`, `min()`, `range()`, `any()`,
  `all()`) now work with multiple data inputs.
* The default data types for floating point numbers and integers can now be
  configured via the `anvl.default_dtypes` field.
  You can configure this for a specific scope via `local_default_dtypes()`
  and `with_default_dtypes()`.
  To convert a function to one running at a specified precision, use
  `with_dtypes()`.
* New `nv_linspace()` and `nv_linspace_like()`, replacing `nv_seq()` with
  a provided `steps` argument.
* `as.vector()` now returns a `bit64::integer64` for integer data types that
  do not fit into R's 32 bit integers.
* New `nv_floor_div()` for flooring (integer) division, and the `%/%` operator
  now works on arrays.
* Added support for more generics:
  * Reversing an array via `rev`.
  * Concatenating vectors via `c()`.
  * Floor division via `nv_floor_div`/`%/%`.
  * Trigonometric functions `sinpi`, `cospi` and `tanpi` and their corresponding `nv_*` functions.
  * The `gamma` generic.
* `log(x, base)` now accepts its second argument like in base R.
* New `nv_range()` returns the minimum and the maximum of an array, stacked
  along a new first axis, and is what the `range()` uses.
* The `nv_*` functions that compute in floating point (`nv_sqrt()`,
  `nv_log()`, `nv_atan2()`, ...) now compute an integer array at the default
  float data type.
* `nv_floor()`, `nv_ceiling()`, `nv_trunc()` and `nv_round()` return an
  integer array unchanged, like base R does.
* Improved documentation of API functions and primitives.
* Printed graphs read as `[captures] (inputs) { ... return ... }`, show
  sub-graphs in full, and wrap long lines to the console width; `format()`
  takes `width` and `digits` arguments.
* New functions for the uniform distribution: `nv_dunif()`, `nv_punif()`,
  and `nv_qunif()`.

## Performance

* `nv_quantile()` and `nv_median()` select the needed order statistics with
  `top_k` instead of a full sort when every requested quantile lies in the
  same half of the axis. Results are unchanged: the interpolation index is
  computed at `f64`, so it agrees with the window the host sizes.
* `prim_top_k()` gained `indices`; without them the CUDA lowering uses an
  unstable sort of the values and a slice instead of the CHLO op, which
  costs no more than a full sort there. `nv_top_k(indices = FALSE)`
  and the quantile fast path use it.

## Bug fixes

* The `_like` constructors (`nv_scalar_like()`, `nv_array_like()`,
  `nv_fill_like()`, `nv_iota_like()`, `nv_empty_like()`) no longer allocate on
  the first CPU device when `like` is an array built inside a trace. The stray
  device made `jit()` abort with "found more than one device" wherever the
  operands were elsewhere, which took out every `nv_qnorm()` call on CUDA.
* `nv_unserialize()` / `nv_read()` place the loaded arrays on
  [`default_device()`], where they always used pjrt's first device.
* `nv_chol()` / `prim_chol()` and `prim_triangular_solve()` accept batched
  inputs again: axes before the last two are batch axes.
* A function returned by `jit()` no longer evaluates its arguments a second
  time. It used to rebuild the call with `match.call()` and evaluate the
  argument expressions again in the caller's frame, which computed them twice
  whenever something had evaluated them already -- most visibly under S3
  dispatch, which evaluates the first argument to choose a method.
* `nv_conv1d()` / `nv_conv2d()` / `nv_conv3d()` now promote `x` and `weight`
  to a common data type.
* The floating-point `nv_*` functions refuse a boolean, `nv_matmul()`,
  `nv_solve()` and `nv_triangular_solve()` included -- a boolean used to meet a
  numeric operand at that operand's data type and pass their data type check.
* `nv_crossprod()` and `nv_tcrossprod()` transpose only the last two axes, so
  they work on batched arrays.
* `nv_top_k()` checks `k` before coercing it, so a fractional or logical `k`
  is refused rather than silently truncated.
* A range that counts down (`x[3:1]`) now selects in reverse instead of failing.
* Coercing a traced array to R inside `jit()` -- `as_array()`, `as.vector()`,
  `as.numeric()`, `as.character()` and friends -- now aborts with an
  explanation instead of falling through to the base R generic. Some of those
  used to fail with a message about lists or dimensions, and `as.vector()`,
  `as.list()` and `as.character()` silently returned the traced box itself.
* `nv_rbinom()` and `nv_sample_int()` reject a boolean `dtype`, which cannot
  hold a count or an index.
* Subsetting with `drop` (e.g. `x[1, , drop = FALSE]`) now gives a better
  error message, as `drop` is not supported.
* `nv_quantile()` and `nv_median()` now compute at the default float data
  type for a non-float input.
* `as.vector()` now works correctly for `AnvlArray`s that are converted
  to `bit64::integer64`. It used to drop that class along with the shape,
  exposing the raw 64-bit pattern as a double.
* The gradient of a conversion into a non-float data type is now zero instead
  of one. `prim_convert()` / `nv_convert()` passed the cotangent through
  whatever the data types were, so `nv_convert(nv_convert(x, "i32"), "f64")`
  reported a gradient of 1 where `nv_floor()` -- the same function on the
  reals -- correctly reported 0. Conversions between floats still pass the
  gradient through.
* `prim_scatter()` now checks that `update_computation` returns one value of
  `x`'s data type, as `prim_reduce()` already did for its `reducer`. A
  combiner returning something else made type inference declare a data type
  the call could not produce, and failed in the backend.
* `prim_reduce()`'s `reducer` no longer has to name its arguments `lhs` and
  `rhs`. They were passed by name, so `function(a, b)` failed with
  `unused arguments (lhs = ..., rhs = ...)`; they are now matched positionally,
  as `prim_scatter()` already matched its `update_computation`.
* Improved the numerics for `nv_mod()`.
* The variadic array functions (`nv_concatenate()`, `nv_rbind()`, `nv_cbind()`,
  `nv_broadcast_scalars()`, `nv_broadcast_arrays()`, `nv_promote_to_common()`)
  say so when given no array, instead of warning or failing internally.
* `nv_array()` of a zero-length vector asks for a `shape` instead of failing
  inside the backend; which axis is empty cannot be inferred from the data.
* `nv_save()` and `nv_serialize()` given a single array say so, instead of
  failing inside `nv_subset()`. `nv_serialize()` to a connection returns
  invisibly.
* The `_like()` functions name `like` when it is an R value with no data type.
* `nv_solve()` and `nv_triangular_solve()` promote their operands, as
  `nv_matmul()` does, instead of refusing two arrays that disagree.
* On the `"quickr"` backend a call whose outputs are all empty emits the empty
  arrays directly, instead of an elementwise operation quickr rejects.
* `nv_any()`, `nv_all()` and `nv_sort()` are jitted, and
  `nv_psigamma()`'s `deriv` is no longer static, so it accepts an array as
  `prim_psigamma()` does.
* `nv_qnorm()` is accurate to its operand's data type rather than to the
  default float; its coefficients used to be materialized at the default.
* `nv_dnorm()`, `nv_pnorm()` and `nv_qnorm()` name their own operand when it
  is not a float.
* `prim_fill()` / `nv_fill()` check that `value` is something `dtype` can hold:
  a whole number for an integer data type, a non-negative one for an unsigned
  one, a logical or `0` / `1` for `bool`.
* Every data type of the float category counts as a float, so `f16` and `bf16`
  pass the checks that used to accept only `f32` and `f64`. `nv_pnorm()` and
  `nv_qnorm()` keep the narrower requirement, as they carry one coefficient
  set per width.
* The gradient of `nv_gamma()` is now correct for positive whole numbers.
* `prim_any()` / `prim_all()` (and `nv_any()` /
  `nv_all()`) now reject a non-boolean input when the call is traced.
  Type inference declared a `bool` output whatever the input was, so an
  integer operand reached the lowering and failed with `Data types of inputs
  and init_values must match`.
* Printed graphs, arrays and error messages now spell a data type the way anvl
  does, so `bool` no longer shows up as its MLIR spelling `i1`.
* Improved the documentation and various error messages.
* `nv_runif()` with `min == max` returns the `state` / `values` pair every
  other sampler returns, instead of the filled array on its own.

## Tests

* The environment variables that configure only the test suite are now spelled
  with an `ANVL_TEST` prefix: `ANVL_TEST_SKIP_QUICKR`. `ANVL_TEST` itself is
  unchanged.
* The suite can be run with `ANVL_DEFAULT_DEVICE=cpu:1`, which makes anything
  allocating on the first CPU device rather than following the trace land on a
  device of its own instead of agreeing with everything else by accident. The
  `default-device` workflow runs it that way on the `full-test` label.
* Moved some of pjrt's dispatcher tests into anvl.

# anvl 0.4.0

## Breaking changes

* Renamed `dim`/`dims` to `axis`/`axes` throughout the package (an axis is an
  index, a dimension is a size); `ndims()` is now `naxes()`.
* Renamed the primary array argument of `prim_*` / `nv_*` functions from
  `operand` to `x`.
* Renamed the `"xla"` backend to `"pjrt"`.
* `xla()` has been removed; use `jit()` instead.
* `nv_rdunif()` has been renamed to `nv_sample_int()`, mirroring R's
  `sample.int()`.
* `nv_runif()`'s `lower`/`upper` arguments are now `min`/`max`, `nv_rnorm()`'s
  `mu`/`sigma` are now `mean`/`sd`, and `nv_rbinom()`'s `n` is now `size`,
  matching the corresponding R functions.

## Bug fixes

* `nv_sample_int()` (formerly `nv_rdunif()`) was off by one: the first integer
  was drawn twice as often as it should have been, and the last integer was
  never drawn at all.

## Features

* New `nv_sample()` samples from an arbitrary population.
* New `nv_lower_tri()` and `nv_upper_tri()` (with `nv_lower_tri_like()` /
  `nv_upper_tri_like()`) return a boolean triangular mask for a given shape,
  mirroring base R's `lower.tri()` / `upper.tri()`.
* New functions for the normal distribution: `nv_dnorm()`, `nv_qnorm()`,
  and `nv_pnorm()` thanks to Louis Aslett.
  They are implemented to be accurate far into either tail.
* `nv_rnorm()`'s `mean` and `sd` now accept arrayish inputs.
* Dimension arguments (`dim`, `dims`, `dimension`, `permutation`) now accept
  negative values that count from the end, so `-1` refers to the last
  dimension.
* Reshaping functions accept a single `-1` in shape indicating a dimension
  to be inferred.
* Added support for 1-3 dimensional convolutions, thanks to
  Troy Hernandez.
* `AnvlArray` constructors and converters have gained a `check` argument
  that opts into scanning for `NA` values, see the "Gotchas" vignette
  for more information.
* `nv_var()` and `nv_sd()` now default to `dims = NULL`, which reduces
  over all dimensions and returns a scalar, consistent with the other
  reductions.
* `trace_fn()` gained an `optimize` argument controlling which graph
  optimization passes run on the traced graph. `TRUE` runs all passes, `FALSE`
  (default) runs none, and a character vector (e.g.
  `c("inline_scalars", "remove_unused_constants")`) selects a subset. `jit()` always traces with all
  passes enabled.
* Improved the installation vignette

## Performance

* Most `nv_*()` API functions are now JIT-compiled internally (via a new
  `@jit` roxygen roclet), speeding up eager-mode execution.
* Tracing (`trace_fn()`) performance has been improved.
* StableHLO lowering has been sped up.
* Calling `jit()`ted functions is now significantly faster.

## Bug fixes

* Reductions now reject dimensions that are out of range for the operand
  instead of silently ignoring them.
* `NULL` is now treated as an empty node when flattening and unflattening trees.
* `nv_argmax()` / `nv_argmin()` and `nv_cummax()` / `nv_cummin()` now break
  ties order-independently, so they return the same result on GPU as on CPU
  (#368). `nv_argmax()` / `nv_argmin()` prefer the smallest index;
  `nv_cummax()` / `nv_cummin()` prefer the last occurrence.
* `nv_diag()` now errors on non-1-D input instead of silently producing an
  incorrect result.
* `jit()` now rejects static arguments with reference semantics.
* Error messages now speak of arrays instead of tensors.

# anvl 0.3.0

## Breaking Changes

* `nv_empty()` / `nv_empty_like()` return arrays with unspecified
  contents (no longer zero-initialized).

## New Features

* On CPU, jitted XLA functions now back every non-aliased output with
  an R-owned RAWSXP. anvl appends a phantom donated input per
  unaliased output during lowering, allocates `pjrt::pjrt_empty()`
  buffers at execute time, and `pjrt` migrates the keepalive onto the
  output XPtr. The output's host bytes are then managed by R's GC.
* Renamed user-facing API functions to match base R names:
  `nv_sine()` -> `nv_sin()`, `nv_cosine()` -> `nv_cos()`,
  `nv_ceil()` -> `nv_ceiling()`, `nv_cholesky()` -> `nv_chol()`.
  The corresponding primitives were renamed in step:
  `prim_sine()` -> `prim_sin()`, `prim_cosine()` -> `prim_cos()`,
  `prim_cholesky()` -> `prim_chol()`.
* `nv_reduce_mean()` was renamed to `nv_mean()`.
* `nv_solve()` no longer requires `a` to be symmetric positive-definite as it
  uses LU instead of Cholesky decomposition.
  Because of this, it is no longer differentiable, as the reverse rule for
  LU is not implemented yet.
* `nv_chol()` / `prim_chol()` now default to `lower = FALSE`
  (upper-triangular factor), matching base R's `chol()`. Previously
  defaulted to `lower = TRUE`.

## New Features

### Linear algebra

* New matrix-decomposition primitives and corresponding `nv_*()`
  functions: `qr`, `lu`, `svd`, `eigh`. None of them implement a
  reverse rule yet.
* New API functions:
  * `nv_triangular_solve()` (wraps the already-existing
    `prim_triangular_solve()`).
  * `nv_det()` and `nv_determinant()`. The latter can also be called
    via the `determinant()` generic.
  * `nv_inv()`, which can also be called via `solve(operand)` (missing
    second argument).
* `qr`, `chol`, and `solve` from base R now dispatch to `nv_qr()`,
  `nv_chol()`, and `nv_solve()` on `AnvlArray` / `AnvlBox` inputs.

### Element-wise math

* New unary primitives and corresponding `nv_*()` functions:
  `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cosh`, `sinh`,
  `digamma`, `lgamma`, `polygamma`, `erf`, `erf_inv`, `erfc`.
* New API functions `nv_mod()` (flooring remainder) and `nv_trunc()`
  (truncation toward zero).

### Cumulative reductions

* New primitives and corresponding `nv_*()` functions: `cumsum`,
  `cumprod`, `cummax`, `cummin`. `prim_cumprod()` does not yet have
  a reverse rule.

### Sorting and searching

* New primitives `prim_sort()`, `prim_top_k()`, `prim_reduce()`,
  `prim_argmax()`, `prim_argmin()`.
* New API functions:
  * `nv_sort()` / `nv_argsort()` -- sort along a dimension, or return
    the permutation that does.
  * `nv_top_k()` -- the `k` largest values along a dimension.
  * `nv_median()` / `nv_quantile()` -- median / quantiles along a
    dimension. `median()` dispatches to `nv_median()`.
  * `nv_argmax()` / `nv_argmin()` -- index of the maximum / minimum
    along a dimension (ties broken by smallest index).
  * `nv_select()` -- select a slice along a dimension by index.

### Array construction / shape

* `nv_array()` gained a `byrow` argument that fills the array from an R
  object in row-major order, mirroring `matrix(byrow = TRUE)` (#165).
* New `nv_matrix(data, nrow, ncol, ...)` which works like R's `matrix()`.
* New API functions `nv_rbind()` and `nv_cbind()` and corresponding
  `rbind()` / `cbind()` generics.
* New API function `nv_flatten()` for flattening to 1-D.

### Misc

* New `AnvlArray` -> R `vector` converters: `as.numeric()`,
  `as.double()`, `as.integer()`, `as.logical()`, `as.vector()`.
* New function `await()` that blocks until the underlying computation
  has finished.
* New tree utilities `map_tree()` and `pmap_tree()` for applying
  functions leaf-wise over (possibly nested) lists.
* Added support for `range` generic.
* Improved NaN handling across various primitives and API functions.

## Other

* `nv_reduce_sum()`, `nv_reduce_prod()`, `nv_reduce_max()`,
  `nv_reduce_min()`, `nv_reduce_any()`, `nv_reduce_all()` and
  `nv_mean()` now default `dims = NULL`, which reduces over all
  dimensions and returns a scalar. Previously, `dims` was required.

## Bug Fixes

* The overloaded `%%` operator now calls the new `nv_mod()` to be
  consistent with base R.
* The reverse rule for `prim_reduce_prod()` no longer produces
  `NaN` / `Inf` gradients when the input contains zeros.
* The CI now actually runs the torch-comparison tests.
* `nv_runif()` not properly respects the `lower` argument.

# anvl 0.2.0

## Breaking Changes

* The package was renamed from `anvil` to `anvl` to avoid a conflict
  with the Bioconductor package `AnVIL`.
* `AnvilTensor`/`nv_tensor` were renamed to `AnvlArray` and `nv_array` to be
  more in line with R's `array()`.
  Also, `nv_aten()` was renamed to `nv_aval()`.
* Subsetting with `list()` (e.g. `x[list(1, 3)]`) is no longer supported.
  Use `array()` to wrap the indices instead, e.g. `x[array(c(1L, 3L))]`.
  This mirrors the input convention used everywhere else in the package.
* Removed *debug mode*.
* Remove NSE support for `nvl_if`. It now requires passing 0-argument
  closures as `true` and `false` arguments.
* Primitives renamed from `nvl_*` to `prim_*`.
  The underlying primitive object containing the rules and metadata
  is now part of the `JitPrimitive` function via the `primitive` attribute.

## New Features

* Better composability:
  `jit()`ted functions can now be used in other `jit()`-calls.
  This is the mechanism underlying the new *eager mode*.
* *Eager mode* was added:
  This means, you can now do `nv_add(1, nv_array(1:2))` and it will
  actually perform the computation and not only do type inference.
* An experimental [{quickr}](https://github.com/t-kalinowski/quickr) backend was added
  It only runs on CPU for now and supports a subset of available operations.
  You can enable it via the `backend` argument in `jit()` and
  `nv_array()` or via the `anvl.default_backend` option.
* New primitives:
  * `nvl_cholesky()` to compute the Cholesky decomposition of a matrix.
  * `nvl_triangular_solve()` to solve a system of linear equations with a triangular matrix.
* New API functions (+ corresponding R generic implementations):
  * `nv_diag()` to create a diagonal matrix from a 1-D tensor.
  * `nv_eye()` to create an identity matrix.
  * `nv_solve()` to solve a system of linear equations.
  * `nv_cholesky()` to compute the Cholesky decomposition of a matrix.
  * `nv_device()` constructs a backend-specific device object (e.g. `nv_device("cpu")`)
    that can be passed as `device` to array constructors like `nv_fill()` or `nv_iota()`.
  * `nv_crossprod()` and `nv_tcrossprod()` for matrix cross-products.
  * `nv_outer()` for the outer product.
  * `nv_extract_diag()` to extract the diagonal of a matrix.
  * `nv_trace()` to compute the trace of a matrix.
  * `nv_tril()` and `nv_triu()` to extract lower/upper triangular parts.
  * `nv_squeeze()` and `nv_unsqueeze()` to drop or add length-1 dimensions.
  * `nv_log2()` and `nv_log10()`.
  * `nv_is_infinite()` and `nv_is_nan()`.
  * `nv_sd()` and `nv_var()` for standard deviation and variance.
* `jit()` now accepts integer positions for the `static` argument.
* New S3 methods `dim()`, `nrow()`, `ncol()`, and `length()` for anvl arrays.
* Printing tensors via `nv_print()` now also works on GPUs.
* R vectors of length 1 and arrays are now auto-converted when being passed
  to `jit`ted functions.
* Improved device handling in `jit()`

## Performance

* Many operations are now done asynchronously, which improves performance,
  especially on GPUs.

## Bug Fixes

* +-Inf/NaN are correctly created for `f64` when inlined into the XLA exectuable (#182).
  This caused wrong results with e.g. `nv_reduce_max()` when working with `f64`.
* Corrected argument checks in `nv_iota()`.
* Fix check that `wrt` arguments in `gradient()` must be floats.
* `nv_subset()` and `nv_subset_assign()` now error on trailing-comma subscripts (#273).

## Documentation

* New vignette on implementing Gaussian Processes.
* New vignette on implementing Metropolis-Hastings sampling.

## Platform support and installation

* An installation guide was added.
* Linux on ARM is now supported (CPU only).
* To use the CUDA backend, it is now possible to install the `cuda12.8`
  package (see installation guide), which only requires a compatible CUDA
  driver.

# anvl 0.1.0

Initial release
