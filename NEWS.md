# anvl (development version)

## Breaking changes

* The type system of {anvl} was changed to avoid the problems reported in issue #373.
  Specifically, the ambiguity system was replaced with the `RData` system and a new system of rules for type promotions.
  With it, also the promotion behavior of various primitives and API
  functions was improved.
* `jit_eval()` was removed as it is no longer needed.
* `nv_reduce_sum()`, `nv_reduce_prod()`, `nv_cumsum()` and `nv_cumprod()` now
  accumulate a boolean array at the default integer data type instead of returning a boolean.
* `as.vector()` on an `AnvlArray` now only accepts `mode = "any"` (the
  default) and errors for any other `mode`.
* The `steps` argument of `nv_seq()` / `nv_seq_like()` was removed.
* `default_backend()` is now called `active_backend()`.
* There is now exactly one backend used at a time and it is configured via the
  `anvl.backend` option.
* A `Shape` is now represented as an integer vector.
* `prim_top_k()`, `prim_cummax()`, `prim_cummin()`,
  `prim_rng_bit_generator()` and the `nv_*` samplers (`nv_runif()`,
  `nv_rnorm()`, `nv_rbinom()`, `nv_sample()`, `nv_sample_int()`) now return
  *named* lists -- `values` / `indices` for the first three, `state` /
  `values` for the rest -- as `prim_qr()` and `prim_svd()` already did.
  Positional indexing keeps working.
* `nv_chol()` / `prim_chol()` accept batched inputs again -- axes before the
  last two are batch axes, as their pages say and as the lowering has always
  handled. A check added earlier in this cycle had narrowed them to exactly two
  axes.
* `nv_inv()` names `x` when its input is not a float, instead of reporting
  `nv_solve()`'s `a` and `b`; `gradient()` accepts any float output data type,
  matching its own message, and its hint no longer suggests an `f32` array that
  the `"quickr"` backend refuses; and the quickr lowering's messages say `bool`
  rather than stablehlo's `pred`.
* `nv_solve()`, `nv_triangular_solve()` and `nv_conv1d()` / `nv_conv2d()` /
  `nv_conv3d()` now promote their operands to a common data type, like every
  other `nv_*` function: `nv_solve(a_f32, b_f64)` gives `f64` where it used to
  refuse the pair, and an integer input meets a float weight at the float. They
  used to apply `promote_rdata_common()`, the primitive layer's rule, so a
  mismatch was rejected -- with a hint telling the caller to use an operation
  that promotes, which is what they were already calling. The solves still
  require the common data type to be a float, and now say so naming `a`
  instead of leaking `prim_lu()`'s operand name.
* `nv_quantile()` and `nv_median()` now interpolate at a float data type, so
  they are correct for a non-float input: `nv_median(nv_array(1:4))` was `1`
  and is now `2.5`. `probs` used to be built at the input's data type, where
  `0.5` rounds to `0` and every quantile came back as the smallest element.
  `interpolation = "lower"` / `"higher"` / `"nearest"` now also return a float,
  as documented.
* `nv_crossprod()` and `nv_tcrossprod()` now require a matrix, like
  [base::crossprod()]. They transpose every axis, so an input above rank 2
  silently contracted the wrong pair.
* `x %/% y` on an `AnvlArray` now performs flooring division instead of
  returning `NULL`, and an unimplemented member of a group generic errors.
* `nv_rbinom()`, `nv_sample_int()`, `nv_seq()`, the twelve functions taking
  `nan_rm`, the reductions' `drop`, `prim_chol()`'s and
  `prim_triangular_solve()`'s flags, `prim_static_slice()`'s `strides`,
  `prim_sort()`'s `xs`, `prim_fill()`'s `shape`, `nv_eye()`'s `n`,
  `nv_quantile()`'s `probs`, the six variadic functions' empty `...` and the
  `_like` functions' `like` are now all checked in anvl, with a message naming
  the argument. Several of these used to reach the backend, or produced a
  base-R warning or `NULL`.
* A cumulative operation or a quantile over a zero-size axis is now refused
  with an anvl error; the reductions still define the empty case and keep it.
* `nv_serialize()` now returns `NULL` invisibly when it writes to a connection,
  as its documentation says (it returned the connection).
* `prim_fill()` (and so `nv_fill()` / `nv_fill_like()`) now checks that
  `value` is something `dtype` can hold: a number for a float, a whole number
  for an integer, a non-negative whole number for an unsigned integer and a
  logical for `bool`, with `NA` rejected. These used to reach the backend.
* `nv_rbinom()` and `nv_sample_int()` now reject a boolean `dtype`. A boolean
  cannot hold a count or an index: `nv_rbinom()` used to return `bool` for
  `size = 1` and silently `i32` above it, and `nv_sample_int()` collapsed every
  drawn index to `TRUE`.
* Every data type of the float category now counts as a float, so `f16` and
  `bf16` pass the checks that used to accept only `f32` and `f64` (and the
  error message is now "must be a float data type"). The RNG, which assembles
  floats out of random bits, still requires a 32- or 64-bit float and says so.

## Features

* The default data types for floating point numbers and integers can now be
  configured via the `anvl.default_dtypes` field.
  You can configure this for a specific scope via `local_default_dtypes()`
  and `with_default_dtypes()`.
  To convert a function to one running at a specified precision, use
  `with_dtypes()`.
* `nv_polygamma()`'s `n` is now an ordinary arrayish argument rather than a
  static one, so it accepts an array and not just a plain R value, matching
  `prim_polygamma()` and JAX's `jax.scipy.special.polygamma()`.
* New `nv_linspace()` and `nv_linspace_like()`, replacing `nv_seq()` with 
  a provided `steps` argument.
* `as.vector` now and returns `bit64::integer64`
  for integer types that don't fit into R's 32 bit integers.
  With this chane the `device_arg` parameter was removed from `jit()` as it is no longer needed.

## Bug fixes

* `as.vector()` now works correctly for `AnvlArray`s that are converted to
  `bit64::integer64`. It used to drop that class along with the shape,
  exposing the raw 64-bit pattern as a double.
* Fixed the reverse rule of `prim_convert`.
* The quickr lowering no longer emits an elementwise operation for a result
  with a zero-size axis, which the development version of quickr rejects even
  where both operand shapes agree. An empty result is emitted directly.
* `prim_scatter()` now checks that `update_computation` returns one value of
  `x`'s data type, as `prim_reduce()` already did for its `reductor`.
* `prim_reduce()` now passes the arguments to the reductor by position.
  Previously, the arguments of the reductor had to be `(lhs, rhs)` and using
  using a function such as `(a, b) a + b` failed.
* `prim_reduce_any()` / `prim_reduce_all()` (and `nv_reduce_any()` /
  `nv_reduce_all()`) now reject a non-boolean input.
* `nv_runif()` with `min == max` now returns the `state` / `values` list every
  other sampler returns, instead of the filled array on its own.
* `nv_qnorm()` now returns `p`'s data type whatever the default float is. Its
  tail coefficients and its infinities were plain R numbers with nothing typed
  to yield to, so they committed at the default and dragged the result up with
  them.
* `prim_while()` now names every state member whose data type or shape changes
  across the body; it used to report the first label repeated once per mismatch,
  without the data types.
* `nv_array()` of a zero-length vector gives a length-0 array instead of
  failing inside pjrt.
* The staging warning (`anvl_staging_widens_warning`) no longer fires where the
  caller has no way to avoid the staging -- under a default integer narrower
  than `i32`, where converting in its own category first would stage through
  `i32` too, or under a default float no backend can materialize. Its hint used
  to name a remedy that could not work.
* `prim_scatter()`'s `update_computation` may close over an array again. The
  inference stub asked for the sub-graph's constants to already be bound, which
  they are not at inference time, so a closed-over array failed with
  `GraphValue not found in environment` where `prim_reduce()`'s `reductor`
  accepted one.
* `nv_serialize()` and `nv_save()` given a single array now say so, instead of
  failing inside `nv_subset()`: `checkmate::assert_list(types = )` subsets its
  input, and an `AnvlArray` has a `[` method.
* `nv_subset()` accepts a plain R array, as its page says.
* `nv_top_k()` checks `k` before coercing it, so a fractional or logical `k` is
  refused rather than silently truncated, and it validates `with_indices`.
* `assert_shapevec()` -- and so every `shape` argument -- rejects a fractional
  axis size instead of truncating it, and one above `.Machine$integer.max`
  instead of turning it into `NA`.
* `prim_while()` names `init` when it is not a named list. A fully unnamed list
  slipped past the check (`names()` is `NULL`) and the call died blaming `body`.
* `nv_pnorm()` and `nv_qnorm()` refuse `f16` / `bf16`, which their page already
  said they do not accept; a narrower float silently took the `f64`
  coefficients.

## Documentation

* Corrections found by auditing every page against the running package:
  `prim_clamp()`'s formula (`min(max(min_val, x), max_val)`, which differs from
  what was documented when `min_val > max_val`), `nv_tril()`'s and
  `nv_triu()`'s `diagonal`, the six shift pages' accepted data types (*integer*,
  not *integerish* -- `bool` is refused), `prim_bitcast_convert()`'s `bool`
  exclusion, `common_dtype()`'s promotion rule (a signed and an unsigned
  integer meet at a wider *signed* type), the `axis = NULL` shape of
  `nv_cumsum()` / `nv_cumprod()` / `nv_cummax()` / `nv_cummin()`,
  `nv_transpose()`'s `NULL` permutation, `prim_triangular_solve()`'s
  `transpose_a` and its example matrix, and the CHLO specification links for
  `prim_erf_inv()` and `prim_top_k()`.
* Every `prim_*` and `nv_*` help page now states, in its parameters and its
  return value, which data types it accepts, what an R value among them
  commits to, whether operands are promoted and whether scalars are broadcast.
  The group names (*numeric*, *integerish*, *float*, ...) are defined once on
  the new `?dtypes` page, which also documents the default data type an R
  value takes, and are shown on `?arrayish`.
* Primitives modelled on a StableHLO or CHLO op now link that op's
  specification instead of restating it.
* `prim_chol()` / `nv_chol()` and `prim_triangular_solve()` /
  `nv_triangular_solve()` say that differentiation is only implemented for a
  single matrix, not a batch, which is what their `reverse` rules do.
* `?dtypes` says that `f16` and `bf16` count as float data types everywhere
  anvl reasons about data types but are not materialized by any backend today,
  and `?common_dtype` that the two have no true common type (they give `f16`).
* `?AnvlBackendQuickr` and `vignette("primitives")` no longer claim the boolean
  reductions have an integer form on pjrt -- `prim_reduce_any()` /
  `prim_reduce_all()` take a boolean operand on every backend.
* `vignette("random-numbers")` had the promotion direction backwards: `mean` and
  `sd` decide what the sample is drawn at when `dtype` is unset, not the other
  way around.
* `vignette("internals")` no longer implies that a use site asking for `f32`
  demonstrates the default float, and says when the staging warning fires.
* `vignette("logistic-regression")` follows the configured default float instead
  of pinning `f32` for the data and the default for the learning rate, which did
  not agree under an `f64` default.
* A number of error messages now speak anvl's vocabulary -- *scalar* rather than
  "0-dimensional array", *axis size* rather than "dimension", and the offending
  shapes rather than "lhs and rhs are not broadcastable".
* These operations now validate their arguments themselves, so the message
  names the argument the caller passed and its shape or data type:
  `prim_clamp()`, `prim_ifelse()`, `prim_polygamma()`,
  `prim_broadcast_in_axes()`, `prim_pad()`, `prim_reduce()`, `prim_if()`,
  `prim_static_slice()`, `prim_dynamic_slice()`,
  `prim_dynamic_update_slice()`, `prim_gather()`, `prim_reshape()`,
  `prim_concatenate()`, `prim_iota()`, `prim_dot_general()`, `nv_matmul()`,
  `nv_concatenate()`, `nv_quantile()` and `nv_conv1d()` / `2d` / `3d`.
  `prim_dynamic_slice()` and `prim_dynamic_update_slice()` in particular now
  check that each start index is an integer scalar and that there is one per
  axis of `x`, and `prim_gather()` that `start_indices` is integral.
* Shapes in error and warning messages are written `(2x3)`, via the new
  `shape_repr()` / `shapes_repr()`. The repr spelling (`f32[2,3]`,
  `RData(double, (2,3))`) is separate and unchanged.
* `?nv_quantile`'s interpolation formula is stated in 1-based terms, so it
  gives the number the function returns.
* `?nv_convert` says what happens to a value the target cannot hold: an
  integer narrowing wraps, a float reaching an integer truncates toward zero
  and saturates, a float narrowing rounds.
* Corrected: `?LiteralArray` (a `nv_fill()` is a recorded operation, not a
  constant), `?to_abstract` (an R value becomes `RData`), `?nv_eye`'s `like`,
  `?nv_if`'s branches (data types must agree too, since nothing is promoted),
  `?nv_concatenate`'s `axis = NULL`, `?nv_polygamma`'s promotion sentence,
  `?assert_shapevec`'s `min_len` and return value, and the `shape` argument of
  `?nv_iota` / `?nv_lower_tri` / `?nv_upper_tri`, which cannot be `integer()`.
* `vignette("jit")` no longer credits the default float for a literal that took
  its data type from the array it met, and `vignette("anvl")` no longer lists
  `bool` among what `default_dtypes()` reports.

## Tests

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
