# Changelog

## anvl (development version)

### Breaking changes

- The type system of {anvl} was changed to avoid the problems reported
  in issue [\#373](https://github.com/r-xla/anvl/issues/373).
  Specifically, the ambiguity system was replaced with the `RData`
  system and a new system of rules for type promotions. With it, also
  the promotion behavior of various primitives and API functions was
  improved.
- `jit_eval()` was removed as it is no longer needed.

### Bug fixes

- [`prim_reduce()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce.md)’s
  `reductor` no longer has to name its arguments `lhs` and `rhs`. They
  were passed by name, so `function(a, b)` failed with
  `unused arguments (lhs = ..., rhs = ...)`; they are now matched
  positionally, as
  [`prim_scatter()`](https://r-xla.github.io/anvl/dev/reference/prim_scatter.md)
  already matched its `update_computation`.
- [`prim_reduce_any()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce_any.md)
  /
  [`prim_reduce_all()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce_all.md)
  (and
  [`nv_reduce_any()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_any.md)
  /
  [`nv_reduce_all()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_all.md))
  now reject a non-boolean input when the call is traced. Type inference
  declared a `bool` output whatever the input was, so an integer operand
  reached the lowering and failed with
  `Data types of inputs and init_values must match`.

### Tests

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
- [`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_runif.md)’s
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
- [`nv_argmax()`](https://r-xla.github.io/anvl/dev/reference/nv_argmax.md)
  /
  [`nv_argmin()`](https://r-xla.github.io/anvl/dev/reference/nv_argmin.md)
  and
  [`nv_cummax()`](https://r-xla.github.io/anvl/dev/reference/nv_cummax.md)
  /
  [`nv_cummin()`](https://r-xla.github.io/anvl/dev/reference/nv_cummin.md)
  now break ties order-independently, so they return the same result on
  GPU as on CPU ([\#368](https://github.com/r-xla/anvl/issues/368)).
  [`nv_argmax()`](https://r-xla.github.io/anvl/dev/reference/nv_argmax.md)
  /
  [`nv_argmin()`](https://r-xla.github.io/anvl/dev/reference/nv_argmin.md)
  prefer the smallest index;
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
  [`prim_argmax()`](https://r-xla.github.io/anvl/dev/reference/prim_argmax.md),
  [`prim_argmin()`](https://r-xla.github.io/anvl/dev/reference/prim_argmin.md).
- New API functions:
  - [`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md)
    /
    [`nv_argsort()`](https://r-xla.github.io/anvl/dev/reference/nv_argsort.md)
    – sort along a dimension, or return the permutation that does.
  - [`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md)
    – the `k` largest values along a dimension.
  - [`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md)
    /
    [`nv_quantile()`](https://r-xla.github.io/anvl/dev/reference/nv_quantile.md)
    – median / quantiles along a dimension.
    [`median()`](https://rdrr.io/r/stats/median.html) dispatches to
    [`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md).
  - [`nv_argmax()`](https://r-xla.github.io/anvl/dev/reference/nv_argmax.md)
    /
    [`nv_argmin()`](https://r-xla.github.io/anvl/dev/reference/nv_argmin.md)
    – index of the maximum / minimum along a dimension (ties broken by
    smallest index).
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

- [`nv_reduce_sum()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_sum.md),
  [`nv_reduce_prod()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_prod.md),
  [`nv_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_max.md),
  [`nv_reduce_min()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_min.md),
  [`nv_reduce_any()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_any.md),
  [`nv_reduce_all()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_all.md)
  and
  [`nv_mean()`](https://r-xla.github.io/anvl/dev/reference/nv_mean.md)
  now default `dims = NULL`, which reduces over all dimensions and
  returns a scalar. Previously, `dims` was required.

### Bug Fixes

- The overloaded `%%` operator now calls the new
  [`nv_mod()`](https://r-xla.github.io/anvl/dev/reference/nv_mod.md) to
  be consistent with base R.
- The reverse rule for
  [`prim_reduce_prod()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce_prod.md)
  no longer produces `NaN` / `Inf` gradients when the input contains
  zeros.
- The CI now actually runs the torch-comparison tests.
- [`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_runif.md)
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
  caused wrong results with
  e.g. [`nv_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_max.md)
  when working with `f64`.
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
