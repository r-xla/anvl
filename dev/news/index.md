# Changelog

## anvl (development version)

### Breaking changes

- The primary array argument (previously `operand`) of array
  transformation functions (`prim_<*>` and `nv_<*>`) is now consistently
  called `x`.
- The `"xla"` backend has been renamed to `"pjrt"`, after the runtime it
  uses. Pass `backend = "pjrt"` to
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md),
  [`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md),
  [`local_backend()`](https://r-xla.github.io/anvl/dev/reference/local_backend.md),
  and friends;
  [`backend()`](https://r-xla.github.io/anvl/dev/reference/backend.md)
  and
  [`default_backend()`](https://r-xla.github.io/anvl/dev/reference/default_backend.md)
  now return `"pjrt"`. The constructor `AnvlBackendXla()` is now
  [`AnvlBackendPjrt()`](https://r-xla.github.io/anvl/dev/reference/AnvlBackendPjrt.md)
  and the internal `compile_xla()` is now
  [`compile_pjrt()`](https://r-xla.github.io/anvl/dev/reference/compile_pjrt.md).
- `xla()` has been removed. Use
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) instead:
  it compiles through the same pipeline, lazily on the first call. Warm
  a jitted function up by calling it once with representative inputs.

### Features

- Dimension arguments (`dim`, `dims`, `dimension`, `permutation`) now
  accept negative values that count from the end, so `-1` refers to the
  last dimension. This works at both layers: in the `prim_*` primitives
  ([`prim_transpose()`](https://r-xla.github.io/anvl/dev/reference/prim_transpose.md),
  [`prim_concatenate()`](https://r-xla.github.io/anvl/dev/reference/prim_concatenate.md),
  `prim_reduce_*()`,
  [`prim_reduce()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce.md),
  [`prim_cumsum()`](https://r-xla.github.io/anvl/dev/reference/prim_cumsum.md)
  /
  [`prim_cumprod()`](https://r-xla.github.io/anvl/dev/reference/prim_cumprod.md)
  /
  [`prim_cummax()`](https://r-xla.github.io/anvl/dev/reference/prim_cummax.md)
  /
  [`prim_cummin()`](https://r-xla.github.io/anvl/dev/reference/prim_cummin.md),
  [`prim_argmax()`](https://r-xla.github.io/anvl/dev/reference/prim_argmax.md),
  [`prim_argmin()`](https://r-xla.github.io/anvl/dev/reference/prim_argmin.md),
  [`prim_reverse()`](https://r-xla.github.io/anvl/dev/reference/prim_reverse.md),
  [`prim_iota()`](https://r-xla.github.io/anvl/dev/reference/prim_iota.md),
  [`prim_sort()`](https://r-xla.github.io/anvl/dev/reference/prim_sort.md))
  and in the `nv_*` functions built on top of them, including
  [`nv_squeeze()`](https://r-xla.github.io/anvl/dev/reference/nv_squeeze.md),
  [`nv_unsqueeze()`](https://r-xla.github.io/anvl/dev/reference/nv_unsqueeze.md),
  [`nv_select()`](https://r-xla.github.io/anvl/dev/reference/nv_select.md),
  [`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md),
  [`nv_quantile()`](https://r-xla.github.io/anvl/dev/reference/nv_quantile.md)
  and
  [`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md)
  ([\#396](https://github.com/r-xla/anvl/issues/396)).
- [`nv_reshape()`](https://r-xla.github.io/anvl/dev/reference/nv_reshape.md)
  and
  [`prim_reshape()`](https://r-xla.github.io/anvl/dev/reference/prim_reshape.md)
  accept a single `-1` in `shape`, whose extent is then inferred from
  the number of elements of the input, e.g. `nv_reshape(x, c(2, -1))`
  ([\#396](https://github.com/r-xla/anvl/issues/396)).
- Reductions now reject dimensions that are out of range for the operand
  instead of silently ignoring them
  ([\#396](https://github.com/r-xla/anvl/issues/396)).
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
  [`upper.tri()`](https://rdrr.io/r/base/lower.tri.html). As in base R,
  the main diagonal is excluded by default; pass `diagonal = 0L` to
  include it. Use
  [`nv_tril()`](https://r-xla.github.io/anvl/dev/reference/nv_tril.md) /
  [`nv_triu()`](https://r-xla.github.io/anvl/dev/reference/nv_triu.md)
  to zero out a triangle of an existing array.
- [`trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md)
  gained an `optimize` argument controlling which graph optimization
  passes run on the traced graph. `TRUE` runs all passes, `FALSE`
  (default) runs none, and a character vector (e.g.
  `c("inline_scalars", "remove_unused_constants")`) selects a subset.
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) always
  traces with all passes enabled.
- New
  [`nv_dnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md)
  computes the normal distribution’s probability density function (or,
  with `log = TRUE`, its log-density).
- New
  [`nv_pnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md)
  computes the normal distribution’s cumulative distribution function
  (`lower_tail = FALSE` for the upper tail, `log_p = TRUE` for the
  log-probability, staying accurate far into either tail).
- [`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md),
  [`nv_scalar()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md),
  [`as_array()`](https://r-xla.github.io/anvl/dev/reference/as_array.md),
  and the [`as.integer()`](https://rdrr.io/r/base/integer.html) /
  [`as.double()`](https://rdrr.io/r/base/double.html) /
  [`as.logical()`](https://rdrr.io/r/base/logical.html) /
  [`as.vector()`](https://rdrr.io/r/base/vector.html) methods for
  `AnvlArray` gained a `check` argument that opts into scanning for `NA`
  values during host -\> device and device -\> host transfers. See the
  “Gotchas” vignette.
- [`nv_var()`](https://r-xla.github.io/anvl/dev/reference/nv_var.md) and
  [`nv_sd()`](https://r-xla.github.io/anvl/dev/reference/nv_sd.md) now
  default to `dims = NULL`, which reduces over all dimensions and
  returns a scalar, consistent with the other reductions.
- Supports 1-3d convolutions.

### Performance

- Most `nv_*()` API functions are now JIT-compiled internally (via a new
  `@jit` roxygen roclet), speeding up eager-mode execution.
- Tracing
  ([`trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md))
  performance has been improved.
- Tracing now accumulates primitive calls in a
  [`fastmap::fastqueue`](https://r-lib.github.io/fastmap/reference/fastqueue.html)
  (amortised-O(1) append) instead of an R list grown with
  [`c()`](https://rdrr.io/r/base/c.html) (copy-on-modify, O(n^2)).
  Tracing large unrolled graphs is substantially faster, e.g. ~1.36x for
  an 8000-op chain, with the gain growing with graph size.
- StableHLO lowering forwards the trace-time output types to the `hlo_*`
  builders (via an `output_types` argument passed to the lowering
  rules), so stablehlo skips redundant type inference when lowering the
  graph.
- Calling
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)ted
  functions is now significantly faster.

### Bug fixes

- `NULL` is now treated as an empty node when flattening and
  unflattening trees. It contributes no leaves but is preserved
  structurally, so functions with optional arguments
  (e.g. `function(x, y = NULL)`) round-trip correctly.

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

- Errors raised while tracing are now phrased in anvl’s own vocabulary
  ([\#298](https://github.com/r-xla/anvl/issues/298)). Messages
  originating in the `stablehlo` package used the StableHLO spec’s
  terminology; they now speak of arrays instead of tensors, and of `x`
  instead of `operand`. For example,
  `` `operand` must have dtype FloatType `` became
  `` `x` must have dtype FloatType ``.

- [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) now
  rejects static arguments with reference semantics – an environment
  (and therefore also an R6 or reference class object) or an external
  pointer, including one nested inside a static list
  ([\#17](https://github.com/r-xla/anvl/issues/17)). Such a value can be
  mutated in place while the compilation cache key stays equal, which
  silently reused a program compiled from its old contents. Functions
  and device objects remain valid static values.

- Errors raised while tracing now speak of arrays instead of tensors
  ([\#298](https://github.com/r-xla/anvl/issues/298)). Previously,
  messages that originated in the `stablehlo` package used the StableHLO
  spec’s terminology,
  e.g. `` `lhs` and `rhs` must have the same tensor type. x Got tensor<4xi32> and tensor<4xf32>. ``

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
