# Changelog

## anvl (development version)

### Breaking Changes

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
  [`map_tree()`](https://r-xla.github.io/anvl/dev/reference/map_tree.md)
  and
  [`pmap_tree()`](https://r-xla.github.io/anvl/dev/reference/pmap_tree.md)
  for applying functions leaf-wise over (possibly nested) lists.
- [`mean()`](https://rdrr.io/r/base/mean.html) and
  [`median()`](https://rdrr.io/r/stats/median.html) now error when
  called with `na.rm = TRUE`, since anvl arrays do not carry `NA`s.
  [`mean()`](https://rdrr.io/r/base/mean.html) also rejects non-zero
  `trim`.
- Added support for `range` generic.

#### NaN handling

- `NaN` handling across reductions, cumulative ops, sorting, and
  [`nv_argmax()`](https://r-xla.github.io/anvl/dev/reference/nv_argmax.md)
  /
  [`nv_argmin()`](https://r-xla.github.io/anvl/dev/reference/nv_argmin.md)
  is now deterministic (previously XLA-backend-defined in several
  cases). A new `nan_rm` argument (default `FALSE`) skips `NaN` values,
  and the base-R generics (`sum`, `max`, `mean`, `median`, …) forward
  `na.rm` to it.

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
- The overloaded `%%` operator now calls the new
  [`nv_mod()`](https://r-xla.github.io/anvl/dev/reference/nv_mod.md) to
  be consistent with base R.
- The reverse rule for
  [`prim_reduce_prod()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce_prod.md)
  no longer produces `NaN`/`Inf` gradients when the input contains
  zeros.
- The CI now actually runs the torch-comparison tests

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
