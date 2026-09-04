# anvl (development version)

## Breaking changes

* The type system of {anvl} was changed to avoid the problems reported in issue #373.
  Specifically, the ambiguity system was replaced with the `RData` system and a new system of rules for type promotions.
  With it, also the promotion behavior of various primitives and API
  functions was improved.
* `jit_eval()` was removed as it is no longer needed.

## Bug fixes

* The gradient of a conversion into a non-float data type is now zero instead
  of one. `prim_convert()` / `nv_convert()` passed the cotangent through
  whatever the data types were, so `nv_convert(nv_convert(x, "i32"), "f64")`
  reported a gradient of 1 where `nv_floor()` -- the same function on the
  reals -- correctly reported 0. Conversions between floats still pass the
  gradient through.
* `prim_scatter()` now checks that `update_computation` returns one value of
  `x`'s data type, as `prim_reduce()` already did for its `reductor`. A
  combiner returning something else made type inference declare a data type
  the call could not produce, and failed in the backend.
* `prim_reduce()`'s `reductor` no longer has to name its arguments `lhs` and
  `rhs`. They were passed by name, so `function(a, b)` failed with
  `unused arguments (lhs = ..., rhs = ...)`; they are now matched positionally,
  as `prim_scatter()` already matched its `update_computation`.
* `prim_reduce_any()` / `prim_reduce_all()` (and `nv_reduce_any()` /
  `nv_reduce_all()`) now reject a non-boolean input when the call is traced.
  Type inference declared a `bool` output whatever the input was, so an
  integer operand reached the lowering and failed with `Data types of inputs
  and init_values must match`.

## Documentation

* The help pages of the binary primitives now have a *Data Types* section
  saying how their operands reach one data type: an operand that has a data
  type keeps it, and an R value is built at the one the others have, within
  its own category. The text comes from a shared roxygen2 template
  (`man-roxygen/section_dtypes.R`), so it is written once for all of them.

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
