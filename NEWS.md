# anvl (development version)

## Breaking changes

* The pytree module moved to pjrt, which owns the native `PJRTNode` tree
  type used by its eager-dispatch cache. `flatten()`, `build_tree()`,
  `unflatten()`, `tree_size()`, `tree_path()`, `map_tree()`, and
  `pmap_tree()` are re-exported from pjrt and keep working; trees are now
  opaque native handles instead of R lists. `filter_list_node()` and
  `reindex_tree()` were removed (superseded by `pjrt::filter_by_names()`
  and `pjrt::tree_concat()`).

## Features

* Jitted functions with static arguments now cache in pjrt's native
  dispatcher instead of falling back to the R-side cache, removing the
  per-call R overhead for static-arg jits.

* The launch overhead of calling a jitted function dropped by roughly 3x:
  outputs are wrapped from the dispatcher's natively-read metadata instead
  of per-output S3 dtype()/shape()/device() reads, the `jit(backend =
  "auto")` wrapper calls the backend's evaluated-args fast entry instead of
  re-capturing the arguments via match.call(), and several per-call
  lookups were hoisted out of the hot path.

* `nv_array()`, `nv_scalar()`, `as_array()`, and the `as.integer()` /
  `as.double()` / `as.logical()` / `as.vector()` methods for
  `AnvlArray` gained a `check` argument that opts into scanning for
  `NA` values during host -> device and device -> host transfers. See
  the "Gotchas" vignette.

* `nv_var()` and `nv_sd()` now default to `dims = NULL`, which reduces
  over all dimensions and returns a scalar, consistent with the other
  reductions.

## Performance

* Most `nv_*()` API functions are now JIT-compiled internally (via a new
  `@jit` roxygen roclet), speeding up eager-mode execution.

## Bug fixes

* `NULL` is now treated as an empty node when flattening and unflattening trees.
  It contributes no leaves but is preserved structurally, so functions with
  optional arguments (e.g. `function(x, y = NULL)`) round-trip correctly.

* `nv_argmax()` / `nv_argmin()` and `nv_cummax()` / `nv_cummin()` now break
  ties order-independently, so they return the same result on GPU as on CPU
  (#368). `nv_argmax()` / `nv_argmin()` prefer the smallest index;
  `nv_cummax()` / `nv_cummin()` prefer the last occurrence.

* `nv_diag()` now errors on non-1-D input instead of silently producing an
  incorrect result.


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
