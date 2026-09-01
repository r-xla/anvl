# anvl (development version)

## Breaking changes

* **Improved type system**:
  - Primitives and `as_anvl_arrays()` now support the specification of promotion rules.
  - R values (length-1 `vector`s and `array`s) are no longer converted to the default dtype
    at the jit-boundary, but instead promoted to whichever data type they are combined with.
    This resolves #373 (issue reported by @louisaslett).
    See the *type promotion* vignette for more details on the improved type system.
  - The `ambiguity` concept no longer exists in the type system.
  - `promote_like()` and `promote_dtype()` refuse an input the target data type
    cannot hold, rather than narrowing it silently: `nv_clamp(0, x_i32, 1.5)` is
    now an error (write the literal in the array's category, e.g. `0L`), and so
    is `x_i32[i] <- 1.5`. Pass `force = TRUE` to the rule where the narrowing is
    the point.
  - `nv_pad()`'s `padding_value` now *yields* to `x` instead of being promoted
    to it: an R value is built at `x`'s data type, within its own category
    (`nv_pad(x_f64, 0)`, not `0L`), and a value that already has a data type
    must have `x`'s -- one that disagrees is reported rather than widened.
  - A primitive no longer promotes its arguments unless it says so: the
    `promote` argument of `new_primitive()` defaults to `NULL`, and the
    primitives whose arrayish arguments must agree declare
    `promote = promote_yield()`. The rule is applied before the primitive's body
    runs, so a body may read a data type or trace a sub-graph without meeting an
    argument that has not taken one yet.
* `nv_rnorm()`'s `dtype` now defaults to `NULL`, which takes the sample's data
  type from `mean` and `sd` where either is a real array, and falls back to
  `"f32"` where both are bare R values. It used to always be `"f32"` unless the
  caller said otherwise, so an `f64` `mean` was silently narrowed. A `mean` and
  `sd` that agree on a data type the generator cannot draw at (both integer
  arrays, say) is an error naming `dtype`, rather than a silent `"f32"`.
* The trace no longer has a node class for an R argument whose data type is not
  decided yet. `GraphRData` is gone: such an argument is an input like any
  other, a `GraphValue` whose aval is an `RData`, and the bookkeeping the trace
  needs for it lives on the descriptor rather than on a node of the graph.
* `RDataInput` is gone with it. A finished graph's inputs all carry a plain
  `AbstractArray`, and `graph$rdata_types` says, one entry per input, which R
  storage type an input is uploaded from (`NA` for one the caller passes through
  as an array). The two together carry what the subclass did, without an
  `AbstractArray` subclass every aval consumer has to tolerate.
  As a side effect the graph printer's `<- integer` annotation, which used to
  follow the aval, now follows the input slot -- so it no longer appears in the
  Outputs section for a graph like `jit(identity)` whose output *is* its input.
* A primitive no longer declares how it promotes. `new_primitive()` and
  `AnvlPrimitive()` lost their `promote` argument, and the wrapper that applied
  the rule around a primitive's body is gone; a primitive whose operands must
  agree calls `promote_operands()` at the top of its own body instead. The
  promotion is where the operands are, rather than in an argument whose effect
  is applied out of sight, and `only =` is no longer needed anywhere: a body
  passes just the operands that have to agree.
* A promotion rule is now a **function** of the call's arguments that returns the
  data type each one is brought to, rather than an opaque `PromoteRule` object
  the framework knows how to interpret. `promote_common()`, `promote_like()`,
  `promote_dtype()` and `promote_yield()` are unchanged to call -- they now
  return such a function -- and a package can write a rule of its own and pass it
  as `.promote`, or combine it with anvl's through `promote_grouped()`. See the
  *Writing a rule* section of `?promote_rule`.
* `promote_common()` gained a `fallback` argument: the data type to settle on
  when every input is a bare R value and there is none to read off the inputs,
  in place of the default those would commit to on their own. An input that has
  a data type wins over it. This is how `nv_rnorm()` reaches the default float.
* `shape_abstract()` and `naxes_abstract()` have been removed: `shape()` and
  `naxes()` now answer for every arrayish value, bare R values included, so the
  two were the same function under a second name.

## Bug fixes

* A negative R integer used at an unsigned data type produced a different answer
  in each mode: eagerly it read back as nonsense, and under `jit()` it was
  written straight into the IR as `dense<-1> : tensor<ui32>`, which is not valid
  StableHLO. An R integer is signed, so it is now built at `i32`/`i64` and
  converted by the program, giving XLA's wrap-around in both modes.
* An R value used at a data type outside its own category from two sibling
  sub-graphs -- `prim_if()`'s two branches, `prim_while()`'s condition and body
  -- failed with "GraphValue not found in environment". The conversion is
  recorded in the sub-graph being traced, so it can no longer be handed to a
  sibling that does not compute it.
* `nv_solve()` and `nv_triangular_solve()` let an R value take `a`'s data type
  instead of committing it to its own default, so `nv_solve(a_f64, b_r_matrix)`
  works. Two typed arrays that disagree are still rejected rather than one being
  widened.
* `nv_pad()` builds its padding value at the array's data type. It used to
  commit the value to its own default first, so `nv_pad(x_f64, 0)` failed with a
  data type mismatch and only an `f32` array with a double padding value
  happened to line up.

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
