#' @include jit.R
NULL

nv_unif_rand <- function(shape, dtype) {
  dtype <- assert_rng_float_dtype(dtype)
  shape <- assert_shapevec(shape)

  # 1. Generate random bits (64)
  # 2. We use these as mantissa bits for float, where we set the exponent to 1.0
  # 3. Because we have an implicit leading 1, we get a number in [1, 2) -> need to shift to [0, 1)

  ui_dtype <- as_dtype(paste0("ui", dtype_width(dtype)))

  # generate random bits
  # use THREE_FRY as rng algorithm: JAX default
  rbits <- prim_rng_bit_generator(
    state = state,
    "THREE_FRY",
    ui_dtype,
    shape = shape
  )

  # shift value: 9 for f32, 11 for f64
  shift <- if (dtype == "f32") 9L else 11L

  # shift to the right, s.t. exponent bits are all 0
  mantissa <- nv_shift_right_logical(rbits$values, shift)

  one_bits <- nv_bitcast_convert(
    nv_fill_like(state, 1.0, shape = integer(), dtype = dtype),
    dtype = ui_dtype
  )

  # bitwise or -> exponent from 1.0 (float), mantissa is random
  U <- nv_or(mantissa, one_bits)

  # convert back to requested dtype
  # resulting RVs  are in [1, 2)
  U <- nv_bitcast_convert(U, dtype = dtype)

  # shift to [0, 1)
  U - 1L
}

# Random Number Generation API
# This file contains user-facing RNG sampling functions

#' @rdname nv_uniform
#' @template param_shape
#' @param dtype (`NULL` | `character(1)` | [`DataType`][xlamisc::DataType])\cr
#'   Floating point data type of the sample.
#'   The default (`NULL`) takes it from `min` and `max`, and uses the
#'   [default float type][default_dtypes] where both are R values.
#' @section Random generation:
#' `nv_runif` samples from the open interval \eqn{(a, b)}.
#'
#' `min` and `max` are [`arrayish`], so they may vary across the sample: they
#' are applied to the draws after they have been reshaped to `shape`, and so
#' may either be scalars or have exactly that shape. As in base R's `runif()`,
#' an element whose `min` equals its `max` is that value, and one whose `min` or
#' `max` is not finite, or whose `max` is less than its `min`, is `NaN`. The RNG
#' state is advanced regardless.
#' @family rng
#' @export
nv_runif <- jit(
  function(shape, min = 0, max = 1, dtype = NULL) {
    shape <- assert_shapevec(shape)
    args <- promote_sample_params(shape, dtype, min = min, max = max)
    min <- args$min
    max <- args$max
    dtype <- dtype(min)

    # generate samples in [0, 1)
    U <- nv_unif_rand(shape = shape, dtype = dtype)

    # check if some values are <= 0
    le_zero <- nv_le(U, 0L)

    # Define smallest step (like R's 0.5 * i2_32m1 philosophy)
    # for f32 and 23 mantissa bits 2^-24 lies between 0 and 2^-23,
    # the next smallest generated value.
    # Same applies for f64 and 2^-53 and 52 mantissa bits.
    smallest_step <- nv_fill_like(
      U,
      ifelse(dtype == "f32", 2^-24, 2^-53),
      shape = shape,
      dtype = dtype
    )

    # Replace values <= 0 with smallest_step
    U <- nv_ifelse(le_zero, smallest_step, U)

    # expand to range and shift to interval (consistent if `min == max`)
    Y <- U * (max - min) + min

    # a reversed or non-finite interval is NaN to match base R
    valid <- nv_is_finite(min) & nv_is_finite(max) & (max >= min)

    nv_ifelse(valid, Y, NaN)
  },
  static = c("shape", "dtype")
)

# Bring the two parameters of a sampler (`min`/`max`, `mean`/`sd`), passed
# named in `...`, to the float data type the sample is drawn at: `dtype`, or
# their common data type when it is `NULL`. Each must be a scalar or have the
# sample's `shape`. Returns the promoted parameters as a named list.
promote_sample_params <- function(shape, dtype, ...) {
  rule <- if (is.null(dtype)) {
    promotion_common(fallback = default_float())
  } else {
    promotion_dtype(assert_rng_float_dtype(dtype, arg = "dtype"))
  }
  args <- as_anvl_arrays(..., .promote = rule)
  assert_rng_float_dtype(
    dtype(args[[1L]]),
    arg = paste(names(args), collapse = "/"),
    hint = "Pass {.arg dtype} to say what data type the sample should be drawn at."
  )
  for (name in names(args)) {
    assert_sample_param_shape(args[[name]], shape, arg = name)
  }
  args
}

# Error unless the sampler parameter `x` is a scalar or has the sample's shape.
assert_sample_param_shape <- function(x, shape, arg = rlang::caller_arg(x)) {
  x_shape <- as.integer(shape(x))
  if (length(x_shape) > 0L && !identical(x_shape, shape)) {
    cli_abort(c(
      "{.arg {arg}} must be a scalar or have the shape of the sample.",
      x = "Got shape {shapes_repr(list(x_shape))}, but the sample has shape {shapes_repr(list(shape))}."
    ))
  }
  invisible(x)
}

# Box-Muller transform: map two batches of uniforms in [0, 1) to twice as many
# standard normals. The radius uses 1 - u1, which lies in (0, 1], so a uniform
# of exactly 0 gives a finite sample.
box_muller <- function(u1, u2) {
  radius <- nv_sqrt(nv_log(1L - u1) * -2L)
  theta <- u2 * (2 * pi)
  nv_concatenate(radius * nv_sin(theta), radius * nv_cos(theta), axis = 1L)
}

#' @rdname nv_normal
#' @template param_shape
#' @param dtype (`NULL` | `character(1)` | [`DataType`][xlamisc::DataType])\cr
#'   Floating point data type of the sample. The default (`NULL`) uses the
#'   common data type of `mean` and `sd`, and the
#'   [default float type][default_dtypes] when both are R values.
#' @section Random generation:
#' `nv_rnorm` samples via the Box-Muller transform. To sample with a covariance
#' structure, use a Cholesky decomposition.
#'
#' `mean` and `sd` are [`arrayish`], so they may vary across the sample: they
#' are applied to the draws after they have been reshaped to `shape`, and so
#' may either be scalars or have exactly that shape. As in base R's `rnorm()`,
#' an element whose `mean` is `NaN`, or whose `sd` is not finite or negative, is
#' `NaN`. The RNG state is advanced regardless.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' with_nv_seed(42L, nv_rnorm(c(2, 3)))
#'
#' # `sd` may also be an array of the same shape as the sample
#' sds <- nv_array(matrix(c(0.01, 0.1, 1, 10, 100, 1000), nrow = 2))
#' nv_rnorm(c(2, 3), sd = sds)
#' @export
nv_rnorm <- jit(
  function(shape, mean = 0, sd = 1, dtype = NULL) {
    shape <- assert_shapevec(shape)
    args <- promote_sample_params(shape, dtype, mean = mean, sd = sd)
    mean <- args$mean
    sd <- args$sd
    dtype <- dtype(mean)

    # n: amount of rvs needed
    n <- prod(shape)

    m <- as.integer(ceiling(n / 2L))
    U1 <- nv_unif_rand(dtype = dtype, shape = m)
    U2 <- nv_unif_rand(dtype = dtype, shape = m)
    Z <- box_muller(U1, U2)

    # if n is uneven, only keep Z(1,...,n), i.e. discard last entry of Z
    if (n %% 2L == 1L) {
      Z <- nv_static_slice(Z, start_indices = 1L, end_indices = n, strides = 1L)
    }

    # reshape Z to match requested shape
    Z <- nv_reshape(Z, shape = shape)

    # Scale and shift the standard normals. This happens after the reshape so
    # that an arrayish `mean`/`sd` broadcasts against `shape` and not against the
    # flat buffer of ceil(n/2) * 2 draws.
    # was:    mean(Z) = 0, var(Z) = 1
    # now:    mean(N) = mean, var(N) = sd^2
    N <- Z * sd + mean

    # an invalid parameter is NaN to match base R
    valid <- !nv_is_nan(mean) & nv_is_finite(sd) & (sd >= 0L)
    nv_ifelse(valid, N, NaN)
  },
  static = c("shape", "dtype")
)

#' @title Sample from a Binomial Distribution
#' @description
#' Samples from a binomial distribution with \eqn{n} trials and success probability \eqn{p}.
#' When `size = 1` (the default), this is a Bernoulli distribution.
#' @template param_shape
#' @param size (`integer(1)`)\cr
#'   Number of trials. `0` gives a sample of zeros, like `rbinom()`.
#' @param prob (`numeric(1)`)\cr
#'   Probability of success on each trial.
#' @param dtype (`NULL` | `character(1)` | [`DataType`][xlamisc::DataType])\cr
#'   Numeric type of the sample.
#'   `NULL` (default) uses the [default integer type][default_dtypes].
#'   The number of successes are converted to it, so it must hold `size`.
#' @return ([`arrayish`])\cr
#'   The sample, of shape `shape` and data type `dtype`.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' # Bernoulli samples
#' with_nv_seed(42L, nv_rbinom(c(2, 3)))
#' @export
nv_rbinom <- jit(
  function(shape, size = 1L, prob = 0.5, dtype = NULL) {
    # The sample counts successes, which `bool` cannot hold.
    dtype <- assert_numeric_dtype(
      dtype %||% default_int(),
      arg = "dtype",
      hint = "A boolean cannot hold a count; use an integer data type and compare it."
    )
    checkmate::assert_int(size, lower = 0L)
    checkmate::assert_number(prob, lower = 0, upper = 1)
    shape <- assert_shapevec(shape)

    n_samples <- prod(shape)
    n_trials <- n_samples * size

    # Generate uniform samples in [0, 1) and compare to prob
    # `nv_runif()` would sample (0, 1), but only by moving a 0 to the smallest
    # step, so it is no less biased than [0, 1) here.
    U <- nv_unif_rand(shape = n_trials, dtype = "f64")

    # Success if U < prob
    successes <- nv_convert(nv_lt(U, prob), dtype = dtype)

    result <- if (size == 1L) {
      nv_reshape(successes, shape = shape)
    } else {
      successes <- nv_reshape(successes, shape = c(size, shape))
      nv_sum(successes, axes = 1L, drop = TRUE)
    }

    result
  },
  static = c("shape", "size", "prob", "dtype")
)

#' @title Sample Integers
#' @description
#' Samples integers from `1` to `n` with equal probability and with
#' replacement, analogous to R's `sample.int()`.
#'
#' To sample from a population other than `1:n`, use [nv_sample()].
#' @template param_shape
#' @param n (`integer(1)`)\cr
#'   Size of the population, i.e. the integers `1` to `n` are sampled.
#' @param dtype (`NULL` | `character(1)` | [`DataType`][xlamisc::DataType])\cr
#'   Numeric type of the sampled integers.
#'   The sampled values are converted to it, so it must hold `n`.
#'   `NULL` (default) uses the [default integer type][default_dtypes].
#' @return ([`arrayish`])\cr
#'   The sampled integers, of shape `shape` and data type `dtype`.
#' @family rng
#' @seealso [nv_sample()] to sample from an arbitrary population.
#' @examplesIf pjrt::plugins_downloaded()
#' # roll six dice
#' with_nv_seed(42L, nv_sample_int(6, 6L))
#' @export
nv_sample_int <- jit(
  function(shape, n, dtype = NULL) {
    # An index is a count too: at `bool` every draw would collapse to `TRUE`.
    dtype <- assert_numeric_dtype(
      dtype %||% default_int(),
      arg = "dtype",
      hint = "A boolean cannot hold an index; use an integer data type."
    )
    assert_int(n, lower = 1L)
    shape <- assert_shapevec(shape)

    idx <- sample_indices(as.integer(n), prod(shape))
    nv_reshape(nv_convert(idx, dtype), shape)
  },
  static = c("shape", "n", "dtype")
)

#' @title Sample from a Population
#' @description
#' Samples elements of a 1-D array with equal probability and with
#' replacement, analogous to R's `sample()`.
#'
#' Unlike R's `sample()`, `x` is always the population itself: sampling the
#' integers `1` to `n` is [nv_sample_int()] and never an overload of `x`.
#' @template param_shape
#' @param x ([`arrayish`])\cr
#'   The population vector to sample from. Must not be empty.
#'   An R value materializes at its [default data type][default_dtypes].
#' @return ([`arrayish`])\cr
#'   The sample, of shape `shape` and `x`'s data type.
#' @family rng
#' @seealso [nv_sample_int()] to sample the integers `1` to `n`.
#' @examplesIf pjrt::plugins_downloaded()
#' # the sample takes the population's data type
#' with_nv_seed(42L, nv_sample(5, nv_array(c(10, 20, 30))))
#' @export
nv_sample <- jit(
  function(shape, x) {
    shape <- assert_shapevec(shape)
    x <- as_anvl_array(x)
    x_shape <- shape(x)
    if (length(x_shape) != 1L) {
      cli_abort("{.arg x} must be a 1-D array, but has {length(x_shape)} axes.")
    }
    n <- x_shape[1L]
    if (n == 0L) {
      cli_abort("{.arg x} must not be empty: there is nothing to sample from.")
    }

    idx <- sample_indices(n, prod(shape))
    nv_reshape(nv_subset(x, idx), shape)
  },
  static = "shape"
)

# Draw `n_sample` uniformly distributed indices into a population of size `n`,
# with replacement, from the active random state.
sample_indices <- function(n, n_sample) {
  # use f64 for higher precision
  U <- nv_unif_rand(shape = n_sample, dtype = "f64")
  # u is in [0, 1), so floor(u * n) is in 0, ..., n - 1. The minimum guards
  # against the product rounding up to n for the largest representable u.
  idx <- nv_convert(nv_floor(nv_mul(U, n)), dtype = "i32")
  nv_pmin(nv_add(idx, 1L), as.integer(n))
}
