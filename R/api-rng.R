nv_unif_rand <- function(
  shape,
  initial_state,
  dtype = "f64"
) {
  dtype <- assert_float_dtype(dtype)
  shape <- assert_shapevec(shape)

  # 1. Generate random bits (64)
  # 2. We use these as mantissa bits for float, where we set the exponent to 1.0
  # 3. Because we have an implicit leading 1, we get a number in [1, 2) -> need to shift to [0, 1)

  ui_dtype <- as_dtype(paste0("ui", dtype_width(dtype)))

  # generate random bits
  # use THREE_FRY as rng algorithm: JAX default
  rbits <- prim_rng_bit_generator(
    initial_state = initial_state,
    "THREE_FRY",
    ui_dtype,
    shape = shape
  )

  # shift value: 9 for f32, 11 for f64
  shift <- if (dtype == "f32") 9L else 11L

  # shift to the right, s.t. exponent bits are all 0
  mantissa <- nv_shift_right_logical(rbits[[2]], shift)

  one_bits <- nv_bitcast_convert(
    nv_fill_like(initial_state, 1.0, shape = integer(), dtype = dtype),
    dtype = ui_dtype
  )

  # bitwise or -> exponent from 1.0 (float), mantissa is random
  U <- nv_or(mantissa, one_bits)

  # convert back to requested dtype
  # resulting RVs  are in [1, 2)
  U <- nv_bitcast_convert(U, dtype = dtype)

  # shift to [0, 1)
  U <- U - 1

  # return state and RVs
  list(rbits[[1]], U)
}

# Random Number Generation API
# This file contains user-facing RNG sampling functions

#' @title Sample from a Uniform Distribution
#' @description
#' Samples from a uniform distribution in the open interval `(min, max)`.
#' @template param_shape
#' @template param_initial_state
#' @param dtype (`NULL` | `character(1)` | [`DataType`])\cr
#'   Data type of the sampled values. `NULL` (default) uses the backend's
#'   default float data type (see [`default_dtypes()`]).
#' @param min,max (`numeric(1)`)\cr
#'   Lower and upper bound.
#' @return (`list()` of [`arrayish`])\cr
#'   List of two elements: the updated RNG state and the sampled values.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' state <- nv_rng_state(42L)
#' result <- nv_runif(c(2, 3), state)
#' result[[2]]
#' @export
#' @jit static c(1L, 3L, 4L, 5L)
nv_runif <- function(
  shape,
  initial_state,
  dtype = NULL,
  min = 0,
  max = 1
) {
  dtype <- assert_float_dtype(dtype %||% default_dtype_r("double"))
  checkmate::assertNumeric(min, len = 1, any.missing = FALSE, upper = max)
  checkmate::assertNumeric(max, len = 1, any.missing = FALSE, lower = min)
  shape <- assert_shapevec(shape)

  if (max == min) {
    return(nv_fill_like(initial_state, max, shape = shape, dtype = dtype))
  }

  .range <- max - min

  # generate samples in [0, 1)
  Unif <- nv_unif_rand(initial_state = initial_state, shape = shape, dtype = dtype)
  U <- Unif[[2]]

  # check if some values are <= 0
  le_zero <- nv_le(U, 0)

  # Define smallest step (like R's 0.5 * i2_32m1 philosophy)
  # for f32 and 23 mantissa bits 2^-24 lies between 0 and 2^-23,
  # the next smallest generated value.
  # Same applies for f64 and 2^-53 and 52 mantissa bits.
  smallest_step <- nv_fill_like(
    initial_state,
    ifelse(dtype == "f32", 2^-24, 2^-53),
    shape = shape,
    dtype = dtype
  )

  # Replace values <= 0 with smallest_step
  U <- nv_ifelse(le_zero, smallest_step, U)

  # expand to range
  U <- nv_mul(U, .range)
  # shift to interval
  Y <- U + min

  return(list(Unif[[1]], Y))
}

#' @rdname nv_normal
#' @template param_shape
#' @template param_initial_state
#' @param dtype (`NULL` | `character(1)` | [`DataType`][tengen::DataType])\cr
#'   Data type of the sample, `"f32"` or `"f64"`. `NULL` (default) takes it from
#'   `mean` and `sd` where either is a real array, and falls back to the default
#'   float data type (see [`default_dtypes()`]) where both are bare R values,
#'   which have none.
#' @section Random generation:
#' `nv_rnorm` samples via the Box-Muller transform. To sample with a covariance
#' structure, use a Cholesky decomposition.
#'
#' `mean` and `sd` are [`arrayish`], so they may vary across the sample: they
#' are applied to the draws after they have been reshaped to `shape`, and so
#' may either be scalars or have exactly that shape.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' state <- nv_rng_state(42L)
#' result <- nv_rnorm(c(2, 3), state)
#' result[[2]]
#'
#' # `sd` may also be an array of the same shape as the sample
#' sds <- nv_array(matrix(c(0.01, 0.1, 1, 10, 100, 1000), nrow = 2))
#' nv_rnorm(c(2, 3), state, sd = sds)[[2]]
#' @export
#' @jit static c(1L, 3L)
nv_rnorm <- function(shape, initial_state, dtype = NULL, mean = 0, sd = 1) {
  shape <- assert_shapevec(shape)

  rule <- if (is.null(dtype)) {
    promote_common(fallback = default_dtype_r("double"))
  } else {
    promote_dtype(assert_float_dtype(dtype))
  }
  args <- as_anvl_arrays(mean = mean, sd = sd, .promote = rule)
  mean <- args$mean
  sd <- args$sd
  dtype <- assert_float_dtype(
    dtype(mean),
    arg = "mean/sd",
    hint = "Pass {.arg dtype} to say what data type the sample should be drawn at."
  )
  # n: amount of rvs needed
  n <- prod(shape)

  # Box-Muller Method:
  # from two random uniform variables u1 and u2 we can produce to normals z1, z2
  # z1 = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
  # z2 = sqrt(-2 * log(u1)) * sin(2 * pi * u2)
  # Box-Muller works via polar representation of coordinates.
  # We scale this approach and genereate ceil(n/2) uniform rvs twice (U, Theta)

  # generate the first ceil(n/2) random uniform variables
  U <- nv_unif_rand(
    initial_state = initial_state,
    dtype = dtype,
    shape = as.integer(ceiling(n / 2))
  )

  # compute the radius R = sqrt(-2 * log(u1))
  R <- nv_mul(nv_log(U[[2]]), -2)
  sqrt_R <- nv_sqrt(R)

  # generate second batch of ceil(n/2) random uniform variables
  Theta <- nv_unif_rand(initial_state = U[[1]], dtype = dtype, shape = as.integer(ceiling(n / 2)))

  # compute cos(2 * pi * u2) / sin(2 * pi * u2)
  Theta[[2]] <- nv_mul(Theta[[2]], 2 * pi)
  sin_Theta <- nv_sin(Theta[[2]])
  cos_Theta <- nv_cos(Theta[[2]])

  # compute z1, z2
  Z1 <- nv_mul(sqrt_R, sin_Theta)
  Z2 <- nv_mul(sqrt_R, cos_Theta)

  # concatenate z = (z1, z2)
  Z <- nv_concatenate(Z1, Z2, axis = 1L)

  # if n is uneven, only keep Z(1,...,n), i.e. discard last entry of Z
  if (n %% 2 == 1) {
    Z <- nv_static_slice(Z, start_indices = 1L, limit_indices = n, strides = 1L)
  }

  # reshape Z to match requested shape
  Z <- nv_reshape(Z, shape = shape)

  # Scale and shift the standard normals. This happens after the reshape so
  # that an arrayish `mean`/`sd` broadcasts against `shape` and not against the
  # flat buffer of ceil(n/2) * 2 draws.
  # was:    mean(Z) = 0, var(Z) = 1
  # now:    mean(N) = mean, var(N) = sd^2
  N <- Z * sd + mean

  # return state and Normals N
  list(Theta[[1]], N)
}

#' @title Sample from a Binomial Distribution
#' @description
#' Samples from a binomial distribution with \eqn{n} trials and success probability \eqn{p}.
#' When `size = 1` (the default), this is a Bernoulli distribution.
#' @template param_shape
#' @template param_initial_state
#' @param size (`integer(1)`)\cr
#'   Number of trials.
#' @param prob (`numeric(1)`)\cr
#'   Probability of success on each trial.
#' @param dtype (`NULL` | `character(1)` | [`DataType`])\cr
#'   Data type of the sampled values. `NULL` (default) uses the backend's
#'   default integer data type (see [`default_dtypes()`]).
#' @return (`list()` of [`arrayish`])\cr
#'   List of two elements: the updated RNG state and the sampled values.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' state <- nv_rng_state(42L)
#' # Bernoulli samples
#' result <- nv_rbinom(c(2, 3), state)
#' result[[2]]
#' @export
#' @jit static c(1L, 3L, 4L, 5L)
nv_rbinom <- function(shape, initial_state, size = 1L, prob = 0.5, dtype = NULL) {
  dtype <- as_dtype(dtype %||% default_dtype_r("integer"))
  checkmate::assert_int(size, lower = 1)
  checkmate::assert_number(prob, lower = 0, upper = 1)
  shape <- assert_shapevec(shape)

  n_samples <- prod(shape)
  n_trials <- n_samples * size

  # Generate uniform samples in [0, 1) and compare to prob
  # Note that using runif() generates in (0, 1), but by shifting the 0 to the smallest value
  # so we don't benefit from using runif w.r.t. unbiasedness
  res <- nv_unif_rand(initial_state, shape = n_trials, dtype = "f64")
  U <- res[[2]]

  # Success if U < prob
  successes <- nv_convert(nv_lt(U, prob), dtype = dtype)

  result <- if (size == 1L) {
    nv_reshape(successes, shape = shape)
  } else {
    successes <- nv_reshape(successes, shape = c(size, shape))
    nv_reduce_sum(successes, axes = 1L, drop = TRUE)
  }

  list(res[[1]], result)
}

#' @title Sample Integers
#' @description
#' Samples integers from `1` to `n` with equal probability and with
#' replacement, analogous to R's `sample.int()`.
#'
#' To sample from a population other than `1:n`, use [nv_sample()].
#' @template param_shape
#' @template param_initial_state
#' @param n (`integer(1)`)\cr
#'   Size of the population, i.e. the integers `1` to `n` are sampled.
#' @param dtype (`NULL` | `character(1)` | [`DataType`])\cr
#'   Data type of the sampled integers. `NULL` (default) uses the backend's
#'   default integer data type (see [`default_dtypes()`]).
#' @return (`list()` of [`arrayish`])\cr
#'   List of two elements: the updated RNG state and the sampled integers,
#'   of shape `shape`.
#' @family rng
#' @seealso [nv_sample()] to sample from an arbitrary population.
#' @examplesIf pjrt::plugins_downloaded()
#' state <- nv_rng_state(42L)
#' # Roll 6 dice
#' result <- nv_sample_int(6, state, 6L)
#' result[[2]]
#' @export
#' @jit static c(1L, 3L, 4L)
nv_sample_int <- function(shape, initial_state, n, dtype = NULL) {
  dtype <- as_dtype(dtype %||% default_dtype_r("integer"))
  assert_int(n, lower = 1)
  shape <- assert_shapevec(shape)

  out <- sample_indices(initial_state, as.integer(n), prod(shape))

  list(out[[1L]], nv_reshape(nv_convert(out[[2L]], dtype), shape))
}

#' @title Sample from a Population
#' @description
#' Samples elements of a 1-D array with equal probability and with
#' replacement, analogous to R's `sample()`.
#'
#' Unlike R's `sample()`, `x` is always the population itself: sampling the
#' integers `1` to `n` is [nv_sample_int()] and never an overload of `x`.
#' @template param_shape
#' @template param_initial_state
#' @param x ([`arrayish`])\cr
#'   The population to sample from, a 1-D array.
#' @return (`list()` of [`arrayish`])\cr
#'   List of two elements: the updated RNG state and the sampled values, of
#'   shape `shape` and with the data type of `x`.
#' @family rng
#' @seealso [nv_sample_int()] to sample the integers `1` to `n`.
#' @examplesIf pjrt::plugins_downloaded()
#' state <- nv_rng_state(42L)
#' pop <- nv_array(c(10, 20, 30))
#' result <- nv_sample(5, state, pop)
#' result[[2]]
#' @export
#' @jit static 1L
nv_sample <- function(shape, initial_state, x) {
  shape <- assert_shapevec(shape)
  x <- as_anvl_array(x)
  x_shape <- shape(x)
  if (length(x_shape) != 1L) {
    cli_abort("{.arg x} must be a 1-D array, but has {length(x_shape)} axes.")
  }
  n <- x_shape[1L]

  out <- sample_indices(initial_state, n, prod(shape))

  list(out[[1L]], nv_reshape(nv_subset(x, out[[2L]]), shape))
}

# Draw `n_sample` uniformly distributed 1-based indices into a population of
# size `n`, with replacement. Returns the updated RNG state and the indices.
sample_indices <- function(initial_state, n, n_sample) {
  # use f64 for higher precision
  res <- nv_unif_rand(initial_state, shape = n_sample, dtype = "f64")
  # u is in [0, 1), so floor(u * n) is in 0, ..., n - 1. The minimum guards
  # against the product rounding up to n for the largest representable u.
  idx <- nv_convert(nv_floor(nv_mul(res[[2L]], n)), dtype = "i32")
  list(res[[1L]], nv_min(nv_add(idx, 1L), as.integer(n)))
}
