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
#' @template param_dtype
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
  dtype = "f32",
  min = 0,
  max = 1
) {
  dtype <- assert_float_dtype(dtype)
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
#' @template param_dtype
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
nv_rnorm <- function(shape, initial_state, dtype = "f32", mean = 0, sd = 1) {
  dtype <- assert_float_dtype(dtype)
  shape <- assert_shapevec(shape)
  # `mean` and `sd` are arrayish: they may be traced values, so they cannot be
  # validated here and are only required to broadcast against `shape`.
  args <- as_anvl_arrays(mean, sd)
  mean <- nv_convert(args[[1L]], dtype)
  sd <- nv_convert(args[[2L]], dtype)
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
#' @template param_dtype
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
nv_rbinom <- function(shape, initial_state, size = 1L, prob = 0.5, dtype = "i32") {
  dtype <- as_dtype(dtype)
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

#' @title Random Samples
#' @description
#' Takes a sample from a population, analogous to R's
#' `sample(x, size, replace, prob)`.
#'
#' As in R, `x` is either the population itself or a single number `n`, in
#' which case the population is the integers `1` to `n`.
#' @template param_shape
#' @template param_initial_state
#' @param x (`numeric(1)` | [`arrayish`])\cr
#'   The population to sample from. A single plain number `n` samples the
#'   integers `1` to `n`; a 1-D array samples its elements.
#' @param replace (`logical(1)`)\cr
#'   Whether to sample with replacement. As in R, defaults to `FALSE`, which
#'   requires `prod(shape)` to be at most the population size.
#' @param probs ([`arrayish`] | `NULL`)\cr
#'   Sampling weights for the population, of the same length. They need not
#'   sum to one; they are normalised internally. If `NULL` (default), all
#'   elements are equally likely.
#' @param dtype (`character(1)` | [`DataType`])\cr
#'   Data type of the sampled integers. Only applies when `x` is a single
#'   number; when sampling from an array the result has the data type of `x`.
#' @return (`list()` of [`arrayish`])\cr
#'   List of two elements: the updated RNG state and the sampled values,
#'   of shape `shape`.
#' @details
#' With `replace = TRUE`, samples are drawn by inverting the cumulative
#' distribution of `probs`.
#'
#' With `replace = FALSE`, samples are drawn via the Gumbel top-`k` trick:
#' each element `i` gets the key \eqn{\log p_i + G_i} with \eqn{G_i} standard
#' Gumbel, and the `prod(shape)` largest keys are taken. This is equivalent to
#' R's sequential scheme, in which each successive element is drawn with
#' probability proportional to its weight among those not yet drawn.
#'
#' Unlike the other RNG functions, `nv_sample` is not itself jit-compiled,
#' because `x` may be either a compile-time count or a traced array, and a
#' static argument cannot be an array. It composes inside [jit()] as usual.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' state <- nv_rng_state(42L)
#' # Roll 6 dice
#' result <- nv_sample(6, state, 6L, replace = TRUE)
#' result[[2]]
#'
#' # A permutation of 1:6
#' nv_sample(6, state, 6L)[[2]]
#'
#' # Sample from a specific array, with weights
#' pop <- nv_array(c(10, 20, 30))
#' nv_sample(5, state, pop, replace = TRUE, probs = nv_array(c(1, 1, 8)))[[2]]
#' @export
nv_sample <- function(shape, initial_state, x, replace = FALSE, probs = NULL, dtype = "i32") {
  assert_flag(replace)
  shape <- assert_shapevec(shape)
  n_sample <- prod(shape)

  # `x` is either a plain count `n` (population 1:n) or the population itself.
  # A traced array is never `is.numeric()`, so this also picks the right branch
  # inside `jit()`.
  if (is.numeric(x) && length(x) == 1L && is.null(dim(x))) {
    assert_int(x, lower = 1)
    n <- as.integer(x)
    population <- NULL
    dtype <- as_dtype(dtype)
  } else {
    if (!missing(dtype)) {
      cli_abort(c(
        "{.arg dtype} only applies when {.arg x} is a single number.",
        i = "When sampling from an array, the result has the data type of {.arg x}."
      ))
    }
    if (is.numeric(x) && is.null(dim(x))) {
      cli_abort(c(
        "Vectors of length > 1 are not allowed as a population.",
        i = "Use {.code array()} to give the population a shape, e.g. {.code array(c(1, 3))}."
      ))
    }
    population <- as_anvl_array(x)
    nd <- naxes_abstract(population)
    if (nd != 1L) {
      cli_abort("{.arg x} must be a single number or a 1-D array, but got a {nd}-D array.")
    }
    n <- shape_abstract(population)[1L]
  }

  if (!replace && n_sample > n) {
    cli_abort(c(
      "Cannot take a sample larger than the population when {.code replace = FALSE}.",
      i = "Requested {n_sample} value{?s} from a population of size {n}."
    ))
  }

  if (!is.null(probs)) {
    probs <- as_anvl_array(probs)
    nd <- naxes_abstract(probs)
    if (nd != 1L || shape_abstract(probs)[1L] != n) {
      cli_abort(
        "{.arg probs} must be a 1-D array of length {n}, matching the population size."
      )
    }
    # use f64 throughout for higher precision
    probs <- nv_convert(probs, "f64")
  }

  if (replace) {
    out <- sample_with_replacement(initial_state, n, n_sample, probs)
  } else {
    out <- sample_without_replacement(initial_state, n, n_sample, probs)
  }
  state <- out[[1L]]
  # 1-based i32 indices into the population, of length n_sample
  idx <- out[[2L]]

  values <- if (is.null(population)) {
    nv_convert(idx, dtype)
  } else {
    nv_subset(population, idx)
  }

  list(state, nv_reshape(values, shape))
}

# Draw `n_sample` 1-based indices into a population of size `n`, with
# replacement, by inverting the cumulative distribution of `probs`.
sample_with_replacement <- function(initial_state, n, n_sample, probs) {
  # use f64 for higher precision
  res <- nv_unif_rand(initial_state, shape = n_sample, dtype = "f64")
  u <- res[[2L]]

  # Cumulative probabilities, normalised so that the final entry is exactly 1.
  # Dividing by the last cumulative sum rather than by the total keeps that
  # exact, so `u < 1` can never select an index past the end.
  cp <- if (is.null(probs)) {
    nv_div(
      nv_iota_like(initial_state, axis = 1L, shape = n, dtype = "f64"),
      nv_fill_like(initial_state, n, shape = integer(), dtype = "f64")
    )
  } else {
    cs <- nv_cumsum(probs, axis = 1L)
    nv_div(cs, nv_subset(cs, n))
  }

  # index i is chosen iff cp[i - 1] <= u < cp[i], i.e. i = 1 + #{j : cp[j] <= u}
  u_col <- nv_reshape(u, c(n_sample, 1L))
  cp_row <- nv_reshape(cp, c(1L, n))
  bc <- nv_broadcast_arrays(u_col, cp_row) # (n_sample, n)
  le_matrix <- nv_convert(nv_le(bc[[2L]], bc[[1L]]), dtype = "i32")
  idx <- nv_add(nv_reduce_sum(le_matrix, axes = 2L), 1L)

  list(res[[1L]], idx)
}

# Draw `n_sample` distinct 1-based indices into a population of size `n` via
# the Gumbel top-k trick: the top k of `log(p_i) + Gumbel_i` is distributed
# exactly like R's sequential weighted sampling without replacement.
sample_without_replacement <- function(initial_state, n, n_sample, probs) {
  # one uniform per population element; f64 for higher precision
  res <- nv_unif_rand(initial_state, shape = n, dtype = "f64")
  # `nv_unif_rand` draws from [0, 1), so clamp away from 0 before taking logs
  u <- nv_max(res[[2L]], 2^-53)

  # G = -log(-log(u)) is standard Gumbel
  keys <- nv_negate(nv_log(nv_negate(nv_log(u))))
  if (!is.null(probs)) {
    # zero-weight elements get key -Inf and so are never drawn
    keys <- nv_add(nv_log(probs), keys)
  }

  idx <- nv_top_k(keys, k = n_sample, with_indices = TRUE)$indices

  list(res[[1L]], idx)
}
