#' @title Generate RNG State
#' @name nv_rng_state
#' @description
#' Creates an initial RNG state from a seed. This state is required by all
#' random sampling functions and is updated after each call.
#' @param seed ([`arrayish`])\cr
#'   Scalar `i32` seed value.
#' @template param_device
#' @return ([`arrayish`])\cr
#'   Has `ui64` data type and shape `(2)`.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' # the state is a 1-D `ui64` array the samplers thread through
#' state <- nv_rng_state(42L)
#' state
#' @export
nv_rng_state <- function(seed, device = NULL) {
  seed <- nv_array(seed, dtype = as_dtype("i32"), shape = integer(), device = device)
  state <- nv_bitcast_convert(seed, dtype = "ui16")
  nv_convert(state, "ui64")
}
