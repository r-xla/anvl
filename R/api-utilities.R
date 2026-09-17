#' @title Generate RNG State
#' @name nv_rng_state
#' @description
#' Creates an initial RNG state from a seed. This state is required by all
#' random sampling functions and is updated after each call.
#' @param seed ([`arrayish`])\cr
#'   Scalar seed, either a plain R value or an `i32` array. It is built at
#'   `i32` whatever the default integer data type is, so a given seed names the
#'   same stream in every configuration -- which is also why an array of
#'   another data type is refused rather than converted.
#' @template param_device
#' @return ([`arrayish`])\cr
#'   Has the `ui64` data type and shape `(2)`, the layout the generator
#'   requires -- fixed, not taken from the default data types.
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
