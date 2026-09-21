#' @title Generate RNG State
#' @name nv_rng_state
#' @description
#' Creates an initial RNG state from a seed. This state is required by all
#' random sampling functions and is updated after each call.
#' @param seed ([`arrayish`])\cr
#'   Scalar integer value.
#' @template param_device
#' @return ([`arrayish`])\cr
#'   Has type `ui64[2]`.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' state <- nv_rng_state(42L)
#' state
#' @export
nv_rng_state <- function(seed, device = NULL) {
  dt <- peek_dtype(seed)
  if (dtype_category(dt) != 2L) {
    what <- if (is_anvl_array(seed)) {
      dtype(what)
    } else {
      class(seed)[1L]
    }
    cli_abort(c(
      "Input seed must be an (un)signed integer.",
      x = "Got {what} instead."
    ))
  }
  # REVIEW: Need .promote for as_anvl_array()
  # Also, they should get argument `device`, then we can use it here and jit() the whole function.
  seed <- nv_array(seed, device = device, shape = integer())
  state <- nv_convert(seed, "i32")
  state <- nv_bitcast_convert(seed, dtype = "ui16")
  nv_convert(state, "ui64")
}
