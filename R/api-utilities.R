#' @title Generate RNG State
#' @name nv_rng_state
#' @description
#' Creates an initial RNG state from a seed. This state is required by all
#' random sampling functions and is updated after each call.
#' @param seed ([`arrayish`])\cr
#'   Scalar seed. Must be a signed or unsigned integer; it is brought to `i32`,
#'   so a wider one is narrowed to its low 32 bits. An R integer is built at
#'   `i32` directly, whatever the default integer data type is.
#' @template param_device
#' @return ([`arrayish`])\cr
#'   A `ui64` array of length 2, whatever `seed`'s data type was.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' state <- nv_rng_state(42L)
#' state
#' @export
nv_rng_state <- function(seed, device = NULL) {
  if (dtype_category(peek_dtype(seed)) != 2L) {
    what <- if (is_rdata(to_abstract(seed))) class(seed)[[1L]] else as.character(dtype(seed))
    cli_abort(c(
      "{.arg seed} must be a signed or unsigned integer.",
      x = "Got {.val {what}} instead."
    ))
  }
  if (length(shape(seed)) != 0L) {
    cli_abort(c(
      "{.arg seed} must be a scalar.",
      x = "Got shape {shape_repr(shape(seed))}."
    ))
  }
  # The seed reaches `i32` whatever it arrived as, which is what fixes the state
  # at `ui64[2]`: the bitcast below reads those 32 bits as two `ui16`. The rule
  # builds an R value at `i32` directly rather than at the default integer and
  # converting, so the state does not depend on `default_dtypes()`.
  seed <- as_anvl_array(seed, device = device, .promote = promotion_dtype("i32", coerce = TRUE))
  state <- nv_bitcast_convert(seed, dtype = "ui16")
  nv_convert(state, "ui64")
}
