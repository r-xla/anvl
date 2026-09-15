# Expectations for the two things nearly every test asserts about a value: its
# data type and its shape. `expect_equal(dtype(x), as_dtype("f64"))` states the
# same thing, but on failure it can only show two `DataType` objects -- these
# name the expression that produced the wrong answer.

#' Assert the data type of an arrayish value.
#'
#' `expect_dtype(nv_scalar(1L), "i32")`.
#'
#' `expected` is anything [as_dtype()] accepts: a string, or a `DataType` such
#' as the one [default_int()] returns. `object` goes through [dtype()], so a
#' bare R value is an error here exactly as it is everywhere else -- assert what
#' such a value *would* commit to with [peek_dtype()] instead.
#' @noRd
expect_dtype <- function(object, expected, info = NULL, label = NULL) {
  act <- testthat::quasi_label(rlang::enquo(object), label, arg = "object")
  exp_dtype <- as_dtype(expected)
  act_dtype <- dtype(act$val)
  testthat::expect(
    act_dtype == exp_dtype,
    sprintf(
      "dtype(%s) is %s, not %s.",
      act$lab,
      encodeString(as.character(act_dtype), quote = '"'),
      encodeString(as.character(exp_dtype), quote = '"')
    ),
    info = info
  )
  invisible(act$val)
}

#' Assert the shape of an arrayish value.
#'
#' `expect_shape(nv_array(array(1:6, c(2, 3))), c(2, 3))`.
#'
#' `expected` is a vector of axis sizes -- `integer()` for a scalar. It is
#' compared as integer, so `c(2, 3)` and `c(2L, 3L)` both work.
#' @noRd
expect_shape <- function(object, expected, info = NULL, label = NULL) {
  act <- testthat::quasi_label(rlang::enquo(object), label, arg = "object")
  exp_shape <- as.integer(expected)
  act_shape <- as.integer(shape(act$val))
  testthat::expect(
    identical(act_shape, exp_shape),
    sprintf("shape(%s) is %s, not %s.", act$lab, shape_repr(act_shape), shape_repr(exp_shape)),
    info = info
  )
  invisible(act$val)
}
