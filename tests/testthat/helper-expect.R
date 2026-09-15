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
