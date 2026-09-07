# Element types supported by anvl.
#
# Maintained here rather than derived from stablehlo's internal list, so anvl's
# documented dtypes are not tied to another package's internals and the two can
# diverge (e.g. when a backend gains a type anvl does not expose yet).
# Note that `"pred"` and `"i1"` are accepted as aliases for `"bool"`.
dtypes_supported <- c(
  "bool",
  "i8",
  "i16",
  "i32",
  "i64",
  "ui8",
  "ui16",
  "ui32",
  "ui64",
  "f32",
  "f64"
)

# Comma-separated list of the supported dtypes, for inline use in roxygen
# blocks as `r roxy_dtypes()`.
roxy_dtypes <- function() {
  paste0(dtypes_supported, collapse = ", ")
}

# The data type rule of a primitive whose operands must reach one data type, for
# inline use in roxygen blocks as `r roxy_agree("x", "update")`. Name every
# argument the primitive's `apply_promotion()` call covers, in the order the
# block documents them; the sentence goes on the primary operand's parameter.
roxy_agree <- function(...) {
  args <- paste0("`", c(...), "`")
  listed <- if (length(args) > 1L) {
    paste0(paste(args[-length(args)], collapse = ", "), " and ", args[[length(args)]])
  } else {
    args
  }
  paste0(
    listed,
    " must have the same data type. An R value among them assumes the data type",
    " of the others when it is in its [data type category][dtypes], and its",
    " [default data type][default_dtypes] when none of them has one."
  )
}
