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
