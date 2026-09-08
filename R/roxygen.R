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
  args <- c(...)
  # Argument names are shown as code; a phrase such as "All inputs" is not.
  is_name <- grepl("^([A-Za-z._][A-Za-z0-9._]*|[.]{3})$", args)
  args[is_name] <- paste0("`", args[is_name], "`")
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

# The CHLO op a primitive lowers to, for inline use in roxygen blocks as
# `r roxy_spec_chlo("erf")`. CHLO ops are not in the StableHLO specification --
# they are lowered to StableHLO during compilation -- so they link the generated
# CHLO reference instead, whose anchors stablehlo's own `op_chlo` template
# builds the same way.
roxy_spec_chlo <- function(op) {
  sprintf(
    paste(
      "Lowers to [hlo_%1$s()], an op of the CHLO dialect, a higher-level",
      "companion to StableHLO that is lowered to it during compilation.",
      "See [chlo.%1$s](https://openxla.org/stablehlo/generated/chlo#chlo%1$s_chlo%1$sop)."
    ),
    op
  )
}

# The StableHLO op a primitive lowers to, with a link to its entry in the
# specification, for inline use in roxygen blocks as `r roxy_spec("gather")`.
# The op name is both the `hlo_*` function's suffix and the spec's anchor.
roxy_spec <- function(op) {
  sprintf(
    "Lowers to [hlo_%1$s()], specified under [%1$s](https://openxla.org/stablehlo/spec#%1$s).",
    op
  )
}
