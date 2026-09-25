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
roxy_wrap <- function(x, width = 74L) {
  paste(strwrap(x, width = width), collapse = "\n")
}

# Titles of the pkgdown articles in `vignettes/articles/`, by file name.
# test-roxygen.R checks that they match the articles' own titles.
article_titles <- c(
  anvl = "Get Started",
  autodiff = "Automatic Differentiation",
  efficiency = "Efficiency",
  extending_api = "Extending the API",
  extending_primitive = "Adding a Primitive",
  faq = "FAQ",
  `gaussian-process` = "Gaussian Process",
  gotchas = "Gotchas",
  installation = "Installation",
  internals = "Internals",
  jit = "JIT Deep Dive",
  `metropolis-hastings` = "Metropolis-Hastings",
  next_steps = "Next Steps",
  primitives = "Primitives Reference",
  `random-numbers` = "Random Number Generation",
  static_shapes = "Static Shape Restriction",
  subsetting = "Subsetting",
  `type-promotion` = "Data Types and Promotion Rules"
)

# URL of a pkgdown article, e.g. `article_url("subsetting")`. The articles are
# not installed with the package, so documentation and messages link to the
# website instead of calling `vignette()`.
article_url <- function(name) {
  if (!name %in% names(article_titles)) {
    cli_abort("Unknown article {.val {name}}.")
  }
  sprintf("https://r-xla.github.io/anvl/articles/%s.html", name)
}

# A link to a pkgdown article, reading "the <title> article", for inline use in
# roxygen blocks as `r roxy_article("subsetting")`.
roxy_article <- function(name) {
  sprintf("the [%s](%s) article", article_titles[[name]], article_url(name))
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
  roxy_wrap(paste0(
    listed,
    " must have the same data type. An R value among them assumes the data type",
    " of the others when it is in its [data type category][dtypes], and its",
    " [default data type][default_dtypes] when none of them has one."
  ))
}

roxy_spec_chlo <- function(op) {
  roxy_wrap(sprintf(
    paste(
      "Lowers to [hlo_%1$s()], an op of the CHLO dialect, a higher-level",
      "companion to StableHLO that is lowered to it during compilation.",
      "See [chlo.%1$s](https://openxla.org/stablehlo/generated/chlo#chlo%1$s_chlo%2$sop)."
    ),
    op,
    gsub("_", "", op, fixed = TRUE)
  ))
}

roxy_spec <- function(op) {
  roxy_wrap(sprintf(
    "Lowers to [hlo_%1$s()], specified under [%1$s](https://openxla.org/stablehlo/spec#%1$s).",
    op
  ))
}
