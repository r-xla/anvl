# Helpers of `prim_custom_call()`.

# The attributes of a custom call as stablehlo attribute objects. A number or a
# logical takes the default data type of its category, so this runs while
# tracing, where the default data types of the program are known. An attribute
# object passes through, so a caller who needs another data type can build the
# attribute itself.
custom_call_attrs <- function(attrs) {
  Map(
    function(nm, value) {
      if (inherits(value, "OpInputAttr")) {
        return(value)
      }
      if (!is.atomic(value) || !length(value) || anyNA(value)) {
        cli_abort(c(
          "Attribute {.val {nm}} must be a non-empty atomic vector without missing values.",
          i = "Alternatively pass a {.cls stablehlo::OpInputAttr} to choose the data type yourself."
        ))
      }
      if (is.character(value)) {
        if (length(value) != 1L) {
          cli_abort("String attribute {.val {nm}} must have length 1.")
        }
        return(stablehlo::StringAttr(name = nm, value = value))
      }
      dt <- default_dtype(value)
      if (length(value) == 1L) {
        if (is.logical(value)) {
          stablehlo::BoolAttr(name = nm, value = value)
        } else {
          stablehlo::ScalarAttr(name = nm, value = value, dtype = dt)
        }
      } else {
        stablehlo::constant_attr(name = nm, value = value, dtype = as.character(dt))
      }
    },
    names(attrs),
    attrs
  )
}

# Layouts are either one spec used everywhere (a list of integer vectors) or
# one spec per platform (a list of such lists, named "cpu" / "cuda").
is_per_platform_layouts <- function(x) {
  is.list(x) &&
    length(x) &&
    !is.null(names(x)) &&
    all(vapply(x, is.list, logical(1L)))
}

# Check the layouts `layouts` of values with `naxes` axes each: one entry per
# value, each a permutation of that value's axes. `what` names the values in
# the messages ("operand" or "result").
assert_custom_call_layouts <- function(layouts, naxes, arg, what) {
  if (is.null(layouts)) {
    return(invisible(NULL))
  }
  specs <- if (is_per_platform_layouts(layouts)) layouts else list(layouts)
  for (i in seq_along(specs)) {
    spec <- specs[[i]]
    where <- if (!is.null(names(specs))) cli::format_inline(" for platform {.val {names(specs)[[i]]}}") else ""
    if (!is.list(spec) || length(spec) != length(naxes)) {
      cli_abort("{.arg {arg}}{where} must have one entry per {what} ({length(naxes)}).")
    }
    is_perm <- vapply(
      seq_along(spec),
      \(j) {
        test_integerish(spec[[j]], any.missing = FALSE) && identical(sort(as.integer(spec[[j]])), seq_len(naxes[[j]]))
      },
      logical(1L)
    )
    if (!all(is_perm)) {
      j <- which(!is_perm)[[1L]]
      cli_abort(
        "Entry {j} of {.arg {arg}}{where} must order the {naxes[[j]]} ax{?is/es} of {what} {j}, as a permutation of {.code seq_len({naxes[[j]]})}."
      )
    }
  }
  invisible(NULL)
}

# The XLA layouts of the layouts `layouts` (R axes, minor to major) of values
# with `naxes` axes each. `NULL` is R's own column-major order, axis 1 varying
# fastest, which is XLA's default layout of the reversed shape. Per-platform
# layouts are resolved here, at lowering, the first point at which the platform
# of the program is known.
custom_call_layouts <- function(layouts, naxes, arg) {
  if (is_per_platform_layouts(layouts)) {
    platform <- current_platform()
    chosen <- if (!is.null(platform)) layouts[[platform]]
    if (is.null(chosen)) {
      cli_abort(c(
        "{.arg {arg}} has no entry for platform {.val {platform %||% 'unknown'}}.",
        i = "It lists {.val {names(layouts)}}."
      ))
    }
    layouts <- chosen
  }
  layouts <- layouts %||% lapply(naxes, seq_len)
  unname(Map(xla_dims, layouts, naxes))
}

# `aliases` is 1-based and one entry per result; NA means "no alias".
custom_call_aliases <- function(aliases, n_results) {
  idx <- which(!is.na(aliases))
  lapply(idx, function(i) {
    stablehlo::OutputOperandAlias(
      operand_index = aliases[[i]] - 1L,
      # a call with a single result is not tuple-typed
      output_tuple_indices = if (n_results > 1L) i - 1L else integer()
    )
  })
}
