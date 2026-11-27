#' @title Call an XLA FFI Handler
#' @description
#' `r lifecycle::badge("experimental")`
#'
#' Calls a handler registered with [`pjrt::pjrt_register_custom_call()`],
#' the way [`prim_custom_call()`] does, but taking the operands through `...`
#' and unwrapping a single result.
#'
#' This is anvl's escape hatch for operations StableHLO cannot express, or
#' cannot express efficiently. How to write, register and call a handler is
#' shown in `r roxy_article("custom_calls")`; XLA documents custom calls, and
#' the FFI API a handler is written against, in its
#' [custom call guide](https://openxla.org/xla/custom_call).
#'
#' Custom calls are a `"pjrt"` backend feature: the handler is registered
#' with the PJRT runtime and invoked by the XLA-compiled program, so there
#' is nothing for the `quickr` backend to call. Under
#' [`local_backend("quickr")`][local_backend] a function containing a custom
#' call fails to compile.
#'
#' Within pjrt the call is platform-agnostic. It names a target, and XLA
#' invokes whichever handler [`pjrt::pjrt_register_custom_call()`] was given
#' for the platform the program is compiled for, so the same
#' `nv_custom_call()` runs on CPU or CUDA depending only on where its
#' operands live.
#'
#' The call is also opaque to XLA and therefore to [`gradient()`]: to
#' differentiate through a handler, wrap it in a primitive of your own with
#' a pullback rule -- see `r roxy_article("extending_primitive")`.
#' @param target_name (`character(1)`)\cr
#'   The name the handler was registered under.
#' @param ... ([`arrayish`])\cr
#'   The operands, in the order the handler binds them.
#' @inheritParams prim_custom_call
#' @return [`arrayish`] or `list` of [`arrayish`]\cr
#'   A single value when `output_types` is a single [`AbstractArray`] (or
#'   when a side-effect only call has a single operand), otherwise a list.
#' @seealso [prim_custom_call()]
#' @examplesIf pjrt::plugins_downloaded()
#' # "eigh" is one of the handlers {pjrt} registers on load. It returns
#' # (eigenvectors, eigenvalues) and, being LAPACK, wants column-major
#' # buffers -- R's own order, which is the default layout.
#' x <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
#' nv_custom_call(
#'   "eigh",
#'   x,
#'   output_types = list(AbstractArray("f64", c(2, 2)), AbstractArray("f64", 2))
#' )
#' @export
nv_custom_call <- function(
  target_name,
  ...,
  output_types = NULL,
  attrs = list(),
  has_side_effect = is.null(output_types),
  operand_layouts = NULL,
  result_layouts = NULL,
  aliases = NULL,
  device = NULL
) {
  # A bare AbstractArray declares one result, and a side-effect only call
  # returns its operands, so one operand comes back on its own too.
  single <- is_abstract_array(output_types) || (is.null(output_types) && ...length() == 1L)
  out <- prim_custom_call(
    list(...),
    target_name = target_name,
    output_types = output_types,
    attrs = attrs,
    has_side_effect = has_side_effect,
    operand_layouts = operand_layouts,
    result_layouts = result_layouts,
    aliases = aliases,
    device = device
  )
  if (single) out[[1L]] else out
}

# Helpers of `prim_custom_call()`.

# The attributes of a custom call as stablehlo attribute objects. A handler
# reads an attribute at a C++ type fixed when it was compiled, so a number or a
# logical takes a data type fixed by its R type, not the default data types: a
# double is `f64`, an integer `i64` and a logical `bool`. An attribute object
# passes through, so a caller who needs another data type can build the
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
      if (!is.numeric(value) && !is.logical(value)) {
        cli_abort("Attribute {.val {nm}} must be a string, a number or a logical, not {.obj_type_friendly {value}}.")
      }
      dt <- as_dtype(switch(typeof(value), double = "f64", integer = "i64", logical = "bool"))
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
