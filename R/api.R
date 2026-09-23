# This is the user-facing API containing the exported array operations.
#' @include primitives.R

# Special array creators

#' @title Fill Constant
#' @description
#' Creates an array filled with a scalar value. More memory-efficient than
#' `nv_array(value, shape = shape)` for large arrays.
#'
#' `nv_fill_like()` is a variant where `dtype`, `shape`, and
#' `device` default to those of `like`.
#' @param value (`numeric(1)`)\cr
#'   Scalar value to fill the array with. It has to be something `dtype` can
#'   hold: a whole number in its range for an integer data type, a non-negative
#'   one for an unsigned integer, and a logical or `0` / `1` for `bool`.
#' @param shape (`integer()`)\cr
#'   Shape of the output array.
#' @param dtype (`NULL` | `character(1)` | [`DataType`][tengen::DataType])\cr
#'   Data type of the result.
#'   The default (`NULL`) uses the [default data type][default_dtypes] for `nv_fill` and
#'   `dtype(like)` for `nv_fill_like`.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_fill_like()`).
#' @template param_device
#' @return ([`arrayish`])\cr
#'   Has the given `shape` and `dtype`.
#' @seealso [prim_fill()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the R double settles the data type, the shape is given
#' nv_fill(0, shape = c(2, 3))
#'
#' # `_like` takes shape, data type and device from an existing array
#' x <- nv_matrix(1:6, nrow = 2)
#' nv_fill_like(x, 0)
#' @export
nv_fill <- function(value, shape, dtype = NULL, device = NULL) {
  if (!is_valid_r_lit(value)) {
    cli_abort(
      "{.arg value} must be an R vector of length 1 of type double, integer, or logical, not {.cls {class(value)[1]}}."
    )
  }
  dtype <- if (is.null(dtype)) {
    default_dtype(value)
  } else {
    as_dtype(dtype)
  }
  prim_fill(value, shape, dtype, device = device)
}

## Conversion ------------------------------------------------------------------

broadcast_shapes <- function(shape_lhs, shape_rhs) {
  # Kept for the message: the two are padded below, and reporting the padded
  # shapes back would name axes the caller never wrote.
  given_lhs <- shape_lhs
  given_rhs <- shape_rhs
  if (length(shape_lhs) > length(shape_rhs)) {
    shape_rhs <- c(shape_rhs, rep(1L, length(shape_lhs) - length(shape_rhs)))
  } else if (length(shape_lhs) < length(shape_rhs)) {
    shape_lhs <- c(shape_lhs, rep(1L, length(shape_rhs) - length(shape_lhs)))
  } else if (identical(shape_lhs, shape_rhs)) {
    return(shape_lhs)
  }
  shape_out <- shape_lhs
  for (i in seq_along(shape_lhs)) {
    d_lhs <- shape_lhs[i]
    d_rhs <- shape_rhs[i]
    if (d_lhs != d_rhs && d_lhs != 1L && d_rhs != 1L) {
      cli_abort(c(
        "Shapes {shape_repr(given_lhs)} and {shape_repr(given_rhs)} are not broadcastable.", # nolint
        x = "Sizes {d_lhs} and {d_rhs} meet, and neither is 1."
      ))
    }
    shape_out[i] <- max(d_lhs, d_rhs)
  }
  shape_out
}

#' @title Broadcast Scalars to Common Shape
#' @description
#' Broadcast scalar arrays to match the shape of non-scalar arrays.
#' All non-scalar arrays must have the same shape.
#' @param ... ([`arrayish`][arrayish])\cr
#'   Arrays to broadcast.
#' @return (`list()` of [`arrayish`])\cr
#'   The inputs, each with its own data type and the common shape.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' # scalar 1 is broadcast to shape [3]
#' nv_broadcast_scalars(x, nv_scalar(1))
#' @export
nv_broadcast_scalars <- jit(function(...) {
  assert_some_arrays(...)
  args <- as_anvl_arrays(...)
  shapes <- lapply(args, shape)
  non_scalar_shapes <- Filter(\(s) length(s) > 0L, shapes)

  if (length(non_scalar_shapes) == 0L) {
    return(args)
  }

  target_shape <- non_scalar_shapes[[1L]]
  if (!all(vapply(non_scalar_shapes, identical, logical(1L), target_shape))) {
    shapes <- shapes_repr(shapes)
    cli_abort(
      "All non-scalar arrays must have the same shape, but got {shapes}. Use {.fn nv_broadcast_arrays} for general broadcasting." # nolint
    )
  }

  lapply(args, \(x) {
    if (length(shape(x)) == 0L) {
      nv_broadcast_to(x, target_shape)
    } else {
      x
    }
  })
})

#' @title Promote Arrays to a Common Data Type
#' @description
#' Promote arrays to a common data type, see [`promotion_common()`] for more details.
#' @param ... ([`arrayish`])\cr
#'   Values to promote.
#' @return (`list()` of [`arrayish`])\cr
#'   The inputs, each at their common data type and with its own shape.
#' @examplesIf pjrt::plugins_downloaded()
#' # An integer is promoted to float
#' x <- nv_scalar(1, dtype = "f32")
#' y <- nv_scalar(1L, dtype = "i32")
#' nv_promote_to_common(x, y)
#' # an R value yields to the data type it meets within its category
#' with_default_dtypes(c(float = "f64"), nv_promote_to_common(1, x))
#' # and settles on its default otherwise
#' with_default_dtypes(c(float = "f64"), nv_promote_to_common(1, y))
#' @export
nv_promote_to_common <- jit(function(...) {
  assert_some_arrays(...)
  as_anvl_arrays(..., .promote = promotion_common())
})

#' @title Broadcast Arrays to a Common Shape
#' @description
#' Broadcasts arrays to a common shape, aligning their axes from the first
#' one, so that a vector meets a matrix as a column.
#'
#' @section Broadcasting Rules:
#' 1. If the arrays have different numbers of axes, append size-1
#'    axes to the shorter shape, so axis 1 meets axis 1. A length-`n`
#'    vector therefore lines up with the rows of an `n` by `m` matrix and
#'    is replicated across its columns. NumPy prepends instead.
#' 2. For each axis: if the sizes match, keep them; if one is 1, expand
#'    it to the other's size; otherwise raise an error.
#'
#' @section Relation to base R:
#' Base R has no broadcasting between arrays -- `matrix(1, 3, 3) +
#' matrix(1, 1, 3)` is a "non-conformable arrays" error. It does recycle a
#' *vector* over a matrix, though, and for a vector as long as the first
#' axis that lands on exactly this broadcast, which is why a vector meets a
#' matrix as a column in both. The two part ways once the lengths stop
#' lining up: `matrix(1, 2, 3) + c(1, 2, 3)` recycles on regardless, where
#' the matching broadcast is an error.
#'
#' The deviation from NumPy's broadcasting rules is still motivated by
#' keeping anvl's behaviour similar to base R in spirit, see the examples
#' for more.
#'
#' @param ... ([`arrayish`])\cr
#'   Arrays to broadcast.
#' @return (`list()` of [`arrayish`])\cr
#'   The inputs, each with its own data type and the common shape.
#' @seealso [nv_broadcast_scalars()], [nv_broadcast_to()]
#' @examplesIf pjrt::plugins_downloaded()
#' # interpreting vectors as columns
#' # base R:
#' x1 <- c(1, 2)
#' m1 <- array(1, dim = c(2, 2))
#' x1 + m1
#' # anvl:
#' args <- nv_broadcast_arrays(nv_array(x1), nv_array(m1))
#' print(args)
#' args[[1]] + args[[2]]
#'
#'
#' # axes of size 1 are expanded to the other operand's size
#' y1 <- nv_array(1:3, shape = c(1, 3))
#' y2 <- nv_array(1:3, shape = c(3, 1))
#' nv_broadcast_arrays(y1, y2)
#' @export
nv_broadcast_arrays <- jit(function(...) {
  # TODO: Better handling of sizes 0, currently the error message is not great
  assert_some_arrays(...)
  args <- as_anvl_arrays(...)
  shape <- Reduce(broadcast_shapes, lapply(args, shape))
  lapply(args, nv_broadcast_to, shape = shape)
})

#' @title Broadcast to Shape
#' @description
#' Broadcasts an array to a target shape, aligning the array's axes with the
#' leading axes of `shape`, so that a vector fills a column.
#' See [`nv_broadcast_arrays`] for more information.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @param shape (`integer()`)\cr
#'   Target shape. The input's axes are matched against its leading axes, and
#'   each must either match or be 1; trailing axes are added.
#' @return ([`arrayish`])\cr
#'   Has the given `shape` and the same data type as `x`.
#' @seealso [nv_broadcast_arrays()], [nv_broadcast_scalars()],
#'   [prim_broadcast_in_axes()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the vector fills a column and is repeated along the new trailing axis
#' x <- nv_array(c(1, 2, 3))
#' nv_broadcast_to(x, shape = c(3, 2))
#' @export
nv_broadcast_to <- function(x, shape) {
  x <- as_anvl_array(x)
  shape_op <- shape(x)
  if (!identical(shape_op, shape)) {
    # Axes align from the first: the array's existing axes map to the leading
    # axes of `shape`, and the axes it lacks are appended. StableHLO wants a
    # mapping for every input axis.
    prim_broadcast_in_axes(x, shape, seq_along(shape_op))
  } else {
    x
  }
}

#' @title Convert Data Type
#' @description
#' Converts the elements of an array to a different data type.
#' Note that R objects are handled differently than `AnvlArray`
#' inputs.
#' For R objects, we check whether the requested data type can
#' meaningfully hold the provided data and otherwise err.
#' For `AnvlArray` inputs such a check is *not* performed.
#'
#' @param x ([`arrayish`])\cr
#'   The input to convert.
#' @param dtype (`character(1)` | [`DataType`])\cr
#'   Target data type.
#' @return ([`arrayish`])\cr
#'   Has the given `dtype` and the input's shape.
#' @seealso [prim_convert()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the values are preserved, the data type changes
#' x <- nv_array(c(1L, 2L, 3L))
#' # For R inputs, we check compatability
#' nv_convert(x, dtype = "f32")
#' try(nv_convert(257L, dtype = "i8"))
#' # For AnvlArrays, we wrap around
#' nv_convert(nv_scalar(257L), dtype = "i8")
#' @export
nv_convert <- function(x, dtype) {
  if (!is_arrayish(x)) {
    cli_abort("Expected arrayish input, but got {.cls {class(x)}}")
  }
  # `materialize_at()` rather than canonicalizing first: an R value is *built*
  # at the target dtype, keeping every digit it had, where converting it from
  # its own default would round through `f32` on the way.
  materialize_at(x, as_dtype(dtype))
}

#' @title Transpose
#' @description
#' Permutes the axes of an array. You can also use `t()` for matrices.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @param permutation (`integer()` | `NULL`)\cr
#'   New ordering of axes. If `NULL` (default), reverses the axes.
#'   Negative values count from the end, i.e. `-1` refers to the last axis.
#' @return ([`arrayish`])\cr
#'   Has `x`'s data type and shape `shape(x)[permutation]`, or `rev(shape(x))`
#'   when `permutation` is `NULL`.
#' @seealso [prim_transpose()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the 2x3 becomes a 3x2, keeping its data type
#' x <- nv_matrix(1:6, nrow = 2)
#' t(x)
#' @export
nv_transpose <- function(x, permutation = NULL) {
  x <- as_anvl_array(x)
  permutation <- permutation %||% rev(seq_len(naxes(x)))
  prim_transpose(x, permutation)
}


#' @title Reshape
#' @description
#' Reshapes an array to a new shape using col-major semantics.
#'
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @param shape (`integer()`)\cr
#'   Target shape. Must have the same number of elements as `x`.
#'   At most one entry may be `-1`, in which case its extent is inferred from
#'   the remaining entries and the number of elements of `x`.
#' @return ([`arrayish`])\cr
#'   Has the given `shape` and the same data type as `x`.
#' @seealso [prim_reshape()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the elements keep their column-major order; the data type is untouched
#' x <- array(1:6, dim = c(3, 2))
#' nv_reshape(x, 6L)
#' # the order base R reads them in, too
#' c(x)
#'
#' # infer the size of the second axis
#' nv_reshape(x, c(2, -1))
#' # flatten
#' nv_reshape(x, -1)
#' @export
nv_reshape <- function(x, shape) {
  x <- as_anvl_array(x)
  # `prim_reshape()` resolves `-1` itself; resolving here too keeps the
  # identity shortcut below able to recognize a no-op reshape.
  shape <- resolve_reshape_shape(shape, prod(shape(x)), arg = "shape")
  if (!identical(shape(x), shape)) {
    prim_reshape(x, shape)
  } else {
    x
  }
}

#' @title Flatten
#' @description
#' Flattens an array with one or more axes into a 1-D array, using col-major semantics.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @return ([`arrayish`])\cr
#'   Has the input's data type, and one axis holding all of its elements --
#'   so a scalar, which has none, becomes a length-1 vector.
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' # the 2x2 matrix becomes a length-4 vector
#' x <- matrix(1:4, nrow = 2)
#' nv_flatten(x)
#' # the same column-major order base R uses
#' c(x)
nv_flatten <- function(x) {
  x <- as_anvl_array(x)
  nv_reshape(x, prod(shape(x)))
}

#' @title Concatenate
#' @description
#' Concatenates arrays along an axis. Operands are promoted to a common
#' data type and scalars are broadcast before concatenation.
#'
#' You can also use `c()` on scalars and 1-D arrays, like base R.
#' @param ... ([`arrayish`])\cr
#'   Arrays to concatenate. Can be of any data type; they are
#'   [promoted to a common data type][nv_promote_to_common()] and scalars are
#'   [broadcast][nv_broadcast_scalars()]. Must have the same shape except
#'   along `axis`.
#' @param axis (`integer(1)` | `NULL`)\cr
#'   Axis along which to concatenate.
#'   Negative values count from the end, i.e. `-1` refers to the last axis.
#'   If `NULL` (default), concatenates along axis 1, which requires every
#'   input to have at most one axis; for anything else `axis` must be given,
#'   since there is no neutral axis to join two matrices along.
#' @return ([`arrayish`])\cr
#'   Has the common data type and a shape matching the inputs in all
#'   axes except `axis`, which is the sum of input sizes.
#' @seealso [prim_concatenate()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the operands are promoted to a common data type; axis 1 grows to 6
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5, 6))
#' nv_concatenate(x, y)
#'
#' m <- nv_matrix(1:4, nrow = 2)
#' nv_concatenate(m, m, axis = 1L) # required: `m` has two axes
#' @export
nv_concatenate <- jit(
  function(..., axis = NULL) {
    assert_some_arrays(...)
    args <- do.call(nv_promote_to_common, list(...))
    shapes <- lapply(args, shape)
    ranks <- lengths(shapes)
    non_scalar_shapes <- shapes[ranks > 0L]
    n_scalars <- sum(ranks == 0L)
    max_axis <- max(max(ranks), 1L)
    if (is.null(axis)) {
      if (max(ranks) > 1L) {
        cli_abort("{.arg axis} must be provided when concatenating arrays with more than one axis.")
      }
      axis <- 1L
    } else {
      axis <- resolve_axis(axis, max_axis)
    }

    non_scalar_ranks <- unique(lengths(non_scalar_shapes))
    if (length(non_scalar_ranks) > 1L) {
      cli_abort(c(
        "All non-scalar arrays must have the same number of axes.",
        x = "Got shapes {shapes_repr(shapes)}."
      ))
    }
    non_scalar_shapes_without_axis <- lapply(non_scalar_shapes, \(shape) {
      shape[-axis]
    })
    if (length(non_scalar_shapes) && length(unique(non_scalar_shapes_without_axis)) != 1L) {
      cli_abort(c(
        "All non-scalar arrays must have the same shape apart from axis {axis}, the one they are joined along.", # nolint
        x = "Got shapes {shapes_repr(shapes)}."
      ))
    }
    size_out_axis <- n_scalars + sum(vapply(non_scalar_shapes, \(shape) shape[axis], integer(1L)))

    out_shape <- if (length(non_scalar_shapes)) {
      x <- non_scalar_shapes[[1L]]
      x[axis] <- size_out_axis
    } else {
      n_scalars
    }
    out_shape_axis_is_one <- out_shape
    out_shape_axis_is_one[axis] <- 1L
    args <- lapply(args, \(arg) {
      if (naxes(arg) == 0L) {
        nv_broadcast_to(arg, out_shape_axis_is_one)
      } else {
        arg
      }
    })
    rlang::exec(prim_concatenate, !!!args, axis = axis)
  },
  static = "axis"
)

#' @title Combine Arrays by Rows or Columns
#' @name nv_bind
#' @description
#' Combine arrays along the row (`nv_rbind`) or column (`nv_cbind`) axis.
#'
#' Each input is handled according to its rank:
#'
#' * a scalar: broadcast to match the non-stacked axes of the other inputs.
#' * 1-D: treated as a single row/column.
#' * Other: used as-is.
#'
#' @section Differences from base R:
#'
#' [base::rbind()] and [base::cbind()] applied to an [`array()`][base::array] of rank > 2
#' flatten the trailing axes into the column axis (so a `c(2, 3, 4)`
#' array becomes a `2 x 12` matrix). `nv_rbind` and `nv_cbind` instead
#' preserve all non-stacked axes: combining two `c(2, 3, 4)` arrays
#' with `nv_rbind` produces a `c(4, 3, 4)` array, and with `nv_cbind` a
#' `c(2, 6, 4)` array.
#'
#' @param ... ([`arrayish`])\cr
#'   Arrays to combine. Can be of any data type; they are
#'   [promoted to a common data type][nv_promote_to_common()], and a scalar is
#'   [broadcast][nv_broadcast_scalars()] to match the non-stacked axes.
#' @return ([`arrayish`])\cr
#'   Has the inputs' common data type. The stacked axis is the sum of their
#'   sizes along it -- rows for `nv_rbind()`, columns for `nv_cbind()` -- and
#'   every other axis is theirs unchanged.
#' @seealso [nv_concatenate()]
#' @examplesIf pjrt::plugins_downloaded()
#' # vectors as rows / columns
#' nv_rbind(nv_array(1:3), nv_array(4:6))
#' nv_cbind(nv_array(1:3), nv_array(4:6))
#'
#' # scalar broadcasting
#' nv_rbind(nv_matrix(1:6, nrow = 2), nv_scalar(0))
#'
#' # rank-3 arrays preserve trailing axes
#' a <- nv_array(1:24, shape = c(2, 3, 4))
#' shape(nv_rbind(a, a)) # c(4, 3, 4)
NULL

# Find the broadcast target shape for scalar (rank 0) inputs and verify
# that all non-scalar inputs are compatible (same rank and same size in
# every non-stacked axis). Rank-1 args are conceptually reshaped to
# a row/column for the comparison. Returns NULL when every arg is a
# scalar.
bind_target_shape <- function(args, stack_axis, fn_name) {
  shapes <- lapply(args, shape)
  non_scalar_idx <- which(lengths(shapes) > 0L)
  if (!length(non_scalar_idx)) {
    return(NULL)
  }

  reshape_for_compare <- function(s) {
    if (length(s) == 1L) {
      if (stack_axis == 1L) c(1L, s) else c(s, 1L)
    } else {
      s
    }
  }
  reshaped <- lapply(shapes[non_scalar_idx], reshape_for_compare)

  ranks <- lengths(reshaped)
  if (length(unique(ranks)) != 1L) {
    cli_abort(c(
      "{.fn {fn_name}} inputs must all have the same rank (treating rank-1 inputs as a row or column)", # nolint
      x = "Got shapes {shapes_repr(shapes)}"
    ))
  }
  non_stack <- lapply(reshaped, \(s) s[-stack_axis])
  if (length(unique(non_stack)) != 1L) {
    cli_abort(c(
      "{.fn {fn_name}} inputs must agree on every non-stacked axis",
      x = "Got shapes {shapes_repr(shapes)}"
    ))
  }
  reshaped[[1L]]
}

bind_reshape <- function(arg, stack_axis, target_shape) {
  s <- shape(arg)
  if (length(s) == 0L) {
    target <- target_shape %||% c(1L, 1L)
    target[stack_axis] <- 1L
    nv_broadcast_to(arg, target)
  } else if (length(s) == 1L) {
    nv_reshape(arg, if (stack_axis == 1L) c(1L, s) else c(s, 1L))
  } else {
    arg
  }
}

#' @rdname nv_bind
#' @export
nv_rbind <- jit(function(...) {
  assert_some_arrays(...)
  # Promoted here rather than in `nv_concatenate()` below: an R value has to be
  # built at the common dtype directly, where materializing it first would round
  # it through its default on the way there.
  args <- as_anvl_arrays(..., .promote = promotion_common())
  target_shape <- bind_target_shape(args, stack_axis = 1L, fn_name = "nv_rbind")
  args <- lapply(args, bind_reshape, stack_axis = 1L, target_shape = target_shape)
  rlang::exec(nv_concatenate, !!!args, axis = 1L)
})

#' @rdname nv_bind
#' @export
nv_cbind <- jit(function(...) {
  assert_some_arrays(...)
  args <- as_anvl_arrays(..., .promote = promotion_common())
  target_shape <- bind_target_shape(args, stack_axis = 2L, fn_name = "nv_cbind")
  args <- lapply(args, bind_reshape, stack_axis = 2L, target_shape = target_shape)
  rlang::exec(nv_concatenate, !!!args, axis = 2L)
})

#' @title Static Slice
#' @description
#' Extracts a slice from an array using static (compile-time) indices.
#' For dynamic indexing, use [nv_subset()] instead.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @param start_indices (`integer()`)\cr
#'   Start indices (inclusive), one per axis.
#' @param limit_indices (`integer()`)\cr
#'   End indices (inclusive), one per axis.
#' @param strides (`integer()`)\cr
#'   Step sizes, one per axis. A stride of 1 selects every element.
#' @return ([`arrayish`])\cr
#'   Has `x`'s data type and shape
#'   `ceiling((limit_indices - start_indices + 1) / strides)` per axis.
#' @seealso [nv_subset()], [prim_static_slice()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # elements 2 through 5, the limit being inclusive
#' x <- nv_array(1:10)
#' nv_static_slice(x, start_indices = 2L, limit_indices = 5L, strides = 1L)
#' @export
nv_static_slice <- prim_static_slice

#' @title Print Array
#' @description
#' Prints an array value to the console during JIT execution and returns the
#' input unchanged. Useful for debugging.
#' For [`RData`] inputs, that do not have an actual data type, the [default data type][default_dtypes]
#' is used for printing.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @return ([`arrayish`])\cr
#'   Returns the input unchanged, data type and shape included.
#' @seealso [prim_print()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the value is printed and handed back unchanged
#' x <- nv_array(c(1, 2, 3))
#' nv_print(x)
#' # RData is printed at the default dtype
#' nv_print(1)
#' @export
nv_print <- prim_print

#' @title Conditional Element Selection
#' @description
#' Selects elements from `true_value` or `false_value` based on `pred`,
#' analogous to R's [ifelse()].
#' @param pred ([`arrayish`])\cr
#'   Predicate array. Must be a boolean or an R logical, and scalar or the same
#'   shape as the non-scalar arguments.
#' @param true_value,false_value ([`arrayish`])\cr
#'   Values to return where `pred` is `TRUE` / `FALSE`. Can be of any data
#'   type; the two are
#'   [promoted to a common data type][nv_promote_to_common()].
#'   Scalars (including `pred`) are
#'   [broadcast][nv_broadcast_scalars()] to the shape of the non-scalar arguments.
#' @return ([`arrayish`])\cr
#'   Has the common data type of `true_value` and `false_value`, and their
#'   broadcast shape.
#' @seealso [prim_ifelse()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' pred <- nv_array(c(TRUE, FALSE, TRUE))
#' nv_ifelse(pred, nv_array(c(1, 2, 3)), nv_array(c(4, 5, 6)))
#' # scalar branches are broadcast and promoted to a common data type
#' nv_ifelse(pred, nv_scalar(1L), nv_scalar(0.5))
#' @export
nv_ifelse <- jit(function(pred, true_value, false_value) {
  # All three are aligned together -- so an R literal branch is built on the
  # device of `pred` rather than on the default one and then conflicting with it
  # -- but only the two branches are promoted, `pred` staying a bool.
  args <- as_anvl_arrays(
    pred = pred,
    true_value = true_value,
    false_value = false_value,
    .promote = promotion_common(on = c("true_value", "false_value"))
  )
  args <- nv_broadcast_scalars(args$pred, args$true_value, args$false_value)
  prim_ifelse(args[[1L]], args[[2L]], args[[3L]])
})

## Binary ops ------------------------------------------------------------------

# Jitted here rather than at each `nv_*`, because promotion and broadcasting
# make this more than one operation -- unlike `make_float_unary()`, which is a
# thin wrapper around a single primitive and stays eager.
make_do_binary <- function(f) {
  jit(function(lhs, rhs) {
    args <- nv_promote_to_common(lhs, rhs)
    args <- nv_broadcast_scalars(args[[1L]], args[[2L]])
    do.call(f, args)
  })
}

#' @title Addition
#' @description
#' Adds two arrays element-wise.
#' You can also use the `+` operator.
#' @templateVar dtypes any data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_add()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5, 6))
#' nv_add(x, y)
#' x + y
#'
#' # different data types are promoted to their common one
#' nv_add(nv_scalar(1, "f32"), nv_scalar(2, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' x + 1L
#' @export
nv_add <- make_do_binary(prim_add)

#' @title Multiplication
#' @description
#' Multiplies two arrays element-wise. You can also use the `*` operator.
#' @templateVar dtypes any data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_mul()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5, 6))
#' nv_mul(x, y)
#' x * y
#'
#' # different data types are promoted to their common one
#' nv_mul(nv_scalar(2, "f32"), nv_scalar(3, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' x * 2L
#' @export
nv_mul <- make_do_binary(prim_mul)

#' @title Subtraction
#' @description
#' Subtracts two arrays element-wise. You can also use the `-` operator.
#' @templateVar dtypes any numeric data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_sub()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(4, 5, 6))
#' y <- nv_array(c(1, 2, 3))
#' nv_sub(x, y)
#' x - y
#'
#' # different data types are promoted to their common one
#' nv_sub(nv_scalar(5, "f32"), nv_scalar(3, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' x - 1L
#' @export
nv_sub <- make_do_binary(prim_sub)

#' @title Division
#' @description
#' Divides two arrays element-wise. You can also use the `/` operator.
#' @templateVar dtypes any numeric data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_div()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(10, 20, 30))
#' y <- nv_array(c(2, 5, 10))
#' nv_div(x, y)
#' x / y
#'
#' # different data types are promoted to their common one
#' nv_div(nv_scalar(10, "f32"), nv_scalar(4, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' x / 2L
#' @export
nv_div <- make_do_binary(prim_div)

#' @title Power
#' @description
#' Raises `lhs` to the power of `rhs` element-wise. You can also use the `^` operator.
#' @templateVar dtypes any numeric data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_pow()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(2, 3, 4))
#' y <- nv_array(c(3, 2, 1))
#' x ^ y
#'
#' # different data types are promoted to their common one
#' nv_pow(nv_scalar(2, "f32"), nv_scalar(3, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' x^2L
#' @export
nv_pow <- make_do_binary(prim_pow)

#' @title Equal
#' @description
#' Element-wise equality comparison. You can also use the `==` operator.
#' @templateVar dtypes any data type
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_eq()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(1, 3, 2))
#' nv_eq(x, y)
#' x == y
#'
#' # different data types are promoted to their common one
#' nv_eq(nv_scalar(1, "f32"), nv_scalar(1, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' x == 2L
#' @export
nv_eq <- make_do_binary(prim_eq)

#' @title Not Equal
#' @description
#' Element-wise inequality comparison. You can also use the `!=` operator.
#' @templateVar dtypes any data type
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_ne()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(1, 3, 2))
#' nv_ne(x, y)
#' x != y
#'
#' # different data types are promoted to their common one
#' nv_ne(nv_scalar(1, "f32"), nv_scalar(2, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' x != 2L
#' @export
nv_ne <- make_do_binary(prim_ne)

#' @title Greater Than
#' @description
#' Element-wise greater than comparison. You can also use the `>` operator.
#' @templateVar dtypes any data type
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_gt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' nv_gt(x, y)
#' x > y
#'
#' # different data types are promoted to their common one
#' nv_gt(nv_scalar(2, "f32"), nv_scalar(1, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' x > 2L
#' @export
nv_gt <- make_do_binary(prim_gt)

#' @title Greater Than or Equal
#' @description
#' Element-wise greater than or equal comparison. You can also use the `>=` operator.
#' @templateVar dtypes any data type
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_ge()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' nv_ge(x, y)
#' x >= y
#'
#' # different data types are promoted to their common one
#' nv_ge(nv_scalar(2, "f32"), nv_scalar(1, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' x >= 2L
#' @export
nv_ge <- make_do_binary(prim_ge)

#' @title Less Than
#' @description
#' Element-wise less than comparison. You can also use the `<` operator.
#' @templateVar dtypes any data type
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_lt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' nv_lt(x, y)
#' x < y
#'
#' # different data types are promoted to their common one
#' nv_lt(nv_scalar(1, "f32"), nv_scalar(2, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' x < 2L
#' @export
nv_lt <- make_do_binary(prim_lt)

#' @title Less Than or Equal
#' @description
#' Element-wise less than or equal comparison. You can also use the `<=` operator.
#' @templateVar dtypes any data type
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_le()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' nv_le(x, y)
#' x <= y
#'
#' # different data types are promoted to their common one
#' nv_le(nv_scalar(1, "f32"), nv_scalar(2, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' x <= 2L
#' @export
nv_le <- make_do_binary(prim_le)

#' @title Maximum
#' @description
#' Element-wise maximum of two arrays.
#' @templateVar dtypes any data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_max()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 5, 3))
#' y <- nv_array(c(4, 2, 6))
#' nv_max(x, y)
#'
#' # different data types are promoted to their common one
#' nv_max(nv_scalar(1, "f32"), nv_scalar(5, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' nv_max(x, 2L)
#' @export
nv_max <- make_do_binary(prim_max)

#' @title Minimum
#' @description
#' Element-wise minimum of two arrays.
#' @templateVar dtypes any data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_min()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 5, 3))
#' y <- nv_array(c(4, 2, 6))
#' nv_min(x, y)
#'
#' # different data types are promoted to their common one
#' nv_min(nv_scalar(1, "f32"), nv_scalar(5, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' nv_min(x, 2L)
#' @export
nv_min <- make_do_binary(prim_min)

#' @title Remainder (Truncating)
#' @description
#' Element-wise remainder. This
#' differs from base R's `%%`, use [`nv_mod()`]/`%%` instead.
#' @templateVar dtypes any numeric data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nv_mod()] for the flooring remainder, [prim_remainder()] for the
#'   underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(7, 8, 9))
#' y <- nv_array(c(3, 3, 4))
#' nv_remainder(x, y)
#'
#' # different data types are promoted to their common one
#' nv_remainder(nv_scalar(7, "f32"), nv_scalar(3, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' nv_remainder(x, 3L)
#' @export
nv_remainder <- make_do_binary(prim_remainder)

#' @title Modulo (Flooring Remainder)
#' @description
#' Element-wise flooring remainder of division. The sign of the result equals the sign of `rhs`, matching base R's `%%`
#' operator.
#' @templateVar dtypes any numeric data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nv_remainder()] for truncating remainder, [nv_floor_div()] for the
#'   matching division, [prim_remainder()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1L, -1L))
#' y <- nv_array(c(-3L, 3L))
#' nv_mod(x, y)
#' as.vector(x) %% as.vector(y)
#'
#' # different data types are promoted to their common one
#' nv_mod(nv_scalar(1L, "i32"), nv_scalar(-3L, "i64"))
#'
#' # a scalar is broadcast
#' x %% 2L
#' @export
nv_mod <- jit(function(lhs, rhs) {
  args <- nv_promote_to_common(lhs, rhs)
  args <- nv_broadcast_scalars(args[[1L]], args[[2L]])
  lhs <- args[[1L]]
  rhs <- args[[2L]]
  rest <- nv_remainder(lhs, rhs)
  if (is_dtype_uint(peek_dtype(lhs))) {
    # Neither operand can be negative, so the truncating remainder already
    # floors and there is nothing to shift. `nv_abs()` below has no unsigned
    # lowering either.
    return(rest)
  }
  shifted <- nv_ifelse((rest != 0L) & ((rest < 0L) != (rhs < 0L)), rest + rhs, rest)
  nv_ifelse(nv_abs(shifted) >= nv_abs(rhs), 0L, shifted)
})

#' @title Flooring Division
#' @description
#' Element-wise flooring division.
#' You can also call this via the `%/%` operator.
#' The result is the largest whole number that does not exceed `lhs / rhs`.
#'
#' @templateVar dtypes any numeric data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nv_mod()] for the matching remainder, [nv_div()] for the
#'   division itself.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(7L, -7L))
#' y <- nv_array(c(2L, 2L))
#' nv_floor_div(x, y)
#' x %/% y
#' @export
nv_floor_div <- jit(function(lhs, rhs) {
  args <- nv_promote_to_common(lhs, rhs)
  args <- nv_broadcast_scalars(args[[1L]], args[[2L]])
  lhs <- args[[1L]]
  rhs <- args[[2L]]
  dt <- peek_dtype(lhs)
  if (is_dtype_float(dt)) {
    return(nv_floor(nv_div(lhs, rhs)))
  }
  if (is_dtype_uint(dt)) {
    # Unsigned division cannot be negative, so there is nothing to floor.
    return(nv_div(lhs, rhs))
  }
  # Integer division truncates towards zero, so subtract the flooring
  # remainder first to make the division exact.
  nv_div(nv_sub(lhs, nv_mod(lhs, rhs)), rhs)
})

#' @title Bitwise AND
#' @description
#' Element-wise bitwise AND -- a logical AND on a boolean input, and a
#' bit-by-bit one on an integer. You can also use the `&` operator, which expects
#' `bool` inputs, however.
#' @templateVar dtypes any integerish data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_and()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' y <- nv_array(c(TRUE, TRUE, FALSE))
#' x & y
#'
#' # different data types are promoted to their common one
#' nv_and(nv_scalar(12L, "i32"), nv_scalar(10L, "i64"))
#'
#' # a scalar is broadcast
#' x & TRUE
#' @export
nv_and <- make_do_binary(prim_and)

#' @title Bitwise OR
#' @description
#' Element-wise bitwise OR -- a logical OR on a boolean input, and a
#' bit-by-bit one on an integer. You can also use the `|` operator,
#' which expects `bool` inputs, however.
#' @templateVar dtypes any integerish data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_or()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' y <- nv_array(c(TRUE, TRUE, FALSE))
#' x | y
#'
#' # different data types are promoted to their common one
#' nv_or(nv_scalar(12L, "i32"), nv_scalar(10L, "i64"))
#'
#' # a scalar is broadcast
#' x | TRUE
#' @export
nv_or <- make_do_binary(prim_or)

#' @title Bitwise XOR
#' @description
#' Element-wise bitwise XOR -- a logical XOR on a boolean input, and a
#' bit-by-bit one on an integer.
#' For *logical* inputs, you can also use `xor`.
#' @templateVar dtypes any integerish data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_xor()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' y <- nv_array(c(TRUE, TRUE, FALSE))
#' nv_xor(x, y)
#'
#' # different data types are promoted to their common one
#' nv_xor(nv_scalar(12L, "i32"), nv_scalar(10L, "i64"))
#'
#' # a scalar is broadcast
#' nv_xor(x, TRUE)
#' @export
nv_xor <- make_do_binary(prim_xor)

#' @title Shift Left
#' @description
#' Element-wise left bit shift.
#' @templateVar dtypes any integer data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_shift_left()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1L, 2L, 4L))
#' y <- nv_array(c(1L, 2L, 1L))
#' nv_shift_left(x, y)
#'
#' # different data types are promoted to their common one
#' nv_shift_left(nv_scalar(8L, "i32"), nv_scalar(2L, "i64"))
#'
#' # a scalar is broadcast
#' nv_shift_left(x, 1L)
#' @export
nv_shift_left <- make_do_binary(prim_shift_left)

#' @title Logical Shift Right
#' @description
#' Element-wise logical right bit shift.
#' @templateVar dtypes any integer data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_shift_right_logical()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(8L, 16L, 32L))
#' y <- nv_array(c(1L, 2L, 3L))
#' nv_shift_right_logical(x, y)
#'
#' # different data types are promoted to their common one
#' nv_shift_right_logical(nv_scalar(32L, "i32"), nv_scalar(2L, "i64"))
#'
#' # a scalar is broadcast
#' nv_shift_right_logical(x, 1L)
#' @export
nv_shift_right_logical <- make_do_binary(prim_shift_right_logical)

#' @title Arithmetic Shift Right
#' @description
#' Element-wise arithmetic right bit shift.
#' @templateVar dtypes any integer data type
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_shift_right_arithmetic()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(8L, -16L, 32L))
#' y <- nv_array(c(1L, 2L, 3L))
#' nv_shift_right_arithmetic(x, y)
#'
#' # different data types are promoted to their common one
#' nv_shift_right_arithmetic(nv_scalar(-32L, "i32"), nv_scalar(2L, "i64"))
#'
#' # a scalar is broadcast
#' nv_shift_right_arithmetic(x, 1L)
#' @export
nv_shift_right_arithmetic <- make_do_binary(prim_shift_right_arithmetic)

#' @title Arctangent 2
#' @description
#' Element-wise two-argument arctangent, i.e. the angle (in radians) between the positive x-axis
#' and the point `(rhs, lhs)`.
#' @template params_lhs_rhs_tofloat
#' @return ([`arrayish`])\cr
#'   Has the inputs' broadcast shape, and their common data type -- or the
#'   default float data type (see [`default_dtypes()`]) where that was an
#'   integer one.
#' @seealso [prim_atan2()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' y <- nv_array(c(1, 0, -1))
#' x <- nv_array(c(0, 1, 0))
#' nv_atan2(y, x)
#'
#' # different data types are promoted to their common one
#' nv_atan2(nv_scalar(1, "f32"), nv_scalar(1, "f64"))
#'
#' # a scalar is broadcast and an R integer is converted to a float
#' nv_atan2(y, 1L)
#' @export
nv_atan2 <- jit(function(lhs, rhs) {
  args <- promote_to_common_float(lhs = lhs, rhs = rhs)
  args <- nv_broadcast_scalars(args$lhs, args$rhs)
  prim_atan2(args[[1L]], args[[2L]])
})


#' @title Bitcast Conversion
#' @name nv_bitcast_convert
#' @description
#' Reinterprets the bits of an array as a different data type without modifying
#' the underlying data. If the target type is narrower, an extra trailing
#' axis is added; if wider, the last axis is consumed.
#' @inheritParams prim_bitcast_convert
#' @return ([`arrayish`])\cr
#'   Has the given `dtype`, and the shape described under `dtype`.
#' @seealso [prim_bitcast_convert()], which this is an alias of, and
#'   [nv_convert()] for value-preserving type conversion.
#' @examplesIf pjrt::plugins_downloaded()
#' # the bits of one i32 reread as four i8, in a new trailing axis
#' x <- nv_array(1L, dtype = "i32")
#' nv_bitcast_convert(x, dtype = "i8")
#' @export
nv_bitcast_convert <- prim_bitcast_convert

## Unary ops ------------------------------------------------------------------

# A unary `nv_*` function that computes in floating point: an int-like array is
# converted to the default float first, the way base R's `sqrt(1L)` returns a
# double. Everything else reaches the primitive unchanged, so a boolean array
# is rejected there rather than silently computed on.
make_float_unary <- function(f) {
  function(x) f(int_to_float(x))
}

#' @title Negation
#' @description
#' Negates an array element-wise. You can also use the unary `-` operator.
#' @templateVar dtypes any numeric data type
#' @template param_unary_x
#' @template return_unary
#' @seealso [prim_negate()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(1, -2, 3))
#' -x
#'
#' # an R value materializes at its default data type
#' nv_negate(1)
#' @export
nv_negate <- prim_negate

#' @title Bitwise Not
#' @description
#' Element-wise bitwise NOT -- a logical negation on a boolean input, and a
#' bit-by-bit complement on an integer, so `nv_not(12L)` is `-13`. You can also
#' use the `!` operator, which expects `bool` inputs.
#' @templateVar dtypes any integerish data type
#' @template param_unary_x
#' @template return_unary
#' @seealso [prim_not()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # on a boolean this is a logical negation
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' !x
#'
#' # on an integer it complements every bit, so `12L` becomes `-13`
#' nv_not(nv_array(12L))
#' @export
nv_not <- prim_not

#' @title Absolute Value
#' @description
#' Element-wise absolute value. You can also use `abs()`.
#' @templateVar dtypes any signed numeric data type
#' @template param_unary_x
#' @template return_unary
#' @seealso [prim_abs()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(-1, 2, -3))
#' abs(x)
#'
#' # an R value materializes at its default data type
#' nv_abs(-1)
#' @export
nv_abs <- prim_abs

#' @title Square Root
#' @description
#' Element-wise square root. You can also use `sqrt()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_sqrt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(1, 4, 9))
#' sqrt(x)
#'
#' # an R value materializes at its default data type
#' nv_sqrt(4)
#' @export
nv_sqrt <- make_float_unary(prim_sqrt)

#' @title Reciprocal Square Root
#' @description
#' Element-wise reciprocal square root, i.e. `1 / sqrt(x)`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_rsqrt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(1, 4, 9))
#' nv_rsqrt(x)
#'
#' # an R value materializes at its default data type
#' nv_rsqrt(4)
#' @export
nv_rsqrt <- make_float_unary(prim_rsqrt)

#' @title Natural Logarithm
#' @description
#' Element-wise natural logarithm. You can also use `log()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_log()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(1, 2.718, 7.389))
#' log(x)
#'
#' # an R value materializes at its default data type
#' nv_log(2)
#' @export
nv_log <- make_float_unary(prim_log)

#' @title Hyperbolic Tangent
#' @description
#' Element-wise hyperbolic tangent. You can also use `tanh()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_tanh()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(-1, 0, 1))
#' tanh(x)
#'
#' # an R value materializes at its default data type
#' nv_tanh(1)
#' @export
nv_tanh <- make_float_unary(prim_tanh)

#' @title Tangent
#' @description
#' Element-wise tangent. You can also use `tan()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_tan()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(0, 0.5, 1))
#' tan(x)
#'
#' # an R value materializes at its default data type
#' nv_tan(0.5)
#' @export
nv_tan <- make_float_unary(prim_tan)

#' @title Sine
#' @description
#' Element-wise sine. You can also use `sin()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_sin()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(0, pi / 2, pi))
#' sin(x)
#'
#' # an R value materializes at its default data type
#' nv_sin(0)
#' @export
nv_sin <- make_float_unary(prim_sin)

#' @title Cosine
#' @description
#' Element-wise cosine. You can also use `cos()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_cos()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(0, pi / 2, pi))
#' cos(x)
#'
#' # an R value materializes at its default data type
#' nv_cos(0)
#' @export
nv_cos <- make_float_unary(prim_cos)

#' @title Sine of a Multiple of Pi
#' @description
#' Element-wise `sin(pi * x)`. You can also use `sinpi()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [nv_cospi()], [nv_tanpi()], [nv_sin()]
#' @examplesIf pjrt::plugins_downloaded()
#' sinpi(nv_array(c(0, 0.5, 1, 1.5)))
#' @export
nv_sinpi <- jit(function(x) {
  x <- as_anvl_array(int_to_float(x))
  n <- nv_round(x, method = "nearest_even")
  reduced <- nv_sin((x - n) * pi)
  # The sine of `pi * n` alternates in sign with the parity of `n`.
  nv_ifelse(nv_mod(n, 2L) == 0L, reduced, -reduced)
})

#' @title Cosine of a Multiple of Pi
#' @description
#' Element-wise `cos(pi * x)`. You can also use `cospi()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [nv_sinpi()], [nv_tanpi()], [nv_cos()]
#' @examplesIf pjrt::plugins_downloaded()
#' cospi(nv_array(c(0, 0.5, 1, 1.5)))
#' @export
nv_cospi <- jit(function(x) {
  # cos(pi * x) == sin(pi * (x + 1/2))
  nv_sinpi(as_anvl_array(int_to_float(x)) + 0.5)
})

#' @title Tangent of a Multiple of Pi
#' @description
#' Element-wise `tan(pi * x)`. You can also use `tanpi()`.
#' Like base R's [base::tanpi()], it is exact for a whole argument and `NaN` at
#' the half integers, where the tangent has its poles.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [nv_sinpi()], [nv_cospi()], [nv_tan()]
#' @examplesIf pjrt::plugins_downloaded()
#' tanpi(nv_array(c(0, 0.25, 0.5, 1)))
#' @export
nv_tanpi <- jit(function(x) {
  x <- as_anvl_array(int_to_float(x))
  denominator <- nv_cospi(x)
  # Otherwise we get (+-)inf depending on which side we land, which is bad
  nv_ifelse(denominator == 0L, NaN, nv_sinpi(x) / denominator)
})

#' @title Floor
#' @description
#' Element-wise floor (round toward negative infinity). You can also use `floor()`.
#' @template param_unary_x_round
#' @template return_unary
#' @seealso [prim_floor()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(1.2, 2.7, -1.5))
#' floor(x)
#' floor(nv_array(1:3)) # an integer array is already whole
#' @export
nv_floor <- function(x) {
  if (is_intlike(x)) as_anvl_array(x) else prim_floor(x)
}

#' @title Ceiling
#' @description
#' Element-wise ceiling (round toward positive infinity). You can also use `ceiling()`.
#' @template param_unary_x_round
#' @template return_unary
#' @seealso [prim_ceil()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(1.2, 2.7, -1.5))
#' ceiling(x)
#' ceiling(nv_array(1:3)) # an integer array is already whole
#' @export
nv_ceiling <- function(x) {
  if (is_intlike(x)) as_anvl_array(x) else prim_ceil(x)
}

#' @title Truncate
#' @description
#' Element-wise truncation (round toward zero). You can also use `trunc()`.
#' @template param_unary_x_round
#' @template return_unary
#' @seealso [nv_floor()], [nv_ceiling()], [nv_round()].
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(1.2, 2.7, -1.5))
#' trunc(x)
#' trunc(nv_array(1:3)) # an integer array is already whole
#' @export
nv_trunc <- jit(function(x) {
  x <- as_anvl_array(x)
  if (is_intlike(x)) {
    return(x)
  }
  nv_mul(nv_sign(x), nv_floor(nv_abs(x)))
})

#' @title Sign
#' @description
#' Element-wise sign function: `-1`, `0` or `1` at the input's data type. An
#' unsigned input holds no negative value, so its sign is `0` or `1`, like
#' base R's `sign()` on a non-negative number. You can also use `sign()`.
#' @templateVar dtypes any numeric data type
#' @template param_unary_x
#' @template return_unary
#' @seealso [prim_sign()] for the underlying primitive, which takes a signed
#'   input only.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(-3, 0, 5))
#' sign(x)
#'
#' # an unsigned input is 0 where it is 0 and 1 everywhere else
#' nv_sign(nv_array(c(0L, 3L), dtype = "ui32"))
#'
#' # an R value materializes at its default data type
#' nv_sign(-3)
#' @export
nv_sign <- jit(function(x) {
  x <- as_anvl_array(x)
  if (is_dtype_uint(dtype(x))) {
    # `prim_sign` takes a signed input only, and an unsigned one is never
    # negative, so its sign is whether it is non-zero.
    return(nv_convert(nv_gt(x, 0L), dtype(x)))
  }
  prim_sign(x)
})

#' @title Exponential
#' @description
#' Element-wise exponential. You can also use `exp()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_exp()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(0, 1, 2))
#' exp(x)
#'
#' # an R value materializes at its default data type
#' nv_exp(1)
#' @export
nv_exp <- make_float_unary(prim_exp)

#' @title Exponential Minus One
#' @description
#' Element-wise `exp(x) - 1`, more accurate for small `x`. You can also use
#' `expm1()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_expm1()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(0, 0.001, 1))
#' nv_expm1(x)
#'
#' # an R value materializes at its default data type
#' nv_expm1(0.001)
#' @export
nv_expm1 <- make_float_unary(prim_expm1)

#' @title Log Plus One
#' @description
#' Element-wise `log(1 + x)`, more accurate for small `x`. You can also use
#' `log1p()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_log1p()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(0, 0.001, 1))
#' nv_log1p(x)
#'
#' # an R value materializes at its default data type
#' nv_log1p(0.001)
#' @export
nv_log1p <- make_float_unary(prim_log1p)

#' @title Cube Root
#' @description
#' Element-wise cube root.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_cbrt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(1, 8, 27))
#' nv_cbrt(x)
#'
#' # an R value materializes at its default data type
#' nv_cbrt(8)
#' @export
nv_cbrt <- make_float_unary(prim_cbrt)

#' @title Logistic (Sigmoid)
#' @description
#' Element-wise logistic sigmoid: `1 / (1 + exp(-x))`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_logistic()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(-2, 0, 2))
#' nv_logistic(x)
#'
#' # an R value materializes at its default data type
#' nv_logistic(2)
#' @export
nv_logistic <- make_float_unary(prim_logistic)

#' @title Arc Cosine
#' @description
#' Element-wise inverse cosine. You can also use `acos()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_acos()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(-1, 0, 1))
#' acos(x)
#'
#' # an R value materializes at its default data type
#' nv_acos(0.5)
#' @export
nv_acos <- make_float_unary(prim_acos)

#' @title Inverse Hyperbolic Cosine
#' @description
#' Element-wise inverse hyperbolic cosine. You can also use `acosh()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_acosh()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(1, 2, 10))
#' acosh(x)
#'
#' # an R value materializes at its default data type
#' nv_acosh(2)
#' @export
nv_acosh <- make_float_unary(prim_acosh)

#' @title Arc Sine
#' @description
#' Element-wise inverse sine. You can also use `asin()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_asin()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(-1, 0, 1))
#' asin(x)
#'
#' # an R value materializes at its default data type
#' nv_asin(0.5)
#' @export
nv_asin <- make_float_unary(prim_asin)

#' @title Inverse Hyperbolic Sine
#' @description
#' Element-wise inverse hyperbolic sine. You can also use `asinh()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_asinh()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(-1, 0, 1))
#' asinh(x)
#'
#' # an R value materializes at its default data type
#' nv_asinh(1)
#' @export
nv_asinh <- make_float_unary(prim_asinh)

#' @title Arc Tangent
#' @description
#' Element-wise inverse tangent. You can also use `atan()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_atan()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(-1, 0, 1))
#' atan(x)
#'
#' # an R value materializes at its default data type
#' nv_atan(1)
#' @export
nv_atan <- make_float_unary(prim_atan)

#' @title Inverse Hyperbolic Tangent
#' @description
#' Element-wise inverse hyperbolic tangent. You can also use `atanh()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_atanh()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(-0.5, 0, 0.5))
#' atanh(x)
#'
#' # an R value materializes at its default data type
#' nv_atanh(0.5)
#' @export
nv_atanh <- make_float_unary(prim_atanh)

#' @title Hyperbolic Cosine
#' @description
#' Element-wise hyperbolic cosine. You can also use `cosh()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_cosh()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(-1, 0, 1))
#' cosh(x)
#'
#' # an R value materializes at its default data type
#' nv_cosh(1)
#' @export
nv_cosh <- make_float_unary(prim_cosh)

#' @title Hyperbolic Sine
#' @description
#' Element-wise hyperbolic sine. You can also use `sinh()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_sinh()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(-1, 0, 1))
#' sinh(x)
#'
#' # an R value materializes at its default data type
#' nv_sinh(1)
#' @export
nv_sinh <- make_float_unary(prim_sinh)

#' @title Digamma
#' @description
#' Element-wise digamma function (logarithmic derivative of the gamma
#' function). You can also use `digamma()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_digamma()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(0.5, 1, 2, 5))
#' digamma(x)
#'
#' # an R value materializes at its default data type
#' nv_digamma(2)
#' @export
nv_digamma <- make_float_unary(prim_digamma)

#' @title Log-Gamma
#' @description
#' Element-wise natural logarithm of the absolute value of the gamma
#' function. You can also use `lgamma()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_lgamma()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(0.5, 1, 2, 5))
#' lgamma(x)
#'
#' # an R value materializes at its default data type
#' nv_lgamma(2)
#' @export
nv_lgamma <- make_float_unary(prim_lgamma)

#' @title Gamma Function
#' @description
#' Element-wise gamma function. You can also use `gamma()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [nv_lgamma()], which is what the hardware computes.
#' @examplesIf pjrt::plugins_downloaded()
#' gamma(nv_array(c(0.5, 1, 5, -1.5)))
#' @export
nv_gamma <- jit(function(x) {
  x <- as_anvl_array(int_to_float(x))
  positive <- nv_exp(nv_lgamma(x))
  # lgamma() is the log of the *absolute* gamma, so for a negative argument use
  # Euler's reflection formula gamma(x) * gamma(1 - x) = pi / sin(pi * x),
  # whose right-hand side is evaluated at 1 - x > 1. Where the reflection is
  # not selected it is evaluated at a regular point: at a positive whole
  # number sin(pi * x) * gamma(1 - x) is 0 * Inf, and the cotangent that
  # nv_ifelse() sends into the discarded branch would pick the NaN up.
  x_reflect <- nv_ifelse(x < 0L, x, -0.5)
  reflected <- pi / (nv_sinpi(x_reflect) * nv_exp(nv_lgamma(1L - x_reflect)))
  out <- nv_ifelse(x < 0L, reflected, positive)
  nv_ifelse((x <= 0L) & (x == nv_floor(x)), NaN, out)
})

#' @title Polygamma
#' @description
#' Element-wise polygamma function: the `(n+1)`-th derivative of the
#' log-gamma function. For `n = 0` this is the digamma function; for `n = 1`,
#' `trigamma()` dispatches here.
#' @param n,x ([`arrayish`])\cr
#'   Order of the polygamma function and the value to evaluate it at. `n`
#'   typically holds non-negative whole numbers. Can be any numeric data type:
#'   the two are [promoted to a common data type][nv_promote_to_common()] and
#'   that is then converted to the default float data type (see
#'   [`default_dtypes()`]) where it is not a float already, since a float is
#'   all [prim_polygamma()] takes. An R value assumes the other operand's data
#'   type within its [data type category][dtypes], and settles on the default
#'   float when neither has one.
#'   Scalars are [broadcast][nv_broadcast_scalars()] to the shape of the other,
#'   so `nv_polygamma(1, x)` works for any float `x`.
#' @return ([`arrayish`])\cr
#'   Has the inputs' broadcast shape, and their common data type -- or the
#'   default float data type (see [`default_dtypes()`]) where that was an
#'   integer one.
#' @seealso [prim_polygamma()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the R `1` is built at `x`'s float data type and broadcast
#' x <- nv_array(c(0.5, 1, 2, 5))
#' nv_polygamma(1, x) # trigamma
#' @export
nv_polygamma <- jit(function(n, x) {
  args <- promote_to_common_float(n = n, x = x)
  args <- nv_broadcast_scalars(args$n, args$x)
  do.call(prim_polygamma, args)
})

#' @title Error Function
#' @description
#' Element-wise error function `erf(x) = (2 / sqrt(pi)) * integral_0^x exp(-t^2) dt`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_erf()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(-1, 0, 1))
#' nv_erf(x)
#'
#' # an R value materializes at its default data type
#' nv_erf(1)
#' @export
nv_erf <- make_float_unary(prim_erf)

#' @title Inverse Error Function
#' @description
#' Element-wise inverse error function (the inverse of `erf` on `(-1, 1)`).
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_erf_inv()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(-0.5, 0, 0.5))
#' nv_erf_inv(x)
#'
#' # an R value materializes at its default data type
#' nv_erf_inv(0.5)
#' @export
nv_erf_inv <- make_float_unary(prim_erf_inv)

#' @title Complementary Error Function
#' @description
#' Element-wise complementary error function `erfc(x) = 1 - erf(x)`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [prim_erfc()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(-1, 0, 1))
#' nv_erfc(x)
#'
#' # an R value materializes at its default data type
#' nv_erfc(1)
#' @export
nv_erfc <- make_float_unary(prim_erfc)

#' @title Is Finite
#' @description
#' Element-wise check if values are finite (not `Inf`, `-Inf`, or `NaN`).
#' Only a float holds a non-finite value, so the answer for any other data
#' type is all `TRUE` and is built as a constant rather than computed, the way
#' base R's `is.finite()` answers `TRUE` for an integer.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @template return_unary_boolean
#' @seealso [prim_is_finite()] for the underlying primitive, which takes a
#'   float only.
#' @examplesIf pjrt::plugins_downloaded()
#' # the result is boolean, whatever float data type the input has
#' x <- nv_array(c(1, Inf, NaN, -Inf, 0))
#' nv_is_finite(x)
#'
#' # all TRUE for an integer input, which has no non-finite value
#' nv_is_finite(nv_array(1:3))
#'
#' # an R value materializes at its default data type before the test
#' nv_is_finite(1)
#' @export
nv_is_finite <- jit(function(x) {
  x <- as_anvl_array(x)
  if (!is_dtype_float(dtype(x))) {
    return(nv_fill_like(x, TRUE, dtype = "bool"))
  }
  prim_is_finite(x)
})

#' @title Population Count
#' @description
#' Element-wise population count (number of set bits).
#' @templateVar dtypes any integer data type
#' @template param_unary_x
#' @template return_unary
#' @seealso [prim_popcnt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the set bits are counted, at the input's own integer data type
#' x <- nv_array(c(7L, 3L, 15L))
#' nv_popcnt(x)
#' @export
nv_popcnt <- prim_popcnt

#' @title Clamp
#' @description
#' Element-wise clamp: `min(max(min_val, x), max_val)`.
#' @param min_val,max_val ([`arrayish`])\cr
#'   Lower and upper bound, each scalar or the same shape as `x`. They are
#'   brought to `x`'s data type: an R value is built at it when its category can
#'   reach it (`0L` serves an integer and a float `x` alike, `0` only a float
#'   one), and a value that already has a data type is converted unless that
#'   would narrow it -- an `f64` bound for an `f32` `x` is an error rather than
#'   a silent narrowing.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @return ([`arrayish`])\cr
#'   Has `x`'s shape and data type.
#' @seealso [prim_clamp()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the bounds are brought to `x`'s data type
#' x <- nv_array(c(-1, 0.5, 2))
#' nv_clamp(nv_scalar(0), x, nv_scalar(1))
#'
#' # an R integer serves a float `x` too, since a float can hold it
#' nv_clamp(0L, x, 1L)
#' @export
nv_clamp <- jit(function(min_val, x, max_val) {
  args <- as_anvl_arrays(min_val = min_val, x = x, max_val = max_val, .promote = promotion_like("x"))
  prim_clamp(args$min_val, args$x, args$max_val)
})

#' @title Reverse
#' @description
#' Reverses the order of elements along the given axes, every axis by default.
#' You can also use `rev()`, which always reverses along every axis.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @param axes (`integer()` | `NULL`)\cr
#'   Axes to reverse. Negative values count from the end, i.e. `-1` refers to
#'   the last axis. If `NULL` (default), reverses along every axis.
#' @return ([`arrayish`])\cr
#'   Has the same shape and data type as `x`.
#' @seealso [prim_reverse()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the order along axis 1 is flipped
#' x <- nv_array(c(1, 2, 3, 4, 5))
#' nv_reverse(x)
#'
#' m <- nv_matrix(1:6, nrow = 2)
#' nv_reverse(m) # every axis
#' nv_reverse(m, axes = 2L) # columns only
#' @export
nv_reverse <- function(x, axes = NULL) {
  x <- as_anvl_array(x)
  axes <- axes %||% seq_len(naxes(x))
  if (length(axes) == 0L) {
    # Nothing to reverse -- a scalar, or an explicitly empty `axes`. The
    # primitive rejects an empty axis list, so answer it here.
    return(x)
  }
  prim_reverse(x, axes = axes)
}

#' @title Iota
#' @description
#' Creates an array with values increasing along the specified axis,
#' starting from `start`.
#'
#' `nv_iota_like()` is a variant where `dtype`, `shape`, and
#' `device` default to those of `like`.
#' @param axis (`integer(1)`)\cr
#'   Axis along which values increase.
#'   Negative values count from the end of `shape`, i.e. `-1` refers to the
#'   last axis.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_iota_like()`).
#' @param dtype (`character(1)` | [`DataType`])\cr
#'   Data type of the result, required here. Can be any numeric data type.
#'   For `nv_iota_like()` it may be `NULL`, which uses `dtype(like)`.
#' @param shape (`integer()`)\cr
#'   Shape of the result.
#' @param start (`integer(1)`)\cr
#'   Starting value (default 1). Built at `dtype`, as the increments are.
#' @template param_device
#' @return ([`arrayish`])\cr
#'   Has the given `dtype` and `shape`.
#' @seealso [nv_seq()] for a simpler 1-D sequence, [nv_linspace()] for evenly
#'   spaced values, [prim_iota()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the sequence is built at the requested data type
#' nv_iota(axis = 1L, dtype = "i32", shape = 5L)
#'
#' # `_like` takes shape, data type and device from an existing array
#' x <- nv_fill(0L, shape = c(2, 3))
#' nv_iota_like(x, axis = 1L)
#' @export
nv_iota <- prim_iota

#' @title Sequence
#' @description
#' Creates a 1-D array with the values from `start` to `end` in steps of `by`,
#' like R's `seq(start, end, by)`. The sequence counts down when `end` lies
#' below `start`, and stops before `end` when `end` is not reachable in whole
#' steps: `nv_seq(0, 9, by = 2)` ends at `8`.
#'
#' `nv_seq_like()` is a variant where `dtype` and `device`
#' default to those of `like`.
#' @param start,end (`integer(1)`)\cr
#'   First value and upper (or, when counting down, lower) limit of the
#'   sequence.
#' @param by (`NULL` | `integer(1)`)\cr
#'   Step size, which must be a non-zero whole number pointing from `start`
#'   towards `end`. `NULL` (default) uses `-1` if `start > end` and `1`
#'   otherwise.
#' @param dtype (`NULL` | `character(1)` | [`DataType`])\cr
#'   Data type of the result. Can be any numeric data type.
#'   `NULL` (default) uses the default integer data
#'   type (see [`default_dtypes()`]), since the values are whole. For
#'   `nv_seq_like()`, `NULL` uses `dtype(like)`.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_seq_like()`).
#' @template param_device
#' @return ([`arrayish`])\cr
#'   Has `dtype` and shape `(end - start) %/% by + 1`.
#' @seealso [nv_linspace()] for a given number of evenly spaced values,
#'   [nv_iota()] for values increasing along an axis of any shape,
#'   [prim_iota()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_seq(3, 7)
#'
#' # a range that counts down needs no `by`
#' nv_seq(7, 3)
#'
#' # `end` is only reached where a whole number of steps lands on it
#' nv_seq(0, 9, by = 2)
#'
#' # a float data type gives the same values as floats
#' nv_seq(3, 7, dtype = "f32")
#'
#' # nv_seq_like() takes the data type and device from an existing array
#' x <- nv_array(c(1, 2, 3), dtype = "f64")
#' nv_seq_like(x, 1, 5)
#' @export
nv_seq <- jit(
  function(start, end, by = NULL, dtype = NULL, device = NULL) {
    dtype <- dtype %||% default_int()
    assert_int(start)
    assert_int(end)
    by <- by %||% if (start > end) -1L else 1L
    assert_int(by)
    if (by == 0) {
      cli_abort("{.arg by} must not be 0.")
    }
    if (start != end && sign(by) != sign(end - start)) {
      cli_abort(c(
        "Wrong sign in {.arg by} argument.",
        x = "Cannot go from {.val {start}} to {.val {end}} in steps of {.val {by}}."
      ))
    }
    n <- as.integer((end - start) %/% by) + 1L
    if (by == 1) {
      return(nv_iota(shape = n, dtype = dtype, axis = 1L, start = start, device = device))
    }
    # prim_iota has no step, so scale a 0-based iota; the literals are integers so
    # that they take the data type of the array instead of promoting it to float
    indices <- nv_iota(shape = n, dtype = dtype, axis = 1L, start = 0L, device = device)
    indices * as.integer(by) + as.integer(start)
  },
  static = 1:5
)

#' @title Evenly Spaced Sequence
#' @description
#' Creates a 1-D array with `steps` evenly spaced values from `start` to `end`
#' (both inclusive), like R's `seq(start, end, length.out = steps)`.
#'
#' The spacing `(end - start) / (steps - 1)` is generally not a whole number,
#' so the result is a float.
#'
#' `nv_linspace_like()` is a variant where `dtype` and `device`
#' default to those of `like`.
#' @param start,end (`numeric(1)`)\cr
#'   First and last value of the sequence. `end` may lie below `start`, in
#'   which case the values decrease.
#' @param steps (`integer(1)`)\cr
#'   Number of values to generate. Must be at least 1; for `steps = 1` the
#'   result is `start`.
#' @param dtype (`NULL` | `character(1)` | [`DataType`])\cr
#'   Data type of the result. Must be a float data type; `NULL` (default) uses
#'   the default float data type (see [`default_dtypes()`]), since
#'   the spacing is fractional. For `nv_linspace_like()`, `NULL` uses
#'   `dtype(like)`, which must then be a float too.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_linspace_like()`).
#' @template param_device
#' @return ([`arrayish`])\cr
#'   Has `dtype` and shape `steps`.
#' @seealso [nv_seq()] for consecutive integers, [nv_iota()] for values
#'   increasing along an axis of any shape, [`dtypes`] for the data type
#'   categories.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_linspace(0, 1, steps = 5L)
#'
#' # end below start counts down
#' nv_linspace(1, 0, steps = 3L)
#'
#' # steps = 1 gives start alone
#' nv_linspace(2.5, 10, steps = 1L)
#'
#' # the data type must be a float; convert afterwards for integers
#' nv_convert(nv_linspace(0, 10, steps = 5L), "i32")
#'
#' # nv_linspace_like() takes the data type and device from an existing array
#' x <- nv_array(c(1, 2, 3), dtype = "f64")
#' nv_linspace_like(x, 0, 1, steps = 3L)
#' @export
nv_linspace <- jit(
  function(start, end, steps, dtype = NULL, device = NULL) {
    assert_number(start)
    assert_number(end)
    assert_int(steps, lower = 1L)
    dtype <- assert_float_dtype(
      dtype %||% default_float(),
      arg = "dtype",
      hint = "Convert the result instead, e.g. {.code nv_convert(x, \"i32\")}."
    )
    if (steps == 1L) {
      return(nv_fill(start, 1L, dtype = dtype, device = device))
    }
    indices <- nv_iota(axis = 1L, shape = steps, dtype = dtype, start = 0L, device = device)
    indices * ((end - start) / (steps - 1L)) + start
  },
  static = 1:5
)

#' @title Pad
#' @description
#' Pads an array with a given value at the edges and optionally between elements.
#' @param x ([`arrayish`])\cr
#'   The array to pad. Can be any data type; `padding_value` is brought to it.
#' @param padding_value ([`arrayish`])\cr
#'   Scalar value to use for padding. It is
#'   brought to `x`'s data type: an R value is built at it when its category can
#'   reach it (`0L` serves an integer and a float `x` alike, `0` only a float
#'   one), and a value that already has a data type is converted unless that
#'   would narrow it -- an `f64` padding value for an `f32` `x` is an error rather than
#'   a silent narrowing.
#' @param edge_padding_low (`integer()`)\cr
#'   Amount of padding to add at the start of each axis.
#' @param edge_padding_high (`integer()`)\cr
#'   Amount of padding to add at the end of each axis.
#' @param interior_padding (`integer()` | `NULL`)\cr
#'   Amount of padding to add between elements in each axis.
#'   If `NULL` (default), no interior padding is applied.
#' @return ([`arrayish`])\cr
#'   Has `x`'s data type. Each axis grows by
#'   `edge_padding_low + edge_padding_high`, plus `interior_padding` between
#'   every pair of elements; negative edge padding trims.
#' @seealso [prim_pad()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # two zeros in front, one behind
#' x <- nv_array(c(1, 2, 3))
#' nv_pad(x, nv_scalar(0), edge_padding_low = 2L, edge_padding_high = 1L)
#' @export
nv_pad <- function(x, padding_value, edge_padding_low, edge_padding_high, interior_padding = NULL) {
  # `promotion_like("x")` rather than the primitive's own rule: crossing a
  # category is the `nv_*` layer's job, so `nv_pad(x_f32, 0L)` works here the
  # way `nv_clamp(0L, x_f32, 1L)` does, while `prim_pad()` stays strict.
  args <- as_anvl_arrays(x = x, padding_value = padding_value, .promote = promotion_like("x"))
  x <- args$x
  padding_value <- args$padding_value
  rank <- naxes(x)
  if (is.null(interior_padding)) {
    interior_padding <- rep(0L, rank)
  }
  prim_pad(x, padding_value, edge_padding_low, edge_padding_high, interior_padding)
}

#' @title Round
#' @description
#' Element-wise rounding to a whole number.
#' @template param_unary_x_round
#' @param method (`character(1)`)\cr
#'   Rounding method.
#'   Either `"nearest_even"` (default) or `"afz"` (away from zero).
#' @template return_unary
#' @seealso [prim_round()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(1.4, 2.5, 3.6))
#' nv_round(x)
#' nv_round(nv_array(1:3)) # an integer array is already whole
#' @export
nv_round <- function(x, method = "nearest_even") {
  if (is_intlike(x)) as_anvl_array(x) else prim_round(x, method = method)
}

## Other operations -----------------------------------------------------------

#' @title Matrix Multiplication
#' @description
#' Matrix multiplication of two arrays. You can also use the `%*%` operator.
#' Supports batched matrix multiplication when inputs have more than 2 axes.
#' @section Shapes:
#' - `lhs`: `(b1, ..., bk, m, n)`
#' - `rhs`: `(b1, ..., bk, n, p)`
#' - output: `(b1, ..., bk, m, p)`
#' @param lhs,rhs ([`arrayish`])\cr
#'   Numeric arrays with at least 2 axes. Can be any numeric data type; the two are
#'   [promoted to a common data type][nv_promote_to_common()]. An R value
#'   assumes the data type of the other operand, and materializes at its
#'   [default data type][default_dtypes] when that has none either.
#' @param precision (`character(1)`)\cr
#'   Controls the trade-off between speed and numerical accuracy of the
#'   operation. One of `"highest"` (default), `"high"` or `"default"`.
#'   See [prim_dot_general()] for details.
#' @return ([`arrayish`])\cr
#'   Has the operands' common data type and the shape given under Shapes.
#' @seealso [prim_dot_general()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # a 2x3 times a 3x2 gives a 2x2 at the operands' common data type
#' x <- nv_matrix(1:6, nrow = 2)
#' y <- nv_matrix(1:6, nrow = 3)
#' nv_matmul(x, y)
#' x %*% y
#' @export
nv_matmul <- jit(
  function(lhs, rhs, precision = "highest") {
    args <- promote_numeric_operands(lhs = lhs, rhs = rhs)
    lhs <- args$lhs
    rhs <- args$rhs
    if (naxes(lhs) < 2L) {
      cli_abort("{.arg lhs} must have at least 2 axes, but it has {naxes(lhs)}.")
    }
    if (naxes(rhs) < 2L) {
      cli_abort("{.arg rhs} must have at least 2 axes, but it has {naxes(rhs)}.")
    }
    # The commonest shape mistake in the package. Left to `prim_dot_general()` it
    # would be reported in terms of `contracting_axes`, which `nv_matmul()` does
    # not have.
    if (naxes(lhs) != naxes(rhs)) {
      cli_abort(c(
        "{.arg lhs} and {.arg rhs} must have the same number of axes.",
        x = "{.arg lhs} is {shape_repr(shape(lhs))} and {.arg rhs} is {shape_repr(shape(rhs))}." # nolint
      ))
    }
    inner_lhs <- shape(lhs)[naxes(lhs)]
    inner_rhs <- shape(rhs)[naxes(rhs) - 1L]
    if (inner_lhs != inner_rhs) {
      cli_abort(c(
        "{.arg lhs} and {.arg rhs} are not conformable.",
        x = "The last axis of {.arg lhs} has size {inner_lhs}, but the second-to-last axis of {.arg rhs} has size {inner_rhs}." # nolint
      ))
    }
    nbatch <- naxes(lhs) - 2L
    batch_lhs <- shape(lhs)[seq_len(nbatch)]
    batch_rhs <- shape(rhs)[seq_len(nbatch)]
    if (!identical(batch_lhs, batch_rhs)) {
      cli_abort(c(
        "{.arg lhs} and {.arg rhs} must have the same batch axes -- the axes before the last two.",
        x = "{.arg lhs} has {shape_repr(batch_lhs)} and {.arg rhs} has {shape_repr(batch_rhs)}." # nolint
      ))
    }
    prim_dot_general(
      lhs,
      rhs,
      contracting_axes = list(naxes(lhs), naxes(rhs) - 1L),
      batching_axes = list(seq_len(nbatch), seq_len(nbatch)),
      precision = precision
    )
  },
  static = "precision"
)

#' @title Cholesky Decomposition
#' @description
#' Computes the Cholesky decomposition of a symmetric positive-definite matrix.
#' Supports batched inputs: axes before the last two are batch axes.
#' @details
#' Differentiation is only implemented for a single matrix: a [gradient()] of a
#' batched decomposition errors.
#' @templateVar shapes a symmetric positive-definite matrix with at least 2 axes, the last two forming the square matrix and any leading ones batch axes
#' @template param_unary_x_tofloat
#' @param lower (`logical(1)`)\cr
#'   If `FALSE` (default, matching base R's [base::chol()]), compute the
#'   upper triangular factor `U` such that `x = t(U) %*% U`. If
#'   `TRUE`, compute the lower triangular factor `L` such that
#'   `x = L %*% t(L)`.
#' @return ([`arrayish`])\cr
#'   Triangular matrix with the input's shape, and its data type -- or the
#'   default float data type (see [`default_dtypes()`]) where the input was an
#'   integer one. The values in the triangle not selected by `lower` are
#'   implementation-defined.
#' @seealso [nv_solve()], [prim_chol()]
#' @examplesIf pjrt::plugins_downloaded()
#' # the factor has the matrix's shape and data type
#' a <- nv_matrix(c(4, 2, 2, 3), nrow = 2, dtype = "f32")
#' nv_chol(a)
#'
#' # an integer matrix is factored at the default float data type
#' nv_chol(nv_matrix(c(4L, 2L, 2L, 3L), nrow = 2))
#' @export
nv_chol <- jit(
  function(x, lower = FALSE) {
    prim_chol(as_anvl_array(int_to_float(x)), lower = lower)
  },
  static = "lower"
)

#' @title Solve Linear System
#' @description
#' Solves the linear system `a %*% x = b` for `x`. Uses LU decomposition
#' with partial pivoting internally, so `a` need only be square and
#' non-singular.
#' @details
#' \deqn{A x = b}
#' \deqn{P A = L U}
#' \deqn{L U x = P b}
#' \deqn{L y = P b}
#' \deqn{U x = y}
#' @section Shapes:
#' - `a`: `(n, n)`
#' - `b`: `(n,)` or `(n, k)`
#' - output: same shape as `b`
#'
#' @param a ([`arrayish`])\cr
#'   Square non-singular matrix with exactly 2 axes. Can be any numeric data
#'   type: `a` and `b` are
#'   [promoted to a common data type][nv_promote_to_common()] and that is then
#'   converted to the default float data type (see [`default_dtypes()`]) where
#'   it is not a float already, since the decomposition is a float one. An R
#'   value assumes the other operand's data type within its
#'   [data type category][dtypes], and settles on the default float when
#'   neither has one.
#' @param b ([`arrayish`])\cr
#'   Right-hand side, vector of length `n` or matrix with `n` rows. Promoted
#'   together with `a` -- see `a`.
#' @return ([`arrayish`])\cr
#'   The solution `x` such that `a %*% x = b`, with `b`'s shape and the
#'   operands' common data type -- or the default float data type (see
#'   [`default_dtypes()`]) where that was an integer one.
#' @seealso [nv_chol()], [nv_triangular_solve()], [prim_lu()]
#' @examplesIf pjrt::plugins_downloaded()
#' # the solution has `b`'s shape and the operands' common data type
#' a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
#' b <- nv_matrix(c(1, 2), nrow = 2, dtype = "f64")
#' nv_solve(a, b)
#'
#' # an integer system is solved at the default float data type
#' nv_solve(nv_matrix(c(3L, 1L, 1L, 2L), nrow = 2), nv_array(c(9L, 8L)))
#' @export
nv_solve <- jit(function(a, b) {
  args <- promote_to_common_float(a = a, b = b)
  a <- args$a
  b <- args$b
  a_shape <- shape(a)
  if (length(a_shape) != 2L || a_shape[1L] != a_shape[2L]) {
    cli_abort("{.arg a} must be a square 2-D matrix")
  }
  n <- a_shape[1L]
  b_shape <- shape(b)
  if (b_shape[1L] != n) {
    cli_abort("{.arg b} must have {n} rows to match {.arg a}")
  }
  if (length(b_shape) > 2L) {
    cli_abort("{.arg b} must be a vector of length {n} or a matrix with {n} rows")
  }

  factored <- prim_lu(a)
  LU <- factored$LU
  permutation <- factored$permutation

  # Apply the row permutation P encoded by `permutation`: gather rows of b
  # so that (P b)[i, ...] == b[permutation[i], ...].
  pb <- nv_select(b, axis = 1L, index = permutation)

  # Forward then back solve via nv_triangular_solve, which handles a
  # vector `b` internally by reshaping to a column matrix and back.
  y <- nv_triangular_solve(LU, pb, lower = TRUE, unit_diagonal = TRUE)
  nv_triangular_solve(LU, y, lower = FALSE)
})

#' @title Triangular Solve
#' @description
#' Solves a triangular system of linear equations. When `left_side = TRUE`,
#' returns `x` such that `op(a) %*% x = b`. When `left_side = FALSE`,
#' returns `x` such that `x %*% op(a) = b`. Here `op` is `a` or `t(a)`
#' depending on `transpose_a`.
#' @details
#' As a convenience, `b` may have one fewer axis than `a` (a single
#' right-hand side per batch, shape `(B..., n)` for `a` of shape
#' `(B..., n, n)`). It is reshaped internally to a column (`left_side =
#' TRUE`) or row (`left_side = FALSE`) and reshaped back on the way out.
#' Because we don't broadcast, this is not ambiguous (as it would be for NumPy).
#'
#' Differentiation is only implemented for a single system: a [gradient()] of a
#' batched solve errors.
#' @param a ([`arrayish`])\cr
#'   Triangular coefficient matrix with at least 2 axes. The last two
#'   axes must be equal; any leading axes are batch axes. Can be any numeric
#'   data type: `a` and `b` are
#'   [promoted to a common data type][nv_promote_to_common()] and that is then
#'   converted to the default float data type (see [`default_dtypes()`]) where
#'   it is not a float already, since the solve is a float one. An R value
#'   assumes the other operand's data type within its
#'   [data type category][dtypes], and settles on the default float when
#'   neither has one.
#' @param b ([`arrayish`])\cr
#'   Right-hand side. For `a` of shape `(B..., n, n)`, `b` may be either:
#'   * full rank — shape `(B..., n, k)` when `left_side = TRUE`, or
#'     `(B..., k, n)` when `left_side = FALSE`;
#'   * one rank less, shape `(B..., n)`, meaning a single column
#'     (`left_side = TRUE`) or row (`left_side = FALSE`) per batch — it
#'     is reshaped internally and the reshape is undone on the result so
#'     the output rank matches `b`.
#'
#'   `b`'s batch axes (`B...`) must match `a`'s exactly. It is promoted
#'   together with `a` -- see `a`.
#' @param left_side (`logical(1)`)\cr
#'   If `TRUE` (default), solve `op(a) %*% x = b`; if `FALSE`,
#'   solve `x %*% op(a) = b`.
#' @param lower (`logical(1)`)\cr
#'   Whether `a` is lower or upper triangular. Defaults to `TRUE`.
#' @param unit_diagonal (`logical(1)`)\cr
#'   If `TRUE`, the diagonal of `a` is treated as all ones (and the actual
#'   values on the diagonal are ignored). Defaults to `FALSE`.
#' @param transpose_a (`logical(1)`)\cr
#'   If `TRUE`, solve with `t(a)` in place of `a`. Defaults to `FALSE`.
#' @return ([`arrayish`])\cr
#'   The solution `x`, with `b`'s shape and the operands' common data type --
#'   or the default float data type (see [`default_dtypes()`]) where that was
#'   an integer one.
#' @seealso [nv_solve()], [nv_chol()], [prim_triangular_solve()]
#' @examplesIf pjrt::plugins_downloaded()
#' L <- nv_matrix(c(2, 1, 0, 3), nrow = 2, dtype = "f32")
#' b <- nv_matrix(c(4, 3), nrow = 2, dtype = "f32")
#' nv_triangular_solve(L, b)
#' @export
nv_triangular_solve <- jit(
  function(
    a,
    b,
    left_side = TRUE,
    lower = TRUE,
    unit_diagonal = FALSE,
    transpose_a = FALSE
  ) {
    args <- promote_to_common_float(a = a, b = b)
    a <- args$a
    b <- args$b

    a_shape <- shape(a)
    b_shape <- shape(b)
    rank_a <- length(a_shape)
    rank_b <- length(b_shape)
    if (rank_a < 2L) {
      cli_abort("{.arg a} must have at least 2 axes, got rank {rank_a}.")
    }
    if (rank_b < rank_a - 1L || rank_b > rank_a) {
      cli_abort(c(
        "{.arg b} must have rank {rank_a - 1L} or {rank_a} to match {.arg a}.",
        "x" = "Got rank {rank_b}."
      ))
    }

    # Convenience: accept a `b` whose rank is one less than `a`'s. The
    # primitive requires rank(b) == rank(a); for left_side = TRUE we append
    # a trailing 1 (column vector per batch), for left_side = FALSE we
    # insert a 1 before the last axis (row vector per batch). The shape is
    # restored on the way out. Nothing is ambiguous about the broadcast since
    # we require exact shape match for the batch axes.
    b_is_vector <- rank_b == rank_a - 1L
    if (b_is_vector) {
      n <- if (left_side) a_shape[rank_a - 1L] else a_shape[rank_a]
      if (b_shape[length(b_shape)] != n) {
        cli_abort("{.arg b} must have size {n} in its last axis to match {.arg a}")
      }
      b <- if (left_side) {
        prim_reshape(b, shape = c(b_shape, 1L))
      } else {
        prim_reshape(b, shape = c(b_shape[-length(b_shape)], 1L, n))
      }
    }

    x <- prim_triangular_solve(
      a,
      b,
      left_side = left_side,
      lower = lower,
      unit_diagonal = unit_diagonal,
      transpose_a = transpose_a
    )

    if (b_is_vector) {
      x <- prim_reshape(x, shape = b_shape)
    }
    x
  },
  static = 3:6
)

# If we took a logarithm in nv_determinant(), the pivot sign is only part of the story
# and we also have to compute the sign from taking the absolute values before the log
# Every proper swap flips the sign
lu_pivot_sign <- function(pivots, n, dt) {
  iota <- nv_seq_like(pivots, 1L, n)
  # Should be simpler after: https://github.com/r-xla/anvl/issues/343
  flips <- nv_ifelse(pivots != iota, nv_fill_like(pivots, -1L), nv_fill_like(pivots, 1L))
  nv_convert(nv_reduce_prod(flips, axes = 1L), dtype = dt)
}

#' @title Determinant
#' @description
#' Computes the determinant of a square matrix via [`nv_determinant()`].
#' @templateVar shapes a square matrix with exactly 2 axes
#' @template param_unary_x_tofloat
#' @return ([`arrayish`])\cr
#'   A scalar with the input's data type -- or the default float data type
#'   (see [`default_dtypes()`]) where the input was an integer one.
#' @seealso [nv_determinant()], [nv_solve()], [prim_lu()]
#' @examplesIf pjrt::plugins_downloaded()
#' # a scalar with the matrix's data type
#' a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
#' nv_det(a)
#' @export
nv_det <- jit(function(x) {
  d <- nv_determinant(x, logarithm = FALSE)
  prim_mul(d$sign, d$modulus)
})

#' @title Determinant in Modulus/Sign Form
#' @description
#' Computes the determinant of a square matrix in the modulus / sign
#' decomposition matching base R's [base::determinant()]. For the plain
#' scalar determinant, use [nv_det()].
#' @details
#' For computing the determinant, we use:
#' \deqn{P A = L U}
#' \deqn{\det(L) = 1}
#' \deqn{\det(A) = \det(U) / \det(P) = \mathrm{sign}(P^{-1}) \, \prod_i U_{ii}
#'   = \mathrm{sign}(P) \, \prod_i U_{ii}}
#'
#' Matching base R's `det_ge_real`, the magnitude is computed in log
#' space when `logarithm = TRUE` (\eqn{\sum_i \log|U_{ii}|}) and as a
#' direct product when `logarithm = FALSE` (\eqn{\prod_i |U_{ii}|}).
#' @templateVar shapes a square matrix with exactly 2 axes
#' @template param_unary_x_tofloat
#' @param logarithm (`logical(1)`)\cr
#'   If `TRUE` (default, matching base R), `modulus` is
#'   `log(abs(det(x)))`. If `FALSE`, `modulus` is `abs(det(x))`.
#' @return (named `list` of two [`arrayish`])\cr
#'   Elements `modulus` and `sign`, both scalar [`arrayish`] with `x`'s data
#'   type -- or the default float data type (see [`default_dtypes()`]) where
#'   `x` was an integer one. The full determinant is `sign * exp(modulus)`
#'   (with `logarithm = TRUE`) or `sign * modulus` (with
#'   `logarithm = FALSE`).
#' @seealso [nv_det()], [nv_solve()], [prim_lu()]
#' @examplesIf pjrt::plugins_downloaded()
#' # `modulus` and `sign` are scalars with the matrix's data type
#' a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
#' nv_determinant(a)
#' nv_determinant(a, logarithm = FALSE)
#' @export
nv_determinant <- jit(
  function(x, logarithm = TRUE) {
    x <- as_anvl_array(int_to_float(x))
    # Adopted from: https://github.com/wch/r-source/blob/ed837b19e0a90df72cedb007583dd4d7604aea2d/src/modules/lapack/Lapack.c#L1408-L1464
    shp <- shape(x)
    if (length(shp) != 2L || shp[[1L]] != shp[[2L]]) {
      cli_abort("{.arg x} must be a square 2-D matrix")
    }
    n <- shp[[1L]]
    dt <- dtype(x)
    # Empty matrix: det of the 0x0 matrix is the empty product = 1, so
    # log|det| = 0 and sign(det) = +1. This matches `base::determinant()`
    # and short-circuits since prim_lu rejects zero-sized inputs.
    if (n == 0L) {
      one <- nv_scalar_like(x, 1L)
      modulus <- if (logarithm) nv_scalar_like(x, 0L) else one
      return(list(modulus = modulus, sign = one))
    }
    factored <- prim_lu(x)
    LU <- factored$LU
    pivots <- factored$pivots
    diag_U <- nv_extract_diag(LU)
    pivot_sign <- lu_pivot_sign(pivots, n, dt)

    if (logarithm) {
      # log|x| discards the sign of each diagonal entry, so accumulate
      # it separately as prod(sign(diag(U))).
      diag_sign <- nv_reduce_prod(nv_sign(diag_U), axes = 1L)
      sign <- prim_mul(pivot_sign, diag_sign)
      modulus <- nv_reduce_sum(prim_log(prim_abs(diag_U)), axes = 1L)
    } else {
      # The signed product carries both magnitude and sign; split at the end.
      signed_prod <- nv_reduce_prod(diag_U, axes = 1L)
      sign <- prim_mul(pivot_sign, nv_sign(signed_prod))
      modulus <- prim_abs(signed_prod)
    }

    list(modulus = modulus, sign = sign)
  },
  static = 2L
)

#' @title Matrix Inverse
#' @description
#' Computes `x^-1`, the inverse of a square non-singular matrix `x`, by
#' solving `x %*% y = I` for `y`.
#'
#' For most use cases prefer [nv_solve()] directly: forming the explicit
#' inverse is both slower and less numerically stable than solving against
#' a right-hand side.
#' @templateVar shapes a square non-singular matrix with exactly 2 axes
#' @template param_unary_x_tofloat
#' @return ([`arrayish`])\cr
#'   The inverse, with the input's shape, and its data type -- or the default
#'   float data type (see [`default_dtypes()`]) where the input was an integer
#'   one.
#' @seealso [nv_solve()]
#' @examplesIf pjrt::plugins_downloaded()
#' # the inverse has the matrix's shape and data type
#' a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
#' nv_inv(a)
#' @export
nv_inv <- jit(function(x) {
  x <- as_anvl_array(int_to_float(x))
  shp <- shape(x)
  if (length(shp) != 2L || shp[[1L]] != shp[[2L]]) {
    cli_abort("{.arg x} must be a square 2-D matrix")
  }
  n <- shp[[1L]]
  # The inverse of the 0x0 matrix is itself; short-circuit since prim_lu
  # rejects zero-sized inputs.
  if (n == 0L) {
    return(x)
  }
  identity <- nv_eye_like(x, n, dtype = dtype(x))
  nv_solve(x, identity)
})

#' @title QR Decomposition
#' @inherit prim_qr description details
#' @templateVar shapes with exactly 2 axes
#' @template param_unary_x_tofloat
#' @return (named `list` of two [`arrayish`])\cr
#'   Elements `Q` (shape `(m, k)`) and `R` (shape `(k, n)`), where
#'   `(m, n) = shape(x)` and `k = min(m, n)`. Both have the input's data type
#'   -- or the default float data type (see [`default_dtypes()`]) where the
#'   input was an integer one.
#' @seealso [prim_qr()]
#' @examplesIf pjrt::plugins_downloaded()
#' # `Q` is 3x2 and `R` 2x2, both at the input's data type
#' x <- nv_matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, dtype = "f32")
#' nv_qr(x)
#'
#' # an integer matrix is decomposed at the default float data type
#' nv_qr(nv_matrix(1:6, nrow = 3))
#' @export
nv_qr <- jit(function(x) {
  prim_qr(as_anvl_array(int_to_float(x)))
})

#' @title LU Decomposition
#' @description
#' Computes the partial-pivoted LU decomposition of a matrix `x`:
#' \deqn{P A = L U,}
#' where \eqn{P} is a permutation matrix, \eqn{L} is unit lower
#' triangular, and \eqn{U} is upper triangular.
#'
#' This function returns `L` and `U` as separate matrices.
#' Use [`prim_lu()`] to get them in packed `LU` form.
#' @templateVar shapes with exactly 2 axes
#' @template param_unary_x_tofloat
#' @return (named `list` of [`arrayish`])\cr
#'   `L` and `U` have the input's data type -- or the default float data type
#'   (see [`default_dtypes()`]) where the input was an integer one; `pivots`
#'   and `permutation` are indices at the default integer data type (see
#'   [`default_dtypes()`]).
#'
#'   * `L` -- unit lower-triangular factor of shape `(m, k)`, where
#'     `(m, n) = shape(x)` and `k = min(m, n)`.
#'   * `U` -- upper-triangular factor of shape `(k, n)`.
#'   * `pivots` -- length `k`, at the default integer data type (see
#'     [`default_dtypes()`]). LAPACK-style sequential row swaps as returned by
#'     `getrf`.
#'   * `permutation` -- length `m`, at that same data type. A permutation
#'     vector representing \eqn{P}.
#' @seealso [prim_lu()]
#' @examplesIf pjrt::plugins_downloaded()
#' # `L` and `U` keep the input's data type; the pivots are the default integer
#' x <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
#' nv_lu(x)
#' @export
nv_lu <- jit(function(x) {
  x <- as_anvl_array(int_to_float(x))
  out <- prim_lu(x)
  LU <- out$LU
  shp <- shape(LU)
  m <- shp[[1L]]
  n <- shp[[2L]]
  k <- min(m, n)
  dt <- dtype(x)

  # L = strict lower triangle of LU (shape (m, k)) + unit diagonal.
  L_strict_full <- nv_tril(LU, diagonal = -1L) # (m, n)
  L_strict <- if (n > k) {
    L_strict_full[1:m, 1:k]
  } else {
    L_strict_full
  }
  rows <- nv_iota_like(x, axis = 1L, shape = c(m, k), dtype = "i32")
  cols <- nv_iota_like(x, axis = 2L, shape = c(m, k), dtype = "i32")
  one <- nv_fill_like(x, 1L, shape = c(m, k), dtype = dt)
  zero <- nv_fill_like(x, 0L, shape = c(m, k), dtype = dt)
  L <- L_strict + nv_ifelse(rows == cols, one, zero)

  # U = upper triangle of the first k rows of LU.
  U_full <- nv_triu(LU, diagonal = 0L) # (m, n)
  U <- if (m > k) {
    U_full[1:k, 1:n]
  } else {
    U_full
  }

  list(L = L, U = U, pivots = out$pivots, permutation = out$permutation)
})

#' @title Singular Value Decomposition
#' @inherit prim_svd description details
#' @templateVar shapes with exactly 2 axes
#' @template param_unary_x_tofloat
#' @return (named `list` of three [`arrayish`])\cr
#'   Elements `d` (length `k`), `u` (shape `(m, k)`), and `vt` (shape
#'   `(k, n)`). All have the input's data type -- or the default float data
#'   type (see [`default_dtypes()`]) where the input was an integer one.
#' @seealso [prim_svd()], [base::svd()]
#' @examplesIf pjrt::plugins_downloaded()
#' # all three outputs have the input's data type
#' x <- nv_matrix(c(1, 0, 0, 1, 0, 1), nrow = 3, dtype = "f64")
#' nv_svd(x)
#' @export
nv_svd <- jit(function(x) {
  prim_svd(as_anvl_array(int_to_float(x)))
})

#' @title Symmetric Eigendecomposition
#' @inherit prim_eigh description details
#' @templateVar shapes a symmetric square matrix with exactly 2 axes
#' @template param_unary_x_tofloat
#' @return (named `list` of two [`arrayish`])\cr
#'   Elements `values` (length `n`) and `vectors` (shape `(n, n)`). Both have
#'   the input's data type -- or the default float data type (see
#'   [`default_dtypes()`]) where the input was an integer one.
#' @seealso [prim_eigh()], [base::eigen()]
#' @examplesIf pjrt::plugins_downloaded()
#' # values and vectors both have the input's data type
#' x <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
#' nv_eigh(x)
#'
#' # an integer matrix is decomposed at the default float data type
#' nv_eigh(nv_matrix(c(2L, 1L, 1L, 2L), nrow = 2))
#' @export
nv_eigh <- jit(function(x) {
  prim_eigh(as_anvl_array(int_to_float(x)))
})

#' @title Diagonal Matrix
#' @description
#' Creates a diagonal matrix from a 1-D array.
#' @templateVar dtypes any data type
#' @templateVar shapes a 1-D array of length `n` whose elements become the diagonal entries
#' @template param_unary_x
#' @return ([`arrayish`])\cr
#'   Has the input's data type and shape `(n, n)`, with the input on the
#'   diagonal and zeros elsewhere.
#' @examplesIf pjrt::plugins_downloaded()
#' # the vector becomes the diagonal of a 3x3 matrix
#' nv_diag(nv_array(c(1, 2, 3)))
#' @export
nv_diag <- jit(function(x) {
  x <- as_anvl_array(x)
  if (naxes(x) != 1L) {
    cli_abort(c(
      "{.arg x} must be a 1-D array.",
      x = "Got shape {shape_repr(shape(x))}."
    ))
  }
  n <- shape(x)[1L]
  zeros <- nv_fill_like(x, 0L, shape = c(n, n))
  idx <- prim_reshape(nv_iota_like(x, axis = 1L, shape = n, dtype = "i32"), shape = c(n, 1L))
  indices <- nv_concatenate(idx, idx, axis = 2L)
  prim_scatter(
    zeros,
    indices,
    x,
    update_window_axes = integer(0L),
    inserted_window_axes = c(1L, 2L),
    x_batching_axes = integer(0L),
    scatter_indices_batching_axes = integer(0L),
    scatter_axes_to_x_axes = c(1L, 2L),
    index_vector_axis = 2L,
    unique_indices = TRUE
  )
})

#' @title Identity Matrix
#' @description
#' Creates an `n x n` identity matrix.
#'
#' `nv_eye_like()` is a variant where `dtype` and `device` default to those of
#' `like`.
#' @param n (`integer(1)`)\cr
#'   Size of the identity matrix.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_eye_like()`).
#' @param dtype (`NULL` | `character(1)` | [`DataType`])\cr
#'   Data type of the result. Can be any data type; `NULL` (default) uses the
#'   default float data type (see [`default_dtypes()`]). For
#'   `nv_eye_like()`, `NULL` uses `dtype(like)`.
#' @template param_device
#' @return ([`arrayish`])\cr
#'   Has the given `dtype` and shape `(n, n)`: ones on the diagonal, zeros
#'   elsewhere.
#' @seealso [nv_diag()] for general diagonal matrices.
#' @examplesIf pjrt::plugins_downloaded()
#' # a 3x3 identity matrix
#' nv_eye(3L)
#'
#' # `_like` takes the data type and device from an existing array
#' x <- nv_fill(0, shape = c(3, 3), dtype = "f64")
#' nv_eye_like(x, 3L)
#' @export
nv_eye <- jit(
  function(n, dtype = NULL, device = NULL) {
    assert_int(n, lower = 0L)
    dtype <- dtype %||% default_float()
    nv_diag(nv_fill(1L, as.integer(n), dtype = dtype, device = device))
  },
  static = 1:3
)

# Expand `axes = NULL` to "all axes". Negative axes are resolved here
# (not just in the primitive) because `nv_mean()` / `nv_var()` / `nv_sd()`
# index `shape(x)[axes]` to compute the number of reduced elements.
.resolve_reduce_axes <- function(x, axes) {
  if (is.null(axes)) {
    return(seq_len(naxes(x)))
  }
  resolve_axes(axes, naxes(x), arg = "axes", unique = TRUE)
}

.count_bool <- function(x) {
  if (is_dtype_bool(peek_dtype(x))) nv_convert(x, default_int()) else x
}

# Gather `axes` into a single trailing axis, so an operation that only ever
# handles one axis -- a sort, an arg-reduction -- can reduce several at once by
# ranking their elements together. `axes` must already be resolved and sorted.
# The reshape is column-major, so a position along the merged axis is the
# column-major linear index within the block of `axes`, like `which.max()`
# reports for a whole array. Returns the reshaped array along with the shape
# its kept axes take afterwards, which `drop = FALSE` restores the reduced axes
# into at size 1.
.flatten_reduce_axes <- function(x, axes, drop) {
  x_shape <- shape(x)
  keep <- setdiff(seq_along(x_shape), axes)
  permutation <- c(keep, axes)
  if (!identical(permutation, seq_along(x_shape))) {
    x <- prim_transpose(x, permutation = permutation)
  }
  flat_shape <- c(x_shape[keep], as.integer(prod(x_shape[axes])))
  if (!identical(shape(x), flat_shape)) {
    x <- prim_reshape(x, flat_shape)
  }
  list(
    x = x,
    keep_shape = if (drop) x_shape[keep] else replace(x_shape, axes, 1L)
  )
}

# Resolve the `axis` of a cumulative op. `NULL` accumulates over every element,
# like base R's `cum*()` functions, which means flattening the input first (in
# column-major order, as they do) -- so this hands back the array as well as
# the axis.
.resolve_cum_input <- function(x, axis) {
  if (is.null(axis)) {
    list(x = nv_reshape(x, prod(shape(x))), axis = 1L)
  } else {
    list(x = x, axis = axis)
  }
}

#' @title Sum Reduction
#' @description
#' Sums array elements along the specified axes.
#' A boolean array is counted, like [base::sum()] does.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @templateVar axes_all If `NULL` (default), reduces over all axes.
#' @template params_reduce
#' @templateVar dtype_out the input's data type, except a boolean input, which is accumulated at the default integer data type (see [`default_dtypes()`])
#' @template return_reduce
#' @template param_nan_rm
#' @seealso [prim_reduce_sum()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' # no axes given: reduce over all of them
#' nv_reduce_sum(x)
#'
#' # reducing axis 1 removes it, drop = FALSE keeps it at size 1
#' nv_reduce_sum(x, axes = 1L)
#' nv_reduce_sum(x, axes = 1L, drop = FALSE)
#'
#' # negative axes count from the end
#' nv_reduce_sum(x, axes = -1L)
#'
#' # NaN propagates unless nan_rm = TRUE
#' nv_reduce_sum(nv_array(c(1, NaN, 3)))
#' nv_reduce_sum(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#' nv_reduce_sum(nv_array(c(TRUE, FALSE, TRUE))) # counts: 2
#' @export
nv_reduce_sum <- jit(
  function(x, axes = NULL, drop = TRUE, nan_rm = FALSE) {
    assert_flag(nan_rm)
    x <- .count_bool(as_anvl_array(x))
    axes <- .resolve_reduce_axes(x, axes)
    if (nan_rm && is_dtype_float(peek_dtype(x))) {
      x <- nv_ifelse(nv_is_nan(x), 0L, x)
    }
    prim_reduce_sum(x, axes = axes, drop = drop)
  },
  static = 2:4
)

#' @title Mean
#' @description
#' Computes the arithmetic mean along the specified axes. You can also
#' use `mean()`.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @templateVar axes_all If `NULL` (default), reduces over all axes.
#' @template params_reduce
#' @templateVar dtype_out the input's data type where that is a float, and the default float data type (see [`default_dtypes()`]) otherwise
#' @template return_reduce
#' @template param_nan_rm
#' @seealso [nv_reduce_sum()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' # an integer input is averaged at the default float data type
#' nv_mean(x)
#'
#' # a float input keeps its own, whatever the default float is
#' nv_mean(nv_array(c(1, 2), dtype = "f64"))
#'
#' # reducing axis 1 removes it, drop = FALSE keeps it at size 1
#' nv_mean(x, axes = 1L)
#' nv_mean(x, axes = 1L, drop = FALSE)
#'
#' # NaN propagates unless nan_rm = TRUE
#' nv_mean(nv_array(c(1, NaN, 3)))
#' nv_mean(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#' @export
nv_mean <- jit(
  function(x, axes = NULL, drop = TRUE, nan_rm = FALSE) {
    assert_flag(nan_rm)
    x <- as_anvl_array(x)
    axes <- .resolve_reduce_axes(x, axes)
    if (nan_rm && is_dtype_float(peek_dtype(x))) {
      is_nan <- nv_is_nan(x)
      total <- prim_reduce_sum(nv_ifelse(is_nan, 0L, x), axes = axes, drop = drop)
      count <- prim_reduce_sum(nv_convert(!is_nan, "i32"), axes = axes, drop = drop)
      return(total / count)
    }
    nelts <- prod(shape(x)[axes])
    nv_reduce_sum(x, axes, drop) / nelts
  },
  static = 2:4
)

#' @title Product Reduction
#' @description
#' Multiplies array elements along the specified axes.
#' A boolean array is multiplied as zeroes and ones, like [base::prod()] does.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @templateVar axes_all If `NULL` (default), reduces over all axes.
#' @template params_reduce
#' @templateVar dtype_out the input's data type, except a boolean input, which is accumulated at the default integer data type (see [`default_dtypes()`])
#' @template return_reduce
#' @template param_nan_rm
#' @seealso [prim_reduce_prod()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' # no axes given: reduce over all of them
#' nv_reduce_prod(x)
#'
#' # reducing axis 1 removes it, drop = FALSE keeps it at size 1
#' nv_reduce_prod(x, axes = 1L)
#' nv_reduce_prod(x, axes = 1L, drop = FALSE)
#'
#' # negative axes count from the end
#' nv_reduce_prod(x, axes = -1L)
#'
#' # NaN propagates unless nan_rm = TRUE
#' nv_reduce_prod(nv_array(c(2, NaN, 3)))
#' nv_reduce_prod(nv_array(c(2, NaN, 3)), nan_rm = TRUE)
#' @export
nv_reduce_prod <- jit(
  function(x, axes = NULL, drop = TRUE, nan_rm = FALSE) {
    assert_flag(nan_rm)
    x <- .count_bool(as_anvl_array(x))
    axes <- .resolve_reduce_axes(x, axes)
    if (nan_rm && is_dtype_float(peek_dtype(x))) {
      x <- nv_ifelse(nv_is_nan(x), 1L, x)
    }
    prim_reduce_prod(x, axes = axes, drop = drop)
  },
  static = 2:4
)

#' @title Max Reduction
#' @description
#' Finds the maximum of array elements along the specified axes.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @templateVar axes_all If `NULL` (default), reduces over all axes.
#' @template params_reduce
#' @template param_nan_rm
#' @templateVar dtype_out the input's data type
#' @template return_reduce
#' @seealso [prim_reduce_max()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' # no axes given: reduce over all of them
#' nv_reduce_max(x)
#'
#' # reducing axis 1 removes it, drop = FALSE keeps it at size 1
#' nv_reduce_max(x, axes = 1L)
#' nv_reduce_max(x, axes = 1L, drop = FALSE)
#'
#' # negative axes count from the end
#' nv_reduce_max(x, axes = -1L)
#'
#' # NaN propagates unless nan_rm = TRUE
#' nv_reduce_max(nv_array(c(1, NaN, 3)))
#' nv_reduce_max(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#' @export
nv_reduce_max <- jit(
  function(x, axes = NULL, drop = TRUE, nan_rm = FALSE) {
    assert_flag(nan_rm)
    x <- as_anvl_array(x)
    axes <- .resolve_reduce_axes(x, axes)
    .nv_reduce_extreme(x, axes, drop, nan_rm, -Inf, prim_reduce_max)
  },
  static = 2:4
)

#' @title Min Reduction
#' @description
#' Finds the minimum of array elements along the specified axes.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @templateVar axes_all If `NULL` (default), reduces over all axes.
#' @template params_reduce
#' @template param_nan_rm
#' @templateVar dtype_out the input's data type
#' @template return_reduce
#' @seealso [prim_reduce_min()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' # no axes given: reduce over all of them
#' nv_reduce_min(x)
#'
#' # reducing axis 1 removes it, drop = FALSE keeps it at size 1
#' nv_reduce_min(x, axes = 1L)
#' nv_reduce_min(x, axes = 1L, drop = FALSE)
#'
#' # negative axes count from the end
#' nv_reduce_min(x, axes = -1L)
#'
#' # NaN propagates unless nan_rm = TRUE
#' nv_reduce_min(nv_array(c(1, NaN, 3)))
#' nv_reduce_min(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#' @export
nv_reduce_min <- jit(
  function(x, axes = NULL, drop = TRUE, nan_rm = FALSE) {
    assert_flag(nan_rm)
    x <- as_anvl_array(x)
    axes <- .resolve_reduce_axes(x, axes)
    .nv_reduce_extreme(x, axes, drop, nan_rm, Inf, prim_reduce_min)
  },
  static = 2:4
)

# Shared NaN-aware reduction for max/min. Unlike sum/prod (which lower to
# arithmetic that propagates NaN), the min/max kernels are comparison-based
# and silently drop NaN, so `nan_rm = FALSE` needs an explicit any-NaN mask
# to re-inject NaN — no input substitution can coax the kernel into emitting
# NaN on output.
.nv_reduce_extreme <- function(x, axes, drop, nan_rm, identity_val, prim_reduce) {
  if (!is_dtype_float(peek_dtype(x))) {
    return(prim_reduce(x, axes = axes, drop = drop))
  }
  is_nan <- nv_is_nan(x)
  if (nan_rm) {
    x <- nv_ifelse(is_nan, identity_val, x)
    return(prim_reduce(x, axes = axes, drop = drop))
  }
  any_nan <- prim_reduce_any(is_nan, axes = axes, drop = drop)
  result <- prim_reduce(x, axes = axes, drop = drop)
  nv_ifelse(any_nan, NaN, result)
}

#' @title Range Reduction
#' @description
#' The smallest and the largest element along the specified axes, stacked along
#' a new first axis. You can also use the `range()` generic.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @param axes (`integer()` | `NULL`)\cr
#'   Axes to reduce over. `NULL` (default) reduces over all of them, which
#'   makes the result a length-2 array like [base::range()]. Negative values
#'   count from the end.
#' @template param_nan_rm
#' @return ([`arrayish`])\cr
#'   Has the same data type as `x` and the shape of the reduced array with a
#'   leading axis of size 2 added: element 1 is the minimum, element 2 the
#'   maximum.
#' @seealso [nv_reduce_min()], [nv_reduce_max()]
#' @examplesIf pjrt::plugins_downloaded()
#' nv_range(nv_array(c(3, 1, 4)))
#' nv_range(nv_matrix(1:6, nrow = 2), axes = 1L)
#' @export
nv_range <- jit(
  function(x, axes = NULL, nan_rm = FALSE) {
    stack_min_max(
      nv_reduce_min(x, axes = axes, nan_rm = nan_rm),
      nv_reduce_max(x, axes = axes, nan_rm = nan_rm)
    )
  },
  static = 2:3
)

# The minimum and the maximum next to each other, which is what base R's
# range() returns. A new leading axis keeps the reduced axes intact, so a
# scalar minimum and maximum become a length-2 array.
stack_min_max <- function(lo, hi) {
  nv_concatenate(nv_unsqueeze(lo, 1L), nv_unsqueeze(hi, 1L), axis = 1L)
}

#' @title Any Reduction
#' @description
#' Performs logical OR along the specified axes.
#' Returns `TRUE` if any element is `TRUE`.
#' @templateVar dtypes a boolean or an R logical
#' @template param_unary_x_must
#' @templateVar axes_all If `NULL` (default), reduces over all axes.
#' @template params_reduce
#' @templateVar dtype_out the boolean data type
#' @template return_reduce
#' @seealso [prim_reduce_any()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
#' # no axes given: reduce over all of them
#' nv_reduce_any(x)
#'
#' # reducing axis 1 removes it, drop = FALSE keeps it at size 1
#' nv_reduce_any(x, axes = 1L)
#' nv_reduce_any(x, axes = 1L, drop = FALSE)
#' @export
nv_reduce_any <- jit(
  function(x, axes = NULL, drop = TRUE) {
    x <- as_anvl_array(x)
    prim_reduce_any(x, axes = .resolve_reduce_axes(x, axes), drop = drop)
  },
  static = 2:3
)

#' @title All Reduction
#' @description
#' Performs logical AND along the specified axes.
#' Returns `TRUE` only if all elements are `TRUE`.
#' @templateVar dtypes a boolean or an R logical
#' @template param_unary_x_must
#' @templateVar axes_all If `NULL` (default), reduces over all axes.
#' @template params_reduce
#' @templateVar dtype_out the boolean data type
#' @template return_reduce
#' @seealso [prim_reduce_all()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
#' # no axes given: reduce over all of them
#' nv_reduce_all(x)
#'
#' # reducing axis 1 removes it, drop = FALSE keeps it at size 1
#' nv_reduce_all(x, axes = 1L)
#' nv_reduce_all(x, axes = 1L, drop = FALSE)
#' @export
nv_reduce_all <- jit(
  function(x, axes = NULL, drop = TRUE) {
    x <- as_anvl_array(x)
    prim_reduce_all(x, axes = .resolve_reduce_axes(x, axes), drop = drop)
  },
  static = 2:3
)

#' @title Cumulative Sum
#' @description
#' Cumulative sum, optionally along a single axis.
#' A boolean array is counted, like [base::cumsum()] does.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @templateVar cum_base_fn cumsum
#' @template param_nv_cum_axis
#' @template param_nan_rm_cum
#' @template return_cum_accumulate
#' @seealso [prim_cumsum()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' nv_cumsum(x)              # flatten, then accumulate
#' nv_cumsum(x, axis = 1L)    # accumulate along rows
#' nv_cumsum(nv_array(c(1, NaN, 3)))                # NaN propagates
#' nv_cumsum(nv_array(c(1, NaN, 3)), nan_rm = TRUE) # NaN treated as 0
#' @export
nv_cumsum <- jit(
  function(x, axis = NULL, nan_rm = FALSE) {
    assert_flag(nan_rm)
    cum <- .resolve_cum_input(.count_bool(as_anvl_array(x)), axis)
    x <- cum$x
    axis <- cum$axis
    if (nan_rm && is_dtype_float(peek_dtype(x))) {
      x <- nv_ifelse(nv_is_nan(x), 0L, x)
    }
    prim_cumsum(x, axis = axis)
  },
  static = 2:3
)

#' @title Cumulative Product
#' @description
#' Cumulative product, optionally along a single axis.
#' A boolean array is multiplied as zeroes and ones, like [base::cumprod()] does.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @templateVar cum_base_fn cumprod
#' @template param_nv_cum_axis
#' @template param_nan_rm_cum
#' @template return_cum_accumulate
#' @seealso [prim_cumprod()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' nv_cumprod(x)              # flatten, then accumulate
#' nv_cumprod(x, axis = 1L)    # accumulate along rows
#' nv_cumprod(nv_array(c(2, NaN, 3)))                # NaN propagates
#' nv_cumprod(nv_array(c(2, NaN, 3)), nan_rm = TRUE) # NaN treated as 1
#' @export
nv_cumprod <- jit(
  function(x, axis = NULL, nan_rm = FALSE) {
    assert_flag(nan_rm)
    cum <- .resolve_cum_input(.count_bool(as_anvl_array(x)), axis)
    x <- cum$x
    axis <- cum$axis
    if (nan_rm && is_dtype_float(peek_dtype(x))) {
      x <- nv_ifelse(nv_is_nan(x), 1L, x)
    }
    prim_cumprod(x, axis = axis)
  },
  static = 2:3
)

#' @title Cumulative Maximum
#' @description
#' Running maximum, optionally along a single axis.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @templateVar cum_base_fn cummax
#' @template param_nv_cum_axis
#' @templateVar cum_extreme_name maximum
#' @template param_nv_cum_indices
#' @template return_nv_cum_extreme
#' @template param_nan_rm_cum
#' @seealso [prim_cummax()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the running maximum keeps the data type; the indices are the default integer
#' x <- nv_matrix(c(3, 1, 4, 1, 5, 9), nrow = 2)
#' nv_cummax(x)
#' nv_cummax(x, axis = 1L)
#' nv_cummax(x, axis = 1L, indices = TRUE)
#' nv_cummax(nv_array(c(1, NaN, 3)))                # NaN propagates
#' nv_cummax(nv_array(c(1, NaN, 3)), nan_rm = TRUE) # NaN skipped
#' @export
nv_cummax <- jit(
  function(x, axis = NULL, indices = FALSE, nan_rm = FALSE) {
    assert_flag(indices)
    assert_flag(nan_rm)
    .nv_cum_extreme(x, axis, indices, nan_rm, -Inf, prim_cummax)
  },
  static = 2:4
)

#' @title Cumulative Minimum
#' @description
#' Running minimum, optionally along a single axis.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @templateVar cum_base_fn cummin
#' @template param_nv_cum_axis
#' @templateVar cum_extreme_name minimum
#' @template param_nv_cum_indices
#' @template return_nv_cum_extreme
#' @template param_nan_rm_cum
#' @seealso [prim_cummin()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the running minimum keeps the data type; the indices are the default integer
#' x <- nv_matrix(c(3, 1, 4, 1, 5, 9), nrow = 2)
#' nv_cummin(x)
#' nv_cummin(x, axis = 1L)
#' nv_cummin(x, axis = 1L, indices = TRUE)
#' nv_cummin(nv_array(c(3, NaN, 1)))                # NaN propagates
#' nv_cummin(nv_array(c(3, NaN, 1)), nan_rm = TRUE) # NaN skipped
#' @export
nv_cummin <- jit(
  function(x, axis = NULL, indices = FALSE, nan_rm = FALSE) {
    assert_flag(indices)
    assert_flag(nan_rm)
    .nv_cum_extreme(x, axis, indices, nan_rm, Inf, prim_cummin)
  },
  static = 2:4
)

# NaN propagation for the default `nan_rm = FALSE` path is now handled in
# `prim_cummax` / `prim_cummin`'s lowering directly. Here we only need to
# sanitize NaN → identity for `nan_rm = TRUE`.
.nv_cum_extreme <- function(x, axis, indices, nan_rm, identity_val, prim_cum) {
  cum <- .resolve_cum_input(as_anvl_array(x), axis)
  x <- cum$x
  axis <- cum$axis
  if (nan_rm && is_dtype_float(peek_dtype(x))) {
    x <- nv_ifelse(nv_is_nan(x), identity_val, x)
  }
  out <- prim_cum(x, axis = axis)
  if (indices) out else out$values
}

# Higher order primitives

#' @title Conditional Branching
#' @description
#' Conditional execution of two branches.
#' Unlike [nv_ifelse()], which selects elements, this executes only one
#' of the two branches depending on a scalar predicate.
#' @param pred ([`arrayish`])\cr
#'   Predicate. Must be a scalar of the boolean data type, or an R logical.
#' @param true (`function()`)\cr
#'   Zero-argument function for the true branch.
#' @param false (`function()`)\cr
#'   Zero-argument function for the false branch.
#'   Must return the same structure, data types and shapes as the true
#'   branch; nothing is promoted.
#' @return ([`arrayish`] | `list`)\cr
#'   Result of the executed branch: an array, or a tree of them in the sense
#'   of pjrt's [`RTree`][pjrt::build_tree] -- a `list`, nested arbitrarily -- with
#'   the structure, data types and shapes both branches share.
#' @seealso [prim_if()] for the underlying primitive, [nv_ifelse()] for
#'   element-wise selection.
#' @examplesIf pjrt::plugins_downloaded()
#' # both branches must return the same structure, data types and shapes
#' nv_if(nv_scalar(TRUE), \() nv_scalar(1), \() nv_scalar(2))
#' @export
nv_if <- prim_if

#' @title While Loop
#' @description
#' Executes a functional while loop.
#' @template param_while_init
#' @param cond (`function`)\cr
#'   Condition function returning a scalar boolean.
#'   Receives the state values as arguments.
#' @param body (`function`)\cr
#'   Body function returning the updated state as a named list with the same
#'   structure, data types and shapes as `init`. Nothing is promoted: a
#'   loop-carried state is meant to be heterogeneous, so each member keeps its
#'   own data type across iterations.
#' @return (named `list`)\cr
#'   A tree of the loop-carried arrays -- see [`RTree`][pjrt::build_tree] -- in its
#'   final state after the loop terminates, with `init`'s structure, data
#'   types and shapes.
#' @seealso [prim_while()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # the loop state is a named list, and each member keeps its data type
#' nv_while(
#'   init = list(i = nv_scalar(0L), total = nv_scalar(0L)),
#'   cond = function(i, total) i < 5L,
#'   body = function(i, total) list(
#'     i = i + 1L,
#'     total = total + i
#'   )
#' )
#' @export
nv_while <- prim_while

#' @title Scan (Loop With Per-Step Outputs)
#' @description
#' Runs a fixed-length loop that threads a carry through `body` while
#' stacking each step's output into preallocated buffers.
#'
#' At step `t`, `body` receives the current carry and the step's slice of
#' `xs` (taken along axis 1, with that unit axis dropped; a 1-D leaf
#' yields a scalar), and must return
#' `list(carry = <same structure as init>, out = <arrays to stack>)`.
#' The stacked `out` buffers gain a new leading axis of size `length`.
#'
#' The whole loop, written out in R:
#'
#' ```r
#' carry <- init
#' out <- <empty, `length` rows>
#' steps <- if (reverse) rev(seq_len(length)) else seq_len(length)
#' for (t in steps) {
#'   step <- body(carry, xs[t, ...])  # `x` is NULL when `xs` is empty
#'   carry <- step$carry
#'   out[t, ...] <- step$out          # position t, not the loop's position
#' }
#' list(carry = carry, out = out)
#' ```
#'
#' @param init ([`arrayish`] | `list()`)\cr
#'   Initial carry: a single array or a (possibly nested) named list.
#'   Every slot must keep a fixed shape and dtype across steps.
#' @param body (`function`)\cr
#'   Step function `function(carry, x)` returning
#'   `list(carry = , out = )`. `out` may be a single array, a (nested)
#'   list of arrays, or `NULL` (loop for the carry only). Its structure
#'   must be identical at every step. `x` is `NULL` when `xs` is empty.
#' @param xs ([`arrayish`] | `list()` | `NULL`)\cr
#'   Per-step inputs, sliced along axis 1. All leaves must agree on
#'   the size of axis 1. `NULL` or a list with no leaves runs a counted
#'   loop over `length` steps instead.
#' @param length (`integer(1)` | `NULL`)\cr
#'   Static trip count. Required when `xs` is empty; otherwise inferred
#'   from (and checked against) axis 1 of `xs`. A trip count of `0` runs
#'   no step.
#' @param reverse (`logical(1)`)\cr
#'   If `TRUE`, steps run `t = length, ..., 1`; each step still reads
#'   `xs` at position `t` and writes its output at position `t`, so a
#'   reverse scan consumes and produces arrays in the original order.
#' @return `list(carry = , out = )`: the final carry (same structure as
#'   `init`) and the stacked outputs (structure of `body`'s `out`, each
#'   leaf gaining a leading axis of size `length`).
#' @seealso [prim_scan()], [nv_while()], [nv_cumsum()] for fixed associative scans.
#' @examplesIf pjrt::plugins_downloaded()
#' # cumulative sum along axis 1
#' x <- nv_array(c(1, 2, 3, 4))
#' nv_scan(
#'   init = nv_scalar(0),
#'   body = function(carry, x) list(carry = carry + x, out = carry + x),
#'   xs = x
#' )$out
#' @export
nv_scan <- function(init, body, xs = NULL, length = NULL, reverse = FALSE) {
  init <- map_tree(init, as_anvl_array)
  xs <- if (is.null(xs)) list() else map_tree(xs, as_anvl_array)

  # The trip count is the one thing this layer settles: `prim_scan()` needs it
  # stated, here it may be read off `xs` instead. The rest of the contract --
  # `body`, `reverse`, `length` itself, every leaf of `xs` against the trip
  # count -- is `prim_scan()`'s and is left to it.
  if (is.null(length)) {
    xs_flat <- flatten(xs)
    if (!base::length(xs_flat)) {
      cli_abort("{.arg length} is required when {.arg xs} is empty")
    }
    # Reading axis 1 needs there to be one; the other leaves are checked
    # against the count that comes out of this one.
    s <- shape(xs_flat[[1L]])
    if (!base::length(s)) {
      cli_abort("every array in {.arg xs} must have at least one axis.")
    }
    length <- as.integer(s[[1L]])
  }

  prim_scan(init, xs, body, length = length, reverse = reverse)
}

## Additional math functions ---------------------------------------------------

#' @title Base-2 Logarithm
#' @description
#' Element-wise base-2 logarithm. You can also use `log2()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [nv_log()], [nv_log10()]
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(1, 2, 4, 8))
#' nv_log2(x)
#'
#' # an R value materializes at its default data type
#' nv_log2(8)
#' @export
nv_log2 <- jit(function(x) {
  x <- as_anvl_array(x)
  nv_log(x) / log(2)
})

#' @title Base-10 Logarithm
#' @description
#' Element-wise base-10 logarithm. You can also use `log10()`.
#' @template param_unary_x_tofloat
#' @template return_unary_tofloat
#' @seealso [nv_log()], [nv_log2()]
#' @examplesIf pjrt::plugins_downloaded()
#' # the input's data type carries through
#' x <- nv_array(c(1, 10, 100, 1000))
#' nv_log10(x)
#'
#' # an R value materializes at its default data type
#' nv_log10(100)
#' @export
nv_log10 <- jit(function(x) {
  x <- as_anvl_array(x)
  nv_log(x) / log(10)
})

#' @title Is NaN
#' @description
#' Element-wise check if values are NaN. You can also use `is.nan()`.
#' Only a float holds a NaN, so the answer for any other data type is all
#' `FALSE` and is built as a constant rather than computed.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @template return_unary_boolean
#' @seealso [nv_is_finite()], [nv_is_infinite()]
#' @examplesIf pjrt::plugins_downloaded()
#' # a boolean result, whatever the input's data type
#' x <- nv_array(c(1, NaN, Inf, -Inf, 0))
#' nv_is_nan(x)
#'
#' # all FALSE for an integer input, which has no NaN to find
#' nv_is_nan(nv_array(1:3))
#' @export
nv_is_nan <- jit(function(x) {
  x <- as_anvl_array(x)
  if (!is_dtype_float(dtype(x))) {
    return(nv_fill_like(x, FALSE, dtype = "bool"))
  }
  x != x
})

#' @title Is Infinite
#' @description
#' Element-wise check if values are infinite (`Inf` or `-Inf`).
#' You can also use `is.infinite()`. Only a float holds an infinity, so the
#' answer for any other data type is all `FALSE` and is built as a constant
#' rather than computed.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @template return_unary_boolean
#' @seealso [nv_is_finite()], [nv_is_nan()]
#' @examplesIf pjrt::plugins_downloaded()
#' # the result is boolean, whatever float data type the input has
#' x <- nv_array(c(1, NaN, Inf, -Inf, 0))
#' nv_is_infinite(x)
#'
#' # all FALSE for an integer input, which has no infinity
#' nv_is_infinite(nv_array(1:3))
#'
#' # an R value materializes at its default data type before the test
#' nv_is_infinite(1)
#' @export
nv_is_infinite <- jit(function(x) {
  x <- as_anvl_array(x)
  if (!is_dtype_float(dtype(x))) {
    return(nv_fill_like(x, FALSE, dtype = "bool"))
  }
  !prim_is_finite(x) & (x == x)
})

## Reduction operations --------------------------------------------------------

#' @title Variance
#' @description
#' Computes the variance along the specified axes.
#' @details
#' Uses Bessel's correction by default (`correction = 1`), matching R's [var()].
#' Set `correction = 0` for population variance.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @templateVar axes_all If `NULL` (default), reduces over all axes.
#' @template params_reduce
#' @param correction (`integer(1)`)\cr
#'   Degrees of freedom correction. Default is `1` (Bessel's correction).
#' @template param_nan_rm
#' @templateVar dtype_out the input's data type where that is a float, and the default float data type (see [`default_dtypes()`]) otherwise
#' @template return_reduce
#' @seealso [nv_sd()], [nv_mean()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:5)
#' # the result is a float, even though the input is an integer
#' nv_var(x)
#'
#' # Bessel's correction by default, correction = 0 for the population variance
#' nv_var(x, correction = 0L)
#'
#' # NaN propagates unless nan_rm = TRUE
#' nv_var(nv_array(c(1, NaN, 3, 5)))
#' nv_var(nv_array(c(1, NaN, 3, 5)), nan_rm = TRUE)
#' @export
nv_var <- jit(
  function(x, axes = NULL, drop = TRUE, correction = 1L, nan_rm = FALSE) {
    assert_flag(nan_rm)
    x <- as_anvl_array(x)
    correction <- assert_int(correction, coerce = TRUE)
    axes <- .resolve_reduce_axes(x, axes)
    mean_bc <- nv_broadcast_to(
      nv_mean(x, axes, drop = FALSE, nan_rm = nan_rm),
      shape(x)
    )
    diff <- x - mean_bc
    ssum <- nv_reduce_sum(diff * diff, axes, drop, nan_rm = nan_rm)
    if (nan_rm && is_dtype_float(peek_dtype(x))) {
      # Counted at `ssum`'s data type, not at an integer one: the divisor then
      # stays there, because `0` and `correction` are R values meeting a float
      # array of their own or a narrower category and so yield to it. Counting
      # into an integer instead would make the R double `0` cross categories and
      # pull the result to the default float, so `nan_rm` alone would change the
      # data type -- which is why the `nan_rm = FALSE` branch below is right.
      count <- nv_reduce_sum(nv_convert(!nv_is_nan(x), dtype(ssum)), axes, drop)
      # When count <= correction the divisor clamps to 0 and ssum is 0
      # (single non-NaN point has zero deviation, all-NaN slice contributes
      # nothing), so 0/0 = NaN propagates naturally — no explicit mask needed.
      return(ssum / nv_max(0L, count - correction))
    }
    nelts <- prod(shape(x)[axes])
    ssum / max(0L, nelts - correction)
  },
  static = 2:5
)

#' @title Standard Deviation
#' @description
#' Computes the standard deviation along the specified axes.
#' @details
#' Uses Bessel's correction by default (`correction = 1`), matching R's [sd()].
#' Set `correction = 0` for population standard deviation.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @templateVar axes_all If `NULL` (default), reduces over all axes.
#' @template params_reduce
#' @param correction (`integer(1)`)\cr
#'   Degrees of freedom correction. Default is `1` (Bessel's correction).
#' @template param_nan_rm
#' @templateVar dtype_out the input's data type where that is a float, and the default float data type (see [`default_dtypes()`]) otherwise
#' @template return_reduce
#' @seealso [nv_var()], [nv_mean()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:5)
#' # the result is a float, even though the input is an integer
#' nv_sd(x)
#'
#' # Bessel's correction by default, correction = 0 for the population value
#' nv_sd(x, correction = 0L)
#' @export
nv_sd <- jit(
  function(x, axes = NULL, drop = TRUE, correction = 1L, nan_rm = FALSE) {
    assert_flag(nan_rm)
    nv_sqrt(nv_var(x, axes, drop, correction, nan_rm = nan_rm))
  },
  static = 2:5
)

## Array manipulation ----------------------------------------------------------

#' @title Squeeze
#' @description
#' Removes axes of size 1 from an array.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @param axes (`integer()` | `NULL`)\cr
#'   Axes to squeeze. Negative values count from the end, i.e. `-1`
#'   refers to the last axis.
#'   If `NULL` (default), all axes of size 1 are removed.
#' @return ([`arrayish`])\cr
#'   Has `x`'s data type, with the specified axes removed from its shape.
#' @seealso [nv_unsqueeze()], [nv_reshape()]
#' @examplesIf pjrt::plugins_downloaded()
#' # the two size-1 axes are dropped
#' x <- nv_array(1:6, shape = c(1, 6, 1))
#' nv_squeeze(x)
#' @export
nv_squeeze <- function(x, axes = NULL) {
  x <- as_anvl_array(x)
  shp <- shape(x)
  if (is.null(axes)) {
    new_shape <- shp[shp != 1L]
  } else {
    axes <- resolve_axes(axes, length(shp), unique = TRUE)
    for (d in axes) {
      if (shp[d] != 1L) {
        cli_abort("Cannot squeeze axis {d}: its size is {shp[d]}, but must be 1")
      }
    }
    new_shape <- shp[-axes]
  }
  if (length(new_shape) == 0L) {
    new_shape <- integer(0L)
  }
  nv_reshape(x, new_shape)
}

#' @title Unsqueeze
#' @description
#' Inserts an axis of size 1 at the specified position.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @param axis (`integer(1)`)\cr
#'   Position at which to insert the new axis. Valid positions range from
#'   1 to `naxes(x) + 1`. Negative values count from the end of the
#'   *result*, i.e. `-1` appends the new axis at the end.
#' @return ([`arrayish`])\cr
#'   Has `x`'s data type, with an extra axis of size 1 in its shape.
#' @seealso [nv_squeeze()], [nv_reshape()]
#' @examplesIf pjrt::plugins_downloaded()
#' # a size-1 axis is inserted, at the front or at the back
#' x <- nv_array(c(1, 2, 3))
#' nv_unsqueeze(x, axis = 1L)
#' nv_unsqueeze(x, axis = -1L)
#' @export
nv_unsqueeze <- function(x, axis) {
  x <- as_anvl_array(x)
  shp <- shape(x)
  axis <- resolve_axis(axis, length(shp) + 1L)
  new_shape <- append(shp, 1L, after = axis - 1L)
  nv_reshape(x, new_shape)
}

## Linear algebra --------------------------------------------------------------

#' @title Outer Product
#' @description
#' Computes the outer product of two 1-D arrays.
#' @param lhs,rhs ([`arrayish`])\cr
#'   Two 1-D arrays. Can be of any data type; the two are
#'   [promoted to a common data type][nv_promote_to_common()].
#' @return ([`arrayish`])\cr
#'   Has the operands' common data type and shape
#'   `(length(lhs), length(rhs))`.
#' @examplesIf pjrt::plugins_downloaded()
#' # a length-3 and a length-2 vector give a 3x2 at their common data type
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5))
#' nv_outer(x, y)
#' @export
nv_outer <- jit(function(lhs, rhs) {
  args <- nv_promote_to_common(lhs, rhs)
  lhs <- args[[1L]]
  rhs <- args[[2L]]
  if (naxes(lhs) != 1L) {
    cli_abort("{.arg lhs} must be a 1-D array, but it has {naxes(lhs)} axes.")
  }
  if (naxes(rhs) != 1L) {
    cli_abort("{.arg rhs} must be a 1-D array, but it has {naxes(rhs)} axes.")
  }
  lhs_exp <- nv_unsqueeze(lhs, axis = 2L)
  rhs_exp <- nv_unsqueeze(rhs, axis = 1L)
  bcast <- nv_broadcast_arrays(lhs_exp, rhs_exp)
  prim_mul(bcast[[1L]], bcast[[2L]])
})

#' @title Extract Diagonal
#' @description
#' Extracts the diagonal elements from a 2-D array.
#' @templateVar dtypes any data type
#' @templateVar shapes with exactly 2 axes
#' @template param_unary_x
#' @return ([`arrayish`])\cr
#'   Has the input's data type, and one axis of length `min(nrow, ncol)`
#'   holding the diagonal elements.
#' @seealso [nv_diag()] for creating a diagonal matrix, [nv_trace()]
#' @examplesIf pjrt::plugins_downloaded()
#' # the diagonal of a 3x3 matrix, keeping its data type
#' x <- nv_array(1:9, shape = c(3, 3))
#' nv_extract_diag(x)
#' @export
nv_extract_diag <- jit(function(x) {
  x <- as_anvl_array(x)
  if (naxes(x) != 2L) {
    cli_abort("{.arg x} must be a 2-D array")
  }
  shp <- shape(x)
  n <- min(shp)
  idx <- prim_reshape(nv_iota_like(x, axis = 1L, shape = n, dtype = "i32"), shape = c(n, 1L))
  indices <- nv_concatenate(idx, idx, axis = 2L)
  prim_gather(
    x,
    start_indices = indices,
    offset_axes = integer(0L),
    collapsed_slice_axes = c(1L, 2L),
    x_batching_axes = integer(0L),
    start_indices_batching_axes = integer(0L),
    start_index_map = c(1L, 2L),
    index_vector_axis = 2L,
    slice_sizes = c(1L, 1L)
  )
})

#' @title Matrix Trace
#' @description
#' Computes the trace (sum of diagonal elements) of a 2-D array.
#' @templateVar dtypes any data type
#' @templateVar shapes with exactly 2 axes
#' @template param_unary_x
#' @return ([`arrayish`])\cr
#'   A scalar with `x`'s data type, except a boolean input, which is counted
#'   at the default integer data type (see [`default_dtypes()`]).
#' @seealso [nv_extract_diag()], [nv_diag()]
#' @examplesIf pjrt::plugins_downloaded()
#' # the diagonal is summed to a scalar
#' x <- nv_array(c(1, 0, 0, 0, 2, 0, 0, 0, 3), shape = c(3, 3))
#' nv_trace(x)
#' @export
nv_trace <- jit(function(x) {
  x <- as_anvl_array(x)
  diag_vals <- nv_extract_diag(x)
  nv_reduce_sum(diag_vals, axes = 1L, drop = TRUE)
})

# Boolean triangular mask for a 2-D `shape`. `diagonal` must already be a
# scalar integer; `lower` selects the lower (TRUE) or upper (FALSE) triangle.
tri_mask <- function(shape, diagonal, lower, device = NULL) {
  rows <- prim_iota(axis = 1L, dtype = "i32", shape = shape, start = 1L, device = device)
  cols <- prim_iota(axis = 2L, dtype = "i32", shape = shape, start = 1L, device = device)
  if (lower) rows >= cols - diagonal else rows <= cols - diagonal
}

assert_tri_args <- function(shape, diagonal) {
  if (length(shape) != 2L) {
    cli_abort("{.arg shape} must have length 2, not {length(shape)}.")
  }
  if (any(shape < 0L)) {
    cli_abort("{.arg shape} must not contain negative extents.")
  }
  assert_int(diagonal)
}

#' @title Lower Triangular Mask
#' @description
#' Returns a boolean matrix that is `TRUE` on and below the given diagonal,
#' mirroring base R's `lower.tri()`. Use [nv_tril()] to zero out the other
#' triangle of an existing array instead.
#' @param shape (`integer()`)\cr
#'   Shape of the result: exactly two axis sizes, since the result is a matrix.
#' @param diagonal (`integer(1)`)\cr
#'   Diagonal offset, with the same meaning as in [nv_tril()]. The default
#'   `-1` excludes the main diagonal, matching `lower.tri()`; use `0` to
#'   include it, matching `lower.tri(diag = TRUE)`.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_lower_tri_like()`).
#' @template param_device
#' @return ([`arrayish`])\cr
#'   Has the given `shape` and boolean data type. It is a mask over positions,
#'   so no array data enters it -- pass it to [nv_ifelse()] or multiply by it
#'   to use it.
#' @seealso [nv_upper_tri()], [nv_tril()], [prim_iota()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # a boolean mask, whatever the array it is later used with
#' nv_lower_tri(c(3, 3))
#' nv_lower_tri(c(3, 3), diagonal = 0L)
#' x <- nv_fill(0, shape = c(3, 3))
#' nv_lower_tri_like(x)
#' @export
nv_lower_tri <- jit(
  function(shape, diagonal = -1L, device = NULL) {
    assert_tri_args(shape, diagonal)
    tri_mask(shape, as.integer(diagonal), lower = TRUE, device = device)
  },
  static = 1:3
)

#' @title Upper Triangular Mask
#' @description
#' Returns a boolean matrix that is `TRUE` on and above the given diagonal,
#' mirroring base R's `upper.tri()`. Use [nv_triu()] to zero out the other
#' triangle of an existing array instead.
#' @param shape (`integer()`)\cr
#'   Shape of the result: exactly two axis sizes, since the result is a matrix.
#' @param diagonal (`integer(1)`)\cr
#'   Diagonal offset, with the same meaning as in [nv_triu()]. The default
#'   `1` excludes the main diagonal, matching `upper.tri()`; use `0` to
#'   include it, matching `upper.tri(diag = TRUE)`.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_upper_tri_like()`).
#' @template param_device
#' @return ([`arrayish`])\cr
#'   Has the given `shape` and boolean data type.
#' @seealso [nv_lower_tri()], [nv_triu()], [prim_iota()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' # a boolean mask, whatever the array it is later used with
#' nv_upper_tri(c(3, 3))
#' nv_upper_tri(c(3, 3), diagonal = 0L)
#' x <- nv_fill(0, shape = c(3, 3))
#' nv_upper_tri_like(x)
#' @export
nv_upper_tri <- jit(
  function(shape, diagonal = 1L, device = NULL) {
    assert_tri_args(shape, diagonal)
    tri_mask(shape, as.integer(diagonal), lower = FALSE, device = device)
  },
  static = 1:3
)

#' @title Lower Triangular Matrix
#' @description
#' Returns the lower triangular part of a 2-D array, setting elements above
#' the specified diagonal to zero.
#' @templateVar dtypes any data type
#' @templateVar shapes with exactly 2 axes
#' @template param_unary_x
#' @param diagonal (`integer(1)`)\cr
#'   Diagonal offset: the kept region is `col - row <= diagonal`. `0` (default)
#'   keeps the main diagonal and everything below it, a positive value keeps
#'   that many diagonals above it as well, and a negative one drops the main
#'   diagonal and `-diagonal - 1` below it.
#' @return ([`arrayish`])\cr
#'   Has the same shape and data type as `x`.
#' @seealso [nv_triu()], [nv_lower_tri()]
#' @examplesIf pjrt::plugins_downloaded()
#' # elements above the main diagonal become zero
#' x <- nv_fill(1, c(3, 3))
#' nv_tril(x)
#' @export
nv_tril <- jit(
  function(x, diagonal = 0L) {
    x <- as_anvl_array(x)
    if (naxes(x) != 2L) {
      cli_abort("{.arg x} must be a 2-D array")
    }
    nv_ifelse(nv_lower_tri_like(x, diagonal), x, nv_fill_like(x, 0L))
  },
  static = 2L
)

#' @title Upper Triangular Matrix
#' @description
#' Returns the upper triangular part of a 2-D array, setting elements below
#' the specified diagonal to zero.
#' @templateVar dtypes any data type
#' @templateVar shapes with exactly 2 axes
#' @template param_unary_x
#' @param diagonal (`integer(1)`)\cr
#'   Diagonal offset: the kept region is `col - row >= diagonal`. `0` (default)
#'   keeps the main diagonal and everything above it, a negative value keeps
#'   that many diagonals below it as well, and a positive one drops the main
#'   diagonal and `diagonal - 1` above it.
#' @return ([`arrayish`])\cr
#'   Has the same shape and data type as `x`.
#' @seealso [nv_tril()], [nv_upper_tri()]
#' @examplesIf pjrt::plugins_downloaded()
#' # elements below the main diagonal become zero
#' x <- nv_fill(1, c(3, 3))
#' nv_triu(x)
#' @export
nv_triu <- jit(
  function(x, diagonal = 0L) {
    x <- as_anvl_array(x)
    if (naxes(x) != 2L) {
      cli_abort("{.arg x} must be a 2-D array")
    }
    nv_ifelse(nv_upper_tri_like(x, diagonal), x, nv_fill_like(x, 0L))
  },
  static = 2L
)

#' @title Cross Product (Matrix)
#' @description
#' Computes `t(lhs) %*% rhs`. If `rhs` is missing, computes `t(lhs) %*% lhs`.
#' Above rank 2 the last two axes are the matrix and the leading ones are batch
#' axes, as in [nv_matmul()]: only the matrix is transposed.
#' @param lhs ([`arrayish`])\cr
#'   An array with at least 2 axes, as for [base::crossprod()]. Can be any numeric
#'   data type; `lhs` and `rhs` are
#'   [promoted to a common data type][nv_promote_to_common()].
#' @param rhs ([`arrayish`] | `NULL`)\cr
#'   Optional second array. If `NULL`, uses `lhs`.
#' @return ([`arrayish`])\cr
#'   Has the operands' common data type, and the shape of `t(lhs) %*% rhs`.
#' @seealso [nv_tcrossprod()], [nv_matmul()]
#' @examplesIf pjrt::plugins_downloaded()
#' # `t(x) %*% x`, so a 3x2 gives a 2x2
#' x <- nv_matrix(1:6, nrow = 3, dtype = "f32")
#' nv_crossprod(x)
#' @export
nv_crossprod <- jit(function(lhs, rhs = NULL) {
  if (is.null(rhs)) {
    lhs <- as_anvl_array(lhs)
    rhs <- lhs
  } else {
    args <- as_anvl_arrays(lhs, rhs, .promote = promotion_common())
    lhs <- args[[1L]]
    rhs <- args[[2L]]
  }
  nv_matmul(transpose_matrix_axes(lhs), rhs)
})

#' @title Transpose Cross Product (Matrix)
#' @description
#' Computes `lhs %*% t(rhs)`. If `rhs` is missing, computes `lhs %*% t(lhs)`.
#' Above rank 2 the last two axes are the matrix and the leading ones are batch
#' axes, as in [nv_matmul()]: only the matrix is transposed.
#' @param lhs ([`arrayish`])\cr
#'   An array with at least 2 axes, as for [base::tcrossprod()]. Can be any numeric
#'   data type; `lhs` and `rhs` are
#'   [promoted to a common data type][nv_promote_to_common()].
#' @param rhs ([`arrayish`] | `NULL`)\cr
#'   Optional second array. If `NULL`, uses `lhs`.
#' @return ([`arrayish`])\cr
#'   Has the operands' common data type, and the shape of `lhs %*% t(rhs)`.
#' @seealso [nv_crossprod()], [nv_matmul()]
#' @examplesIf pjrt::plugins_downloaded()
#' # `x %*% t(x)`, so a 2x3 gives a 2x2
#' x <- nv_matrix(1:6, nrow = 2, dtype = "f32")
#' nv_tcrossprod(x)
#' @export
nv_tcrossprod <- jit(function(lhs, rhs = NULL) {
  if (is.null(rhs)) {
    lhs <- as_anvl_array(lhs)
    rhs <- lhs
  } else {
    args <- as_anvl_arrays(lhs, rhs, .promote = promotion_common())
    lhs <- args[[1L]]
    rhs <- args[[2L]]
  }
  nv_matmul(lhs, transpose_matrix_axes(rhs))
})

# Sorting and searching --------------------------------------------------------

#' @title Select Elements Along an Axis
#' @description
#' Picks one or more elements along axis `axis` of `x`.
#' Use this instead of `[` or `nv_subset` when the index to select is provided
#' programmatically.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @param axis (`integer(1)`)\cr
#'   Axis to index into.
#'   Negative values count from the end, i.e. `-1` refers to the last axis.
#' @param index ([`arrayish`])\cr
#'   Scalar or 1-D array of an integer data type, which it keeps -- the index
#'   takes no part in `x`'s data type.
#' @return ([`arrayish`])\cr
#'   Has `x`'s data type. `axis` is dropped if `index` was scalar, and
#'   otherwise resized to the number of selected elements.
#' @seealso [nv_subset()] for general subsetting, [prim_static_slice()].
#' @examplesIf pjrt::plugins_downloaded()
#' # a scalar index drops the axis, an index array keeps it
#' m <- nv_matrix(1:6, nrow = 2)
#' nv_select(m, axis = 2L, index = 2L)
#' nv_select(m, axis = 1L, index = 1L)
#' nv_select(m, axis = 2L, index = array(c(1L, 3L)))
#' @export
nv_select <- function(x, axis, index) {
  x <- as_anvl_array(x)
  rank <- naxes(x)
  if (rank == 0L) {
    cli_abort("{.arg x} must have at least one axis to select along, but it is a scalar.")
  }
  axis <- resolve_axis(axis, rank)

  args <- rep(list(quote(expr = )), rank)
  args[[axis]] <- index
  do.call(nv_subset, c(list(x), args))
}

# Per-slice gather along `axis`. `index` has the same rank as `x`,
# matching shape on every non-`axis` axis. Returns shape `shape(index)`.
# Used internally by nv_quantile; not exposed as part of nv_select.
.gather_along_axis <- function(x, index, axis, rank, shp) {
  idx_shape <- shape(index)
  if (!identical(shp[-axis], idx_shape[-axis])) {
    cli_abort(
      "Per-slice {.arg index} must match {.arg x}'s shape on every axis except {.arg axis}"
    )
  }
  axis_idx <- function(d) {
    raw <- if (d == axis) {
      nv_convert(index, "i32")
    } else {
      nv_iota_like(x, axis = d, shape = idx_shape, dtype = "i32")
    }
    prim_reshape(raw, c(idx_shape, 1L))
  }
  start_indices <- do.call(
    nv_concatenate,
    c(lapply(seq_len(rank), axis_idx), list(axis = rank + 1L))
  )
  prim_gather(
    x,
    start_indices = start_indices,
    slice_sizes = rep(1L, rank),
    offset_axes = integer(0L),
    collapsed_slice_axes = seq_len(rank),
    x_batching_axes = integer(0L),
    start_indices_batching_axes = integer(0L),
    start_index_map = seq_len(rank),
    index_vector_axis = rank + 1L
  )
}

#' @title Sort
#' @name nv_sort
#' @description
#' Sorts an array along an axis.
#'
#' You can also use `sort()` directly.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @param axis (`integer(1)` | `NULL`)\cr
#'   Axis along which to sort. Negative values count from the end,
#'   i.e. `-1` refers to the last axis. If `NULL` (default), the input is
#'   first flattened to a 1-D array, like [base::sort()].
#' @param decreasing (`logical(1)`)\cr
#'   If `TRUE`, sort in decreasing order. Default `FALSE`.
#' @param stable (`logical(1)`)\cr
#'   If `TRUE`, the sort is stable: equal values keep their original
#'   relative order along `axis`. Default `FALSE`. Stability is only
#'   observable for floats when `-0` / `+0` or `-NaN` / `+NaN` are mixed
#'   (they compare equal under the total order used here); for distinct
#'   values the result is identical either way.
#' @return ([`arrayish`])\cr
#'   Has the input's shape and data type.
#' @section NaN handling:
#' `NaN` values sort to the **end** (ascending) or **beginning**
#' (descending), regardless of sign. `+0` and `-0` compare equal.
#' @section The `sort()` generic:
#' Like [base::sort()], `nv_sort()` with `axis = NULL` flattens a multi-axis
#' array into one sorted vector, so `sort()` on an anvl array agrees with base
#' R (the flatten order does not matter once the elements are sorted). It
#' differs in one respect: base R drops `NA` by default, whereas `NaN` is kept
#' and sorted to the end. Pass `axis` to sort each slice along one axis
#' instead, which keeps the shape.
#' @seealso [prim_sort()] for the underlying primitive,
#'   [nv_argsort()], [nv_top_k()], [nv_median()],
#'   [nv_argmax()], [nv_argmin()].
#' @examplesIf pjrt::plugins_downloaded()
#' # sorting moves elements, so the data type and shape stay
#' x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
#' nv_sort(x)
#' sort(x) # via the S3 generic
#' nv_sort(x, decreasing = TRUE)
#'
#' m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
#' nv_sort(m) # one sorted vector, like base R
#' nv_sort(m, axis = 2L) # each row sorted, shape kept
#' @export
nv_sort <- jit(
  function(x, axis = NULL, decreasing = FALSE, stable = FALSE) {
    x <- as_anvl_array(x)
    if (naxes(x) == 0L) {
      cli_abort("{.arg x} must have at least one axis to sort along, but it is a scalar.")
    }
    if (is.null(axis)) {
      x <- nv_flatten(x)
      axis <- 1L
    }
    prim_sort(list(x), axis = axis, descending = decreasing, is_stable = stable)[[1L]]
  },
  static = 2:4
)

#' @title Argsort
#' @description
#' Returns the indices that would sort the array: over every element by
#' default, or along one axis. It is the index twin of [nv_sort()] and takes
#' the same `axis`, so the two always describe the same ordering.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @param axis (`integer(1)` | `NULL`)\cr
#'   Axis along which to compute the sort permutation. Negative values
#'   count from the end, i.e. `-1` refers to the last axis. If `NULL`
#'   (default), the input is first flattened to a 1-D array, like
#'   [nv_sort()], and the indices refer to that flattening.
#' @param decreasing (`logical(1)`)\cr
#'   If `TRUE`, returns indices that produce a decreasing sort. Default
#'   `FALSE`.
#' @param stable (`logical(1)`)\cr
#'   If `TRUE`, the sort is stable: indices for equal values keep their
#'   original relative order. Default `FALSE`.
#' @return ([`arrayish`])\cr
#'   Has the default integer data type (see [`default_dtypes()`]) regardless of
#'   the input's, and the input's shape -- or 1-D holding every element's index
#'   when `axis` is `NULL`. For a size-0 axis, the output is an empty array of
#'   the same shape (a valid empty permutation). Indexing the flattened input
#'   by the result reproduces [nv_sort()]'s output.
#' @inheritSection nv_sort NaN handling
#' @seealso [nv_sort()], [prim_sort()].
#' @examplesIf pjrt::plugins_downloaded()
#' # the indices come out at the default integer data type
#' x <- nv_array(c(3, 1, 4, 1, 5))
#' nv_argsort(x)
#'
#' m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
#' nv_argsort(m) # indexes the flattened matrix
#' nv_argsort(m, axis = 2L) # a permutation per row
#' @export
nv_argsort <- jit(
  function(x, axis = NULL, decreasing = FALSE, stable = FALSE) {
    x <- as_anvl_array(x)
    if (naxes(x) == 0L) {
      cli_abort("{.arg x} must have at least one axis to sort along, but it is a scalar.")
    }
    if (is.null(axis)) {
      x <- nv_flatten(x)
      axis <- 1L
    }
    idx <- nv_iota_like(x, axis = axis, dtype = default_int())
    prim_sort(list(x, idx), axis = axis, descending = decreasing, is_stable = stable)[[2L]]
  },
  static = 2:4
)

#' @title Top-K Elements
#' @description
#' Returns the `k` largest values over one or more axes, sorted in decreasing
#' order.
#' @templateVar dtypes any numeric data type
#' @templateVar shapes with at least 1 axis
#' @template param_unary_x
#' @param k (`integer(1)`)\cr
#'   Number of top elements to return. Must be a whole number between 1 and
#'   the number of elements `axes` holds together; a fractional or logical `k`
#'   is refused rather than truncated.
#' @param axes (`integer()` | `NULL`)\cr
#'   Axes to take the top `k` over. Negative values count from the end, i.e.
#'   `-1` refers to the last axis. If `NULL` (default), ranks over every axis.
#' @param indices (`logical(1)`)\cr
#'   If `FALSE` (default), returns just the top-`k` values. If `TRUE`,
#'   returns `list(values = ..., indices = ...)` where `indices` holds the
#'   position of each top-`k` value.
#' @return ([`arrayish`] | named `list` of two [`arrayish`])\cr
#'   One array when `indices = FALSE`, a named `list` of `values` and
#'   `indices` when `indices = TRUE`. The values have the input's data type and
#'   the indices the default integer data type (see [`default_dtypes()`]). Both
#'   have the input's shape with `axes` replaced by a single axis of size `k`,
#'   sitting where the first of them was; values are sorted decreasing along
#'   that axis.
#' @section Ranking several axes:
#' Taking the top `k` over several axes ranks all of their elements together,
#' so `nv_top_k(x, k)` equals `nv_top_k(nv_flatten(x), k)`. The indices then
#' index the column-major flattening of those axes -- the order
#' [nv_flatten()] produces -- rather than any single axis.
#' @section NaN handling:
#' `NaN` ranks larger than any finite value (so it appears first in the
#' top-`k` output); `-NaN` ranks smaller. Unlike [nv_sort()], the sign
#' bit is not canonicalized.
#' @seealso [prim_top_k()] for the underlying primitive, [nv_sort()].
#' @examplesIf pjrt::plugins_downloaded()
#' # the values keep the input's data type, the indices the default integer
#' x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
#' nv_top_k(x, k = 3L)
#' nv_top_k(x, k = 3L, indices = TRUE)
#'
#' m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
#' nv_top_k(m, k = 2L, axes = 2L) # the top 2 of each row
#' nv_top_k(m, k = 2L) # the top 2 of the whole matrix
#' @export
nv_top_k <- jit(
  function(x, k, axes = NULL, indices = FALSE) {
    assert_flag(indices)
    x <- as_anvl_array(x)
    rank <- naxes(x)
    if (rank == 0L) {
      cli_abort("{.arg x} must have at least one axis to take the top {.arg k} along, but it is a scalar.")
    }
    axes <- sort(.resolve_reduce_axes(x, axes))
    n <- as.integer(prod(shape(x)[axes]))
    # Check before coercing: `as.integer()` first would silently truncate a
    # fractional `k` and accept a logical one, where `prim_top_k()` refuses both.
    if (!checkmate::test_int(k, lower = 1L, upper = n)) {
      cli_abort(c(
        "{.arg k} must be a single whole number between 1 and the number of elements {.arg axes} holds together.",
        x = "The ranked ax{cli::qty(length(axes))}{?is/es} hold{?s/} {n} of them, and {.arg k} is {.val {k}}."
      ))
    }
    k <- as.integer(k)

    # `prim_top_k` reads the last axis only, so several axes are gathered into
    # one first -- which is what makes the indices index their column-major
    # flattening -- and the `k` axis is moved back to where the first of them
    # was.
    flat <- .flatten_reduce_axes(x, axes, drop = TRUE)
    out <- prim_top_k(flat$x, k = k, indices = indices)
    keep <- setdiff(seq_len(rank), axes)
    out_rank <- length(keep) + 1L
    permutation <- append(
      seq_len(out_rank - 1L),
      out_rank,
      after = sum(keep < axes[[1L]])
    )
    restore <- function(v) {
      if (identical(permutation, seq_len(out_rank))) {
        v
      } else {
        prim_transpose(v, permutation = permutation)
      }
    }
    if (indices) {
      list(values = restore(out$values), indices = restore(out$indices))
    } else {
      restore(out$values)
    }
  },
  static = 2:4
)

#' @title Quantile
#' @description
#' Computes the `probs` quantile(s) of an array over one or more axes.
#'
#' `probs` follows the same scalar-vs-array convention as [nv_select()]'s
#' `index`:
#'
#' * a length-1 numeric (e.g. `0.5`) treats `probs` as scalar — the result is
#'   the reduction alone;
#' * a 1-D R array (e.g. `array(c(0.25, 0.5, 0.75))`) prepends a leading
#'   axis of size `length(probs)`.
#'
#' Plain length-K (K > 1) vectors are rejected; wrap with `array()` to
#' make the array intent explicit.
#'
#' A quantile generally falls between two elements, so a non-float `x` is
#' computed (and returned) at the default float data type.
#' @section Interpolation modes:
#' For `n` reduced elements and a probability `q`, let
#' `h = 1 + (n - 1) * q` be the position `q` falls at in the sorted values,
#' with `lo = floor(h)`, `hi = ceiling(h)` and `frac = h - lo`. Then, writing
#' `sorted` for the reduced values in sorted order:
#'
#' * `"linear"` (default): `(1 - frac) * sorted[lo] + frac * sorted[hi]`.
#' * `"lower"`: `sorted[lo]` — the lower bracket of `linear`.
#' * `"higher"`: `sorted[hi]` — the upper bracket of `linear`.
#' * `"nearest"`: `sorted[lo]` if `frac < 0.5` else `sorted[hi]`.
#' * `"midpoint"`: `(sorted[lo] + sorted[hi]) / 2`.
#'
#' Reducing several axes at once ranks all of their elements together, so
#' `nv_quantile(x, q, axes = c(1, 2))` equals
#' `nv_quantile(nv_flatten(x), q)` for a matrix `x`.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @param probs (`numeric(1)` | 1-D `array`)\cr
#'   One or more probabilities in `[0, 1]`. Either a length-1 numeric
#'   (scalar) or a 1-D `array` (a leading axis of size `length(probs)` is
#'   prepended). Plain length-K (K > 1) vectors are rejected — wrap with
#'   `array()`.
#' @templateVar axes_all If `NULL` (default), reduces over all axes.
#' @template params_reduce
#' @param interpolation (`character(1)`)\cr
#'   One of `"linear"` (default), `"lower"`, `"higher"`, `"nearest"`,
#'   `"midpoint"`. See "Interpolation modes".
#' @template param_nan_rm
#' @return ([`arrayish`])\cr
#'   Same shape as `x` with `axes` removed (or set to 1 if `drop = FALSE`).
#'   For array `probs`, a **leading** axis of size `length(probs)` is
#'   prepended. The data type is that of `x`, or the default float for a
#'   non-float `x`.
#' @seealso [nv_median()], [nv_sort()].
#' @examplesIf pjrt::plugins_downloaded()
#' # a float result even for an integer input, since it interpolates
#' x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
#' nv_quantile(x, 0.5) # = nv_median(x)
#' nv_quantile(x, array(c(0.25, 0.5, 0.75)))
#' nv_quantile(x, 0.5, interpolation = "lower")
#' m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
#' nv_quantile(m, 0.5) # over every element
#' nv_quantile(m, 0.5, axes = 2L) # one quantile per row
#' nv_quantile(nv_array(c(1, NaN, 3, 5)), 0.5)
#' nv_quantile(nv_array(c(1, NaN, 3, 5)), 0.5, nan_rm = TRUE)
#' @export
nv_quantile <- jit(
  function(x, probs, axes = NULL, drop = TRUE, interpolation = "linear", nan_rm = FALSE) {
    assert_flag(drop)
    assert_flag(nan_rm)
    x <- as_anvl_array(x)
    # A quantile lies between two elements, so -- like base R's quantile() --
    # a non-float array is computed at the default float instead of rounding the
    # interpolation weights down to whole numbers.
    if (!is_dtype_float(peek_dtype(x))) {
      x <- nv_convert(x, default_float())
    }
    assert_choice(interpolation, c("linear", "lower", "higher", "nearest", "midpoint"))
    if (!is_valid_r(probs)) {
      cli_abort("{.arg probs} must either be a length-1 numeric or 1-D R array.")
    }
    if (!checkmate::test_numeric(probs, lower = 0, upper = 1, any.missing = FALSE, min.len = 1L)) {
      cli_abort(c(
        "{.arg probs} must be probabilities: numbers between 0 and 1, none missing.",
        x = "Got {.val {as.vector(probs)}}."
      ))
    }

    is_probs_array <- !is.null(dim(probs))
    if (is_probs_array && length(dim(probs)) != 1L) {
      cli_abort(c(
        "{.arg probs} must be a length-1 numeric or a 1-D array.",
        x = "Got an array with {length(dim(probs))} axes."
      ))
    }

    # The sort below only ever orders one axis, so reducing several means
    # ranking their elements together in one flattened axis.
    axes <- sort(.resolve_reduce_axes(x, axes))
    flat <- .flatten_reduce_axes(x, axes, drop)
    x <- flat$x

    rank <- naxes(x)
    axis <- rank
    shp <- shape(x)
    K <- length(probs)
    probs <- as.numeric(probs)
    shp_kd <- replace(shp, axis, 1L)
    shp_K <- replace(shp, axis, K)
    out_dtype <- dtype(x)

    # Selection instead of a full sort when every requested order statistic lies
    # in one end of the axis. Ascending position j of a slice's n_valid sorted
    # values is at most ceil((n_valid - 1) * max(probs)) + 1, and n_valid <= n
    # only ever moves it down, so the ascending prefix of k_lo elements always
    # holds it; mirrored, the descending prefix of k_hi elements holds every
    # position from floor((n_valid - 1) * min(probs)) + 1 up. Either prefix comes
    # from top_k (of the negated values for the low end), which is cheaper than
    # a sort when the window is at most about half the axis. With nan_rm = FALSE,
    # NaNs rank to the front of the window instead of the back, but any slice
    # containing NaN has its output forced to NaN below, so the gathered values
    # never surface.
    #
    # The window is sized here in R doubles while the gather index is computed
    # on device, so the two agree only if the device arithmetic matches R's. At
    # `dtype(x)` it does not: `21 * (1/7)` is 3 exactly in a double and
    # 3.0000002 in `f32`, so the index lands past the window, where the gather
    # clamps and quietly returns a neighbouring order statistic. The index
    # arithmetic therefore runs at `f64`, which is bit-for-bit what R does, and
    # each window below is the device's own index expression evaluated at
    # `n_valid = n_axis` -- the same operations on the same bits, rather than a
    # second formula for the same quantity, whose own rounding could put the
    # index outside the window at equal precision. The index is nondecreasing
    # in `n_valid`, so `n_axis` gives the largest index any slice can reach and
    # the window is exactly big enough. Only `frac` returns to `out_dtype`, so
    # the result keeps its data type.
    #
    # TODO(metal): Metal has no `f64`, so a program that reaches here cannot run
    # on it at all. Supporting Metal means making the two sides agree the other
    # way round -- rounding the host-side window computation through the
    # device's data type -- instead of widening the device to R's.
    idx_dtype <- "f64"

    n_axis <- shp[axis]
    budget <- ceiling(n_axis / 2L) + 1L
    k_lo <- as.integer(ceiling((n_axis - 1L) * max(probs)) + 1L)
    k_hi <- as.integer(n_axis - floor((n_axis - 1L) * min(probs)))
    path <- if (n_axis > 0L && k_lo <= budget) {
      "low"
    } else if (n_axis > 0L && k_hi <= budget) {
      "high"
    } else {
      "sort"
    }

    # Find the NaN positions: nan_rm = TRUE sanitizes them to the end the window
    # does not read from (+Inf, or -Inf for the high window) so the valid values
    # keep their ranks; nan_rm = FALSE uses them post-hoc to propagate.
    # The count of valid elements is kept at `i32`, since it is a count.
    count_kd <- nv_broadcast_to(
      nv_fill_like(x, shp[axis], shape = integer(), dtype = "i32"),
      shp_kd
    )
    nan_mask <- nv_is_nan(x)
    nan_fill <- if (path == "high") -Inf else Inf
    to_sort <- if (nan_rm) nv_ifelse(nan_mask, nan_fill, x) else x
    n_valid_kd <- if (nan_rm) {
      # At `idx_dtype`, the data type the index arithmetic below runs at; the
      # `i32` count the other branch takes is converted to it as well.
      prim_reduce_sum(nv_convert(!nan_mask, idx_dtype), axes = axis, drop = FALSE)
    } else {
      count_kd
    }
    # All three paths index the same multiset, so the order statistic does not
    # depend on which one ran -- except in the sign of a zero: `top_k` ranks
    # `-0` below `+0`, as `chlo.top_k` does, where `prim_sort()` folds the two
    # together. Nothing short of `1/x` tells those two results apart.
    sorted <- switch(
      path,
      "low" = -nv_top_k(-to_sort, k = k_lo, axes = axis),
      "high" = nv_top_k(to_sort, k = k_hi, axes = axis),
      "sort" = prim_sort(list(to_sort), axis = axis)[[1L]]
    )

    # Broadcast `(K,) probs` along `axis` and `(shp_kd,) n_valid_kd` across
    # `axis` → both shaped `shp_K`, with K varying along `axis`.
    probs_shape <- replace(rep(1L, rank), axis, K)
    probs_b <- nv_broadcast_to(
      prim_reshape(
        nv_array_like(sorted, probs, shape = K, dtype = idx_dtype),
        probs_shape
      ),
      shp_K
    )
    n_valid_b <- nv_convert(nv_broadcast_to(n_valid_kd, shp_K), idx_dtype)
    h <- (n_valid_b - 1L) * probs_b
    lo_f <- nv_floor(h)
    hi_f <- nv_ceiling(h)
    frac <- nv_convert(h - lo_f, out_dtype)

    # `sorted` is ascending, except the high window, which top_k returns in
    # descending order: ascending position j of the slice's n_valid values is
    # its element n_valid - j + 1.
    if (path == "high") {
      lo_idx <- n_valid_b - lo_f
      hi_idx <- n_valid_b - hi_f
    } else {
      lo_idx <- lo_f + 1L
      hi_idx <- hi_f + 1L
    }
    lo_val <- .gather_along_axis(sorted, nv_convert(lo_idx, "i32"), axis, rank, shp)
    hi_val <- .gather_along_axis(sorted, nv_convert(hi_idx, "i32"), axis, rank, shp)

    out <- switch(
      interpolation,
      "lower" = lo_val,
      "higher" = hi_val,
      "nearest" = nv_ifelse(frac < 0.5, lo_val, hi_val),
      "linear" = lo_val * (1L - frac) + hi_val * frac,
      "midpoint" = (lo_val + hi_val) / 2L
    )

    # Propagate NaN: nan_rm = TRUE produces NaN only for all-NaN slices;
    # nan_rm = FALSE produces NaN for any slice that contained a NaN (XLA's
    # sort places NaN unpredictably, so we can't rely on the gather hitting it).
    bad <- if (nan_rm) n_valid_kd == 0L else prim_reduce_any(nan_mask, axes = axis, drop = FALSE)
    out <- nv_ifelse(nv_broadcast_to(bad, shp_K), NaN, out)

    # Unpack the trailing axis the reduced ones were gathered into: dropped, or
    # restored at size 1 in their original positions. For array `probs`, the K
    # axis moves from the back, where it is now, to the front.
    if (is_probs_array) {
      out <- prim_transpose(out, permutation = c(axis, seq_len(rank - 1L)))
      prim_reshape(out, c(K, flat$keep_shape))
    } else {
      prim_reshape(out, flat$keep_shape)
    }
  },
  static = 2:6
)


#' @title Median
#' @name nv_median
#' @description
#' Computes the median over one or more axes. Equivalent to
#' `nv_quantile(x, 0.5, axes, drop, interpolation)`; for an even number of
#' reduced elements with the default `"linear"` interpolation, the average of
#' the two middle values is returned, matching base R's `median()`.
#'
#' You can also use `median()` directly on an [`AnvlArray`] or [`AnvlBox`];
#' extra arguments (e.g. `interpolation`) are forwarded via `...`.
#' @section The `median()` generic:
#' [stats::median()] reduces every axis of a multi-axis array, and so does
#' `nv_median()` by default, so the two agree. Pass `axes` to reduce a subset
#' instead. A non-float `x` is computed at the default float, like base R
#' returns a double.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @templateVar axes_all If `NULL` (default), reduces over all axes.
#' @template params_reduce
#' @param interpolation (`character(1)`)\cr
#'   Forwarded to [nv_quantile()]. One of `"linear"` (default), `"lower"`,
#'   `"higher"`, `"nearest"`, `"midpoint"`.
#' @param nan_rm (`logical(1)`)\cr
#'   Forwarded to [nv_quantile()]. See its documentation for details.
#' @return ([`arrayish`])\cr
#'   Same shape as `x` with `axes` removed (or set to 1 if `drop = FALSE`).
#'   The data type is that of `x`, or the default float for a non-float `x`.
#' @seealso [nv_quantile()], [nv_sort()], [prim_sort()].
#' @examplesIf pjrt::plugins_downloaded()
#' nv_median(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#' median(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#' m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
#' nv_median(m) # over every element
#' nv_median(m, axes = 2L) # one median per row
#' # forwards through the S3 generic via `...`
#' median(nv_array(c(1, 2, 3, 4)), interpolation = "lower")
#' nv_median(nv_array(c(1, NaN, 3, 5)))
#' nv_median(nv_array(c(1, NaN, 3, 5)), nan_rm = TRUE)
#' @export
nv_median <- jit(
  function(x, axes = NULL, drop = TRUE, interpolation = "linear", nan_rm = FALSE) {
    assert_flag(nan_rm)
    nv_quantile(
      x,
      probs = 0.5,
      axes = axes,
      drop = drop,
      interpolation = interpolation,
      nan_rm = nan_rm
    )
  },
  static = 2:5
)

#' @title Index of the Maximum
#' @description
#' Returns the index of the maximum value over one or more axes. Ties are
#' broken by returning the smallest index.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @templateVar axes_all If `NULL` (default), reduces over all axes.
#' @template params_reduce
#' @template param_nan_rm
#' @return ([`arrayish`])\cr
#'   Has the default integer data type (see [`default_dtypes()`]) regardless of
#'   the input's, and the input's shape with `axes` removed (`drop = TRUE`) or
#'   set to 1 (`drop = FALSE`).
#' @section Reducing several axes:
#' `nv_argmax()` is the index to [nv_reduce_max()]'s value: called with the
#' same `axes` and `drop`, it points at the element whose value
#' `nv_reduce_max()` returns. Reducing several axes ranks their elements
#' together, and the result indexes the column-major flattening of those axes
#' -- the order [nv_flatten()] produces and [base::which.max()] reports -- which
#' is also the order ties are broken in.
#' @section NaN handling:
#' With `nan_rm = FALSE` (default), if any entry being reduced is `NaN`, the
#' returned index points at the first such `NaN`. With `nan_rm = TRUE`, `NaN`
#' entries are skipped.
#' @seealso [nv_argmin()], [nv_reduce_max()].
#' @examplesIf pjrt::plugins_downloaded()
#' # the index comes out at the default integer data type
#' nv_argmax(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#' m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
#' nv_argmax(m) # indexes the flattened matrix
#' nv_argmax(m, axes = 2L) # one index per row
#' nv_argmax(nv_array(c(1, NaN, 3)))
#' nv_argmax(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#' @export
nv_argmax <- jit(
  function(x, axes = NULL, drop = TRUE, nan_rm = FALSE) {
    assert_flag(drop)
    assert_flag(nan_rm)
    .nv_arg_extreme(as_anvl_array(x), axes, drop, nan_rm, prim_argmax)
  },
  static = 2:4
)

#' @title Index of the Minimum
#' @description
#' Returns the index of the minimum value over one or more axes. Ties are
#' broken by returning the smallest index.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @templateVar axes_all If `NULL` (default), reduces over all axes.
#' @template params_reduce
#' @template param_nan_rm
#' @return ([`arrayish`])\cr
#'   Has the default integer data type (see [`default_dtypes()`]) regardless of
#'   the input's, and the input's shape with `axes` removed (`drop = TRUE`) or
#'   set to 1 (`drop = FALSE`).
#' @section Reducing several axes:
#' `nv_argmin()` is the index to [nv_reduce_min()]'s value: called with the
#' same `axes` and `drop`, it points at the element whose value
#' `nv_reduce_min()` returns. Reducing several axes ranks their elements
#' together, and the result indexes the column-major flattening of those axes
#' -- the order [nv_flatten()] produces and [base::which.min()] reports -- which
#' is also the order ties are broken in.
#' @inheritSection nv_argmax NaN handling
#' @seealso [nv_argmax()], [nv_reduce_min()].
#' @examplesIf pjrt::plugins_downloaded()
#' # the index comes out at the default integer data type
#' nv_argmin(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#' m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
#' nv_argmin(m) # indexes the flattened matrix
#' nv_argmin(m, axes = 2L) # one index per row
#' nv_argmin(nv_array(c(2, NaN, 1, 3)))
#' nv_argmin(nv_array(c(2, NaN, 1, 3)), nan_rm = TRUE)
#' @export
nv_argmin <- jit(
  function(x, axes = NULL, drop = TRUE, nan_rm = FALSE) {
    assert_flag(drop)
    assert_flag(nan_rm)
    .nv_arg_extreme(as_anvl_array(x), axes, drop, nan_rm, prim_argmin)
  },
  static = 2:4
)

# Shared NaN-aware argmax/argmin. The primitives read a single axis, so several
# axes are gathered into one trailing axis first, exactly as `nv_quantile()`
# does -- which is what makes the result index the column-major flattening of
# `axes`, the one `which.max()` reports.
#
# The XLA arg-reduction kernels are comparison-based and silently skip NaN, so
# `nan_rm = TRUE` is free -- we just call the primitive. For `nan_rm = FALSE` we
# want NaN to propagate, mirroring `.nv_reduce_extreme`'s contract; there's no
# NaN in an integer, so we surface "a NaN was here" by returning the first NaN's
# index instead.
.nv_arg_extreme <- function(x, axes, drop, nan_rm, prim_arg) {
  flat <- .flatten_reduce_axes(x, sort(.resolve_reduce_axes(x, axes)), drop)
  x <- flat$x
  axis <- naxes(x)
  result <- prim_arg(x, axis = axis, drop = TRUE)
  if (!nan_rm && is_dtype_float(peek_dtype(x))) {
    # argmax on the bool mask returns the index of the first TRUE (tie-break:
    # smallest index) — exactly the first NaN's position — or 1 if no NaN
    # exists. `any_nan` disambiguates those two cases.
    nan_mask <- nv_is_nan(x)
    any_nan <- prim_reduce_any(nan_mask, axes = axis, drop = TRUE)
    first_nan_idx <- prim_argmax(nan_mask, axis = axis, drop = TRUE)
    result <- nv_ifelse(any_nan, first_nan_idx, result)
  }
  prim_reshape(result, flat$keep_shape)
}

# Build the NCHW/NC(D)HW axis numbers (1-based) for nv_conv*.
.nv_conv_axis_numbers <- function(n_spatial) {
  spatial <- 3:(2L + n_spatial)
  list(
    input_batch_axis = 1L,
    input_feature_axis = 2L,
    input_spatial_axes = spatial,
    kernel_output_feature_axis = 1L,
    kernel_input_feature_axis = 2L,
    kernel_spatial_axes = spatial,
    output_batch_axis = 1L,
    output_feature_axis = 2L,
    output_spatial_axes = spatial
  )
}

# Normalize a stride/dilation/padding arg to length n_spatial.
.nv_conv_vec <- function(x, n, name) {
  x <- as.integer(x)
  if (length(x) == 1L) {
    x <- rep(x, n)
  }
  if (length(x) != n) {
    cli_abort("{.arg {name}} must have length 1 or {n}.")
  }
  x
}

#' @title 1D Convolution
#' @description
#' Torch-style 1D convolution in NCW layout: `x` is
#' `[batch, in_channels, width]`, `weight` is
#' `[out_channels, in_channels / groups, kW]`, output is
#' `[batch, out_channels, out_w]`. Symmetric zero padding.
#' @param x ([`arrayish`])\cr `[N, C_in, W]`. Can be any data type; `x` and
#'   `weight` are [promoted to a common data type][nv_promote_to_common()]. An
#'   R value assumes the other operand's data type, and materializes at its
#'   [default data type][default_dtypes] when that has none either.
#' @param weight ([`arrayish`])\cr `[C_out, C_in / groups, kW]`.
#'   Promoted together with `x` -- see `x`.
#' @param stride,padding,dilation (`integer()`)\cr Length 1.
#' @param groups (`integer(1)`)\cr Grouped/depthwise convolution.
#' @param precision (`character(1)`)\cr `"highest"`, `"high"` or `"default"`.
#' @return ([`arrayish`])\cr
#'   Has the operands' common data type, and shape
#'   `[N, C_out, out_W]`.
#' @seealso [nv_conv2d()], [nv_conv3d()], [prim_convolution()].
#' @examplesIf pjrt::plugins_downloaded()
#' # one batch, one channel, width 5, convolved with a width-3 kernel
#' x <- nv_array(1:5, shape = c(1, 1, 5), dtype = "f32")
#' weight <- nv_array(c(1, 0, -1), shape = c(1, 1, 3), dtype = "f32")
#' nv_conv1d(x, weight)
#'
#' # `padding = 1` keeps the input width, `stride = 2` visits every other
#' # window position
#' nv_conv1d(x, weight, padding = 1L)
#' nv_conv1d(x, weight, stride = 2L)
#' @export
nv_conv1d <- function(x, weight, stride = 1L, padding = 0L, dilation = 1L, groups = 1L, precision = "highest") {
  .nv_convnd(x, weight, 1L, stride, padding, dilation, groups, precision)
}

#' @title 2D Convolution
#' @description
#' Torch-style 2D convolution in NCHW layout: `x` is
#' `[batch, in_channels, height, width]`, `weight` is
#' `[out_channels, in_channels / groups, kh, kw]`, output is
#' `[batch, out_channels, out_h, out_w]`. Symmetric zero padding.
#' @param x ([`arrayish`])\cr `[N, C_in, H, W]`. Can be any data type; `x` and
#'   `weight` are [promoted to a common data type][nv_promote_to_common()]. An
#'   R value assumes the other operand's data type, and materializes at its
#'   [default data type][default_dtypes] when that has none either.
#' @param weight ([`arrayish`])\cr `[C_out, C_in / groups, kH, kW]`.
#'   Promoted together with `x` -- see `x`.
#' @param stride (`integer()`)\cr Length 1 or 2.
#' @param padding (`integer()`)\cr Symmetric padding, length 1 or 2.
#' @param dilation (`integer()`)\cr Kernel dilation, length 1 or 2.
#' @param groups (`integer(1)`)\cr Grouped/depthwise convolution.
#' @param precision (`character(1)`)\cr `"highest"`, `"high"` or `"default"`.
#' @return ([`arrayish`])\cr
#'   Has the operands' common data type, and shape
#'   `[N, C_out, out_H, out_W]`.
#' @seealso [nv_conv1d()], [nv_conv3d()], [prim_convolution()].
#' @examplesIf pjrt::plugins_downloaded()
#' # one batch, one channel, 4x4, convolved with a 3x3 kernel
#' x <- nv_array(1:16, shape = c(1, 1, 4, 4), dtype = "f32")
#' weight <- nv_fill(1, shape = c(1, 1, 3, 3), dtype = "f32")
#' nv_conv2d(x, weight)
#'
#' # two output channels give a result with two channels
#' weight2 <- nv_fill(1, shape = c(2, 1, 3, 3), dtype = "f32")
#' shape(nv_conv2d(x, weight2))
#' @export
nv_conv2d <- function(x, weight, stride = 1L, padding = 0L, dilation = 1L, groups = 1L, precision = "highest") {
  .nv_convnd(x, weight, 2L, stride, padding, dilation, groups, precision)
}

#' @title 3D Convolution
#' @description
#' Torch-style 3D convolution in NCDHW layout. `x` is
#' `[batch, in_channels, depth, height, width]`, `weight` is
#' `[out_channels, in_channels / groups, kD, kH, kW]`. Asymmetric
#' padding (e.g. causal temporal padding) is available via
#' [prim_convolution()].
#' @param x ([`arrayish`])\cr `[N, C_in, D, H, W]`. Can be any data type; `x` and
#'   `weight` are [promoted to a common data type][nv_promote_to_common()]. An
#'   R value assumes the other operand's data type, and materializes at its
#'   [default data type][default_dtypes] when that has none either.
#' @param weight ([`arrayish`])\cr `[C_out, C_in / groups, kD, kH, kW]`.
#'   Promoted together with `x` -- see `x`.
#' @inheritParams nv_conv2d
#' @param stride,padding,dilation (`integer()`)\cr Length 1 or 3.
#' @return ([`arrayish`])\cr
#'   Has the operands' common data type, and shape
#'   `[N, C_out, out_D, out_H, out_W]`.
#' @seealso [nv_conv1d()], [nv_conv2d()], [prim_convolution()].
#' @examplesIf pjrt::plugins_downloaded()
#' # one batch, one channel, 2x3x3, convolved with a 1x2x2 kernel
#' x <- nv_array(1:18, shape = c(1, 1, 2, 3, 3), dtype = "f32")
#' weight <- nv_fill(1, shape = c(1, 1, 1, 2, 2), dtype = "f32")
#' shape(nv_conv3d(x, weight))
#' @export
nv_conv3d <- function(x, weight, stride = 1L, padding = 0L, dilation = 1L, groups = 1L, precision = "highest") {
  .nv_convnd(x, weight, 3L, stride, padding, dilation, groups, precision)
}

.nv_convnd <- function(x, weight, n, stride, padding, dilation, groups, precision) {
  # The `nv_*` layer promotes across data types; `prim_convolution()` would
  # require `x` and `weight` to agree already, and would name its own operand.
  args <- as_anvl_arrays(x = x, weight = weight, .promote = promotion_common())
  x <- args$x
  weight <- args$weight
  assert_int(groups, lower = 1L)
  # `prim_convolution()` and stablehlo below both speak of `lhs`, `rhs` and
  # `kernel_input_feature_dimension`; none of those is an argument here.
  for (nm in c("x", "weight")) {
    value <- get(nm)
    if (naxes(value) != n + 2L) {
      cli_abort(c(
        "{.arg {nm}} must have {n + 2L} axes for a {n}-D convolution.",
        x = "Got shape {shape_repr(shape(value))}."
      ))
    }
  }
  in_channels <- shape(x)[2L]
  if (in_channels %% groups != 0L) {
    cli_abort(c(
      "{.arg groups} must divide the number of input channels of {.arg x}.",
      x = "{.arg x} has {in_channels} input channel{?s}, and {.arg groups} is {groups}."
    ))
  }
  if (shape(weight)[2L] != in_channels / groups) {
    cli_abort(c(
      "{.arg weight}'s second axis must be {.arg x}'s input channels divided by {.arg groups}.",
      x = "Expected {in_channels / groups}, but {.arg weight} is {shape_repr(shape(weight))}."
    ))
  }
  if (shape(weight)[1L] %% groups != 0L) {
    cli_abort(c(
      "{.arg groups} must divide the number of output channels of {.arg weight}.",
      x = "{.arg weight} has {shape(weight)[1L]} output channel{?s}, and {.arg groups} is {groups}."
    ))
  }
  stride <- .nv_conv_vec(stride, n, "stride")
  pad <- .nv_conv_vec(padding, n, "padding")
  dilation <- .nv_conv_vec(dilation, n, "dilation")
  do.call(
    prim_convolution,
    c(
      list(x, weight),
      .nv_conv_axis_numbers(n), # individual 1-based axis params
      list(
        window_strides = stride,
        padding = cbind(pad, pad), # symmetric [n, 2]
        x_dilation = rep(1L, n),
        kernel_dilation = dilation,
        feature_group_count = as.integer(groups),
        batch_group_count = 1L,
        precision = precision
      )
    )
  )
}
