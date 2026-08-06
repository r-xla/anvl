# This is the user-facing API containing the exported array operations.
#' @include primitives.R

# Special array creators

#' @title Fill Constant
#' @description
#' Creates an array filled with a scalar value. More memory-efficient than
#' `nv_array(value, shape = shape)` for large arrays.
#'
#' `nv_fill_like()` is a variant where `dtype`, `shape`, `ambiguous`, and
#' `device` default to those of `like`.
#' @param value (`numeric(1)`)\cr
#'   Scalar value to fill the array with.
#' @param shape (`integer()`)\cr
#'   Shape of the output array.
#' @param dtype (`character(1)` | `NULL`)\cr
#'   Data type.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_fill_like()`).
#' @template param_ambiguous
#' @template param_device
#' @return [`arrayish`]\cr
#'   Has the given `shape` and `dtype`.
#' @seealso [prim_fill()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_fill(0, shape = c(2, 3))
#' x <- nv_matrix(1:6, nrow = 2)
#' nv_fill_like(x, 0)
#' @export
nv_fill <- function(value, shape, dtype = NULL, ambiguous = FALSE, device = NULL) {
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
  prim_fill(value, shape, dtype, ambiguous, device = device)
}

## Conversion ------------------------------------------------------------------

broadcast_shapes <- function(shape_lhs, shape_rhs) {
  if (length(shape_lhs) > length(shape_rhs)) {
    shape_rhs <- c(rep(1L, length(shape_lhs) - length(shape_rhs)), shape_rhs)
  } else if (length(shape_lhs) < length(shape_rhs)) {
    shape_lhs <- c(rep(1L, length(shape_rhs) - length(shape_lhs)), shape_lhs)
  } else if (identical(shape_lhs, shape_rhs)) {
    return(shape_lhs)
  }
  shape_out <- shape_lhs
  for (i in seq_along(shape_lhs)) {
    d_lhs <- shape_lhs[i]
    d_rhs <- shape_rhs[i]
    if (d_lhs != d_rhs && d_lhs != 1L && d_rhs != 1L) {
      cli_abort("lhs and rhs are not broadcastable")
    }
    shape_out[i] <- max(d_lhs, d_rhs)
  }
  shape_out
}

make_broadcast_axes <- function(shape_in, shape_out) {
  rank_in <- length(shape_in)
  rank_out <- length(shape_out)
  if (rank_in == rank_out) {
    # When ranks match, each input axis maps to the same output axis
    # StableHLO expects a mapping for every input axis
    return(seq_along(shape_out))
  }
  tail(seq_len(rank_out), rank_in)
}


#' @title Broadcast Scalars to Common Shape
#' @description
#' Broadcast scalar arrays to match the shape of non-scalar arrays.
#' All non-scalar arrays must have the same shape.
#' @param ... ([`arrayish`][arrayish])\cr
#'   Arrays to broadcast. Scalars will be broadcast to the common non-scalar shape.
#' @return (`list()` of [`arrayish`])\cr
#'   List of broadcasted arrays.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' # scalar 1 is broadcast to shape [3]
#' nv_broadcast_scalars(x, nv_scalar(1))
#' @export
nv_broadcast_scalars <- function(...) {
  args <- as_anvl_arrays(...)
  shapes <- lapply(args, shape)
  non_scalar_shapes <- Filter(\(s) length(s) > 0L, shapes)

  if (length(non_scalar_shapes) == 0L) {
    return(args)
  }

  target_shape <- non_scalar_shapes[[1L]]
  if (!all(vapply(non_scalar_shapes, identical, logical(1L), target_shape))) {
    shapes <- paste0(sapply(shapes, shape2string), collapse = ", ")
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
}

#' @title Promote Arrays to a Common Dtype
#' @description
#' Promote arrays to a common data type, see [`common_dtype`] for more details.
#' @param ... ([`arrayish`])\cr
#'   Arrays to promote.
#' @return (`list()` of [`arrayish`])
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1L)
#' y <- nv_array(1.5)
#' # integer is promoted to float
#' nv_promote_to_common(x, y)
#' @export
nv_promote_to_common <- function(...) {
  args <- as_anvl_arrays(...)
  tmp <- do.call(common_type_info, args)
  cdt <- tmp[[1L]]
  ambiguous <- tmp[[2L]]
  out <- lapply(seq_along(args), \(i) {
    if (cdt == dtype(args[[i]])) {
      args[[i]]
    } else {
      prim_convert(args[[i]], dtype = cdt, ambiguous = ambiguous)
    }
  })
  return(out)
}

#' @title Broadcast Arrays to a Common Shape
#' @description
#' Broadcasts arrays to a common shape using NumPy-style broadcasting rules.
#'
#' @section Broadcasting Rules:
#' 1. If the arrays have different numbers of axes, prepend size-1
#'    axes to the shorter shape.
#' 2. For each axis: if the sizes match, keep them; if one is 1, expand
#'    it to the other's size; otherwise raise an error.
#'
#' @param ... ([`arrayish`])\cr
#'   Arrays to broadcast.
#' @return (`list()` of [`arrayish`])\cr
#'   List of arrays, all with the same shape.
#' @seealso [nv_broadcast_scalars()], [nv_broadcast_to()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' y <- nv_array(c(10, 20, 30))
#' nv_broadcast_arrays(x, y)
#' @export
#' @jit
nv_broadcast_arrays <- function(...) {
  args <- as_anvl_arrays(...)
  shape <- Reduce(broadcast_shapes, lapply(args, shape))
  lapply(args, nv_broadcast_to, shape = shape)
}

#' @title Broadcast to Shape
#' @description
#' Broadcasts an array to a target shape using NumPy-style broadcasting rules.
#' @template param_x
#' @param shape (`integer()`)\cr
#'   Target shape. Each existing axis must either match or be 1.
#' @return [`arrayish`]\cr
#'   Has the given `shape` and the same data type as `x`.
#' @seealso [nv_broadcast_arrays()], [nv_broadcast_scalars()],
#'   [prim_broadcast_in_axes()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' nv_broadcast_to(x, shape = c(2, 3))
#' @export
nv_broadcast_to <- function(x, shape) {
  x <- as_anvl_array(x)
  shape_op <- shape(x)
  if (!identical(shape_op, shape)) {
    broadcast_axes <- make_broadcast_axes(shape_op, shape)
    prim_broadcast_in_axes(x, shape, broadcast_axes)
  } else {
    x
  }
}

#' @title Convert Data Type
#' @description
#' Converts the elements of an array to a different data type.
#' Returns the input unchanged if it already has the target type.
#' @template param_x
#' @template param_dtype
#' @return [`arrayish`]\cr
#'   Has the given `dtype` and the same shape as `x`.
#' @seealso [prim_convert()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1L, 2L, 3L))
#' nv_convert(x, dtype = "f32")
#' @export
nv_convert <- function(x, dtype) {
  x <- as_anvl_array(x)
  if (dtype(x) != as_dtype(dtype)) {
    prim_convert(x, dtype = as_dtype(dtype), ambiguous = FALSE)
  } else {
    x
  }
}

#' @rdname nv_transpose
#' @template param_x
#' @export
nv_transpose <- function(x, permutation = NULL) {
  x <- as_anvl_array(x)
  permutation <- permutation %||% rev(seq_len(naxes(x)))
  prim_transpose(x, permutation)
}


#' @title Reshape
#' @description
#' Reshapes an array to a new shape without changing the underlying data.
#' Returns the input unchanged if it already has the target shape.
#' @details
#' Note that row-major order is used, which differs from R's column-major order.
#' @template param_x
#' @param shape (`integer()`)\cr
#'   Target shape. Must have the same number of elements as `x`.
#'   At most one entry may be `-1`, in which case its extent is inferred from
#'   the remaining entries and the number of elements of `x`.
#' @return [`arrayish`]\cr
#'   Has the given `shape` and the same data type as `x`.
#' @seealso [prim_reshape()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:6)
#' nv_reshape(x, c(2, 3))
#' nv_reshape(x, c(2, -1)) # infer the second dimension
#' nv_reshape(x, -1) # flatten
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

#' @title Flatte
#' @description
#' Flattens an N-dimensional array into a 1-dimensional array.
#' Fails with scalar inputs.
#' @template param_x
#' @return ([`arrayish`])\cr
#'   1-D array.
#' @export
#' @examples
#' nv_flatten(matrix(1:4, nrow = 2))
nv_flatten <- function(x) {
  x <- as_anvl_array(x)
  if (naxes(x) == 0) {
    cli_abort("Cannot flatten a scalar array.")
  }
  nv_reshape(x, prod(shape(x)))
}

#' @title Concatenate
#' @description
#' Concatenates arrays along an axis. Operands are promoted to a common
#' data type and scalars are broadcast before concatenation.
#' @param ... ([`arrayish`])\cr
#'   Arrays to concatenate. Must have the same shape except along `axis`.
#' @param axis (`integer(1)` | `NULL`)\cr
#'   Axis along which to concatenate.
#'   Negative values count from the end, i.e. `-1` refers to the last axis.
#'   If `NULL` (default), assumes all inputs are at most 1-D and concatenates along axis 1.
#' @return [`arrayish`]\cr
#'   Has the common data type and a shape matching the inputs in all
#'   axes except `axis`, which is the sum of input sizes.
#' @seealso [prim_concatenate()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5, 6))
#' nv_concatenate(x, y)
#' @export
#' @jit static "axis"
nv_concatenate <- function(..., axis = NULL) {
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

  non_scalar_shapes_without_axis <- lapply(non_scalar_shapes, \(shape) {
    shape[-axis]
  })
  if (length(non_scalar_shapes) && length(unique(non_scalar_shapes_without_axis)) != 1L) {
    cli_abort(c(
      "All non-scalar arrays must have the same shape (except for the concatenation axis)",
      x = "Got shapes {shapes2string(shapes)} and axis {axis}"
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
}

#' @title Combine arrays by rows or columns
#' @name nv_bind
#' @description
#' Combine arrays along the row (`nv_rbind`) or column (`nv_cbind`) axis.
#' Arguments are first promoted to a common data type
#' (see [nv_promote_to_common()]).
#'
#' Each input is then handled according to its rank:
#'
#' * 0-D: broadcast to match the non-stacked axes of the other inputs.
#' * 1-D: treated as a single row/column.
#' * Other: used as-is.
#'
#' # Differences from base R
#'
#' [base::rbind()] and [base::cbind()] applied to an [array()] of rank > 2
#' flatten the trailing axes into the column axis (so a `c(2, 3, 4)`
#' array becomes a `2 x 12` matrix). `nv_rbind` and `nv_cbind` instead
#' preserve all non-stacked axes: combining two `c(2, 3, 4)` arrays
#' with `nv_rbind` produces a `c(4, 3, 4)` array, and with `nv_cbind` a
#' `c(2, 6, 4)` array.
#'
#' @param ... ([`arrayish`])\cr
#'   Arrays to combine. Inputs are promoted to a common data type.
#' @return [`arrayish`]\cr
#' @seealso [nv_concatenate()]
#' @examplesIf pjrt::plugins_downloaded()
#' # Vectors as rows / columns
#' nv_rbind(nv_array(1:3), nv_array(4:6))
#' nv_cbind(nv_array(1:3), nv_array(4:6))
#'
#' # Scalar broadcasting
#' nv_rbind(nv_matrix(1:6, nrow = 2), nv_scalar(0))
#'
#' # Rank-3 arrays preserve trailing axes
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
      x = "Got shapes {shapes2string(shapes)}"
    ))
  }
  non_stack <- lapply(reshaped, \(s) s[-stack_axis])
  if (length(unique(non_stack)) != 1L) {
    cli_abort(c(
      "{.fn {fn_name}} inputs must agree on every non-stacked axis",
      x = "Got shapes {shapes2string(shapes)}"
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
#' @jit
nv_rbind <- function(...) {
  args <- as_anvl_arrays(...)
  target_shape <- bind_target_shape(args, stack_axis = 1L, fn_name = "nv_rbind")
  args <- lapply(args, bind_reshape, stack_axis = 1L, target_shape = target_shape)
  rlang::exec(nv_concatenate, !!!args, axis = 1L)
}

#' @rdname nv_bind
#' @export
#' @jit
nv_cbind <- function(...) {
  args <- as_anvl_arrays(...)
  target_shape <- bind_target_shape(args, stack_axis = 2L, fn_name = "nv_cbind")
  args <- lapply(args, bind_reshape, stack_axis = 2L, target_shape = target_shape)
  rlang::exec(nv_concatenate, !!!args, axis = 2L)
}

#' @title Static Slice
#' @description
#' Extracts a slice from an array using static (compile-time) indices.
#' For dynamic indexing, use [nv_subset()] instead.
#' @template param_x
#' @param start_indices (`integer()`)\cr
#'   Start indices (inclusive), one per axis.
#' @param limit_indices (`integer()`)\cr
#'   End indices (inclusive), one per axis.
#' @param strides (`integer()`)\cr
#'   Step sizes, one per axis. A stride of 1 selects every element.
#' @return [`arrayish`]\cr
#'   Has the same data type as `x`.
#' @seealso [nv_subset()], [prim_static_slice()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:10)
#' nv_static_slice(x, start_indices = 2L, limit_indices = 5L, strides = 1L)
#' @export
nv_static_slice <- prim_static_slice

#' @title Print Array
#' @description
#' Prints an array value to the console during JIT execution and returns the
#' input unchanged. Useful for debugging.
#' @template param_x
#' @return [`arrayish`]\cr
#'   Returns `x` unchanged.
#' @seealso [prim_print()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' nv_print(x)
#' @export
nv_print <- prim_print

#' @title Conditional Element Selection
#' @description
#' Selects elements from `true_value` or `false_value` based on `pred`,
#' analogous to R's [ifelse()].
#' @param pred ([`arrayish`] of boolean type)\cr
#'   Predicate array. Must be scalar or have the same shape as the
#'   non-scalar arguments.
#' @param true_value,false_value ([`arrayish`])\cr
#'   Values to return where `pred` is `TRUE` / `FALSE`.
#'   `true_value` and `false_value` are
#'   [promoted to a common data type][nv_promote_to_common()].
#'   Scalars (including `pred`) are
#'   [broadcast][nv_broadcast_scalars()] to the shape of the non-scalar arguments.
#' @return [`arrayish`]\cr
#'   Has the common data type of `true_value` and `false_value` and the
#'   shape of the non-scalar arguments.
#' @seealso [prim_ifelse()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' pred <- nv_array(c(TRUE, FALSE, TRUE))
#' nv_ifelse(pred, nv_array(c(1, 2, 3)), nv_array(c(4, 5, 6)))
#' # scalar branches are broadcast and promoted to a common dtype
#' nv_ifelse(pred, nv_scalar(1L), nv_scalar(0.5))
#' @export
#' @jit
nv_ifelse <- function(pred, true_value, false_value) {
  # Canonicalize all three inputs together so an R literal `true_value` /
  # `false_value` inherits the device of `pred` (and vice versa). Doing the
  # `nv_promote_to_common` step first would convert literals on the default
  # device and then conflict with a non-default-device `pred`.
  args <- as_anvl_arrays(pred, true_value, false_value)
  promoted <- nv_promote_to_common(args[[2L]], args[[3L]])
  args <- nv_broadcast_scalars(args[[1L]], promoted[[1L]], promoted[[2L]])
  prim_ifelse(args[[1L]], args[[2L]], args[[3L]])
}

## Binary ops ------------------------------------------------------------------

make_do_binary <- function(f) {
  function(lhs, rhs) {
    args <- nv_promote_to_common(lhs, rhs)
    args <- nv_broadcast_scalars(args[[1L]], args[[2L]])
    do.call(f, args)
  }
}

#' @title Addition
#' @description
#' Adds two arrays element-wise. You can also use the `+` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_add()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5, 6))
#' x + y
#' @export
#' @jit
nv_add <- make_do_binary(prim_add)

#' @title Multiplication
#' @description
#' Multiplies two arrays element-wise. You can also use the `*` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_mul()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5, 6))
#' x * y
#' @export
#' @jit
nv_mul <- make_do_binary(prim_mul)

#' @title Subtraction
#' @description
#' Subtracts two arrays element-wise. You can also use the `-` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_sub()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(4, 5, 6))
#' y <- nv_array(c(1, 2, 3))
#' x - y
#' @export
#' @jit
nv_sub <- make_do_binary(prim_sub)

#' @title Division
#' @description
#' Divides two arrays element-wise. You can also use the `/` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_div()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(10, 20, 30))
#' y <- nv_array(c(2, 5, 10))
#' x / y
#' @export
#' @jit
nv_div <- make_do_binary(prim_div)

#' @title Power
#' @description
#' Raises `lhs` to the power of `rhs` element-wise. You can also use the `^` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_pow()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(2, 3, 4))
#' y <- nv_array(c(3, 2, 1))
#' x ^ y
#' @export
#' @jit
nv_pow <- make_do_binary(prim_pow)

#' @title Equal
#' @description
#' Element-wise equality comparison. You can also use the `==` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_eq()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(1, 3, 2))
#' x == y
#' @export
#' @jit
nv_eq <- make_do_binary(prim_eq)

#' @title Not Equal
#' @description
#' Element-wise inequality comparison. You can also use the `!=` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_ne()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(1, 3, 2))
#' x != y
#' @export
#' @jit
nv_ne <- make_do_binary(prim_ne)

#' @title Greater Than
#' @description
#' Element-wise greater than comparison. You can also use the `>` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_gt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' x > y
#' @export
#' @jit
nv_gt <- make_do_binary(prim_gt)

#' @title Greater Than or Equal
#' @description
#' Element-wise greater than or equal comparison. You can also use the `>=` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_ge()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' x >= y
#' @export
#' @jit
nv_ge <- make_do_binary(prim_ge)

#' @title Less Than
#' @description
#' Element-wise less than comparison. You can also use the `<` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_lt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' x < y
#' @export
#' @jit
nv_lt <- make_do_binary(prim_lt)

#' @title Less Than or Equal
#' @description
#' Element-wise less than or equal comparison. You can also use the `<=` operator.
#' @template params_lhs_rhs
#' @template return_compare
#' @seealso [prim_le()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' x <= y
#' @export
#' @jit
nv_le <- make_do_binary(prim_le)

#' @title Maximum
#' @description
#' Element-wise maximum of two arrays.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_max()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 5, 3))
#' y <- nv_array(c(4, 2, 6))
#' nv_max(x, y)
#' @export
#' @jit
nv_max <- make_do_binary(prim_max)

#' @title Minimum
#' @description
#' Element-wise minimum of two arrays.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_min()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 5, 3))
#' y <- nv_array(c(4, 2, 6))
#' nv_min(x, y)
#' @export
#' @jit
nv_min <- make_do_binary(prim_min)

#' @title Remainder (Truncating)
#' @description
#' Element-wise remainder.
#' This differs from base R's `%%`, use [`nv_mod()`]/`%%` instead.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nv_mod()] for the flooring remainder, [prim_remainder()] for the
#'   underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(7, 8, 9))
#' y <- nv_array(c(3, 3, 4))
#' nv_remainder(x, y)
#' @export
#' @jit
nv_remainder <- make_do_binary(prim_remainder)

#' @title Modulo (Flooring Remainder)
#' @description
#' Element-wise flooring remainder of division. The sign of the result equals
#' the sign of `rhs`, matching base R's `%%` operator.
#'
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [nv_remainder()] for truncating remainder, [prim_remainder()] for
#'   the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1L, -1L))
#' y <- nv_array(c(-3L, 3L))
#' nv_mod(x, y)
#' as.vector(x) %% as.vector(y)
#' @export
#' @jit
nv_mod <- function(lhs, rhs) {
  args <- nv_promote_to_common(lhs, rhs)
  args <- nv_broadcast_scalars(args[[1L]], args[[2L]])
  lhs <- args[[1L]]
  rhs <- args[[2L]]
  nv_remainder(nv_remainder(lhs, rhs) + rhs, rhs)
}

#' @title Logical And
#' @description
#' Element-wise logical AND. You can also use the `&` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_and()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' y <- nv_array(c(TRUE, TRUE, FALSE))
#' x & y
#' @export
#' @jit
nv_and <- make_do_binary(prim_and)

#' @title Logical Or
#' @description
#' Element-wise logical OR. You can also use the `|` operator.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_or()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' y <- nv_array(c(TRUE, TRUE, FALSE))
#' x | y
#' @export
#' @jit
nv_or <- make_do_binary(prim_or)

#' @title Logical Xor
#' @description
#' Element-wise logical XOR.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_xor()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' y <- nv_array(c(TRUE, TRUE, FALSE))
#' nv_xor(x, y)
#' @export
#' @jit
nv_xor <- make_do_binary(prim_xor)

#' @title Shift Left
#' @description
#' Element-wise left bit shift.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_shift_left()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1L, 2L, 4L))
#' y <- nv_array(c(1L, 2L, 1L))
#' nv_shift_left(x, y)
#' @export
#' @jit
nv_shift_left <- make_do_binary(prim_shift_left)

#' @title Logical Shift Right
#' @description
#' Element-wise logical right bit shift.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_shift_right_logical()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(8L, 16L, 32L))
#' y <- nv_array(c(1L, 2L, 3L))
#' nv_shift_right_logical(x, y)
#' @export
#' @jit
nv_shift_right_logical <- make_do_binary(prim_shift_right_logical)

#' @title Arithmetic Shift Right
#' @description
#' Element-wise arithmetic right bit shift.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_shift_right_arithmetic()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(8L, -16L, 32L))
#' y <- nv_array(c(1L, 2L, 3L))
#' nv_shift_right_arithmetic(x, y)
#' @export
#' @jit
nv_shift_right_arithmetic <- make_do_binary(prim_shift_right_arithmetic)

#' @title Arctangent 2
#' @description
#' Element-wise two-argument arctangent, i.e. the angle (in radians) between the positive
#' x-axis and the point `(rhs, lhs)`.
#' @template params_lhs_rhs
#' @template return_binary
#' @seealso [prim_atan2()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' y <- nv_array(c(1, 0, -1))
#' x <- nv_array(c(0, 1, 0))
#' nv_atan2(y, x)
#' @export
#' @jit
nv_atan2 <- make_do_binary(prim_atan2)


#' @title Bitcast Conversion
#' @name nv_bitcast_convert
#' @description
#' Reinterprets the bits of an array as a different data type without modifying
#' the underlying data. If the target type is narrower, an extra trailing
#' axis is added; if wider, the last axis is consumed.
#' @template param_x
#' @param dtype (`character(1)` | [`DataType`])\cr
#'   Target data type.
#' @return [`arrayish`]\cr
#'   Has the given `dtype`.
#' @seealso [prim_bitcast_convert()] for the underlying primitive, [nv_convert()]
#'   for value-preserving type conversion.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1L)
#' prim_bitcast_convert(x, dtype = "i8")
#' @export
nv_bitcast_convert <- prim_bitcast_convert

## Unary ops ------------------------------------------------------------------

#' @title Negation
#' @description
#' Negates an array element-wise. You can also use the unary `-` operator.
#' @template param_x
#' @template return_unary
#' @seealso [prim_negate()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, -2, 3))
#' -x
#' @export
nv_negate <- prim_negate

#' @title Logical Not
#' @description
#' Element-wise logical NOT. You can also use the `!` operator.
#' @template param_x
#' @template return_unary
#' @seealso [prim_not()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' !x
#' @export
nv_not <- prim_not

#' @title Absolute Value
#' @description
#' Element-wise absolute value. You can also use `abs()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_abs()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 2, -3))
#' abs(x)
#' @export
nv_abs <- prim_abs

#' @title Square Root
#' @description
#' Element-wise square root. You can also use `sqrt()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_sqrt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 4, 9))
#' sqrt(x)
#' @export
nv_sqrt <- prim_sqrt

#' @title Reciprocal Square Root
#' @description
#' Element-wise reciprocal square root, i.e. `1 / sqrt(x)`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_rsqrt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 4, 9))
#' nv_rsqrt(x)
#' @export
nv_rsqrt <- prim_rsqrt

#' @title Natural Logarithm
#' @description
#' Element-wise natural logarithm. You can also use `log()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_log()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2.718, 7.389))
#' log(x)
#' @export
nv_log <- prim_log

#' @title Hyperbolic Tangent
#' @description
#' Element-wise hyperbolic tangent. You can also use `tanh()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_tanh()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' tanh(x)
#' @export
nv_tanh <- prim_tanh

#' @title Tangent
#' @description
#' Element-wise tangent. You can also use `tan()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_tan()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 0.5, 1))
#' tan(x)
#' @export
nv_tan <- prim_tan

#' @title Sine
#' @description
#' Element-wise sine. You can also use `sin()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_sin()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, pi / 2, pi))
#' sin(x)
#' @export
nv_sin <- prim_sin

#' @title Cosine
#' @description
#' Element-wise cosine. You can also use `cos()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_cos()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, pi / 2, pi))
#' cos(x)
#' @export
nv_cos <- prim_cos

#' @title Floor
#' @description
#' Element-wise floor (round toward negative infinity). You can also use `floor()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_floor()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1.2, 2.7, -1.5))
#' floor(x)
#' @export
nv_floor <- prim_floor

#' @title Ceiling
#' @description
#' Element-wise ceiling (round toward positive infinity). You can also use `ceiling()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_ceil()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1.2, 2.7, -1.5))
#' ceiling(x)
#' @export
nv_ceiling <- prim_ceil

#' @title Truncate
#' @description
#' Element-wise truncation (round toward zero). You can also use `trunc()`.
#' @template param_x
#' @template return_unary
#' @seealso [nv_floor()], [nv_ceiling()], [nv_round()].
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1.2, 2.7, -1.5))
#' trunc(x)
#' @export
#' @jit
nv_trunc <- function(x) {
  x <- as_anvl_array(x)
  nv_mul(nv_sign(x), nv_floor(nv_abs(x)))
}

#' @title Sign
#' @description
#' Element-wise sign function. You can also use `sign()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_sign()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-3, 0, 5))
#' sign(x)
#' @export
nv_sign <- prim_sign

#' @title Exponential
#' @description
#' Element-wise exponential. You can also use `exp()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_exp()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 1, 2))
#' exp(x)
#' @export
nv_exp <- prim_exp

#' @title Exponential Minus One
#' @description
#' Element-wise `exp(x) - 1`, more accurate for small `x`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_expm1()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 0.001, 1))
#' nv_expm1(x)
#' @export
nv_expm1 <- prim_expm1

#' @title Log Plus One
#' @description
#' Element-wise `log(1 + x)`, more accurate for small `x`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_log1p()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 0.001, 1))
#' nv_log1p(x)
#' @export
nv_log1p <- prim_log1p

#' @title Cube Root
#' @description
#' Element-wise cube root.
#' @template param_x
#' @template return_unary
#' @seealso [prim_cbrt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 8, 27))
#' nv_cbrt(x)
#' @export
nv_cbrt <- prim_cbrt

#' @title Logistic (Sigmoid)
#' @description
#' Element-wise logistic sigmoid: `1 / (1 + exp(-x))`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_logistic()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-2, 0, 2))
#' nv_logistic(x)
#' @export
nv_logistic <- prim_logistic

#' @title Arc Cosine
#' @description
#' Element-wise inverse cosine. You can also use `acos()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_acos()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' acos(x)
#' @export
nv_acos <- prim_acos

#' @title Inverse Hyperbolic Cosine
#' @description
#' Element-wise inverse hyperbolic cosine. You can also use `acosh()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_acosh()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 10))
#' acosh(x)
#' @export
nv_acosh <- prim_acosh

#' @title Arc Sine
#' @description
#' Element-wise inverse sine. You can also use `asin()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_asin()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' asin(x)
#' @export
nv_asin <- prim_asin

#' @title Inverse Hyperbolic Sine
#' @description
#' Element-wise inverse hyperbolic sine. You can also use `asinh()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_asinh()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' asinh(x)
#' @export
nv_asinh <- prim_asinh

#' @title Arc Tangent
#' @description
#' Element-wise inverse tangent. You can also use `atan()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_atan()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' atan(x)
#' @export
nv_atan <- prim_atan

#' @title Inverse Hyperbolic Tangent
#' @description
#' Element-wise inverse hyperbolic tangent. You can also use `atanh()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_atanh()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-0.5, 0, 0.5))
#' atanh(x)
#' @export
nv_atanh <- prim_atanh

#' @title Hyperbolic Cosine
#' @description
#' Element-wise hyperbolic cosine. You can also use `cosh()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_cosh()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' cosh(x)
#' @export
nv_cosh <- prim_cosh

#' @title Hyperbolic Sine
#' @description
#' Element-wise hyperbolic sine. You can also use `sinh()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_sinh()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' sinh(x)
#' @export
nv_sinh <- prim_sinh

#' @title Digamma
#' @description
#' Element-wise digamma function (logarithmic derivative of the gamma
#' function). You can also use `digamma()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_digamma()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0.5, 1, 2, 5))
#' digamma(x)
#' @export
nv_digamma <- prim_digamma

#' @title Log-Gamma
#' @description
#' Element-wise natural logarithm of the absolute value of the gamma
#' function. You can also use `lgamma()`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_lgamma()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0.5, 1, 2, 5))
#' lgamma(x)
#' @export
nv_lgamma <- prim_lgamma

#' @title Polygamma
#' @description
#' Element-wise polygamma function: the `(n+1)`-th derivative of the
#' log-gamma function. The order `n` is broadcast against `x` (so
#' `nv_polygamma(1, x)` works for any `x`). For `n = 0` this is the
#' digamma function; for `n = 1`, `trigamma()` dispatches here.
#'
#' Inputs are
#' [promoted to a common floating data type][nv_promote_to_common()] and
#' scalar arguments are
#' [broadcast][nv_broadcast_scalars()] to the shape of the non-scalar
#' arguments.
#' @param n,x ([`arrayish`])\cr
#'   Floating-point arrayish values. After promotion and broadcasting,
#'   `n` and `x` must have the same shape; `n` typically holds
#'   non-negative integer values.
#' @template return_binary
#' @seealso [prim_polygamma()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0.5, 1, 2, 5))
#' nv_polygamma(1, x) # trigamma
#' @export
#' @jit static 1L
nv_polygamma <- function(n, x) {
  args <- nv_promote_to_common(n, x)
  args <- nv_broadcast_scalars(args[[1L]], args[[2L]])
  do.call(prim_polygamma, args)
}

#' @title Error Function
#' @description
#' Element-wise error function `erf(x) = (2 / sqrt(pi)) * integral_0^x exp(-t^2) dt`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_erf()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' nv_erf(x)
#' @export
nv_erf <- prim_erf

#' @title Inverse Error Function
#' @description
#' Element-wise inverse error function (the inverse of `erf` on `(-1, 1)`).
#' @template param_x
#' @template return_unary
#' @seealso [prim_erf_inv()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-0.5, 0, 0.5))
#' nv_erf_inv(x)
#' @export
nv_erf_inv <- prim_erf_inv

#' @title Complementary Error Function
#' @description
#' Element-wise complementary error function `erfc(x) = 1 - erf(x)`.
#' @template param_x
#' @template return_unary
#' @seealso [prim_erfc()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' nv_erfc(x)
#' @export
nv_erfc <- prim_erfc

#' @title Is Finite
#' @description
#' Element-wise check if values are finite (not `Inf`, `-Inf`, or `NaN`).
#' @template param_x
#' @template return_unary_boolean
#' @seealso [prim_is_finite()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, Inf, NaN, -Inf, 0))
#' nv_is_finite(x)
#' @export
nv_is_finite <- prim_is_finite

#' @title Population Count
#' @description
#' Element-wise population count (number of set bits).
#' @template param_x
#' @template return_unary
#' @seealso [prim_popcnt()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(7L, 3L, 15L))
#' nv_popcnt(x)
#' @export
nv_popcnt <- prim_popcnt

#' @title Clamp
#' @description
#' Element-wise clamp: `max(min_val, min(x, max_val))`.
#' Converts `min_val` and `max_val` to the data type of `x`.
#' @details
#' The underlying stableHLO function already broadcasts scalars, so no need to broadcast manually.
#' @param min_val,max_val ([`arrayish`])\cr
#'   Minimum and maximum values (scalar or same shape as `x`).
#' @template param_x
#' @template return_unary
#' @seealso [prim_clamp()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0.5, 2))
#' nv_clamp(nv_scalar(0), x, nv_scalar(1))
#' @export
#' @jit
nv_clamp <- function(min_val, x, max_val) {
  args <- as_anvl_arrays(min_val, x, max_val)
  min_val <- args[[1L]]
  x <- args[[2L]]
  max_val <- args[[3L]]
  op_dtype <- dtype(x)
  min_val <- nv_convert(min_val, op_dtype)
  max_val <- nv_convert(max_val, op_dtype)
  prim_clamp(min_val, x, max_val)
}

#' @title Reverse
#' @description
#' Reverses the order of elements along specified axes.
#' @template param_x
#' @param axes (`integer()`)\cr
#'   Axes to reverse.
#'   Negative values count from the end, i.e. `-1` refers to the last axis.
#' @return [`arrayish`]\cr
#'   Has the same shape and data type as `x`.
#' @seealso [prim_reverse()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3, 4, 5))
#' nv_reverse(x, axes = 1L)
#' @export
nv_reverse <- prim_reverse

#' @title Iota
#' @description
#' Creates an array with values increasing along the specified axis,
#' starting from `start`.
#'
#' `nv_iota_like()` is a variant where `dtype`, `shape`, `ambiguous`, and
#' `device` default to those of `like`.
#' @param axis (`integer(1)`)\cr
#'   Axis along which values increase.
#'   Negative values count from the end of `shape`, i.e. `-1` refers to the
#'   last axis.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_iota_like()`).
#' @template param_dtype
#' @template param_shape
#' @param start (`integer(1)`)\cr
#'   Starting value (default 1).
#' @template param_ambiguous
#' @template param_device
#' @return [`arrayish`]\cr
#'   Has the given `dtype` and `shape`.
#' @seealso [nv_seq()] for a simpler 1-D sequence, [prim_iota()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_iota(axis = 1L, dtype = "i32", shape = 5L)
#' x <- nv_fill(0L, shape = c(2, 3))
#' nv_iota_like(x, axis = 1L)
#' @export
nv_iota <- prim_iota

#' @title Sequence
#' @description
#' Creates a 1-D array with values from `start` to `end` (inclusive).
#'
#' Without `steps`, behaves like R's `seq(start, end)` producing integer values.
#' With `steps`, produces `steps` evenly spaced values (like `seq(start, end, length.out = steps)`).
#'
#' `nv_seq_like()` is a variant where `dtype`, `ambiguous`, and `device`
#' default to those of `like`.
#' @param start,end (`numeric(1)`)\cr
#'   Start and end values. When `steps` is `NULL`, must satisfy `start <= end`.
#' @param steps (`integer(1)` or `NULL`)\cr
#'   Number of evenly spaced values to generate. Must be at least 1.
#'   When `NULL` (default), generates consecutive integer values from `start` to `end`.
#' @param dtype (`character(1)`)\cr
#'   Data type. Default `"i32"` when `steps` is `NULL`, `"f32"` when `steps` is given.
#'   For `nv_seq_like()`, `NULL` uses `dtype(like)`.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_seq_like()`).
#' @template param_ambiguous
#' @template param_device
#' @return [`arrayish`]\cr
#'   1-D array of length `end - start + 1`.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_seq(3, 7)
#' x <- nv_array(c(1, 2, 3), dtype = "f64")
#' nv_seq_like(x, 1, 5)
#' @export
#' @jit static 1:6
nv_seq <- function(start, end, steps = NULL, dtype = NULL, ambiguous = FALSE, device = NULL) {
  if (is.null(steps)) {
    dtype <- dtype %||% "i32"
    assert_int(start)
    assert_int(end)
    assert(start <= end)
    return(nv_iota(
      shape = end - start + 1,
      dtype = dtype,
      ambiguous = ambiguous,
      axis = 1L,
      start = start,
      device = device
    ))
  }
  dtype <- dtype %||% "f32"
  assert_int(steps, lower = 1L)
  if (steps == 1L) {
    return(nv_fill(start, 1L, dtype = dtype, device = device))
  }
  indices <- nv_iota(axis = 1L, shape = steps, dtype = dtype, start = 0L, device = device)
  indices * ((end - start) / (steps - 1L)) + start
}

#' @title Pad
#' @description
#' Pads an array with a given value at the edges and optionally between elements.
#' @template param_x
#' @param padding_value ([`arrayish`])\cr
#'   Scalar value to use for padding. Must have the same dtype as `x`.
#' @param edge_padding_low (`integer()`)\cr
#'   Amount of padding to add at the start of each axis.
#' @param edge_padding_high (`integer()`)\cr
#'   Amount of padding to add at the end of each axis.
#' @param interior_padding (`integer()` | `NULL`)\cr
#'   Amount of padding to add between elements in each axis.
#'   If `NULL` (default), no interior padding is applied.
#' @return [`arrayish`]\cr
#'   Has the same data type as `x`.
#' @seealso [prim_pad()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' nv_pad(x, nv_scalar(0), edge_padding_low = 2L, edge_padding_high = 1L)
#' @export
nv_pad <- function(x, padding_value, edge_padding_low, edge_padding_high, interior_padding = NULL) {
  args <- as_anvl_arrays(x, padding_value)
  x <- args[[1L]]
  padding_value <- args[[2L]]
  rank <- naxes(x)
  if (is.null(interior_padding)) {
    interior_padding <- rep(0L, rank)
  }
  prim_pad(x, padding_value, edge_padding_low, edge_padding_high, interior_padding)
}

#' @title Round
#' @description
#' Element-wise rounding. You can also use the `round()` generic.
#' @template param_x
#' @param method (`character(1)`)\cr
#'   Rounding method.
#'   Either `"nearest_even"` (default) or `"afz"` (away from zero).
#' @template return_unary
#' @seealso [prim_round()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1.4, 2.5, 3.6))
#' round(x)
#' @export
nv_round <- prim_round

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
#'   Arrays with at least 2 axes.
#'   Operands are [promoted to a common data type][nv_promote_to_common()].
#' @param precision (`character(1)`)\cr
#'   Controls the trade-off between speed and numerical accuracy of the
#'   operation. One of `"highest"` (default), `"high"` or `"default"`.
#'   See [prim_dot_general()] for details.
#' @return [`arrayish`]
#' @seealso [prim_dot_general()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' y <- nv_matrix(1:6, nrow = 3)
#' x %*% y
#' @export
#' @jit static "precision"
nv_matmul <- function(lhs, rhs, precision = "highest") {
  args <- nv_promote_to_common(lhs, rhs)
  lhs <- args[[1L]]
  rhs <- args[[2L]]
  if (naxes(lhs) < 2L) {
    cli_abort("lhs of matmul must have at least 2 axes")
  }
  if (naxes(rhs) < 2L) {
    cli_abort("rhs of matmul must have at least 2 axes")
  }
  nbatch <- naxes(lhs) - 2L
  prim_dot_general(
    lhs,
    rhs,
    contracting_axes = list(naxes(lhs), naxes(rhs) - 1L),
    batching_axes = list(seq_len(nbatch), seq_len(nbatch)),
    precision = precision
  )
}

#' @title Cholesky Decomposition
#' @description
#' Computes the Cholesky decomposition of a symmetric positive-definite matrix.
#' Supports batched inputs: axes before the last two are batch axes.
#' @param x ([`arrayish`])\cr
#'   Symmetric positive-definite matrix with at least 2 axes.
#'   The last two axes form the square matrix; any leading axes
#'   are batch axes.
#' @param lower (`logical(1)`)\cr
#'   If `FALSE` (default, matching base R's [base::chol()]), compute the
#'   upper triangular factor `U` such that `x = t(U) %*% U`. If
#'   `TRUE`, compute the lower triangular factor `L` such that
#'   `x = L %*% t(L)`.
#' @return [`arrayish`]\cr
#'   Triangular matrix with the same shape and data type as the input.
#' @seealso [nv_solve()], [prim_chol()]
#' @examplesIf pjrt::plugins_downloaded()
#' a <- nv_matrix(c(4, 2, 2, 3), nrow = 2, dtype = "f32")
#' nv_chol(a)
#' @export
nv_chol <- prim_chol

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
#'   Square non-singular matrix.
#' @param b ([`arrayish`])\cr
#'   Right-hand side, vector of length `n` or matrix with `n` rows. Must
#'   have the same data type as `a`.
#' @return [`arrayish`]\cr
#'   The solution `x` such that `a %*% x = b`.
#' @seealso [nv_chol()], [nv_triangular_solve()], [prim_lu()]
#' @examplesIf pjrt::plugins_downloaded()
#' a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
#' b <- nv_matrix(c(1, 2), nrow = 2, dtype = "f64")
#' nv_solve(a, b)
#' @export
#' @jit
nv_solve <- function(a, b) {
  args <- as_anvl_arrays(a, b)
  a <- args[[1L]]
  b <- args[[2L]]
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
}

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
#' @param a ([`arrayish`])\cr
#'   Triangular coefficient matrix with at least 2 axes. The last two
#'   axes must be equal; any leading axes are batch axes.
#' @param b ([`arrayish`])\cr
#'   Right-hand side. For `a` of shape `(B..., n, n)`, `b` may be either:
#'   * full rank — shape `(B..., n, k)` when `left_side = TRUE`, or
#'     `(B..., k, n)` when `left_side = FALSE`;
#'   * one rank less, shape `(B..., n)`, meaning a single column
#'     (`left_side = TRUE`) or row (`left_side = FALSE`) per batch — it
#'     is reshaped internally and the reshape is undone on the result so
#'     the output rank matches `b`.
#'
#'   `b`'s batch axes (`B...`) must match `a`'s exactly.
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
#' @return [`arrayish`]\cr
#'   The solution `x`, with the same shape and dtype as `b`.
#' @seealso [nv_solve()], [nv_chol()], [prim_triangular_solve()]
#' @examplesIf pjrt::plugins_downloaded()
#' L <- nv_matrix(c(2, 1, 0, 3), nrow = 2, dtype = "f32")
#' b <- nv_matrix(c(4, 3), nrow = 2, dtype = "f32")
#' nv_triangular_solve(L, b)
#' @export
#' @jit static 3:6
nv_triangular_solve <- function(
  a,
  b,
  left_side = TRUE,
  lower = TRUE,
  unit_diagonal = FALSE,
  transpose_a = FALSE
) {
  args <- as_anvl_arrays(a, b)
  a <- args[[1L]]
  b <- args[[2L]]

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
  # restored on the way out. There's no ambiguity from broadcasting since
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
}

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
#' @param x ([`arrayish`])\cr
#'   Square matrix of floating-point data type.
#' @return Scalar [`arrayish`] with the same dtype as `x`.
#' @seealso [nv_determinant()], [nv_solve()], [prim_lu()]
#' @examplesIf pjrt::plugins_downloaded()
#' a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
#' nv_det(a)
#' @export
#' @jit
nv_det <- function(x) {
  d <- nv_determinant(x, logarithm = FALSE)
  prim_mul(d$sign, d$modulus)
}

#' @title Determinant in modulus/sign form
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
#' @param x ([`arrayish`])\cr
#'   Square matrix of floating-point data type.
#' @param logarithm (`logical(1)`)\cr
#'   If `TRUE` (default, matching base R), `modulus` is
#'   `log(abs(det(x)))`. If `FALSE`, `modulus` is `abs(det(x))`.
#' @return Named `list` with elements `modulus` and `sign`, both scalar
#'   [`arrayish`] with the same dtype as `x`. The full determinant
#'   is `sign * exp(modulus)` (with `logarithm = TRUE`) or
#'   `sign * modulus` (with `logarithm = FALSE`).
#' @seealso [nv_det()], [nv_solve()], [prim_lu()]
#' @examplesIf pjrt::plugins_downloaded()
#' a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
#' nv_determinant(a)
#' nv_determinant(a, logarithm = FALSE)
#' @export
#' @jit static 2L
nv_determinant <- function(x, logarithm = TRUE) {
  x <- as_anvl_array(x)
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
    one <- nv_scalar_like(x, 1)
    modulus <- if (logarithm) nv_scalar_like(x, 0) else one
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
}

#' @title Matrix Inverse
#' @description
#' Computes `x^-1`, the inverse of a square non-singular matrix `x`, by
#' solving `x %*% y = I` for `y`.
#'
#' For most use cases prefer [nv_solve()] directly: forming the explicit
#' inverse is both slower and less numerically stable than solving against
#' a right-hand side.
#' @param x ([`arrayish`])\cr
#'   Square non-singular matrix.
#' @return [`arrayish`]\cr
#'   The inverse, same shape and dtype as `x`.
#' @seealso [nv_solve()]
#' @examplesIf pjrt::plugins_downloaded()
#' a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
#' nv_inv(a)
#' @export
#' @jit
nv_inv <- function(x) {
  x <- as_anvl_array(x)
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
}

#' @title QR Decomposition
#' @inherit prim_qr description params return details
#' @seealso [prim_qr()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, dtype = "f32")
#' nv_qr(x)
#' @export
nv_qr <- prim_qr

#' @title LU Decomposition
#' @description
#' Computes the partial-pivoted LU decomposition of a matrix `x`:
#' \deqn{P A = L U,}
#' where \eqn{P} is a permutation matrix, \eqn{L} is unit lower
#' triangular, and \eqn{U} is upper triangular.
#'
#' This function returns `L` and `U` as separate matrices.
#' Use [`prim_lu()`] to get them in packed `LU` form.
#' @inheritParams prim_lu
#' @return Named `list`:
#'   * `L` -- unit lower-triangular factor of shape `(m, k)`, where
#'     `(m, n) = shape(x)` and `k = min(m, n)`.
#'   * `U` -- upper-triangular factor of shape `(k, n)`.
#'   * `pivots` -- length `k`, dtype `i32`. LAPACK-style sequential
#'     1-based row swaps as returned by `getrf`.
#'   * `permutation` -- length `m`, dtype `i32`. A 1-based permutation
#'     vector representing \eqn{P}.
#' @seealso [prim_lu()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
#' nv_lu(x)
#' @export
#' @jit
nv_lu <- function(x) {
  x <- as_anvl_array(x)
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
}

#' @title Singular Value Decomposition
#' @inherit prim_svd description params return details
#' @seealso [prim_svd()], [base::svd()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(1, 0, 0, 1, 0, 1), nrow = 3, dtype = "f64")
#' nv_svd(x)
#' @export
nv_svd <- prim_svd

#' @title Symmetric Eigendecomposition
#' @inherit prim_eigh description params return details
#' @seealso [prim_eigh()], [base::eigen()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
#' nv_eigh(x)
#' @export
nv_eigh <- prim_eigh

#' @title Diagonal Matrix
#' @description
#' Creates a diagonal matrix from a 1-D array.
#' @param x ([`arrayish`])\cr
#'   A 1-D array of length `n` whose elements become the diagonal entries.
#' @return [`arrayish`]\cr
#'   An `n x n` matrix with `x` on the diagonal and zeros elsewhere.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_diag(nv_array(c(1, 2, 3)))
#' @export
#' @jit
nv_diag <- function(x) {
  x <- as_anvl_array(x)
  if (naxes(x) != 1L) {
    cli_abort(c(
      "{.arg x} must be a 1-D array.",
      x = "Got shape {xlamisc::shapevec_repr(shape(x))}."
    ))
  }
  n <- shape(x)[1L]
  zeros <- nv_fill_like(x, 0, shape = c(n, n))
  idx <- prim_reshape(nv_iota_like(x, axis = 1L, shape = n, dtype = "i32"), shape = c(n, 1L))
  indices <- nv_concatenate(idx, idx, axis = 2L)
  prim_scatter(
    zeros,
    indices,
    x,
    update_window_axes = integer(0),
    inserted_window_axes = c(1L, 2L),
    x_batching_axes = integer(0),
    scatter_indices_batching_axes = integer(0),
    scatter_axes_to_x_axes = c(1L, 2L),
    index_vector_axis = 2L,
    unique_indices = TRUE
  )
}

#' @title Identity Matrix
#' @description
#' Creates an `n x n` identity matrix.
#'
#' `nv_eye_like()` is a variant where `dtype` and `device` default to those of
#' `like`.
#' @param n (`integer(1)`)\cr
#'   Size of the identity matrix.
#' @param like ([`arrayish`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_eye_like()`).
#' @template param_dtype
#' @template param_device
#' @return [`arrayish`]\cr
#'   An `n x n` identity matrix.
#' @seealso [nv_diag()] for general diagonal matrices.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_eye(3L)
#' x <- nv_fill(0, shape = c(3, 3), dtype = "f64")
#' nv_eye_like(x, 3L)
#' @export
#' @jit static 1:3
nv_eye <- function(n, dtype = "f32", device = NULL) {
  nv_diag(nv_fill(1, n, dtype = dtype, device = device))
}

# Expand `axes = NULL` to "all axes". Negative axes are resolved here
# (not just in the primitive) because `nv_mean()` / `nv_var()` / `nv_sd()`
# index `shape(x)[axes]` to compute the number of reduced elements.
.resolve_reduce_axes <- function(x, axes) {
  if (is.null(axes)) {
    return(seq_len(naxes(x)))
  }
  resolve_axes(axes, naxes(x), arg = "axes", unique = TRUE)
}

#' @title Sum Reduction
#' @description
#' Sums array elements along the specified axes.
#' @template param_x
#' @template params_reduce
#' @template return_reduce
#' @template param_nan_rm
#' @seealso [prim_reduce_sum()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' nv_reduce_sum(x)            # all axes -> scalar
#' nv_reduce_sum(x, axes = 1L)
#' nv_reduce_sum(nv_array(c(1, NaN, 3)))
#' nv_reduce_sum(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#' @export
nv_reduce_sum <- function(x, axes = NULL, drop = TRUE, nan_rm = FALSE) {
  x <- as_anvl_array(x)
  axes <- .resolve_reduce_axes(x, axes)
  if (nan_rm && is_dtype_float(dtype(x))) {
    x <- nv_ifelse(nv_is_nan(x), 0, x)
  }
  prim_reduce_sum(x, axes = axes, drop = drop)
}

#' @title Mean
#' @description
#' Computes the arithmetic mean along the specified axes. You can also
#' use `mean()`.
#' @template param_x
#' @template params_reduce
#' @template return_reduce
#' @template param_nan_rm
#' @seealso [nv_reduce_sum()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' nv_mean(x)            # all axes -> scalar
#' nv_mean(x, axes = 1L)
#' nv_mean(nv_array(c(1, NaN, 3)))
#' nv_mean(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#' @export
#' @jit static 2:4
nv_mean <- function(x, axes = NULL, drop = TRUE, nan_rm = FALSE) {
  x <- as_anvl_array(x)
  axes <- .resolve_reduce_axes(x, axes)
  if (nan_rm && is_dtype_float(dtype(x))) {
    is_nan <- nv_is_nan(x)
    total <- prim_reduce_sum(nv_ifelse(is_nan, 0, x), axes = axes, drop = drop)
    count <- prim_reduce_sum(nv_convert(!is_nan, "i32"), axes = axes, drop = drop)
    return(total / count)
  }
  nelts <- prod(shape(x)[axes])
  nv_reduce_sum(x, axes, drop) / nelts
}

#' @title Product Reduction
#' @description
#' Multiplies array elements along the specified axes.
#' @template param_x
#' @template params_reduce
#' @template return_reduce
#' @template param_nan_rm
#' @seealso [prim_reduce_prod()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' nv_reduce_prod(x)            # all axes -> scalar
#' nv_reduce_prod(x, axes = 1L)
#' nv_reduce_prod(nv_array(c(2, NaN, 3)))
#' nv_reduce_prod(nv_array(c(2, NaN, 3)), nan_rm = TRUE)
#' @export
nv_reduce_prod <- function(x, axes = NULL, drop = TRUE, nan_rm = FALSE) {
  x <- as_anvl_array(x)
  axes <- .resolve_reduce_axes(x, axes)
  if (nan_rm && is_dtype_float(dtype(x))) {
    x <- nv_ifelse(nv_is_nan(x), 1, x)
  }
  prim_reduce_prod(x, axes = axes, drop = drop)
}

#' @title Max Reduction
#' @description
#' Finds the maximum of array elements along the specified axes.
#' @template param_x
#' @template params_reduce
#' @template param_nan_rm
#' @template return_reduce
#' @seealso [prim_reduce_max()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' nv_reduce_max(x)            # all axes -> scalar
#' nv_reduce_max(x, axes = 1L)
#' nv_reduce_max(nv_array(c(1, NaN, 3)))
#' nv_reduce_max(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#' @export
nv_reduce_max <- function(x, axes = NULL, drop = TRUE, nan_rm = FALSE) {
  x <- as_anvl_array(x)
  axes <- .resolve_reduce_axes(x, axes)
  .nv_reduce_extreme(x, axes, drop, nan_rm, -Inf, prim_reduce_max)
}

#' @title Min Reduction
#' @description
#' Finds the minimum of array elements along the specified axes.
#' @template param_x
#' @template params_reduce
#' @template param_nan_rm
#' @template return_reduce
#' @seealso [prim_reduce_min()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' nv_reduce_min(x)            # all axes -> scalar
#' nv_reduce_min(x, axes = 1L)
#' nv_reduce_min(nv_array(c(1, NaN, 3)))
#' nv_reduce_min(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#' @export
nv_reduce_min <- function(x, axes = NULL, drop = TRUE, nan_rm = FALSE) {
  x <- as_anvl_array(x)
  axes <- .resolve_reduce_axes(x, axes)
  .nv_reduce_extreme(x, axes, drop, nan_rm, Inf, prim_reduce_min)
}

# Shared NaN-aware reduction for max/min. Unlike sum/prod (which lower to
# arithmetic that propagates NaN), the min/max kernels are comparison-based
# and silently drop NaN, so `nan_rm = FALSE` needs an explicit any-NaN mask
# to re-inject NaN — no input substitution can coax the kernel into emitting
# NaN on output.
.nv_reduce_extreme <- function(x, axes, drop, nan_rm, identity_val, prim_reduce) {
  if (!is_dtype_float(dtype(x))) {
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

#' @title Any Reduction
#' @description
#' Performs logical OR along the specified axes.
#' Returns `TRUE` if any element is `TRUE`.
#' @template param_x
#' @template params_reduce
#' @template return_reduce_boolean
#' @seealso [prim_reduce_any()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
#' nv_reduce_any(x)            # all axes -> scalar
#' nv_reduce_any(x, axes = 1L)
#' @export
nv_reduce_any <- function(x, axes = NULL, drop = TRUE) {
  x <- as_anvl_array(x)
  prim_reduce_any(x, axes = .resolve_reduce_axes(x, axes), drop = drop)
}

#' @title All Reduction
#' @description
#' Performs logical AND along the specified axes.
#' Returns `TRUE` only if all elements are `TRUE`.
#' @template param_x
#' @template params_reduce
#' @template return_reduce_boolean
#' @seealso [prim_reduce_all()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
#' nv_reduce_all(x)            # all axes -> scalar
#' nv_reduce_all(x, axes = 1L)
#' @export
nv_reduce_all <- function(x, axes = NULL, drop = TRUE) {
  x <- as_anvl_array(x)
  prim_reduce_all(x, axes = .resolve_reduce_axes(x, axes), drop = drop)
}

#' @title Cumulative Sum
#' @description
#' Cumulative sum, optionally along a single axis.
#' @template param_x
#' @templateVar cum_base_fn cumsum
#' @template param_nv_cum_axis
#' @template param_nan_rm_cum
#' @template return_unary
#' @templateVar cum_nv_name nv_cumsum
#' @template section_nv_cum_relation
#' @seealso [prim_cumsum()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' nv_cumsum(x)              # row-major flatten, then accumulate
#' nv_cumsum(x, axis = 1L)    # accumulate along rows
#' nv_cumsum(nv_array(c(1, NaN, 3)))                # NaN propagates
#' nv_cumsum(nv_array(c(1, NaN, 3)), nan_rm = TRUE) # NaN treated as 0
#' @export
#' @jit static 2:3
nv_cumsum <- function(x, axis = NULL, nan_rm = FALSE) {
  x <- as_anvl_array(x)
  if (is.null(axis)) {
    x <- nv_reshape(x, prod(shape(x)))
    axis <- 1L
  }
  if (nan_rm && is_dtype_float(dtype(x))) {
    x <- nv_ifelse(nv_is_nan(x), 0, x)
  }
  prim_cumsum(x, axis = axis)
}

#' @title Cumulative Product
#' @description
#' Cumulative product, optionally along a single axis.
#' @template param_x
#' @templateVar cum_base_fn cumprod
#' @template param_nv_cum_axis
#' @template param_nan_rm_cum
#' @template return_unary
#' @templateVar cum_nv_name nv_cumprod
#' @template section_nv_cum_relation
#' @seealso [prim_cumprod()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' nv_cumprod(x)              # row-major flatten, then accumulate
#' nv_cumprod(x, axis = 1L)    # accumulate along rows
#' nv_cumprod(nv_array(c(2, NaN, 3)))                # NaN propagates
#' nv_cumprod(nv_array(c(2, NaN, 3)), nan_rm = TRUE) # NaN treated as 1
#' @export
#' @jit static 2:3
nv_cumprod <- function(x, axis = NULL, nan_rm = FALSE) {
  x <- as_anvl_array(x)
  if (is.null(axis)) {
    x <- nv_reshape(x, prod(shape(x)))
    axis <- 1L
  }
  if (nan_rm && is_dtype_float(dtype(x))) {
    x <- nv_ifelse(nv_is_nan(x), 1, x)
  }
  prim_cumprod(x, axis = axis)
}

#' @title Cumulative Maximum
#' @description
#' Running maximum, optionally along a single axis.
#' @template param_x
#' @templateVar cum_base_fn cummax
#' @template param_nv_cum_axis
#' @templateVar cum_extreme_name maximum
#' @template param_nv_cum_with_indices
#' @template return_nv_cum_extreme
#' @templateVar cum_nv_name nv_cummax
#' @template section_nv_cum_relation
#' @template param_nan_rm_cum
#' @seealso [prim_cummax()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(3, 1, 4, 1, 5, 9), nrow = 2)
#' nv_cummax(x)
#' nv_cummax(x, axis = 1L)
#' nv_cummax(x, axis = 1L, with_indices = TRUE)
#' nv_cummax(nv_array(c(1, NaN, 3)))                # NaN propagates
#' nv_cummax(nv_array(c(1, NaN, 3)), nan_rm = TRUE) # NaN skipped
#' @export
#' @jit static 2:4
nv_cummax <- function(x, axis = NULL, with_indices = FALSE, nan_rm = FALSE) {
  .nv_cum_extreme(x, axis, with_indices, nan_rm, -Inf, prim_cummax)
}

#' @title Cumulative Minimum
#' @description
#' Running minimum, optionally along a single axis.
#' @template param_x
#' @templateVar cum_base_fn cummin
#' @template param_nv_cum_axis
#' @templateVar cum_extreme_name minimum
#' @template param_nv_cum_with_indices
#' @template return_nv_cum_extreme
#' @templateVar cum_nv_name nv_cummin
#' @template section_nv_cum_relation
#' @template param_nan_rm_cum
#' @seealso [prim_cummin()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(3, 1, 4, 1, 5, 9), nrow = 2)
#' nv_cummin(x)
#' nv_cummin(x, axis = 1L)
#' nv_cummin(x, axis = 1L, with_indices = TRUE)
#' nv_cummin(nv_array(c(3, NaN, 1)))                # NaN propagates
#' nv_cummin(nv_array(c(3, NaN, 1)), nan_rm = TRUE) # NaN skipped
#' @export
#' @jit static 2:4
nv_cummin <- function(x, axis = NULL, with_indices = FALSE, nan_rm = FALSE) {
  .nv_cum_extreme(x, axis, with_indices, nan_rm, Inf, prim_cummin)
}

# NaN propagation for the default `nan_rm = FALSE` path is now handled in
# `prim_cummax` / `prim_cummin`'s lowering directly. Here we only need to
# sanitize NaN → identity for `nan_rm = TRUE`.
.nv_cum_extreme <- function(x, axis, with_indices, nan_rm, identity_val, prim_cum) {
  x <- as_anvl_array(x)
  if (is.null(axis)) {
    x <- nv_reshape(x, prod(shape(x)))
    axis <- 1L
  }
  if (nan_rm && is_dtype_float(dtype(x))) {
    x <- nv_ifelse(nv_is_nan(x), identity_val, x)
  }
  out <- prim_cum(x, axis = axis)
  if (with_indices) list(values = out[[1L]], indices = out[[2L]]) else out[[1L]]
}

# Higher order primitives

#' @title Conditional Branching
#' @description
#' Conditional execution of two branches.
#' Unlike [nv_ifelse()], which selects elements, this executes only one
#' of the two branches depending on a scalar predicate.
#' @param pred ([`arrayish`] of boolean type, scalar)\cr
#'   Predicate.
#' @param true (`function()`)\cr
#'   Zero-argument function for the true branch.
#' @param false (`function()`)\cr
#'   Zero-argument function for the false branch.
#'   Must return outputs with the same shapes as the true branch.
#' @return Result of the executed branch.
#' @seealso [prim_if()] for the underlying primitive, [nv_ifelse()] for
#'   element-wise selection.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_if(nv_scalar(TRUE), \() nv_scalar(1), \() nv_scalar(2))
#' @export
nv_if <- prim_if

#' @title While Loop
#' @description
#' Executes a functional while loop.
#' @param init (`list()`)\cr
#'   Named list of initial state values.
#' @param cond (`function`)\cr
#'   Condition function returning a scalar boolean.
#'   Receives the state values as arguments.
#' @param body (`function`)\cr
#'   Body function returning the updated state as a named list
#'   with the same structure as `init`.
#' @return Final state after the loop terminates (same structure as `init`).
#' @seealso [prim_while()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
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

## Additional math functions ---------------------------------------------------

#' @title Base-2 Logarithm
#' @description
#' Element-wise base-2 logarithm. You can also use `log2()`.
#' @template param_x
#' @template return_unary
#' @seealso [nv_log()], [nv_log10()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 4, 8))
#' nv_log2(x)
#' @export
#' @jit
nv_log2 <- function(x) {
  x <- as_anvl_array(x)
  nv_log(x) / log(2)
}

#' @title Base-10 Logarithm
#' @description
#' Element-wise base-10 logarithm. You can also use `log10()`.
#' @template param_x
#' @template return_unary
#' @seealso [nv_log()], [nv_log2()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 10, 100, 1000))
#' nv_log10(x)
#' @export
#' @jit
nv_log10 <- function(x) {
  x <- as_anvl_array(x)
  nv_log(x) / log(10)
}

#' @title Is NaN
#' @description
#' Element-wise check if values are NaN. You can also use `is.nan()`.
#' @template param_x
#' @template return_unary_boolean
#' @seealso [nv_is_finite()], [nv_is_infinite()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, NaN, Inf, -Inf, 0))
#' nv_is_nan(x)
#' @export
#' @jit
nv_is_nan <- function(x) {
  x <- as_anvl_array(x)
  x != x
}

#' @title Is Infinite
#' @description
#' Element-wise check if values are infinite (`Inf` or `-Inf`).
#' You can also use `is.infinite()`.
#' @template param_x
#' @template return_unary_boolean
#' @seealso [nv_is_finite()], [nv_is_nan()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, NaN, Inf, -Inf, 0))
#' nv_is_infinite(x)
#' @export
nv_is_infinite <- function(x) {
  x <- as_anvl_array(x)
  !nv_is_finite(x) & (x == x)
}

## Reduction operations --------------------------------------------------------

#' @title Variance
#' @description
#' Computes the variance along the specified axes.
#' @details
#' Uses Bessel's correction by default (`correction = 1`), matching R's [var()].
#' Set `correction = 0` for population variance.
#' @template param_x
#' @template params_reduce
#' @param correction (`integer(1)`)\cr
#'   Degrees of freedom correction. Default is `1` (Bessel's correction).
#' @template param_nan_rm
#' @template return_reduce
#' @seealso [nv_sd()], [nv_mean()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3, 4, 5))
#' nv_var(x)             # all axes -> scalar
#' nv_var(x, axes = 1L)
#' nv_var(nv_array(c(1, NaN, 3, 5)), axes = 1L, nan_rm = TRUE)
#' @export
#' @jit static 2:5
nv_var <- function(x, axes = NULL, drop = TRUE, correction = 1L, nan_rm = FALSE) {
  x <- as_anvl_array(x)
  assert_int(correction)
  axes <- .resolve_reduce_axes(x, axes)
  mean_bc <- nv_broadcast_to(
    nv_mean(x, axes, drop = FALSE, nan_rm = nan_rm),
    shape(x)
  )
  diff <- x - mean_bc
  ssum <- nv_reduce_sum(diff * diff, axes, drop, nan_rm = nan_rm)
  if (nan_rm && is_dtype_float(dtype(x))) {
    count <- nv_reduce_sum(nv_convert(!nv_is_nan(x), "i32"), axes, drop)
    # When count <= correction the divisor clamps to 0 and ssum is 0
    # (single non-NaN point has zero deviation, all-NaN slice contributes
    # nothing), so 0/0 = NaN propagates naturally — no explicit mask needed.
    return(ssum / nv_max(0, count - correction))
  }
  nelts <- prod(shape(x)[axes])
  ssum / max(0L, nelts - correction)
}

#' @title Standard Deviation
#' @description
#' Computes the standard deviation along the specified axes.
#' @details
#' Uses Bessel's correction by default (`correction = 1`), matching R's [sd()].
#' Set `correction = 0` for population standard deviation.
#' @template param_x
#' @template params_reduce
#' @param correction (`integer(1)`)\cr
#'   Degrees of freedom correction. Default is `1` (Bessel's correction).
#' @template param_nan_rm
#' @template return_reduce
#' @seealso [nv_var()], [nv_mean()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3, 4, 5))
#' nv_sd(x)              # all axes -> scalar
#' nv_sd(x, axes = 1L)
#' @export
#' @jit static 2:5
nv_sd <- function(x, axes = NULL, drop = TRUE, correction = 1L, nan_rm = FALSE) {
  nv_sqrt(nv_var(x, axes, drop, correction, nan_rm = nan_rm))
}

## Array manipulation ----------------------------------------------------------

#' @title Squeeze
#' @description
#' Removes axes of size 1 from an array.
#' @template param_x
#' @param axes (`integer()` | `NULL`)\cr
#'   Axes to squeeze. Negative values count from the end, i.e. `-1`
#'   refers to the last axis.
#'   If `NULL` (default), all axes of size 1 are removed.
#' @return [`arrayish`]\cr
#'   Has the same data type as `x` with the specified axes removed.
#' @seealso [nv_unsqueeze()], [nv_reshape()]
#' @examplesIf pjrt::plugins_downloaded()
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
    new_shape <- integer(0)
  }
  nv_reshape(x, new_shape)
}

#' @title Unsqueeze
#' @description
#' Inserts an axis of size 1 at the specified position.
#' @template param_x
#' @param axis (`integer(1)`)\cr
#'   Position at which to insert the new axis. Valid positions range from
#'   1 to `naxes(x) + 1`. Negative values count from the end of the
#'   *result*, i.e. `-1` appends the new axis at the end.
#' @return [`arrayish`]\cr
#'   Has the same data type as `x` with an extra axis of size 1.
#' @seealso [nv_squeeze()], [nv_reshape()]
#' @examplesIf pjrt::plugins_downloaded()
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
#'   1-D arrays.
#' @return [`arrayish`]\cr
#'   A 2-D array of shape `(length(lhs), length(rhs))`.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5))
#' nv_outer(x, y)
#' @export
#' @jit
nv_outer <- function(lhs, rhs) {
  args <- nv_promote_to_common(lhs, rhs)
  lhs <- args[[1L]]
  rhs <- args[[2L]]
  if (naxes(lhs) != 1L) {
    cli_abort("lhs must be a 1-D array")
  }
  if (naxes(rhs) != 1L) {
    cli_abort("rhs must be a 1-D array")
  }
  lhs_exp <- nv_unsqueeze(lhs, axis = 2L)
  rhs_exp <- nv_unsqueeze(rhs, axis = 1L)
  bcast <- nv_broadcast_arrays(lhs_exp, rhs_exp)
  prim_mul(bcast[[1L]], bcast[[2L]])
}

#' @title Extract Diagonal
#' @description
#' Extracts the diagonal elements from a 2-D array.
#' @template param_x
#' @return [`arrayish`]\cr
#'   A 1-D array of length `min(nrow, ncol)` containing the diagonal elements.
#' @seealso [nv_diag()] for creating a diagonal matrix, [nv_trace()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:9, shape = c(3, 3))
#' nv_extract_diag(x)
#' @export
#' @jit
nv_extract_diag <- function(x) {
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
    offset_axes = integer(0),
    collapsed_slice_axes = c(1L, 2L),
    x_batching_axes = integer(0),
    start_indices_batching_axes = integer(0),
    start_index_map = c(1L, 2L),
    index_vector_axis = 2L,
    slice_sizes = c(1L, 1L)
  )
}

#' @title Matrix Trace
#' @description
#' Computes the trace (sum of diagonal elements) of a 2-D array.
#' @template param_x
#' @return [`arrayish`]\cr
#'   A scalar with the same data type as `x`.
#' @seealso [nv_extract_diag()], [nv_diag()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 0, 0, 0, 2, 0, 0, 0, 3), shape = c(3, 3))
#' nv_trace(x)
#' @export
#' @jit
nv_trace <- function(x) {
  x <- as_anvl_array(x)
  diag_vals <- nv_extract_diag(x)
  nv_reduce_sum(diag_vals, axes = 1L, drop = TRUE)
}

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
#' @template param_shape
#' @param diagonal (`integer(1)`)\cr
#'   Diagonal offset, with the same meaning as in [nv_tril()]. The default
#'   `-1` excludes the main diagonal, matching `lower.tri()`; use `0` to
#'   include it, matching `lower.tri(diag = TRUE)`.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_lower_tri_like()`).
#' @template param_device
#' @return [`arrayish`]\cr
#'   Has the given `shape` and dtype `bool`.
#' @seealso [nv_upper_tri()], [nv_tril()], [prim_iota()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_lower_tri(c(3, 3))
#' nv_lower_tri(c(3, 3), diagonal = 0L)
#' x <- nv_fill(0, shape = c(3, 3))
#' nv_lower_tri_like(x)
#' @export
#' @jit static 1:3
nv_lower_tri <- function(shape, diagonal = -1L, device = NULL) {
  assert_tri_args(shape, diagonal)
  tri_mask(shape, as.integer(diagonal), lower = TRUE, device = device)
}

#' @title Upper Triangular Mask
#' @description
#' Returns a boolean matrix that is `TRUE` on and above the given diagonal,
#' mirroring base R's `upper.tri()`. Use [nv_triu()] to zero out the other
#' triangle of an existing array instead.
#' @template param_shape
#' @param diagonal (`integer(1)`)\cr
#'   Diagonal offset, with the same meaning as in [nv_triu()]. The default
#'   `1` excludes the main diagonal, matching `upper.tri()`; use `0` to
#'   include it, matching `upper.tri(diag = TRUE)`.
#' @param like ([`AnvlArray`])\cr
#'   Existing array whose attributes are used as defaults
#'   (only for `nv_upper_tri_like()`).
#' @template param_device
#' @return [`arrayish`]\cr
#'   Has the given `shape` and dtype `bool`.
#' @seealso [nv_lower_tri()], [nv_triu()], [prim_iota()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_upper_tri(c(3, 3))
#' nv_upper_tri(c(3, 3), diagonal = 0L)
#' x <- nv_fill(0, shape = c(3, 3))
#' nv_upper_tri_like(x)
#' @export
#' @jit static 1:3
nv_upper_tri <- function(shape, diagonal = 1L, device = NULL) {
  assert_tri_args(shape, diagonal)
  tri_mask(shape, as.integer(diagonal), lower = FALSE, device = device)
}

#' @title Lower Triangular Matrix
#' @description
#' Returns the lower triangular part of a 2-D array, setting elements above
#' the specified diagonal to zero.
#' @template param_x
#' @param diagonal (`integer(1)`)\cr
#'   Diagonal offset. `0` (default) is the main diagonal, positive values
#'   include diagonals above, negative values exclude diagonals below.
#' @return [`arrayish`]\cr
#'   Has the same shape and data type as `x`.
#' @seealso [nv_triu()], [nv_lower_tri()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_fill(1, c(3, 3))
#' nv_tril(x)
#' @export
#' @jit static 2L
nv_tril <- function(x, diagonal = 0L) {
  x <- as_anvl_array(x)
  if (naxes(x) != 2L) {
    cli_abort("{.arg x} must be a 2-D array")
  }
  nv_ifelse(nv_lower_tri_like(x, diagonal), x, nv_fill_like(x, 0))
}

#' @title Upper Triangular Matrix
#' @description
#' Returns the upper triangular part of a 2-D array, setting elements below
#' the specified diagonal to zero.
#' @template param_x
#' @param diagonal (`integer(1)`)\cr
#'   Diagonal offset. `0` (default) is the main diagonal, positive values
#'   exclude diagonals above, negative values include diagonals below.
#' @return [`arrayish`]\cr
#'   Has the same shape and data type as `x`.
#' @seealso [nv_tril()], [nv_upper_tri()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_fill(1, c(3, 3))
#' nv_triu(x)
#' @export
#' @jit static 2L
nv_triu <- function(x, diagonal = 0L) {
  x <- as_anvl_array(x)
  if (naxes(x) != 2L) {
    cli_abort("{.arg x} must be a 2-D array")
  }
  nv_ifelse(nv_upper_tri_like(x, diagonal), x, nv_fill_like(x, 0))
}

#' @title Cross Product (Matrix)
#' @description
#' Computes `t(lhs) %*% rhs`. If `rhs` is missing, computes `t(lhs) %*% lhs`.
#' @param lhs ([`arrayish`])\cr
#'   An array with at least 2 axes.
#' @param rhs ([`arrayish`] | `NULL`)\cr
#'   Optional second array. If `NULL`, uses `lhs`.
#' @return [`arrayish`]
#' @seealso [nv_tcrossprod()], [nv_matmul()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 3, dtype = "f32")
#' nv_crossprod(x)
#' @export
#' @jit
nv_crossprod <- function(lhs, rhs = NULL) {
  if (is.null(rhs)) {
    lhs <- as_anvl_array(lhs)
    rhs <- lhs
  } else {
    args <- as_anvl_arrays(lhs, rhs)
    lhs <- args[[1L]]
    rhs <- args[[2L]]
  }
  nv_matmul(nv_transpose(lhs), rhs)
}

#' @title Transpose Cross Product (Matrix)
#' @description
#' Computes `lhs %*% t(rhs)`. If `rhs` is missing, computes `lhs %*% t(lhs)`.
#' @param lhs ([`arrayish`])\cr
#'   An array with at least 2 axes.
#' @param rhs ([`arrayish`] | `NULL`)\cr
#'   Optional second array. If `NULL`, uses `lhs`.
#' @return [`arrayish`]
#' @seealso [nv_crossprod()], [nv_matmul()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2, dtype = "f32")
#' nv_tcrossprod(x)
#' @export
#' @jit
nv_tcrossprod <- function(lhs, rhs = NULL) {
  if (is.null(rhs)) {
    lhs <- as_anvl_array(lhs)
    rhs <- lhs
  } else {
    args <- as_anvl_arrays(lhs, rhs)
    lhs <- args[[1L]]
    rhs <- args[[2L]]
  }
  nv_matmul(lhs, nv_transpose(rhs))
}

# Sorting and searching --------------------------------------------------------

#' @title Select Elements Along an Axis
#' @description
#' Picks one or more elements along axis `axis` of `x`.
#' Use this instead of `[` or `nv_subset` when the index to select is provided
#' programmatically.
#' @template param_x
#' @param axis (`integer(1)`)\cr
#'   Axis to index into.
#'   Negative values count from the end, i.e. `-1` refers to the last axis.
#' @param index ([`arrayish`])\cr
#'   Scalar or 1D arrayish input (integer).
#' @return [`arrayish`]\cr
#'   Same data type as `x`. `axis` is dropped if `index` was scalar.
#' @seealso [nv_subset()] for general subsetting, [prim_static_slice()].
#' @examplesIf pjrt::plugins_downloaded()
#' m <- nv_matrix(1:6, nrow = 2)
#' nv_select(m, axis = 2L, index = 2L)
#' nv_select(m, axis = 1L, index = 1L)
#' nv_select(m, axis = 2L, index = array(c(1L, 3L)))
#' @export
nv_select <- function(x, axis, index) {
  x <- as_anvl_array(x)
  rank <- naxes(x)
  if (rank == 0L) {
    cli_abort("Cannot select along a 0-dimensional array")
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
    offset_axes = integer(0),
    collapsed_slice_axes = seq_len(rank),
    x_batching_axes = integer(0),
    start_indices_batching_axes = integer(0),
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
#' @template param_x
#' @param axis (`integer(1)` | `NULL`)\cr
#'   Axis along which to sort. Negative values count from the end,
#'   i.e. `-1` refers to the last axis. If `NULL` (default), uses the last
#'   axis.
#' @param decreasing (`logical(1)`)\cr
#'   If `TRUE`, sort in decreasing order. Default `FALSE`.
#' @param stable (`logical(1)`)\cr
#'   If `TRUE`, the sort is stable: equal values keep their original
#'   relative order along `axis`. Default `FALSE`. Stability is only
#'   observable for floats when `-0` / `+0` or `-NaN` / `+NaN` are mixed
#'   (they compare equal under the total order used here); for distinct
#'   values the result is identical either way.
#' @return [`arrayish`]\cr
#'   Same shape and data type as `x`.
#' @section NaN handling:
#' `NaN` values sort to the **end** (ascending) or **beginning**
#' (descending), regardless of sign. `+0` and `-0` compare equal.
#' @seealso [prim_sort()] for the underlying primitive,
#'   [nv_argsort()], [nv_top_k()], [nv_median()],
#'   [nv_argmax()], [nv_argmin()].
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
#' nv_sort(x)
#' sort(x) # via the S3 generic
#' nv_sort(x, decreasing = TRUE)
#'
#' m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
#' nv_sort(m, axis = 2L)
#' @export
nv_sort <- function(x, axis = NULL, decreasing = FALSE, stable = FALSE) {
  x <- as_anvl_array(x)
  if (naxes(x) == 0L) {
    cli_abort("Cannot sort a 0-dimensional array")
  }
  prim_sort(list(x), axis = axis %||% naxes(x), descending = decreasing, is_stable = stable)[[1L]]
}

#' @title Argsort
#' @description
#' Returns the indices that would sort the array along an axis.
#' @template param_x
#' @param axis (`integer(1)` | `NULL`)\cr
#'   Axis along which to compute the sort permutation. Negative values
#'   count from the end, i.e. `-1` refers to the last axis. If `NULL`
#'   (default), uses the last axis.
#' @param decreasing (`logical(1)`)\cr
#'   If `TRUE`, returns indices that produce a decreasing sort. Default
#'   `FALSE`.
#' @param stable (`logical(1)`)\cr
#'   If `TRUE`, the sort is stable: indices for equal values keep their
#'   original relative order. Default `FALSE`.
#' @return [`arrayish`] of dtype `i32`\cr
#'   Same shape as `x`. For a size-0 axis, the output is an empty `i32`
#'   array of the same shape (a valid empty permutation).
#'   `as_array(x)[as_array(nv_argsort(x))]` reproduces the sorted
#'   array (for 1-D inputs).
#' @inheritSection nv_sort NaN handling
#' @seealso [nv_sort()], [prim_sort()].
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(3, 1, 4, 1, 5))
#' nv_argsort(x)
#' @export
#' @jit static 2:4
nv_argsort <- function(x, axis = NULL, decreasing = FALSE, stable = FALSE) {
  x <- as_anvl_array(x)
  if (naxes(x) == 0L) {
    cli_abort("Cannot argsort a 0-dimensional array")
  }
  axis <- axis %||% naxes(x)
  idx <- nv_iota_like(x, axis = axis, dtype = "i32")
  prim_sort(list(x, idx), axis = axis, descending = decreasing, is_stable = stable)[[2L]]
}

#' @title Top-K Elements
#' @description
#' Returns the `k` largest values along an axis, sorted in decreasing order.
#' @template param_x
#' @param k (`integer(1)`)\cr
#'   Number of top elements to return. Must satisfy
#'   `1 <= k <= shape(x)[axis]`.
#' @param axis (`integer(1)` | `NULL`)\cr
#'   Axis along which to take the top `k`. Negative values count from the
#'   end, i.e. `-1` refers to the last axis. If `NULL` (default),
#'   uses the last axis.
#' @param with_indices (`logical(1)`)\cr
#'   If `FALSE` (default), returns just the top-`k` values. If `TRUE`,
#'   returns `list(values = ..., indices = ...)` where `indices` is the
#'   1-based position of each top-`k` value along `axis` (dtype `i32`).
#' @return [`arrayish`] (when `with_indices = FALSE`) or named list of two
#'   arrays (when `with_indices = TRUE`). Output shape matches `x` with
#'   `axis` resized to `k`; values are sorted decreasing along `axis`.
#' @section NaN handling:
#' `NaN` ranks larger than any finite value (so it appears first in the
#' top-`k` output); `-NaN` ranks smaller. Unlike [nv_sort()], the sign
#' bit is not canonicalized.
#' @seealso [prim_top_k()] for the underlying primitive, [nv_sort()].
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
#' nv_top_k(x, k = 3L)
#' nv_top_k(x, k = 3L, with_indices = TRUE)
#'
#' m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
#' nv_top_k(m, k = 2L, axis = 2L)
#' @export
#' @jit static 2:4
nv_top_k <- function(x, k, axis = NULL, with_indices = FALSE) {
  x <- as_anvl_array(x)
  rank <- naxes(x)
  if (rank == 0L) {
    cli_abort("Cannot take top-k of a 0-dimensional array")
  }
  axis <- resolve_axis(axis %||% rank, rank, arg = "axis")
  k <- as.integer(k)
  assert_int(k, lower = 1L, upper = shape(x)[axis])

  # prim_top_k operates on the last axis; transpose axis to last and back.
  if (axis != rank) {
    perm <- seq_len(rank)
    perm[c(axis, rank)] <- c(rank, axis)
    out <- prim_top_k(prim_transpose(x, permutation = perm), k = k)
    values <- prim_transpose(out[[1L]], permutation = perm)
    if (with_indices) {
      indices <- prim_transpose(out[[2L]], permutation = perm)
      list(values = values, indices = indices)
    } else {
      values
    }
  } else {
    out <- prim_top_k(x, k = k)
    if (with_indices) list(values = out[[1L]], indices = out[[2L]]) else out[[1L]]
  }
}

#' @title Quantile
#' @description
#' Computes the `probs` quantile(s) of an array along an axis.
#'
#' `probs` follows the same scalar-vs-array convention as [nv_select()]'s
#' `index`:
#'
#' * a length-1 numeric (e.g. `0.5`) treats `probs` as scalar — the output
#'   has `axis` removed, like a reduction;
#' * a 1-D R array (e.g. `array(c(0.25, 0.5, 0.75))`) prepends a leading
#'   axis of size `length(probs)`.
#'
#' Plain length-K (K > 1) vectors are rejected; wrap with `array()` to
#' make the array intent explicit.
#' @section Interpolation modes:
#' Let `h = (n - 1) * q` be the 0-based fractional index for an axis of
#' length `n` and probability `q`, with `lo = floor(h)`, `hi = ceil(h)`,
#' `frac = h - lo`. Then:
#'
#' * `"linear"` (default): `(1 - frac) * sorted[lo] + frac * sorted[hi]`.
#' * `"lower"`: `sorted[lo]` — the lower bracket of `linear`.
#' * `"higher"`: `sorted[hi]` — the upper bracket of `linear`.
#' * `"nearest"`: `sorted[lo]` if `frac < 0.5` else `sorted[hi]`.
#' * `"midpoint"`: `(sorted[lo] + sorted[hi]) / 2`.
#' @template param_x
#' @param probs (`numeric(1)` | 1-D `array`)\cr
#'   One or more probabilities in `[0, 1]`. Either a length-1 numeric
#'   (scalar; `axis` is dropped) or a 1-D `array` (a leading axis of size
#'   `length(probs)` is prepended). Plain length-K (K > 1) vectors are
#'   rejected — wrap with `array()`.
#' @param axis (`integer(1)` | `NULL`)\cr
#'   Axis along which to compute the quantile. Negative values count from
#'   the end, i.e. `-1` refers to the last axis. If `NULL` (default),
#'   uses the last axis.
#' @param interpolation (`character(1)`)\cr
#'   One of `"linear"` (default), `"lower"`, `"higher"`, `"nearest"`,
#'   `"midpoint"`. See "Interpolation modes".
#' @template param_nan_rm
#' @return [`arrayish`]\cr
#'   For scalar `probs`: same shape as `x` with `axis` removed. For
#'   array `probs`: a **leading** axis of size `length(probs)` is
#'   prepended.
#' @seealso [nv_median()], [nv_sort()].
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
#' nv_quantile(x, 0.5) # = nv_median(x)
#' nv_quantile(x, array(c(0.25, 0.5, 0.75)))
#' nv_quantile(x, 0.5, interpolation = "lower")
#' nv_quantile(nv_array(c(1, NaN, 3, 5)), 0.5)
#' nv_quantile(nv_array(c(1, NaN, 3, 5)), 0.5, nan_rm = TRUE)
#' @export
#' @jit static 2:5
nv_quantile <- function(x, probs, axis = NULL, interpolation = "linear", nan_rm = FALSE) {
  x <- as_anvl_array(x)
  rank <- naxes(x)
  if (rank == 0L) {
    cli_abort("Cannot compute quantile of a 0-dimensional array")
  }
  assert_choice(interpolation, c("linear", "lower", "higher", "nearest", "midpoint"))
  if (!is_valid_r(probs)) {
    cli_abort("{.arg probs} must either be a length-1 numeric or 1-D R array.")
  }
  checkmate::assert_numeric(probs, lower = 0, upper = 1, any.missing = FALSE, min.len = 1L)

  is_probs_array <- !is.null(dim(probs))
  axis <- resolve_axis(axis %||% rank, rank, arg = "axis")
  shp <- shape(x)
  K <- length(probs)
  probs <- as.numeric(probs)
  is_float <- is_dtype_float(dtype(x))
  shp_kd <- replace(shp, axis, 1L)
  shp_K <- replace(shp, axis, K)

  # For float input, find NaN positions: nan_rm = TRUE sanitizes them to +Inf
  # so they sort to the end; nan_rm = FALSE uses them post-hoc to propagate.
  if (is_float) {
    nan_mask <- nv_is_nan(x)
    to_sort <- if (nan_rm) nv_ifelse(nan_mask, Inf, x) else x
    n_valid_kd <- if (nan_rm) {
      prim_reduce_sum(nv_convert(!nan_mask, "i32"), axes = axis, drop = FALSE)
    } else {
      nv_broadcast_to(nv_array_like(x, shp[axis], shape = integer()), shp_kd)
    }
  } else {
    to_sort <- x
    n_valid_kd <- nv_broadcast_to(nv_array_like(x, shp[axis], shape = integer()), shp_kd)
  }
  sorted <- prim_sort(list(to_sort), axis = axis)[[1L]]

  # Broadcast `(K,) probs` along `axis` and `(shp_kd,) n_valid_kd` across
  # `axis` → both shaped `shp_K`, with K varying along `axis`.
  probs_shape <- replace(rep(1L, rank), axis, K)
  probs_b <- nv_broadcast_to(
    prim_reshape(nv_array_like(sorted, probs, shape = K), probs_shape),
    shp_K
  )
  n_valid_b <- nv_broadcast_to(n_valid_kd, shp_K)
  h <- (n_valid_b - 1) * probs_b
  lo_f <- nv_floor(h)
  hi_f <- nv_ceiling(h)
  frac <- h - lo_f

  lo_val <- .gather_along_axis(sorted, nv_convert(lo_f + 1, "i32"), axis, rank, shp)
  hi_val <- .gather_along_axis(sorted, nv_convert(hi_f + 1, "i32"), axis, rank, shp)

  out <- switch(
    interpolation,
    "lower" = lo_val,
    "higher" = hi_val,
    "nearest" = nv_ifelse(frac < 0.5, lo_val, hi_val),
    "linear" = lo_val * (1 - frac) + hi_val * frac,
    "midpoint" = (lo_val + hi_val) / 2
  )

  # Propagate NaN: nan_rm = TRUE produces NaN only for all-NaN slices;
  # nan_rm = FALSE produces NaN for any slice that contained a NaN (XLA's
  # sort places NaN unpredictably, so we can't rely on the gather hitting it).
  if (is_float) {
    bad <- if (nan_rm) n_valid_kd == 0 else prim_reduce_any(nan_mask, axes = axis, drop = FALSE)
    out <- nv_ifelse(nv_broadcast_to(bad, shp_K), NaN, out)
  }

  # For scalar probs, drop the (now size-1) reduced axis. For array probs,
  # move the K axis (currently at `axis`) to the front.
  if (is_probs_array) {
    prim_transpose(out, permutation = c(axis, seq_len(rank)[-axis]))
  } else {
    prim_reshape(out, shp[-axis])
  }
}


#' @title Median
#' @name nv_median
#' @description
#' Computes the median along an axis. Equivalent to
#' `nv_quantile(x, 0.5, axis, interpolation)`; for an even-length axis
#' with the default `"linear"` interpolation, the average of the two middle
#' values is returned, matching base R's `median()`.
#'
#' You can also use `median()` directly on an [`AnvlArray`] or [`AnvlBox`];
#' extra arguments (e.g. `interpolation`) are forwarded via `...`.
#' @template param_x
#' @param axis (`integer(1)` | `NULL`)\cr
#'   Axis along which to compute the median. Negative values count from
#'   the end, i.e. `-1` refers to the last axis. If `NULL` (default),
#'   uses the last axis.
#' @param interpolation (`character(1)`)\cr
#'   Forwarded to [nv_quantile()]. One of `"linear"` (default), `"lower"`,
#'   `"higher"`, `"nearest"`, `"midpoint"`.
#' @param nan_rm (`logical(1)`)\cr
#'   Forwarded to [nv_quantile()]. See its documentation for details.
#' @return [`arrayish`]\cr
#'   Same shape as `x` with `axis` removed.
#' @seealso [nv_quantile()], [nv_sort()], [prim_sort()].
#' @examplesIf pjrt::plugins_downloaded()
#' nv_median(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#' median(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#' nv_median(nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE),
#'   axis = 2L
#' )
#' # forwards through the S3 generic via `...`
#' median(nv_array(c(1, 2, 3, 4)), interpolation = "lower")
#' nv_median(nv_array(c(1, NaN, 3, 5)))
#' nv_median(nv_array(c(1, NaN, 3, 5)), nan_rm = TRUE)
#' @export
#' @jit static 2:4
nv_median <- function(x, axis = NULL, interpolation = "linear", nan_rm = FALSE) {
  nv_quantile(x, probs = 0.5, axis = axis, interpolation = interpolation, nan_rm = nan_rm)
}

#' @title Index of the Maximum
#' @description
#' Returns the index of the maximum value along an axis. Ties are broken
#' by returning the smallest index.
#' @template param_x
#' @param axis (`integer(1)` | `NULL`)\cr
#'   Axis along which to find the index. Negative values count from the
#'   end, i.e. `-1` refers to the last axis. If `NULL` (default), uses
#'   the last axis.
#' @param drop (`logical(1)`)\cr
#'   If `TRUE` (default) the reduced axis is removed; if `FALSE` it
#'   is kept with size 1.
#' @template param_nan_rm
#' @return [`arrayish`] of dtype `i32`\cr
#'   Same shape as `x` with `axis` removed (or set to 1 if `drop = FALSE`).
#' @section NaN handling:
#' With `nan_rm = FALSE` (default), if any entry along the reduced axis is
#' `NaN`, the returned index points at the first such `NaN`. With
#' `nan_rm = TRUE`, `NaN` entries are skipped.
#' @seealso [nv_argmin()], [nv_reduce_max()].
#' @examplesIf pjrt::plugins_downloaded()
#' nv_argmax(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#' nv_argmax(nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE),
#'   axis = 2L
#' )
#' nv_argmax(nv_array(c(1, NaN, 3)))
#' nv_argmax(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#' @export
nv_argmax <- function(x, axis = NULL, drop = TRUE, nan_rm = FALSE) {
  x <- as_anvl_array(x)
  if (naxes(x) == 0L) {
    cli_abort("Cannot compute the arg-extremum of a 0-dimensional array")
  }
  axis <- axis %||% naxes(x)
  .nv_arg_extreme(x, axis, drop, nan_rm, prim_argmax)
}

#' @title Index of the Minimum
#' @description
#' Returns the index of the minimum value along an axis. Ties are broken
#' by returning the smallest index.
#' @template param_x
#' @param axis (`integer(1)` | `NULL`)\cr
#'   Axis along which to find the index. Negative values count from the
#'   end, i.e. `-1` refers to the last axis. If `NULL` (default), uses
#'   the last axis.
#' @param drop (`logical(1)`)\cr
#'   If `TRUE` (default) the reduced axis is removed; if `FALSE` it
#'   is kept with size 1.
#' @template param_nan_rm
#' @return [`arrayish`] of dtype `i32`\cr
#'   Same shape as `x` with `axis` removed (or set to 1 if `drop = FALSE`).
#' @inheritSection nv_argmax NaN handling
#' @seealso [nv_argmax()], [nv_reduce_min()].
#' @examplesIf pjrt::plugins_downloaded()
#' nv_argmin(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#' nv_argmin(nv_array(c(2, NaN, 1, 3)))
#' nv_argmin(nv_array(c(2, NaN, 1, 3)), nan_rm = TRUE)
#' @export
nv_argmin <- function(x, axis = NULL, drop = TRUE, nan_rm = FALSE) {
  x <- as_anvl_array(x)
  if (naxes(x) == 0L) {
    cli_abort("Cannot compute the arg-extremum of a 0-dimensional array")
  }
  axis <- axis %||% naxes(x)
  .nv_arg_extreme(x, axis, drop, nan_rm, prim_argmin)
}

# Shared NaN-aware argmax/argmin. The XLA arg-reduction kernels are
# comparison-based and silently skip NaN, so `nan_rm = TRUE` is free — we
# just call the primitive. For `nan_rm = FALSE` we want NaN to propagate,
# mirroring `.nv_reduce_extreme`'s contract; there's no NaN in i32, so we
# surface "a NaN was here" by returning the first NaN's index instead.
#
.nv_arg_extreme <- function(x, axis, drop, nan_rm, prim_arg) {
  result <- prim_arg(x, axis = axis, drop = drop)
  if (nan_rm || !is_dtype_float(dtype(x))) {
    return(result)
  }
  # argmax on the bool mask returns the index of the first TRUE (tie-break:
  # smallest index) — exactly the first NaN's position — or 1 if no NaN
  # exists. `any_nan` disambiguates those two cases.
  nan_mask <- nv_is_nan(x)
  any_nan <- prim_reduce_any(nan_mask, axes = axis, drop = drop)
  first_nan_idx <- prim_argmax(nan_mask, axis = axis, drop = drop)
  nv_ifelse(any_nan, first_nan_idx, result)
}

# Build the NCHW/NC(D)HW axis numbers (1-based) for nv_conv*.
.nv_conv_axis_numbers <- function(n_spatial) {
  spatial <- 3:(2 + n_spatial)
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
#' @param x ([`arrayish`])\cr `[N, C_in, W]`.
#' @param weight ([`arrayish`])\cr `[C_out, C_in / groups, kW]`.
#' @param stride,padding,dilation (`integer()`)\cr Length 1.
#' @param groups (`integer(1)`)\cr Grouped/depthwise convolution.
#' @param precision (`character(1)`)\cr `"highest"`, `"high"` or `"default"`.
#' @return [`arrayish`] `[N, C_out, out_W]`.
#' @seealso [nv_conv2d()], [nv_conv3d()], [prim_convolution()].
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
#' @param x ([`arrayish`])\cr `[N, C_in, H, W]`.
#' @param weight ([`arrayish`])\cr `[C_out, C_in / groups, kH, kW]`.
#' @param stride (`integer()`)\cr Length 1 or 2.
#' @param padding (`integer()`)\cr Symmetric padding, length 1 or 2.
#' @param dilation (`integer()`)\cr Kernel dilation, length 1 or 2.
#' @param groups (`integer(1)`)\cr Grouped/depthwise convolution.
#' @param precision (`character(1)`)\cr `"highest"`, `"high"` or `"default"`.
#' @return [`arrayish`] `[N, C_out, out_H, out_W]`.
#' @seealso [nv_conv1d()], [nv_conv3d()], [prim_convolution()].
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
#' @inheritParams nv_conv2d
#' @param stride,padding,dilation (`integer()`)\cr Length 1 or 3.
#' @return [`arrayish`] `[N, C_out, out_D, out_H, out_W]`.
#' @seealso [nv_conv1d()], [nv_conv2d()], [prim_convolution()].
#' @export
nv_conv3d <- function(x, weight, stride = 1L, padding = 0L, dilation = 1L, groups = 1L, precision = "highest") {
  .nv_convnd(x, weight, 3L, stride, padding, dilation, groups, precision)
}

.nv_convnd <- function(x, weight, n, stride, padding, dilation, groups, precision) {
  # `x`/`weight` are left as raw arrayish; prim_convolution's machinery
  # (graph_desc_add -> maybe_box_arrayish) coerces them.
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
