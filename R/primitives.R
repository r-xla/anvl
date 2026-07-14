#' @include utils.R
#' @include type-converters.R
#' @include primitive.R
#' @include jit.R

make_binary_op <- function(stablehlo_infer) {
  force(stablehlo_infer)
  infer_fn <- function(lhs, rhs) {
    both_ambiguous <- lhs$ambiguous && rhs$ambiguous
    out <- stablehlo_infer(at2vt(lhs), at2vt(rhs))[[1L]]
    out <- vt2at(out)
    out$ambiguous <- both_ambiguous
    list(out)
  }
  function(lhs, rhs) {
    graph_desc_add(self, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
  }
}

make_unary_op <- function(stablehlo_infer) {
  force(stablehlo_infer)
  infer_fn <- function(operand) {
    out <- stablehlo_infer(at2vt(operand))[[1L]]
    out <- vt2at(out)
    out$ambiguous <- operand$ambiguous
    list(out)
  }
  function(operand) {
    graph_desc_add(self, list(operand = operand), infer_fn = infer_fn)[[1L]]
  }
}


infer_reduce <- function(operand, dims, drop) {
  old_shape <- shape(operand)
  if (drop) {
    new_shape <- old_shape[-dims]
  } else {
    new_shape <- old_shape
    new_shape[dims] <- 1L
  }
  list(AbstractArray(
    dtype = dtype(operand),
    shape = Shape(new_shape),
    ambiguous = operand$ambiguous
  ))
}

infer_reduce_boolean <- function(operand, dims, drop) {
  old_shape <- shape(operand)
  if (drop) {
    new_shape <- old_shape[-dims]
  } else {
    new_shape <- old_shape
    new_shape[dims] <- 1L
  }
  list(AbstractArray(
    dtype = "bool",
    shape = Shape(new_shape),
    ambiguous = FALSE
  ))
}

#' @title Primitive Fill
#' @description
#' Creates an array of a given shape and data type, filled with a scalar value.
#' The advantage of using this function instead of e.g. doing
#' `nv_array(1, shape = c(100, 100))` is that lowering of [prim_fill()] is
#' efficiently represented in the compiled program, while the latter uses
#' 100 * 100 * 4 bytes of memory.
#' @param value (`numeric(1)`)\cr
#'   Scalar value to fill the array with.
#' @param shape (`integer()`)\cr
#'   Shape of the output array.
#' @template param_dtype
#' @template param_ambiguous
#' @template param_device
#' @return [`arrayish`]\cr
#'   Has the given `shape` and `dtype`.
#' @templateVar primitive_id fill
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_tensor()].
#' @seealso [nv_fill()]
#' @examplesIf pjrt::plugins_downloaded()
#' prim_fill(3.14, shape = c(2, 3), dtype = "f32")
#' @export
prim_fill <- new_primitive(
  "fill",
  function(value, shape, dtype, ambiguous = FALSE, device = NULL) {
    infer_fill <- function(value, shape, dtype, ambiguous) {
      list(AbstractArray(dtype = as_dtype(dtype), shape = shape, ambiguous = ambiguous))
    }
    graph_desc_add(
      self,
      list(),
      params = list(value = value, dtype = dtype, shape = shape, ambiguous = ambiguous),
      infer_fn = infer_fill
    )[[1L]]
  },
  static = 1:5,
  device = device_arg("device")
)

#' @title Primitive Addition
#' @description
#' Adds two arrays element-wise.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_binary
#' @templateVar primitive_id add
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_add()].
#' @seealso [nv_add()], `+`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5, 6))
#' prim_add(x, y)
#' @export
prim_add <- new_primitive("add", make_binary_op(stablehlo::infer_types_add))

#' @title Primitive Multiplication
#' @description
#' Multiplies two arrays element-wise.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_binary
#' @templateVar primitive_id mul
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_multiply()].
#' @seealso [nv_mul()], `*`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5, 6))
#' prim_mul(x, y)
#' @export
prim_mul <- new_primitive("mul", make_binary_op(stablehlo::infer_types_multiply))

#' @title Primitive Subtraction
#' @description
#' Subtracts two arrays element-wise.
#' @template params_prim_lhs_rhs_numeric
#' @template return_prim_binary
#' @templateVar primitive_id sub
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_subtract()].
#' @seealso [nv_sub()], `-`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5, 6))
#' prim_sub(x, y)
#' @export
prim_sub <- new_primitive("sub", make_binary_op(stablehlo::infer_types_subtract))

#' @title Primitive Negation
#' @description
#' Negates an array element-wise.
#' @param operand ([`arrayish`])\cr
#'   Arrayish value of data type integer or floating-point.
#' @template return_prim_unary
#' @templateVar primitive_id negate
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_negate()].
#' @seealso [nv_negate()], unary `-`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, -2, 3))
#' prim_negate(x)
#' @export
prim_negate <- new_primitive("negate", make_unary_op(stablehlo::infer_types_negate))

#' @title Primitive Division
#' @description
#' Divides two arrays element-wise.
#' @template params_prim_lhs_rhs_numeric
#' @template return_prim_binary
#' @templateVar primitive_id divide
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_divide()].
#' @seealso [nv_div()], `/`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(10, 20, 30))
#' y <- nv_array(c(2, 5, 10))
#' prim_div(x, y)
#' @export
prim_div <- new_primitive("divide", make_binary_op(stablehlo::infer_types_divide))

#' @title Primitive Power
#' @description
#' Raises lhs to the power of rhs element-wise.
#' @template params_prim_lhs_rhs_numeric
#' @template return_prim_binary
#' @templateVar primitive_id power
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_power()].
#' @seealso [nv_pow()], `^`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(2, 3, 4))
#' y <- nv_array(c(3, 2, 1))
#' prim_pow(x, y)
#' @export
prim_pow <- new_primitive("power", make_binary_op(stablehlo::infer_types_power))

#' @title Primitive Broadcast
#' @description
#' Broadcasts an array to a new shape by replicating the data along new or size-1 dimensions.
#' @template param_prim_operand_any
#' @param shape (`integer()`)\cr
#'   Target shape. Each mapped dimension must either match the corresponding
#'   operand dimension or the operand dimension must be 1.
#' @param broadcast_dimensions (`integer()`)\cr
#'   Maps each dimension of `operand` to a dimension of the output.
#'   Must have length equal to the number of dimensions of `operand`.
#' @return [`arrayish`]\cr
#'   Has the same data type as the input and the given `shape`.
#'   It is ambiguous if the input is ambiguous.
#' @importFrom stablehlo r_to_constant
#' @templateVar primitive_id broadcast_in_dim
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_broadcast_in_dim()].
#' @seealso [nv_broadcast_to()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' prim_broadcast_in_dim(x, shape = c(2, 3), broadcast_dimensions = 2L)
#' @export
prim_broadcast_in_dim <- new_primitive(
  "broadcast_in_dim",
  function(operand, shape, broadcast_dimensions) {
    infer_fn <- function(operand, shape, broadcast_dimensions) {
      bd_attr <- r_to_constant(
        as.integer(broadcast_dimensions - 1L),
        dtype = "i64",
        shape = length(broadcast_dimensions)
      )
      out <- stablehlo::infer_types_broadcast_in_dim(
        at2vt(operand),
        broadcast_dimensions = bd_attr,
        shape = shape
      )[[1L]]
      out <- vt2at(out)
      out$ambiguous <- operand$ambiguous
      list(out)
    }
    graph_desc_add(
      self,
      list(operand = operand),
      params = list(
        shape = shape,
        broadcast_dimensions = broadcast_dimensions
      ),
      infer_fn = infer_fn
    )[[1L]]
  },
  static = 2:3
)

#' @title Primitive Dot General
#' @description
#' General dot product of two arrays, supporting contraction over arbitrary
#' dimensions and batching.
#' @template params_lhs_rhs
#' @param contracting_dims (`list(integer(), integer())`)\cr
#'   A list of two integer vectors specifying which dimensions of `lhs` and
#'   `rhs` to contract over. The contracted dimensions must have matching sizes.
#' @param batching_dims (`list(integer(), integer())`)\cr
#'   A list of two integer vectors specifying which dimensions of `lhs` and
#'   `rhs` are batch dimensions. These must have matching sizes.
#' @param precision (`character(1)`)\cr
#'   Controls the trade-off between speed and numerical accuracy of the
#'   operation. One of `"highest"` (default), `"high"` or `"default"`.
#'   Only the StableHLO backend honors this; it is ignored by the quickr backend.
#' @return [`arrayish`]\cr
#'   The output shape is the batch dimensions followed by the remaining
#'   (non-contracted, non-batched) dimensions of `lhs`, then `rhs`.
#' @templateVar primitive_id dot_general
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_dot_general()].
#' @seealso [nv_matmul()], `%*%`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' y <- nv_matrix(1:6, nrow = 3)
#' prim_dot_general(x, y,
#'   contracting_dims = list(2L, 1L),
#'   batching_dims = list(integer(0), integer(0))
#' )
#' @export
prim_dot_general <- new_primitive(
  "dot_general",
  function(lhs, rhs, contracting_dims, batching_dims, precision = "highest") {
    precision <- match.arg(precision, c("default", "high", "highest"))
    infer_fn <- function(lhs, rhs, contracting_dims, batching_dims, precision) {
      ddn <- stablehlo::DotDimensionNumbers(
        contracting_dims = lapply(contracting_dims, \(x) x - 1L),
        batching_dims = lapply(batching_dims, \(x) x - 1L)
      )
      out <- stablehlo::infer_types_dot_general(at2vt(lhs), at2vt(rhs), dot_dimension_numbers = ddn)[[1L]]
      list(vt2at(out))
    }
    graph_desc_add(
      self,
      list(lhs = lhs, rhs = rhs),
      list(
        contracting_dims = contracting_dims,
        batching_dims = batching_dims,
        precision = precision
      ),
      infer_fn = infer_fn
    )[[1L]]
  },
  static = 3:5
)

#' @title Primitive Transpose
#' @description
#' Permutes the dimensions of an array.
#' @template param_prim_operand_any
#' @param permutation (`integer()`)\cr
#'   Specifies the new ordering of dimensions. Must be a permutation of
#'   `seq_len(ndims)` where `ndims` is the number of dimensions of `operand`.
#' @return [`arrayish`]\cr
#'   Has the same data type as the input and shape `nv_shape(operand)[permutation]`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id transpose
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_transpose()].
#' @seealso [nv_transpose()], [t()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' prim_transpose(x, permutation = c(2L, 1L))
#' @export
prim_transpose <- new_primitive(
  "transpose",
  function(operand, permutation) {
    infer_fn <- function(operand, permutation) {
      perm_attr <- r_to_constant(
        as.integer(permutation - 1L),
        dtype = "i64",
        shape = length(permutation)
      )
      out <- stablehlo::infer_types_transpose(at2vt(operand), permutation = perm_attr)[[1L]]
      out <- vt2at(out)
      out$ambiguous <- operand$ambiguous
      list(out)
    }
    graph_desc_add(
      self,
      list(operand = operand),
      list(permutation = permutation),
      infer_fn = infer_fn
    )[[1L]]
  },
  static = 2L
)

#' @title Primitive Reshape
#' @description
#' Reshapes an array to a new shape without changing the underlying data.
#' Note that row-major order is used, which differs from R's column-major order.
#' @template param_prim_operand_any
#' @param shape (`integer()`)\cr
#'   Target shape. Must have the same number of elements as `operand`.
#' @return [`arrayish`]\cr
#'   Has the same data type as the input and the given `shape`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id reshape
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_reshape()].
#' @seealso [nv_reshape()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:6)
#' prim_reshape(x, shape = c(2, 3))
#' @export
prim_reshape <- new_primitive(
  "reshape",
  function(operand, shape) {
    infer_fn <- function(operand, shape) {
      out <- stablehlo::infer_types_reshape(at2vt(operand), shape = shape)[[1L]]
      out <- vt2at(out)
      out$ambiguous <- operand$ambiguous
      list(out)
    }
    graph_desc_add(
      self,
      list(operand = operand),
      params = list(shape = shape),
      infer_fn = infer_fn
    )[[1L]]
  },
  static = 2L
)

#' @title Primitive Concatenate
#' @description
#' Concatenates arrays along a dimension.
#' @param ... ([`arrayish`])\cr
#'   Arrays to concatenate. Must all have the same data type, ndims,
#'   and shape except along `dimension`.
#' @param dimension (`integer(1)`)\cr
#'   Dimension along which to concatenate (1-indexed).
#' @return [`arrayish`]\cr
#'   Has the same data type as the inputs.
#'   The output shape matches the inputs in all dimensions except `dimension`,
#'   which is the sum of the input sizes along that dimension.
#'   It is ambiguous if all inputs are ambiguous.
#' @templateVar primitive_id concatenate
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_concatenate()].
#' @seealso [nv_concatenate()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5, 6))
#' prim_concatenate(x, y, dimension = 1L)
#' @export
prim_concatenate <- new_primitive(
  "concatenate",
  function(..., dimension) {
    dots <- list(...)
    infer_fn <- function(..., dimension) {
      operands <- list(...)
      all_ambiguous <- all(vapply(operands, \(x) x$ambiguous, logical(1L)))
      vts <- lapply(operands, at2vt)
      # Convert dimension to Constant as required by stablehlo
      dim_const <- stablehlo::r_to_constant(
        as.integer(dimension - 1L),
        dtype = "i64",
        shape = integer(0)
      )
      out <- rlang::exec(stablehlo::infer_types_concatenate, !!!vts, dimension = dim_const)[[1L]]
      out <- vt2at(out)
      out$ambiguous <- all_ambiguous
      list(out)
    }
    graph_desc_add(
      self,
      args = dots,
      params = list(dimension = dimension),
      infer_fn = infer_fn
    )[[1L]]
  },
  static = "dimension"
)

#' @title Primitive Static Slice
#' @description
#' Extracts a slice from an array using static (compile-time) indices.
#' All indices, limits, and strides are fixed R integers.
#'
#' Use [prim_dynamic_slice()] instead when the start position must be
#' computed at runtime (e.g. depends on array values).
#' @template param_prim_operand_any
#' @param start_indices (`integer()`)\cr
#'   Start indices (inclusive), one per dimension. Must satisfy
#'   `1 <= start_indices <= limit_indices` per dimension.
#' @param limit_indices (`integer()`)\cr
#'   End indices (inclusive), one per dimension. Must satisfy
#'   `limit_indices <= nv_shape(operand)` per dimension.
#' @param strides (`integer()`)\cr
#'   Step sizes, one per dimension. Must be `>= 1`. A stride of `1`
#'   selects every element; a stride of `2` selects every other element, etc.
#' @return [`arrayish`]\cr
#'   Has the same data type as the input and shape
#'   `ceiling((limit_indices - start_indices + 1) / strides)`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id static_slice
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_slice()].
#' @seealso [prim_dynamic_slice()], [prim_scatter()], [prim_gather()], [nv_subset()], `[`
#' @examplesIf pjrt::plugins_downloaded()
#' # 1-D: extract elements 2 through 4 (limit is exclusive)
#' x <- nv_array(1:10)
#' prim_static_slice(x, start_indices = 2L, limit_indices = 5L, strides = 1L)
#'
#' # 1-D: every other element using strides
#' x <- nv_array(1:10)
#' prim_static_slice(x, start_indices = 1L, limit_indices = 10L, strides = 2L)
#'
#' # 2-D: extract a submatrix (rows 1-2, columns 2-3)
#' x <- nv_matrix(1:12, nrow = 3, ncol = 4)
#' prim_static_slice(x,
#'   start_indices = c(1L, 2L),
#'   limit_indices = c(3L, 4L),
#'   strides       = c(1L, 1L)
#' )
#' @export
prim_static_slice <- new_primitive(
  "static_slice",
  function(operand, start_indices, limit_indices, strides) {
    infer_fn <- function(operand, start_indices, limit_indices, strides) {
      start_attr <- r_to_constant(start_indices - 1L, dtype = "i64", shape = length(start_indices))
      limit_attr <- r_to_constant(limit_indices, dtype = "i64", shape = length(limit_indices))
      strides_attr <- r_to_constant(strides, dtype = "i64", shape = length(strides))
      out <- stablehlo::infer_types_slice(at2vt(operand), start_attr, limit_attr, strides_attr)[[1L]]
      out <- vt2at(out)
      out$ambiguous <- operand$ambiguous
      list(out)
    }
    graph_desc_add(
      self,
      args = list(
        operand = operand
      ),
      params = list(
        start_indices = start_indices,
        limit_indices = limit_indices,
        strides = strides
      ),
      infer_fn = infer_fn
    )[[1L]]
  },
  static = 2:4
)

#' @title Primitive Dynamic Slice
#' @description
#' Extracts a slice from an array whose start position is determined at
#' runtime via array-valued indices. The slice shape (`slice_sizes`) is
#' a fixed R integer vector.
#'
#' Use [prim_static_slice()] instead when all indices are known at compile
#' time and you need stride support.
#' @template param_prim_operand_any
#' @param ... ([`arrayish`] of integer type)\cr
#'   Scalar start indices, one per dimension. Each must be a
#'   scalar array. Pass one scalar per dimension of `operand`.
#' @param slice_sizes (`integer()`)\cr
#'   Size of the slice in each dimension. Must have length equal to
#'   `ndims(operand)` and satisfy `1 <= slice_sizes <= nv_shape(operand)`
#'   per dimension.
#' @section Out Of Bounds Behavior:
#' Start indices are clamped before the slice is extracted:
#' `adjusted_start_indices = clamp(1, start_indices, nv_shape(operand) - slice_sizes + 1)`.
#' This means that out-of-bounds indices will not cause an error, but
#' the effective start position may differ from the requested one.
#' @return [`arrayish`]\cr
#'   Has the same data type as the input and shape `slice_sizes`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id dynamic_slice
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_dynamic_slice()].
#' @seealso [prim_static_slice()], [prim_dynamic_update_slice()], [prim_scatter()], [prim_gather()], [nv_subset()], `[`
#' @examplesIf pjrt::plugins_downloaded()
#' # 1-D: extract 3 elements starting at position 3
#' x <- nv_array(1:10)
#' start <- nv_scalar(3L)
#' prim_dynamic_slice(x, start, slice_sizes = 3L)
#'
#' # 2-D: extract a 2x2 block from a matrix
#' x <- nv_matrix(1:12, nrow = 3, ncol = 4)
#' row_start <- nv_scalar(2L)
#' col_start <- nv_scalar(1L)
#' prim_dynamic_slice(x, row_start, col_start, slice_sizes = c(2L, 2L))
#' @export
prim_dynamic_slice <- new_primitive(
  "dynamic_slice",
  function(operand, ..., slice_sizes) {
    start_indices <- list(...)
    infer_fn <- function(operand, ..., slice_sizes) {
      start_indices_avals <- list(...)
      for (i in seq_along(start_indices_avals)) {
        aval <- start_indices_avals[[i]]
        if (length(shape(aval)) != 0L) {
          cli_abort("Start index {i} must be a scalar, but has shape {shape(aval)}")
        }
      }
      out <- AbstractArray(dtype = operand$dtype, shape = slice_sizes, ambiguous = operand$ambiguous)
      list(out)
    }
    graph_desc_add(
      self,
      args = c(list(operand = operand), start_indices),
      params = list(slice_sizes = slice_sizes),
      infer_fn = infer_fn
    )[[1L]]
  },
  static = "slice_sizes"
)

#' @title Primitive Dynamic Update Slice
#' @description
#' Returns a copy of `operand` with a slice replaced by `update` at a
#' runtime-determined position. This is the write counterpart of
#' [prim_dynamic_slice()]: dynamic slice reads a block from an array,
#' while dynamic update slice writes a block into an array.
#' @template param_prim_operand_any
#' @param update ([`arrayish`])\cr
#'   The values to write at the specified position. Must have the same
#'   data type and number of dimensions as `operand`, with
#'   `nv_shape(update) <= nv_shape(operand)` per dimension.
#' @param ... ([`arrayish`] of integer type)\cr
#'   Scalar start indices, one per dimension of `operand`.
#'   Each must be a scalar array.
#' @inheritSection prim_dynamic_slice Out Of Bounds Behavior
#' @return [`arrayish`]\cr
#'   Has the same data type and shape as `operand`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id dynamic_update_slice
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_dynamic_update_slice()].
#' @seealso [prim_dynamic_slice()], [prim_scatter()], [prim_gather()], [nv_subset_assign()], `[<-`
#' @examplesIf pjrt::plugins_downloaded()
#' # 1-D: overwrite two elements starting at position 2
#' x <- nv_array(1:5)
#' update <- nv_array(c(10L, 20L))
#' start <- nv_scalar(2L)
#' prim_dynamic_update_slice(x, update, start)
#'
#' # 2-D: write a 2x2 block into a 3x4 matrix
#' x <- nv_fill(0L, shape = c(3, 4))
#' update <- nv_matrix(c(1L, 2L, 3L, 4L), nrow = 2, ncol = 2)
#' row_start <- nv_scalar(2L)
#' col_start <- nv_scalar(3L)
#' prim_dynamic_update_slice(x, update, row_start, col_start)
#' @export
prim_dynamic_update_slice <- new_primitive(
  "dynamic_update_slice",
  function(operand, update, ...) {
    start_indices <- list(...)
    infer_fn <- function(operand, update, ...) {
      start_indices_avals <- list(...)
      for (i in seq_along(start_indices_avals)) {
        aval <- start_indices_avals[[i]]
        if (length(shape(aval)) != 0L) {
          cli_abort("Start index {i} must be a scalar, but has shape {shape(aval)}")
        }
      }
      out <- AbstractArray(dtype = operand$dtype, shape = shape(operand), ambiguous = operand$ambiguous)
      list(out)
    }
    graph_desc_add(
      self,
      args = c(list(operand = operand, update = update), start_indices),
      params = list(),
      infer_fn = infer_fn
    )[[1L]]
  }
)


# reduction operators

make_reduce_op <- function(infer_fn = infer_reduce) {
  force(infer_fn)
  function(operand, dims, drop = TRUE) {
    graph_desc_add(
      self,
      list(operand = operand),
      params = list(dims = dims, drop = drop),
      infer_fn = infer_fn
    )[[1L]]
  }
}

#' @title Primitive Sum Reduction
#' @description
#' Sums array elements along the specified dimensions.
#' @template param_prim_operand_any
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce over.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced dimensions from the output shape.
#'   If `TRUE`, the reduced dimensions are removed.
#'   If `FALSE`, the reduced dimensions are set to 1.
#' @template return_prim_reduce
#' @templateVar primitive_id reduce_sum
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_reduce()] with [hlo_add()] as the reducer.
#' @seealso [nv_reduce_sum()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' prim_reduce_sum(x, dims = 1L)
#' @export
prim_reduce_sum <- new_primitive("reduce_sum", make_reduce_op(), static = 2:3)

#' @title Primitive Product Reduction
#' @description
#' Multiplies array elements along the specified dimensions.
#' @template param_prim_operand_any
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce over.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced dimensions from the output shape.
#'   If `TRUE`, the reduced dimensions are removed.
#'   If `FALSE`, the reduced dimensions are set to 1.
#' @template return_prim_reduce
#' @templateVar primitive_id reduce_prod
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_reduce()] with [hlo_multiply()] as the reducer.
#' @seealso [nv_reduce_prod()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' prim_reduce_prod(x, dims = 1L)
#' @export
prim_reduce_prod <- new_primitive("reduce_prod", make_reduce_op(), static = 2:3)

#' @title Primitive Max Reduction
#' @description
#' Finds the maximum of array elements along the specified dimensions.
#' @template param_prim_operand_any
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce over.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced dimensions from the output shape.
#'   If `TRUE`, the reduced dimensions are removed.
#'   If `FALSE`, the reduced dimensions are set to 1.
#' @template return_prim_reduce
#' @templateVar primitive_id reduce_max
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_reduce()] with [hlo_maximum()] as the reducer.
#' @seealso [nv_reduce_max()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' prim_reduce_max(x, dims = 1L)
#' @export
prim_reduce_max <- new_primitive("reduce_max", make_reduce_op(), static = 2:3)

#' @title Primitive Min Reduction
#' @description
#' Finds the minimum of array elements along the specified dimensions.
#' @template param_prim_operand_any
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce over.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced dimensions from the output shape.
#'   If `TRUE`, the reduced dimensions are removed.
#'   If `FALSE`, the reduced dimensions are set to 1.
#' @template return_prim_reduce
#' @templateVar primitive_id reduce_min
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_reduce()] with [hlo_minimum()] as the reducer.
#' @seealso [nv_reduce_min()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' prim_reduce_min(x, dims = 1L)
#' @export
prim_reduce_min <- new_primitive("reduce_min", make_reduce_op(), static = 2:3)

#' @title Primitive Any Reduction
#' @description
#' Performs logical OR along the specified dimensions.
#' @template param_prim_operand_boolean
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce over.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced dimensions from the output shape.
#'   If `TRUE`, the reduced dimensions are removed.
#'   If `FALSE`, the reduced dimensions are set to 1.
#' @template return_prim_reduce_boolean
#' @templateVar primitive_id reduce_any
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_reduce()] with [hlo_or()] as the reducer.
#' @seealso [nv_reduce_any()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
#' prim_reduce_any(x, dims = 1L)
#' @export
prim_reduce_any <- new_primitive("reduce_any", make_reduce_op(infer_reduce_boolean), static = 2:3)

#' @title Primitive All Reduction
#' @description
#' Performs logical AND along the specified dimensions.
#' @template param_prim_operand_boolean
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce over.
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced dimensions from the output shape.
#'   If `TRUE`, the reduced dimensions are removed.
#'   If `FALSE`, the reduced dimensions are set to 1.
#' @template return_prim_reduce_boolean
#' @templateVar primitive_id reduce_all
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_reduce()] with [hlo_and()] as the reducer.
#' @seealso [nv_reduce_all()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
#' prim_reduce_all(x, dims = 1L)
#' @export
prim_reduce_all <- new_primitive("reduce_all", make_reduce_op(infer_reduce_boolean), static = 2:3)

# cumulative (scan) primitives -------------------------------------------------

infer_cum <- function(operand, dim) {
  rank <- length(shape(operand))
  if (rank == 0L) {
    cli_abort("cumulative ops require at least a 1-dimensional operand, but operand is a scalar")
  }
  if (!checkmate::test_integerish(dim, lower = 1, upper = rank, len = 1L)) {
    cli_abort("{.arg dim} must be a single integer in 1:{rank}, but is {.val {dim}}")
  }
  list(AbstractArray(
    dtype = dtype(operand),
    shape = Shape(shape(operand)),
    ambiguous = operand$ambiguous
  ))
}

cum_op <- function(operand, dim) {
  graph_desc_add(self, list(operand = operand), params = list(dim = dim), infer_fn = infer_cum)[[1L]]
}

infer_cum_extreme <- function(operand, dim) {
  rank <- length(shape(operand))
  if (rank == 0L) {
    cli_abort("cumulative ops require at least a 1-dimensional operand, but operand is a scalar")
  }
  if (!checkmate::test_integerish(dim, lower = 1, upper = rank, len = 1L)) {
    cli_abort("{.arg dim} must be a single integer in 1:{rank}, but is {.val {dim}}")
  }
  list(
    AbstractArray(
      dtype = dtype(operand),
      shape = Shape(shape(operand)),
      ambiguous = operand$ambiguous
    ),
    AbstractArray(
      dtype = "i32",
      shape = Shape(shape(operand)),
      ambiguous = FALSE
    )
  )
}

cum_extreme_op <- function(operand, dim) {
  graph_desc_add(self, list(operand = operand), params = list(dim = dim), infer_fn = infer_cum_extreme)
}

#' @title Primitive Cumulative Sum
#' @description
#' Cumulative sum of array elements along a single dimension.
#' Output position `j` along `dim` equals the sum of input positions `1:j`.
#' @template param_prim_operand_any
#' @template param_prim_cum_dim
#' @template return_prim_unary
#' @templateVar primitive_id cumsum
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_reduce_window()] with [hlo_add()] as the reducer.
#' @seealso [nv_cumsum()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' prim_cumsum(x, dim = 1L)
#' @export
prim_cumsum <- new_primitive("cumsum", cum_op, static = 2L)

#' @title Primitive Cumulative Product
#' @description
#' Cumulative product of array elements along a single dimension.
#' Output position `j` along `dim` equals the product of input positions `1:j`.
#' @template param_prim_operand_any
#' @template param_prim_cum_dim
#' @template return_prim_unary
#' @templateVar primitive_id cumprod
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_reduce_window()] with [hlo_multiply()] as the reducer.
#' @seealso [nv_cumprod()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' prim_cumprod(x, dim = 1L)
#' @export
prim_cumprod <- new_primitive("cumprod", cum_op, static = 2L)

#' @title Primitive Cumulative Maximum
#' @description
#' Running maximum of array elements along a single dimension along with
#' the index of the last occurrence of the running maximum.
#' At output position `j`, the values output is `max(input[1:j])` and the
#' indices output is the largest `i` in `1:j` with
#' `input[i] == values[j]` (last-occurrence tiebreak).
#' @template param_prim_operand_any
#' @template param_prim_cum_dim
#' @templateVar cum_extreme_name maximum
#' @templateVar cum_extreme_arg argmax
#' @template return_prim_cum_extreme
#' @templateVar primitive_id cummax
#' @template section_rules
#' @section StableHLO:
#' Lowers to a variadic [hlo_reduce_window()] over `(values, iota)`.
#' @seealso [nv_cummax()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(3, 1, 4, 1, 5, 9), nrow = 2)
#' prim_cummax(x, dim = 1L)
#' @export
prim_cummax <- new_primitive("cummax", cum_extreme_op, static = 2L)

#' @title Primitive Cumulative Minimum
#' @description
#' Running minimum of array elements along a single dimension along with
#' the index of the last occurrence of the running minimum.
#' At output position `j`, the values output is `min(input[1:j])` and the
#' indices output is the largest `i` in `1:j` with
#' `input[i] == values[j]` (last-occurrence tiebreak).
#' @template param_prim_operand_any
#' @template param_prim_cum_dim
#' @templateVar cum_extreme_name minimum
#' @templateVar cum_extreme_arg argmin
#' @template return_prim_cum_extreme
#' @templateVar primitive_id cummin
#' @template section_rules
#' @section StableHLO:
#' Lowers to a variadic [hlo_reduce_window()] over `(values, iota)`.
#' @seealso [nv_cummin()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(3, 1, 4, 1, 5, 9), nrow = 2)
#' prim_cummin(x, dim = 1L)
#' @export
prim_cummin <- new_primitive("cummin", cum_extreme_op, static = 2L)

#' @title Primitive Generic Reduce
#' @description
#' Reduces an array along the specified dimensions using a user-supplied
#' associative reducer.
#' @section Associativity Requirement:
#' The order in which `reductor` is applied across the reduction window is
#' implementation-defined. If the reductor is not associative, the result
#' is ill-defined.
#' Furthermore, `init` must be the neutral element for this reductor.
#' Because floating point math is non-associative, the output of
#' the reduction can differ between backends (GPU, CPU), even if the underlying mathematical
#' function (like `+`) is associative.
#'
#' @template param_prim_operand_any
#' @param init ([`arrayish`])\cr
#'   Scalar (0-dimensional) initial value. Must have the same data type as
#'   `operand` and be the neutral element w.r.t. `reductor`.
#' @param dims (`integer()`)\cr
#'   Dimensions to reduce over.
#' @param drop (`logical(1)`)\cr
#'   If `TRUE` (default) the reduced dimensions are removed; if `FALSE`
#'   they are kept with size 1.
#' @param reductor (`function(lhs, rhs)`)\cr
#'   Binary reducer producing a scalar of the same dtype as `operand`.
#'   Must be associative (see "Associativity Requirement").
#' @return [`arrayish`]\cr
#'   Same data type as `operand`. Shape is `operand` with `dims` removed
#'   (or set to 1 if `drop = FALSE`).
#' @templateVar primitive_id reduce
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_reduce()] with `reductor` as the body.
#' @seealso [prim_reduce_sum()], [prim_reduce_max()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3, 4))
#' prim_reduce(x, init = nv_scalar(0), dims = 1L, reductor = prim_add)
#' prim_reduce(x, init = nv_scalar(1), dims = 1L, reductor = prim_mul)
#' @export
prim_reduce <- new_primitive(
  "reduce",
  function(operand, init, dims, drop = TRUE, reductor) {
    force(operand)
    force(init)
    force(reductor)

    if (!checkmate::test_integerish(dims, lower = 1)) {
      cli_abort("{.arg dims} must be a positive integer vector")
    }
    if (!checkmate::test_flag(drop)) {
      cli_abort("{.arg drop} must be a flag")
    }
    if (!is.function(reductor)) {
      cli_abort("{.arg reductor} must be a function")
    }

    op_dtype <- dtype(operand)
    init_dtype <- dtype(init)
    if (init_dtype != op_dtype) {
      cli_abort(c(
        "{.arg init} must have the same dtype as {.arg operand}.",
        x = "Got operand dtype {.field {repr(op_dtype)}} and init dtype {.field {repr(init_dtype)}}."
      ))
    }
    if (ndims(init) != 0L) {
      cli_abort("{.arg init} must be a scalar (0-dimensional)")
    }

    current_desc <- .current_descriptor(silent = TRUE)
    desc_red <- local_descriptor()

    dummy_args <- list(
      lhs = nv_aval(op_dtype, integer()),
      rhs = nv_aval(op_dtype, integer())
    )
    reductor_graph <- trace_fn(reductor, dummy_args, desc = desc_red, mode = "subgraph")

    if (length(reductor_graph$outputs) != 1L) {
      cli_abort(c(
        "{.arg reductor} must return exactly one value.",
        x = "Got {length(reductor_graph$outputs)} outputs."
      ))
    }
    out_aval <- reductor_graph$outputs[[1L]]$aval
    if (out_aval$dtype != op_dtype) {
      cli_abort(c(
        "{.arg reductor} must return a value with the same dtype as {.arg operand}.",
        x = "Got reductor output dtype {.field {repr(out_aval$dtype)}}."
      ))
    }

    for (const in reductor_graph$constants) {
      get_box_or_register_const(current_desc, const)
    }

    infer_fn <- function(operand, init, dims, drop, reductor_graph) {
      stub_body <- stablehlo(reductor_graph)[[1L]]
      dims0 <- as.integer(dims) - 1L
      vts <- stablehlo::infer_types_reduce(
        inputs = list(at2vt(operand)),
        init_values = list(at2vt(init)),
        body = stub_body,
        dimensions = stablehlo::r_to_constant(
          dims0,
          dtype = "i64",
          shape = length(dims0)
        )
      )
      out <- vt2at(vts[[1L]])
      out$ambiguous <- operand$ambiguous
      if (!drop) {
        new_shape <- shape(operand)
        new_shape[dims] <- 1L
        out <- AbstractArray(
          dtype = out$dtype,
          shape = Shape(new_shape),
          ambiguous = out$ambiguous
        )
      }
      list(out)
    }

    graph_desc_add(
      self,
      args = list(operand = operand, init = init),
      params = list(dims = dims, drop = drop, reductor_graph = reductor_graph),
      infer_fn = infer_fn,
      desc = current_desc
    )[[1L]]
  },
  subgraphs = "reductor_graph",
  static = c("dims", "drop", "reductor")
)

# Shared shape inference for prim_argmax / prim_argmin: operand -> i32
# with `dim` dropped (or kept as size 1).
infer_fn_arg_extreme <- function(operand, dim, drop) {
  shp <- shape(operand)
  if (dim > length(shp)) {
    cli_abort(c(
      "{.arg dim} is out of bounds.",
      x = "Operand has {length(shp)} dims, got {.arg dim} = {dim}."
    ))
  }
  # The reduction lowering uses `init_v = +/-Inf` and `init_i = 0`. Reducing
  # along a size-0 axis would silently emit those sentinels (i.e. index 1)
  # rather than failing. Argmax/argmin of an empty axis is undefined, so
  # reject it here at trace time.
  if (shp[dim] == 0L) {
    cli_abort(c(
      "argmax/argmin is undefined for an empty axis.",
      x = "Operand has shape {xlamisc::shapevec_repr(shp)}; {.arg dim} = {dim} has size 0."
    ))
  }
  if (drop) {
    new_shape <- shp[-dim]
  } else {
    new_shape <- shp
    new_shape[dim] <- 1L
  }
  list(AbstractArray(
    dtype = "i32",
    shape = Shape(new_shape),
    ambiguous = FALSE
  ))
}

#' @title Primitive Argmax
#' @description
#' Returns the index of the maximum value along a single dimension. Ties
#' are broken by returning the smallest index.
#' @template param_prim_operand_any
#' @param dim (`integer(1)`)\cr
#'   Dimension along which to find the index of the maximum.
#' @param drop (`logical(1)`)\cr
#'   If `TRUE` (default) the reduced dimension is removed; if `FALSE` it is
#'   kept with size 1.
#' @return [`arrayish`] of dtype `i32`\cr
#'   Same shape as `operand` with `dim` removed (or set to 1 if
#'   `drop = FALSE`).
#' @templateVar primitive_id argmax
#' @template section_rules
#' @section StableHLO:
#' Lowers to a variadic [hlo_reduce()] over `(values, indices)`
#' with a (value > value | (value == value & idx < idx)) selector.
#' @seealso [prim_argmin()], [nv_argmax()]
#' @examplesIf pjrt::plugins_downloaded()
#' prim_argmax(nv_array(c(3, 1, 4, 1, 5)), dim = 1L)
#' @export
prim_argmax <- new_primitive(
  "argmax",
  function(operand, dim, drop = TRUE) {
    assert_integerish(dim, lower = 1, len = 1L)
    assert_flag(drop)
    graph_desc_add(
      self,
      args = list(operand = operand),
      params = list(dim = dim, drop = drop),
      infer_fn = infer_fn_arg_extreme
    )[[1L]]
  },
  static = 2:3
)

#' @title Primitive Argmin
#' @description
#' Returns the index of the minimum value along a single dimension. Ties
#' are broken by returning the smallest index.
#' @template param_prim_operand_any
#' @inheritParams prim_argmax
#' @return [`arrayish`] of dtype `i32`\cr
#'   Same shape as `operand` with `dim` removed (or set to 1 if
#'   `drop = FALSE`).
#' @templateVar primitive_id argmin
#' @template section_rules
#' @section StableHLO:
#' Lowers to a variadic [hlo_reduce()] over `(values, indices)`
#' with a (value < value | (value == value & idx < idx)) selector.
#' @seealso [prim_argmax()], [nv_argmin()]
#' @examplesIf pjrt::plugins_downloaded()
#' prim_argmin(nv_array(c(3, 1, 4, 1, 5)), dim = 1L)
#' @export
prim_argmin <- new_primitive(
  "argmin",
  function(operand, dim, drop = TRUE) {
    assert_integerish(dim, lower = 1, len = 1L)
    assert_flag(drop)
    graph_desc_add(
      self,
      args = list(operand = operand),
      params = list(dim = dim, drop = drop),
      infer_fn = infer_fn_arg_extreme
    )[[1L]]
  },
  static = 2:3
)

# comparison primitives --------------------------------------------------------

infer_compare <- function(lhs, rhs, comparison_direction) {
  check_dtype <- as.character(dtype(lhs))
  compare_type <- if ((check_dtype == "bool") || grepl("^ui", check_dtype)) {
    "UNSIGNED"
  } else if (grepl("^i", check_dtype)) {
    "SIGNED"
  } else {
    "FLOAT"
  }
  out <- stablehlo::infer_types_compare(at2vt(lhs), at2vt(rhs), comparison_direction, compare_type)[[1L]]
  out <- vt2at(out)
  out$ambiguous <- lhs$ambiguous && rhs$ambiguous
  list(out)
}

make_compare_op <- function(direction) {
  force(direction)
  infer_fn <- function(lhs, rhs) infer_compare(lhs, rhs, direction)
  function(lhs, rhs) {
    graph_desc_add(self, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
  }
}

#' @title Primitive Equal
#' @description
#' Element-wise equality comparison.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id equal
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_compare()] with `comparison_direction = "EQ"`.
#' @seealso [nv_eq()], `==`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(1, 3, 2))
#' prim_eq(x, y)
#' @export
prim_eq <- new_primitive("equal", make_compare_op("EQ"))

#' @title Primitive Not Equal
#' @description
#' Element-wise inequality comparison.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id not_equal
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_compare()] with `comparison_direction = "NE"`.
#' @seealso [nv_ne()], `!=`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(1, 3, 2))
#' prim_ne(x, y)
#' @export
prim_ne <- new_primitive("not_equal", make_compare_op("NE"))

#' @title Primitive Greater Than
#' @description
#' Element-wise greater than comparison.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id greater
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_compare()] with `comparison_direction = "GT"`.
#' @seealso [nv_gt()], `>`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' prim_gt(x, y)
#' @export
prim_gt <- new_primitive("greater", make_compare_op("GT"))

#' @title Primitive Greater Equal
#' @description
#' Element-wise greater than or equal comparison.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id greater_equal
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_compare()] with `comparison_direction = "GE"`.
#' @seealso [nv_ge()], `>=`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' prim_ge(x, y)
#' @export
prim_ge <- new_primitive("greater_equal", make_compare_op("GE"))

#' @title Primitive Less Than
#' @description
#' Element-wise less than comparison.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id less
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_compare()] with `comparison_direction = "LT"`.
#' @seealso [nv_lt()], `<`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' prim_lt(x, y)
#' @export
prim_lt <- new_primitive("less", make_compare_op("LT"))

#' @title Primitive Less Equal
#' @description
#' Element-wise less than or equal comparison.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_compare
#' @templateVar primitive_id less_equal
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_compare()] with `comparison_direction = "LE"`.
#' @seealso [nv_le()], `<=`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(3, 2, 1))
#' prim_le(x, y)
#' @export
prim_le <- new_primitive("less_equal", make_compare_op("LE"))

# additional simple binary primitives -----------------------------------------

#' @title Primitive Maximum
#' @description
#' Element-wise maximum of two arrays.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_binary
#' @templateVar primitive_id maximum
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_maximum()].
#' @seealso [nv_max()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 5, 3))
#' y <- nv_array(c(4, 2, 6))
#' prim_max(x, y)
#' @export
prim_max <- new_primitive("maximum", make_binary_op(stablehlo::infer_types_maximum))

#' @title Primitive Minimum
#' @description
#' Element-wise minimum of two arrays.
#' @template params_prim_lhs_rhs_any
#' @template return_prim_binary
#' @templateVar primitive_id minimum
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_minimum()].
#' @seealso [nv_min()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 5, 3))
#' y <- nv_array(c(4, 2, 6))
#' prim_min(x, y)
#' @export
prim_min <- new_primitive("minimum", make_binary_op(stablehlo::infer_types_minimum))

#' @title Primitive Remainder
#' @description
#' Element-wise remainder.
#' Result has sign of the divident, which differs from base R's `%%`, which is available
#' via [`nv_mod()`] and has sign of divisor.
#' @template params_prim_lhs_rhs_numeric
#' @template return_prim_binary
#' @templateVar primitive_id remainder
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_remainder()].
#' @seealso [nv_remainder()]
#' @examplesIf pjrt::plugins_downloaded()
#' prim_remainder(1, -3)
#' 1 %% -3
#' @export
prim_remainder <- new_primitive("remainder", make_binary_op(stablehlo::infer_types_remainder))

#' @title Primitive And
#' @description
#' Element-wise logical AND.
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id and
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_and()].
#' @seealso [nv_and()], `&`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' y <- nv_array(c(TRUE, TRUE, FALSE))
#' prim_and(x, y)
#' @export
prim_and <- new_primitive("and", make_binary_op(stablehlo::infer_types_and))

#' @title Primitive Not
#' @description
#' Element-wise logical NOT.
#' @param operand ([`arrayish`])\cr
#'   Arrayish value of data type boolean, integer, or unsigned integer.
#' @template return_prim_unary
#' @templateVar primitive_id not
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_not()].
#' @seealso [nv_not()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' prim_not(x)
#' @export
prim_not <- new_primitive("not", make_unary_op(stablehlo::infer_types_not))

#' @title Primitive Or
#' @description
#' Element-wise logical OR.
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id or
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_or()].
#' @seealso [nv_or()], `|`
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' y <- nv_array(c(TRUE, TRUE, FALSE))
#' prim_or(x, y)
#' @export
prim_or <- new_primitive("or", make_binary_op(stablehlo::infer_types_or))

#' @title Primitive Xor
#' @description
#' Element-wise logical XOR.
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id xor
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_xor()].
#' @seealso [nv_xor()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(TRUE, FALSE, TRUE))
#' y <- nv_array(c(TRUE, TRUE, FALSE))
#' prim_xor(x, y)
#' @export
prim_xor <- new_primitive("xor", make_binary_op(stablehlo::infer_types_xor))

infer_shift <- function(lhs, rhs, shift_fn) {
  both_ambiguous <- lhs$ambiguous && rhs$ambiguous
  out <- shift_fn(at2vt(lhs), at2vt(rhs))[[1L]]
  out <- vt2at(out)
  out$ambiguous <- both_ambiguous
  list(out)
}

#' @title Primitive Shift Left
#' @description
#' Element-wise left bit shift.
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id shift_left
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_shift_left()].
#' @seealso [nv_shift_left()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1L, 2L, 4L))
#' y <- nv_array(c(1L, 2L, 1L))
#' prim_shift_left(x, y)
#' @export
prim_shift_left <- new_primitive(
  "shift_left",
  function(lhs, rhs) {
    infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_left)
    graph_desc_add(self, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
  }
)

#' @title Primitive Logical Shift Right
#' @description
#' Element-wise logical right bit shift.
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id shift_right_logical
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_shift_right_logical()].
#' @seealso [nv_shift_right_logical()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(8L, 16L, 32L))
#' y <- nv_array(c(1L, 2L, 3L))
#' prim_shift_right_logical(x, y)
#' @export
prim_shift_right_logical <- new_primitive(
  "shift_right_logical",
  function(lhs, rhs) {
    infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_right_logical)
    graph_desc_add(self, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
  }
)

#' @title Primitive Arithmetic Shift Right
#' @description
#' Element-wise arithmetic right bit shift.
#' @template params_prim_lhs_rhs_intlike
#' @template return_prim_binary
#' @templateVar primitive_id shift_right_arithmetic
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_shift_right_arithmetic()].
#' @seealso [nv_shift_right_arithmetic()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(8L, -16L, 32L))
#' y <- nv_array(c(1L, 2L, 3L))
#' prim_shift_right_arithmetic(x, y)
#' @export
prim_shift_right_arithmetic <- new_primitive(
  "shift_right_arithmetic",
  function(lhs, rhs) {
    infer_fn <- function(lhs, rhs) infer_shift(lhs, rhs, stablehlo::infer_types_shift_right_arithmetic)
    graph_desc_add(self, list(lhs = lhs, rhs = rhs), infer_fn = infer_fn)[[1L]]
  }
)

#' @title Primitive Atan2
#' @description
#' Element-wise atan2 operation.
#' @template params_prim_lhs_rhs_float
#' @template return_prim_binary
#' @templateVar primitive_id atan2
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_atan2()].
#' @seealso [nv_atan2()]
#' @examplesIf pjrt::plugins_downloaded()
#' y <- nv_array(c(1, 0, -1))
#' x <- nv_array(c(0, 1, 0))
#' prim_atan2(y, x)
#' @export
prim_atan2 <- new_primitive("atan2", make_binary_op(stablehlo::infer_types_atan2))

#' @title Primitive Bitcast Convert
#' @description
#' Reinterprets the bits of an array as a different data type without
#' modifying the underlying data.
#' @template param_prim_operand_any
#' @param dtype (`character(1)` | [`DataType`])\cr
#'   Target data type. If it has the same bit width as the input, the output
#'   shape is unchanged. If narrower, an extra trailing dimension is added.
#'   If wider, the last dimension is consumed.
#' @return [`arrayish`]\cr
#'   Has the given `dtype`.
#' @templateVar primitive_id bitcast_convert
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_bitcast_convert()].
#' @seealso [nv_bitcast_convert()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1L)
#' prim_bitcast_convert(x, dtype = "i8")
#' x <- nv_array(rep(1L, 4), dtype = "i8")
#' prim_bitcast_convert(x, dtype = "i32")
#' @export
prim_bitcast_convert <- new_primitive(
  "bitcast_convert",
  function(operand, dtype) {
    infer_fn <- function(operand, dtype) {
      lapply(stablehlo::infer_types_bitcast_convert(at2vt(operand), dtype), vt2at)
    }
    graph_desc_add(self, list(operand = operand), params = list(dtype = dtype), infer_fn = infer_fn)[[1L]]
  },
  static = 2L
)

# unary math primitives ---------------------------------------------------------

#' @title Primitive Absolute Value
#' @description
#' Element-wise absolute value.
#' @template param_prim_operand_signed_numeric
#' @template return_prim_unary
#' @templateVar primitive_id abs
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_abs()].
#' @seealso [nv_abs()], [abs()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 2, -3))
#' prim_abs(x)
#' @export
prim_abs <- new_primitive("abs", make_unary_op(stablehlo::infer_types_abs))

#' @title Primitive Square Root
#' @description
#' Element-wise square root.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id sqrt
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_sqrt()].
#' @seealso [nv_sqrt()], [sqrt()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 4, 9))
#' prim_sqrt(x)
#' @export
prim_sqrt <- new_primitive("sqrt", make_unary_op(stablehlo::infer_types_sqrt))

#' @title Primitive Reciprocal Square Root
#' @description
#' Element-wise reciprocal square root.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id rsqrt
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_rsqrt()].
#' @seealso [nv_rsqrt()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 4, 9))
#' prim_rsqrt(x)
#' @export
prim_rsqrt <- new_primitive("rsqrt", make_unary_op(stablehlo::infer_types_rsqrt))

#' @title Primitive Logarithm
#' @description
#' Element-wise natural logarithm.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id log
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_log()].
#' @seealso [nv_log()], [log()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2.718, 7.389))
#' prim_log(x)
#' @export
prim_log <- new_primitive("log", make_unary_op(stablehlo::infer_types_log))

#' @title Primitive Hyperbolic Tangent
#' @description
#' Element-wise hyperbolic tangent.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id tanh
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_tanh()].
#' @seealso [nv_tanh()], [tanh()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' prim_tanh(x)
#' @export
prim_tanh <- new_primitive("tanh", make_unary_op(stablehlo::infer_types_tanh))

#' @title Primitive Tangent
#' @description
#' Element-wise tangent.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id tan
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_tan()].
#' @seealso [nv_tan()], [tan()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 0.5, 1))
#' prim_tan(x)
#' @export
prim_tan <- new_primitive("tan", make_unary_op(stablehlo::infer_types_tan))

#' @title Primitive Sine
#' @description
#' Element-wise sine.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id sine
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_sine()].
#' @seealso [nv_sin()], [sin()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, pi / 2, pi))
#' prim_sin(x)
#' @export
prim_sin <- new_primitive("sine", make_unary_op(stablehlo::infer_types_sine))

#' @title Primitive Cosine
#' @description
#' Element-wise cosine.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id cosine
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_cosine()].
#' @seealso [nv_cos()], [cos()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, pi / 2, pi))
#' prim_cos(x)
#' @export
prim_cos <- new_primitive("cosine", make_unary_op(stablehlo::infer_types_cosine))

#' @title Primitive Floor
#' @description
#' Element-wise floor.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id floor
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_floor()].
#' @seealso [nv_floor()], [floor()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1.2, 2.7, -1.5))
#' prim_floor(x)
#' @export
prim_floor <- new_primitive("floor", make_unary_op(stablehlo::infer_types_floor))

#' @title Primitive Ceiling
#' @description
#' Element-wise ceiling.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id ceil
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_ceil()].
#' @seealso [nv_ceiling()], [ceiling()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1.2, 2.7, -1.5))
#' prim_ceil(x)
#' @export
prim_ceil <- new_primitive("ceil", make_unary_op(stablehlo::infer_types_ceil))

#' @title Primitive Sign
#' @description
#' Element-wise sign.
#' @template param_prim_operand_signed_numeric
#' @template return_prim_unary
#' @templateVar primitive_id sign
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_sign()].
#' @seealso [nv_sign()], [sign()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-3, 0, 5))
#' prim_sign(x)
#' @export
prim_sign <- new_primitive("sign", make_unary_op(stablehlo::infer_types_sign))

#' @title Primitive Exponential
#' @description
#' Element-wise exponential.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id exp
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_exponential()].
#' @seealso [nv_exp()], [exp()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 1, 2))
#' prim_exp(x)
#' @export
prim_exp <- new_primitive("exp", make_unary_op(stablehlo::infer_types_exponential))

#' @title Primitive Exponential Minus One
#' @description
#' Element-wise exp(x) - 1, more accurate for small x.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id expm1
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_exponential_minus_one()].
#' @seealso [nv_expm1()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 0.001, 1))
#' prim_expm1(x)
#' @export
prim_expm1 <- new_primitive("expm1", make_unary_op(stablehlo::infer_types_exponential_minus_one))

#' @title Primitive Log Plus One
#' @description
#' Element-wise log(1 + x), more accurate for small x.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id log1p
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_log_plus_one()].
#' @seealso [nv_log1p()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 0.001, 1))
#' prim_log1p(x)
#' @export
prim_log1p <- new_primitive("log1p", make_unary_op(stablehlo::infer_types_log_plus_one))

#' @title Primitive Cube Root
#' @description
#' Element-wise cube root.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id cbrt
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_cbrt()].
#' @seealso [nv_cbrt()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 8, 27))
#' prim_cbrt(x)
#' @export
prim_cbrt <- new_primitive("cbrt", make_unary_op(stablehlo::infer_types_cbrt))

#' @title Primitive Logistic (Sigmoid)
#' @description
#' Element-wise logistic sigmoid: 1 / (1 + exp(-x)).
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id logistic
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_logistic()].
#' @seealso [nv_logistic()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-2, 0, 2))
#' prim_logistic(x)
#' @export
prim_logistic <- new_primitive("logistic", make_unary_op(stablehlo::infer_types_logistic))

#' @title Primitive Arc Cosine
#' @description
#' Element-wise inverse cosine.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id acos
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_acos()].
#' @seealso [nv_acos()], [acos()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' prim_acos(x)
#' @export
prim_acos <- new_primitive("acos", make_unary_op(stablehlo::infer_types_acos))

#' @title Primitive Inverse Hyperbolic Cosine
#' @description
#' Element-wise inverse hyperbolic cosine.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id acosh
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_acosh()].
#' @seealso [nv_acosh()], [acosh()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 10))
#' prim_acosh(x)
#' @export
prim_acosh <- new_primitive("acosh", make_unary_op(stablehlo::infer_types_acosh))

#' @title Primitive Arc Sine
#' @description
#' Element-wise inverse sine.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id asin
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_asin()].
#' @seealso [nv_asin()], [asin()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' prim_asin(x)
#' @export
prim_asin <- new_primitive("asin", make_unary_op(stablehlo::infer_types_asin))

#' @title Primitive Inverse Hyperbolic Sine
#' @description
#' Element-wise inverse hyperbolic sine.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id asinh
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_asinh()].
#' @seealso [nv_asinh()], [asinh()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' prim_asinh(x)
#' @export
prim_asinh <- new_primitive("asinh", make_unary_op(stablehlo::infer_types_asinh))

#' @title Primitive Arc Tangent
#' @description
#' Element-wise inverse tangent.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id atan
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_atan()].
#' @seealso [nv_atan()], [atan()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' prim_atan(x)
#' @export
prim_atan <- new_primitive("atan", make_unary_op(stablehlo::infer_types_atan))

#' @title Primitive Inverse Hyperbolic Tangent
#' @description
#' Element-wise inverse hyperbolic tangent.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id atanh
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_atanh()].
#' @seealso [nv_atanh()], [atanh()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-0.5, 0, 0.5))
#' prim_atanh(x)
#' @export
prim_atanh <- new_primitive("atanh", make_unary_op(stablehlo::infer_types_atanh))

#' @title Primitive Hyperbolic Cosine
#' @description
#' Element-wise hyperbolic cosine.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id cosh
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_cosh()].
#' @seealso [nv_cosh()], [cosh()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' prim_cosh(x)
#' @export
prim_cosh <- new_primitive("cosh", make_unary_op(stablehlo::infer_types_cosh))

#' @title Primitive Hyperbolic Sine
#' @description
#' Element-wise hyperbolic sine.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id sinh
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_sinh()].
#' @seealso [nv_sinh()], [sinh()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' prim_sinh(x)
#' @export
prim_sinh <- new_primitive("sinh", make_unary_op(stablehlo::infer_types_sinh))

#' @title Primitive Digamma
#' @description
#' Element-wise digamma function (logarithmic derivative of the gamma function).
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id digamma
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_digamma()].
#' @seealso [nv_digamma()], [digamma()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0.5, 1, 2, 5))
#' prim_digamma(x)
#' @export
prim_digamma <- new_primitive("digamma", make_unary_op(stablehlo::infer_types_digamma))

#' @title Primitive Log-Gamma
#' @description
#' Element-wise natural logarithm of the absolute value of the gamma function.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id lgamma
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_lgamma()].
#' @seealso [nv_lgamma()], [lgamma()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0.5, 1, 2, 5))
#' prim_lgamma(x)
#' @export
prim_lgamma <- new_primitive("lgamma", make_unary_op(stablehlo::infer_types_lgamma))

#' @title Primitive Polygamma
#' @description
#' Element-wise polygamma function: the `(n+1)`-th derivative of the
#' log-gamma function. Both `n` and `x` must have the same shape; `n`
#' typically holds non-negative integer values.
#' @param n,x ([`arrayish`])\cr
#'   Arrayish values of data type floating-point.
#'   Must have the same shape.
#' @template return_prim_binary
#' @templateVar primitive_id polygamma
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_polygamma()].
#' @seealso [nv_polygamma()]
#' @examplesIf pjrt::plugins_downloaded()
#' n <- nv_array(c(1, 1, 2))
#' x <- nv_array(c(0.5, 1, 2))
#' prim_polygamma(n, x)
#' @export
prim_polygamma <- new_primitive(
  "polygamma",
  function(n, x) {
    infer_fn <- function(n, x) {
      both_ambiguous <- n$ambiguous && x$ambiguous
      out <- stablehlo::infer_types_polygamma(at2vt(n), at2vt(x))[[1L]]
      out <- vt2at(out)
      out$ambiguous <- both_ambiguous
      list(out)
    }
    graph_desc_add(self, list(n = n, x = x), infer_fn = infer_fn)[[1L]]
  }
)

#' @title Primitive Error Function
#' @description
#' Element-wise error function `erf(x) = (2 / sqrt(pi)) * integral_0^x exp(-t^2) dt`.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id erf
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_erf()].
#' @seealso [nv_erf()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' prim_erf(x)
#' @export
prim_erf <- new_primitive("erf", make_unary_op(stablehlo::infer_types_erf))

#' @title Primitive Inverse Error Function
#' @description
#' Element-wise inverse error function (the inverse of `erf` on `(-1, 1)`).
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id erf_inv
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_erf_inv()].
#' @seealso [nv_erf_inv()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-0.5, 0, 0.5))
#' prim_erf_inv(x)
#' @export
prim_erf_inv <- new_primitive("erf_inv", make_unary_op(stablehlo::infer_types_erf_inv))

#' @title Primitive Complementary Error Function
#' @description
#' Element-wise complementary error function `erfc(x) = 1 - erf(x)`.
#' @template param_prim_operand_float
#' @template return_prim_unary
#' @templateVar primitive_id erfc
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_erfc()].
#' @seealso [nv_erfc()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' prim_erfc(x)
#' @export
prim_erfc <- new_primitive("erfc", make_unary_op(stablehlo::infer_types_erfc))

#' @title Primitive Is Finite
#' @description
#' Element-wise check if values are finite (not Inf, -Inf, or NaN).
#' @template param_prim_operand_float
#' @return [`arrayish`]\cr
#'   Has the same shape as the input and boolean data type.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id is_finite
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_is_finite()].
#' @seealso [nv_is_finite()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, Inf, NaN, -Inf, 0))
#' prim_is_finite(x)
#' @export
prim_is_finite <- new_primitive(
  "is_finite",
  function(operand) {
    infer_fn <- function(operand) {
      out <- stablehlo::infer_types_is_finite(at2vt(operand))[[1L]]
      list(vt2at(out))
    }
    graph_desc_add(self, list(operand = operand), list(), infer_fn = infer_fn)[[1L]]
  }
)

#' @title Primitive Population Count
#' @description
#' Element-wise population count (number of set bits).
#' @param operand ([`arrayish`])\cr
#'   Arrayish value of data type integer or unsigned integer.
#' @template return_prim_unary
#' @templateVar primitive_id popcnt
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_popcnt()].
#' @seealso [nv_popcnt()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(7L, 3L, 15L))
#' prim_popcnt(x)
#' @export
prim_popcnt <- new_primitive(
  "popcnt",
  function(operand) {
    infer_fn <- function(operand) {
      out <- stablehlo::infer_types_popcnt(at2vt(operand))[[1L]]
      out <- vt2at(out)
      out$ambiguous <- operand$ambiguous
      list(out)
    }
    graph_desc_add(self, list(operand = operand), list(), infer_fn = infer_fn)[[1L]]
  }
)

#' @title Primitive Clamp
#' @description
#' Clamps every element of `operand` to the range `[min_val, max_val]`,
#' i.e. `max(min_val, min(operand, max_val))`.
#' @param min_val ([`arrayish`])\cr
#'   Minimum value. Must be scalar or the same shape as `operand`.
#' @template param_prim_operand_any
#' @param max_val ([`arrayish`])\cr
#'   Maximum value. Must be scalar or the same shape as `operand`.
#' @return [`arrayish`]\cr
#'   Has the same data type and shape as `operand`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id clamp
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_clamp()].
#' @seealso [nv_clamp()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0.5, 2))
#' prim_clamp(nv_scalar(0), x, nv_scalar(1))
#' @export
prim_clamp <- new_primitive(
  "clamp",
  function(min_val, operand, max_val) {
    infer_fn <- function(min_val, operand, max_val) {
      out <- stablehlo::infer_types_clamp(at2vt(min_val), at2vt(operand), at2vt(max_val))[[1L]]
      out <- vt2at(out)
      out$ambiguous <- operand$ambiguous
      list(out)
    }
    graph_desc_add(
      self,
      list(min_val = min_val, operand = operand, max_val = max_val),
      list(),
      infer_fn = infer_fn
    )[[
      1L
    ]]
  }
)

#' @title Primitive Reverse
#' @description
#' Reverses the order of elements along specified dimensions.
#' @template param_prim_operand_any
#' @param dims (`integer()`)\cr
#'   Dimensions to reverse (1-indexed).
#' @return [`arrayish`]\cr
#'   Has the same data type and shape as `operand`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id reverse
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_reverse()].
#' @seealso [nv_reverse()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3, 4, 5))
#' prim_reverse(x, dims = 1L)
#' @export
prim_reverse <- new_primitive(
  "reverse",
  function(operand, dims) {
    infer_fn <- function(operand, dims) {
      # stablehlo uses 0-based indexing
      dims_attr <- r_to_constant(dims - 1L, dtype = "i64", shape = length(dims))
      out <- stablehlo::infer_types_reverse(at2vt(operand), dimensions = dims_attr)[[1L]]
      out <- vt2at(out)
      out$ambiguous <- operand$ambiguous
      list(out)
    }
    graph_desc_add(self, list(operand = operand), list(dims = dims), infer_fn = infer_fn)[[1L]]
  },
  static = 2L
)

#' @title Primitive Iota
#' @description
#' Creates an array with values increasing along the specified dimension.
#' @param dim (`integer(1)`)\cr
#'   Dimension along which values increase (1-indexed).
#' @template param_dtype
#' @param shape (`integer()`)\cr
#'   Shape of the output array.
#' @param start (`integer(1)`)\cr
#'   Starting value.
#' @template param_ambiguous
#' @template param_device
#' @return [`arrayish`]\cr
#'   Has the given `dtype` and `shape`.
#' @templateVar primitive_id iota
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_iota()].
#' @seealso [nv_iota()]
#' @examplesIf pjrt::plugins_downloaded()
#' prim_iota(dim = 1L, dtype = "i32", shape = 5L)
#' @export
prim_iota <- new_primitive(
  "iota",
  function(dim, dtype, shape, start = 1L, ambiguous = FALSE, device = NULL) {
    infer_fn <- function(dim, dtype, shape, start, ambiguous) {
      # stablehlo uses 0-based indexing, anvl uses 1-based
      # Convert dim to Constant as required by stablehlo
      iota_dim_const <- stablehlo::r_to_constant(
        as.integer(dim - 1L),
        dtype = "i64",
        shape = integer(0)
      )
      # Just for the checks
      stablehlo::infer_types_iota(iota_dimension = iota_dim_const, dtype = dtype, shape = shape)[[1L]]

      list(IotaArray(shape = shape, dtype = dtype, dimension = dim, start = start, ambiguous = ambiguous))
    }
    result <- graph_desc_add(
      self,
      list(),
      list(dim = dim, dtype = dtype, shape = shape, start = start, ambiguous = ambiguous),
      infer_fn = infer_fn
    )[[1L]]

    result
  },
  static = 1:6,
  device = device_arg("device")
)

#' @title Primitive Pad
#' @description
#' Pads an array with a given padding value.
#' @template param_prim_operand_any
#' @param padding_value ([`arrayish`])\cr
#'   Scalar value to use for padding. Must have the same dtype as `operand`.
#' @param edge_padding_low (`integer()`)\cr
#'   Amount of padding to add at the start of each dimension.
#' @param edge_padding_high (`integer()`)\cr
#'   Amount of padding to add at the end of each dimension.
#' @param interior_padding (`integer()`)\cr
#'   Amount of padding to add between elements in each dimension.
#' @return [`arrayish`]\cr
#'   Has the same data type as `operand`.
#'   For the output shape see the underlying stablehlo documentation ([hlo_pad()]).
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id pad
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_pad()].
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' prim_pad(x, nv_scalar(0),
#'   edge_padding_low = 2L, edge_padding_high = 1L, interior_padding = 0L
#' )
#' @export
prim_pad <- new_primitive(
  "pad",
  function(operand, padding_value, edge_padding_low, edge_padding_high, interior_padding) {
    infer_fn <- function(operand, padding_value, edge_padding_low, edge_padding_high, interior_padding) {
      rank <- ndims_abstract(operand)
      low_attr <- r_to_constant(edge_padding_low, dtype = "i64", shape = rank)
      high_attr <- r_to_constant(edge_padding_high, dtype = "i64", shape = rank)
      interior_attr <- r_to_constant(interior_padding, dtype = "i64", shape = rank)
      out <- stablehlo::infer_types_pad(
        at2vt(operand),
        at2vt(padding_value),
        edge_padding_low = low_attr,
        edge_padding_high = high_attr,
        interior_padding = interior_attr
      )[[1L]]
      out <- vt2at(out)
      out$ambiguous <- operand$ambiguous
      list(out)
    }

    graph_desc_add(
      self,
      list(operand = operand, padding_value = padding_value),
      list(
        edge_padding_low = edge_padding_low,
        edge_padding_high = edge_padding_high,
        interior_padding = interior_padding
      ),
      infer_fn = infer_fn
    )[[1L]]
  },
  static = 3:5
)

#' @title Primitive Round
#' @description
#' Rounds the elements of an array to the nearest integer.
#' @template param_prim_operand_float
#' @param method (`character(1)`)\cr
#'   Rounding method. `"nearest_even"` (default) rounds to the nearest even
#'   integer on a tie, `"afz"` rounds away from zero on a tie.
#' @return [`arrayish`]\cr
#'   Has the same dtype and shape as `operand`.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id round
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_round_nearest_even()] or
#' [hlo_round_nearest_afz()] depending on the `method` parameter.
#' @seealso [nv_round()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1.4, 2.5, 3.6))
#' prim_round(x)
#' @export
prim_round <- new_primitive(
  "round",
  function(operand, method = "nearest_even") {
    if (!(method %in% c("nearest_even", "afz"))) {
      cli_abort("method must be one of: 'nearest_even', 'afz', but is {method}")
    }
    infer_fn <- function(operand, method) {
      # both rounding functions have the same inference, so just pick one:
      stablehlo_infer <- stablehlo::infer_types_round_nearest_even
      out <- stablehlo_infer(at2vt(operand))[[1L]]
      out <- vt2at(out)
      out$ambiguous <- operand$ambiguous
      list(out)
    }
    graph_desc_add(self, list(operand = operand), list(method = method), infer_fn = infer_fn)[[1L]]
  },
  static = 2L
)

# dtype conversion ----------------------------------------------------------------

#' @title Primitive Convert
#' @description
#' Converts the elements of an array to a different data type.
#' @template param_prim_operand_any
#' @param dtype (`character(1)` | [`DataType`])\cr
#'   Target data type.
#' @template param_ambiguous
#' @return [`arrayish`]\cr
#'   Has the given `dtype` and the same shape as `operand`.
#'   Ambiguity is controlled by the `ambiguous` parameter.
#' @templateVar primitive_id convert
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_convert()].
#' @seealso [nv_convert()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1L, 2L, 3L))
#' prim_convert(x, dtype = "f32")
#' @export
prim_convert <- new_primitive(
  "convert",
  function(operand, dtype, ambiguous = FALSE) {
    dtype <- as_dtype(dtype)
    infer_fn <- function(operand, dtype, ambiguous) {
      list(AbstractArray(
        dtype = dtype,
        shape = Shape(shape(operand)),
        ambiguous = ambiguous
      ))
    }
    graph_desc_add(
      self,
      list(operand = operand),
      params = list(dtype = dtype, ambiguous = ambiguous),
      infer_fn = infer_fn
    )[[1L]]
  },
  static = 2:3
)


#' @title Primitive Ifelse
#' @description
#' Element-wise selection based on a boolean predicate, like R's [ifelse()].
#' For each element, returns the corresponding element from `true_value` where
#' `pred` is `TRUE` and from `false_value` where `pred` is `FALSE`.
#' @param pred ([`arrayish`] of boolean type)\cr
#'   Predicate array. Must be scalar or have the same shape as
#'   `true_value`.
#' @param true_value,false_value ([`arrayish`])\cr
#'   Values to select from. Must have the same dtype and shape.
#' @return [`arrayish`]\cr
#'   Has the same dtype and shape as `true_value`.
#'   It is ambiguous if both `true_value` and `false_value` are ambiguous.
#' @templateVar primitive_id select
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_select()].
#' @seealso [nv_ifelse()]
#' @examplesIf pjrt::plugins_downloaded()
#' pred <- nv_array(c(TRUE, FALSE, TRUE))
#' prim_ifelse(pred, nv_array(c(1, 2, 3)), nv_array(c(4, 5, 6)))
#' @export
prim_ifelse <- new_primitive(
  "select",
  function(pred, true_value, false_value) {
    infer_fn <- function(pred, true_value, false_value) {
      both_ambiguous <- true_value$ambiguous && false_value$ambiguous
      out <- stablehlo::infer_types_select(
        at2vt(pred),
        on_true = at2vt(true_value),
        on_false = at2vt(false_value)
      )[[1L]]
      out <- vt2at(out)
      out$ambiguous <- both_ambiguous
      list(out)
    }
    graph_desc_add(
      self,
      list(pred = pred, true_value = true_value, false_value = false_value),
      infer_fn = infer_fn
    )[[
      1L
    ]]
  }
)

# Higher order primitives -------------------------------------------------------

#' @title Primitive If
#' @description
#' Conditional execution of one of two branches based on a scalar boolean
#' predicate. Unlike [prim_ifelse()] which operates element-wise, this
#' evaluates only the selected branch.
#' @param pred ([`arrayish`])\cr
#'   Scalar boolean predicate that determines which branch to execute.
#' @param true,false (`function()`)\cr
#'   Zero-argument functions for the true and false branches. Both must return outputs
#'   with the same structure, dtypes, and shapes.
#' @return Result of the executed branch.\cr
#'   An output is ambiguous if it is ambiguous in both branches.
#' @templateVar primitive_id if
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_if()].
#' @seealso [nv_if()], [prim_ifelse()]
#' @examplesIf pjrt::plugins_downloaded()
#' prim_if(nv_scalar(TRUE), \() nv_scalar(1), \() nv_scalar(2))
#' @export
prim_if <- new_primitive(
  "if",
  function(pred, true, false) {
    force(pred)
    force(true)
    force(false)

    # Build sub-graphs for each branch (no inputs, just capture closed-over values)
    # We need to ensure that constants that are captured in both branches receive the same
    # GraphValue if they capture the same constant

    current_desc <- .current_descriptor(silent = TRUE)

    desc_true <- local_descriptor()
    true_graph <- trace_fn(true, list(), desc = desc_true, mode = "subgraph")
    desc_false <- local_descriptor()

    register_consts(desc_false, desc_true$constants)
    false_graph <- trace_fn(false, list(), desc = desc_false, mode = "subgraph")

    register_consts(current_desc, desc_false$constants)

    if (!pjrt::tree_equal(true_graph$out_tree, false_graph$out_tree)) {
      cli_abort("true and false branches must have the same output structure")
    }

    # TODO: Apply promotion rules to the outputs of the branches

    infer_fn <- function(pred, true_graph, false_graph) {
      # The returned values might have different ambiguity, so we need to handle it.
      # An output is ambiguous if its type is ambiguous in both branches.
      lapply(seq_along(true_graph$outputs), function(i) {
        aval_true <- true_graph$outputs[[i]]$aval
        aval_false <- false_graph$outputs[[i]]$aval
        if (aval_true$ambiguous && aval_false$ambiguous) {
          return(aval_true)
        }

        aval_true$ambiguous <- FALSE
        return(aval_true)
      })
    }

    out <- graph_desc_add(
      self,
      list(pred = pred),
      params = list(true_graph = true_graph, false_graph = false_graph),
      infer_fn = infer_fn,
      desc = current_desc
    )
    unflatten(true_graph$out_tree, out)
  },
  subgraphs = c("true_graph", "false_graph"),
  static = 2:3
)

#' @title Primitive While Loop
#' @description
#' Repeatedly executes `body` while `cond` returns `TRUE`, like R's
#' `while` loop. The loop state is initialized with `init` and
#' passed through each iteration.
#' Otherwise, no state is maintained between iterations.
#' @param init (`named list()`)\cr
#'   Named list of initial state values.
#' @param cond (`function`)\cr
#'   Condition function that receives the current state as arguments
#'   and outputs whether to continue the loop.
#' @param body (`function`)\cr
#'   Body function that receives the current state as arguments and
#'   returns a named list with the same structure, dtypes, and shapes
#'   as `init`.
#' @return Named list with the same structure as `init` containing the
#'   final state after the loop terminates.
#' @templateVar primitive_id while
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_while()].
#' @seealso [nv_while()]
#' @examplesIf pjrt::plugins_downloaded()
#' prim_while(
#'   init = list(i = nv_scalar(0L), total = nv_scalar(0L)),
#'   cond = function(i, total) i <= 5L,
#'   body = function(i, total) list(
#'     i = i + 1L,
#'     total = total + i
#'   )
#' )
#' @export
prim_while <- new_primitive(
  "while",
  function(init, cond, body) {
    # delayed promise evaluation can cause the value to be added to the wrong graph descriptor
    force(init)
    if (!is.function(body)) {
      cli_abort("body must be a function")
    }
    if (!is.function(cond)) {
      cli_abort("cond must be a function")
    }

    state_names <- names(init)

    if (any(state_names == "")) {
      cli_abort("init must have only named arguments")
    }

    current_desc <- .current_descriptor(silent = TRUE)

    desc_cond <- local_descriptor()

    cond_graph <- trace_fn(cond, init, desc = desc_cond, mode = "subgraph")

    desc_body <- local_descriptor()

    # ensure that constant ids are the same between cond and body
    # inputs don't matter, because we don't inline the sub-graphs into the parent graph
    register_consts(desc_body, desc_cond$constants)
    body_graph <- trace_fn(body, init, desc_body, mode = "subgraph")

    if (!pjrt::tree_equal(cond_graph$in_tree, body_graph$in_tree)) {
      cli_abort("cond and body must have the same input structure")
    }

    if (!pjrt::tree_equal(body_graph$in_tree, body_graph$out_tree)) {
      cli_abort("body must have the same input and output structure")
    }

    # now we register the constants of both sub-graphs (body includes cond's constants) into the graph
    register_consts(current_desc, body_graph$constants)

    infer_fn <- function(..., cond_graph, body_graph) {
      outs <- list(...)
      outs_body <- lapply(body_graph$outputs, \(out) out$aval)
      inputs_body <- lapply(body_graph$inputs, \(inp) inp$aval)
      # ignore ambiguity when comparing dtypes
      if (!all(sapply(seq_along(outs), \(i) eq_type(outs[[i]], outs_body[[i]], ambiguity = FALSE)))) {
        cli_abort("outs must be have same type as outs_body")
      }
      if (!all(sapply(seq_along(inputs_body), \(i) eq_type(inputs_body[[i]], outs_body[[i]], ambiguity = FALSE)))) {
        cli_abort("inputs_body must be have same type as outs_body")
      }
      # function might change the ambiguity, so we return the body outputs and not the inputs
      return(outs_body)
    }

    out <- graph_desc_add(
      self,
      args = flatten(init),
      params = list(cond_graph = cond_graph, body_graph = body_graph),
      infer_fn = infer_fn,
      desc = current_desc
    )

    unflatten(body_graph$out_tree, out)
  },
  subgraphs = c("cond_graph", "body_graph"),
  static = 2:3
)

#' @title Primitive Sort
#' @description
#' Sorts arrays along the given dimension.
#'
#' Sorting is determined by the *first operand* only: it is the sort key,
#' and any additional operands are reordered with the same permutation
#' that sorts the first. This enables idioms like *argsort* (sort `x`
#' paired with an `iota` and read off the second output) and key-value
#' sorts (sort `keys` paired with `values`).
#'
#' All operands must have the same shape; their dtypes may differ.
#' 1-dimensional slices along `dim` are sorted independently; other
#' dimensions are preserved.
#' @param operands (`list` of [`arrayish`])\cr
#'   One or more arrays to sort. The first is the sort key; the rest are
#'   carried along under the same permutation. All must share the same shape.
#' @param dim (`integer(1)`)\cr
#'   Dimension along which to sort.
#' @param descending (`logical(1)`)\cr
#'   If `TRUE`, sort the key in descending order (largest first). Default
#'   `FALSE`. Additional operands are reordered by the same permutation
#'   regardless.
#' @param is_stable (`logical(1)`)\cr
#'   If `TRUE`, the sort is stable: the relative order of equal *keys* is
#'   preserved. Default `FALSE`.
#' @return `list` of [`arrayish`]\cr
#'   One sorted output per element of `operands`, in the same order. Each
#'   output has the same shape, data type, and ambiguity as the
#'   corresponding input.
#' @templateVar primitive_id sort
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_sort()] with a comparator that uses
#' [hlo_compare()] (`LT` for ascending, `GT` for descending) on
#' the first operand. For float keys the comparator uses
#' `compare_type = "TOTALORDER"` and canonicalizes `-0`/`+0` and
#' `-NaN`/`+NaN` to their positive form before comparing, so all `NaN`
#' values land at one end of the result regardless of sign. Integer keys
#' use `SIGNED` / `UNSIGNED` as appropriate.
#' @seealso [nv_sort()], [nv_argsort()], [nv_top_k()], [nv_median()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(3, 1, 4, 1, 5))
#' prim_sort(list(x), dim = 1L)[[1L]]
#'
#' # Sort indices by the values (argsort): pair x with iota and read off
#' # the second result.
#' idx <- nv_iota(dim = 1L, dtype = "i64", shape = 5L)
#' out <- prim_sort(list(x, idx), dim = 1L)
#' out[[1L]] # sorted x
#' out[[2L]] # permutation indices
#' @export
prim_sort <- new_primitive(
  "sort",
  function(operands, dim = 1L, descending = FALSE, is_stable = FALSE) {
    assert_integerish(dim, lower = 1, len = 1)
    assert_flag(descending)
    assert_flag(is_stable)
    if (!is.list(operands) || !length(operands)) {
      cli_abort("{.arg operands} must be a non-empty list of arrayish values")
    }
    ref_shape <- shape(operands[[1L]])
    for (i in seq_along(operands)[-1L]) {
      if (!identical(shape(operands[[i]]), ref_shape)) {
        cli_abort(c(
          "All operands of {.fn prim_sort} must have the same shape.",
          x = "Operand 1 has shape {xlamisc::shapevec_repr(ref_shape)}, operand {i} has shape {xlamisc::shapevec_repr(shape(operands[[i]]))}."
        ))
      }
    }
    if (dim > length(ref_shape)) {
      cli_abort(c(
        "{.arg dim} not in valid range.",
        x = "Operand has {length(ref_shape)} dim(s), got {.arg dim} = {dim}."
      ))
    }

    # Output shape/dtype mirrors each input — sort only permutes along `dim`.
    infer_fn <- function(..., dim, descending, is_stable) {
      ops <- list(...)
      lapply(ops, function(op) {
        AbstractArray(
          dtype = dtype(op),
          shape = Shape(shape(op)),
          ambiguous = op$ambiguous
        )
      })
    }

    graph_desc_add(
      self,
      args = operands,
      params = list(dim = dim, descending = descending, is_stable = is_stable),
      infer_fn = infer_fn
    )
  },
  static = c("dim", "descending", "is_stable")
)

#' @title Primitive Top-K
#' @description
#' Returns the `k` largest values along the last dimension, sorted in
#' descending order, together with their indices into that dimension.
#'
#' For other dimensions, transpose so the target dimension is last, call
#' `prim_top_k()`, then transpose back. [nv_top_k()] does this.
#' @param operand ([`arrayish`])\cr
#'   Tensor of integer, unsigned integer, or floating-point dtype with rank >= 1.
#' @param k (`integer(1)`)\cr
#'   Number of top elements. Must satisfy
#'   `1 <= k <= shape(operand)[ndims(operand)]`.
#' @return `list` of two [`arrayish`] values:\cr
#'   The top-`k` values (same dtype as `operand`) and their indices along
#'   the last dimension (dtype `i32`, matching JAX). Both have the same
#'   shape as `operand` with the last dimension replaced by `k`. Ties are
#'   broken by lower index first.
#' @templateVar primitive_id top_k
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_top_k()].
#' @seealso [nv_top_k()], [prim_sort()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
#' prim_top_k(x, k = 3L)
#' @export
prim_top_k <- new_primitive(
  "top_k",
  function(operand, k) {
    assert_integerish(k, lower = 1L, len = 1L)
    k <- as.integer(k)

    infer_fn <- function(operand, k) {
      k_const <- stablehlo::r_to_constant(
        k,
        dtype = "i64",
        shape = integer()
      )
      vts <- stablehlo::infer_types_top_k(at2vt(operand), k = k_const)
      values <- vt2at(vts[[1L]])
      values$ambiguous <- operand$ambiguous
      indices <- vt2at(vts[[2L]])
      indices$ambiguous <- FALSE
      list(values, indices)
    }

    graph_desc_add(
      self,
      args = list(operand = operand),
      params = list(k = k),
      infer_fn = infer_fn
    )
  },
  static = "k"
)

# Print primitive
#' @title Primitive Print
#' @description
#' Prints an array value to the console during execution and returns the
#' input unchanged. This is useful for debugging JIT-compiled code.
#' @template param_prim_operand_any
#' @return [`arrayish`]\cr
#'   Returns `operand` as-is.
#' @templateVar primitive_id print
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_custom_call()].
#' @seealso [nv_print()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3))
#' prim_print(x)
#' @export
prim_print <- new_primitive(
  "print",
  function(operand) {
    # HACK: ambiguity is not available in stablehlo, so we need to pre-compute this
    # and pass it as a "param", although it is not really one
    # TODO: We should also include the platform/device, but it is currently not avilable in GraphDescriptor
    dtype_str <- paste0(as.character(dtype(operand)), if (ambiguous_abstract(operand)) "?")
    footer <- sprintf("[ %s{%s} ]", dtype_str, paste0(shape(operand), collapse = ","))
    # slig
    graph_desc_add(self, list(operand = operand), list(footer = footer), infer_fn = function(operand, ...) {
      list(operand)
    })[[1L]]
  }
)

# RNG primitives
#' @title Primitive RNG Bit Generator
#' @description
#' Generates pseudo-random numbers using the specified algorithm and returns
#' the updated RNG state together with the generated values.
#' @template param_initial_state
#' @param rng_algorithm (`character(1)`)\cr
#'   RNG algorithm name. Default is `"THREE_FRY"`.
#' @param dtype (`character(1)` | [`DataType`])\cr
#'   Data type of the generated random values.
#' @template param_shape
#' @return `list` of two [`arrayish`] values:\cr
#'   The first element is the updated RNG state with the same dtype and shape
#'   as `initial_state`. The second element is an array of random values with
#'   the given `dtype` and `shape`.
#' @templateVar primitive_id rng_bit_generator
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_rng_bit_generator()].
#' @seealso [nv_runif()], [nv_rnorm()]
#' @examplesIf pjrt::plugins_downloaded()
#' state <- nv_array(c(0L, 0L), dtype = "ui64")
#' prim_rng_bit_generator(state, dtype = "f32", shape = c(3, 2))
#' @export
prim_rng_bit_generator <- new_primitive(
  "rng_bit_generator",
  function(initial_state, rng_algorithm = "THREE_FRY", dtype, shape) {
    infer_fn <- function(initial_state, rng_algorithm, dtype, shape) {
      lapply(stablehlo::infer_types_rng_bit_generator(at2vt(initial_state), rng_algorithm, dtype, shape), vt2at)
    }
    graph_desc_add(
      self,
      list(initial_state = initial_state),
      params = list(rng_algorithm = rng_algorithm, dtype = dtype, shape = shape),
      infer_fn = infer_fn
    )
  },
  static = 2:4
)

#' @title Primitive Scatter
#' @description
#' Produces a result array identical to `input` except that slices at
#' positions specified by `scatter_indices` are updated with values from
#' the `update` array. When multiple indices point to the same location,
#' the `update_computation` function determines how to combine the values
#' (by default the new value replaces the old one).
#'
#' This is the inverse of [prim_gather()]: gather reads slices from an array
#' at given indices, while scatter writes slices into an array at given
#' indices.
#' @param input ([`arrayish`])\cr
#'   Arrayish value of any data type. The base array to scatter into.
#' @param scatter_indices ([`arrayish`] of integer type)\cr
#'   Array of indices. Contains index vectors that map to positions in
#'   `input` via `scatter_dims_to_operand_dims`. The dimension specified
#'   by `index_vector_dim` holds the index vectors.
#' @param update ([`arrayish`])\cr
#'   Update values array. Must have the same data type as `input`.
#' @param update_window_dims (`integer()`)\cr
#'   Dimensions of `update` that are window dimensions, i.e. they
#'   correspond to the slice being written into `input`.
#' @param inserted_window_dims (`integer()`)\cr
#'   Dimensions of `input` whose slices have size 1 and are inserted
#'   (not present) in the `update` window. Together with
#'   `update_window_dims` and `input_batching_dims`, these must account
#'   for all dimensions of `input`.
#' @param input_batching_dims (`integer()`)\cr
#'   Dimensions of `input` that are batch dimensions.
#'   Use `integer(0)` when there are no batch dimensions.
#' @param scatter_indices_batching_dims (`integer()`)\cr
#'   Dimensions of `scatter_indices` that correspond to batch
#'   dimensions. Must have the same length as `input_batching_dims`.
#' @param scatter_dims_to_operand_dims (`integer()`)\cr
#'   Maps each component of the index vector to an `input` dimension.
#'   For example, `scatter_dims_to_operand_dims = c(1L)` means each
#'   index vector indexes into the first dimension of `input`.
#' @param index_vector_dim (`integer(1)`)\cr
#'   Dimension of `scatter_indices` that contains the index vectors.
#'   If set to `ndims(scatter_indices) + 1`, each scalar element of
#'   `scatter_indices` is treated as a length-1 index vector.
#' @param indices_are_sorted (`logical(1)`)\cr
#'   Whether indices are guaranteed to be sorted. Setting to `TRUE`
#'   may improve performance but produces undefined behavior if the
#'   indices are not actually sorted. Default `FALSE`.
#' @param unique_indices (`logical(1)`)\cr
#'   Whether indices are guaranteed to be unique (no duplicates).
#'   Setting to `TRUE` may improve performance but produces undefined
#'   behavior if the indices are not actually unique. Default `FALSE`.
#' @param update_computation (`function`)\cr
#'   Binary function `f(old, new)` that combines the existing value in
#'   `input` with the value from `update`. The default (`NULL`) uses
#'   `function(old, new) new`, which replaces the old value.
#' @return [`arrayish`]\cr
#'   Has the same data type and shape as `input`.
#'   It is ambiguous if `input` is ambiguous.
#' @section Out Of Bounds Behavior:
#' If a computed result index falls outside the bounds of `input`, the
#' update for that index is silently ignored.
#' @section Update Order:
#' When multiple indices in `scatter_indices` map to the same element
#' of `input`, the order in which `update_computation` is applied is
#' implementation-defined and may vary between plugins ("cpu", "cuda").
#' @templateVar primitive_id scatter
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_scatter()].
#' @seealso [prim_gather()], [nv_subset()], [nv_subset_assign()], `[`, `[<-`
#' @examplesIf pjrt::plugins_downloaded()
#' # Scatter values 10 and 30 into positions 1 and 3 of a zero vector
#' input <- nv_array(c(0, 0, 0, 0, 0))
#' indices <- nv_matrix(c(1L, 3L), ncol = 1)
#' updates <- nv_array(c(10, 30))
#' prim_scatter(
#'   input, indices, updates,
#'   update_window_dims = integer(0),
#'   inserted_window_dims = 1L,
#'   input_batching_dims = integer(0),
#'   scatter_indices_batching_dims = integer(0),
#'   scatter_dims_to_operand_dims = 1L,
#'   index_vector_dim = 2L
#' )
#' @export
prim_scatter <- new_primitive(
  "scatter",
  function(
    input,
    scatter_indices,
    update,
    update_window_dims,
    inserted_window_dims,
    input_batching_dims,
    scatter_indices_batching_dims,
    scatter_dims_to_operand_dims,
    index_vector_dim,
    indices_are_sorted = FALSE,
    unique_indices = FALSE,
    update_computation = NULL
  ) {
    # otherwise, delayed promise evaluation means they might be added to the update_descriptor
    force(input)
    force(scatter_indices)
    force(update)
    if (is.null(update_computation)) {
      update_computation <- function(old, new) new
    } else if (!is.function(update_computation)) {
      cli_abort("update_computation must be a function")
    }

    current_desc <- .current_descriptor(silent = TRUE)

    # Trace the update computation function
    # For scatter, the update computation takes 2 scalar arguments (current, update)
    desc_update <- local_descriptor()

    # Create dummy arguments for tracing - use the input's dtype
    input_dtype <- dtype_abstract(input)
    update_dtype <- dtype_abstract(update)
    if (input_dtype != update_dtype) {
      cli_abort("input and update must have the same dtype")
    }

    dummy_args <- list(
      AbstractArray(dtype = input_dtype, shape = Shape(integer()), ambiguous = ambiguous_abstract(input)),
      AbstractArray(dtype = input_dtype, shape = Shape(integer()), ambiguous = ambiguous_abstract(update))
    )

    update_computation_graph <- trace_fn(update_computation, dummy_args, desc = desc_update, mode = "subgraph")

    # Register constants from the update computation graph
    register_consts(current_desc, update_computation_graph$constants)

    infer_fn <- function(
      input,
      scatter_indices,
      update,
      update_window_dims,
      inserted_window_dims,
      input_batching_dims,
      scatter_indices_batching_dims,
      scatter_dims_to_operand_dims,
      index_vector_dim,
      indices_are_sorted,
      unique_indices,
      update_computation_graph
    ) {
      # Convert 1-based dimension numbers to 0-based
      scatter_dimension_numbers <- stablehlo::ScatterDimensionNumbers(
        update_window_dims = update_window_dims - 1L,
        inserted_window_dims = inserted_window_dims - 1L,
        input_batching_dims = input_batching_dims - 1L,
        scatter_indices_batching_dims = scatter_indices_batching_dims - 1L,
        scatter_dims_to_operand_dims = scatter_dims_to_operand_dims - 1L,
        index_vector_dim = index_vector_dim - 1L
      )

      indices_sorted_attr <- r_to_constant(indices_are_sorted, dtype = "bool", shape = integer())
      unique_indices_attr <- r_to_constant(unique_indices, dtype = "bool", shape = integer())

      out <- stablehlo::infer_types_scatter(
        inputs = list(at2vt(input)),
        scatter_indices = at2vt(scatter_indices),
        updates = list(at2vt(update)),
        scatter_dimension_numbers = scatter_dimension_numbers,
        indices_are_sorted = indices_sorted_attr,
        unique_indices = unique_indices_attr,
        update_computation = stablehlo(update_computation_graph, id = "", constants_as_inputs = FALSE)[[1L]]
      )[[1L]]

      out <- vt2at(out)
      out$ambiguous <- input$ambiguous
      list(out)
    }

    out <- graph_desc_add(
      self,
      args = list(input = input, scatter_indices = scatter_indices, update = update),
      params = list(
        update_window_dims = update_window_dims,
        inserted_window_dims = inserted_window_dims,
        input_batching_dims = input_batching_dims,
        scatter_indices_batching_dims = scatter_indices_batching_dims,
        scatter_dims_to_operand_dims = scatter_dims_to_operand_dims,
        index_vector_dim = index_vector_dim,
        indices_are_sorted = indices_are_sorted,
        unique_indices = unique_indices,
        update_computation_graph = update_computation_graph
      ),
      infer_fn = infer_fn,
      desc = current_desc
    )

    out[[1L]]
  },
  subgraphs = "update_computation_graph",
  static = 4:12
)

#' @title Primitive Gather
#' @description
#' Gathers slices from the `operand` array at positions specified by
#' `start_indices`. Each index vector in `start_indices` identifies a
#' starting position in `operand`, and a slice of size `slice_sizes` is
#' extracted from that position. The gathered slices are assembled into
#' the output array.
#'
#' This is the inverse of [prim_scatter()]: gather reads slices from a
#' array at given indices, while scatter writes slices into an array at
#' given indices.
#' @template param_prim_operand_any
#' @param start_indices ([`arrayish`] of integer type)\cr
#'   Array of starting indices. Contains index vectors that map to
#'   positions in `operand` via `start_index_map`. The dimension
#'   specified by `index_vector_dim` holds the index vectors.
#' @param slice_sizes (`integer()`)\cr
#'   Size of the slice to gather from `operand` in each dimension.
#'   Must have length equal to `ndims(operand)`.
#' @param offset_dims (`integer()`)\cr
#'   Dimensions in the output that correspond to the non-collapsed
#'   slice dimensions of `operand`.
#' @param collapsed_slice_dims (`integer()`)\cr
#'   Dimensions of `operand` that are collapsed (removed) from the
#'   slice. The corresponding entries in `slice_sizes` must be `1`.
#'   Together with `offset_dims` and `operand_batching_dims`, these
#'   must account for all dimensions of `operand`.
#' @param operand_batching_dims (`integer()`)\cr
#'   Dimensions of `operand` that are batch dimensions.
#'   Use `integer(0)` when there are no batch dimensions.
#' @param start_indices_batching_dims (`integer()`)\cr
#'   Dimensions of `start_indices` that correspond to batch
#'   dimensions. Must have the same length as `operand_batching_dims`.
#' @param start_index_map (`integer()`)\cr
#'   Maps each component of the index vector to an `operand`
#'   dimension. For example, `start_index_map = c(1L)` means each
#'   index vector indexes into the first dimension of `operand`.
#' @param index_vector_dim (`integer(1)`)\cr
#'   Dimension of `start_indices` that contains the index vectors.
#'   If set to `ndims(start_indices) + 1`, each scalar element of
#'   `start_indices` is treated as a length-1 index vector.
#' @param indices_are_sorted (`logical(1)`)\cr
#'   Whether indices are guaranteed to be sorted. Setting to `TRUE`
#'   may improve performance but produces undefined behavior if the
#'   indices are not actually sorted. Default `FALSE`.
#' @param unique_indices (`logical(1)`)\cr
#'   Whether indices are guaranteed to be unique (no duplicates).
#'   Setting to `TRUE` may improve performance but produces undefined
#'   behavior if the indices are not actually unique. Default `FALSE`.
#' @return [`arrayish`]\cr
#'   Has the same data type as `operand`. The output shape is composed
#'   of the offset dimensions (from the slice) and the remaining
#'   dimensions from `start_indices`. See the underluing stableHLO function
#'   for more details.
#' @section Out Of Bounds Behavior:
#' Start indices are clamped before the slice is extracted:
#' `clamp(1, start_index, nv_shape(operand) - slice_sizes + 1)`.
#' This means that out-of-bounds indices will not cause an error, but
#' the effective start position may differ from the requested one.
#' @templateVar primitive_id gather
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_gather()].
#' @seealso [prim_scatter()], [nv_subset()], [nv_subset_assign()], `[`, `[<-`
#' @examplesIf pjrt::plugins_downloaded()
#' # Gather rows 1 and 3 from a 3x3 matrix
#' operand <- nv_matrix(1:9, nrow = 3)
#' indices <- nv_matrix(c(1L, 3L), ncol = 1)
#' prim_gather(
#'   operand, indices,
#'   slice_sizes = c(1L, 3L),
#'   offset_dims = 2L,
#'   collapsed_slice_dims = 1L,
#'   operand_batching_dims = integer(0),
#'   start_indices_batching_dims = integer(0),
#'   start_index_map = 1L,
#'   index_vector_dim = 2L
#' )
#' @export
prim_gather <- new_primitive(
  "gather",
  function(
    operand,
    start_indices,
    slice_sizes,
    offset_dims,
    collapsed_slice_dims,
    operand_batching_dims,
    start_indices_batching_dims,
    start_index_map,
    index_vector_dim,
    indices_are_sorted = FALSE,
    unique_indices = FALSE
  ) {
    infer_fn <- function(
      operand,
      start_indices,
      slice_sizes,
      offset_dims,
      collapsed_slice_dims,
      operand_batching_dims,
      start_indices_batching_dims,
      start_index_map,
      index_vector_dim,
      indices_are_sorted,
      unique_indices
    ) {
      gather_dimension_numbers <- stablehlo::GatherDimensionNumbers(
        offset_dims = offset_dims - 1L,
        collapsed_slice_dims = collapsed_slice_dims - 1L,
        operand_batching_dims = operand_batching_dims - 1L,
        start_indices_batching_dims = start_indices_batching_dims - 1L,
        start_index_map = start_index_map - 1L,
        index_vector_dim = index_vector_dim - 1L
      )

      slice_sizes_attr <- r_to_constant(slice_sizes, dtype = "i64", shape = length(slice_sizes))
      indices_sorted_attr <- r_to_constant(indices_are_sorted, dtype = "bool", shape = integer())

      out <- stablehlo::infer_types_gather(
        at2vt(operand),
        at2vt(start_indices),
        gather_dimension_numbers = gather_dimension_numbers,
        slice_sizes = slice_sizes_attr,
        indices_are_sorted = indices_sorted_attr
      )[[1L]]

      out <- vt2at(out)
      out$ambiguous <- operand$ambiguous
      list(out)
    }
    graph_desc_add(
      self,
      args = list(operand = operand, start_indices = start_indices),
      params = list(
        slice_sizes = slice_sizes,
        offset_dims = offset_dims,
        collapsed_slice_dims = collapsed_slice_dims,
        operand_batching_dims = operand_batching_dims,
        start_indices_batching_dims = start_indices_batching_dims,
        start_index_map = start_index_map,
        index_vector_dim = index_vector_dim,
        indices_are_sorted = indices_are_sorted,
        unique_indices = unique_indices
      ),
      infer_fn = infer_fn
    )[[1L]]
  },
  static = 3:11
)

#' @title Primitive Cholesky Decomposition
#' @description
#' Computes the Cholesky decomposition of a symmetric positive-definite matrix.
#' Dimensions before the last two are batch dimensions.
#' @param operand ([`arrayish`])\cr
#'   Arrayish value of data type floating-point with at least 2 dimensions.
#'   The last two dimensions must be equal (square matrix); any leading
#'   dimensions are batch dimensions.
#' @param lower (`logical(1)`)\cr
#'   If `FALSE` (default, matching base R's [base::chol()]), compute the
#'   upper triangular factor `U` such that `operand = t(U) %*% U`. If
#'   `TRUE`, compute the lower triangular factor `L` such that
#'   `operand = L %*% t(L)`.
#' @return [`arrayish`]\cr
#'   Has the same shape and data type as the input.
#'   The values in the triangle not specified by `lower` are implementation-defined.
#'   It is ambiguous if the input is ambiguous.
#' @templateVar primitive_id cholesky
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_cholesky()].
#' @seealso [nv_solve()]
#' @examplesIf pjrt::plugins_downloaded()
#' # Create a positive-definite matrix
#' x <- nv_matrix(c(4, 2, 2, 3), nrow = 2, dtype = "f32")
#' prim_chol(x, lower = TRUE)
#' @export
prim_chol <- new_primitive(
  "cholesky",
  function(operand, lower = FALSE) {
    infer_fn <- function(operand, lower) {
      # Output has same shape and dtype as input (square matrix)
      list(AbstractArray(
        dtype = dtype(operand),
        shape = Shape(shape(operand)),
        ambiguous = operand$ambiguous
      ))
    }
    graph_desc_add(
      self,
      list(operand = operand),
      list(lower = lower),
      infer_fn = infer_fn
    )[[1L]]
  },
  static = 2L
)

#' @title Primitive Triangular Solve
#' @description
#' Solves a system of linear equations with a triangular coefficient matrix.
#' When `left_side` is `TRUE`, solves `op(a) %*% x = b` for `x`.
#' When `left_side` is `FALSE`, solves `x %*% op(a) = b` for `x`.
#' Dimensions before the last two are batch dimensions and must match
#' between `a` and `b` (no broadcasting).
#' Here `op` is `A` or `A^T` depending on `transpose_a`.
#' @param a ([`arrayish`])\cr
#'   Triangular coefficient matrix of data type floating-point with at least 2
#'   dimensions. The last two dimensions must be equal (square matrix); any
#'   leading dimensions are batch dimensions.
#' @param b ([`arrayish`])\cr
#'   Right-hand side. Same data type and rank as `a` (rank >= 2), with
#'   matching leading batch dimensions. The size of `a`'s last two (square)
#'   dimensions must equal `b`'s second-to-last dimension when
#'   `left_side = TRUE`, or `b`'s last dimension when `left_side = FALSE`.
#' @param left_side (`logical(1)`)\cr
#'   If `TRUE`, solve `op(a) %*% x = b`. If `FALSE`, solve `x %*% op(a) = b`.
#' @param lower (`logical(1)`)\cr
#'   If `TRUE`, `a` is lower triangular. If `FALSE`, `a` is upper triangular.
#' @param unit_diagonal (`logical(1)`)\cr
#'   If `TRUE`, assume diagonal elements of `a` are 1.
#' @param transpose_a (`logical(1)`)\cr
#'   If `TRUE`, solve with `t(a)` in place of `a`. Defaults to `FALSE`.
#' @return [`arrayish`]\cr
#'   Has the same shape and data type as `b`.
#'   It is ambiguous if both `a` and `b` are ambiguous.
#' @templateVar primitive_id triangular_solve
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_triangular_solve()].
#' @seealso [nv_solve()]
#' @examplesIf pjrt::plugins_downloaded()
#' # Solve L %*% x = b where L is lower triangular
#' L <- nv_matrix(c(2, 0, 1, 3), nrow = 2, dtype = "f32")
#' b <- nv_matrix(c(4, 3), nrow = 2, dtype = "f32")
#' prim_triangular_solve(L, b,
#'   left_side = TRUE, lower = TRUE,
#'   unit_diagonal = FALSE, transpose_a = FALSE
#' )
#' @export
prim_triangular_solve <- new_primitive(
  "triangular_solve",
  function(a, b, left_side, lower, unit_diagonal, transpose_a) {
    infer_fn <- function(a, b, left_side, lower, unit_diagonal, transpose_a) {
      left_side_attr <- r_to_constant(as.logical(left_side), dtype = "bool", shape = integer())
      lower_attr <- r_to_constant(as.logical(lower), dtype = "bool", shape = integer())
      unit_diagonal_attr <- r_to_constant(as.logical(unit_diagonal), dtype = "bool", shape = integer())
      out <- stablehlo::infer_types_triangular_solve(
        at2vt(a),
        at2vt(b),
        left_side = left_side_attr,
        lower = lower_attr,
        unit_diagonal = unit_diagonal_attr,
        transpose_a = if (transpose_a) "TRANSPOSE" else "NO_TRANSPOSE"
      )[[1L]]
      out <- vt2at(out)
      out$ambiguous <- a$ambiguous && b$ambiguous
      list(out)
    }
    graph_desc_add(
      self,
      list(a = a, b = b),
      list(
        left_side = left_side,
        lower = lower,
        unit_diagonal = unit_diagonal,
        transpose_a = transpose_a
      ),
      infer_fn = infer_fn
    )[[1L]]
  },
  static = 3:6
)

#' @title Primitive QR Decomposition
#' @description
#' Computes the reduced QR decomposition of a matrix `operand`:
#' \deqn{A = Q R,}
#' where \eqn{Q} has orthonormal columns (\eqn{Q^\top Q = I}) and
#' \eqn{R} is upper triangular.
#' For an \eqn{m \times n} input with \eqn{k = \min(m, n)}, \eqn{Q} has
#' shape \eqn{m \times k} and \eqn{R} has shape \eqn{k \times n}.
#' @param operand ([`arrayish`])\cr
#'   Matrix of data type floating-point with exactly 2 dimensions.
#' @return Named `list` with elements `Q` (shape `(m, k)`) and `R`
#'   (shape `(k, n)`), where `(m, n) = shape(operand)` and
#'   `k = min(m, n)`. Both have the same data type as `operand`.
#' @templateVar primitive_id qr
#' @template section_rules
#' @section StableHLO:
#' Lowers to a `"geqrf"` + `"orgqr"` [hlo_custom_call()] pair
#' (backed by LAPACK on CPU and cuSOLVER on CUDA) + postprocessing.
#' @seealso [nv_qr()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(1:6, shape = c(3, 2), dtype = "f32")
#' prim_qr(x)
#' @export
prim_qr <- new_primitive(
  "qr",
  function(operand) {
    infer_fn <- function(operand) {
      assert_linalg_matrix(operand, "operand")
      dt <- dtype(operand)
      s <- shape(operand)
      m <- s[1L]
      n <- s[2L]
      k <- min(m, n)
      list(
        Q = AbstractArray(dtype = dt, shape = Shape(c(m, k))),
        R = AbstractArray(dtype = dt, shape = Shape(c(k, n)))
      )
    }
    graph_desc_add(
      self,
      list(operand = operand),
      params = list(),
      infer_fn = infer_fn
    )
  }
)

#' @title Primitive LU Decomposition
#' @description
#' Computes the partial-pivoted LU decomposition of a matrix `operand`:
#' \deqn{P A = L U,}
#' where \eqn{P} is a permutation matrix, \eqn{L} is unit lower triangular,
#' and \eqn{U} is upper triangular. `L` (with implicit unit diagonal) and
#' `U` are packed into a single `LU` output matching LAPACK's `getrf`
#' layout. \eqn{P} is returned in two equivalent forms: `pivots` (LAPACK's
#' sequential row-swap encoding) and `permutation` (an explicit
#' permutation vector).
#'
#' @param operand ([`arrayish`])\cr
#'   Matrix of data type floating-point with exactly 2 dimensions.
#' @return `list` of three [`arrayish`] values: `LU` `(m, n)` with the same
#'   dtype as the input; `pivots` `(k,)` of dtype `i32` with
#'   `k = min(m, n)` (1-based row swaps such that row `i` was exchanged
#'   with row `pivots[i]` during elimination step `i`); and `permutation`
#'   `(m,)` of dtype `i32`, a 1-based permutation vector for \eqn{P} such
#'   that `(P %*% A)[i, ]` equals `A[permutation[i], ]`.
#' @templateVar primitive_id lu
#' @template section_rules
#' @section StableHLO:
#' Lowers to a `"lu"` [hlo_custom_call()] (backed by LAPACK on
#' CPU and cuSOLVER on CUDA) for `LU` and `pivots`, followed by a
#' [hlo_while()] loop that converts `pivots` to `permutation`
#' in-graph.
#' @seealso [nv_lu()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
#' prim_lu(x)
#' @export
prim_lu <- new_primitive(
  "lu",
  function(operand) {
    infer_fn <- function(operand) {
      assert_linalg_matrix(operand, "operand")
      dt <- dtype(operand)
      s <- shape(operand)
      m <- s[1L]
      n <- s[2L]
      k <- min(m, n)
      list(
        LU = AbstractArray(dtype = dt, shape = Shape(c(m, n))),
        pivots = AbstractArray(dtype = "i32", shape = Shape(k)),
        permutation = AbstractArray(dtype = "i32", shape = Shape(m))
      )
    }
    graph_desc_add(
      self,
      list(operand = operand),
      params = list(),
      infer_fn = infer_fn
    )
  }
)

#' @title Primitive Singular Value Decomposition
#' @description
#' Computes the reduced ("economy") singular value decomposition of a
#' matrix `operand` of shape `(m, n)`:
#' \deqn{A = u \, \mathrm{diag}(d) \, vt,}
#' where `u` has orthonormal columns, `vt` has orthonormal rows, and `d`
#' is the length-`k` (`k = min(m, n)`) vector of non-negative singular
#' values in descending order.
#'
#' Note: unlike `base::svd()`, which returns the right singular vectors
#' as `v` of shape `(n, k)` (so that `a = u %*% diag(d) %*% t(v)`), this
#' primitive returns them already transposed as `vt` of shape `(k, n)`
#' (matching the underlying LAPACK / cuSOLVER output and avoiding an
#' extra transpose).
#'
#' Supports any matrix shape on both the host (LAPACK `gesdd`) and CUDA
#' (cuSOLVER `gesvd`) backends. cuSOLVER's `m >= n` requirement is handled
#' transparently via a layout flip for wide matrices.
#' @param operand ([`arrayish`])\cr
#'   Matrix of data type floating-point with exactly 2 dimensions.
#' @return Named `list` with elements `d` (length `k`), `u` (shape
#'   `(m, k)`), and `vt` (shape `(k, n)`). All have the same dtype as
#'   the input.
#' @templateVar primitive_id svd
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_custom_call()] with target `"svd"`.
#' @seealso [nv_svd()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 0, 0, 1, 0, 1), shape = c(3, 2))
#' prim_svd(x)
#' @export
prim_svd <- new_primitive(
  "svd",
  function(operand) {
    infer_fn <- function(operand) {
      assert_linalg_matrix(operand, "operand")
      dt <- dtype(operand)
      s <- shape(operand)
      m <- s[1L]
      n <- s[2L]
      k <- min(m, n)
      list(
        d = AbstractArray(dtype = dt, shape = Shape(k)),
        u = AbstractArray(dtype = dt, shape = Shape(c(m, k))),
        vt = AbstractArray(dtype = dt, shape = Shape(c(k, n)))
      )
    }
    graph_desc_add(
      self,
      list(operand = operand),
      params = list(),
      infer_fn = infer_fn
    )
  }
)

#' @title Primitive Symmetric Eigendecomposition
#' @description
#' Computes the eigendecomposition of a symmetric matrix `operand` of
#' shape `(n, n)`:
#' \deqn{A = \mathrm{vectors} \, \mathrm{diag}(\mathrm{values}) \,
#'   \mathrm{vectors}^\top.}
#' Only the lower triangle of `operand` is read. The columns of `vectors`
#' are the (orthonormal) eigenvectors and `values` is the length-`n`
#' vector of (real) eigenvalues in ascending order. Output names and
#' order match [base::eigen()].
#' @param operand ([`arrayish`])\cr
#'   Symmetric square matrix of floating-point data type.
#' @return Named `list` with elements `values` (length `n`) and `vectors`
#'   (shape `(n, n)`). Both have the same dtype as the input.
#' @templateVar primitive_id eigh
#' @template section_rules
#' @section StableHLO:
#' Lowers to [hlo_custom_call()] with target `"eigh"`.
#' @seealso [nv_eigh()]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(2, 1, 1, 2), shape = c(2, 2), dtype = "f64")
#' prim_eigh(x)
#' @export
prim_eigh <- new_primitive(
  "eigh",
  function(operand) {
    infer_fn <- function(operand) {
      assert_linalg_matrix(operand, "operand", square = TRUE)
      dt <- dtype(operand)
      s <- shape(operand)
      n <- s[1L]
      # Names + order mirror `base::eigen()`: list(values, vectors).
      list(
        values = AbstractArray(dtype = dt, shape = Shape(n)),
        vectors = AbstractArray(dtype = dt, shape = Shape(c(n, n)))
      )
    }
    graph_desc_add(
      self,
      list(operand = operand),
      params = list(),
      infer_fn = infer_fn
    )
  }
)
