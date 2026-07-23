SubsetFull <- function(size) {
  structure(list(size = size), class = "SubsetFull")
}

SubsetRange <- function(start, end) {
  structure(list(start = start, size = end - start + 1L), class = "SubsetRange")
}

SubsetIndex <- function(index) {
  static <- is.numeric(index)
  if (static) {
    if (length(index) != 1L) cli_abort("Internal error")
  } else {
    if (naxes_abstract(index) != 0L) cli_abort("Internal error")
  }
  structure(list(index = index, size = 1L, static = static), class = "SubsetIndex")
}

SubsetIndices <- function(indices) {
  static <- is.numeric(indices)
  if (static) {
    size <- length(indices)
  } else {
    nd <- naxes_abstract(indices)
    if (nd != 1L) {
      cli_abort("Internal error")
    }
    size <- shape_abstract(indices)[1L]
  }
  structure(list(indices = indices, size = size, static = static), class = "SubsetIndices")
}

is_subset_full <- function(x) inherits(x, "SubsetFull")
is_subset_range <- function(x) inherits(x, "SubsetRange")
is_subset_index <- function(x) inherits(x, "SubsetIndex")
is_subset_indices <- function(x) inherits(x, "SubsetIndices")

subset_spec_to_shape <- function(specs) {
  shp <- integer()
  for (spec in specs) {
    if (is_subset_index(spec)) {
      next
    }
    shp <- c(shp, spec$size)
  }
  return(shp)
}


subset_start_positions <- function(subsets) {
  subset_start_position <- function(s) {
    if (is_subset_index(s)) {
      s$index
    } else if (inherits(s, "SubsetIndices")) {
      s$indices
    } else if (is_subset_full(s)) {
      1L
    } else if (is_subset_range(s)) {
      s$start
    } else {
      cli_abort("Internal error")
    }
  }

  lapply(subsets, subset_start_position)
}

# Takes a list of 1-D start arrays (one per axis). Scalar axes have length 1,
# multi-index axes have length > 1.
# Returns an array where each row is one index tuple into the original array,
# covering all combinations of the multi-index axes (cartesian product).
# Shape [rank] if all scalar, or [multi_index_sizes..., rank] otherwise.
dynamic_start_indices <- function(starts) {
  rank <- length(starts)
  sizes <- vapply(starts, function(s) shape_abstract(s)[1L], integer(1L))
  multi_index_axes <- which(sizes > 1L)

  if (length(multi_index_axes) == 0L) {
    start <- do.call(nv_concatenate, c(starts, list(axis = 1L)))
    return(start)
  }

  # Each "row" (last axis) is an index tuple [d1, d2, ..., d_rank].
  # Scalar axes contribute the same value to every row.
  # Multi-index axes vary across their own axis, forming the cartesian product.

  multi_index_sizes <- sizes[multi_index_axes]
  n_gather <- length(multi_index_axes)

  slices <- vector("list", rank)
  multi_index_i <- 1L
  for (d in seq_len(rank)) {
    if (identical(shape_abstract(starts[[d]]), 1L)) {
      slices[[d]] <- nv_broadcast_to(starts[[d]], c(multi_index_sizes, 1L))
    } else {
      slices[[d]] <- prim_broadcast_in_axes(starts[[d]], c(multi_index_sizes, 1L), multi_index_i)
      multi_index_i <- multi_index_i + 1L
    }
  }
  out <- do.call(nv_concatenate, c(slices, list(axis = n_gather + 1L)))
  out
}

static_start_indices <- function(starts, like = NULL) {
  sizes <- lengths(starts)
  multi_index_axes <- which(sizes > 1L)
  if (length(multi_index_axes) == 0L) {
    data <- unlist(starts)
    if (is.null(like)) {
      return(nv_array(data, dtype = "i32"))
    }
    return(nv_array_like(like, data, dtype = "i32", shape = length(data), ambiguous = FALSE))
  }
  grid <- as.matrix(do.call(expand.grid, starts))
  out <- array(grid, dim = c(sizes[multi_index_axes], length(starts)))
  if (is.null(like)) {
    return(nv_array(out, dtype = "i32"))
  }
  nv_array_like(like, out, dtype = "i32", shape = dim(out), ambiguous = FALSE)
}


#' Build an array of start indices (aka scatter_indices) from subset specs
#'
#' For each subset spec, extracts the start index and combines them into an array.
#' The dtype is determined automatically: i32 for static R ints, or the maximum
#' integer type present among dynamic array indices. Conversion is only performed
#' when at least one subset is dynamic.
#'
#' - Without multi_index_axes: returns a 1D array of shape `(rank)` (all starts are scalar).
#' - With multi_index_axes: returns an array of shape `(gather_shape..., rank)` where the
#'   gather axes' indices are broadcast across the cartesian product.
#'
#' @param subsets List of SubsetSpec objects (from parse_subset_specs)
#' @return An array of start indices
#' @noRd
subset_specs_start_indices <- function(subsets, like = NULL) {
  starts <- subset_start_positions(subsets)
  all_static <- all(vapply(starts, is.numeric, logical(1L)))
  if (all_static) {
    static_start_indices(starts, like = like)
  } else {
    # Convert R integers to 1D arrays, reshape 0D arrays to 1D
    starts <- lapply(starts, function(s) {
      if (is.numeric(s)) {
        if (is.null(like)) {
          nv_array(s, dtype = "i32")
        } else {
          nv_array_like(like, s, dtype = "i32", shape = length(s), ambiguous = FALSE)
        }
      } else if (naxes_abstract(s) == 0L) {
        nv_reshape(s, 1L)
      } else {
        s
      }
    })
    dynamic_start_indices(starts)
  }
}

#' Convert subset specs to gather parameters
#'
#' @param subsets List of SubsetSpec objects (from parse_subset_specs)
#' @return A list with all parameters needed for prim_gather:
#'   - start_indices: array of start indices (shape `(gather_shape..., rank)` or `(1, rank)`)
#'   - slice_sizes: integer vector
#'   - offset_axes: integer vector
#'   - collapsed_slice_axes: integer vector
#'   - start_index_map: integer vector
#'   - index_vector_axis: integer
#'   - indices_are_sorted: logical
#'   - unique_indices: logical
#'   - multi_index_subset: logical
#' @noRd
subset_specs_to_gather <- function(subsets, like = NULL) {
  rank <- length(subsets)

  # Identify gather axes (SubsetIndices with multiple elements)
  multi_index_axes <- which(vapply(
    subsets,
    function(s) {
      is_subset_indices(s) && s$size > 1L
    },
    logical(1L)
  ))

  multi_index_subset <- length(multi_index_axes) > 0L

  # slice_sizes: 1 for multi_index_axes, the size for others
  slice_sizes <- vapply(
    seq_len(rank),
    function(i) {
      if (i %in% multi_index_axes) 1L else subsets[[i]]$size
    },
    integer(1L)
  )

  collapsed_slice_axes <- sort(c(
    multi_index_axes,
    which(vapply(
      seq_len(rank),
      function(i) {
        !(i %in% multi_index_axes) && is_subset_index(subsets[[i]])
      },
      logical(1L)
    ))
  ))

  start_indices <- subset_specs_start_indices(subsets, like = like)

  # offset_axes: positions in the output for non-collapsed axes of `x`.
  # The output interleaves batch (gather) axes and offset (slice) axes
  # in the order of the original axes of `x`.
  subset_index_axes <- which(vapply(subsets, is_subset_index, logical(1L)))
  surviving_axes <- setdiff(seq_len(rank), subset_index_axes)
  multi_among_surviving <- which(surviving_axes %in% multi_index_axes)
  offset_axes <- setdiff(seq_along(surviving_axes), multi_among_surviving)

  index_vector_axis <- length(multi_index_axes) + 1L

  list(
    start_indices = start_indices,
    slice_sizes = slice_sizes,
    offset_axes = offset_axes,
    collapsed_slice_axes = collapsed_slice_axes,
    start_index_map = seq_len(rank),
    index_vector_axis = index_vector_axis,
    indices_are_sorted = !multi_index_subset,
    # TODO: Could improve this
    unique_indices = !multi_index_subset,
    multi_index_subset = multi_index_subset
  )
}

#' Convert subset specs to scatter parameters
#'
#' @param subsets List of SubsetSpec objects (from parse_subset_specs)
#' @return A list with all parameters needed for prim_scatter:
#'   - scatter_indices: array of scatter indices
#'   - update_window_axes: integer vector
#'   - inserted_window_axes: integer vector
#'   - scatter_axes_to_x_axes: integer vector
#'   - index_vector_axis: integer
#'   - indices_are_sorted: logical
#'   - unique_indices: logical
#'   - update_shape: integer vector (expected shape of the update array)
#' @noRd
subset_specs_to_scatter <- function(subsets, like = NULL) {
  rank <- length(subsets)

  multi_index_axes <- which(vapply(
    subsets,
    function(s) {
      is_subset_indices(s) && s$size > 1L
    },
    logical(1L)
  ))

  multi_index_subset <- length(multi_index_axes) > 0L

  # slice_sizes: 1 for gather axes (individually addressed), normal for others
  slice_sizes <- vapply(
    seq_len(rank),
    function(i) {
      if (i %in% multi_index_axes) 1L else subsets[[i]]$size
    },
    integer(1L)
  )

  scatter_indices <- subset_specs_start_indices(subsets, like = like)

  # SubsetIndex axes are individually addressed (dropped from update),
  # just like collapsed_slice_axes in the gather path.
  index_axes <- which(vapply(subsets, is_subset_index, logical(1L)))
  inserted_window_axes <- sort(c(multi_index_axes, index_axes))
  surviving_axes <- setdiff(seq_len(rank), index_axes)

  if (multi_index_subset) {
    # scatter_indices shape: [gather_shape..., rank]
    n_gather <- length(multi_index_axes)
    multi_among_surviving <- which(surviving_axes %in% multi_index_axes)
    update_window_axes <- setdiff(seq_along(surviving_axes), multi_among_surviving)
    update_shape <- vapply(
      surviving_axes,
      function(i) {
        if (i %in% multi_index_axes) subsets[[i]]$size else slice_sizes[i]
      },
      integer(1L)
    )
    index_vector_axis <- n_gather + 1L
  } else {
    # scatter_indices shape: [rank] (no batch axes)
    update_window_axes <- seq_along(surviving_axes)
    update_shape <- slice_sizes[surviving_axes]
    index_vector_axis <- 1L
  }

  list(
    scatter_indices = scatter_indices,
    update_window_axes = update_window_axes,
    inserted_window_axes = inserted_window_axes,
    scatter_axes_to_x_axes = seq_len(rank),
    index_vector_axis = index_vector_axis,
    # TODO: Could improve this
    indices_are_sorted = !multi_index_subset,
    unique_indices = !multi_index_subset,
    update_shape = update_shape,
    multi_index_subset = multi_index_subset
  )
}

# Helper functions for subset operations ======================================

#' Parse subset specifications and fill unspecified axes
#' @param quos List of quosures (from enquos)
#' @param x_shape Shape of the input array
#' @return List of SubsetSpec objects
#' @noRd
parse_subset_specs <- function(quos, x_shape) {
  rank <- length(x_shape)

  if (length(quos) > rank) {
    cli_abort("Too many subset specifications: got {length(quos)}, expected at most {rank}")
  }

  subsets <- lapply(seq_along(quos), function(i) {
    parse_subset_spec(quos[[i]], x_shape[i])
  })

  # Trailing subsets don't need to be specified, so we fill them with full selections
  if (length(subsets) < rank) {
    for (i in seq(length(subsets) + 1L, rank)) {
      subsets[[i]] <- SubsetFull(x_shape[i])
    }
  }

  subsets
}

#' Parse a single subset specification
#' @param quo Quosure to parse
#' @param axis_size Size of the axis being indexed
#' @return A SubsetSpec object (SubsetFull, SubsetRange, or SubsetIndices)
#' @noRd
parse_subset_spec <- function(quo, axis_size) {
  is_integerish <- function(x) {
    is.null(dim(x)) && test_integerish(x, len = 1L, any.missing = FALSE)
  }

  # Missing argument - select all
  if (rlang::quo_is_missing(quo)) {
    return(SubsetFull(axis_size))
  }

  e <- rlang::quo_get_expr(quo)

  # Check for range expression (a:b) before evaluating
  if (rlang::is_call(e, ":")) {
    env <- rlang::quo_get_env(quo)
    start <- rlang::eval_tidy(e[[2]], env = env)
    end <- rlang::eval_tidy(e[[3]], env = env)

    if (!is_integerish(start) || !is_integerish(end)) {
      cli_abort("Range indices must be scalar integers")
    }

    start <- as.integer(start)
    end <- as.integer(end)

    if (start < 1L || end > axis_size) {
      cli_abort("Range {start}:{end} is out of bounds for axis of size {axis_size}")
    }

    return(SubsetRange(start, end))
  }

  # Evaluate the quosure
  e <- rlang::eval_tidy(quo)

  # Single integer - drops axis. `array(i)` (length-1, with axis attr)
  # falls through to the array branch below so the axis is kept.
  if (is_integerish(e)) {
    idx <- as.integer(e)
    if (idx < 1L || idx > axis_size) {
      cli_abort("Index {idx} is out of bounds for axis of size {axis_size}")
    }
    return(SubsetIndex(idx))
  }

  # R vectors of length > 1 without a dim attribute - not allowed (ambiguous shape)
  if (is.numeric(e) && length(e) > 1L && is.null(dim(e))) {
    cli_abort(c(
      "Vectors of length > 1 are not allowed as subset indices.",
      "i" = "Use {.code array()} to select multiple elements, e.g. {.code x[array(c(1L, 3L)), ]}."
    ))
  }

  # Atomic numeric array - static indices (preserves axis)
  if (is.array(e) && is.numeric(e)) {
    if (length(dim(e)) != 1L) {
      cli_abort("Array indices must be 1D, but got {length(dim(e))}D")
    }
    indices <- as.integer(e)
    oob <- indices < 1L | indices > axis_size
    if (any(oob)) {
      bad <- indices[oob][1L] # nolint
      cli_abort("Index {bad} is out of bounds for axis of size {axis_size}")
    }
    return(SubsetIndices(indices))
  }

  # AnvlRange (dynamic range) - not supported
  if (inherits(e, "IotaArray")) {
    if (length(shape) != 1L) {
      cli_abort("IotaArray must be 1D, but got {length(shape)}D")
    }
    return(SubsetRange(e$start, e$end))
  }

  # Array indices (AnvlArray or GraphBox)
  if (is_arrayish(e) && !is.atomic(e)) {
    dt <- dtype_abstract(e)
    if (!(inherits(dt, "IntegerType") || inherits(dt, "UIntegerType"))) {
      cli_abort("Dynamic indices must be integers, but got {.cls {class(dt)[1]}}")
    }
    nd <- naxes_abstract(e)
    if (nd > 1L) {
      cli_abort("Dynamic indices must be at most 1D, but got {nd}D array")
    }
    # Scalar array drops axis, 1D array preserves
    if (nd == 0L) {
      return(SubsetIndex(e))
    }
    return(SubsetIndices(e))
  }

  cli_abort("Invalid subset expression")
}

#' @title Subset an Array
#' @description
#' Extracts a subset from an array. You can also use the `[` operator.
#' Supports R-style indexing including scalar indices (which drop axes),
#' ranges (`a:b`), and `array(c(...))` for selecting multiple elements along a
#' axis.
#' @template param_x
#' @param ... Subset specifications, one per axis. Omitted trailing
#'   axes select all elements. See `vignette("subsetting")` for details.
#' @return [`arrayish`]
#' @seealso [nv_subset_assign()] for updating subsets, `vignette("subsetting")`
#'   for a comprehensive guide.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:12, nrow = 3)
#' x
#' # Select row 2
#' x[2, ]
#'
#' # Select rows 1 to 2, all columns
#' x[1:2, ]
#' @export
nv_subset <- function(x, ...) {
  if (!is_arrayish(x)) {
    cli_abort(c(
      "Argument {.arg x} must be arrayish",
      "x" = "Got {.cls {class(x)[1]}}"
    ))
  }
  x_shape <- shape_abstract(x)
  quos <- rlang::enquos(...)

  subsets <- parse_subset_specs(quos, x_shape)
  params <- subset_specs_to_gather(subsets, like = x)

  out <- prim_gather(
    x = x,
    start_indices = params$start_indices,
    slice_sizes = params$slice_sizes,
    offset_axes = params$offset_axes,
    collapsed_slice_axes = params$collapsed_slice_axes,
    x_batching_axes = integer(),
    start_indices_batching_axes = integer(),
    start_index_map = params$start_index_map,
    index_vector_axis = params$index_vector_axis,
    indices_are_sorted = params$indices_are_sorted,
    unique_indices = params$unique_indices
  )

  out
}

# Jitted core of `nv_subset_assign()`. The entry point parses the NSE `...`
# subscripts (which jit cannot trace) and materialises the scatter indices; the
# array work -- dtype promotion, value broadcast, and the scatter itself -- is
# fused into a single program here. `scatter_indices` is passed in as a traced
# input (never reconstructed from the static params), so distinct index *values*
# reuse one compiled program; only distinct subset *patterns* recompile.
subset_scatter_core <- jit(
  function(
    x,
    value,
    scatter_indices,
    update_window_axes,
    inserted_window_axes,
    scatter_axes_to_x_axes,
    index_vector_axis,
    indices_are_sorted,
    unique_indices,
    update_shape
  ) {
    if (dtype(x) != dtype(value)) {
      dt_x <- dtype(x)
      dt_value <- dtype(value)
      if (!promotable_to(dt_value, dt_x)) {
        cli_abort(
          "Value type {dtype2string(dt_value)} is not promotable to left-hand side type {dtype2string(dt_x)}"
        )
      }
      value <- nv_convert(value, dtype = dt_x)
    }

    if (!naxes(value)) {
      value <- nv_broadcast_to(value, update_shape)
    } else {
      value_shape <- shape(value)
      if (!identical(value_shape, update_shape)) {
        cli_abort(c(
          "Update shape does not match subset shape.",
          x = "Got {shape2string(value_shape)} and {shape2string(update_shape)}"
        ))
      }
    }

    prim_scatter(
      x = x,
      scatter_indices = scatter_indices,
      update = value,
      update_window_axes = update_window_axes,
      inserted_window_axes = inserted_window_axes,
      x_batching_axes = integer(),
      scatter_indices_batching_axes = integer(),
      scatter_axes_to_x_axes = scatter_axes_to_x_axes,
      index_vector_axis = index_vector_axis,
      indices_are_sorted = indices_are_sorted,
      unique_indices = unique_indices
    )
  },
  backend = "auto",
  static = c(
    "update_window_axes",
    "inserted_window_axes",
    "scatter_axes_to_x_axes",
    "index_vector_axis",
    "indices_are_sorted",
    "unique_indices",
    "update_shape"
  )
)

#' @title Update Subset
#' @description
#' Updates elements of an array at specified positions, returning a new array.
#' You can also use the `[<-` operator.
#' @template param_x
#' @param ... Subset specifications, one per axis. See
#'   `vignette("subsetting")` for details.
#' @param value ([`arrayish`])\cr
#'   Replacement values. Scalars are broadcast to the subset shape.
#'   Non-scalar values must match the subset shape.
#' @return [`arrayish`]\cr
#'   A new array with the same shape as `x` and the subset replaced.
#' @seealso [nv_subset()], `vignette("subsetting")` for a comprehensive guide.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:12, nrow = 3)
#' # Set row 1 to zeros
#' x[1, ] <- nv_scalar(0L)
#' x
#' @export
# Not `@jit`-tagged: the `...` subscripts are captured via NSE (`enquos()`),
# which jit's argument handling cannot trace. The parsing stays here (eager) and
# the array work is delegated to the jitted [subset_scatter_core()].
nv_subset_assign <- function(x, ..., value) {
  if (!is_arrayish(x)) {
    cli_abort("Expected arrayish `x`, but got {.cls {class(x)[1]}}")
  }
  if (!is_arrayish(value)) {
    cli_abort("Expected arrayish `value`, but got {.cls {class(value)[1]}}")
  }
  aligned <- as_anvl_arrays(x, value)
  x <- aligned[[1L]]
  value <- aligned[[2L]]

  lhs_shape <- shape_abstract(x)
  # because we do NSE to determine `:`-calls
  quos <- rlang::enquos(...)

  subsets <- parse_subset_specs(quos, lhs_shape)
  params <- subset_specs_to_scatter(subsets, like = x)

  subset_scatter_core(
    x = x,
    value = value,
    scatter_indices = params$scatter_indices,
    update_window_axes = params$update_window_axes,
    inserted_window_axes = params$inserted_window_axes,
    scatter_axes_to_x_axes = params$scatter_axes_to_x_axes,
    index_vector_axis = params$index_vector_axis,
    indices_are_sorted = params$indices_are_sorted,
    unique_indices = params$unique_indices,
    update_shape = params$update_shape
  )
}
