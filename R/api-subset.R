#' @include jit.R
NULL

SubsetFull <- function(size) {
  structure(list(size = size), class = "SubsetFull")
}

SubsetRange <- function(start, end, reversed = FALSE) {
  structure(
    list(start = start, size = end - start + 1L, reversed = reversed),
    class = "SubsetRange"
  )
}

SubsetIndex <- function(index) {
  static <- is.numeric(index)
  if (static) {
    if (length(index) != 1L) cli_abort("Internal error")
  } else {
    if (naxes(index) != 0L) cli_abort("Internal error")
  }
  structure(list(index = index, size = 1L, static = static), class = "SubsetIndex")
}

SubsetIndices <- function(indices) {
  static <- is.numeric(indices)
  if (static) {
    size <- length(indices)
  } else {
    nd <- naxes(indices)
    if (nd != 1L) {
      cli_abort("Internal error")
    }
    size <- shape(indices)[1L]
  }
  structure(list(indices = indices, size = size, static = static), class = "SubsetIndices")
}

is_subset_full <- function(x) inherits(x, "SubsetFull")
is_subset_range <- function(x) inherits(x, "SubsetRange")
is_subset_index <- function(x) inherits(x, "SubsetIndex")
is_subset_indices <- function(x) inherits(x, "SubsetIndices")

# Output axes carrying a decreasing range. A range always survives into the
# output, so its output axis is its position among the axes that no scalar
# index drops.
subset_reversed_axes <- function(subsets) {
  surviving <- !vapply(subsets, is_subset_index, logical(1L))
  reversed <- vapply(subsets, function(s) isTRUE(s$reversed), logical(1L))
  which(reversed[surviving])
}

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
# multi-index axes have any other length (including 0, for an empty selection).
# Returns an array where each row is one index tuple into the original array,
# covering all combinations of the multi-index axes (cartesian product).
# Shape [rank] if all scalar, or [multi_index_sizes..., rank] otherwise.
dynamic_start_indices <- function(starts) {
  rank <- length(starts)
  sizes <- vapply(starts, function(s) shape(s)[1L], integer(1L))
  multi_index_axes <- which(sizes != 1L)

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
    if (identical(shape(starts[[d]]), 1L)) {
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
  multi_index_axes <- which(sizes != 1L)
  if (length(multi_index_axes) == 0L) {
    data <- unlist(starts)
    return(new_index_array(data, length(data), like = like))
  }
  grid <- as.matrix(do.call(expand.grid, starts))
  new_index_array(grid, c(sizes[multi_index_axes], length(starts)), like = like)
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
#' @return ([`arrayish`])\cr
#'   An array of start indices
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
          nv_array_like(like, s, dtype = "i32", shape = length(s))
        }
      } else if (naxes(s) == 0L) {
        nv_reshape(s, 1L)
      } else {
        s
      }
    })
    dynamic_start_indices(starts)
  }
}

#' Build an index array for gather/scatter from static R integers
#' @noRd
new_index_array <- function(data, shape, like = NULL) {
  data <- array(as.integer(data), dim = shape)
  if (is.null(like)) {
    return(nv_array(data, dtype = "i32"))
  }
  nv_array_like(like, data, dtype = "i32", shape = shape)
}

#' Resolve a whole-array mask subscript
#'
#' A single subscript whose shape equals the shape of `x` selects elements
#' across the entire array, flattening the result. Deciding this requires
#' evaluating the subscript, so the quosures are returned alongside the mask
#' with the evaluated value spliced back in -- otherwise `parse_subset_specs()`
#' would evaluate the subscript a second time.
#'
#' Rank-1 arrays are left alone: there a whole-array mask and a mask on the
#' single axis mean the same thing, so the regular path already covers it.
#'
#' @param quos List of quosures (from `enquos()`)
#' @param x_shape Shape of the array being subset
#' @return A list with `mask` (an R logical array, or `NULL` if this is not a
#'   whole-array mask) and `quos`.
#' @noRd
resolve_flat_mask <- function(quos, x_shape) {
  if (length(quos) != 1L || length(x_shape) < 2L) {
    return(list(mask = NULL, quos = quos))
  }
  quo <- quos[[1L]]
  if (rlang::quo_is_missing(quo) || rlang::is_call(rlang::quo_get_expr(quo), ":")) {
    return(list(mask = NULL, quos = quos))
  }

  e <- rlang::eval_tidy(quo)
  quos[[1L]] <- rlang::new_quosure(e)

  if (!is_mask_subscript(e)) {
    return(list(mask = NULL, quos = quos))
  }
  mask <- as_r_mask(e)
  if (!identical(as.integer(dim(mask)), as.integer(x_shape))) {
    # splice the host-side mask back in so it is not read from the device twice
    quos[[1L]] <- rlang::new_quosure(mask)
    return(list(mask = NULL, quos = quos))
  }
  list(mask = mask, quos = quos)
}

#' Convert a whole-array mask to gather parameters
#'
#' Every selected element is addressed individually, so all axes of `x` are
#' collapsed and the result is 1-D. `which(arr.ind = TRUE)` enumerates the
#' index tuples in column-major order, which is the order R uses for `x[mask]`.
#' @noRd
flat_mask_to_gather <- function(mask, like = NULL) {
  rank <- length(dim(mask))
  indices <- which(mask, arr.ind = TRUE, useNames = FALSE)

  list(
    start_indices = new_index_array(indices, c(nrow(indices), rank), like = like),
    slice_sizes = rep(1L, rank),
    offset_axes = integer(),
    collapsed_slice_axes = seq_len(rank),
    start_index_map = seq_len(rank),
    index_vector_axis = 2L,
    # the index tuples are in column-major order, which is not the lexicographic
    # order that `indices_are_sorted` refers to
    indices_are_sorted = FALSE,
    unique_indices = TRUE
  )
}

#' Convert a whole-array mask to scatter parameters
#' @noRd
flat_mask_to_scatter <- function(mask, like = NULL) {
  rank <- length(dim(mask))
  indices <- which(mask, arr.ind = TRUE, useNames = FALSE)

  list(
    scatter_indices = new_index_array(indices, c(nrow(indices), rank), like = like),
    update_window_axes = integer(),
    inserted_window_axes = seq_len(rank),
    scatter_axes_to_x_axes = seq_len(rank),
    index_vector_axis = 2L,
    indices_are_sorted = FALSE,
    unique_indices = TRUE,
    update_shape = nrow(indices)
  )
}

#' Convert subset specs to gather parameters
#'
#' @param subsets List of SubsetSpec objects (from parse_subset_specs)
#' @return (`list`)\cr
#'   All parameters needed for `prim_gather()`:
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
      is_subset_indices(s) && s$size != 1L
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
#' @return (`list`)\cr
#'   All parameters needed for `prim_scatter()`:
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
      is_subset_indices(s) && s$size != 1L
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

abort_too_many_subsets <- function(n, x_shape, call = rlang::caller_env()) {
  rank <- length(x_shape)
  cli_abort(
    c(
      "Too many subset specifications.",
      x = "Got {n} for an array of shape {shape_repr(x_shape)}, which has
           {rank} {cli::qty(rank)}ax{?is/es}.",
      i = "Trailing axes can be left out; they select all elements."
    ),
    call = call
  )
}

#' Parse subset specifications and fill unspecified axes
#' @param quos List of quosures (from enquos)
#' @param x_shape Shape of the input array
#' @return (`list` of `SubsetSpec`)
#' @noRd
parse_subset_specs <- function(quos, x_shape) {
  rank <- length(x_shape)

  if (length(quos) > rank) {
    abort_too_many_subsets(length(quos), x_shape)
  }

  subsets <- lapply(seq_along(quos), function(i) {
    parse_subset_spec(quos[[i]], x_shape[i], axis = i)
  })

  # Trailing subsets don't need to be specified, so we fill them with full selections
  if (length(subsets) < rank) {
    for (i in seq(length(subsets) + 1L, rank)) {
      subsets[[i]] <- SubsetFull(x_shape[i])
    }
  }

  subsets
}

#' Is this subscript a boolean mask?
#'
#' A mask is either an R logical array or an arrayish value of dtype `bool`.
#' Bare R logical vectors are recognised here so that `as_r_mask()` can reject
#' them with a helpful message rather than "Invalid subset expression".
#' @param x Evaluated subscript
#' @noRd
is_mask_subscript <- function(x) {
  if (is.logical(x)) {
    return(TRUE)
  }
  is_arrayish(x, convert_ok = FALSE) && identical(as.character(peek_dtype(x)), "bool")
}

#' Convert a boolean mask subscript to a plain R logical array
#'
#' Masks are resolved to positions with `which()`, so their values must be known
#' on the host. For an R logical array that is free; for an arrayish mask it
#' requires reading the array back, which is only possible in eager mode.
#' @param e Evaluated subscript, as recognised by `is_mask_subscript()`
#' @return An R logical array
#' @noRd
as_r_mask <- function(e) {
  if (is.logical(e)) {
    if (is.null(dim(e))) {
      cli_abort(c(
        "Logical vectors are not allowed as subset indices.",
        "i" = "Use {.fn arr} to create a mask, e.g. {.code x[arr(TRUE, FALSE, TRUE), ]}."
      ))
    }
    mask <- e
  } else {
    if (currently_tracing()) {
      cli_abort(c(
        "Boolean masks from arrays are only supported in eager mode.",
        "x" = "The number of selected elements, and hence the output shape, depends on the data.",
        "i" = "Use an R logical mask (e.g. {.code arr(TRUE, FALSE, TRUE)}) for a mask that is known at compile time."
      ))
    }
    mask <- as_array(e)
  }
  if (anyNA(mask)) {
    cli_abort("Boolean masks must not contain missing values.")
  }
  mask
}

#' Parse a single subset specification
#' @param quo Quosure to parse
#' @param axis_size Size of the axis being indexed
#' @param axis Axis the subset applies to, used to name it in errors
#' @return (`SubsetSpec`)\cr
#'   One of `SubsetFull`, `SubsetRange` or `SubsetIndices`.
#' @noRd
parse_subset_spec <- function(quo, axis_size, axis) {
  in_bounds <- "Axis {axis} has size {axis_size}, so indices must be between 1 and {axis_size}."
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
    start <- rlang::eval_tidy(e[[2L]], env = env)
    end <- rlang::eval_tidy(e[[3L]], env = env)

    if (!is_integerish(start) || !is_integerish(end)) {
      bad <- if (!is_integerish(start)) start else end
      side <- if (!is_integerish(start)) "start" else "end"
      cli_abort(c(
        "The {side} of a range subset must be a single whole number.",
        x = "For axis {axis}, got {.obj_type_friendly {bad}}."
      ))
    }

    start <- as.integer(start)
    end <- as.integer(end)

    # Either end may be the larger one, since a range is allowed to count down.
    if (min(start, end) < 1L || max(start, end) > axis_size) {
      cli_abort(c(
        "The range subset {start}:{end} is out of bounds for axis {axis}.",
        x = in_bounds
      ))
    }

    # A range that counts down selects in reverse, as in base R. There is no
    # descending slice, so it becomes the ascending one plus a reverse of the
    # output axis
    if (end < start) {
      return(SubsetRange(end, start, reversed = TRUE))
    }

    return(SubsetRange(start, end))
  }

  # Evaluate the quosure
  e <- rlang::eval_tidy(quo)

  # Boolean mask - selects the TRUE positions, never drops the axis
  if (is_mask_subscript(e)) {
    mask <- as_r_mask(e)
    if (length(dim(mask)) != 1L) {
      cli_abort(c(
        "A mask for a single axis must have exactly one axis.",
        x = "Got {length(dim(mask))} axes.",
        "i" = "A mask over the whole array must be the only subscript and have the same shape as {.arg x}."
      ))
    }
    if (length(mask) != axis_size) {
      cli_abort(
        "Mask of length {length(mask)} does not match an axis of size {axis_size}."
      )
    }
    return(SubsetIndices(which(mask)))
  }

  # Single integer - drops axis. `array(i)` (length-1, with axis attr)
  # falls through to the array branch below so the axis is kept.
  if (is_integerish(e)) {
    idx <- as.integer(e)
    if (idx < 1L || idx > axis_size) {
      cli_abort(c(
        "The index {idx} is out of bounds for axis {axis}.",
        x = in_bounds
      ))
    }
    return(SubsetIndex(idx))
  }

  # R vectors of length > 1 without a dim attribute - not allowed (ambiguous shape)
  if (is.numeric(e) && length(e) > 1L && is.null(dim(e))) {
    cli_abort(c(
      "Vectors of length > 1 are not allowed as subset indices.",
      x = "Got one of length {length(e)} for axis {axis}.",
      i = "Use {.code array()} to select multiple elements, e.g. {.code x[array(c(1L, 3L)), ]}."
    ))
  }

  # Atomic numeric array - static indices (preserves axis)
  if (is.array(e) && is.numeric(e)) {
    if (length(dim(e)) != 1L) {
      cli_abort(c(
        "An array of indices must have exactly one axis.",
        x = "The array given for axis {axis} has {length(dim(e))} axes."
      ))
    }
    indices <- as.integer(e)
    oob <- indices < 1L | indices > axis_size
    if (any(oob)) {
      bad <- unique(indices[oob])
      # Each `{?}` gets its own quantity: left to infer one from `{.val {bad}}`,
      # cli reads the index *values* as the count.
      nbad <- length(bad)
      cli_abort(c(
        "{cli::qty(nbad)}The ind{?ex/ices} {.val {bad}} {cli::qty(nbad)}{?is/are}
         out of bounds for axis {axis}.",
        x = in_bounds
      ))
    }
    return(SubsetIndices(indices))
  }

  # A dynamic range is not supported. This used to build a SubsetRange from
  # `e$end`, which an IotaArray does not have, so the call failed further down
  # with a length-0 slice size instead of saying what was wrong.
  if (inherits(e, "IotaArray")) {
    cli_abort(c(
      "A dynamic range is not supported as a subset index.",
      x = "Got {.cls IotaArray} for axis {axis}.",
      i = "Use a literal range such as {.code 2:4}, or an array of indices."
    ))
  }

  # Array indices (AnvlArray or GraphBox)
  if (is_arrayish(e) && !is.atomic(e)) {
    dt <- peek_dtype(e)
    if (!(is_dtype_int(dt) || is_dtype_uint(dt))) {
      cli_abort(c(
        "An array of indices must have an integer data type.",
        x = "The array given for axis {axis} has data type {.val {as.character(dt)}}.",
        i = "Convert it with {.fn nv_convert}."
      ))
    }
    nd <- naxes(e)
    if (nd > 1L) {
      cli_abort(c(
        "An array of indices must have at most one axis.",
        x = "The array given for axis {axis} has {nd} axes."
      ))
    }
    # Scalar array drops axis, 1D array preserves
    if (nd == 0L) {
      return(SubsetIndex(e))
    }
    return(SubsetIndices(e))
  }

  detail <- if (is.numeric(e) && length(e) == 1L && is.finite(e)) {
    "For axis {axis}, got {.val {e}}, which is not whole."
  } else {
    "For axis {axis}, got {.obj_type_friendly {e}}."
  }
  cli_abort(c(
    "Each subset must be missing, a whole number, a range, or an array of an integer data type.",
    x = detail,
    i = "See {.code vignette(\"subsetting\")}."
  ))
}

#' @title Subset an Array
#' @description
#' Extracts a subset from an array. You can also use the `[` operator.
#' Supports R-style indexing including scalar indices (which drop axes),
#' ranges (`a:b`), `array(c(...))` for selecting multiple elements along an
#' axis, and boolean masks.
#' @templateVar dtypes any data type
#' @template param_unary_x
#' @param ... Subset specifications, one per axis. Omitted trailing
#'   axes select all elements.
#'
#'   A boolean mask (an R logical array such as `arr(TRUE, FALSE)`, or an
#'   arrayish value of dtype `bool`) selects the elements at the `TRUE`
#'   positions. A mask for one axis must have as many elements as the size of
#'   that axis. A mask that is the only subscript and has the same shape as
#'   `x` selects across the whole array, yielding a 1-D result. Masks whose
#'   values come from an array only work in eager mode, because the number of
#'   selected elements determines the output shape.
#'
#'   See `vignette("subsetting")` for details.
#' @return ([`arrayish`])\cr
#'   Has the input's data type, and the shape the specifications select --
#'   a scalar index drops its axis, a range, an index array or an axis mask
#'   keeps it, and a whole-array mask yields a 1-D result.
#' @seealso [nv_subset_assign()] for updating subsets, `vignette("subsetting")`
#'   for a comprehensive guide.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:12, nrow = 3)
#' x
#' # select row 2
#' nv_subset(x, 2)
#' x[2, ]
#'
#' # select rows 1 to 2, all columns
#' nv_subset(x, 1:2)
#' x[1:2, ]
#'
#' # Select rows 1 and 3 with a mask
#' x[arr(TRUE, FALSE, TRUE), ]
#'
#' # Select all elements greater than 6 (eager mode only)
#' x[x > 6]
#' @export
nv_subset <- function(x, ...) {
  if (!is_arrayish(x)) {
    cli_abort(c(
      "{.arg x} must be arrayish.",
      "x" = "Got {.cls {class(x)[1]}}."
    ))
  }
  x_shape <- shape(x)
  quos <- rlang::enquos(...)

  flat <- resolve_flat_mask(quos, x_shape)
  if (is.null(flat$mask)) {
    subsets <- parse_subset_specs(flat$quos, x_shape)
    params <- subset_specs_to_gather(subsets, like = x)
  } else {
    subsets <- list()
    params <- flat_mask_to_gather(flat$mask, like = x)
  }

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

  reversed_axes <- subset_reversed_axes(subsets)
  if (length(reversed_axes)) {
    out <- prim_rev(out, axes = reversed_axes)
  }

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
    # `x` and `value` arrive at one data type: nv_subset_assign() brought them
    # there, which is also where a value `x`'s data type cannot hold is refused.
    if (!naxes(value)) {
      value <- nv_broadcast_to(value, update_shape)
    } else {
      value_shape <- shape(value)
      if (!identical(value_shape, update_shape)) {
        cli_abort(c(
          "Update shape does not match subset shape.",
          x = "Got {shape_repr(value_shape)} and {shape_repr(update_shape)}"
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
#' @param x ([`arrayish`])\cr
#'   The array to update. Can be any data type.
#'   An R object is materialized at its [default data type][default_dtypes].
#' @inheritParams nv_subset
#' @param value ([`arrayish`])\cr
#'   Replacement values. Scalars are broadcast to the subset shape and non-scalar
#'   values must match it.
#'   The value is converted to the data type of `x`.
#' @return ([`arrayish`])\cr
#'   Has `x`'s data type and shape, with the subset replaced.
#' @seealso [nv_subset()], `vignette("subsetting")` for a comprehensive guide.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:12, nrow = 3)
#' # set row 1 to zeros
#' nv_subset_assign(x, 1, value = nv_scalar(0L))
#' x[1, ] <- nv_scalar(0L)
#' x
#'
#' # Zero out every element greater than 6 (eager mode only)
#' x[x > 6] <- 0L
#' x
#' @export
# Not wrapped in `jit()`: the `...` subscripts are captured via NSE (`enquos()`),
# which jit's argument handling cannot trace. The parsing stays here (eager) and
# the array work is delegated to the jitted [subset_scatter_core()].
nv_subset_assign <- function(x, ..., value) {
  if (!is_arrayish(x)) {
    cli_abort("Expected arrayish `x`, but got {.cls {class(x)[1]}}")
  }
  if (!is_arrayish(value)) {
    cli_abort("Expected arrayish `value`, but got {.cls {class(value)[1]}}")
  }
  # `value` is assigned at the data type of the array it goes into, so an R
  # value is *built* there rather than converted to it -- and one that data type
  # cannot hold (`x_i32[i] <- 1.5`) is an error, not a silent truncation.
  args <- as_anvl_arrays(x = x, value = value, .promote = promotion_like("x"))
  x <- args$x
  value <- args$value

  lhs_shape <- shape(x)
  # because we do NSE to determine `:`-calls
  quos <- rlang::enquos(...)

  flat <- resolve_flat_mask(quos, lhs_shape)
  if (is.null(flat$mask)) {
    subsets <- parse_subset_specs(flat$quos, lhs_shape)
    params <- subset_specs_to_scatter(subsets, like = x)
  } else {
    subsets <- list()
    params <- flat_mask_to_scatter(flat$mask, like = x)
  }

  # The scatter writes the ascending slice, so a decreasing range means the
  # value goes in back to front. A scalar value broadcasts either way.
  reversed_axes <- subset_reversed_axes(subsets)
  if (length(reversed_axes) && naxes(value)) {
    value <- prim_rev(value, axes = reversed_axes)
  }

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
