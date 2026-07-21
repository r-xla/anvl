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
    if (ndims_abstract(index) != 0L) cli_abort("Internal error")
  }
  structure(list(index = index, size = 1L, static = static), class = "SubsetIndex")
}

SubsetIndices <- function(indices) {
  static <- is.numeric(indices)
  if (static) {
    size <- length(indices)
  } else {
    nd <- ndims_abstract(indices)
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

# Takes a list of 1-D start arrays (one per dim). Scalar dims have length 1,
# multi-index dims have any other length (including 0, for an empty selection).
# Returns an array where each row is one index tuple into the original array,
# covering all combinations of the multi-index dims (cartesian product).
# Shape [rank] if all scalar, or [multi_index_sizes..., rank] otherwise.
dynamic_start_indices <- function(starts) {
  rank <- length(starts)
  sizes <- vapply(starts, function(s) shape_abstract(s)[1L], integer(1L))
  multi_index_dims <- which(sizes != 1L)

  if (length(multi_index_dims) == 0L) {
    start <- do.call(nv_concatenate, c(starts, list(dimension = 1L)))
    return(start)
  }

  # Each "row" (last axis) is an index tuple [d1, d2, ..., d_rank].
  # Scalar dims contribute the same value to every row.
  # Multi-index dims vary across their own axis, forming the cartesian product.

  multi_index_sizes <- sizes[multi_index_dims]
  n_gather <- length(multi_index_dims)

  slices <- vector("list", rank)
  multi_index_i <- 1L
  for (d in seq_len(rank)) {
    if (identical(shape_abstract(starts[[d]]), 1L)) {
      slices[[d]] <- nv_broadcast_to(starts[[d]], c(multi_index_sizes, 1L))
    } else {
      slices[[d]] <- prim_broadcast_in_dim(starts[[d]], c(multi_index_sizes, 1L), multi_index_i)
      multi_index_i <- multi_index_i + 1L
    }
  }
  out <- do.call(nv_concatenate, c(slices, list(dimension = n_gather + 1L)))
  out
}

static_start_indices <- function(starts, like = NULL) {
  sizes <- lengths(starts)
  multi_index_dims <- which(sizes != 1L)
  if (length(multi_index_dims) == 0L) {
    data <- unlist(starts)
    return(new_index_array(data, length(data), like = like))
  }
  grid <- as.matrix(do.call(expand.grid, starts))
  new_index_array(grid, c(sizes[multi_index_dims], length(starts)), like = like)
}


#' Build an array of start indices (aka scatter_indices) from subset specs
#'
#' For each subset spec, extracts the start index and combines them into an array.
#' The dtype is determined automatically: i32 for static R ints, or the maximum
#' integer type present among dynamic array indices. Conversion is only performed
#' when at least one subset is dynamic.
#'
#' - Without multi_index_dims: returns a 1D array of shape `(rank)` (all starts are scalar).
#' - With multi_index_dims: returns an array of shape `(gather_shape..., rank)` where the
#'   gather dimensions' indices are broadcast across the cartesian product.
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
      } else if (ndims_abstract(s) == 0L) {
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
  nv_array_like(like, data, dtype = "i32", shape = shape, ambiguous = FALSE)
}

#' Resolve a whole-array mask subscript
#'
#' A single subscript whose shape equals the operand's shape selects elements
#' across the entire array, flattening the result. Deciding this requires
#' evaluating the subscript, so the quosures are returned alongside the mask
#' with the evaluated value spliced back in -- otherwise `parse_subset_specs()`
#' would evaluate the subscript a second time.
#'
#' Rank-1 operands are left alone: there a whole-array mask and a mask on the
#' single dimension mean the same thing, so the regular path already covers it.
#'
#' @param quos List of quosures (from `enquos()`)
#' @param operand_shape Shape of the operand array
#' @return A list with `mask` (an R logical array, or `NULL` if this is not a
#'   whole-array mask) and `quos`.
#' @noRd
resolve_flat_mask <- function(quos, operand_shape) {
  if (length(quos) != 1L || length(operand_shape) < 2L) {
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
  if (!identical(as.integer(dim(mask)), as.integer(operand_shape))) {
    # splice the host-side mask back in so it is not read from the device twice
    quos[[1L]] <- rlang::new_quosure(mask)
    return(list(mask = NULL, quos = quos))
  }
  list(mask = mask, quos = quos)
}

#' Convert a whole-array mask to gather parameters
#'
#' Every selected element is addressed individually, so all operand dimensions
#' are collapsed and the result is 1-D. `which(arr.ind = TRUE)` enumerates the
#' index tuples in column-major order, which is the order R uses for `x[mask]`.
#' @noRd
flat_mask_to_gather <- function(mask, like = NULL) {
  rank <- length(dim(mask))
  indices <- which(mask, arr.ind = TRUE, useNames = FALSE)

  list(
    start_indices = new_index_array(indices, c(nrow(indices), rank), like = like),
    slice_sizes = rep(1L, rank),
    offset_dims = integer(),
    collapsed_slice_dims = seq_len(rank),
    start_index_map = seq_len(rank),
    index_vector_dim = 2L,
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
    update_window_dims = integer(),
    inserted_window_dims = seq_len(rank),
    scatter_dims_to_operand_dims = seq_len(rank),
    index_vector_dim = 2L,
    indices_are_sorted = FALSE,
    unique_indices = TRUE,
    update_shape = nrow(indices)
  )
}

#' Convert subset specs to gather parameters
#'
#' @param subsets List of SubsetSpec objects (from parse_subset_specs)
#' @return A list with all parameters needed for prim_gather:
#'   - start_indices: array of start indices (shape `(gather_shape..., rank)` or `(1, rank)`)
#'   - slice_sizes: integer vector
#'   - offset_dims: integer vector
#'   - collapsed_slice_dims: integer vector
#'   - start_index_map: integer vector
#'   - index_vector_dim: integer
#'   - indices_are_sorted: logical
#'   - unique_indices: logical
#'   - multi_index_subset: logical
#' @noRd
subset_specs_to_gather <- function(subsets, like = NULL) {
  rank <- length(subsets)

  # Identify gather dimensions (SubsetIndices selecting other than one element)
  multi_index_dims <- which(vapply(
    subsets,
    function(s) {
      is_subset_indices(s) && s$size != 1L
    },
    logical(1L)
  ))

  multi_index_subset <- length(multi_index_dims) > 0L

  # slice_sizes: 1 for multi_index_dims, the size for others
  slice_sizes <- vapply(
    seq_len(rank),
    function(i) {
      if (i %in% multi_index_dims) 1L else subsets[[i]]$size
    },
    integer(1L)
  )

  collapsed_slice_dims <- sort(c(
    multi_index_dims,
    which(vapply(
      seq_len(rank),
      function(i) {
        !(i %in% multi_index_dims) && is_subset_index(subsets[[i]])
      },
      logical(1L)
    ))
  ))

  start_indices <- subset_specs_start_indices(subsets, like = like)

  # offset_dims: positions in the output for non-collapsed operand dims.
  # The output interleaves batch (gather) dims and offset (slice) dims
  # in the order of the original operand dimensions.
  subset_index_dims <- which(vapply(subsets, is_subset_index, logical(1L)))
  surviving_dims <- setdiff(seq_len(rank), subset_index_dims)
  multi_among_surviving <- which(surviving_dims %in% multi_index_dims)
  offset_dims <- setdiff(seq_along(surviving_dims), multi_among_surviving)

  index_vector_dim <- length(multi_index_dims) + 1L

  list(
    start_indices = start_indices,
    slice_sizes = slice_sizes,
    offset_dims = offset_dims,
    collapsed_slice_dims = collapsed_slice_dims,
    start_index_map = seq_len(rank),
    index_vector_dim = index_vector_dim,
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
#'   - update_window_dims: integer vector
#'   - inserted_window_dims: integer vector
#'   - scatter_dims_to_operand_dims: integer vector
#'   - index_vector_dim: integer
#'   - indices_are_sorted: logical
#'   - unique_indices: logical
#'   - update_shape: integer vector (expected shape of the update array)
#' @noRd
subset_specs_to_scatter <- function(subsets, like = NULL) {
  rank <- length(subsets)

  multi_index_dims <- which(vapply(
    subsets,
    function(s) {
      is_subset_indices(s) && s$size != 1L
    },
    logical(1L)
  ))

  multi_index_subset <- length(multi_index_dims) > 0L

  # slice_sizes: 1 for gather dims (individually addressed), normal for others
  slice_sizes <- vapply(
    seq_len(rank),
    function(i) {
      if (i %in% multi_index_dims) 1L else subsets[[i]]$size
    },
    integer(1L)
  )

  scatter_indices <- subset_specs_start_indices(subsets, like = like)

  # SubsetIndex dims are individually addressed (dropped from update),
  # just like collapsed_slice_dims in the gather path.
  index_dims <- which(vapply(subsets, is_subset_index, logical(1L)))
  inserted_window_dims <- sort(c(multi_index_dims, index_dims))
  surviving_dims <- setdiff(seq_len(rank), index_dims)

  if (multi_index_subset) {
    # scatter_indices shape: [gather_shape..., rank]
    n_gather <- length(multi_index_dims)
    multi_among_surviving <- which(surviving_dims %in% multi_index_dims)
    update_window_dims <- setdiff(seq_along(surviving_dims), multi_among_surviving)
    update_shape <- vapply(
      surviving_dims,
      function(i) {
        if (i %in% multi_index_dims) subsets[[i]]$size else slice_sizes[i]
      },
      integer(1L)
    )
    index_vector_dim <- n_gather + 1L
  } else {
    # scatter_indices shape: [rank] (no batch dims)
    update_window_dims <- seq_along(surviving_dims)
    update_shape <- slice_sizes[surviving_dims]
    index_vector_dim <- 1L
  }

  list(
    scatter_indices = scatter_indices,
    update_window_dims = update_window_dims,
    inserted_window_dims = inserted_window_dims,
    scatter_dims_to_operand_dims = seq_len(rank),
    index_vector_dim = index_vector_dim,
    # TODO: Could improve this
    indices_are_sorted = !multi_index_subset,
    unique_indices = !multi_index_subset,
    update_shape = update_shape,
    multi_index_subset = multi_index_subset
  )
}

# Helper functions for subset operations ======================================

#' Parse subset specifications and fill unspecified dimensions
#' @param quos List of quosures (from enquos)
#' @param operand_shape Shape of the operand array
#' @return List of SubsetSpec objects
#' @noRd
parse_subset_specs <- function(quos, operand_shape) {
  rank <- length(operand_shape)

  if (length(quos) > rank) {
    cli_abort("Too many subset specifications: got {length(quos)}, expected at most {rank}")
  }

  subsets <- lapply(seq_along(quos), function(i) {
    parse_subset_spec(quos[[i]], operand_shape[i])
  })

  # Trailing subsets don't need to be specified, so we fill them with full selections
  if (length(subsets) < rank) {
    for (i in seq(length(subsets) + 1L, rank)) {
      subsets[[i]] <- SubsetFull(operand_shape[i])
    }
  }

  subsets
}

#' Is this subscript a boolean mask?
#'
#' A mask is either an R logical array or an arrayish value of dtype `bool`.
#' Bare R logical vectors are recognised here so that [as_r_mask()] can reject
#' them with a helpful message rather than "Invalid subset expression".
#' @param x Evaluated subscript
#' @noRd
is_mask_subscript <- function(x) {
  if (is.logical(x)) {
    return(TRUE)
  }
  is_arrayish(x, convert_ok = FALSE) && identical(as.character(dtype_abstract(x)), "bool")
}

#' Convert a boolean mask subscript to a plain R logical array
#'
#' Masks are resolved to positions with `which()`, so their values must be known
#' on the host. For an R logical array that is free; for an arrayish mask it
#' requires reading the array back, which is only possible in eager mode.
#' @param e Evaluated subscript, as recognised by [is_mask_subscript()]
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
#' @param dim_size Size of the dimension being indexed
#' @return A SubsetSpec object (SubsetFull, SubsetRange, or SubsetIndices)
#' @noRd
parse_subset_spec <- function(quo, dim_size) {
  is_integerish <- function(x) {
    is.null(dim(x)) && test_integerish(x, len = 1L, any.missing = FALSE)
  }

  # Missing argument - select all
  if (rlang::quo_is_missing(quo)) {
    return(SubsetFull(dim_size))
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

    if (start < 1L || end > dim_size) {
      cli_abort("Range {start}:{end} is out of bounds for dimension of size {dim_size}")
    }

    return(SubsetRange(start, end))
  }

  # Evaluate the quosure
  e <- rlang::eval_tidy(quo)

  # Boolean mask - selects the TRUE positions, never drops the dimension
  if (is_mask_subscript(e)) {
    mask <- as_r_mask(e)
    if (length(dim(mask)) != 1L) {
      cli_abort(c(
        "Masks for a single dimension must be 1D, but got {length(dim(mask))}D",
        "i" = "A mask over the whole array must be the only subscript and have the same shape as the operand."
      ))
    }
    if (length(mask) != dim_size) {
      cli_abort(
        "Mask of length {length(mask)} does not match dimension of size {dim_size}"
      )
    }
    return(SubsetIndices(which(mask)))
  }

  # Single integer - drops dimension. `array(i)` (length-1, with dim attr)
  # falls through to the array branch below so the dim is kept.
  if (is_integerish(e)) {
    idx <- as.integer(e)
    if (idx < 1L || idx > dim_size) {
      cli_abort("Index {idx} is out of bounds for dimension of size {dim_size}")
    }
    return(SubsetIndex(idx))
  }

  # R vectors of length > 1 without dim - not allowed (ambiguous shape)
  if (is.numeric(e) && length(e) > 1L && is.null(dim(e))) {
    cli_abort(c(
      "Vectors of length > 1 are not allowed as subset indices.",
      "i" = "Use {.code array()} to select multiple elements, e.g. {.code x[array(c(1L, 3L)), ]}."
    ))
  }

  # Atomic numeric array - static indices (preserves dim)
  if (is.array(e) && is.numeric(e)) {
    if (length(dim(e)) != 1L) {
      cli_abort("Array indices must be 1D, but got {length(dim(e))}D")
    }
    indices <- as.integer(e)
    oob <- indices < 1L | indices > dim_size
    if (any(oob)) {
      bad <- indices[oob][1L] # nolint
      cli_abort("Index {bad} is out of bounds for dimension of size {dim_size}")
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
    nd <- ndims_abstract(e)
    if (nd > 1L) {
      cli_abort("Dynamic indices must be at most 1D, but got {nd}D array")
    }
    # Scalar array drops dimension, 1D array preserves
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
#' Supports R-style indexing including scalar indices (which drop dimensions),
#' ranges (`a:b`), `array(c(...))` for selecting multiple elements along a
#' dimension, and boolean masks.
#' @template param_operand
#' @param ... Subset specifications, one per dimension. Omitted trailing
#'   dimensions select all elements.
#'
#'   A boolean mask (an R logical array such as `arr(TRUE, FALSE)`, or an
#'   arrayish value of dtype `bool`) selects the elements at the `TRUE`
#'   positions. A mask for one dimension must have as many elements as that
#'   dimension. A mask that is the only subscript and has the same shape as
#'   `operand` selects across the whole array, yielding a 1-D result. Masks
#'   whose values come from an array only work in eager mode, because the number
#'   of selected elements determines the output shape.
#'
#'   See `vignette("subsetting")` for details.
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
#'
#' # Select rows 1 and 3 with a mask
#' x[arr(TRUE, FALSE, TRUE), ]
#'
#' # Select all elements greater than 6 (eager mode only)
#' x[x > 6]
#' @export
nv_subset <- function(operand, ...) {
  if (!is_arrayish(operand)) {
    cli_abort(c(
      "Argument operand must be arrayish",
      "x" = "Got {.cls {class(operand)[1]}}"
    ))
  }
  operand_shape <- shape_abstract(operand)
  quos <- rlang::enquos(...)

  flat <- resolve_flat_mask(quos, operand_shape)
  params <- if (is.null(flat$mask)) {
    subset_specs_to_gather(parse_subset_specs(flat$quos, operand_shape), like = operand)
  } else {
    flat_mask_to_gather(flat$mask, like = operand)
  }

  out <- prim_gather(
    operand = operand,
    start_indices = params$start_indices,
    slice_sizes = params$slice_sizes,
    offset_dims = params$offset_dims,
    collapsed_slice_dims = params$collapsed_slice_dims,
    operand_batching_dims = integer(),
    start_indices_batching_dims = integer(),
    start_index_map = params$start_index_map,
    index_vector_dim = params$index_vector_dim,
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
    operand,
    value,
    scatter_indices,
    update_window_dims,
    inserted_window_dims,
    scatter_dims_to_operand_dims,
    index_vector_dim,
    indices_are_sorted,
    unique_indices,
    update_shape
  ) {
    if (dtype(operand) != dtype(value)) {
      dt_operand <- dtype(operand)
      dt_value <- dtype(value)
      if (!promotable_to(dt_value, dt_operand)) {
        cli_abort(
          "Value type {dtype2string(dt_value)} is not promotable to left-hand side type {dtype2string(dt_operand)}"
        )
      }
      value <- nv_convert(value, dtype = dt_operand)
    }

    if (!ndims(value)) {
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
      input = operand,
      scatter_indices = scatter_indices,
      update = value,
      update_window_dims = update_window_dims,
      inserted_window_dims = inserted_window_dims,
      input_batching_dims = integer(),
      scatter_indices_batching_dims = integer(),
      scatter_dims_to_operand_dims = scatter_dims_to_operand_dims,
      index_vector_dim = index_vector_dim,
      indices_are_sorted = indices_are_sorted,
      unique_indices = unique_indices
    )
  },
  backend = "auto",
  static = c(
    "update_window_dims",
    "inserted_window_dims",
    "scatter_dims_to_operand_dims",
    "index_vector_dim",
    "indices_are_sorted",
    "unique_indices",
    "update_shape"
  )
)

#' @title Update Subset
#' @description
#' Updates elements of an array at specified positions, returning a new array.
#' You can also use the `[<-` operator.
#' @template param_operand
#' @inheritParams nv_subset
#' @param value ([`arrayish`])\cr
#'   Replacement values. Scalars are broadcast to the subset shape.
#'   Non-scalar values must match the subset shape.
#' @return [`arrayish`]\cr
#'   A new array with the same shape as `operand` and the subset replaced.
#' @seealso [nv_subset()], `vignette("subsetting")` for a comprehensive guide.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:12, nrow = 3)
#' # Set row 1 to zeros
#' x[1, ] <- nv_scalar(0L)
#' x
#'
#' # Zero out every element greater than 6 (eager mode only)
#' x[x > 6] <- 0L
#' x
#' @export
# Not `@jit`-tagged: the `...` subscripts are captured via NSE (`enquos()`),
# which jit's argument handling cannot trace. The parsing stays here (eager) and
# the array work is delegated to the jitted [subset_scatter_core()].
nv_subset_assign <- function(operand, ..., value) {
  if (!is_arrayish(operand)) {
    cli_abort("Expected arrayish `operand`, but got {.cls {class(operand)[1]}}")
  }
  if (!is_arrayish(value)) {
    cli_abort("Expected arrayish `value`, but got {.cls {class(value)[1]}}")
  }
  aligned <- as_anvl_arrays(operand, value)
  operand <- aligned[[1L]]
  value <- aligned[[2L]]

  lhs_shape <- shape_abstract(operand)
  # because we do NSE to determine `:`-calls
  quos <- rlang::enquos(...)

  flat <- resolve_flat_mask(quos, lhs_shape)
  params <- if (is.null(flat$mask)) {
    subset_specs_to_scatter(parse_subset_specs(flat$quos, lhs_shape), like = operand)
  } else {
    flat_mask_to_scatter(flat$mask, like = operand)
  }

  subset_scatter_core(
    operand = operand,
    value = value,
    scatter_indices = params$scatter_indices,
    update_window_dims = params$update_window_dims,
    inserted_window_dims = params$inserted_window_dims,
    scatter_dims_to_operand_dims = params$scatter_dims_to_operand_dims,
    index_vector_dim = params$index_vector_dim,
    indices_are_sorted = params$indices_are_sorted,
    unique_indices = params$unique_indices,
    update_shape = params$update_shape
  )
}
