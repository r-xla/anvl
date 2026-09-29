dtype_from_buffer <- function(x) {
  d <- as.character(dtype(x))
  as_dtype(d)
}

hashvalues <- function(h) {
  val <- vector("list", numhash(h))
  idx <- 0L
  maphash(h, function(k, v) {
    idx <<- idx + 1L
    val[[idx]] <<- v
  })
  val
}

# these functions also work with primitives etc.
formalArgs2 <- function(f) {
  names(formals2(f))
}

formals2 <- function(f) {
  formals(args(f))
}


# We assume little endian
minmax_raw <- function(bits, signed = TRUE) {
  stopifnot(bits %% 8L == 0L, bits >= 8L)
  n <- bits %/% 8L
  if (!signed) {
    return(list(
      min = as.raw(rep(0x00, n)),
      max = as.raw(rep(0xFF, n))
    ))
  }
  hi_min <- as.raw(0x80) # 1000 0000
  hi_max <- as.raw(0x7F) # 0111 1111
  zeros <- as.raw(rep(0x00, n - 1L))
  ff <- as.raw(rep(0xFF, n - 1L))
  list(min = c(zeros, hi_min), max = c(ff, hi_max))
}


# The smallest / largest value of a data type as a scalar: `-Inf` / `Inf` for a
# float, `FALSE` / `TRUE` for `bool`, and the bounds of the integer range
# otherwise.
nv_minval <- function(dtype, device = NULL) {
  nv_extremal_value(dtype, "min", -Inf, FALSE, device)
}

nv_maxval <- function(dtype, device = NULL) {
  nv_extremal_value(dtype, "max", Inf, TRUE, device)
}

nv_extremal_value <- function(dtype, which, float_value, bool_value, device) {
  dtype <- as.character(dtype)
  if (grepl("^f", dtype)) {
    return(nv_scalar(float_value, dtype = dtype, device = device))
  }
  if (dtype == "bool") {
    return(nv_scalar(bool_value, dtype = "bool", device = device))
  }
  nv_scalar(pjrt_buffer(
    globals$ranges_raw[[dtype]][[which]],
    dtype = dtype,
    device = device,
    row_major = TRUE,
    shape = integer()
  ))
}

without <- function(x, indices) {
  if (length(indices)) {
    x[-indices]
  } else {
    x
  }
}

shape2string <- function(x, parenthesize = TRUE) {
  s <- paste0(x, collapse = ",")
  if (parenthesize) sprintf("(%s)", s) else s
}

# The shape spelling for user-facing messages: `(2x3)`, and `()` for a scalar.
# `shape2string()` above is the *repr* spelling -- it is what `f32[2,3]` and
# `RData(double, (2,3))` are built from and stays as it is -- so everything a
# caller reads in an error or warning goes through these two instead.
#
# A shape can also be one a caller typed (`shape = 1:1000`), so past
# `repr_max_entries` axes it is cut short, with the rank stated. That is enough
# for an array's real shape to print whole.
shape_repr <- function(shape) {
  n <- length(shape)
  if (n <= repr_max_entries) {
    return(sprintf("(%s)", paste0(shape, collapse = "x")))
  }
  sprintf("(%sx...) with %d axes", paste0(shape[seq_len(repr_max_entries)], collapse = "x"), n)
}

shapes_repr <- function(shapes) {
  paste0(vapply(shapes, shape_repr, character(1L)), collapse = ", ")
}

# `prim_fill()` takes a whole number at any data type -- `0L` builds at `bool`,
# at an integer one and at a float one alike -- so the fills that do not know
# their data type statically write `0L` / `1L`.
zeros <- function(dtype, shape) {
  prim_fill(0L, dtype = dtype, shape = shape)
}

ones <- function(dtype, shape) {
  prim_fill(1L, dtype = dtype, shape = shape)
}


zeros_like <- function(x) {
  zeros(dtype(x), shape(x))
}

ones_like <- function(x) {
  ones(dtype(x), shape(x))
}

is_valid_r_lit <- function(x) {
  length(x) == 1L &&
    is.null(dim(x)) &&
    (is.numeric(x) || is.logical(x)) &&
    # Accept NaN/Inf but reject NA (NA has no obvious dtype).
    (is.nan(x) || !is.na(x))
}

is_valid_r_array <- function(x) {
  is.array(x) && (is.numeric(x) || is.logical(x))
}

is_valid_r <- function(x) {
  (is.numeric(x) || is.logical(x)) && (is.array(x) || (length(x) == 1L))
}

# Clamp gather start indices to valid ranges, matching XLA's forward pass behavior.
# This ensures that out-of-bounds indices are clamped to [1, x_size - slice_size + 1]
# for each axis.
gather_clamp_indices <- function(
  start_indices,
  x_shape,
  slice_sizes,
  start_index_map,
  index_vector_axis
) {
  if (length(x_shape) != length(slice_sizes)) {
    cli_abort("{.arg x_shape} and {.arg slice_sizes} must have the same length")
  }

  n_index_coords <- length(start_index_map)
  if (n_index_coords == 0L) {
    return(start_indices)
  }

  # The largest valid start index of each coordinate. `slice_sizes` is in the
  # order of `x_shape`, so it is read through `start_index_map`.
  max_bounds <- pmax(1L, x_shape[start_index_map] - slice_sizes[start_index_map] + 1L)

  indices_shape <- shape(start_indices)
  index_dtype <- dtype(start_indices)
  if (index_vector_axis > length(indices_shape)) {
    # Implicit index vector (single coordinate)
    min_bound <- prim_fill(1L, dtype = index_dtype, shape = integer())
    max_bound <- prim_fill(max_bounds[1L], dtype = index_dtype, shape = integer())
    return(prim_clamp(start_indices, min_bound, max_bound))
  }

  # Explicit index vector axis - build bounds arrays
  bounds_shape <- rep(1L, length(indices_shape))
  bounds_shape[index_vector_axis] <- n_index_coords

  min_bound <- prim_broadcast_in_axes(
    prim_fill(1L, dtype = index_dtype, shape = integer()),
    indices_shape,
    integer()
  )

  # The max bound is the same for a given slice along the index_vector_axis
  max_bound_vals <- prim_reshape(
    prim_convert(
      nv_array(max_bounds, dtype = default_int()),
      dtype = index_dtype
    ),
    bounds_shape
  )
  max_bound <- nv_broadcast_to(max_bound_vals, indices_shape)

  prim_clamp(start_indices, min_bound, max_bound)
}

# Compute gather slice_sizes from scatter parameters.
# This inverts a scatter into a gather: for each axis of `x`, the slice
# size is 1 for inserted/batching axes, or the update's window size otherwise.
scatter_to_gather_slice_sizes <- function(
  update_shape,
  x_shape,
  update_window_axes,
  inserted_window_axes,
  x_batching_axes
) {
  slice_sizes <- integer(length(x_shape))
  update_window_pos <- 1L
  for (i in seq_along(x_shape)) {
    if (i %in% inserted_window_axes || i %in% x_batching_axes) {
      slice_sizes[i] <- 1L
    } else {
      slice_sizes[i] <- update_shape[update_window_axes[update_window_pos]]
      update_window_pos <- update_window_pos + 1L
    }
  }
  slice_sizes
}

col_major_layout <- function(naxes) {
  seq_len(naxes) - 1L
}

col_major_layouts <- function(...) {
  lapply(list(...), col_major_layout)
}

# Transpose the matrix an array's last two axes form, leaving any leading batch
# axes in place -- what `t()` means for the batched operands `nv_matmul()`
# takes. `nv_aperm()` reverses *every* axis, which would put a batch axis
# into the contraction slot. An array with fewer than two axes is handed on
# unchanged, for `nv_matmul()` to report.
transpose_matrix_axes <- function(x) {
  n <- naxes(x)
  if (n < 2L) {
    return(x)
  }
  nv_aperm(x, replace(seq_len(n), c(n - 1L, n), c(n, n - 1L)))
}

# Where `prim_bitcast_convert()` puts the axis holding an element's pieces when
# the two data types differ in width. StableHLO puts it last, where the pieces
# of one element sit next to each other under its row-major reading; anvl is
# column-major, so the axis belongs first instead. Returns the number of pieces
# and which way the conversion goes, so the shape rule and the lowering agree.
bitcast_lane <- function(dtype_in, dtype_out) {
  width_in <- dtype_width(as_dtype(dtype_in))
  width_out <- dtype_width(as_dtype(dtype_out))
  list(
    pieces = as.integer(max(width_in, width_out) / min(width_in, width_out)),
    splits = width_in > width_out,
    joins = width_in < width_out
  )
}
