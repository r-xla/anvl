dtype_from_buffer <- function(x) {
  d <- as.character(dtype(x))
  as_dtype(d)
}

#' @title Apply a `@jit` registry
#' @description
#' Iterates over a registry produced by [`jit_roclet()`] and rebinds each
#' listed function in `envir` to
#' `jit(f, backend = "auto", static = entry$static)`.
#'
#' Call this from the top level of your package's `R/zzz.R`, right next to
#' `.onLoad`, so the wrappers are byte-compiled during package install
#' instead of being rebuilt on every `.onLoad`:
#'
#' ```r
#' anvl::apply_jit_registry(.jit_registry)
#' ```
#'
#' `.jit_registry` is the variable defined by `R/jit-registry.R`, which is
#' regenerated on every `devtools::document()`.
#'
#' @param registry (`list`)\cr
#'   List of `list(name = <chr>, static = <chr|int>)` entries. Typically the
#'   `.jit_registry` object emitted by the roclet.
#' @param envir (`environment`)\cr
#'   Environment in which to look up and rebind functions. Defaults to
#'   `parent.frame()`, which at top-level package source time is the package
#'   namespace.
#' @return Invisibly returns `envir`.
#' @seealso [`jit_roclet()`], [`jit()`]
#' @export
apply_jit_registry <- function(registry, envir = parent.frame()) {
  for (entry in registry) {
    assign(
      entry$name,
      jit(
        get(entry$name, envir = envir, inherits = FALSE),
        backend = "auto",
        static = entry$static
      ),
      envir = envir
    )
  }
  invisible(envir)
}

hashvalues <- function(h) {
  val <- vector("list", numhash(h))
  idx <- 0
  maphash(h, function(k, v) {
    idx <<- idx + 1
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
  stopifnot(bits %% 8 == 0, bits >= 8)
  n <- bits %/% 8
  if (!signed) {
    return(list(
      min = as.raw(rep(0x00, n)),
      max = as.raw(rep(0xFF, n))
    ))
  }
  hi_min <- as.raw(0x80) # 1000 0000
  hi_max <- as.raw(0x7F) # 0111 1111
  zeros <- as.raw(rep(0x00, n - 1))
  ff <- as.raw(rep(0xFF, n - 1))
  list(min = c(zeros, hi_min), max = c(ff, hi_max))
}


nv_minval <- function(dtype, device) {
  dtype <- as.character(dtype)
  if (grepl("^f", dtype)) {
    nv_scalar(-Inf, dtype = dtype, device = device)
  } else if (dtype == "bool") {
    nv_scalar(FALSE, dtype = "bool", device = device)
  } else {
    nv_scalar(pjrt_buffer(
      globals$ranges_raw[[dtype]]$min,
      dtype = dtype,
      device = device,
      row_major = TRUE,
      shape = integer()
    ))
  }
}

nv_maxval <- function(dtype, device) {
  dtype <- as.character(dtype)
  if (grepl("^f", dtype)) {
    nv_scalar(Inf, dtype = dtype, device = device)
  } else if (dtype == "bool") {
    nv_scalar(TRUE, dtype = "bool", device = device)
  } else {
    nv_scalar(pjrt_buffer(
      globals$ranges_raw[[dtype]]$max,
      dtype = dtype,
      device = device,
      row_major = TRUE,
      shape = integer()
    ))
  }
}

without <- function(x, indices) {
  if (length(indices)) {
    x[-indices]
  } else {
    x
  }
}

shape2string <- function(x, parenthesize = TRUE) {
  if (is_shape(x)) {
    x <- x$dims
  }
  if (parenthesize) {
    sprintf("(%s)", paste0(x, collapse = ","))
  } else {
    paste0(x, collapse = ",")
  }
}

shapes2string <- function(shapes) {
  paste0(sapply(shapes, shape2string), sep = ", ")
}

zeros <- function(dtype, shape, ambiguous) {
  prim_fill(0L, dtype = dtype, shape = shape, ambiguous = ambiguous)
}

ones <- function(dtype, shape, ambiguous) {
  prim_fill(1L, dtype = dtype, shape = shape, ambiguous = ambiguous)
}


zeros_like <- function(x, ambiguous = FALSE) {
  zeros(dtype(x), shape(x), ambiguous)
}

ones_like <- function(x, ambiguous = FALSE) {
  ones(dtype(x), shape(x), ambiguous)
}

#' @title Abstract Properties
#' @name abstract_properties
#' @description
#' Calls the extractor after converting the input to an [`AbstractArray`].
#' @param x ([`arrayish`])\cr
#' @export
shape_abstract <- function(x) {
  shape(to_abstract(x))
}

#' @rdname abstract_properties
#' @export
naxes_abstract <- function(x) {
  length(shape_abstract(x))
}

#' @rdname abstract_properties
#' @export
dtype_abstract <- function(x) {
  dtype(to_abstract(x))
}

#' @export
#' @rdname abstract_properties
ambiguous_abstract <- function(x) {
  to_abstract(x)$ambiguous
}

dtype2string <- function(dtype, ambiguous = FALSE) {
  paste0(repr(dtype), if (ambiguous) "?")
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

cache_size <- function(f) {
  # All jit paths cache in pjrt's native dispatcher.
  dispatcher <- environment(f)$dispatcher
  if (is.null(dispatcher)) {
    cli_abort("{.arg f} has no dispatcher; is it a jitted function?")
  }
  pjrt::dispatcher_size(dispatcher)
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
  # slice_sizes are in the order of `x_shape`, so we need to reverse the start_index_map
  if (length(x_shape) != length(slice_sizes)) {
    cli_abort("{.arg x_shape} and {.arg slice_sizes} must have the same length")
  }

  indices_shape <- shape(start_indices)
  n_index_coords <- length(start_index_map)

  if (n_index_coords == 0L) {
    return(start_indices)
  }

  # Build max bounds for each coordinate
  max_bounds <- integer(n_index_coords)
  for (coord_idx in seq_len(n_index_coords)) {
    x_axis <- start_index_map[coord_idx]
    x_size <- x_shape[x_axis]
    slice_size_for_axis <- slice_sizes[x_axis]
    max_bounds[coord_idx] <- max(1L, x_size - slice_size_for_axis + 1L)
  }

  if (index_vector_axis <= length(indices_shape)) {
    # Explicit index vector axis - build bounds arrays
    bounds_shape <- rep(1L, length(indices_shape))
    bounds_shape[index_vector_axis] <- n_index_coords

    min_tensor <- prim_broadcast_in_axes(
      prim_fill(1L, dtype = dtype(start_indices), shape = integer()),
      indices_shape,
      integer()
    )

    # The max bound is the same for a given slice along the index_vector_axis
    max_tensor_vals <- prim_reshape(
      nv_convert(nv_array(max_bounds, dtype = "i64"), dtype = dtype(start_indices)),
      bounds_shape
    )
    max_tensor <- nv_broadcast_to(max_tensor_vals, indices_shape)

    prim_clamp(min_tensor, start_indices, max_tensor)
  } else {
    # Implicit index vector (single coordinate)
    min_tensor <- prim_fill(1L, dtype = dtype(start_indices), shape = integer())
    max_tensor <- prim_fill(max_bounds[1L], dtype = dtype(start_indices), shape = integer())
    prim_clamp(min_tensor, start_indices, max_tensor)
  }
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
    if (i %in% inserted_window_axes) {
      slice_sizes[i] <- 1L
    } else if (i %in% x_batching_axes) {
      slice_sizes[i] <- 1L
    } else {
      slice_sizes[i] <- update_shape[update_window_axes[update_window_pos]]
      update_window_pos <- update_window_pos + 1L
    }
  }
  slice_sizes
}

col_major_layout <- function(naxes) {
  as.integer(seq.int(0L, naxes - 1L))
}

col_major_layouts <- function(...) {
  lapply(list(...), col_major_layout)
}

is_device_arg <- function(x) {
  inherits(x, "AnvlDeviceArg")
}

# returns list(device | NULL, backend)
resolve_device <- function(device, backend) {
  if (is.character(device)) {
    backend <- backend %||% default_backend()
    device <- if (backend == "auto") {
      nv_device(device, default_backend())
    } else {
      nv_device(device, backend)
    }
    return(list(device, backend))
  }
  if (is.null(device)) {
    return(list(NULL, backend %||% default_backend()))
  }
  # concrete device
  if (is.null(backend) || (backend == "auto")) {
    return(list(device, backend(device)))
  }
  if (backend(device) != backend) {
    cli_abort(c(
      "Backend of requested device does not match requested backend",
      i = "backend(device) = {backend(device)}",
      i = "backend = {backend}"
    ))
  }
  list(device, backend)
}
