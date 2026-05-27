#' @title AnvlArray
#' @description
#' The main array object.
#' Its type is determined by a data type and a shape.
#'
#' @section Extractors:
#' The following generic functions can be used to extract information from an `AnvlArray`:
#' - [`dtype()`][tengen::dtype]: Get the data type of the array.
#' - [`shape()`][tengen::shape]: Get the shape (dimensions) of the array.
#' - [`ndims()`][tengen::ndims]: Get the number of dimensions.
#' - [`device()`][tengen::device]: Get the device of the array.
#' - [`platform()`]: Get the platform (e.g. `"cpu"`, `"cuda"`).
#' - [`ambiguous()`]: Get whether the dtype is ambiguous.
#'
#' @section Serialization:
#' Arrays can be serialized to and from the
#' [safetensors](https://huggingface.co/docs/safetensors/index) format:
#' - [`nv_save()`] / [`nv_read()`]: Save/load arrays to/from a file.
#' - [`nv_serialize()`] / [`nv_unserialize()`]:
#'   Serialize/deserialize arrays to/from raw vectors.
#'
#' @seealso [nv_fill], [nv_iota], [nv_seq], [as_array], [nv_serialize]
#'
#' @param data (any)\cr
#'   `integer()`, `double()`, or `logical()` scalar, vector, or array.
#' @param dtype (`NULL` | `character(1)` | [`DataType`])\cr
#'   One of `r stablehlo:::roxy_dtypes()` or a [`tengen::DataType`].
#'   The default (`NULL`) uses the current backend's default dtype:
#'   `f32` for numeric data on `"xla"`, `f64` for numeric data on `"quickr"`,
#'   `i32` for integer data, and `bool` for logical data.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The device for the array (`"cpu"`, `"cuda"`).
#'   Default is to use the CPU for new arrays.
#'   This can be changed by setting the `PJRT_PLATFORM` environment variable.
#' @param shape (`NULL` | `integer()`)\cr
#'   The output shape of the array.
#'   The default (`NULL`) is to infer it from the data if possible.
#'   Note that [`nv_array`] interprets length 1 vectors as having shape `(1)`.
#'   To create a "scalar" with dimension `()`, use [`nv_scalar`] or explicitly specify `shape = c()`.
#' @param ambiguous (`NULL` | `logical(1)`)\cr
#'   Whether the dtype should be marked as ambiguous.
#'   Defaults to `FALSE` for new arrays.
#' @param backend (`NULL` | `character(1)`)\cr
#'   Backend to use (`"xla"` or `"quickr"`).
#'   Defaults to `default_backend()`.
#'   Must not be specified inside [`jit()`].
#' @param byrow (`logical(1)`)\cr
#'   When constructing from an R object and the result has at least two
#'   dimensions, fill the array in row-major order rather than the
#'   default column-major order, mirroring [`base::matrix()`]'s `byrow`.
#'   Only allowed when `data` is an R object — passing an existing
#'   `AnvlArray` together with `byrow = TRUE` is an error.
#' @return ([`AnvlArray`])
#' @examplesIf pjrt::plugins_downloaded()
#' # A 1-d array (vector) with shape (4). Default type for integers is `i32`
#' nv_array(1:4)
#'
#' # Specify a dtype
#' nv_array(c(1.5, 2.5, 3.5), dtype = "f64")
#'
#' # A 2x3 matrix
#' nv_array(1:6, shape = c(2L, 3L))
#'
#' # A 2x3 matrix filled by row, like `matrix(1:6, 2, 3, byrow = TRUE)`.
#' nv_array(1:6, shape = c(2L, 3L), byrow = TRUE)
#'
#' # A scalar array.
#' nv_scalar(3.14)
#'
#' # A 0x3 array
#' nv_empty("f32", shape = c(0L, 3L))
#'
#' # --- Extractors ---
#' x <- nv_array(1:6, shape = c(2L, 3L))
#' dtype(x)
#' shape(x)
#' ndims(x)
#' device(x)
#' platform(x)
#' ambiguous(x)
#'
#' # --- Transforming arrays with jit ---
#' add_one <- jit(function(x) x + 1)
#' add_one(nv_array(1:4))
#'
#' # --- Eager mode (calling operations directly) ---
#' nv_add(nv_array(1:3), nv_array(4:6))
#'
#' @name AnvlArray
NULL

#' @rdname AnvlArray
#' @export
nv_array <- function(data, dtype = NULL, device = NULL, shape = NULL, ambiguous = NULL, backend = NULL, byrow = FALSE) {
  assert_flag(byrow)
  if (is_anvl_array(data)) {
    if (byrow) {
      cli_abort("{.arg byrow} only applies when constructing an {.cls AnvlArray} from an R object.")
    }
    if (!is.null(device) && !eq_device(device(data), nv_device(device, backend))) {
      cli_abort("Cannot change device of existing AnvlArray from {.val {device(data)}} to {.val {device}}")
    }
    if (!is.null(shape) && !identical(shape(data), as.integer(shape))) {
      cli_abort("Cannot change shape of existing AnvlArray")
    }
    if (!is.null(dtype)) {
      if (dtype(data) != as_dtype(dtype)) {
        cli_abort("Cannot change dtype of existing AnvlArray from {.val {dtype(data)}} to {.val {dtype}}")
      }
    }
    if (!is.null(ambiguous) && ambiguous(data) != ambiguous) {
      cli_abort("Cannot change ambiguous of existing AnvlArray from {.val {ambiguous(data)}} to {.val {ambiguous}}")
    }
    return(data)
  }
  if (is.null(ambiguous)) {
    ambiguous <- FALSE
  }
  if (!is.null(dtype)) {
    dtype <- as_dtype(dtype)
  }
  if (!is.null(shape)) {
    shape <- as.integer(shape)
  }
  if (byrow) {
    fill_shape <- shape %||% (if (!is.null(dim(data))) as.integer(dim(data)) else as.integer(length(data)))
    if (length(fill_shape) >= 2L) {
      # Fill column-major into the reversed shape, then permute axes back —
      # this is equivalent to placing `data` row-major into `fill_shape`.
      data <- aperm(array(data, dim = rev(fill_shape)), rev(seq_along(fill_shape)))
    }
  }
  if (currently_tracing() && is.null(device)) {
    # The functions we jit should be backend-agnostic
    if (!is.null(backend)) {
      cli_abort("{.arg backend} must not be specified when calling {.fn nv_array} inside {.fn jit}.")
    }
    return(globals$backends[["plain"]]$new_data(data, dtype, shape, device, ambiguous))
  }
  if (is.null(backend) && is_device(device)) {
    backend <- backend(device)
  }
  backend <- backend %||% default_backend()
  globals$backends[[backend]]$new_data(data, dtype, shape, device, ambiguous)
}

#' @title Convert to AnvlArray
#' @description
#' Use this to canonicalize inputs at the start of a function so it works
#' both with eager executing and in combination with [`jit()`].
#' Use [`as_anvl_array()`] for a single input and [`as_anvl_arrays()`] for multiple inputs.
#' The latter will also ensure all arrays are from the same backend and live on the same device.
#'
#' @details
#' During tracing, [boxes][GraphBox] are returned as is.
#' During eager mode, R literals and arrays are converted to `AnvlArray`s on the specified device.
#' For `AnvlArray` inputs, we check that they live on provided device (if specified).
#'
#' @param x ([`arrayish`])\cr
#'   Input to standardize.
#' @param ... ([`arrayish`])\cr
#'   Inputs to align.
#' @param device (`NULL` | [`device`])\cr
#'   Target device. If `x` is an `AnvlArray` on a different device, an error
#'   is raised.
#' @return ([`AnvlArray`] for `as_anvl_array()`, `list` of [`AnvlArray`]s
#'   for `as_anvl_arrays()`).
#' @examplesIf pjrt::plugins_downloaded()
#' as_anvl_array(1L)
#' as_anvl_arrays(nv_array(1:3), 1L)
#' @name as_anvl_array
NULL

#' @rdname as_anvl_array
#' @export
as_anvl_array <- function(x, device = NULL) {
  if (is_box(x)) {
    return(x)
  }
  if (!is_arrayish(x)) {
    cli_abort("Expected arrayish input, but got {.cls {class(x)}}")
  }
  if (is_anvl_array(x)) {
    if (!is.null(device) && !eq_device(device(x), nv_device(device, backend(x)))) {
      cli_abort(c(
        "Input is on an unexpected device.",
        i = "Expected {.val {as.character(nv_device(device, backend(x)))}}.",
        i = "Got {.val {as.character(device(x))}}."
      ))
    }
    return(x)
  }
  # Raw R literals have no explicit dtype, so they are ambiguous (except for
  # logicals, which unambiguously map to `bool`).
  ambiguous <- !is.logical(x)
  if (currently_tracing()) {
    # `maybe_box_arrayish()` lifts scalars into inlined GraphLiterals and
    # materializes R arrays as plain-backend named constants.
    return(maybe_box_arrayish(x))
  }
  if (is_valid_r_lit(x)) {
    return(nv_scalar(x, device = device, ambiguous = ambiguous))
  }
  nv_array(x, device = device, ambiguous = ambiguous)
}

#' @rdname as_anvl_array
#' @export
as_anvl_arrays <- function(...) {
  args <- list(...)
  # During tracing, device placement is handled by jit; just canonicalize each
  # input so `shape()` / `dtype()` etc. work downstream.
  # Without specifiying device, this will create arrays from the "plain" backend
  if (currently_tracing()) {
    return(lapply(args, as_anvl_array))
  }
  # Target device is the first concrete input's device, else the default.
  dev <- default_device()
  for (a in args) {
    if (is_anvl_array(a) && backend(a) != "plain") {
      dev <- device(a)
      break
    }
  }
  # Every other concrete input must match that device/backend.
  for (a in args) {
    if (!is_anvl_array(a) || backend(a) == "plain") {
      next
    }
    if (backend(a) != backend(dev)) {
      cli_abort(c(
        "Found inputs from multiple backends.",
        i = "Found backends {.val {backend(dev)}} and {.val {backend(a)}}."
      ))
    }
    if (!eq_device(device(a), dev)) {
      cli_abort(c(
        "Found inputs living on multiple devices, which is currently not supported.",
        i = "Found devices {.val {as.character(dev)}} and {.val {as.character(device(a))}}."
      ))
    }
  }
  lapply(args, as_anvl_array, device = dev)
}


is_anvl_array <- function(x) {
  inherits(x, "AnvlArray")
}

#' Get the underlying PJRT buffer from an AnvlArray or pass through other values
#' @param x An AnvlArray or any other value
#' @return The underlying PJRT buffer if x is an AnvlArray, otherwise x unchanged
#' @keywords internal
unwrap_if_array <- function(x) {
  if (is_anvl_array(x)) {
    x$data
  } else {
    x
  }
}

#' @rdname AnvlArray
#' @export
nv_scalar <- function(data, dtype = NULL, device = NULL, ambiguous = NULL, backend = NULL) {
  nv_array(data, dtype = dtype, device = device, shape = integer(), ambiguous = ambiguous, backend = backend)
}

infer_matrix_dim <- function(n, other, given) {
  if (other == 0L) {
    if (n != 0L) {
      cli_abort("{.arg {given}} is 0 but {.arg data} has {n} element{?s}.")
    }
    return(0L)
  }
  if (n %% other != 0L) {
    cli_abort("Data length ({n}) is not a multiple of {.arg {given}} ({other}).")
  }
  n %/% other
}

#' @rdname AnvlArray
#' @param nrow (`NULL` | `integer(1)`)\cr
#'   Number of rows. Inferred from `ncol` and the data length if `NULL`.
#'   Defaults to `1` when `data` is a scalar.
#' @param ncol (`NULL` | `integer(1)`)\cr
#'   Number of columns. Inferred from `nrow` and the data length if `NULL`.
#'   Defaults to `1` when `data` is a scalar.
#' @export
nv_matrix <- function(
  data,
  nrow = NULL,
  ncol = NULL,
  dtype = NULL,
  device = NULL,
  ambiguous = NULL,
  backend = NULL,
  byrow = FALSE
) {
  assert_int(nrow, lower = 0L, null.ok = TRUE)
  assert_int(ncol, lower = 0L, null.ok = TRUE)
  is_r_scalar <- !is_anvl_array(data) && is_valid_r_lit(data)
  is_array_scalar <- is_anvl_array(data) && length(shape(data)) == 0L
  if (is_r_scalar || is_array_scalar) {
    nrow <- nrow %||% 1L
    ncol <- ncol %||% 1L
    if (is_array_scalar) {
      data <- nv_broadcast_to(data, c(nrow, ncol))
    } else {
      data <- rep(data, nrow * ncol)
    }
    return(nv_array(
      data,
      dtype = dtype,
      device = device,
      shape = c(nrow, ncol),
      ambiguous = ambiguous,
      backend = backend
    ))
  }
  if (is.null(nrow) && is.null(ncol)) {
    cli_abort("At least one of {.arg nrow} and {.arg ncol} must be supplied.")
  }
  n <- if (is_anvl_array(data)) prod(shape(data)) else length(data)
  if (is.null(nrow)) {
    nrow <- infer_matrix_dim(n, ncol, given = "ncol")
  } else if (is.null(ncol)) {
    ncol <- infer_matrix_dim(n, nrow, given = "nrow")
  } else if (nrow * ncol != n) {
    cli_abort("Data length ({n}) does not match {.code nrow * ncol} ({nrow * ncol}).")
  }
  nv_array(
    data,
    dtype = dtype,
    device = device,
    shape = c(nrow, ncol),
    ambiguous = ambiguous,
    backend = backend,
    byrow = byrow
  )
}

#' @rdname AnvlArray
#' @export
nv_empty <- function(dtype, shape, device = NULL, ambiguous = FALSE) {
  shape <- as.integer(shape)
  storage_mode <- switch(
    substr(as.character(if (is_dtype(dtype)) dtype else as_dtype(dtype)), 1L, 1L),
    "f" = "double",
    "i" = ,
    "u" = "integer",
    "b" = "logical",
    "double"
  )
  data <- array(vector(storage_mode, prod(shape)), dim = shape)
  nv_array(data, dtype = dtype, device = device, shape = shape, ambiguous = ambiguous)
}

#' @rdname AbstractArray
#' @export
nv_aval <- function(dtype, shape, ambiguous = FALSE) {
  AbstractArray(dtype = dtype, shape = shape, ambiguous = ambiguous)
}

#' @export
dtype.AnvlArray <- function(x, ...) {
  globals$backends[[x$backend]]$dtype(x)
}

#' @title Get Ambiguity of an Array
#' @description
#' Returns whether the array's dtype is ambiguous.
#' @param x An array object
#' @param ... Additional arguments (unused)
#' @return `logical(1)` - `TRUE` if the dtype is ambiguous, `FALSE` otherwise
#' @export
ambiguous <- function(x, ...) {
  UseMethod("ambiguous")
}

#' @export
ambiguous.AnvlArray <- function(x, ...) {
  globals$backends[[x$backend]]$ambiguous(x)
}

#' @export
ambiguous.AbstractArray <- function(x, ...) {
  x$ambiguous
}

#' @export
shape.AnvlArray <- function(x, ...) {
  globals$backends[[x$backend]]$shape(x)
}

#' @export
as_array.AnvlArray <- function(x, ...) {
  globals$backends[[x$backend]]$as_array(x)
}

#' @method as.array AnvlArray
#' @export
as.array.AnvlArray <- function(x, ...) {
  as_array(x)
}

#' @method as.matrix AnvlArray
#' @export
as.matrix.AnvlArray <- function(x, ...) {
  nd <- ndims(x)
  if (nd != 2L) {
    cli_abort("{.fn as.matrix} requires a 2-D array, but got a {nd}-D array.")
  }
  as_array(x)
}

#' @export
as_raw.AnvlArray <- function(x, row_major = FALSE, ...) {
  globals$backends[[x$backend]]$as_raw(x, row_major)
}

#' @export
await.AnvlArray <- function(x, ...) {
  globals$backends[[x$backend]]$await_data(x)
  invisible(x)
}

#' @title Coerce AnvlArray to an R Vector
#' @description
#' Convert an [`AnvlArray`] to a bare R vector.
#' The array's shape is discarded; the result is always a flat vector.
#' Each method requires a compatible dtype:
#' * `as.double()` / `as.numeric()`: float or (signed/unsigned) integer dtypes.
#' * `as.integer()`: signed or unsigned integer dtypes.
#' * `as.logical()`: `bool`.
#' * `as.vector()`: any dtype; the R type is chosen by the dtype, or
#'   forced via `mode` (e.g. `"integer"`, `"double"`, `"logical"`, `"list"`).
#'
#' Use [`as_array()`] to obtain an R array that preserves the shape, or
#' [`nv_convert()`] to change the dtype of an [`AnvlArray`] before coercing.
#' @param x ([`AnvlArray`])\cr
#'   Array to coerce.
#' @param mode (`character(1)`)\cr
#'   For `as.vector()` only. See [base::as.vector()]. Defaults to `"any"`,
#'   meaning the natural R type for the array's dtype.
#' @param ... Unused.
#' @return An R vector of the corresponding type (`double`, `integer`, or `logical`).
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1.5, 2.5, 3.5, 4.5), shape = c(2L, 2L))
#' as.numeric(x)
#' as.integer(nv_array(1:6, shape = c(2L, 3L)))
#' as.logical(nv_array(c(TRUE, FALSE), dtype = "bool"))
#' as.vector(x)
#' @name as-AnvlArray
NULL

#' @rdname as-AnvlArray
#' @method as.double AnvlArray
#' @export
as.double.AnvlArray <- function(x, ...) {
  dt <- dtype(x)
  if (!(inherits(dt, "FloatType") || inherits(dt, "IntegerType") || inherits(dt, "UIntegerType"))) {
    cli_abort("{.fn as.double} requires a float or integer dtype, but got {.val {as.character(dt)}}.")
  }
  as.double(as_array(x))
}

#' @rdname as-AnvlArray
#' @method as.integer AnvlArray
#' @export
as.integer.AnvlArray <- function(x, ...) {
  dt <- dtype(x)
  if (!(inherits(dt, "IntegerType") || inherits(dt, "UIntegerType"))) {
    cli_abort("{.fn as.integer} requires a (signed or unsigned) integer dtype, but got {.val {as.character(dt)}}.")
  }
  as.integer(as_array(x))
}

#' @rdname as-AnvlArray
#' @method as.logical AnvlArray
#' @export
as.logical.AnvlArray <- function(x, ...) {
  if (!inherits(dtype(x), "BooleanType")) {
    cli_abort("{.fn as.logical} requires a {.val bool} dtype, but got {.val {as.character(dtype(x))}}.")
  }
  as.logical(as_array(x))
}

#' @rdname as-AnvlArray
#' @method as.vector AnvlArray
#' @export
as.vector.AnvlArray <- function(x, mode = "any") {
  as.vector(as_array(x), mode = mode)
}

#' @rdname platform
#' @export
platform.AnvlArray <- function(x, ...) {
  globals$backends[[x$backend]]$platform(x)
}

#' @export
device.AnvlArray <- function(x, ...) {
  globals$backends[[x$backend]]$device(x)
}

#' @title Get Backend of an Array
#' @param x An array object
#' @param ... Additional arguments (unused)
#' @return `character(1)` - the backend name
#' @export
backend <- function(x, ...) {
  UseMethod("backend")
}

#' @export
backend.AnvlArray <- function(x, ...) {
  x$backend
}

#' @export
backend.PJRTDevice <- function(x, ...) {
  "xla"
}

#' @export
backend.QuickrDevice <- function(x, ...) {
  "quickr"
}

#' @title Abstract Array Class
#' @description
#' Representation of an abstract array type.
#' During tracing, it is wrapped in a [`GraphNode`] held by a [`GraphBox`].
#' In the lowered [`AnvlGraph`] it is also part of [`GraphNode`]s representing the values in the program.
#'
#' The base class represents an *unknown* value, but child classes exist for:
#' * closed-over constants: [`ConcreteArray`]
#' * scalar arrays arising from R literals: [`LiteralArray`]
#' * sequence patterns: [`IotaArray`]
#'
#' To convert a [`arrayish`] value to an abstract array, use [`to_abstract()`].
#'
#' @section Extractors:
#' The following extractors are available on `AbstractArray` objects:
#' - [`dtype()`][tengen::dtype]: Get the data type of the array.
#' - [`shape()`][tengen::shape]: Get the shape (dimensions) of the array.
#' - [`ambiguous()`]: Get whether the dtype is ambiguous.
#' - [`ndims()`][tengen::ndims]: Get the number of dimensions.
#'
#' @param dtype ([`tengen::DataType`] | `character(1)`)\cr
#'   The data type of the array.
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the array. Can be provided as an integer vector.
#' @template param_ambiguous
#' @seealso [LiteralArray], [ConcreteArray], [IotaArray], [GraphValue], [to_abstract()], [GraphBox]
#'
#' @examplesIf pjrt::plugins_downloaded()
#' # -- Creating abstract arrays --
#' a <- AbstractArray("f32", c(2L, 3L))
#' a
#' dtype(a)
#' shape(a)
#' ambiguous(a)
#'
#' # Shorthand
#' nv_aval("f32", c(2L, 3L))
#'
#' # How AbstractArrays appear in an AnvlGraph
#' graph <- trace_fn(function(x) x + 1, list(x = nv_aval("i32", 4L)))
#' graph
#' graph$inputs[[1]]$aval
#'
#' @export
AbstractArray <- function(dtype, shape, ambiguous = FALSE) {
  shape <- as_shape(shape)
  dtype <- as_dtype(dtype)
  if (!test_flag(ambiguous)) {
    cli_abort("ambiguous must be a flag")
  }

  structure(
    list(dtype = dtype, shape = shape, ambiguous = ambiguous),
    class = "AbstractArray"
  )
}

is_abstract_array <- function(x) {
  inherits(x, "AbstractArray")
}

is_concrete_tensor <- function(x) {
  inherits(x, "ConcreteArray")
}

#' @method dtype AbstractArray
#' @export
dtype.AbstractArray <- function(x, ...) {
  x$dtype
}

#' @method shape AbstractArray
#' @export
shape.AbstractArray <- function(x, ...) {
  x$shape$dims
}

#' @title Concrete Array Class
#' @description
#' An [`AbstractArray`] that also holds a reference to the actual array data.
#' Usually represents a closed-over constant in a program.
#' Inherits from [`AbstractArray`].
#'
#' @section Lowering:
#' When lowering to XLA, these become inputs to the executable instead of embedding them into
#' programs as constants.
#' This is to avoid increasing compilation time and bloating the size of the executable.
#'
#' @param data ([`AnvlArray`])\cr
#'   The actual array data.
#'
#' @examplesIf pjrt::plugins_downloaded()
#' y <- nv_array(c(0.5, 0.6))
#' x <- ConcreteArray(y)
#' x
#' ambiguous(x)
#' shape(x)
#' ndims(x)
#' dtype(x)
#'
#' # How it appears during tracing
#' graph <- trace_fn(function() y, list())
#' graph
#' graph$outputs[[1]]$aval
#' @export
ConcreteArray <- function(data) {
  if (!inherits(data, "AnvlArray")) {
    cli_abort("data must be an AnvlArray")
  }

  structure(
    list(
      dtype = dtype_from_buffer(data),
      shape = Shape(shape(data)),
      data = data,
      ambiguous = ambiguous(data)
    ),
    class = c("ConcreteArray", "AbstractArray")
  )
}

#' @title Literal Array Class
#' @description
#' An [`AbstractArray`] where all elements have the same constant value.
#' This either arises when using literals in traced code (e.g. `x + 1`) or when using
#' [`nv_fill()`] to create a constant.
#'
#' @section Type Ambiguity:
#' When arising from R literals, the resulting `LiteralArray` is ambiguous because no type
#' information was available. See the `vignette("type-promotion")` for more details.
#'
#' @section Lowering:
#' `LiteralArray`s become constants inlined into the stableHLO program.
#' I.e., they lower to [`hlo_tensor()`].
#'
#' @param data (`double(1)` | `integer(1)` | `logical(1)` | [`AnvlArray`])\cr
#'   The scalar value or scalarish AnvlArray (contains 1 element).
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the array.
#' @param dtype ([`tengen::DataType`])\cr
#'   The data type. Defaults to the current backend's default floating dtype,
#'   `i32` for integer, and `bool` for logical.
#' @template param_ambiguous
#'
#' @examplesIf pjrt::plugins_downloaded()
#' x <- LiteralArray(1L, shape = integer(), ambiguous = TRUE)
#' x
#' ambiguous(x)
#' shape(x)
#' ndims(x)
#' dtype(x)
#' # How it appears during tracing:
#' # 1. via R literals
#' graph <- trace_fn(function() 1, list())
#' graph
#' graph$outputs[[1]]$aval
#' # 2. via nv_fill()
#' graph <- trace_fn(function() nv_fill(2L, shape = c(2, 2)), list())
#' graph
#' graph$outputs[[1]]$aval
#' @export
LiteralArray <- function(data, shape, dtype = default_dtype(data), ambiguous) {
  if (!is_valid_r_lit(data) && !inherits(data, "AnvlArray")) {
    cli_abort("LiteralArrays expect scalars or AnvlArray")
  }
  if (inherits(data, "AnvlArray")) {
    if (prod(shape(data)) != 1L) {
      cli_abort("AnvlArray must contain exactly one element.")
    }
  }
  shape <- as_shape(shape)
  dtype <- as_dtype(dtype)

  structure(
    list(
      data = data,
      dtype = dtype,
      shape = shape,
      ambiguous = ambiguous
    ),
    class = c("LiteralArray", "AbstractArray")
  )
}

#' @title Iota Array Class
#' @description
#' An [`AbstractArray`] representing an integer sequence.
#' Usually created by [`nv_iota()`] / [`nv_seq()`], which both call [`prim_iota()`] internally.
#' Inherits from [`AbstractArray`].
#'
#' @section Lowering:
#' When lowering to stableHLO, these become `iota` operations that generate the integer sequence
#' so they do not need to actually hold the data in the executable, similar to `ALTREP`s in R.
#' It lowers to [`hlo_iota()`], optionally shifting the starting value via
#' [`hlo_add()`].
#'
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the array.
#' @param dtype ([`tengen::DataType`])\cr
#'   The data type.
#' @param dimension (`integer(1)`)\cr
#'   The dimension along which values increase.
#' @param start (`integer(1)`)\cr
#'   The starting value.
#' @template param_ambiguous
#'
#' @examplesIf pjrt::plugins_downloaded()
#' x <- IotaArray(shape = 4L, dtype = "i32", dimension = 1L)
#' x
#' ambiguous(x)
#' shape(x)
#' ndims(x)
#' dtype(x)
#' # How it appears during tracing:
#' graph <- trace_fn(function() nv_iota(dim = 1L, dtype = "i32", shape = 4L), list())
#' graph
#' graph$outputs[[1]]$aval
#' @export
IotaArray <- function(shape, dtype, dimension, start = 1L, ambiguous = FALSE) {
  shape <- as_shape(shape)
  dtype <- as_dtype(dtype)
  assert_flag(ambiguous)
  # stablehlo::Shape is a wrapper object; its rank is length(shape$dims), not length(shape)
  assert_int(dimension, lower = 1L, upper = length(shape$dims))
  assert_int(start)
  structure(
    list(shape = shape, dtype = dtype, dimension = dimension, start = start, ambiguous = ambiguous),
    class = c("IotaArray", "AbstractArray")
  )
}

#' @export
format.IotaArray <- function(x, ...) {
  sprintf(
    "IotaArray(shape=%s, dtype=%s, dimension=%s, start=%s)",
    shape2string(x$shape),
    dtype2string(x$dtype, x$ambiguous),
    x$dimension,
    x$start
  )
}

#' @export
print.IotaArray <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
`==.AbstractArray` <- function(e1, e2) {
  cli_abort("Use {.fn eq_type} instead of {.code ==} for comparing AbstractArrays")
}

#' @export
`!=.AbstractArray` <- function(e1, e2) {
  cli_abort("Use {.fn neq_type} instead of {.code !=} for comparing AbstractArrays")
}

#' @title Compare AbstractArray Types
#' @description
#' Compare two abstract arrays for type equality.
#' @param e1 ([`AbstractArray`])\cr
#'   First array to compare.
#' @param e2 ([`AbstractArray`])\cr
#'   Second array to compare.
#' @param ambiguity (`logical(1)`)\cr
#'   Whether to consider the ambiguous field when comparing.
#'   If `TRUE`, arrays with different ambiguity are not equal.
#'   If `FALSE`, only dtype and shape are compared.
#' @return `logical(1)` - `TRUE` if the arrays are equal, `FALSE` otherwise.
#' @examples
#' a <- nv_aval("f32", c(2L, 3L))
#' b <- nv_aval("f32", c(2L, 3L))
#'
#' # Same dtype and shape
#' eq_type(a, b, ambiguity = FALSE)
#'
#' # Different dtype
#' eq_type(a, nv_aval("i32", c(2L, 3L)), ambiguity = FALSE)
#'
#' # Different shape
#' eq_type(a, nv_aval("f32", c(3L, 2L)), ambiguity = FALSE)
#'
#' # ambiguity parameter controls whether ambiguous field is compared
#' c <- nv_aval("f32", c(2L, 3L), ambiguous = TRUE)
#' eq_type(a, c, ambiguity = FALSE)
#' eq_type(a, c, ambiguity = TRUE)
#'
#' # neq_type is the negation of eq_type
#' neq_type(a, b, ambiguity = FALSE)
#' @export
eq_type <- function(e1, e2, ambiguity) {
  if (!inherits(e1, "AbstractArray") || !inherits(e2, "AbstractArray")) {
    cli_abort("e1 and e2 must be AbstractArrays")
  }
  if (e1$dtype != e2$dtype || !identical(e1$shape, e2$shape)) {
    return(FALSE)
  }
  if (ambiguity && (e1$ambiguous != e2$ambiguous)) {
    return(FALSE)
  }
  TRUE
}

#' @rdname eq_type
#' @export
neq_type <- function(e1, e2, ambiguity) {
  !eq_type(e1, e2, ambiguity)
}

#' @export
repr.AbstractArray <- function(x, ...) {
  sprintf("%s[%s]", paste0(repr(x$dtype), if (x$ambiguous) "?"), repr(x$shape))
}

#' @export
format.AbstractArray <- function(x, ...) {
  sprintf(
    "AbstractArray(dtype=%s, shape=%s)",
    if (x$ambiguous) paste0(repr(x$dtype), "?") else repr(x$dtype),
    repr(x$shape)
  )
}

#' @export
format.ConcreteArray <- function(x, ...) {
  sprintf("ConcreteArray(%s, %s)", dtype2string(x$dtype, x$ambiguous), shape2string(x$shape))
}

#' @export
format.LiteralArray <- function(x, ...) {
  data_str <- if (is_anvl_array(x$data)) {
    trimws(capture.output(print(x$data, ..., header = FALSE))[1L])
  } else {
    x$data
  }
  sprintf("LiteralArray(%s, %s, %s)", data_str, dtype2string(x$dtype, x$ambiguous), shape2string(x$shape))
}

#' @export
print.AbstractArray <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
print.ConcreteArray <- function(x, ...) {
  cat("ConcreteArray\n")
  print(x$data, header = FALSE)
  invisible(x)
}

#' @export
format.AnvlArray <- function(x, ...) {
  dtype_str <- if (ambiguous(x)) paste0(repr(dtype(x)), "?") else repr(dtype(x))
  sprintf("AnvlArray(dtype=%s, shape=%s)", dtype_str, paste(shape(x), collapse = "x"))
}

#' @export
print.AnvlArray <- function(x, header = TRUE, ...) {
  if (header) {
    cat("AnvlArray\n")
  }
  dtype_str <- paste0(as.character(dtype(x)), if (ambiguous(x)) "?")
  footer <- sprintf("[ %s%s{%s} ]", toupper(platform(x)), dtype_str, paste0(shape(x), collapse = ","))
  globals$backends[[x$backend]]$print_data(x, footer)
  invisible(x)
}

# fmt: skip
compare_proxy.AnvlArray <- function(x, path) { # nolint
  list(
    object = list(
      data = as_array(x),
      dtype = as.character(dtype(x)),
      ambiguous = ambiguous(x),
      backend = backend(x),
      device = as.character(device(x))
    ),
    path = path
  )
}

#' @title Convert to Abstract Array
#' @description
#' Convert an object to its abstract array representation ([`AbstractArray`]).
#' @param x (`any`)\cr
#'   Object to convert.
#' @param pure (`logical(1)`)\cr
#'   Whether to convert to a pure `AbstractArray` and not e.g. `LiteralArray` or `ConcreteArray`.
#' @return [`AbstractArray`]
#' @examplesIf pjrt::plugins_downloaded()
#' # R literals become LiteralArrays (ambiguous by default, except logicals)
#' to_abstract(1.5)
#' to_abstract(1L)
#' to_abstract(TRUE)
#'
#' # AnvlArrays become ConcreteArrays
#' to_abstract(nv_array(1:4))
#'
#' # Use pure = TRUE to strip subclass info
#' to_abstract(nv_array(1:4), pure = TRUE)
#'
#' @export
to_abstract <- function(x, pure = FALSE) {
  x <- if (is_anvl_array(x)) {
    ConcreteArray(x)
  } else if (is_abstract_array(x)) {
    x
  } else if (test_atomic(x) && (is.logical(x) || is.numeric(x))) {
    # logicals are not ambiguous
    LiteralArray(x, integer(), ambiguous = !is.logical(x))
  } else if (is_graph_box(x)) {
    gnode <- x$gnode
    gnode$aval
  } else {
    cli_abort("internal error: {.cls {class(x)}} is not an array-like object")
  }
  if (pure && class(x)[[1L]] != "AbstractArray") {
    AbstractArray(dtype = x$dtype, shape = x$shape, ambiguous = x$ambiguous)
  } else {
    x
  }
}

as_shape <- function(x) {
  if (test_integerish(x, any.missing = FALSE, lower = 0)) {
    Shape(as.integer(x))
  } else if (is_shape(x)) {
    x
  } else if (is.null(x)) {
    Shape(integer())
  } else {
    cli_abort("x must be an integer vector or a stablehlo::Shape")
  }
}

is_shape <- function(x) {
  inherits(x, "Shape")
}


#' @title Array-like Objects
#' @description
#' A `arrayish` value is any object that can be input to a primitive such as [`prim_add`].
#'
#' During runtime of a JIT-compiled function, these are [`AnvlArray`] objects.
#'
#' The following types are arrayish (during tracing):
#' * [`AnvlArray`]: a concrete array holding data on a device.
#' * [`GraphBox`]: a boxed abstract array representing a value in a graph.
#' * Length-1 vectors: `numeric(1)` and `logical(1)`
#' * R arrays of types: `numeric` and `logical`.
#'
#' Use [`is_arrayish()`] to check whether a value is arrayish.
#'
#' @param x (`any`)\cr
#'   Object to check.
#' @param convert_ok (`logical(1)`)\cr
#'   Whether to accept `numeric(1)` and `logical(1)` and R arrays of type `numeric` and `logical`.
#' @return `logical(1)`
#' @name arrayish
#' @seealso [AnvlArray], [GraphBox]
#' @examplesIf pjrt::plugins_downloaded()
#' # AnvlArrays are arrayish
#' is_arrayish(nv_array(1:4))
#'
#' # Scalar R literals are arrayish by default
#' is_arrayish(1.5)
#' # R arrays are arrayish by default
#' is_arrayish(array(1.5))
#'
#' # R arrays
#' is_arrayish(array(1:4), convert_ok = TRUE)
#' is_arrayish(array(1:4), convert_ok = FALSE)
#'
#' # Length 1 vectors
#' is_arrayish(1.5, convert_ok = FALSE)
#' is_arrayish(1.5, convert_ok = TRUE)
NULL

#' @rdname arrayish
#' @export
is_arrayish <- function(x, convert_ok = TRUE) {
  ok <- inherits(x, "AnvlArray") ||
    is_box(x)

  if (ok) {
    return(TRUE)
  }

  if (!convert_ok) {
    return(FALSE)
  }
  is_valid_r(x)
}


#' @title Create an R array
#' @description
#' Create an R array without having to wrap data in `c()`
#' @param ... (any)\cr
#'   Values of new array.
#' @param shape (`NULL` | `integer()`)\cr
#'   Shape of new array. If `NULL` (default), uses length of elements to create a 1D array.
#' @export
#' @examples
#' arr(1, 2, 3)
#' arr(1, 2, 3, 4, shape = c(2, 2))
arr <- function(..., shape = NULL) {
  vals <- c(...)
  if (is.null(vals)) {
    cli_abort("Invalid input values")
  }
  assert_integerish(shape, null.ok = TRUE)
  assert_vector(vals, min.len = 1L)
  nvals <- length(vals)
  if (!is.null(shape) && (nvals != 1) && (prod(shape) != nvals)) {
    cli_abort("Number of elements is {nvals}, but {.arg shape} is {shape}")
  }
  array(vals, dim = shape %||% length(vals))
}
