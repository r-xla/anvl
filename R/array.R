#' @title AnvlArray
#' @description
#' The main array object.
#' Its type is determined by a data type and a shape.
#'
#' @section Terminology:
#' An array's **axes** are the indices that identify its directions, numbered
#' `1`, `2`, `3`, ... The **size** of an axis (its *axis size*) is the extent
#' along that axis, and the **shape** is the vector of all axis sizes. For
#' example, `nv_array(1:6, shape = c(2, 3))` has two axes; the size of axis `1`
#' is `2` and the size of axis `2` is `3`, so its shape is `c(2, 3)`. Use
#' [`naxes()`][tengen::naxes] for the number of axes and
#' [`shape()`][tengen::shape] for the axis sizes. We speak of the *size of an
#' axis* rather than an array's "dimensions", as the latter is generally
#' overloaded as it is used to refer to both the axis and it's size.
#'
#'
#' @section Extractors:
#' The following generic functions can be used to extract information from an `AnvlArray`:
#' - [`dtype()`][tengen::dtype]: Get the data type of the array.
#' - [`shape()`][tengen::shape]: Get the shape (axis sizes) of the array.
#' - [`naxes()`][tengen::naxes]: Get the number of axes.
#' - [`device()`][tengen::device]: Get the device of the array.
#' - [`platform()`]: Get the platform (e.g. `"cpu"`, `"cuda"`).
#'
#' @section Serialization:
#' Arrays can be serialized to and from the
#' [safetensors](https://huggingface.co/docs/safetensors/index) format:
#' - [`nv_save()`] / [`nv_read()`]: Save/load arrays to/from a file.
#' - [`nv_serialize()`] / [`nv_unserialize()`]:
#'   Serialize/deserialize arrays to/from raw vectors.
#'
#' @section Backend:
#' An `AnvlArray` is backend-dependent: it belongs to exactly one backend
#' (`"pjrt"` or the experimental `"quickr"`) and lives on a device of that backend.
#' The supported data types and devices differ between backends.
#'
#' @seealso [nv_fill], [nv_iota], [nv_seq], [as_array], [nv_serialize]
#'
#' @param data (any)\cr
#'   `integer()`, `double()`, or `logical()` scalar, vector, or array.
#' @param dtype (`NULL` | `character(1)` | [`DataType`])\cr
#'   One of `r roxy_dtypes()` or a [`tengen::DataType`].
#'   The default (`NULL`) uses the current backend's default dtype:
#'   `f32` for numeric data on `"pjrt"`, `f64` for numeric data on `"quickr"`,
#'   `i32` for integer data, and `bool` for logical data.
#' @template param_device
#' @param shape (`NULL` | `integer()`)\cr
#'   The output shape of the array.
#'   The default (`NULL`) is to infer it from the data if possible.
#'   Note that [`nv_array`] interprets length 1 vectors as having shape `(1)`.
#'   To create a "scalar" with no axes (shape `()`), use [`nv_scalar`] or explicitly specify `shape = c()`.
#' @param backend (`NULL` | `character(1)`)\cr
#'   Backend the array belongs to (`"pjrt"` or `"quickr"`).
#'   The default (`NULL`) is inferred from `device` when `device` is a
#'   backend-specific device object, and otherwise falls back to
#'   [`default_backend()`].
#'   Must not be specified inside [`jit()`].
#' @param byrow (`logical(1)`)\cr
#'   When constructing from an R object and the result has at least two
#'   axes, fill the array in row-major order rather than the
#'   default column-major order, mirroring [`base::matrix()`]'s `byrow`.
#'   Only allowed when `data` is an R object — passing an existing
#'   `AnvlArray` together with `byrow = TRUE` is an error.
#' @param check (`logical(1)`)\cr
#'   If `TRUE`, error when `data` contains any `NA` values. XLA has no
#'   representation for missing values, so they are otherwise silently
#'   coerced to the closest available value of the target dtype (e.g. `NaN`
#'   for floats, the bit pattern `-2147483648` for `i32`, `TRUE` for
#'   `bool`). Defaults to `FALSE`. See the "Gotchas" vignette.
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
#' # An uninitialized 2x3 array (contents are unspecified)
#' nv_empty("f32", shape = c(2L, 3L))
#'
#' # --- Extractors ---
#' x <- nv_array(1:6, shape = c(2L, 3L))
#' dtype(x)
#' shape(x)
#' naxes(x)
#' device(x)
#' platform(x)
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
nv_array <- function(
  data,
  dtype = NULL,
  device = NULL,
  shape = NULL,
  backend = NULL,
  byrow = FALSE,
  check = FALSE
) {
  assert_flag(byrow)
  assert_flag(check)
  if (check && !is_anvl_array(data) && anyNA(data)) {
    n_na <- sum(is.na(data))
    cli_abort(c(
      "Input {.arg data} contains {n_na} {.val NA} value{?s}, which {?has/have} no representation at the XLA level.",
      i = "Replace or drop missing values before transferring, or set {.code check = FALSE} to skip this check."
    ))
  }
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
    return(data)
  }
  if (is_rdata_box(data)) {
    # An R value that has not taken a data type yet: build it at the one asked
    # for, which is what makes `nv_array(x, dtype = )` the way to give a traced
    # R value a type.
    if (!is.null(shape) && !identical(as.integer(shape), shape(data))) {
      cli_abort("Cannot change shape of a traced value from {.val {shape(data)}} to {.val {shape}}")
    }
    if (is.null(dtype)) {
      return(trace_commit_rdata(data))
    }
    return(materialize_rdata(data, as_dtype(dtype)))
  }
  if (is_box(data)) {
    cli_abort(c(
      "Cannot build an {.cls AnvlArray} from a traced value.",
      i = "It already has a data type; use {.fn nv_convert} to change it."
    ))
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
    return(globals$backends[["plain"]]$new_data(data, dtype, shape, device))
  }
  if (is.null(backend) && is_device(device)) {
    backend <- backend(device)
  }
  backend <- backend %||% default_backend()
  globals$backends[[backend]]$new_data(data, dtype, shape, device)
}

#' @title Convert to AnvlArray
#' @description
#' Use this to canonicalize inputs at the start of a function so it works
#' both with eager executing and in combination with [`jit()`].
#' Use [`as_anvl_array()`] for a single input and [`as_anvl_arrays()`] for multiple inputs.
#' The latter will also ensure all arrays are from the same backend and live on the same device,
#' and can additionally apply type promotion rules via the `.promote` argument.
#'
#' @details
#' [Boxes][GraphBox] and [`AnvlArray`]s are returned as they are -- for an
#' `AnvlArray` we check that it lives on `device`, if one is given.
#'
#' A bare R value is converted, always -- the same conversion while tracing and
#' in eager mode. This is what the names say, and it matters that they mean one
#' thing in both modes: an `nv_*` function built on them must not do two
#' different things depending on whether it is called under [`jit()`].
#'
#' Which dtype it is converted at is the question these functions answer. An R
#' value has none of its own (see [`RDataArray`]); it can take the one the
#' operation settles on, which is what makes `x_f64 / sqrt(2)` exact, or -- with
#' nothing to take it from -- its default (`f32` for a double, `i32` for an
#' integer, `bool` for a logical).
#'
#' `as_anvl_arrays(.promote = )` is the one that can be exact, because it decides
#' the dtype and does the conversion in the same call. Without a `.promote` rule
#' there is nothing to decide from, so an R value takes its default, and a
#' caller that goes on to convert it (`nv_convert(args[[2]], dtype(args[[1]]))`)
#' has already lost the digits below that default. **An `nv_*` function whose
#' result dtype depends on its arguments should say so with a
#' [rule][promote_rule]** rather than canonicalize first and convert
#' afterwards.
#'
#' To keep an R value open -- to hold it until the dtype is known -- do not
#' canonicalize it at all: [`nv_convert()`] and the primitives take it as it is,
#' and [`peek_dtype()`] answers what it would commit to without committing it.
#'
#' @param x ([`arrayish`])\cr
#'   Input to standardize.
#' @param ... ([`arrayish`])\cr
#'   Inputs to align. Name them to be able to point `.promote` at one of them.
#' @param device (`NULL` | [`device`])\cr
#'   Target device. If `x` is an `AnvlArray` on a different device, an error
#'   is raised.
#' @param .promote (`NULL` | [`PromoteRule`][promote_rule])\cr
#'   Which dtype every input is brought to: [`promote_common()`] for the common
#'   one, [`promote_like()`] for the one a particular argument has, or
#'   [`promote_dtype()`] for one the caller names. `NULL` (default) decides
#'   none, so each input keeps the dtype it has and an R value takes its
#'   default.
#'
#'   [`promote_grouped()`] combines rules to promote several groups of arguments
#'   independently; they must name disjoint sets with `only`.
#'
#'   The name is dotted because the inputs go in `...`: an argument the caller
#'   happens to call `promote` is data, not the rule.
#'
#'   Inputs are *realized* at that dtype rather than converted to it: an R value
#'   has no dtype to convert from, so it is built at the target directly, which
#'   is what keeps `x_f64 / sqrt(2)` exact.
#' @return ([`AnvlArray`], or a [`GraphBox`] while tracing, for
#'   `as_anvl_array()`; a `list` of them for `as_anvl_arrays()`, keeping the
#'   names of `...`).
#' @seealso [peek_dtype()], [nv_promote_to_common()]
#' @examplesIf pjrt::plugins_downloaded()
#' as_anvl_array(1L)
#' as_anvl_arrays(nv_array(1:3), 1L)
#' # each input keeps its own dtype by default, brought to a common one with
#' # `.promote` -- and only then is an R value exact at it
#' as_anvl_arrays(nv_array(1L), nv_array(1.5), .promote = promote_common())
#' # ... or to one particular argument's
#' as_anvl_arrays(x = nv_array(1L), y = nv_array(1.5), .promote = promote_like("x"))
#' @name as_anvl_array
NULL

#' @rdname as_anvl_array
#' @export
as_anvl_array <- function(x, device = NULL) {
  if (is_box(x)) {
    return(trace_commit_rdata_box(x))
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
  # A bare R value: it has no dtype of its own, and nothing here says what it
  # should be, so it takes its default.
  if (currently_tracing()) {
    return(trace_commit_rdata_box(maybe_box_arrayish(x)))
  }
  if (is_valid_r_lit(x)) {
    return(nv_scalar(x, device = device))
  }
  nv_array(x, device = device)
}

#' @rdname as_anvl_array
#' @export
as_anvl_arrays <- function(..., .promote = NULL) {
  aligned <- align_arrayish(list(...))
  args <- aligned$args
  if (is.null(.promote)) {
    return(lapply(args, as_anvl_array, device = aligned$device))
  }
  # Realized at the target dtype rather than converted to it: an R value has no
  # dtype to convert *from*, so it is built at the target directly (see
  # `realize_at()`), with every digit it had.
  # Every rule is resolved before any is applied, so a rule's target is read off
  # the arguments as the caller passed them.
  resolved <- resolve_promote_rules(.promote, args)
  args <- apply_promote_rules(args, resolved, device = aligned$device)
  # An argument no rule names is still aligned and converted, just not to a
  # target.
  rest <- setdiff(seq_along(args), unlist(lapply(resolved, `[[`, "positions")))
  args[rest] <- lapply(args[rest], as_anvl_array, device = aligned$device)
  args
}

# The device every input of an operation shares. Nothing is converted here: an R
# value stays an R value, so the caller can build it at the dtype the operation
# settles on rather than at the default. Returns the arguments and that device
# -- `NULL` while tracing, where jit places the inputs.
align_arrayish <- function(args) {
  for (i in seq_along(args)) {
    if (!is_arrayish(args[[i]])) {
      cli_abort("Expected arrayish input, but got {.cls {class(args[[i]])}}")
    }
  }
  # While tracing, device placement is handled by jit.
  if (currently_tracing()) {
    return(list(args = args, device = NULL))
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
  list(args = args, device = dev)
}

# Build `x` at `dtype`. An R value -- an [`RDataArray`] while tracing, the R
# value itself when not -- is built from its R data, so it arrives with every
# digit it had; anything that already has a dtype is converted.
realize_at <- function(x, dtype, device = NULL) {
  if (!is_box(x) && !is_anvl_array(x) && is_valid_r(x) && currently_tracing()) {
    # Inside a trace, box the value *without* committing it first, so that
    # materializing it below yields the node kind it would have had anyway --
    # an inlined literal for a scalar, a constant for an R array.
    x <- maybe_box_arrayish(x)
  }
  if (is_rdata_box(x)) {
    return(materialize_rdata(x, dtype))
  }
  if (!is_anvl_array(x) && !is_box(x) && is_valid_r(x)) {
    # Outside a trace the same rule applies as inside it: build the R value
    # where it is exact, and let a conversion out of its category be the
    # program's, not R's (see `materialize_rdata()`).
    build_at <- if (rdata_builds_directly(typeof(x), dtype)) dtype else rdata_natural_dtype(typeof(x))
    out <- if (is_valid_r_lit(x)) {
      nv_scalar(x, dtype = build_at, device = device)
    } else {
      nv_array(x, dtype = build_at, device = device)
    }
    if (build_at == dtype) {
      return(out)
    }
    return(prim_convert(out, dtype = dtype))
  }
  if (dtype(x) == dtype) {
    return(x)
  }
  prim_convert(x, dtype = dtype)
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
nv_scalar <- function(data, dtype = NULL, device = NULL, backend = NULL, check = FALSE) {
  nv_array(
    data,
    dtype = dtype,
    device = device,
    shape = integer(),
    backend = backend,
    check = check
  )
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
    backend = backend,
    byrow = byrow
  )
}

#' @rdname AnvlArray
#' @export
nv_empty <- function(dtype, shape, device = NULL, backend = NULL) {
  shape <- as.integer(shape)
  if (is.null(backend) && is_device(device)) {
    backend <- backend(device)
  }
  backend <- backend %||% default_backend()
  globals$backends[[backend]]$new_empty(
    dtype = dtype,
    shape = shape,
    device = device
  )
}

#' @rdname AbstractArray
#' @export
nv_aval <- function(dtype, shape) {
  r_type <- r_type_of_aval_spec(dtype)
  if (!is.null(r_type)) {
    return(RDataArray(NULL, shape = shape, r_type = r_type))
  }
  AbstractArray(dtype = dtype, shape = shape)
}

# The R storage types `nv_aval()` names, spelled as `typeof()` spells them.
# None of them is a dtype name, so there is nothing for them to be confused
# with.
r_type_aval_specs <- c("double", "integer", "logical")

# The R storage type an `nv_aval()` spec names, or NULL if it names a dtype.
r_type_of_aval_spec <- function(dtype) {
  if (!is.character(dtype) || length(dtype) != 1L || !dtype %in% r_type_aval_specs) {
    return(NULL)
  }
  dtype
}

#' @export
dtype.AnvlArray <- function(x, ...) {
  globals$backends[[x$backend]]$dtype(x)
}

#' @export
shape.AnvlArray <- function(x, ...) {
  globals$backends[[x$backend]]$shape(x)
}

#' @rdname as_array
#' @param check (`logical(1)`)\cr
#'   If `TRUE`, sanity-check the materialized R vector against losing
#'   information across the device-to-host boundary, and abort if any
#'   problematic value is detected. Forwarded to the backend; for the
#'   `pjrt` backend the relevant cases are `i32`/`i64` values colliding
#'   with the `NA` bit pattern and `ui64` values `>= 2^63` wrapping
#'   through `bit64::integer64`. See [`pjrt::as_array.PJRTBuffer()`] for
#'   the full list. Defaults to `FALSE`. See the "Gotchas" vignette.
#' @export
as_array.AnvlArray <- function(x, check = FALSE, ...) {
  assert_flag(check)
  globals$backends[[x$backend]]$as_array(x, check = check)
}

#' @method as.array AnvlArray
#' @export
as.array.AnvlArray <- function(x, ...) {
  as_array(x)
}

#' @method as.matrix AnvlArray
#' @export
as.matrix.AnvlArray <- function(x, ...) {
  nd <- naxes(x)
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
#' @param check (`logical(1)`)\cr
#'   Forwarded to [`as_array()`]; see there for details.
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
as.double.AnvlArray <- function(x, check = FALSE, ...) {
  dt <- dtype(x)
  if (!(is_dtype_float(dt) || is_dtype_int(dt) || is_dtype_uint(dt))) {
    cli_abort("{.fn as.double} requires a float or integer dtype, but got {.val {as.character(dt)}}.")
  }
  as.double(as_array(x, check = check))
}

#' @rdname as-AnvlArray
#' @method as.integer AnvlArray
#' @export
as.integer.AnvlArray <- function(x, check = FALSE, ...) {
  dt <- dtype(x)
  if (!(is_dtype_int(dt) || is_dtype_uint(dt))) {
    cli_abort("{.fn as.integer} requires a (signed or unsigned) integer dtype, but got {.val {as.character(dt)}}.")
  }
  as.integer(as_array(x, check = check))
}

#' @rdname as-AnvlArray
#' @method as.logical AnvlArray
#' @export
as.logical.AnvlArray <- function(x, check = FALSE, ...) {
  if (!is_dtype_bool(dtype(x))) {
    cli_abort("{.fn as.logical} requires a {.val bool} dtype, but got {.val {as.character(dtype(x))}}.")
  }
  as.logical(as_array(x, check = check))
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
  "pjrt"
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
#' - [`shape()`][tengen::shape]: Get the shape (axis sizes) of the array.
#' - [`naxes()`][tengen::naxes]: Get the number of axes.
#'
#' @section R data:
#' A value can also be one that has *no* dtype yet -- a bare R value, which only
#' takes one where it is used (see [`RDataArray`]). `nv_aval()` builds that aval
#' from the R storage type instead of a dtype: `"double"`, `"integer"` or
#' `"logical"`. This is the aval of a bare R argument of a jitted function,
#' whose value is unknown while tracing.
#'
#' @param dtype ([`tengen::DataType`] | `character(1)`)\cr
#'   The data type of the array, or -- for `nv_aval()` -- one of `"double"`,
#'   `"integer"`, `"logical"` for a value that has none yet.
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the array. Can be provided as an integer vector.
#' @seealso [LiteralArray], [ConcreteArray], [IotaArray], [RDataArray], [GraphValue], [to_abstract()], [GraphBox]
#'
#' @examplesIf pjrt::plugins_downloaded()
#' # -- Creating abstract arrays --
#' a <- AbstractArray("f32", c(2L, 3L))
#' a
#' dtype(a)
#' shape(a)
#'
#' # Shorthand
#' nv_aval("f32", c(2L, 3L))
#'
#' # An R value, which has no dtype until it is used
#' nv_aval("double", c(2L, 3L))
#'
#' # How AbstractArrays appear in an AnvlGraph
#' graph <- trace_fn(function(x) x + 1, list(x = nv_aval("i32", 4L)))
#' graph
#' graph$inputs[[1]]$aval
#'
#' @export
AbstractArray <- function(dtype, shape) {
  shape <- as_shape(shape)
  dtype <- as_dtype(dtype)

  structure(
    list(dtype = dtype, shape = shape),
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
#' shape(x)
#' naxes(x)
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
      data = data
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
#'
#' @examplesIf pjrt::plugins_downloaded()
#' x <- LiteralArray(1L, shape = integer())
#' x
#' shape(x)
#' naxes(x)
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
LiteralArray <- function(data, shape, dtype = default_dtype(data)) {
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
      shape = shape
    ),
    class = c("LiteralArray", "AbstractArray")
  )
}

#' @title R Data Array Class
#' @description
#' An [`AbstractArray`] for an R value that entered a program without a data
#' type: a length-1 vector or an [`array()`], either written in the body of a
#' traced function (`x / sqrt(2)`) or passed as an argument to a jitted one.
#'
#' Unlike every other abstract array it has **no** data type. R has no dtype to
#' give -- `1.5` is neither an `f32` nor an `f64` -- so instead of stamping one
#' at the boundary, the value is kept as R data and built into the program at
#' each use site, at the dtype that use site turns out to need. `x_f64 /
#' sqrt(2)` therefore sees the exact double, rather than one that was rounded
#' to `f32` on the way in and widened again.
#'
#' Which dtype a use site needs is normally not decided value by value: an
#' `nv_*` function canonicalizes its whole argument set once at the top with
#' [`as_anvl_arrays()`], and the [`.promote` rule][promote_rule] it passes
#' there is what names the dtype every R value among them is built at.
#'
#' A value that is never combined with a typed array has nothing to take its
#' dtype from and *commits* to the default for its R type: `f32` for a double,
#' `i32` for an integer, `bool` for a logical.
#'
#' @section Extractors:
#' [`shape()`][tengen::shape] and [`naxes()`][tengen::naxes] answer as they
#' would for the R value: `()` for a length-1 vector, `dim()` for an
#' [`array()`], and an error for a longer vector without a `dim()`, which is not
#' an arrayish value at all. [`dtype()`][tengen::dtype] **errors**: there is no
#' dtype to report until the value commits, exactly as `dtype(1.5)` has none to
#' report. Give the value a dtype (e.g. [`nv_array()`], [`nv_scalar()`]) to ask.
#'
#' @param data (`NULL` | `numeric()` | `logical()`)\cr
#'   The R data. `NULL` for an argument of a jitted function, whose value is
#'   deliberately unknown while tracing: the compiled program is cached by the
#'   argument's R type and shape, so it must not depend on the value.
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the value: `()` for a length-1 vector, its `dim()` for an
#'   R array.
#' @param r_type (`character(1)`)\cr
#'   The R storage type: `"double"`, `"integer"` or `"logical"`. Inferred from
#'   `data` when that is given.
#'
#' @seealso [AbstractArray], [LiteralArray]
#' @examplesIf pjrt::plugins_downloaded()
#' x <- RDataArray(1.5, shape = integer())
#' x
#' shape(x)
#' # dtype(x) would error: 1.5 has no data type of its own
#'
#' # How it appears during tracing
#' graph <- trace_fn(function(x) x, list(x = RDataArray(NULL, integer(), "double")))
#' graph
#' @export
RDataArray <- function(data, shape, r_type = typeof(data)) {
  shape <- as_shape(shape)
  r_type <- match.arg(r_type, c("double", "integer", "logical"))
  structure(
    list(
      data = data,
      r_type = r_type,
      # The dtype this value commits to when nothing tells it otherwise. Not
      # called `dtype`: it is what the value *would* become, and code that
      # reaches for `$dtype` must not silently get it.
      default_dtype = default_dtype_r(r_type),
      shape = shape
    ),
    class = c("RDataArray", "AbstractArray")
  )
}

is_rdata_array <- function(x) {
  inherits(x, "RDataArray")
}

#' @title R Data Input Class
#' @description
#' The [`AbstractArray`] of a program input that the caller supplies as bare R
#' data. Unlike every other input it has no data type of its own to be supplied
#' at: the program decides one (see [`RDataArray`]), and the runtime uploads the
#' R value at that dtype rather than at the default for its R storage type.
#'
#' It is the resolved form of an [`RDataArray`]: an `RDataArray` is a value
#' whose dtype is still open, an `RDataInput` is the same value once the
#' finished trace has settled which dtype its input is supplied at. It is what
#' makes an [`AnvlGraph`] self-describing about its inputs -- the backends read
#' the upload dtypes off the inputs themselves, via `graph_input_dtypes()`.
#'
#' @param dtype ([`tengen::DataType`] | `character(1)`)\cr
#'   The dtype the R value is uploaded at.
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the value.
#' @param r_type (`character(1)`)\cr
#'   The R storage type the caller passes: `"double"`, `"integer"` or
#'   `"logical"`.
#' @return (`RDataInput`)
#' @seealso [RDataArray], [AbstractArray]
#' @examplesIf pjrt::plugins_downloaded()
#' # An f64 program consuming an R double argument uploads it as f64.
#' graph <- trace_fn(function(x) nv_scalar(1, dtype = "f64") + x, list(x = nv_aval("double", integer()))) # nolint
#' graph$inputs[[1]]$aval
#' @export
# REVIEW: Why do we need this in addition to RDataArray?
# Okay, I get it (known vs dynamic, but do we also have this difference for anvl
# arrays?)
# RESPONSE: Not in the same shape, because a typed value needs only one class:
# `AbstractArray(dtype, shape)` is the aval of an input whose value is unknown,
# and `ConcreteArray` / `LiteralArray` / `IotaArray` are the ones that also
# carry the value -- but all four already have a dtype, so the input form is
# just the aval with the data left out. `RDataArray` cannot be its own input
# form: it has *no* dtype, and an input has to name the one it is supplied at.
# `RDataInput` is what an `RDataArray` becomes once the finished trace has
# settled that dtype, which is what makes the graph self-describing about its
# inputs (`graph_input_dtypes()`).
RDataInput <- function(dtype, shape, r_type) {
  structure(
    list(dtype = as_dtype(dtype), shape = as_shape(shape), r_type = r_type),
    class = c("RDataInput", "AbstractArray")
  )
}

is_rdata_input <- function(x) {
  inherits(x, "RDataInput")
}

#' @export
format.RDataInput <- function(x, ...) {
  sprintf("RDataInput(%s, %s, %s)", as.character(x$dtype), x$r_type, shape2string(x$shape))
}

#' @export
print.RDataInput <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
repr.RDataInput <- function(x, ...) {
  # The shape belongs to the dtype the program takes the value at, so it is
  # written next to it: `f64[2x3]<-double` reads as "an f64[2x3], supplied as an
  # R double", the way the graph printer spells it.
  sprintf("%s[%s]<-%s", repr(x$dtype), repr(x$shape), x$r_type)
}

#' @method dtype RDataArray
#' @export
dtype.RDataArray <- function(x, ...) {
  abort_no_dtype(x$default_dtype)
}

# An R value is the same value whether it is boxed for a trace or not, so the
# extractors have to answer the same way in both places -- otherwise an `nv_*`
# function means one thing under `jit()` and another eagerly. `shape.array()`
# (tengen) already covers an R array; these cover a length-1 vector, and refuse
# a dtype for either.
# REVIEW: What are the implications of this?
# RESPONSE: Three. (1) `shape(1.5)` answers instead of erroring, so an `nv_*`
# function can ask for a shape before anything has decided a dtype, and gets the
# same answer eagerly as it does for the boxed value under `jit()` -- which is
# the whole point, since a function must not mean two things depending on where
# it is called. (2) `shape(c(1, 2, 3))` errors rather than returning `3`: a
# length-3 vector with no `dim()` is not an arrayish value, and reporting a
# shape for it would only push the error further downstream. (3) It made
# `shape_abstract()` redundant, which is why that is gone. This is a method on
# our own generic (`tengen::shape`), so it only affects code that already calls
# it; base R is untouched.
#' @method shape numeric
#' @export
shape.numeric <- function(x, ...) {
  r_value_shape(x)
}

#' @method shape logical
#' @export
shape.logical <- function(x, ...) {
  r_value_shape(x)
}

#' @method dtype numeric
#' @export
dtype.numeric <- function(x, ...) {
  abort_no_dtype(default_dtype(x))
}

# REVIEW: Hmm, here it is kind of un-ambiguous? but keep it consistent i guess
# RESPONSE: Consistent, and also right on its own: a logical yields like any
# other R value -- `nv_array(1:2) + TRUE` is `i32`, not `bool` -- so `TRUE` has
# no more of a data type of its own than `1.5` has. `bool` is only what it
# commits to when nothing claims it, which is what `peek_dtype()` reports.
#' @method dtype logical
#' @export
dtype.logical <- function(x, ...) {
  abort_no_dtype(default_dtype(x))
}

r_value_shape <- function(x) {
  if (!is.null(dim(x))) {
    return(as.integer(dim(x)))
  }
  if (length(x) != 1L) {
    cli_abort(c(
      "{.fn shape} is undefined for a length-{length(x)} R vector.",
      i = "Only a length-1 R value and an {.fn array} are arrayish; use {.fn nv_array} to make one an array."
    ))
  }
  integer()
}

abort_no_dtype <- function(default_dtype) {
  cli_abort(
    c(
      "An R value has no data type of its own until it is used.",
      i = "{.fn dtype} is undefined here for the same reason {.code dtype(1.5)} is: the value only takes a data type when it meets a typed array, or when it commits to the default ({.val {as.character(default_dtype)}}).", # nolint
      i = "Give it one explicitly, e.g. {.fn nv_array} or {.fn nv_scalar} with {.arg dtype}."
    ),
    call = NULL
  )
}

#' @export
format.RDataArray <- function(x, ...) {
  # The data itself is deliberately left out: it can be a whole array, and what
  # matters about an RDataArray is what it is, not what it holds.
  sprintf("RDataArray(%s, %s)", x$r_type, shape2string(x$shape))
}

#' @export
print.RDataArray <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
repr.RDataArray <- function(x, ...) {
  sprintf("%s[%s]", x$r_type, repr(x$shape))
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
#' @param axis (`integer(1)`)\cr
#'   The axis along which values increase.
#' @param start (`integer(1)`)\cr
#'   The starting value.
#'
#' @examplesIf pjrt::plugins_downloaded()
#' x <- IotaArray(shape = 4L, dtype = "i32", axis = 1L)
#' x
#' shape(x)
#' naxes(x)
#' dtype(x)
#' # How it appears during tracing:
#' graph <- trace_fn(function() nv_iota(axis = 1L, dtype = "i32", shape = 4L), list())
#' graph
#' graph$outputs[[1]]$aval
#' @export
IotaArray <- function(shape, dtype, axis, start = 1L) {
  shape <- as_shape(shape)
  dtype <- as_dtype(dtype)
  # stablehlo::Shape is a wrapper object; its rank is length(shape$dims), not length(shape)
  assert_int(axis, lower = 1L, upper = length(shape$dims))
  assert_int(start)
  structure(
    list(shape = shape, dtype = dtype, axis = axis, start = start),
    class = c("IotaArray", "AbstractArray")
  )
}

#' @export
format.IotaArray <- function(x, ...) {
  sprintf(
    "IotaArray(shape=%s, dtype=%s, axis=%s, start=%s)",
    shape2string(x$shape),
    repr(x$dtype),
    x$axis,
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
#' @return `logical(1)` - `TRUE` if the arrays are equal, `FALSE` otherwise.
#' @examples
#' a <- nv_aval("f32", c(2L, 3L))
#' b <- nv_aval("f32", c(2L, 3L))
#'
#' # Same dtype and shape
#' eq_type(a, b)
#'
#' # Different dtype
#' eq_type(a, nv_aval("i32", c(2L, 3L)))
#'
#' # Different shape
#' eq_type(a, nv_aval("f32", c(3L, 2L)))
#'
#' # neq_type is the negation of eq_type
#' neq_type(a, b)
#' @export
eq_type <- function(e1, e2) {
  if (!inherits(e1, "AbstractArray") || !inherits(e2, "AbstractArray")) {
    cli_abort("e1 and e2 must be AbstractArrays")
  }
  # An `RDataArray` compares as the dtype it would commit to; it has no other.
  if (peek_dtype(e1) != peek_dtype(e2) || !identical(e1$shape, e2$shape)) {
    return(FALSE)
  }
  TRUE
}

#' @rdname eq_type
#' @export
neq_type <- function(e1, e2) {
  !eq_type(e1, e2)
}

#' @export
repr.AbstractArray <- function(x, ...) {
  sprintf("%s[%s]", repr(x$dtype), repr(x$shape))
}

#' @export
format.AbstractArray <- function(x, ...) {
  sprintf(
    "AbstractArray(dtype=%s, shape=%s)",
    repr(x$dtype),
    repr(x$shape)
  )
}

#' @export
format.ConcreteArray <- function(x, ...) {
  sprintf("ConcreteArray(%s, %s)", repr(x$dtype), shape2string(x$shape))
}

#' @export
format.LiteralArray <- function(x, ...) {
  data_str <- if (is_anvl_array(x$data)) {
    trimws(capture.output(print(x$data, ..., header = FALSE))[1L])
  } else {
    x$data
  }
  sprintf("LiteralArray(%s, %s, %s)", data_str, repr(x$dtype), shape2string(x$shape))
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
  sprintf("AnvlArray(dtype=%s, shape=%s)", repr(dtype(x)), paste(shape(x), collapse = "x"))
}

#' @export
print.AnvlArray <- function(x, header = TRUE, ...) {
  if (header) {
    cat("AnvlArray\n")
  }
  dtype_str <- as.character(dtype(x))
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
#' # R literals become LiteralArrays
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
    RDataArray(x, shape = if (is.null(dim(x))) integer() else as.integer(dim(x)))
  } else if (is_graph_box(x)) {
    gnode <- x$gnode
    gnode$aval
  } else {
    cli_abort("internal error: {.cls {class(x)}} is not an array-like object")
  }
  if (pure && class(x)[[1L]] != "AbstractArray") {
    AbstractArray(dtype = peek_dtype(x), shape = x$shape)
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
