like_defaults <- function(like, ...) {
  args <- list(...)
  getters <- list(
    dtype = dtype,
    shape = shape,
    # `device` and `backend` only come from concrete AnvlArrays. For a
    # GraphBox (during tracing) they stay NULL so downstream constructors
    # pick them up from the tracing context.
    device = function(x) if (is_anvl_array(x)) device(x),
    ambiguous = ambiguous,
    backend = function(x) if (is_anvl_array(x)) backend(x)
  )
  for (name in names(args)) {
    if (is.null(args[[name]])) {
      args[[name]] <- getters[[name]](like)
    }
  }
  args
}

#' @rdname AnvlArray
#' @param like ([`AnvlArray`])\cr
#'   An existing array. Any of `dtype`, `device`, `shape`, `ambiguous`, and
#'   `backend` that are `NULL` (the default) are taken from `like`.
#' @export
nv_array_like <- function(like, data, dtype = NULL, device = NULL, shape = NULL, ambiguous = NULL, backend = NULL) {
  do.call(
    nv_array,
    c(
      list(data = data),
      like_defaults(like, dtype = dtype, device = device, shape = shape, ambiguous = ambiguous, backend = backend)
    )
  )
}

#' @rdname AnvlArray
#' @export
nv_scalar_like <- function(like, data, dtype = NULL, device = NULL, ambiguous = NULL, backend = NULL) {
  do.call(
    nv_scalar,
    c(
      list(data = data),
      like_defaults(like, dtype = dtype, device = device, ambiguous = ambiguous, backend = backend)
    )
  )
}

#' @rdname AnvlArray
#' @export
#' @jit static 2:5
nv_empty_like <- function(like, dtype = NULL, shape = NULL, device = NULL, ambiguous = NULL) {
  do.call(nv_empty, like_defaults(like, dtype = dtype, shape = shape, device = device, ambiguous = ambiguous))
}

#' @rdname nv_fill
#' @export
nv_fill_like <- function(like, value, shape = NULL, dtype = NULL, ambiguous = NULL, device = NULL) {
  do.call(
    nv_fill,
    c(
      list(value = value),
      like_defaults(like, shape = shape, dtype = dtype, ambiguous = ambiguous, device = device)
    )
  )
}

#' @rdname nv_iota
#' @export
nv_iota_like <- function(like, axis, shape = NULL, start = 1L, dtype = NULL, ambiguous = NULL, device = NULL) {
  do.call(
    nv_iota,
    c(
      list(axis = axis, start = start),
      like_defaults(like, shape = shape, dtype = dtype, ambiguous = ambiguous, device = device)
    )
  )
}

#' @rdname nv_seq
#' @export
#' @jit static 2:7
nv_seq_like <- function(like, start, end, steps = NULL, dtype = NULL, ambiguous = NULL, device = NULL) {
  do.call(
    nv_seq,
    c(
      list(start = start, end = end, steps = steps),
      like_defaults(like, dtype = dtype, ambiguous = ambiguous, device = device)
    )
  )
}

#' @rdname nv_eye
#' @export
#' @jit static 2:4
nv_eye_like <- function(like, n, dtype = NULL, device = NULL) {
  do.call(nv_eye, c(list(n = n), like_defaults(like, dtype = dtype, device = device)))
}

#' @rdname nv_lower_tri
#' @export
#' @jit static 2:4
nv_lower_tri_like <- function(like, diagonal = -1L, shape = NULL, device = NULL) {
  do.call(
    nv_lower_tri,
    c(list(diagonal = diagonal), like_defaults(like, shape = shape, device = device))
  )
}

#' @rdname nv_upper_tri
#' @export
#' @jit static 2:4
nv_upper_tri_like <- function(like, diagonal = 1L, shape = NULL, device = NULL) {
  do.call(
    nv_upper_tri,
    c(list(diagonal = diagonal), like_defaults(like, shape = shape, device = device))
  )
}
