#' @include jit.R
NULL

like_defaults <- function(like, ...) {
  if (is_rdata(to_abstract(like))) {
    cli_abort(c(
      "{.arg like} must be an array to take defaults from.",
      x = "Got an R {typeof(like)}, which has no data type of its own.",
      i = "Pass {.arg dtype} and {.arg shape} directly, or build an array with {.fn nv_array}."
    ))
  }
  args <- list(...)
  getters <- list(
    dtype = dtype,
    shape = shape,
    # `device` only comes from an array that is placed on one. A traced value
    # is not, and neither is a constant of the trace -- reading the
    # `PlainDeviceCpu()` of one back would allocate the result on the first CPU
    # device, which under `jit()` is a device the graph never asked for. It
    # stays `NULL` instead, so the constructor below builds a constant of the
    # trace as well.
    device = placement_device
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
#'   An existing array. Any of `dtype`, `device` and `shape` that are `NULL`
#'   (the default) are taken from `like`.
#' @export
nv_array_like <- function(like, data, dtype = NULL, device = NULL, shape = NULL) {
  do.call(
    nv_array,
    c(
      list(data = data),
      like_defaults(like, dtype = dtype, device = device, shape = shape)
    )
  )
}

#' @rdname AnvlArray
#' @export
nv_scalar_like <- function(like, data, dtype = NULL, device = NULL) {
  do.call(
    nv_scalar,
    c(
      list(data = data),
      like_defaults(like, dtype = dtype, device = device)
    )
  )
}

#' @rdname AnvlArray
#' @export
nv_empty_like <- jit(
  function(like, dtype = NULL, shape = NULL, device = NULL) {
    do.call(nv_empty, like_defaults(like, dtype = dtype, shape = shape, device = device))
  },
  static = 2:4
)

#' @rdname nv_fill
#' @export
nv_fill_like <- function(like, value, shape = NULL, dtype = NULL, device = NULL) {
  do.call(
    nv_fill,
    c(
      list(value = value),
      like_defaults(like, shape = shape, dtype = dtype, device = device)
    )
  )
}

#' @rdname nv_iota
#' @export
nv_iota_like <- function(like, axis, shape = NULL, start = 1L, dtype = NULL, device = NULL) {
  do.call(
    nv_iota,
    c(
      list(axis = axis, start = start),
      like_defaults(like, shape = shape, dtype = dtype, device = device)
    )
  )
}

#' @rdname nv_seq
#' @export
nv_seq_like <- jit(
  function(like, start, end, by = NULL, dtype = NULL, device = NULL) {
    do.call(
      nv_seq,
      c(
        list(start = start, end = end, by = by),
        like_defaults(like, dtype = dtype, device = device)
      )
    )
  },
  static = 2:6
)

#' @rdname nv_linspace
#' @export
nv_linspace_like <- jit(
  function(like, start, end, steps, dtype = NULL, device = NULL) {
    do.call(
      nv_linspace,
      c(
        list(start = start, end = end, steps = steps),
        like_defaults(like, dtype = dtype, device = device)
      )
    )
  },
  static = 2:6
)

#' @rdname nv_eye
#' @export
nv_eye_like <- jit(
  function(like, n, dtype = NULL, device = NULL) {
    do.call(nv_eye, c(list(n = n), like_defaults(like, dtype = dtype, device = device)))
  },
  static = 2:4
)

#' @rdname nv_lower_tri
#' @export
nv_lower_tri_like <- jit(
  function(like, diagonal = -1L, shape = NULL, device = NULL) {
    do.call(
      nv_lower_tri,
      c(list(diagonal = diagonal), like_defaults(like, shape = shape, device = device))
    )
  },
  static = 2:4
)

#' @rdname nv_upper_tri
#' @export
nv_upper_tri_like <- jit(
  function(like, diagonal = 1L, shape = NULL, device = NULL) {
    do.call(
      nv_upper_tri,
      c(list(diagonal = diagonal), like_defaults(like, shape = shape, device = device))
    )
  },
  static = 2:4
)
