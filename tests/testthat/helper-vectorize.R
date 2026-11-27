# Checks `vectorize()` on `f` against applying `f` to one slice at a time.
#
# `...` are the arguments of `f`, each an array whose last axis is the batch
# axis; they must agree on its size. For every non-empty subset of them,
# `vectorize(f, in_axes = , out_axes = )` is called with the arguments of the
# subset in full and the others at their first slice, and each of its outputs
# must have the data type of `f`'s and equal the results of `f` on the slices,
# stacked along the output's batch axis.
#
# The batch axes are drawn at random for every subset: each argument mapped
# over has its batch axis moved to a random axis of its own, named in
# `in_axes`, and each output is stacked along a random axis it has room for,
# passed as `out_axes` with the structure of what `f` returns. Either is named
# from the front or, as negative, from the end, at random. Call it under
# `withr::local_seed()`.
#
# Two more checks map over all arguments: one at a batch size of 1 or 7,
# drawn at random, made of the slices of `...` (repeated as needed); and one
# of `vectorize(f)` itself mapped over a batch of 2, so that the rules also see
# the primitives as an inner `vectorize()` leaves them, e.g. with batching axes
# of their own. `extra = FALSE` skips these two.
#
# A random draw in `f` is checked against a loop over the slices from the
# random state the vectorized call started from, which `vectorize()` must draw
# what it draws (`randomness = "independent"`).
#
# `f` only takes arrays: bind the parameters a primitive takes in a closure.
# Every primitive whose vectorize rule runs is recorded in `vectorize_tested`,
# which the end of test-primitives-vectorize.R checks against the primitives
# with a rule.
autotest_vectorize <- function(f, ..., tolerance = 1e-6, extra = TRUE) {
  args <- list(...)
  checkmate::assert_list(args, min.len = 1L, names = "unique")
  size <- unique(vapply(args, batch_size, integer(1L)))
  if (length(size) != 1L) {
    stop("All arguments must have the same size along the last axis.")
  }
  apply_rule <- apply_vectorize_rule
  testthat::local_mocked_bindings(apply_vectorize_rule = function(primitive, ...) {
    vectorize_tested[[primitive$name]] <- TRUE
    apply_rule(primitive, ...)
  })
  subsets <- unlist(
    lapply(seq_along(args), function(k) utils::combn(names(args), k, simplify = FALSE)),
    recursive = FALSE
  )
  check_vectorize(f, args, subsets, tolerance)
  if (!extra) {
    return(invisible(NULL))
  }

  other_size <- sample(c(1L, 7L), 1L)
  resized <- lapply(args, take_batch, idx = rep_len(seq_len(size), other_size))
  check_vectorize(f, resized, list(names(args)), tolerance)

  f_vec <- vectorize(f)
  inner <- function() do.call(f_vec, as.list(environment()))
  formals(inner) <- stats::setNames(rep(list(quote(expr = )), length(args)), names(args))
  rotated <- c(seq_len(size)[-1L], 1L)
  outer <- lapply(args, function(x) stack_arrays(x, take_batch(x, rotated)))
  check_vectorize(inner, outer, list(names(args)), tolerance)
  invisible(NULL)
}

# The check of `autotest_vectorize()` for the mapped subsets `subsets` of the
# arguments `args`.
check_vectorize <- function(f, args, subsets, tolerance) {
  size <- batch_size(args[[1L]])
  slice_avals <- lapply(args, function(x) nv_aval(dtype(x), slice_shape(x)))
  slice_graph <- trace_fn(f, slice_avals)
  prims <- unique(vapply(slice_graph$statements, \(s) s$primitive$name, character(1L)))
  out_naxes <- vapply(slice_graph$outputs, \(out) length(shape(out$aval)), integer(1L))

  f_jit <- jit(f)
  for (mapped in subsets) {
    in_axes <- lapply(stats::setNames(nm = mapped), function(nm) sample.int(length(shape(args[[nm]])), 1L))
    out_axes <- lapply(out_naxes, function(n) sample.int(n + 1L, 1L))
    # the axes as `vectorize()` is given them, each counted from the end or not
    in_names <- lapply(stats::setNames(nm = mapped), function(nm) {
      from_end(in_axes[[nm]], length(shape(args[[nm]])))
    })
    out_names <- Map(from_end, out_axes, out_naxes + 1L)
    call_args <- args
    for (nm in setdiff(names(args), mapped)) {
      call_args[[nm]] <- slice_batch(args[[nm]], 1L)
    }
    vec_args <- call_args
    for (nm in mapped) {
      vec_args[[nm]] <- move_batch_axis(args[[nm]], in_axes[[nm]])
    }
    f_vec <- vectorize(f, in_axes = in_names, out_axes = unflatten(slice_graph$out_tree, out_names))
    rng_state <- if (length(slice_graph$side_effects)) nv_get_rng_state()
    actual <- flatten(do.call(jit(f_vec), vec_args))
    if (!is.null(rng_state)) {
      nv_set_rng_state(rng_state)
    }
    slices <- lapply(seq_len(size), function(i) {
      slice_args <- call_args
      for (nm in mapped) {
        slice_args[[nm]] <- slice_batch(args[[nm]], i)
      }
      flatten(do.call(f_jit, slice_args))
    })
    for (j in seq_along(actual)) {
      axis <- out_axes[[j]]
      info <- sprintf(
        "%s, in_axes: %s, output %d at axis %d",
        paste0("prim_", prims, collapse = ", "),
        paste(names(in_names), unlist(in_names), sep = " = ", collapse = ", "),
        j,
        out_names[[j]]
      )
      testthat::expect_equal(dtype(actual[[j]]), dtype(slices[[1L]][[j]]), info = info)
      testthat::expect_equal(
        shape(actual[[j]]),
        append(shape(slices[[1L]][[j]]), size, after = axis - 1L),
        info = info
      )
      expected <- stack_batch(lapply(slices, function(out) as_r(out[[j]])), axis)
      testthat::expect_equal(as_r(actual[[j]]), expected, tolerance = tolerance, info = info)
    }
  }
  invisible(NULL)
}

vectorize_tested <- new.env()

# An array as plain R data of its shape: a vector for a scalar, an array
# otherwise. A 64-bit integer or `ui32` array comes back as `integer64`, which
# `as.vector()` would turn into garbage, so it is read as doubles, which hold its
# values exactly up to 2^53. A `ui64` array, such as random bits, spans the whole
# range instead, so it is read as strings of the `integer64` its values wrap to,
# which keep them apart exactly.
as_r <- function(x) {
  v <- if (dtype(x) == "ui64") {
    as.character(as_array(x, check = FALSE))
  } else {
    a <- as_array(x)
    if (inherits(a, "integer64")) as.double(a) else as.vector(a)
  }
  s <- shape(x)
  if (length(s)) array(v, dim = s) else v
}

# The size of the batch axis of `x`, its last.
batch_size <- function(x) {
  s <- shape(x)
  s[[length(s)]]
}

# The shape of one slice of `x`, without its batch axis.
slice_shape <- function(x) {
  s <- shape(x)
  s[-length(s)]
}

# The axis `axis` of `n` axes, at random as it is or counted from the end.
from_end <- function(axis, n) {
  if (stats::runif(1L) < 0.5) axis else axis - n - 1L
}

# The slices `idx` of the last axis of `x`, as an array of the same data type.
take_batch <- function(x, idx) {
  a <- array(as_r(x), dim = shape(x))
  n <- length(shape(x))
  sub <- do.call(`[`, c(list(a), rep(list(TRUE), n - 1L), list(idx), list(drop = FALSE)))
  nv_array(as.vector(sub), shape = c(slice_shape(x), length(idx)), dtype = dtype(x))
}

# Arrays `x` and `y` of one shape, stacked along a new last axis.
stack_arrays <- function(x, y) {
  stacked <- stack_batch(list(as_r(x), as_r(y)), length(shape(x)) + 1L)
  nv_array(as.vector(stacked), shape = c(shape(x), 2L), dtype = dtype(x))
}

# Slice `i` of the last axis of `x`, as an array of the same data type.
slice_batch <- function(x, i) {
  a <- array(as_r(x), dim = shape(x))
  n <- length(shape(x))
  if (n == 1L) {
    return(nv_scalar(a[[i]], dtype = dtype(x)))
  }
  sub <- do.call(`[`, c(list(a), rep(list(TRUE), n - 1L), list(i), list(drop = FALSE)))
  nv_array(as.vector(sub), shape = slice_shape(x), dtype = dtype(x))
}

# Stacks plain R arrays of one shape along a new axis `axis`.
stack_batch <- function(xs, axis) {
  s <- dim(xs[[1L]]) %||% integer()
  v <- unlist(lapply(xs, as.vector))
  if (!length(s)) {
    return(array(v, dim = length(xs)))
  }
  n <- length(s) + 1L
  aperm(array(v, dim = c(s, length(xs))), append(seq_along(s), n, after = axis - 1L))
}

# `x` with its last axis moved to `axis`, as an array of the same data type.
move_batch_axis <- function(x, axis) {
  n <- length(shape(x))
  if (axis == n) {
    return(x)
  }
  perm <- append(seq_len(n - 1L), n, after = axis - 1L)
  nv_array(as.vector(aperm(as_r(x), perm)), shape = shape(x)[perm], dtype = dtype(x))
}

# An array of shape `shape` with values drawn by `sample(n)`.
rand_array <- function(shape, dtype = "f32", sample = stats::rnorm) {
  nv_array(sample(prod(shape)), shape = shape, dtype = dtype)
}

# An `i32` array of shape `shape` of indices into an axis of size `n`.
rand_indices <- function(shape, n) {
  rand_array(shape, dtype = "i32", sample = function(k) sample.int(n, k, replace = TRUE))
}
