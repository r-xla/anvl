# Checks `vectorize()` on `f` against applying `f` to one slice at a time.
#
# `...` are the arguments of `f`, each an array whose first axis is the batch
# axis; they must agree on its size. For every non-empty subset of them,
# `vectorize(f, in_axes = , out_axes = )` is called with the arguments of the
# subset in full and the others at their first slice, and each of its outputs
# must have the data type of `f`'s and equal the results of `f` on the slices,
# stacked along the output's batch axis.
#
# The batch axes are drawn at random for every subset: each argument mapped
# over has its batch axis moved to a random axis of its own, named in
# `in_axes`, and each output is stacked along a random axis it has room for,
# passed as `out_axes` with the structure of what `f` returns. Call it under
# `withr::local_seed()`.
#
# Two more checks map over all arguments: one at a batch size of 1 or 7,
# drawn at random, made of the slices of `...` (repeated as needed); and one
# of `vectorize(f)` itself mapped over a batch of 2, so that the rules also see
# the primitives as an inner `vectorize()` leaves them, e.g. with batching axes
# of their own. `extra = FALSE` skips these two.
#
# `f` only takes arrays: bind the parameters a primitive takes in a closure.
# Every primitive whose vectorize rule runs is recorded in `vectorize_tested`,
# which the end of test-primitives-vectorize.R checks against the primitives
# with a rule.
autotest_vectorize <- function(f, ..., tolerance = 1e-6, extra = TRUE) {
  args <- list(...)
  checkmate::assert_list(args, min.len = 1L, names = "unique")
  size <- unique(vapply(args, function(x) shape(x)[[1L]], integer(1L)))
  if (length(size) != 1L) {
    stop("All arguments must have the same size along the first axis.")
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
  size <- shape(args[[1L]])[[1L]]
  slice_avals <- lapply(args, function(x) nv_aval(dtype(x), shape(x)[-1L]))
  slice_graph <- trace_fn(f, slice_avals)
  prims <- unique(vapply(slice_graph$statements, \(s) s$primitive$name, character(1L)))
  out_naxes <- vapply(slice_graph$outputs, \(out) length(shape(out$aval)), integer(1L))

  f_jit <- jit(f)
  for (mapped in subsets) {
    in_axes <- lapply(stats::setNames(nm = mapped), function(nm) sample.int(length(shape(args[[nm]])), 1L))
    out_axes <- lapply(out_naxes, function(n) sample.int(n + 1L, 1L))
    call_args <- args
    for (nm in setdiff(names(args), mapped)) {
      call_args[[nm]] <- slice_batch(args[[nm]], 1L)
    }
    vec_args <- call_args
    for (nm in mapped) {
      vec_args[[nm]] <- move_batch_axis(args[[nm]], in_axes[[nm]])
    }
    f_vec <- vectorize(f, in_axes = in_axes, out_axes = unflatten(slice_graph$out_tree, out_axes))
    actual <- flatten(do.call(jit(f_vec), vec_args))
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
        paste(names(in_axes), unlist(in_axes), sep = " = ", collapse = ", "),
        j,
        axis
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
# values exactly up to 2^53.
as_r <- function(x) {
  a <- as_array(x)
  v <- if (inherits(a, "integer64")) as.double(a) else as.vector(a)
  s <- shape(x)
  if (length(s)) array(v, dim = s) else v
}

# The slices `idx` of the first axis of `x`, as an array of the same data type.
take_batch <- function(x, idx) {
  a <- as_r(x)
  s <- shape(x)
  sub <- do.call(`[`, c(list(array(a, dim = s), idx), rep(list(TRUE), length(s) - 1L), list(drop = FALSE)))
  nv_array(as.vector(sub), shape = c(length(idx), s[-1L]), dtype = dtype(x))
}

# Arrays `x` and `y` of one shape, stacked along a new first axis.
stack_arrays <- function(x, y) {
  nv_array(as.vector(stack_batch(list(as_r(x), as_r(y)))), shape = c(2L, shape(x)), dtype = dtype(x))
}

# Slice `i` of the first axis of `x`, as an array of the same data type.
slice_batch <- function(x, i) {
  a <- as_r(x)
  s <- shape(x)
  if (length(s) == 1L) {
    return(nv_scalar(a[[i]], dtype = dtype(x)))
  }
  idx <- c(list(a, i), rep(list(TRUE), length(s) - 1L), list(drop = FALSE))
  nv_array(as.vector(do.call(`[`, idx)), shape = s[-1L], dtype = dtype(x))
}

# Stacks plain R arrays of one shape along a new axis `axis`.
stack_batch <- function(xs, axis = 1L) {
  s <- dim(xs[[1L]]) %||% integer()
  v <- unlist(lapply(xs, as.vector))
  if (!length(s)) {
    return(array(v, dim = length(xs)))
  }
  n <- length(s) + 1L
  aperm(array(v, dim = c(s, length(xs))), append(seq_along(s), n, after = axis - 1L))
}

# `x` with its first axis moved to `axis`, as an array of the same data type.
move_batch_axis <- function(x, axis) {
  if (axis == 1L) {
    return(x)
  }
  perm <- append(seq_along(shape(x))[-1L], 1L, after = axis - 1L)
  nv_array(as.vector(aperm(as_r(x), perm)), shape = shape(x)[perm], dtype = dtype(x))
}

# An array of shape `shape` with values drawn by `sample(n)`.
rand_array <- function(shape, dtype = "f32", sample = stats::rnorm) {
  nv_array(sample(prod(shape)), shape = shape, dtype = dtype)
}
