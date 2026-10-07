# Checks `vectorize()` on `f` against applying `f` to one slice at a time.
#
# `...` are the arguments of `f`, each an array whose first axis is the batch
# axis; they must agree on its size. For every non-empty subset of them,
# `vectorize(f, args = <subset>)` is called with the arguments of the subset in
# full and the others at their first slice, and each of its outputs must have
# the data type of `f`'s and equal the results of `f` on the slices, stacked
# along the first axis.
#
# With `axis`, the batch axis of the arguments is moved to `axis` first, and the
# outputs are stacked along `axis` instead, so that every argument mapped over
# and every output must have that many axes.
#
# `f` only takes arrays: bind the parameters a primitive takes in a closure.
# Every primitive whose vectorize rule runs is recorded in `vectorize_tested`,
# which the end of test-primitives-vectorize.R checks against the primitives
# with a rule.
autotest_vectorize <- function(f, ..., axis = 1L, tolerance = 1e-6) {
  args <- list(...)
  checkmate::assert_list(args, min.len = 1L, names = "unique")
  size <- unique(vapply(args, function(x) shape(x)[[1L]], integer(1L)))
  if (length(size) != 1L) {
    stop("All arguments must have the same size along the first axis.")
  }
  slice_avals <- lapply(args, function(x) nv_aval(dtype(x), shape(x)[-1L]))
  prims <- unique(vapply(trace_fn(f, slice_avals)$statements, \(s) s$primitive$name, character(1L)))
  apply_rule <- apply_vectorize_rule
  testthat::local_mocked_bindings(apply_vectorize_rule = function(primitive, ...) {
    vectorize_tested[[primitive$name]] <- TRUE
    apply_rule(primitive, ...)
  })

  f_jit <- jit(f)
  subsets <- unlist(
    lapply(seq_along(args), function(k) utils::combn(names(args), k, simplify = FALSE)),
    recursive = FALSE
  )
  for (mapped in subsets) {
    call_args <- args
    for (nm in setdiff(names(args), mapped)) {
      call_args[[nm]] <- slice_batch(args[[nm]], 1L)
    }
    vec_args <- call_args
    for (nm in mapped) {
      vec_args[[nm]] <- move_batch_axis(args[[nm]], axis)
    }
    actual <- flatten(do.call(jit(vectorize(f, args = mapped, axis = axis)), vec_args))
    slices <- lapply(seq_len(size), function(i) {
      slice_args <- call_args
      for (nm in mapped) {
        slice_args[[nm]] <- slice_batch(args[[nm]], i)
      }
      flatten(do.call(f_jit, slice_args))
    })
    for (j in seq_along(actual)) {
      info <- sprintf(
        "%s, mapped over: %s, output %d",
        paste0("prim_", prims, collapse = ", "),
        paste(mapped, collapse = ", "),
        j
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
# otherwise. A 64-bit or `ui32` array comes back as `integer64`, which
# `as.vector()` would turn into garbage, so it is read as doubles, which hold its
# values exactly up to 2^53.
as_r <- function(x) {
  a <- as_array(x)
  v <- if (inherits(a, "integer64")) as.double(a) else as.vector(a)
  s <- shape(x)
  if (length(s)) array(v, dim = s) else v
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
