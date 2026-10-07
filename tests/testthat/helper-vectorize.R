# Checks `vectorize()` on `f` against applying `f` to one slice at a time.
#
# `...` are the arguments of `f`, each an array whose first axis is the batch
# axis; they must agree on its size. For every non-empty subset of them,
# `vectorize(f, args = <subset>)` is called with the arguments of the subset in
# full and the others at their first slice, and each of its outputs must equal
# the results of `f` on the slices, stacked along the first axis.
#
# `f` only takes arrays: bind the parameters a primitive takes in a closure.
autotest_vectorize <- function(f, ..., tolerance = 1e-6) {
  args <- list(...)
  checkmate::assert_list(args, min.len = 1L, names = "unique")
  size <- unique(vapply(args, function(x) shape(x)[[1L]], integer(1L)))
  if (length(size) != 1L) {
    stop("All arguments must have the same size along the first axis.")
  }
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
    actual <- flatten(do.call(jit(vectorize(f, args = mapped)), call_args))
    slices <- lapply(seq_len(size), function(i) {
      slice_args <- call_args
      for (nm in mapped) {
        slice_args[[nm]] <- slice_batch(args[[nm]], i)
      }
      flatten(do.call(f_jit, slice_args))
    })
    for (j in seq_along(actual)) {
      expected <- stack_batch(lapply(slices, function(out) as_r(out[[j]])))
      testthat::expect_equal(
        as_r(actual[[j]]),
        expected,
        tolerance = tolerance,
        info = sprintf("mapped over: %s, output %d", paste(mapped, collapse = ", "), j)
      )
    }
  }
  invisible(NULL)
}

# An array as plain R data of its shape: a vector for a scalar, an array
# otherwise. An `i64` array comes back as `integer64`, which `as.vector()` would
# turn into garbage, so it is read as integers.
as_r <- function(x) {
  a <- as_array(x)
  v <- if (inherits(a, "integer64")) as.integer(a) else as.vector(a)
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

# Stacks plain R arrays of one shape along a new first axis.
stack_batch <- function(xs) {
  s <- dim(xs[[1L]]) %||% integer()
  v <- unlist(lapply(xs, as.vector))
  if (!length(s)) {
    return(array(v, dim = length(xs)))
  }
  aperm(array(v, dim = c(s, length(xs))), c(length(s) + 1L, seq_along(s)))
}

# An array of shape `shape` with values drawn by `sample(n)`.
rand_array <- function(shape, dtype = "f32", sample = stats::rnorm) {
  nv_array(sample(prod(shape)), shape = shape, dtype = dtype)
}
