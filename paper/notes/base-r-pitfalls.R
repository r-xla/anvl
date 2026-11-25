# Pitfalls of base R's array programming API, and how anvl handles them.
#
# Personal notes, not part of the paper. Each section shows a base R operation that is hard to
# express or gives a surprising result, followed by the anvl counterpart.
# Run with `Rscript base-r-pitfalls.R`; uses the current anvl source via pkgload::load_all().

suppressMessages(pkgload::load_all("/Users/sebi/r-xla/anvl-paper", quiet = TRUE))

section <- function(title) cat("\n", strrep("=", 78), "\n", title, "\n", strrep("=", 78), "\n", sep = "")
show <- function(label, expr) {
  cat("\n#", label, "\n")
  res <- tryCatch(
    withCallingHandlers(expr, warning = function(w) {
      cat("  [warning]", conditionMessage(w), "\n")
      invokeRestart("muffleWarning")
    }),
    error = function(e) structure(strsplit(conditionMessage(e), "\n")[[1L]][1L], class = "shown_error")
  )
  if (inherits(res, "shown_error")) cat("  [error]", unclass(res), "\n") else print(res)
}
a <- function(x) anvl::as_array(x)

m <- matrix(1:6, nrow = 2) # 2 x 3
M <- nv_array(m)

# ------------------------------------------------------------------------------------------------
section("1. Dropping axes depends on how many elements are selected")
rows <- c(1L, 2L)
show("base R: m[rows, ] with two rows is a matrix", dim(m[rows, ]))
rows <- 1L
show("base R: the same code with one row is a vector (dim NULL)", dim(m[rows, ]))
show("base R: the fix is drop = FALSE, easy to forget", dim(m[rows, , drop = FALSE]))
show("anvl: a scalar index drops the axis", shape(M[1L, ]))
show("anvl: an array index keeps it, whatever its length", shape(M[array(1L), ]))

# ------------------------------------------------------------------------------------------------
section("2. Negative indices with an empty index vector select nothing")
x <- c(10, 20, 30)
drop_idx <- which(x > 100) # integer(0): nothing to remove
show("base R: x[-drop_idx] is meant to remove nothing but returns an empty vector", x[-drop_idx])
show("anvl: negative indices are not supported, so this cannot happen silently", nv_array(x)[-1L])
show("anvl: keep elements by masking instead", a(nv_array(x)[nv_array(x <= 100)]))

# ------------------------------------------------------------------------------------------------
section("3. Recycling instead of broadcasting")
show("base R: vectors of different lengths are recycled silently if the lengths divide", c(1, 2, 3, 4) + c(10, 20))
show("base R: ... and with only a warning if they don't", c(1, 2, 3) + c(10, 20))
show("base R: a matrix and a vector recycle over the column-major elements", m + c(10, 20, 30))
show("anvl: arrays of different shapes are an error", nv_array(c(1, 2, 3, 4)) + nv_array(c(10, 20)))
show("anvl: scalars broadcast automatically", a(M * 2L))

# ------------------------------------------------------------------------------------------------
section("4. No broadcasting between arrays")
row_vec <- matrix(c(10L, 20L, 30L), nrow = 1) # 1 x 3
show("base R: adding a 1 x 3 matrix to a 2 x 3 matrix is an error", m + row_vec)
show("base R: needs sweep() or rep()", sweep(m, 2L, c(10L, 20L, 30L), `+`))
show("anvl: explicit broadcasting along size-1 axes", {
  xs <- nv_broadcast_arrays(M, nv_array(row_vec))
  a(xs[[1L]] + xs[[2L]])
})

# ------------------------------------------------------------------------------------------------
section("5. Functions that silently flatten matrices")
show("base R: cumsum() of a matrix flattens it and drops the dim", cumsum(m))
show("base R: sort() of a matrix flattens it", sort(matrix(c(3, 1, 2, 6, 5, 4), 2)))
show("base R: rev() of a matrix flattens it", rev(m))
show("anvl: cumulative sum along an axis, shape kept", a(nv_cumsum(M, axis = 2L)))
show("anvl: sort along an axis, shape kept", a(nv_sort(nv_array(matrix(c(3, 1, 2, 6, 5, 4), 2)), axis = 1L)))
show("anvl: reverse along chosen axes", a(nv_rev(M, axes = 2L)))

# ------------------------------------------------------------------------------------------------
section("6. Reductions over arbitrary axes of higher-rank arrays")
arr <- array(1:24, dim = c(2, 3, 4))
show("base R: rowSums()/colSums() only reduce leading/trailing blocks of axes", dim(colSums(arr)))
show("base R: other axes need apply(), which loops in R and permutes the result", dim(apply(arr, c(1L, 3L), sum)))
show("anvl: any set of axes, optionally keeping them", shape(nv_sum(nv_array(arr), axes = 2L)))
show("anvl: drop = FALSE keeps a size-1 axis", shape(nv_sum(nv_array(arr), axes = 2L, drop = FALSE)))

# ------------------------------------------------------------------------------------------------
section("7. Batched matrix products")
A <- array(rnorm(2 * 3 * 4), dim = c(2, 3, 4)) # four 2 x 3 matrices along axis 3
B <- array(rnorm(3 * 2 * 4), dim = c(3, 2, 4))
show("base R: %*% of two 3-d arrays silently returns the inner product of the flattened arrays", A %*% B)
show("base R: a batch needs a loop or sapply()", dim(simplify2array(lapply(1:4, function(k) A[, , k] %*% B[, , k]))))
show("anvl: shape of a batched product (batch along the leading axes)", {
  Ab <- nv_array(aperm(A, c(3L, 1L, 2L))) # 4 x 2 x 3
  Bb <- nv_array(aperm(B, c(3L, 1L, 2L))) # 4 x 3 x 2
  shape(Ab %*% Bb)
})

# ------------------------------------------------------------------------------------------------
section("8. Vectors versus matrices in %*%")
v <- c(1, 2, 3)
show("base R: v %*% v is a 1 x 1 matrix, not a scalar", v %*% v)
show("base R: m %*% v treats v as a column, v %*% t(m) as a row", dim(t(m) %*% c(1, 2)))
show("anvl: %*% needs at least 2 axes; vectors are not silently turned into rows or columns", nv_array(v) %*% nv_array(v))
show("anvl: an inner product is written as such", a(sum(nv_array(v) * nv_array(v))))

# ------------------------------------------------------------------------------------------------
section("9. Data types: no single precision, no 64-bit or unsigned integers, overflow to NA")
show("base R: integer overflow gives NA with a warning", .Machine$integer.max + 1L)
show("base R: there is no float32, int64, or uint8 storage type", typeof(1L))
show("anvl: explicit data types", vapply(
  list(nv_array(1, dtype = "f32"), nv_array(1L, dtype = "i64"), nv_array(1L, dtype = "ui8")),
  function(z) as.character(dtype(z)), character(1L)
))
show("anvl: an R value that a data type cannot represent is an error, not a wrap-around", nv_array(1L, dtype = "ui8") + (-2L))

# ------------------------------------------------------------------------------------------------
section("10. Length-one vectors change the meaning of sample() and seq()")
show("base R: sample(x) with length(x) == 1 samples from 1:x", {
  set.seed(1)
  x <- 5
  sample(x)
})
show("base R: 1:length(x) for an empty x is c(1, 0)", 1:length(integer(0)))
show("anvl: nv_sample() always samples from the given array (here with replacement)", {
  r <- nv_sample(shape = 3L, state = nv_rng_state(1L), x = nv_array(array(5)))
  a(r$values)
})

cat("\n")
