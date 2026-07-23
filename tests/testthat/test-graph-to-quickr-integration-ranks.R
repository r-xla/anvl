test_that("integration: `%*%` matches PJRT for array ranks 1..5", {
  skip_if_no_quickr_or_pjrt()

  withr::local_seed(4)

  m <- 3L
  n <- 4L
  p <- 5L
  batch_dim <- 2L

  matmul_any_rank <- function(A, B) {
    if (naxes(A) == 1L) {
      A <- nv_reshape(A, c(1L, shape(A)))
    }
    if (naxes(B) == 1L) {
      B <- nv_reshape(B, c(shape(B), 1L))
    }

    pad_and_broadcast <- function(x, target_shape) {
      if (identical(shape(x), target_shape)) {
        return(x)
      }
      if (naxes(x) < length(target_shape)) {
        x <- nv_reshape(x, c(rep.int(1L, length(target_shape) - naxes(x)), shape(x)))
      }
      prim_broadcast_in_dim(x, shape = target_shape, broadcast_dimensions = seq_along(target_shape))
    }

    shA <- shape(A)
    shB <- shape(B)
    rA <- length(shA)
    rB <- length(shB)
    target_batch <- if (rA >= rB) {
      shA[seq_len(rA - 2L)]
    } else {
      shB[seq_len(rB - 2L)]
    }

    A <- pad_and_broadcast(A, c(target_batch, shA[(rA - 1L):rA]))
    B <- pad_and_broadcast(B, c(target_batch, shB[(rB - 1L):rB]))

    A %*% B
  }

  template_tensor <- function(x) {
    shp <- dim(x)
    if (is.null(shp)) {
      shp <- c(length(x))
    }
    nv_array(x, dtype = "f64", shape = shp)
  }

  make_input <- function(rank, side) {
    if (rank == 1L) {
      return(rnorm(n, sd = 0.2))
    }
    leading <- rep.int(batch_dim, rank - 2L)
    if (side == "lhs") {
      dims <- c(leading, m, n)
    } else {
      dims <- c(leading, n, p)
    }
    array(rnorm(prod(dims), sd = 0.2), dim = dims)
  }

  rank_pairs <- list(
    c(1L, 1L),
    c(1L, 2L),
    c(2L, 1L),
    c(2L, 2L),
    c(3L, 5L),
    c(5L, 3L)
  )

  for (pair in rank_pairs) {
    a_rank <- pair[[1L]]
    b_rank <- pair[[2L]]

    A <- make_input(a_rank, "lhs")
    B <- make_input(b_rank, "rhs")

    templates <- list(A = template_tensor(A), B = template_tensor(B))
    run <- list(
      args = list(A = A, B = B),
      info = paste0("a_rank=", a_rank, ", b_rank=", b_rank)
    )
    expect_quickr_matches_pjrt_fn(matmul_any_rank, templates, list(run), tolerance = 1e-12)
  }
})
