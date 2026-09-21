cumsum_body <- function(carry, x) {
  s <- carry + x
  list(carry = s, out = s)
}

describe("nv_scan", {
  it("matches nv_cumsum on a 1-D array", {
    x <- c(1, 2, 3, 4)
    res <- nv_scan(nv_scalar(0), cumsum_body, xs = nv_array(x))
    expect_equal(as.numeric(as.array(res$out)), cumsum(x))
    expect_equal(as.numeric(as.array(res$carry)), sum(x))
    expect_equal(
      as.numeric(as.array(res$out)),
      as.numeric(as.array(nv_cumsum(nv_array(x))))
    )
  })

  it("batches the carry over the non-scanned axes", {
    a <- array(as.numeric(1:24), dim = c(4L, 2L, 3L))
    res <- nv_scan(
      init = nv_fill(0, shape = c(2L, 3L), dtype = "f64"),
      body = cumsum_body,
      xs = nv_array(a)
    )
    expect_equal(as.array(res$out), apply(a, c(2, 3), cumsum))
    expect_equal(as.array(res$carry), apply(a, c(2, 3), sum))
  })

  it("computes a recursive EWMA matching stats::filter", {
    x <- c(2, 5, 1, 4, 3, 6)
    # R double literals materialise as f32 constants, so use explicit f64
    # scalars to keep the recursion in full double precision
    alpha <- nv_scalar(0.3, dtype = "f64")
    one_m_alpha <- nv_scalar(0.7, dtype = "f64")
    res <- nv_scan(
      init = nv_scalar(0, dtype = "f64"),
      body = function(carry, v) {
        y <- one_m_alpha * carry + alpha * v
        list(carry = y, out = y)
      },
      xs = nv_array(x, dtype = "f64")
    )
    ref <- as.numeric(stats::filter(0.3 * x, 0.7, method = "recursive"))
    expect_equal(as.numeric(as.array(res$out)), ref)
  })

  it("reverse scan reads and writes at the original positions", {
    x <- c(1, 2, 3, 4)
    res <- nv_scan(nv_scalar(0), cumsum_body, xs = nv_array(x), reverse = TRUE)
    expect_equal(as.numeric(as.array(res$out)), rev(cumsum(rev(x))))
    expect_equal(as.numeric(as.array(res$carry)), sum(x))
  })

  it("supports nested carries and multiple out leaves", {
    x <- c(3, 1, 4, 1, 5)
    res <- nv_scan(
      init = list(s = nv_scalar(0), m = nv_scalar(-Inf)),
      body = function(carry, v) {
        s <- carry$s + v
        m <- nv_max(carry$m, v)
        list(carry = list(s = s, m = m), out = list(sum = s, max = m))
      },
      xs = nv_array(x)
    )
    expect_named(res$out, c("sum", "max"))
    expect_equal(as.numeric(as.array(res$out$sum)), cumsum(x))
    expect_equal(as.numeric(as.array(res$out$max)), cummax(x))
    expect_equal(as.numeric(as.array(res$carry$m)), max(x))
  })

  it("slices multiple xs leaves in lockstep", {
    x <- c(1, 2, 3, 4)
    w <- c(10, 20, 30, 40)
    res <- nv_scan(
      init = nv_scalar(0),
      body = function(carry, v) {
        s <- carry + v$x * v$w
        list(carry = s, out = s)
      },
      xs = list(x = nv_array(x), w = nv_array(w))
    )
    expect_equal(as.numeric(as.array(res$out)), cumsum(x * w))
  })

  it("runs fori-style with xs = NULL and an explicit length", {
    res <- nv_scan(
      init = nv_scalar(1L),
      body = function(carry, x) {
        expect_null(x)
        list(carry = carry + 1L, out = carry * 2L)
      },
      length = 3L
    )
    expect_equal(as.numeric(as.array(res$out)), c(2, 4, 6))
    expect_equal(as.numeric(as.array(res$carry)), 4)
  })

  it("allows out = NULL (carry-only loop)", {
    res <- nv_scan(
      nv_scalar(0),
      function(carry, v) list(carry = carry + v, out = NULL),
      xs = nv_array(c(1, 2, 3, 4))
    )
    expect_null(res$out)
    expect_equal(as.numeric(as.array(res$carry)), 10)
  })

  it("handles length 1", {
    res <- nv_scan(nv_scalar(0), cumsum_body, xs = nv_array(7))
    expect_equal(as.numeric(as.array(res$out)), 7)
    expect_equal(as.numeric(as.array(res$carry)), 7)
  })

  it("nests: a scan inside a scan body", {
    # row-wise cumulative sums, then a running total of the row totals
    m <- matrix(as.numeric(1:6), 2L, 3L)
    res <- nv_scan(
      init = nv_scalar(0),
      body = function(carry, row) {
        inner <- nv_scan(nv_scalar(0), cumsum_body, xs = row)
        total <- carry + inner$carry
        list(carry = total, out = inner$out)
      },
      xs = nv_array(m)
    )
    expect_equal(as.array(res$out), t(apply(m, 1L, cumsum)))
    expect_equal(as.numeric(as.array(res$carry)), sum(m))
  })

  it("works under jit", {
    f <- jit(function(x) nv_scan(nv_scalar(0), cumsum_body, xs = x)$out)
    out <- f(nv_array(c(1, 2, 3, 4)))
    expect_equal(as.numeric(as.array(out)), cumsum(c(1, 2, 3, 4)))
  })

  it("stacks boolean and integer outputs", {
    x <- nv_array(c(3L, -1L, 4L, -1L, 5L), dtype = "i32")
    res <- nv_scan(
      init = nv_scalar(0L, dtype = "i32"),
      body = function(carry, v) {
        s <- carry + v
        list(carry = s, out = list(pos = v > 0L, sum = s))
      },
      xs = x
    )
    expect_equal(dtype(res$out$pos), as_dtype("bool"))
    expect_equal(dtype(res$out$sum), as_dtype("i32"))
    expect_equal(as.logical(as.array(res$out$pos)), c(TRUE, FALSE, TRUE, FALSE, TRUE))
    expect_equal(as.integer(as.array(res$out$sum)), cumsum(c(3L, -1L, 4L, -1L, 5L)))
  })

  it("validates its arguments", {
    x <- nv_array(c(1, 2, 3, 4))
    expect_error(nv_scan(nv_scalar(0), body = "not a function", xs = x), "must be a function")
    expect_error(nv_scan(nv_scalar(0), cumsum_body, xs = x, reverse = NA), "TRUE or FALSE")
    expect_error(nv_scan(nv_scalar(0), cumsum_body, xs = list()), "at least one array")
    expect_error(nv_scan(nv_scalar(0), cumsum_body, length = 0L), "positive integer")
  })

  it("errors clearly on contract violations", {
    x <- nv_array(c(1, 2, 3, 4))
    expect_error(
      nv_scan(nv_scalar(0), function(c, v) c + v, xs = x),
      "list\\(carry = , out = \\)"
    )
    expect_error(
      nv_scan(nv_scalar(0), function(c, v) list(carry = list(c), out = c), xs = x),
      "same structure as `init`"
    )
    expect_error(
      nv_scan(nv_scalar(0), cumsum_body, xs = x, length = 9L),
      "disagrees with axis 1"
    )
    expect_error(
      nv_scan(nv_scalar(0), cumsum_body),
      "`length` is required"
    )
    expect_error(
      nv_scan(nv_scalar(0), cumsum_body, xs = nv_scalar(1)),
      "at least one axis"
    )
    expect_error(
      nv_scan(
        nv_scalar(0),
        function(c, v) list(carry = c, out = c),
        xs = list(a = nv_array(c(1, 2)), b = nv_array(c(1, 2, 3)))
      ),
      "agree on the size of axis 1"
    )
  })
})

test_that("quickr pipeline matches PJRT: nv_scan", {
  skip_if_no_quickr_or_pjrt()

  scan_ops <- function(x1, x3) {
    zero <- nv_scalar(0, dtype = "f64")
    cs <- nv_scan(zero, cumsum_body, xs = x1)
    rs <- nv_scan(zero, cumsum_body, xs = x1, reverse = TRUE)
    a1 <- nv_scalar(0.7, dtype = "f64")
    a2 <- nv_scalar(0.3, dtype = "f64")
    ew <- nv_scan(
      init = zero,
      body = function(carry, v) {
        y <- a1 * carry + a2 * v
        list(carry = y, out = y)
      },
      xs = x1
    )
    b <- nv_scan(
      init = nv_fill(0, shape = c(2L, 3L), dtype = "f64"),
      body = cumsum_body,
      xs = x3
    )
    list(cs = cs$out, cs_carry = cs$carry, rs = rs$out, ew = ew$out, b = b$out)
  }

  templates <- list(
    x1 = nv_array(rep(0, 4), dtype = "f64"),
    x3 = nv_array(array(0, dim = c(4L, 2L, 3L)), dtype = "f64")
  )
  run <- list(
    args = list(
      x1 = c(1, 2, 3, 4),
      x3 = array(as.numeric(1:24), dim = c(4L, 2L, 3L))
    ),
    info = "scan run"
  )

  expect_quickr_matches_pjrt_fn(scan_ops, templates, list(run))
})
