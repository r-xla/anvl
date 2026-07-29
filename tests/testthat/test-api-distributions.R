describe("nv_dnorm", {
  it("matches base R dnorm() with default mean/sd", {
    x <- c(-2, -1, 0, 0.5, 1, 2)
    expect_equal(
      as.vector(nv_dnorm(nv_array(x))),
      dnorm(x),
      tolerance = 1e-6
    )
  })

  it("matches base R dnorm() with custom mean/sd", {
    x <- c(-2, -1, 0, 0.5, 1, 2)
    expect_equal(
      as.vector(nv_dnorm(nv_array(x), mean = 1, sd = 2)),
      dnorm(x, mean = 1, sd = 2),
      tolerance = 1e-6
    )
  })

  it("log = TRUE matches base R dnorm(..., log = TRUE)", {
    x <- c(-2, -1, 0, 0.5, 1, 2)
    expect_equal(
      as.vector(nv_dnorm(nv_array(x), log = TRUE)),
      dnorm(x, log = TRUE),
      tolerance = 1e-6
    )
  })

  it("log = TRUE stays finite where the plain density underflows to 0", {
    x <- nv_array(40)
    expect_equal(as.vector(nv_dnorm(x)), 0)
    expect_equal(
      as.vector(nv_dnorm(x, log = TRUE)),
      dnorm(40, log = TRUE),
      tolerance = 1e-6
    )
  })

  it("non-scalar mean/sd works", {
    x <- c(0, 0, 0)
    mean <- c(-1, 0, 1)
    sd <- c(1, 2, 3)
    expect_equal(
      as.vector(nv_dnorm(
        nv_array(x),
        mean = nv_array(mean),
        sd = nv_array(sd)
      )),
      dnorm(x, mean = mean, sd = sd),
      tolerance = 1e-6
    )
  })

  it("converts mean/sd to the dtype of x", {
    out <- nv_dnorm(nv_array(c(0, 1), dtype = "f32"), mean = 0L, sd = 1L)
    expect_equal(dtype(out), as_dtype("f32"))
  })

  it("works under jit with log as a static argument", {
    x <- c(-1, 0, 1)
    expect_equal(
      as.vector(nv_dnorm(nv_array(x))),
      dnorm(x),
      tolerance = 1e-6
    )
    expect_equal(
      as.vector(nv_dnorm(nv_array(x), log = TRUE)),
      dnorm(x, log = TRUE),
      tolerance = 1e-6
    )
  })
})

describe("nv_pnorm", {
  it("matches base R pnorm() with default mean/sd", {
    q <- c(-2, -1, 0, 0.5, 1, 2)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q))),
      pnorm(q),
      tolerance = 1e-6
    )
  })

  it("matches base R pnorm() with custom mean/sd", {
    q <- c(-2, -1, 0, 0.5, 1, 2)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q), mean = 1, sd = 2)),
      pnorm(q, mean = 1, sd = 2),
      tolerance = 1e-6
    )
  })

  it("lower_tail = FALSE matches base R pnorm(..., lower.tail = FALSE)", {
    q <- c(-2, -1, 0, 0.5, 1, 2)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q), lower_tail = FALSE)),
      pnorm(q, lower.tail = FALSE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE matches base R pnorm(..., log.p = TRUE) around the direct/asymptotic threshold", {
    q <- c(-2, -1, 0, 0.5, 1, 2, -15, -20, -25)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q, dtype = "f64"), log_p = TRUE)),
      pnorm(q, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE and lower_tail = FALSE compose correctly, including via the asymptotic branch", {
    q <- c(-2, -1, 0, 0.5, 1, 2, 25)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q, dtype = "f64"), lower_tail = FALSE, log_p = TRUE)),
      pnorm(q, lower.tail = FALSE, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE stays finite deep in the tail where erfc() underflows to 0", {
    q <- nv_array(-40, dtype = "f64")
    expect_equal(as.vector(nv_pnorm(q)), 0)
    expect_equal(
      as.vector(nv_pnorm(q, log_p = TRUE)),
      pnorm(-40, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("gradient stays finite deep in the tail (asymptotic branch doesn't poison it via nv_ifelse)", {
    f <- function(q) nv_pnorm(q, log_p = TRUE)
    g <- as.vector(jit(gradient(f, wrt = "q"))(nv_scalar(-40, dtype = "f64"))[[1L]])
    expect_true(is.finite(g))
  })

  it("log_p = TRUE has a dtype-aware lower threshold, closing the f32 gap between where erfc() underflows and a fixed f64 threshold would kick in", {
    q <- c(-11, -13, -15, -17, -19)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q, dtype = "f32"), log_p = TRUE)),
      pnorm(q, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE stays accurate far in the upper tail (probability close to 1), in both f32 and f64", {
    q32 <- c(6, 10, 12.9)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q32, dtype = "f32"), log_p = TRUE)),
      pnorm(q32, log.p = TRUE),
      tolerance = 1e-5
    )
    q64 <- c(9, 20, 40, 100)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q64, dtype = "f64"), log_p = TRUE)),
      pnorm(q64, log.p = TRUE),
      tolerance = 1e-5
    )
  })

  it("log_p = TRUE upper tail composes correctly with lower_tail = FALSE (mirrors the lower tail)", {
    q <- c(-9, -20, -40)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q, dtype = "f64"), lower_tail = FALSE, log_p = TRUE)),
      pnorm(q, lower.tail = FALSE, log.p = TRUE),
      tolerance = 1e-5
    )
  })

  it("gradient stays finite far in the upper tail (upper branch doesn't poison it via nv_ifelse)", {
    f <- function(q) nv_pnorm(q, log_p = TRUE)
    g <- as.vector(jit(gradient(f, wrt = "q"))(nv_scalar(40, dtype = "f64"))[[1L]])
    expect_true(is.finite(g))
  })

  it("non-scalar mean/sd works", {
    q <- c(0, 0, 0)
    mean <- c(-1, 0, 1)
    sd <- c(1, 2, 3)
    expect_equal(
      as.vector(nv_pnorm(
        nv_array(q),
        mean = nv_array(mean),
        sd = nv_array(sd)
      )),
      pnorm(q, mean = mean, sd = sd),
      tolerance = 1e-6
    )
  })

  it("converts mean/sd to the dtype of q", {
    out <- nv_pnorm(nv_array(c(0, 1), dtype = "f32"), mean = 0L, sd = 1L)
    expect_equal(dtype(out), as_dtype("f32"))
  })
})

describe("nv_qnorm", {
  it("matches base R qnorm() with default mean/sd", {
    p <- c(0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999)
    expect_equal(
      as.vector(nv_qnorm(nv_array(p))),
      qnorm(p),
      tolerance = 1e-6
    )
  })

  it("matches base R qnorm() with custom mean/sd", {
    p <- c(0.001, 0.025, 0.5, 0.975)
    expect_equal(
      as.vector(nv_qnorm(nv_array(p), mean = 1, sd = 2)),
      qnorm(p, mean = 1, sd = 2),
      tolerance = 1e-6
    )
  })

  it("lower_tail = FALSE matches base R qnorm(..., lower.tail = FALSE)", {
    p <- c(0.001, 0.025, 0.5, 0.975)
    expect_equal(
      as.vector(nv_qnorm(nv_array(p), lower_tail = FALSE)),
      qnorm(p, lower.tail = FALSE),
      tolerance = 1e-6
    )
  })

  it("matches base R across both rational regimes and their crossover", {
    # exp(-2) is the central <-> tail threshold and exp(-32) the near <-> far
    # tail one; the near-1 values exercise the upper reflection.
    p <- c(1e-300, exp(-32), 1e-5, exp(-2), 0.3, 1 - exp(-2), 1 - 1e-10)
    expect_equal(
      as.vector(nv_qnorm(nv_array(p, dtype = "f64"))),
      qnorm(p),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE matches base R qnorm(..., log.p = TRUE)", {
    lp <- c(-729, -100, -32, -25, -2, -0.7, -0.1, -1e-10)
    expect_equal(
      as.vector(nv_qnorm(nv_array(lp, dtype = "f64"), log_p = TRUE)),
      qnorm(lp, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE reaches quantiles a bare probability cannot express", {
    expect_equal(as.vector(nv_qnorm(nv_array(0, dtype = "f64"))), -Inf)
    expect_equal(
      as.vector(nv_qnorm(nv_array(-1000, dtype = "f64"), log_p = TRUE)),
      qnorm(-1000, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE and lower_tail = FALSE compose correctly", {
    lp <- c(-100, -2, -0.7, -0.1)
    expect_equal(
      as.vector(nv_qnorm(
        nv_array(lp, dtype = "f64"),
        lower_tail = FALSE,
        log_p = TRUE
      )),
      qnorm(lp, lower.tail = FALSE, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("returns the infinite boundaries and NaN outside [0, 1]", {
    p <- c(0, 1, -0.25, 1.25, NaN)
    expect_equal(
      as.vector(nv_qnorm(nv_array(p, dtype = "f64"))),
      c(-Inf, Inf, NaN, NaN, NaN)
    )
    lp <- c(-Inf, 0, 0.5, NaN)
    expect_equal(
      as.vector(nv_qnorm(nv_array(lp, dtype = "f64"), log_p = TRUE)),
      c(-Inf, Inf, NaN, NaN)
    )
  })

  it("gradient matches 1 / dnorm(qnorm(p))", {
    p <- c(0.001, 0.025, 0.1, 0.5, 0.9)
    f <- function(p) nv_reduce_sum(nv_qnorm(p))
    g <- as.vector(jit(gradient(f, wrt = "p"))(nv_array(p, dtype = "f64"))[[1L]])
    expect_equal(g, 1 / dnorm(qnorm(p)), tolerance = 1e-6)
  })

  it("gradient is not halved at the central/tail threshold", {
    f <- function(p) nv_reduce_sum(nv_qnorm(p))
    g <- as.vector(jit(gradient(f, wrt = "p"))(
      nv_array(exp(-2), dtype = "f64")
    )[[1L]])
    expect_equal(g, 1 / dnorm(qnorm(exp(-2))), tolerance = 1e-6)

    flog <- function(p) nv_reduce_sum(nv_qnorm(p, log_p = TRUE))
    glog <- as.vector(jit(gradient(flog, wrt = "p"))(
      nv_array(-2, dtype = "f64")
    )[[1L]])
    expect_equal(
      glog,
      exp(-2) / dnorm(qnorm(-2, log.p = TRUE)),
      tolerance = 1e-6
    )
  })

  it("gradient stays finite deep in the log tail", {
    f <- function(p) nv_reduce_sum(nv_qnorm(p, log_p = TRUE))
    g <- as.vector(jit(gradient(f, wrt = "p"))(
      nv_array(c(-1e4, -1e5), dtype = "f64")
    )[[1L]])
    expect_true(all(is.finite(g)))
  })

  it("gradients wrt mean/sd are exact", {
    p <- c(0.025, 0.9)
    f <- function(p, mean, sd) nv_reduce_sum(nv_qnorm(p, mean, sd))
    g <- jit(gradient(f, wrt = c("mean", "sd")))(
      nv_array(p, dtype = "f64"),
      nv_array(c(1, 1), dtype = "f64"),
      nv_array(c(2, 2), dtype = "f64")
    )
    expect_equal(as.vector(g[[1L]]), c(1, 1))
    expect_equal(as.vector(g[[2L]]), qnorm(p), tolerance = 1e-6)
  })

  it("inverts nv_pnorm", {
    x <- c(-4, -1, 0, 1, 4)
    expect_equal(
      as.vector(nv_qnorm(nv_pnorm(nv_array(x, dtype = "f64")))),
      x,
      tolerance = 1e-6
    )
  })

  it("non-scalar mean/sd works", {
    p <- c(0.1, 0.5, 0.9)
    mean <- c(-1, 0, 1)
    sd <- c(1, 2, 3)
    expect_equal(
      as.vector(nv_qnorm(
        nv_array(p),
        mean = nv_array(mean),
        sd = nv_array(sd)
      )),
      qnorm(p, mean = mean, sd = sd),
      tolerance = 1e-6
    )
  })

  it("converts mean/sd to the dtype of p", {
    out <- nv_qnorm(nv_array(c(0.25, 0.75), dtype = "f32"), mean = 0L, sd = 1L)
    expect_equal(dtype(out), as_dtype("f32"))
  })
})
