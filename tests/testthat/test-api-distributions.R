# Limit configurations exercised against base R by the nv_dunif/nv_punif/nv_qunif
# agreement tests: ordinary, degenerate, reversed, infinite and NaN.
uniform_limit_cases <- function() {
  list(
    c(0, 1),
    c(-1, 2),
    c(1, 1),
    c(2, 1),
    c(-Inf, Inf),
    c(0, Inf),
    c(-Inf, 0),
    c(Inf, Inf),
    c(-Inf, -Inf),
    c(NaN, 1),
    c(0, NaN)
  )
}
as_f64 <- function(v) nv_array(v, dtype = "f64")
as_f64_scalar <- function(v) nv_scalar(v, dtype = "f64")

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
    expect_dtype(out, "f32")
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
    expect_dtype(out, "f32")
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
    # At `f64` the answer is accurate to `f64`, not to whatever the default
    # float is. The coefficients are plain R numbers with nothing typed to
    # yield to, so they used to materialize at the default -- and `qnorm(0.975)`
    # came back with an error of 1e-8, `f32` accuracy in an `f64` computation.
    expect_equal(
      as.vector(nv_qnorm(nv_array(p, dtype = "f64"))),
      qnorm(p),
      tolerance = 1e-13
    )
  })

  it("names the operand when it is a boolean", {
    expect_error(nv_qnorm(nv_array(TRUE)), "`p` must be a numeric data type")
    expect_error(nv_pnorm(nv_array(TRUE)), "`q` must be a numeric data type")
    expect_error(nv_dnorm(nv_array(TRUE)), "`x` must be a numeric data type")
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
    f <- function(p) nv_sum(nv_qnorm(p))
    g <- as.vector(jit(gradient(f, wrt = "p"))(nv_array(p, dtype = "f64"))[[1L]])
    expect_equal(g, 1 / dnorm(qnorm(p)), tolerance = 1e-6)
  })

  it("gradient is not halved at the central/tail threshold", {
    f <- function(p) nv_sum(nv_qnorm(p))
    g <- as.vector(jit(gradient(f, wrt = "p"))(
      nv_array(exp(-2), dtype = "f64")
    )[[1L]])
    expect_equal(g, 1 / dnorm(qnorm(exp(-2))), tolerance = 1e-6)

    flog <- function(p) nv_sum(nv_qnorm(p, log_p = TRUE))
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
    f <- function(p) nv_sum(nv_qnorm(p, log_p = TRUE))
    g <- as.vector(jit(gradient(f, wrt = "p"))(
      nv_array(c(-1e4, -1e5), dtype = "f64")
    )[[1L]])
    expect_true(all(is.finite(g)))
  })

  it("gradients wrt mean/sd are exact", {
    p <- c(0.025, 0.9)
    f <- function(p, mean, sd) nv_sum(nv_qnorm(p, mean, sd))
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
    # `p`'s data type, not the default float. The two coincide under the
    # standard defaults, which is what let the result follow the default
    # unnoticed while the coefficients materialized there.
    out <- nv_qnorm(nv_array(c(0.25, 0.75), dtype = "f32"), mean = 0L, sd = 1L)
    expect_dtype(out, "f32")
    out64 <- nv_qnorm(nv_array(c(0.25, 0.75), dtype = "f64"), mean = 0L, sd = 1L)
    expect_dtype(out64, "f64")
  })
})

describe("nv_dunif", {
  it("matches base R dunif() with default min/max", {
    x <- c(-0.5, 0, 0.25, 0.75, 1, 1.5)
    expect_equal(
      as.vector(nv_dunif(nv_array(x))),
      dunif(x),
      tolerance = 1e-6
    )
  })

  it("matches base R dunif() with custom min/max", {
    x <- c(-2, -1, 0, 1, 2, 3)
    expect_equal(
      as.vector(nv_dunif(nv_array(x), min = -1, max = 2)),
      dunif(x, min = -1, max = 2),
      tolerance = 1e-6
    )
  })

  it("log = TRUE matches base R dunif(..., log = TRUE)", {
    x <- c(-2, -1, 0, 1, 2, 3)
    expect_equal(
      as.vector(nv_dunif(nv_array(x), min = -1, max = 2, log = TRUE)),
      dunif(x, min = -1, max = 2, log = TRUE),
      tolerance = 1e-6
    )
  })

  it("is constant on a support that includes both endpoints and zero outside it", {
    x <- nv_array(c(-1e-6, 0, 0.5, 1, 1 + 1e-6))
    expect_equal(as.vector(nv_dunif(x)), c(0, 1, 1, 1, 0))
    expect_equal(as.vector(nv_dunif(x, log = TRUE)), c(-Inf, 0, 0, 0, -Inf))
  })

  it("propagates NaN rather than reading it as outside the support", {
    # Every comparison against NaN is FALSE, so without explicit handling NaN
    # would silently become a density of zero
    x <- nv_array(c(NaN, 0.5))
    expect_equal(as.vector(nv_dunif(x)), c(NaN, 1))
    expect_equal(as.vector(nv_dunif(x, log = TRUE)), c(NaN, 0))
  })

  it("matches base R dunif() for degenerate, reversed, infinite and NaN limits", {
    # base R's dunif() is NaN whenever `max <= min` (so a degenerate interval is
    # NaN, unlike punif()/qunif()) but has no finiteness test, so an unbounded
    # interval has density zero rather than NaN
    x <- c(-Inf, -1, 0, 0.5, 1, 2, Inf, NaN)
    got <- unlist(lapply(uniform_limit_cases(), function(l) {
      lapply(c(FALSE, TRUE), function(lg) {
        as.vector(nv_dunif(as_f64(x), min = as_f64_scalar(l[[1L]]), max = as_f64_scalar(l[[2L]]), log = lg))
      })
    }))
    want <- unlist(lapply(uniform_limit_cases(), function(l) {
      lapply(c(FALSE, TRUE), function(lg) {
        suppressWarnings(dunif(x, l[[1L]], l[[2L]], log = lg))
      })
    }))
    expect_equal(got, want)
  })

  it("non-scalar min/max works", {
    x <- c(0.5, 0.5, 0.5)
    min <- c(0, -1, 0.6)
    max <- c(1, 3, 2)
    expect_equal(
      as.vector(nv_dunif(
        nv_array(x),
        min = nv_array(min),
        max = nv_array(max)
      )),
      dunif(x, min = min, max = max),
      tolerance = 1e-6
    )
  })

  it("converts min/max to the dtype of x", {
    out <- nv_dunif(nv_array(c(0, 1), dtype = "f32"), min = 0L, max = 1L)
    expect_equal(dtype(out), as_dtype("f32"))
  })
})

describe("nv_punif", {
  it("matches base R punif() with default min/max", {
    q <- c(-0.5, 0, 0.25, 0.75, 1, 1.5)
    expect_equal(
      as.vector(nv_punif(nv_array(q))),
      punif(q),
      tolerance = 1e-6
    )
  })

  it("matches base R punif() with custom min/max", {
    q <- c(-2, -1, 0, 1, 2, 3)
    expect_equal(
      as.vector(nv_punif(nv_array(q), min = -1, max = 2)),
      punif(q, min = -1, max = 2),
      tolerance = 1e-6
    )
  })

  it("lower_tail = FALSE matches base R punif(..., lower.tail = FALSE)", {
    q <- c(-2, -1, 0, 1, 2, 3)
    expect_equal(
      as.vector(nv_punif(nv_array(q), min = -1, max = 2, lower_tail = FALSE)),
      punif(q, min = -1, max = 2, lower.tail = FALSE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE matches base R punif(..., log.p = TRUE)", {
    q <- c(-2, -1, 0, 1, 2, 3)
    expect_equal(
      as.vector(nv_punif(nv_array(q), min = -1, max = 2, log_p = TRUE)),
      punif(q, min = -1, max = 2, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE and lower_tail = FALSE compose correctly", {
    q <- c(-2, -1, 0, 1, 2, 3)
    expect_equal(
      as.vector(nv_punif(
        nv_array(q),
        min = -1,
        max = 2,
        lower_tail = FALSE,
        log_p = TRUE
      )),
      punif(q, min = -1, max = 2, lower.tail = FALSE, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("saturates at 0 and 1 outside the support in both tails", {
    q <- nv_array(c(-10, 10))
    expect_equal(as.vector(nv_punif(q)), c(0, 1))
    expect_equal(as.vector(nv_punif(q, lower_tail = FALSE)), c(1, 0))
    expect_equal(as.vector(nv_punif(q, log_p = TRUE)), c(-Inf, 0))
    expect_equal(
      as.vector(nv_punif(q, lower_tail = FALSE, log_p = TRUE)),
      c(0, -Inf)
    )
  })

  it("log_p = TRUE keeps full relative accuracy where the probability is close to one", {
    # (q - min) / (max - min) rounds to within 1 ulp of 1 here, so log() of it
    # retains only ~5 digits; log1p() of the small opposite tail retains all of
    # them. base R's punif() gives -9.9997787828e-13 for the same input.
    q <- nv_array(1e12 - 1, dtype = "f64")
    min <- nv_array(0, dtype = "f64")
    max <- nv_array(1e12, dtype = "f64")
    expect_equal(
      as.vector(nv_punif(q, min = min, max = max, log_p = TRUE)),
      log1p(-1 / 1e12),
      tolerance = 1e-12
    )
    # the upper tail mirrors it
    expect_equal(
      as.vector(nv_punif(
        nv_array(1, dtype = "f64"),
        min = min,
        max = max,
        lower_tail = FALSE,
        log_p = TRUE
      )),
      log1p(-1 / 1e12),
      tolerance = 1e-12
    )
  })

  it("propagates NaN through the clamp to the support", {
    q <- nv_array(c(NaN, 0.25))
    expect_equal(as.vector(nv_punif(q)), c(NaN, 0.25))
    expect_equal(as.vector(nv_punif(q, log_p = TRUE)), c(NaN, log(0.25)))
  })

  it("gradients stay finite at an infinite q (endpoint branch doesn't poison them via nv_ifelse)", {
    # nv_ifelse() differentiates through both branches, so the untaken interior
    # (q - min) / width is evaluated even where an endpoint is selected. At an
    # infinite `q` that branch is infinite, and the reverse pass combines it as
    # 0 * Inf = NaN unless the interior is fed a clamped stand-in.
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    f <- function(q, min, max, lower_tail = TRUE, log_p = FALSE) {
      nv_sum(nv_punif(q, min, max, lower_tail = lower_tail, log_p = log_p))
    }
    grad <- jit(gradient(f, wrt = c("q", "min", "max")), static = c("lower_tail", "log_p"))
    for (k in seq_len(nrow(flags))) {
      g <- grad(
        nv_array(c(-Inf, Inf), dtype = "f64"),
        nv_array(c(-1, -1), dtype = "f64"),
        nv_array(c(2, 2), dtype = "f64"),
        lower_tail = flags$lower_tail[k],
        log_p = flags$log_p[k]
      )
      expect_equal(as.vector(g$q), c(0, 0))
      expect_equal(as.vector(g$min), c(0, 0))
      expect_equal(as.vector(g$max), c(0, 0))
    }
  })

  it("gradient is the density, either side of the log/log1p branch", {
    f <- function(q) nv_sum(nv_punif(q, log_p = TRUE))
    q <- c(0.4, 0.6, 1 - 1e-7)
    g <- as.vector(jit(gradient(f, wrt = "q"))(nv_array(q, dtype = "f64"))[[1L]])
    expect_equal(g, 1 / q, tolerance = 1e-9)

    fp <- function(q) nv_sum(nv_punif(q, min = -1, max = 2))
    gp <- as.vector(jit(gradient(fp, wrt = "q"))(
      nv_array(c(-2, 0.5, 3), dtype = "f64")
    )[[1L]])
    expect_equal(gp, c(0, 1 / 3, 0), tolerance = 1e-9)
  })

  it("matches base R punif() for degenerate, reversed, infinite and NaN limits", {
    # unlike dunif(), base R's punif() admits a degenerate interval but rejects
    # any non-finite limit
    q <- c(-Inf, -1, 0, 0.5, 1, 2, Inf, NaN)
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    got <- unlist(lapply(uniform_limit_cases(), function(l) {
      lapply(seq_len(nrow(flags)), function(k) {
        as.vector(nv_punif(
          as_f64(q),
          as_f64_scalar(l[[1L]]),
          as_f64_scalar(l[[2L]]),
          lower_tail = flags$lower_tail[k],
          log_p = flags$log_p[k]
        ))
      })
    }))
    want <- unlist(lapply(uniform_limit_cases(), function(l) {
      lapply(seq_len(nrow(flags)), function(k) {
        suppressWarnings(punif(q, l[[1L]], l[[2L]], lower.tail = flags$lower_tail[k], log.p = flags$log_p[k]))
      })
    }))
    expect_equal(got, want)
  })

  it("non-scalar min/max works", {
    q <- c(0.5, 0.5, 0.5)
    min <- c(0, -1, 0.6)
    max <- c(1, 3, 2)
    expect_equal(
      as.vector(nv_punif(
        nv_array(q),
        min = nv_array(min),
        max = nv_array(max)
      )),
      punif(q, min = min, max = max),
      tolerance = 1e-6
    )
  })

  it("converts min/max to the dtype of q", {
    out <- nv_punif(nv_array(c(0, 1), dtype = "f32"), min = 0L, max = 1L)
    expect_equal(dtype(out), as_dtype("f32"))
  })
})

describe("nv_qunif", {
  it("matches base R qunif() with default min/max", {
    p <- c(0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999)
    expect_equal(
      as.vector(nv_qunif(nv_array(p))),
      qunif(p),
      tolerance = 1e-6
    )
  })

  it("matches base R qunif() with custom min/max", {
    p <- c(0.001, 0.025, 0.5, 0.975)
    expect_equal(
      as.vector(nv_qunif(nv_array(p), min = -1, max = 2)),
      qunif(p, min = -1, max = 2),
      tolerance = 1e-6
    )
  })

  it("lower_tail = FALSE matches base R qunif(..., lower.tail = FALSE)", {
    p <- c(0.001, 0.025, 0.5, 0.975)
    expect_equal(
      as.vector(nv_qunif(nv_array(p), min = -1, max = 2, lower_tail = FALSE)),
      qunif(p, min = -1, max = 2, lower.tail = FALSE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE matches base R qunif(..., log.p = TRUE)", {
    lp <- c(-700, -10, -2, -0.7, -0.1, 0)
    expect_equal(
      as.vector(nv_qunif(nv_array(lp, dtype = "f64"), min = -1, max = 2, log_p = TRUE)),
      qunif(lp, min = -1, max = 2, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE and lower_tail = FALSE compose correctly", {
    lp <- c(-700, -10, -2, -0.7, -0.1, 0)
    expect_equal(
      as.vector(nv_qunif(
        nv_array(lp, dtype = "f64"),
        min = -1,
        max = 2,
        lower_tail = FALSE,
        log_p = TRUE
      )),
      qunif(lp, min = -1, max = 2, lower.tail = FALSE, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("returns the endpoints of the interval at p = 0 and p = 1", {
    p <- nv_array(c(0, 1), dtype = "f64")
    expect_equal(as.vector(nv_qunif(p, min = -1, max = 2)), c(-1, 2))
    expect_equal(
      as.vector(nv_qunif(p, min = -1, max = 2, lower_tail = FALSE)),
      c(2, -1)
    )
    expect_equal(
      as.vector(nv_qunif(nv_array(c(-Inf, 0), dtype = "f64"), min = -1, max = 2, log_p = TRUE)),
      c(-1, 2)
    )
  })

  it("returns NaN outside [0, 1], and for positive log probabilities", {
    p <- c(-0.25, 1.25, NaN)
    expect_equal(
      as.vector(nv_qunif(nv_array(p, dtype = "f64"))),
      c(NaN, NaN, NaN)
    )
    lp <- c(0.5, Inf, NaN)
    expect_equal(
      as.vector(nv_qunif(nv_array(lp, dtype = "f64"), log_p = TRUE)),
      c(NaN, NaN, NaN)
    )
  })

  it("log_p = TRUE and lower_tail = FALSE keep full accuracy for probabilities close to one", {
    # 1 - exp(p) would cancel away all but ~7 digits of the complement here
    lp <- nv_array(-1e-10, dtype = "f64")
    expect_equal(
      as.vector(nv_qunif(lp, lower_tail = FALSE, log_p = TRUE)),
      -expm1(-1e-10),
      tolerance = 1e-12
    )
  })

  it("inverts nv_punif", {
    q <- c(-1, -0.5, 0.5, 1.5, 2)
    expect_equal(
      as.vector(nv_qunif(
        nv_punif(nv_array(q, dtype = "f64"), min = -1, max = 2),
        min = -1,
        max = 2
      )),
      q,
      tolerance = 1e-9
    )
  })

  it("gradient wrt p is the width of the interval", {
    f <- function(p) nv_sum(nv_qunif(p, min = -1, max = 2))
    g <- as.vector(jit(gradient(f, wrt = "p"))(
      nv_array(c(0.1, 0.5, 0.9), dtype = "f64")
    )[[1L]])
    expect_equal(g, c(3, 3, 3), tolerance = 1e-9)
  })

  it("gradients wrt min/max are exact", {
    p <- c(0.25, 0.75)
    f <- function(p, min, max) nv_sum(nv_qunif(p, min, max))
    g <- jit(gradient(f, wrt = c("min", "max")))(
      nv_array(p, dtype = "f64"),
      nv_array(c(0, 0), dtype = "f64"),
      nv_array(c(1, 1), dtype = "f64")
    )
    expect_equal(as.vector(g[[1L]]), 1 - p)
    expect_equal(as.vector(g[[2L]]), p)
  })

  it("gradients stay finite outside the admissible range (invalid branch doesn't poison them via nv_ifelse)", {
    # nv_ifelse() differentiates through both branches, so `min + u * (max - min)`
    # is evaluated even where the NaN branch is selected. At p = +Inf -- or at a
    # positive log probability, where exp(p) overflows -- that branch is
    # infinite, and the reverse pass combines it as 0 * Inf = NaN unless `p` is
    # replaced by an in-range stand-in first.
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    f <- function(p, min, max, lower_tail = TRUE, log_p = FALSE) {
      nv_sum(nv_qunif(p, min, max, lower_tail = lower_tail, log_p = log_p))
    }
    grad <- jit(gradient(f, wrt = c("p", "min", "max")), static = c("lower_tail", "log_p"))
    for (k in seq_len(nrow(flags))) {
      # -Inf is a legitimate log probability, so the out-of-range side differs
      p <- if (flags$log_p[k]) c(0.5, Inf) else c(-Inf, Inf)
      g <- grad(
        nv_array(p, dtype = "f64"),
        nv_array(c(-1, -1), dtype = "f64"),
        nv_array(c(2, 2), dtype = "f64"),
        lower_tail = flags$lower_tail[k],
        log_p = flags$log_p[k]
      )
      expect_equal(as.vector(g$p), c(0, 0))
      expect_equal(as.vector(g$min), c(0, 0))
      expect_equal(as.vector(g$max), c(0, 0))
    }
  })

  it("matches base R qunif() for degenerate, reversed, infinite and NaN limits", {
    # a degenerate interval collapses to `min`; a non-finite limit is NaN
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    grid <- function(log_p) if (log_p) c(-Inf, -2, -0.7, -1e-10, 0, 0.5, NaN) else c(-0.5, 0, 0.25, 1, 1.5, NaN)
    got <- unlist(lapply(uniform_limit_cases(), function(l) {
      lapply(seq_len(nrow(flags)), function(k) {
        as.vector(nv_qunif(
          as_f64(grid(flags$log_p[k])),
          as_f64_scalar(l[[1L]]),
          as_f64_scalar(l[[2L]]),
          lower_tail = flags$lower_tail[k],
          log_p = flags$log_p[k]
        ))
      })
    }))
    want <- unlist(lapply(uniform_limit_cases(), function(l) {
      lapply(seq_len(nrow(flags)), function(k) {
        suppressWarnings(qunif(
          grid(flags$log_p[k]),
          l[[1L]],
          l[[2L]],
          lower.tail = flags$lower_tail[k],
          log.p = flags$log_p[k]
        ))
      })
    }))
    expect_equal(got, want)
  })

  it("non-scalar min/max works", {
    p <- c(0.1, 0.5, 0.9)
    min <- c(0, -1, 0.6)
    max <- c(1, 3, 2)
    expect_equal(
      as.vector(nv_qunif(
        nv_array(p),
        min = nv_array(min),
        max = nv_array(max)
      )),
      qunif(p, min = min, max = max),
      tolerance = 1e-6
    )
  })

  it("converts min/max to the dtype of p", {
    out <- nv_qunif(nv_array(c(0.25, 0.75), dtype = "f32"), min = 0L, max = 1L)
    expect_equal(dtype(out), as_dtype("f32"))
  })

  it("names the operand when it is a boolean", {
    expect_error(nv_qunif(nv_array(TRUE)), "`p` must be a numeric data type")
    expect_error(nv_punif(nv_array(TRUE)), "`q` must be a numeric data type")
    expect_error(nv_dunif(nv_array(TRUE)), "`x` must be a numeric data type")
  })
})

# Rates exercised against base R by the nv_dexp/nv_pexp/nv_qexp agreement tests:
# ordinary, degenerate (zero and infinite), negative, NaN, and -0, which base R
# rejects because its scale 1 / rate is -Inf. A literal `-0` would not do: R's
# byte compiler folds it into the constant `0` once this function is compiled.
exp_rate_cases <- function() {
  c(1, 2.5, 0, Inf, -1, NaN, as.numeric("-0"))
}

# Element-by-element relative comparison, with NaN and infinities matched
# exactly. `expect_equal()` measures the mean relative difference over the whole
# vector, which tiny elements barely move, and compares absolutely when the
# expected values are smaller than the tolerance.
expect_equal_elementwise <- function(object, expected, tolerance = 1e-13) {
  close <- (abs(object - expected) <= tolerance * abs(expected)) %in% TRUE
  ok <- ifelse(is.nan(expected), is.nan(object), !is.nan(object) & (object == expected | close))
  bad <- which(!ok)
  expect(
    length(bad) == 0L,
    sprintf(
      "Elements differ at %s: got %s, expected %s.",
      paste(bad, collapse = ", "),
      paste(format(object[bad], digits = 17), collapse = ", "),
      paste(format(expected[bad], digits = 17), collapse = ", ")
    )
  )
  invisible(object)
}

describe("nv_dexp", {
  it("matches base R dexp() with default rate", {
    x <- c(-1, 0, 0.25, 0.5, 1, 2, 5)
    expect_equal(
      as.vector(nv_dexp(nv_array(x))),
      dexp(x),
      tolerance = 1e-6
    )
  })

  it("matches base R dexp() with custom rate", {
    x <- c(-1, 0, 0.25, 0.5, 1, 2, 5)
    expect_equal(
      as.vector(nv_dexp(nv_array(x), rate = 2.5)),
      dexp(x, rate = 2.5),
      tolerance = 1e-6
    )
  })

  it("log = TRUE matches base R dexp(..., log = TRUE)", {
    x <- c(-1, 0, 0.25, 0.5, 1, 2, 5)
    expect_equal(
      as.vector(nv_dexp(nv_array(x), rate = 2.5, log = TRUE)),
      dexp(x, rate = 2.5, log = TRUE),
      tolerance = 1e-6
    )
  })

  it("log = TRUE stays finite where the plain density underflows to 0", {
    x <- nv_array(800)
    expect_equal(as.vector(nv_dexp(x)), 0)
    expect_equal(as.vector(nv_dexp(x, log = TRUE)), -800, tolerance = 1e-6)
  })

  it("includes zero in the support and is zero below it", {
    x <- nv_array(c(-1e-6, 0, 1e-6))
    expect_equal(as.vector(nv_dexp(x, rate = 2)), c(0, 2, 2 * exp(-2e-6)), tolerance = 1e-6)
    expect_equal(as.vector(nv_dexp(x, rate = 2, log = TRUE)), c(-Inf, log(2), log(2) - 2e-6), tolerance = 1e-6)
  })

  it("propagates NaN rather than reading it as outside the support", {
    x <- nv_array(c(NaN, 0))
    expect_equal(as.vector(nv_dexp(x)), c(NaN, 1))
    expect_equal(as.vector(nv_dexp(x, log = TRUE)), c(NaN, 0))
  })

  it("matches base R dexp() for degenerate, negative and NaN rates", {
    # base R tests the scale 1 / rate > 0, so `rate = Inf` is NaN while
    # `rate = 0` is admitted
    x <- c(-Inf, -1, 0, 0.5, 1, 800, Inf, NaN)
    got <- unlist(lapply(exp_rate_cases(), function(r) {
      lapply(c(FALSE, TRUE), function(lg) {
        as.vector(nv_dexp(as_f64(x), rate = as_f64_scalar(r), log = lg))
      })
    }))
    want <- unlist(lapply(exp_rate_cases(), function(r) {
      lapply(c(FALSE, TRUE), function(lg) {
        suppressWarnings(dexp(x, r, log = lg))
      })
    }))
    expect_equal_elementwise(got, want)
  })

  it("gradients stay finite at an infinite x (resolved branch doesn't poison them via nv_ifelse)", {
    # At `x = Inf` the untaken branch would combine -x * exp(-rate * x) as
    # Inf * 0 = NaN in the gradient wrt `rate`
    f <- function(x, rate, log = FALSE) nv_sum(nv_dexp(x, rate, log = log))
    grad <- jit(gradient(f, wrt = c("x", "rate")), static = "log")
    for (lg in c(FALSE, TRUE)) {
      g <- grad(
        nv_array(c(-Inf, Inf), dtype = "f64"),
        nv_array(c(2, 2), dtype = "f64"),
        log = lg
      )
      expect_equal(as.vector(g$x), c(0, 0))
      expect_equal(as.vector(g$rate), c(0, 0))
    }
  })

  it("gradients match the analytic derivatives", {
    x <- c(0.25, 1, 3)
    rate <- c(0.5, 2, 2)
    f <- function(x, rate, log = FALSE) nv_sum(nv_dexp(x, rate, log = log))
    grad <- jit(gradient(f, wrt = c("x", "rate")), static = "log")

    g <- grad(nv_array(x, dtype = "f64"), nv_array(rate, dtype = "f64"))
    expect_equal(as.vector(g$x), -rate^2 * exp(-rate * x), tolerance = 1e-9)
    expect_equal(as.vector(g$rate), (1 - rate * x) * exp(-rate * x), tolerance = 1e-9)

    g <- grad(nv_array(x, dtype = "f64"), nv_array(rate, dtype = "f64"), log = TRUE)
    expect_equal(as.vector(g$x), -rate, tolerance = 1e-9)
    expect_equal(as.vector(g$rate), 1 / rate - x, tolerance = 1e-9)
  })

  it("non-scalar rate works", {
    x <- c(0.5, 0.5, 0.5)
    rate <- c(0.5, 1, 3)
    expect_equal(
      as.vector(nv_dexp(nv_array(x), rate = nv_array(rate))),
      dexp(x, rate = rate),
      tolerance = 1e-6
    )
  })

  it("accepts large finite rates, whose reciprocal flushes to zero", {
    for (dt in c("f32", "f64")) {
      rate <- nv_scalar(if (dt == "f32") 1e38 else 1e308, dtype = dt)
      x <- nv_scalar(0, dtype = dt)
      expect_equal(as.vector(nv_dexp(x, rate)) / as.vector(rate), 1)
      expect_true(is.nan(as.vector(nv_dexp(x, -rate))))
    }
  })

  it("rescales density tails by a large rate before they underflow, either side of the split", {
    for (dt in c("f32", "f64")) {
      r <- if (dt == "f32") 1e30 else 1e300
      t <- if (dt == "f32") c(79, 80, 81, 110) else c(699, 700, 701, 750)
      x <- nv_array(t / r, dtype = dt)
      rate <- nv_scalar(r, dtype = dt)
      want <- exp(log(as.vector(rate)) - as.vector(rate) * as.vector(x))
      expect_equal_elementwise(
        as.vector(nv_dexp(x, rate)),
        want,
        tolerance = if (dt == "f32") 2e-5 else 1e-12
      )
    }
  })

  it("converts rate to the dtype of x", {
    out <- nv_dexp(nv_array(c(0, 1), dtype = "f32"), rate = 2L)
    expect_equal(dtype(out), as_dtype("f32"))
  })
})

describe("nv_pexp", {
  it("matches base R pexp() with default rate", {
    q <- c(-1, 0, 0.25, 0.5, 1, 2, 5)
    expect_equal(
      as.vector(nv_pexp(nv_array(q))),
      pexp(q),
      tolerance = 1e-6
    )
  })

  it("matches base R pexp() with custom rate", {
    q <- c(-1, 0, 0.25, 0.5, 1, 2, 5)
    expect_equal(
      as.vector(nv_pexp(nv_array(q), rate = 2.5)),
      pexp(q, rate = 2.5),
      tolerance = 1e-6
    )
  })

  it("lower_tail = FALSE matches base R pexp(..., lower.tail = FALSE)", {
    q <- c(-1, 0, 0.25, 0.5, 1, 2, 5)
    expect_equal(
      as.vector(nv_pexp(nv_array(q), rate = 2.5, lower_tail = FALSE)),
      pexp(q, rate = 2.5, lower.tail = FALSE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE matches base R pexp(..., log.p = TRUE) either side of the log1mexp switch", {
    # the switch is at rate * q = log(2)
    q <- c(-1, 0, 0.1, 0.25, log(2) / 2.5, 0.5, 1, 5)
    expect_equal(
      as.vector(nv_pexp(nv_array(q), rate = 2.5, log_p = TRUE)),
      pexp(q, rate = 2.5, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE and lower_tail = FALSE compose correctly", {
    q <- c(-1, 0, 0.25, 0.5, 1, 5)
    expect_equal(
      as.vector(nv_pexp(
        nv_array(q),
        rate = 2.5,
        lower_tail = FALSE,
        log_p = TRUE
      )),
      pexp(q, rate = 2.5, lower.tail = FALSE, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("saturates at 0 and 1 outside the support in both tails", {
    q <- nv_array(c(-10, -Inf, Inf))
    expect_equal(as.vector(nv_pexp(q)), c(0, 0, 1))
    expect_equal(as.vector(nv_pexp(q, lower_tail = FALSE)), c(1, 1, 0))
    expect_equal(as.vector(nv_pexp(q, log_p = TRUE)), c(-Inf, -Inf, 0))
    expect_equal(
      as.vector(nv_pexp(q, lower_tail = FALSE, log_p = TRUE)),
      c(0, 0, -Inf)
    )
  })

  it("keeps full relative accuracy for tiny and near-one probabilities, in both f32 and f64", {
    # 1 - exp(-q) would cancel to 0 at q = 1e-20, and log(1 - exp(-q)) would
    # round to 0 at q = 40
    for (dt in c("f32", "f64")) {
      tiny <- nv_array(1e-20, dtype = dt)
      expect_equal(as.vector(nv_pexp(tiny)) / 1e-20, 1, tolerance = 1e-6)
      expect_equal(as.vector(nv_pexp(tiny, log_p = TRUE)), log(1e-20), tolerance = 1e-6)
      expect_equal(
        as.vector(nv_pexp(nv_array(40, dtype = dt), log_p = TRUE)) / -exp(-40),
        1,
        tolerance = 1e-6
      )
    }
  })

  it("lower_tail = FALSE and log_p = TRUE stays finite where the probability underflows", {
    q <- nv_array(c(1e3, 1e5), dtype = "f64")
    expect_equal(as.vector(nv_pexp(q, lower_tail = FALSE)), c(0, 0))
    expect_equal(as.vector(nv_pexp(q, lower_tail = FALSE, log_p = TRUE)), c(-1e3, -1e5))
  })

  it("propagates NaN through the resolved ends", {
    q <- nv_array(c(NaN, 1))
    expect_equal(as.vector(nv_pexp(q)), c(NaN, pexp(1)), tolerance = 1e-6)
    expect_equal(as.vector(nv_pexp(q, log_p = TRUE)), c(NaN, pexp(1, log.p = TRUE)), tolerance = 1e-6)
  })

  it("matches base R pexp() for degenerate, negative and NaN rates", {
    # unlike dexp(), base R's pexp() tests the scale 1 / rate >= 0, so it admits
    # `rate = Inf`
    q <- c(-Inf, -1, 0, 1e-300, 0.5, 1, 800, Inf, NaN)
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    got <- unlist(lapply(exp_rate_cases(), function(r) {
      lapply(seq_len(nrow(flags)), function(k) {
        as.vector(nv_pexp(
          as_f64(q),
          as_f64_scalar(r),
          lower_tail = flags$lower_tail[k],
          log_p = flags$log_p[k]
        ))
      })
    }))
    want <- unlist(lapply(exp_rate_cases(), function(r) {
      lapply(seq_len(nrow(flags)), function(k) {
        suppressWarnings(pexp(q, r, lower.tail = flags$lower_tail[k], log.p = flags$log_p[k]))
      })
    }))
    expect_equal_elementwise(got, want)
  })

  it("gradients stay finite at an infinite q (resolved branch doesn't poison them via nv_ifelse)", {
    # At `q = Inf` the untaken branch would combine -q * exp(-rate * q) as
    # Inf * 0 = NaN in the gradient wrt `rate`; at `q = -Inf` the clamp keeps
    # the interior branch finite
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    f <- function(q, rate, lower_tail = TRUE, log_p = FALSE) {
      nv_sum(nv_pexp(q, rate, lower_tail = lower_tail, log_p = log_p))
    }
    grad <- jit(gradient(f, wrt = c("q", "rate")), static = c("lower_tail", "log_p"))
    for (k in seq_len(nrow(flags))) {
      g <- grad(
        nv_array(c(-Inf, Inf), dtype = "f64"),
        nv_array(c(2, 2), dtype = "f64"),
        lower_tail = flags$lower_tail[k],
        log_p = flags$log_p[k]
      )
      expect_equal(as.vector(g$q), c(0, 0))
      expect_equal(as.vector(g$rate), c(0, 0))
    }
  })

  it("gradient is the density, either side of the log1mexp switch", {
    q <- c(0.1, log(2) / 2, 0.5, 3)
    f <- function(q) nv_sum(nv_pexp(q, rate = 2))
    g <- as.vector(jit(gradient(f, wrt = "q"))(nv_array(q, dtype = "f64"))[[1L]])
    expect_equal(g, dexp(q, rate = 2), tolerance = 1e-9)

    fl <- function(q) nv_sum(nv_pexp(q, rate = 2, log_p = TRUE))
    gl <- as.vector(jit(gradient(fl, wrt = "q"))(nv_array(q, dtype = "f64"))[[1L]])
    expect_equal(gl, dexp(q, rate = 2) / pexp(q, rate = 2), tolerance = 1e-9)
  })

  it("non-scalar rate works", {
    q <- c(0.5, 0.5, 0.5)
    rate <- c(0.5, 1, 3)
    expect_equal(
      as.vector(nv_pexp(nv_array(q), rate = nv_array(rate))),
      pexp(q, rate = rate),
      tolerance = 1e-6
    )
  })

  it("accepts large finite rates and rejects large negative ones", {
    for (dt in c("f32", "f64")) {
      rate <- nv_scalar(if (dt == "f32") 1e38 else 1e308, dtype = dt)
      q <- nv_scalar(1, dtype = dt)
      expect_equal(as.vector(nv_pexp(q, rate)), 1)
      expect_true(is.nan(as.vector(nv_pexp(q, -rate))))
      expect_true(is.nan(as.vector(nv_pexp(q, -rate, log_p = TRUE))))
    }
  })

  it("log_p = TRUE stays finite where rate * q underflows", {
    for (dt in c("f32", "f64")) {
      r <- if (dt == "f32") 1e-30 else 1e-200
      rate <- nv_scalar(r, dtype = dt)
      # Scalar and length-one programs can be optimized differently to vectors.
      for (q in list(nv_scalar(r, dtype = dt), nv_array(r, dtype = dt), nv_array(c(r, 2 * r), dtype = dt))) {
        want <- log(as.vector(rate)) + log(as.vector(q))
        expect_equal_elementwise(
          as.vector(nv_pexp(q, rate, log_p = TRUE)),
          want,
          tolerance = if (dt == "f32") 3e-7 else 1e-15
        )
      }
    }
  })

  it("log_p = TRUE agrees either side of where rate * q underflows", {
    # The two smaller products fall below the smallest normal and flush to zero
    for (dt in c("f32", "f64")) {
      smallest <- if (dt == "f32") 2^-126 else 2^-1022
      rate <- nv_scalar(2^-20, dtype = dt)
      q <- nv_array(smallest * 2^20 * c(0.5, 0.99, 1.01, 2), dtype = dt)
      want <- log(as.vector(rate)) + log(as.vector(q))
      expect_equal_elementwise(
        as.vector(nv_pexp(q, rate, log_p = TRUE)),
        want,
        tolerance = if (dt == "f32") 3e-7 else 1e-15
      )
    }
  })

  it("log_p = TRUE keeps full accuracy where rate * q is tiny but representable", {
    # log(rate) + log(q) is exact in principle here, but cancels several ulps
    # away when the two logs are large and of opposite sign
    rate <- c(0x1.bd318f07885f5p-450, 1e-100, 1e100, 1e-20, 1e150)
    q <- c(0x1.362c5bca9c0c2p+389, 1e-150, 1e-217, 1e-5, 1e-300)
    expect_equal_elementwise(
      as.vector(nv_pexp(as_f64(q), as_f64(rate), log_p = TRUE)),
      pexp(q, rate, log.p = TRUE),
      tolerance = 5e-16
    )
  })

  it("log_p = TRUE has accurate gradients where rate * q underflows", {
    f <- function(q, rate) nv_sum(nv_pexp(q, rate, log_p = TRUE))
    grad <- jit(gradient(f, wrt = c("q", "rate")))
    for (dt in c("f32", "f64")) {
      r <- if (dt == "f32") 1e-30 else 1e-200
      tolerance <- if (dt == "f32") 5e-7 else 1e-12
      rate <- nv_scalar(r, dtype = dt)
      for (q in list(nv_scalar(r, dtype = dt), nv_array(r, dtype = dt), nv_array(c(r, 2 * r), dtype = dt))) {
        g <- grad(q, rate)
        q_values <- as.vector(q)
        expect_equal(as.vector(g$q) / (1 / q_values), rep(1, length(q_values)), tolerance = tolerance)
        expect_equal(as.vector(g$rate) / (length(q_values) / as.vector(rate)), 1, tolerance = tolerance)
      }
    }
  })

  it("log_p = TRUE ignores overflow in unused rescaled factors", {
    f <- function(q, rate) nv_sum(nv_pexp(q, rate, log_p = TRUE))
    grad <- jit(gradient(f, wrt = c("q", "rate")))
    for (dt in c("f32", "f64")) {
      large <- if (dt == "f32") 1e30 else 1e200
      tolerance <- if (dt == "f32") 5e-7 else 1e-12
      q <- nv_array(c(large, 1 / large), dtype = dt)
      rate <- nv_array(c(1 / large, large), dtype = dt)
      q_values <- as.vector(q)
      rate_values <- as.vector(rate)
      product <- q_values * rate_values
      expect_equal(
        as.vector(nv_pexp(q, rate, log_p = TRUE)),
        pexp(q_values, rate_values, log.p = TRUE),
        tolerance = tolerance
      )
      g <- grad(q, rate)
      expect_equal(as.vector(g$q) / (rate_values / expm1(product)), c(1, 1), tolerance = tolerance)
      expect_equal(as.vector(g$rate) / (q_values / expm1(product)), c(1, 1), tolerance = tolerance)
    }
  })

  it("log_p = TRUE resolves a subnormal q as q = 0, in the gradient as in the value", {
    # Arithmetic flushes a subnormal q to zero; the gradient must then take the
    # q = 0 branch too, not the interior formula, whose derivative is singular
    for (dt in c("f32", "f64")) {
      q <- nv_array(c(if (dt == "f32") 1e-40 else 1e-315, 0), dtype = dt)
      rate <- nv_array(c(2, 2), dtype = dt)
      f <- function(q, rate) nv_sum(nv_pexp(q, rate, log_p = TRUE))
      g <- jit(gradient(f, wrt = c("q", "rate")))(q, rate)
      expect_equal(as.vector(nv_pexp(q, rate, log_p = TRUE)), c(-Inf, -Inf))
      expect_equal(as.vector(g$q), c(0, 0))
      expect_equal(as.vector(g$rate), c(0, 0))
    }
  })

  it("converts rate to the dtype of q", {
    out <- nv_pexp(nv_array(c(0, 1), dtype = "f32"), rate = 2L)
    expect_equal(dtype(out), as_dtype("f32"))
  })
})

describe("nv_qexp", {
  it("matches base R qexp() with default rate", {
    # -log1p(-p) is ill-conditioned as p -> 1, so compare at the probabilities
    # the array holds: at the default float 0.999 is already ~1e-8 out
    p <- nv_array(c(0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999))
    expect_equal(
      as.vector(nv_qexp(p)),
      qexp(as.vector(p)),
      tolerance = 1e-6
    )
  })

  it("matches base R qexp() with custom rate", {
    p <- c(0.001, 0.025, 0.5, 0.975)
    expect_equal(
      as.vector(nv_qexp(nv_array(p), rate = 2.5)),
      qexp(p, rate = 2.5),
      tolerance = 1e-6
    )
  })

  it("lower_tail = FALSE matches base R qexp(..., lower.tail = FALSE)", {
    p <- c(0.001, 0.025, 0.5, 0.975)
    expect_equal(
      as.vector(nv_qexp(nv_array(p), rate = 2.5, lower_tail = FALSE)),
      qexp(p, rate = 2.5, lower.tail = FALSE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE matches base R qexp(..., log.p = TRUE) either side of the log1mexp switch", {
    lp <- c(-700, -10, -2, -log(2), -0.5, -0.1, 0)
    expect_equal(
      as.vector(nv_qexp(nv_array(lp, dtype = "f64"), rate = 2.5, log_p = TRUE)),
      qexp(lp, rate = 2.5, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE and lower_tail = FALSE compose correctly", {
    lp <- c(-700, -10, -2, -0.7, -0.1, 0)
    expect_equal(
      as.vector(nv_qexp(
        nv_array(lp, dtype = "f64"),
        rate = 2.5,
        lower_tail = FALSE,
        log_p = TRUE
      )),
      qexp(lp, rate = 2.5, lower.tail = FALSE, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("returns 0 and Inf at the ends of [0, 1]", {
    p <- nv_array(c(0, 1), dtype = "f64")
    expect_equal(as.vector(nv_qexp(p, rate = 2)), c(0, Inf))
    expect_equal(as.vector(nv_qexp(p, rate = 2, lower_tail = FALSE)), c(Inf, 0))
    expect_equal(
      as.vector(nv_qexp(nv_array(c(-Inf, 0), dtype = "f64"), rate = 2, log_p = TRUE)),
      c(0, Inf)
    )
  })

  it("returns NaN outside [0, 1], and for positive log probabilities", {
    p <- c(-0.25, 1.25, NaN)
    expect_equal(
      as.vector(nv_qexp(nv_array(p, dtype = "f64"))),
      c(NaN, NaN, NaN)
    )
    lp <- c(0.5, Inf, NaN)
    expect_equal(
      as.vector(nv_qexp(nv_array(lp, dtype = "f64"), log_p = TRUE)),
      c(NaN, NaN, NaN)
    )
  })

  it("keeps full relative accuracy for tiny probabilities, in both f32 and f64", {
    # -log(1 - p) would cancel to 0 at p = 1e-20, as would -log(1 - exp(lp)) at
    # lp = log(1e-20)
    for (dt in c("f32", "f64")) {
      expect_equal(as.vector(nv_qexp(nv_array(1e-20, dtype = dt))) / 1e-20, 1, tolerance = 1e-6)
      lp <- nv_array(log(1e-20), dtype = dt)
      expect_equal(
        as.vector(nv_qexp(lp, log_p = TRUE)) / qexp(as.vector(lp), log.p = TRUE),
        1,
        tolerance = 1e-6
      )
    }
  })

  it("inverts nv_pexp", {
    # Large quantiles round-trip through the upper tail, where their
    # probabilities are not crowded against 1
    q <- c(0, 0.01, 0.5, 2)
    expect_equal(
      as.vector(nv_qexp(
        nv_pexp(nv_array(q, dtype = "f64"), rate = 2.5),
        rate = 2.5
      )),
      q,
      tolerance = 1e-9
    )
    q <- c(0.01, 0.5, 2, 10, 100)
    expect_equal(
      as.vector(nv_qexp(
        nv_pexp(nv_array(q, dtype = "f64"), rate = 2.5, lower_tail = FALSE),
        rate = 2.5,
        lower_tail = FALSE
      )),
      q,
      tolerance = 1e-9
    )
  })

  it("gradient wrt p is 1 / dexp(qexp(p))", {
    p <- c(0.1, 0.5, 0.9)
    f <- function(p) nv_sum(nv_qexp(p, rate = 2))
    g <- as.vector(jit(gradient(f, wrt = "p"))(nv_array(p, dtype = "f64"))[[1L]])
    expect_equal(g, 1 / dexp(qexp(p, rate = 2), rate = 2), tolerance = 1e-9)
  })

  it("gradient wrt rate is -qexp(p) / rate", {
    p <- c(0.25, 0.75)
    f <- function(p, rate) nv_sum(nv_qexp(p, rate))
    g <- jit(gradient(f, wrt = "rate"))(
      nv_array(p, dtype = "f64"),
      nv_array(c(2, 2), dtype = "f64")
    )
    expect_equal(as.vector(g[[1L]]), -qexp(p, rate = 2) / 2, tolerance = 1e-9)
  })

  it("gradients stay finite outside the admissible range (invalid branch doesn't poison them via nv_ifelse)", {
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    f <- function(p, rate, lower_tail = TRUE, log_p = FALSE) {
      nv_sum(nv_qexp(p, rate, lower_tail = lower_tail, log_p = log_p))
    }
    grad <- jit(gradient(f, wrt = c("p", "rate")), static = c("lower_tail", "log_p"))
    for (k in seq_len(nrow(flags))) {
      # -Inf is a legitimate log probability, so the out-of-range side differs
      p <- if (flags$log_p[k]) c(0.5, Inf) else c(-Inf, Inf)
      g <- grad(
        nv_array(p, dtype = "f64"),
        nv_array(c(2, 2), dtype = "f64"),
        lower_tail = flags$lower_tail[k],
        log_p = flags$log_p[k]
      )
      expect_equal(as.vector(g$p), c(0, 0))
      expect_equal(as.vector(g$rate), c(0, 0))
    }
  })

  it("matches base R qexp() for degenerate, negative and NaN rates", {
    # `rate = 0` puts every quantile but the one at probability zero at Inf
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    grid <- function(log_p) {
      if (log_p) c(-Inf, -700, -2, -0.7, -1e-10, 0, 0.5, NaN) else c(-0.5, 0, 1e-300, 0.25, 1, 1.5, NaN)
    }
    got <- unlist(lapply(exp_rate_cases(), function(r) {
      lapply(seq_len(nrow(flags)), function(k) {
        as.vector(nv_qexp(
          as_f64(grid(flags$log_p[k])),
          as_f64_scalar(r),
          lower_tail = flags$lower_tail[k],
          log_p = flags$log_p[k]
        ))
      })
    }))
    want <- unlist(lapply(exp_rate_cases(), function(r) {
      lapply(seq_len(nrow(flags)), function(k) {
        suppressWarnings(qexp(
          grid(flags$log_p[k]),
          r,
          lower.tail = flags$lower_tail[k],
          log.p = flags$log_p[k]
        ))
      })
    }))
    expect_equal_elementwise(got, want)
  })

  it("non-scalar rate works", {
    p <- c(0.1, 0.5, 0.9)
    rate <- c(0.5, 1, 3)
    expect_equal(
      as.vector(nv_qexp(nv_array(p), rate = nv_array(rate))),
      qexp(p, rate = rate),
      tolerance = 1e-6
    )
  })

  it("accepts large finite rates and rejects large negative ones", {
    for (dt in c("f32", "f64")) {
      r <- if (dt == "f32") 1e38 else 1e308
      lp <- if (dt == "f32") -80 else -700
      rate <- nv_scalar(r, dtype = dt)
      p <- nv_scalar(lp, dtype = dt)
      expect_equal(
        as.vector(nv_qexp(p, rate, lower_tail = FALSE, log_p = TRUE)) / (-lp / as.vector(rate)),
        1,
        tolerance = 1e-6
      )
      expect_true(is.nan(as.vector(nv_qexp(nv_scalar(0.5, dtype = dt), -rate))))
    }
  })

  it("log_p = TRUE rescales by a small rate before exp(p) underflows", {
    for (dt in c("f32", "f64")) {
      r <- if (dt == "f32") 1e-30 else 1e-300
      lp <- if (dt == "f32") c(-110, -120) else c(-750, -760)
      p <- nv_array(lp, dtype = dt)
      rate <- nv_scalar(r, dtype = dt)
      want <- exp(lp - log(as.vector(rate)))
      expect_equal_elementwise(
        as.vector(nv_qexp(p, rate, log_p = TRUE)),
        want,
        tolerance = if (dt == "f32") 1e-5 else 1e-12
      )
    }
  })

  it("log_p = TRUE agrees either side of the underflow split", {
    for (dt in c("f32", "f64")) {
      split <- if (dt == "f32") -80 else -700
      p <- nv_array(split + c(-0.01, 0, 0.01), dtype = dt)
      want <- exp(as.vector(p)) / 2
      expect_equal_elementwise(
        as.vector(nv_qexp(p, rate = 2, log_p = TRUE)),
        want,
        tolerance = if (dt == "f32") 5e-7 else 1e-15
      )
    }
  })

  it("log_p = TRUE keeps full accuracy for tiny probabilities that do not underflow", {
    # -log(1 - exp(p)) = exp(p) to working precision below about p = -37. A
    # tolerance of about 2 ulp: rescaling through exp(p / 2) costs up to 4.
    lp <- seq(-700, -37, length.out = 4000)
    expect_equal_elementwise(
      as.vector(nv_qexp(as_f64(lp), rate = 2.5, log_p = TRUE)),
      exp(lp) / 2.5,
      tolerance = 4e-16
    )
  })

  it("log_p = TRUE has accurate gradients where exp(p) underflows", {
    p <- nv_array(c(-750, -760), dtype = "f64")
    rate <- nv_scalar(1e-300, dtype = "f64")
    f <- function(p, rate) nv_sum(nv_qexp(p, rate, log_p = TRUE))
    g <- jit(gradient(f, wrt = c("p", "rate")))(p, rate)
    want <- exp(as.vector(p) - log(1e-300))
    expect_equal_elementwise(as.vector(g$p), want, tolerance = 1e-12)
    expect_equal(as.vector(g$rate) / (-sum(want) / 1e-300), 1, tolerance = 1e-12)
  })

  it("resolves the zero quantile before differentiating at rate zero", {
    f <- function(p, rate, lower_tail, log_p) nv_sum(nv_qexp(p, rate, lower_tail, log_p))
    grad <- jit(gradient(f, wrt = "rate"), static = c("lower_tail", "log_p"))
    for (lt in c(FALSE, TRUE)) {
      for (lp in c(FALSE, TRUE)) {
        p <- if (lp) {
          if (lt) -Inf else 0
        } else {
          if (lt) 0 else 1
        }
        g <- grad(nv_scalar(p, dtype = "f64"), nv_scalar(0, dtype = "f64"), lt, lp)
        expect_equal(as.vector(g$rate), 0)
      }
    }
  })

  it("the zero quantile keeps its finite one-sided d/dp", {
    f <- function(p, rate, lower_tail, log_p) nv_sum(nv_qexp(p, rate, lower_tail, log_p))
    grad <- jit(gradient(f, wrt = c("p", "rate")), static = c("lower_tail", "log_p"))
    cases <- list(
      list(lower_tail = TRUE, log_p = FALSE, p = c(0, as.numeric("-0")), d_p = 1 / 2),
      list(lower_tail = FALSE, log_p = FALSE, p = c(1, 1), d_p = -1 / 2),
      list(lower_tail = FALSE, log_p = TRUE, p = c(0, as.numeric("-0")), d_p = -1 / 2)
    )
    for (cs in cases) {
      p <- as_f64(cs$p)
      rate <- as_f64(c(2, 2))
      value <- as.vector(nv_qexp(p, rate, lower_tail = cs$lower_tail, log_p = cs$log_p))
      expect_equal(value, c(0, 0))
      g <- grad(p, rate, cs$lower_tail, cs$log_p)
      expect_equal(as.vector(g$p), rep(cs$d_p, 2))
      expect_equal(as.vector(g$rate), c(0, 0))
    }
  })

  it("converts rate to the dtype of p", {
    out <- nv_qexp(nv_array(c(0.25, 0.75), dtype = "f32"), rate = 2L)
    expect_equal(dtype(out), as_dtype("f32"))
  })

  it("names the operand when it is a boolean", {
    expect_error(nv_qexp(nv_array(TRUE)), "`p` must be a numeric data type")
    expect_error(nv_pexp(nv_array(TRUE)), "`q` must be a numeric data type")
    expect_error(nv_dexp(nv_array(TRUE)), "`x` must be a numeric data type")
  })
})

describe("nv_dexp(), nv_pexp() and nv_qexp()", {
  it("reject a rate of -Inf", {
    for (f in list(nv_dexp, nv_pexp, nv_qexp)) {
      expect_true(all(is.nan(as.vector(f(nv_array(c(0, 0.5, 1, Inf)), rate = -Inf)))))
    }
  })

  it("broadcast a scalar operand against a non-scalar rate", {
    for (f in list(nv_dexp, nv_pexp, nv_qexp)) {
      got <- f(nv_scalar(0.5), rate = nv_array(c(1, 2)))
      expect_shape(got, 2L)
      expect_equal(as.vector(got), as.vector(f(nv_array(c(0.5, 0.5)), rate = nv_array(c(1, 2)))))
    }
  })
})

# Size/prob configurations exercised against base R by the nv_dbinom/nv_pbinom/
# nv_qbinom agreement tests: ordinary, degenerate, invalid, infinite and NaN.
binomial_parameter_cases <- function() {
  list(
    c(10, 0.3),
    c(0, 0.3),
    c(10, 0),
    c(10, 1),
    c(10.5, 0.3),
    c(-1, 0.3),
    c(10, -0.1),
    c(10, 1.1),
    c(Inf, 0.3),
    c(Inf, 0),
    c(NaN, 0.3),
    c(10, NaN)
  )
}

describe("nv_dbinom", {
  it("matches base R dbinom()", {
    x <- c(0, 1, 2, 3, 5, 9, 10)
    expect_equal(
      as.vector(nv_dbinom(nv_array(x), size = 10, prob = 0.3)),
      dbinom(x, size = 10, prob = 0.3),
      tolerance = 1e-6
    )
  })

  it("log = TRUE matches base R dbinom(..., log = TRUE)", {
    x <- c(0, 1, 2, 3, 5, 9, 10)
    expect_equal(
      as.vector(nv_dbinom(nv_array(x), size = 10, prob = 0.3, log = TRUE)),
      dbinom(x, size = 10, prob = 0.3, log = TRUE),
      tolerance = 1e-6
    )
  })

  it("keeps f64 accuracy across the support for small and large size", {
    for (args in list(c(20, 0.3), c(1000, 0.001), c(1e5, 0.5), c(1e5, 0.999))) {
      size <- args[[1L]]
      prob <- args[[2L]]
      x <- unique(round(c(0, 1, size * prob + c(-5, 0, 5) * sqrt(size * prob * (1 - prob)), size - 1, size)))
      x <- x[x >= 0 & x <= size]
      for (lg in c(FALSE, TRUE)) {
        expect_equal(
          as.vector(nv_dbinom(as_f64(x), size = size, prob = prob, log = lg)),
          dbinom(x, size = size, prob = prob, log = lg),
          tolerance = 1e-13
        )
      }
    }
  })

  it("stays accurate next to x = size, where log1p(-x / size) would cancel", {
    # base R's dbinom() itself is only good to ~2e-12 here; the exact value
    # comes from the closed form n p^(n - 1) (1 - p).
    x <- as_f64(99999)
    expect_equal(
      as.vector(nv_dbinom(x, size = 1e5, prob = 0.999, log = TRUE)),
      log(1e5) + 99999 * log(0.999) + log1p(-0.999),
      tolerance = 1e-14
    )
  })

  it("stays accurate for very large size, where size * prob must not be rounded", {
    # Reference values from Rmpfr at 400 bits. base R's dbinom() is only good to
    # ~1e-9 here, as it rounds size * prob before taking x - size * prob.
    expect_equal(
      as.vector(nv_dbinom(
        as_f64(c(7699989354000, 2999995653000)),
        size = 1e13,
        prob = as_f64(c(0.77, 0.3)),
        log = TRUE
      )),
      c(-47.018301202312358, -19.604563591068437),
      tolerance = 1e-14
    )
  })

  it("keeps f32 accuracy for large size", {
    # Against base R on the same f32 inputs, so only the f32 computation differs
    prob <- as.vector(nv_array(0.3, dtype = "f32"))
    x <- 3e6 + c(-4000, -1000, 0, 1000, 4000)
    expect_equal(
      as.vector(nv_dbinom(nv_array(x, dtype = "f32"), size = 1e7, prob = prob)),
      dbinom(x, size = 1e7, prob = prob),
      tolerance = 1e-5
    )
  })

  it("log = TRUE stays finite where the plain density underflows to 0", {
    x <- as_f64(0)
    expect_equal(as.vector(nv_dbinom(x, size = 1e4, prob = 0.5)), 0)
    expect_equal(
      as.vector(nv_dbinom(x, size = 1e4, prob = 0.5, log = TRUE)),
      dbinom(0, size = 1e4, prob = 0.5, log = TRUE),
      tolerance = 1e-14
    )
  })

  it("matches base R dbinom() off the support and for degenerate, invalid, infinite and NaN parameters", {
    x <- c(-Inf, -1, 0, 2.5, 3, 10, 11, Inf, NaN)
    got <- unlist(lapply(binomial_parameter_cases(), function(a) {
      lapply(c(FALSE, TRUE), function(lg) {
        as.vector(nv_dbinom(as_f64(x), size = as_f64_scalar(a[[1L]]), prob = as_f64_scalar(a[[2L]]), log = lg))
      })
    }))
    want <- unlist(lapply(binomial_parameter_cases(), function(a) {
      lapply(c(FALSE, TRUE), function(lg) {
        suppressWarnings(dbinom(x, a[[1L]], a[[2L]], log = lg))
      })
    }))
    expect_equal(got, want)
  })

  it("non-scalar size/prob works", {
    x <- c(0, 3, 7)
    size <- c(5, 10, 20)
    prob <- c(0.1, 0.5, 0.9)
    expect_equal(
      as.vector(nv_dbinom(
        nv_array(x),
        size = nv_array(size),
        prob = nv_array(prob)
      )),
      dbinom(x, size = size, prob = prob),
      tolerance = 1e-6
    )
  })

  it("gradient with respect to prob matches the score, including at the ends of the support", {
    # d/dp log p(x) = x / p - (n - x) / (1 - p)
    x <- c(0, 1, 3, 9, 10)
    f <- function(prob) nv_sum(nv_dbinom(as_f64(x), size = 10, prob = prob, log = TRUE))
    g <- as.vector(jit(gradient(f, wrt = "prob"))(as_f64_scalar(0.3))[[1L]])
    expect_equal(g, sum(x / 0.3 - (10 - x) / 0.7), tolerance = 1e-12)
  })

  it("gradient is right at prob of 0 and 1, where the density is a polynomial with a zero", {
    # d/dp (1 - p)^10 = -10 at p = 0 and d/dp p^10 = 10 at p = 1. Next to them,
    # d/dp 10 p (1 - p)^9 = 10 at p = 0 and d/dp 10 (1 - p) p^9 = -10 at p = 1,
    # although the density itself is zero there. Further in it is flat at zero.
    x <- c(0, 10, 1, 9, 2, 8, 5, 5)
    prob <- c(0, 1, 0, 1, 0, 1, 0, 1)
    f <- function(prob) nv_sum(nv_dbinom(as_f64(x), size = 10, prob = prob))
    g <- as.vector(jit(gradient(f, wrt = "prob"))(as_f64(prob))[[1L]])
    expect_equal(g, c(-10, 10, 10, -10, 0, 0, 0, 0))
  })

  it("gradient stays right for a tiny prob whose density is still representable", {
    # d/dp 10 p (1 - p)^9 is about 10 near p = 0. The density there is about
    # 10 p, so the gradient must not be scaled by it before the 1 / p from the
    # log-density is applied, which would underflow.
    for (dt in c("f32", "f64")) {
      prob <- if (dt == "f32") c(1e-30, 1e-20, 1e-15) else c(2.3e-308, 1e-300, 1e-200, 1e-155, 1e-150)
      prob <- as.vector(nv_array(prob, dtype = dt))
      f <- function(prob) nv_sum(nv_dbinom(nv_scalar(1, dtype = dt), size = 10, prob = prob))
      g <- as.vector(jit(gradient(f, wrt = "prob"))(nv_array(prob, dtype = dt))[[1L]])
      expect_equal(g, 10 * (1 - prob)^8 * (1 - 10 * prob), tolerance = if (dt == "f32") 1e-5 else 1e-12, info = dt)
    }
  })

  it("gradient is right for size of 0 and 1, and unpoisoned by invalid elements", {
    # size = 0: the density is 1 at x = 0 whatever prob is. size = 1: it is
    # 1 - p at x = 0 and p at x = 1. Elements with an invalid size or prob, or
    # outside the support, have a NaN or zero density and a zero gradient, and
    # must leave the others' finite.
    x <- c(0, 0, 1, 0, 1, 5, 5, 5, 11, 3)
    size <- c(0, 1, 1, 1, 1, -1, 10.5, 10, 10, 10)
    prob <- c(0.3, 0.3, 0.3, 1, 0, 0.3, 0.3, 1.5, 0.3, 0.3)
    f <- function(prob) nv_sum(nv_dbinom(as_f64(x), size = as_f64(size), prob = prob))
    g <- as.vector(jit(gradient(f, wrt = "prob"))(as_f64(prob))[[1L]])
    expect_equal(g, c(0, -1, 1, -1, 1, 0, 0, 0, 0, dbinom(3, 10, 0.3) * (3 / 0.3 - 7 / 0.7)))
  })

  it("converts size/prob to the dtype of x", {
    out <- nv_dbinom(nv_array(c(0, 1), dtype = "f32"), size = 2L, prob = 0.5)
    expect_dtype(out, "f32")
  })
})

describe("nv_pbinom", {
  it("matches base R pbinom()", {
    q <- c(-1, 0, 1, 2, 3, 5, 9, 10, 11)
    expect_equal(
      as.vector(nv_pbinom(nv_array(q), size = 10, prob = 0.3)),
      pbinom(q, size = 10, prob = 0.3),
      tolerance = 1e-6
    )
  })

  it("lower_tail and log_p match base R in every combination", {
    q <- c(-1, 0, 1, 2, 3, 5, 9, 10, 11)
    for (lt in c(TRUE, FALSE)) {
      for (lp in c(FALSE, TRUE)) {
        expect_equal(
          as.vector(nv_pbinom(as_f64(q), size = 10, prob = 0.3, lower_tail = lt, log_p = lp)),
          pbinom(q, size = 10, prob = 0.3, lower.tail = lt, log.p = lp),
          tolerance = 1e-14
        )
      }
    }
  })

  it("keeps f64 accuracy in both tails and across both continued fraction branches", {
    # The branch switches at the mean, where the directly computed tail changes
    # sides; 8 sd out the other tail comes as a complement.
    for (args in list(c(20, 0.3), c(1000, 0.001), c(1e4, 0.5), c(1e5, 0.77))) {
      size <- args[[1L]]
      prob <- args[[2L]]
      q <- unique(floor(size * prob + c(-8, -3, -1, 0, 1, 3, 8) * sqrt(size * prob * (1 - prob))))
      q <- q[q >= 0 & q < size]
      for (lt in c(TRUE, FALSE)) {
        for (lp in c(FALSE, TRUE)) {
          expect_equal(
            as.vector(nv_pbinom(as_f64(q), size = size, prob = prob, lower_tail = lt, log_p = lp)),
            pbinom(q, size = size, prob = prob, lower.tail = lt, log.p = lp),
            tolerance = 1e-13
          )
        }
      }
    }
  })

  it("stays accurate for very large size, near the mean and in the tails", {
    # Near the mean the continued fraction would need too many iterations here;
    # the asymptotic expansion takes over
    for (size in c(1e12, 1e15)) {
      for (prob in c(0.001, 0.5)) {
        q <- floor(size * prob + c(-5, -1, 0, 1, 5) * sqrt(size * prob * (1 - prob)))
        for (lp in c(FALSE, TRUE)) {
          expect_equal(
            as.vector(nv_pbinom(as_f64(q), size = size, prob = prob, log_p = lp)),
            pbinom(q, size = size, prob = prob, log.p = lp),
            tolerance = 1e-12
          )
        }
      }
    }
  })

  it("stays accurate where size is huge but size * prob is about 1", {
    # The continued fraction's terms then span many orders of magnitude, which
    # its scaling must keep representable. Against base R on the same rounded
    # inputs, in both tails and on both scales.
    for (dt in c("f32", "f64")) {
      for (size in c(1e10, 1e20)) {
        prob <- as.vector(nv_array(1 / size, dtype = dt))
        q <- c(0, 1, 2, 5)
        for (lt in c(TRUE, FALSE)) {
          for (lp in c(FALSE, TRUE)) {
            expect_equal(
              as.vector(nv_pbinom(nv_array(q, dtype = dt), size = size, prob = prob, lower_tail = lt, log_p = lp)),
              pbinom(q, size = size, prob = prob, lower.tail = lt, log.p = lp),
              tolerance = if (dt == "f32") 1e-5 else 1e-12,
              info = paste(dt, size, lt, lp)
            )
          }
        }
      }
    }
    # P(X <= 0) = (1 - prob)^size exactly
    expect_equal(
      as.vector(nv_pbinom(as_f64(0), size = 1e50, prob = 1e-50, log_p = TRUE)),
      1e50 * log1p(-1e-50),
      tolerance = 1e-14
    )
  })

  it("stays accurate deep in the lower tail for huge size, where base R fails", {
    # Reference values from Rmpfr at 400 bits, summing the exact Binomial
    # probabilities. base R gives -Inf or positive log probabilities here.
    q <- 0:5
    expect_equal(
      as.vector(nv_pbinom(as_f64(q), size = 1e20, prob = 1e3 / 1e20, log_p = TRUE)),
      c(
        -1000.0000000000001,
        -993.09124522068487,
        -986.87563662392711,
        -981.06549213528399,
        -975.54302871710217,
        -970.24370784624102
      ),
      tolerance = 1e-14
    )
    expect_equal(
      as.vector(nv_pbinom(as_f64(q), size = 1e50, prob = 1e3 / 1e50, log_p = TRUE)),
      c(-1000, -993.09124522068487, -986.87563662392711, -981.06549213528388, -975.54302871710206, -970.24370784624102),
      tolerance = 1e-14
    )
  })

  it("agrees with base R across a batch mixing the asymptotic expansion and the continued fraction", {
    q <- c(0, 10, 299000, 299800, 300000, 300200, 301000, 999999)
    expect_equal(
      as.vector(nv_pbinom(as_f64(q), size = 1e6, prob = 0.3)),
      pbinom(q, size = 1e6, prob = 0.3),
      tolerance = 1e-12
    )
  })

  it("keeps f32 accuracy for large size", {
    # Against base R on the same f32 inputs, so only the f32 computation differs
    prob <- as.vector(nv_array(0.99, dtype = "f32"))
    q <- 9.9e6 + c(-1000, -300, 0, 300, 1000)
    for (lt in c(TRUE, FALSE)) {
      expect_equal(
        as.vector(nv_pbinom(nv_array(q, dtype = "f32"), size = 1e7, prob = prob, lower_tail = lt)),
        pbinom(q, size = 1e7, prob = prob, lower.tail = lt),
        tolerance = 1e-5
      )
    }
  })

  it("log_p = TRUE stays finite where the probability underflows", {
    q <- as_f64(c(10, 9990))
    expect_equal(
      as.vector(nv_pbinom(q, size = 1e4, prob = 0.5, log_p = TRUE))[[1L]],
      pbinom(10, size = 1e4, prob = 0.5, log.p = TRUE),
      tolerance = 1e-14
    )
    expect_equal(
      as.vector(nv_pbinom(q, size = 1e4, prob = 0.5, lower_tail = FALSE, log_p = TRUE))[[2L]],
      pbinom(9990, size = 1e4, prob = 0.5, lower.tail = FALSE, log.p = TRUE),
      tolerance = 1e-14
    )
  })

  it("rounds q down, with base R's fuzz just below a whole number", {
    q <- as_f64(c(2.5, 3 - 1e-9, 3 - 1e-6))
    expect_equal(
      as.vector(nv_pbinom(q, size = 10, prob = 0.3)),
      pbinom(c(2.5, 3 - 1e-9, 3 - 1e-6), size = 10, prob = 0.3),
      tolerance = 1e-14
    )
  })

  it("applies base R's fuzz without rounding at f32", {
    # At f32, q + 1e-7 would round 1 - 2^-23 up to 1
    q <- c(1 - 2^-23, 2 - 2^-23, 3 - 2^-22, 1 - 1e-7)
    prob <- as.vector(nv_array(0.3, dtype = "f32"))
    q32 <- as.vector(nv_array(q, dtype = "f32"))
    expect_equal(
      as.vector(nv_pbinom(nv_array(q, dtype = "f32"), size = 20, prob = prob)),
      pbinom(q32, size = 20, prob = prob),
      tolerance = 1e-5
    )
  })

  it("matches base R pbinom() off the support and for degenerate, invalid, infinite and NaN parameters", {
    q <- c(-Inf, -1, 0, 2.5, 3, 10, 11, Inf, NaN)
    combos <- expand.grid(lt = c(TRUE, FALSE), lp = c(FALSE, TRUE))
    got <- unlist(lapply(binomial_parameter_cases(), function(a) {
      Map(
        function(lt, lp) {
          as.vector(nv_pbinom(
            as_f64(q),
            size = as_f64_scalar(a[[1L]]),
            prob = as_f64_scalar(a[[2L]]),
            lower_tail = lt,
            log_p = lp
          ))
        },
        combos$lt,
        combos$lp
      )
    }))
    want <- unlist(lapply(binomial_parameter_cases(), function(a) {
      Map(
        function(lt, lp) {
          suppressWarnings(pbinom(q, a[[1L]], a[[2L]], lower.tail = lt, log.p = lp))
        },
        combos$lt,
        combos$lp
      )
    }))
    expect_equal(got, want, tolerance = 1e-14)
  })

  it("non-scalar size/prob works", {
    q <- c(0, 3, 7)
    size <- c(5, 10, 20)
    prob <- c(0.1, 0.5, 0.9)
    expect_equal(
      as.vector(nv_pbinom(
        nv_array(q),
        size = nv_array(size),
        prob = nv_array(prob)
      )),
      pbinom(q, size = size, prob = prob),
      tolerance = 1e-6
    )
  })

  it("converts size/prob to the dtype of q", {
    out <- nv_pbinom(nv_array(c(0, 1), dtype = "f32"), size = 2L, prob = 0.5)
    expect_dtype(out, "f32")
  })
})

describe("nv_qbinom", {
  it("matches base R qbinom()", {
    p <- c(0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999)
    expect_equal(
      as.vector(nv_qbinom(nv_array(p), size = 10, prob = 0.3)),
      qbinom(p, size = 10, prob = 0.3)
    )
  })

  it("lower_tail and log_p match base R in every combination, for small and large size", {
    u <- c(1e-12, 0.001, 0.01, 0.2, 0.5, 0.8, 0.99, 0.999, 1 - 1e-12)
    for (args in list(c(20, 0.3), c(1000, 0.001), c(1e5, 0.5))) {
      for (lt in c(TRUE, FALSE)) {
        for (lp in c(FALSE, TRUE)) {
          p <- if (lp) log(u) else u
          expect_equal(
            as.vector(nv_qbinom(as_f64(p), size = args[[1L]], prob = args[[2L]], lower_tail = lt, log_p = lp)),
            qbinom(p, size = args[[1L]], prob = args[[2L]], lower.tail = lt, log.p = lp)
          )
        }
      }
    }
  })

  it("matches base R qbinom() for very large size", {
    u <- c(1e-10, 0.001, 0.3, 0.5, 0.7, 0.999)
    for (size in c(1e6, 1e12)) {
      expect_equal(
        as.vector(nv_qbinom(as_f64(u), size = size, prob = 0.3)),
        qbinom(u, size = size, prob = 0.3)
      )
    }
  })

  it("is NaN where an evaluation of the distribution function fails, and unaffected elsewhere", {
    # No input is known to make the distribution function fail, so one is made
    # to: for size = 20 only. The search must not read the NaN as an answer.
    # The fresh jit() traces nv_qbinom() anew, so the substitute is seen, and
    # nothing compiled with it is cached by nv_qbinom() itself.
    real_cdf <- binom_cdf
    local_mocked_bindings(binom_cdf = function(k, n, p, lower_tail, log_p, op_dtype) {
      nv_ifelse(n == 20, NaN, real_cdf(k, n, p, lower_tail, log_p, op_dtype))
    })
    p <- c(0.1, 0.5, 0.9)
    size <- c(10, 20, 30)
    out <- jit(function(p, size) nv_qbinom(p, size, 0.3))(as_f64(p), as_f64(size))
    expect_equal(as.vector(out), c(qbinom(0.1, 10, 0.3), NaN, qbinom(0.9, 30, 0.3)))
  })

  it("finds the quantile at extreme log probabilities, where the starting guess overflows", {
    # z * z from the normal quantile overflows here; the search must still run
    for (dt in c("f32", "f64")) {
      lp <- if (dt == "f32") c(-1e30, -1.7e38) else c(-1e300, -9e307)
      for (args in list(c(20, 0.3), c(1e6, 0.3))) {
        expect_equal(
          as.vector(nv_qbinom(nv_array(lp, dtype = dt), size = args[[1L]], prob = args[[2L]], log_p = TRUE)),
          qbinom(lp, size = args[[1L]], prob = args[[2L]], log.p = TRUE),
          info = paste(dt, args[[1L]])
        )
        expect_equal(
          as.vector(nv_qbinom(
            nv_array(lp, dtype = dt),
            size = args[[1L]],
            prob = args[[2L]],
            lower_tail = FALSE,
            log_p = TRUE
          )),
          qbinom(lp, size = args[[1L]], prob = args[[2L]], lower.tail = FALSE, log.p = TRUE),
          info = paste(dt, args[[1L]], "upper")
        )
      }
    }
  })

  it("finds the quantile of the smallest normal probability", {
    # The fuzz on p would take it subnormal, where XLA flushes it to zero
    for (dt in c("f32", "f64")) {
      p <- if (dt == "f32") c(2^-126, 2^-125) else c(2^-1022, 2^-1021)
      prob <- as.vector(nv_array(0.3, dtype = dt))
      expect_equal(
        as.vector(nv_qbinom(nv_array(p, dtype = dt), size = 1e6, prob = prob)),
        qbinom(p, size = 1e6, prob = prob),
        info = dt
      )
    }
  })

  it("matches base R qbinom() where size is huge but size * prob is about 1", {
    for (dt in c("f32", "f64")) {
      for (size in c(1e10, 1e20)) {
        prob <- as.vector(nv_array(1 / size, dtype = dt))
        p <- as.vector(nv_array(c(0.1, 0.5, 0.73, 0.95), dtype = dt))
        expect_equal(
          as.vector(nv_qbinom(nv_array(p, dtype = dt), size = size, prob = prob)),
          qbinom(p, size = size, prob = prob),
          info = paste(dt, size)
        )
      }
    }
  })

  it("matches base R qbinom() at f32 for large size", {
    # On the same f32 inputs. Away from p that round onto a value of the
    # distribution function, where the coarser f32 tolerance decides.
    p <- as.vector(nv_array(c(0.001, 0.2, 0.5, 0.8), dtype = "f32"))
    prob <- as.vector(nv_array(0.3, dtype = "f32"))
    expect_equal(
      as.vector(nv_qbinom(nv_array(p, dtype = "f32"), size = 1e7, prob = prob)),
      qbinom(p, size = 1e7, prob = prob)
    )
  })

  it("maps the distribution function at each support point back to it", {
    # The search is continuous from the left, so a probability computed at a
    # support point is mapped back to that point despite its rounding.
    q <- 0:20
    for (lt in c(TRUE, FALSE)) {
      for (lp in c(FALSE, TRUE)) {
        p <- nv_pbinom(as_f64(q), size = 20, prob = 0.3, lower_tail = lt, log_p = lp)
        got <- as.vector(nv_qbinom(p, size = 20, prob = 0.3, lower_tail = lt, log_p = lp))
        # The upper tail reaches 0 at q = 20, which maps to the start of the
        # support it is attained on
        expect_equal(got, if (lt) q else c(q[-21L], 20))
      }
    }
  })

  it("matches base R qbinom() at the ends of p and for degenerate, invalid, infinite and NaN parameters", {
    p <- c(-0.1, 0, 0.3, 1, 1.1, NaN)
    lp <- c(-Inf, -1, 0, 0.1, NaN)
    combos <- expand.grid(lt = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    got <- unlist(lapply(binomial_parameter_cases(), function(a) {
      Map(
        function(lt, log_p) {
          as.vector(nv_qbinom(
            as_f64(if (log_p) lp else p),
            size = as_f64_scalar(a[[1L]]),
            prob = as_f64_scalar(a[[2L]]),
            lower_tail = lt,
            log_p = log_p
          ))
        },
        combos$lt,
        combos$log_p
      )
    }))
    want <- unlist(lapply(binomial_parameter_cases(), function(a) {
      Map(
        function(lt, log_p) {
          suppressWarnings(qbinom(if (log_p) lp else p, a[[1L]], a[[2L]], lower.tail = lt, log.p = log_p))
        },
        combos$lt,
        combos$log_p
      )
    }))
    expect_equal(got, want)
  })

  it("non-scalar size/prob works", {
    p <- c(0.1, 0.5, 0.9)
    size <- c(5, 10, 20)
    prob <- c(0.1, 0.5, 0.9)
    expect_equal(
      as.vector(nv_qbinom(
        nv_array(p),
        size = nv_array(size),
        prob = nv_array(prob)
      )),
      qbinom(p, size = size, prob = prob)
    )
  })

  it("converts size/prob to the dtype of p", {
    out <- nv_qbinom(nv_array(c(0.1, 0.9), dtype = "f32"), size = 2L, prob = 0.5)
    expect_dtype(out, "f32")
  })

  it("names the operand when it is a boolean", {
    expect_error(nv_qbinom(nv_array(TRUE), 2, 0.5), "`p` must be a numeric data type")
    expect_error(nv_pbinom(nv_array(TRUE), 2, 0.5), "`q` must be a numeric data type")
    expect_error(nv_dbinom(nv_array(TRUE), 2, 0.5), "`x` must be a numeric data type")
  })
})

describe("eager/jit equivalence", {
  it("agrees for nv_dnorm(), nv_pnorm() and nv_qnorm()", {
    f64 <- function() nv_array(c(0.25, 0.75), dtype = "f64")
    expect_eager_jit_equal_grid(list(
      dnorm = function(x, v) nv_dnorm(f64(), mean = v),
      pnorm = function(x, v) nv_pnorm(f64(), sd = v),
      qnorm = function(x, v) nv_qnorm(f64(), mean = v)
    ))
  })

  it("agrees for nv_dexp(), nv_pexp() and nv_qexp()", {
    f64 <- function() nv_array(c(0.25, 0.75), dtype = "f64")
    expect_eager_jit_equal_grid(list(
      dexp = function(x, v) nv_dexp(f64(), rate = v),
      pexp = function(x, v) nv_pexp(f64(), rate = v),
      qexp = function(x, v) nv_qexp(f64(), rate = v)
    ))
  })

  it("agrees for nv_dbinom(), nv_pbinom() and nv_qbinom()", {
    f64 <- function() nv_array(c(0.25, 0.75), dtype = "f64")
    expect_eager_jit_equal_grid(list(
      dbinom = function(x, v) nv_dbinom(f64(), size = 3, prob = v),
      pbinom = function(x, v) nv_pbinom(f64(), size = v, prob = 0.5),
      qbinom = function(x, v) nv_qbinom(f64(), size = 3, prob = v)
    ))
  })
})

describe("an integer operand", {
  # Each function as a function of its operand and one parameter, which may be
  # an R value or an array.
  families <- list(
    dnorm = function(x, v) nv_dnorm(x, mean = v),
    pnorm = function(x, v) nv_pnorm(x, mean = v),
    qnorm = function(x, v) nv_qnorm(x, mean = v),
    dunif = function(x, v) nv_dunif(x, min = v, max = 4),
    punif = function(x, v) nv_punif(x, min = v, max = 4),
    qunif = function(x, v) nv_qunif(x, min = v, max = 4),
    dexp = function(x, v) nv_dexp(x, rate = v),
    pexp = function(x, v) nv_pexp(x, rate = v),
    qexp = function(x, v) nv_qexp(x, rate = v),
    dbinom = function(x, v) nv_dbinom(x, size = v, prob = 0.5),
    pbinom = function(x, v) nv_pbinom(x, size = v, prob = 0.5),
    qbinom = function(x, v) nv_qbinom(x, size = v, prob = 0.5)
  )

  it("computes at the default float when the parameters are R values", {
    for (name in names(families)) {
      expected <- as.vector(families[[name]](nv_array(c(0, 1), dtype = default_float()), 1))
      for (int_dtype in c("i8", "i32", "i64", "ui32")) {
        out <- families[[name]](nv_array(c(0L, 1L), dtype = int_dtype), 1)
        expect_dtype(out, default_float(), info = paste(name, int_dtype))
        expect_equal(as.vector(out), expected, info = paste(name, int_dtype))
      }
    }
  })

  it("follows the default float", {
    for (float_dtype in c("f32", "f64")) {
      with_default_dtypes(c(float = float_dtype), {
        expect_dtype(nv_dbinom(nv_array(c(0L, 1L)), 3, 0.5), float_dtype)
      })
    }
  })

  it("computes at the widest float among the parameter arrays", {
    for (name in names(families)) {
      for (float_dtype in c("f32", "f64")) {
        out <- families[[name]](nv_array(c(0L, 1L)), nv_array(c(1, 1), dtype = float_dtype))
        expect_dtype(out, float_dtype, info = paste(name, float_dtype))
      }
    }
    out <- nv_dnorm(
      nv_array(c(0L, 1L)),
      mean = nv_array(c(0, 0), dtype = "f32"),
      sd = nv_scalar(1, dtype = "f64")
    )
    expect_dtype(out, "f64")
  })

  it("ignores integer parameter arrays when choosing the float", {
    out <- nv_dbinom(
      nv_array(c(0L, 1L), dtype = "i64"),
      size = nv_scalar(3L, dtype = "ui64"),
      prob = 0.5
    )
    expect_dtype(out, default_float())
  })

  it("accepts an R integer", {
    expect_dtype(nv_dnorm(1L), default_float())
    expect_equal(as.vector(nv_dbinom(1L, 3, 0.5)), dbinom(1, 3, 0.5), tolerance = 1e-6)
  })

  it("is converted directly to f64, not through the default float", {
    # 2^30 + 1, which is not an f32. Rounding it through f32 moves the result
    # by about 1e-10 relative, so the tolerance has to be tighter than that.
    x <- 1073741825L
    expect_equal(
      as.vector(nv_dbinom(nv_array(x), size = nv_scalar(2^31, dtype = "f64"), prob = 0.5, log = TRUE)),
      dbinom(x, 2^31, 0.5, log = TRUE),
      tolerance = 1e-14
    )
  })

  it("agrees between eager and jit", {
    expect_eager_jit_equal_grid(families)
  })
})

describe("the float category", {
  it("nv_pnorm() and nv_qnorm() still need a 32- or 64-bit float", {
    # `f16` / `bf16` are float data types, so the general float check accepts
    # them -- but these two carry one coefficient set per width, and a narrower
    # float would silently take the `f64` set.
    #
    # The `jit()` is needed: eagerly, `nv_convert()` runs at once and has to
    # materialise a `bf16` buffer, which no backend does, so the call dies with
    # "Unsupported type: bf16" before it reaches the check. Under tracing the
    # array stays abstract and the check runs.
    expect_error(
      jit(function(x) nv_pnorm(nv_convert(x, "bf16")))(nv_array(c(0.5, 0.5))),
      "must be a 32- or 64-bit float data type"
    )
    expect_error(
      jit(function(x) nv_qnorm(nv_convert(x, "bf16")))(nv_array(c(0.5, 0.5))),
      "must be a 32- or 64-bit float data type"
    )
  })

  it("nv_dexp(), nv_pexp() and nv_qexp() need a 32- or 64-bit float", {
    # They carry one underflow threshold or rescaling per width; see above for
    # the `jit()`
    for (f in list(nv_dexp, nv_pexp, nv_qexp)) {
      expect_error(
        jit(function(x) f(nv_convert(x, "bf16")))(nv_array(c(0.5, 0.5))),
        "must be a 32- or 64-bit float data type"
      )
    }
  })

  it("nv_dbinom(), nv_pbinom() and nv_qbinom() still need a 32- or 64-bit float", {
    # As above: they carry one series length or tolerance per width.
    for (f in list(nv_dbinom, nv_pbinom, nv_qbinom)) {
      expect_error(
        jit(function(x) f(nv_convert(x, "bf16"), 2, 0.5))(nv_array(c(0.5, 0.5))),
        "must be a 32- or 64-bit float data type"
      )
    }
  })
})
