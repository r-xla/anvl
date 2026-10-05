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

  it("names the operand when it is not a float", {
    # Reported as a failure to bring `mean` to the operand's data type before.
    expect_error(nv_qnorm(nv_array(1L)), "`p` must be a float data type")
    expect_error(nv_pnorm(nv_array(1L)), "`q` must be a float data type")
    expect_error(nv_dnorm(nv_array(1L)), "`x` must be a float data type")
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

  it("names the operand when it is not a float", {
    # Reported as a failure to bring `min` to the operand's data type before.
    expect_error(nv_qunif(nv_array(1L)), "`p` must be a float data type")
    expect_error(nv_punif(nv_array(1L)), "`q` must be a float data type")
    expect_error(nv_dunif(nv_array(1L)), "`x` must be a float data type")
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
      q <- nv_array(c(r, 2 * r), dtype = dt)
      rate <- nv_scalar(r, dtype = dt)
      want <- log(as.vector(rate)) + log(as.vector(q))
      expect_equal_elementwise(
        as.vector(nv_pexp(q, rate, log_p = TRUE)),
        want,
        tolerance = if (dt == "f32") 3e-7 else 1e-15
      )
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
    q <- nv_array(c(1e-200, 2e-200), dtype = "f64")
    rate <- nv_scalar(1e-200, dtype = "f64")
    f <- function(q, rate) nv_sum(nv_pexp(q, rate, log_p = TRUE))
    g <- jit(gradient(f, wrt = c("q", "rate")))(q, rate)
    expect_equal(as.vector(g$q) / (1 / as.vector(q)), c(1, 1))
    expect_equal(as.vector(g$rate) / 2e200, 1)
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

  it("d/dp at the zero end of the probability scale is the limit from inside, for either sign of zero", {
    # the quantile tends to Inf as log p -> 0 from below, and as p -> 0 from above
    zeros <- c(0, as.numeric("-0"))
    f <- function(p, rate, lower_tail, log_p) nv_sum(nv_qexp(p, rate, lower_tail, log_p))
    grad <- jit(gradient(f, wrt = "p"), static = c("lower_tail", "log_p"))
    g <- function(lt, lp) as.vector(grad(as_f64(zeros), as_f64(c(2, 2)), lt, lp)[[1L]])
    expect_equal(g(TRUE, TRUE), c(Inf, Inf))
    expect_equal(g(FALSE, FALSE), c(-Inf, -Inf))
  })

  it("the zero quantile is +0 and keeps its one-sided d/dp", {
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
      expect_identical(1 / value, c(Inf, Inf))
      g <- grad(p, rate, cs$lower_tail, cs$log_p)
      expect_equal(as.vector(g$p), rep(cs$d_p, 2))
      expect_equal(as.vector(g$rate), c(0, 0))
    }
  })

  it("converts rate to the dtype of p", {
    out <- nv_qexp(nv_array(c(0.25, 0.75), dtype = "f32"), rate = 2L)
    expect_equal(dtype(out), as_dtype("f32"))
  })

  it("names the operand when it is not a float", {
    expect_error(nv_qexp(nv_array(1L)), "`p` must be a float data type")
    expect_error(nv_pexp(nv_array(1L)), "`q` must be a float data type")
    expect_error(nv_dexp(nv_array(1L)), "`x` must be a float data type")
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
})
