# statistical validity checks are in inst/random

describe("the samplers", {
  it("return only the sample", {
    local_nv_seed(1L)
    expect_shape(nv_runif(c(2L, 3L)), c(2L, 3L))
    expect_shape(nv_rnorm(4L), 4L)
    expect_shape(nv_rbinom(3L, size = 2L), 3L)
    values <- as_array(nv_sample_int(5L, n = 3L))
    expect_true(all(values %in% 1:3))
    values <- as_array(nv_sample(5L, x = nv_array(c(10, 20))))
    expect_true(all(values %in% c(10, 20)))
  })

  it("draw the same in a jitted function as eagerly, and reproducibly", {
    pop <- nv_array(c(10, 20, 30))
    draw <- function() {
      list(
        nv_rbinom(5L, size = 3L),
        nv_sample_int(5L, n = 6L),
        nv_sample(5L, pop),
        nv_rnorm(3L),
        nv_runif(3L)
      )
    }
    eager <- with_nv_seed(1L, lapply(draw(), as_array))
    expect_identical(with_nv_seed(1L, lapply(jit(draw)(), as_array)), eager)
    expect_identical(with_nv_seed(1L, lapply(draw(), as_array)), eager)
    # a second set of draws continues the sequence rather than repeating it
    again <- with_nv_seed(1L, {
      draw()
      lapply(draw(), as_array)
    })
    expect_false(identical(again[[2L]], eager[[2L]]) && identical(again[[3L]], eager[[3L]]))
    expect_false(identical(again[[4L]], eager[[4L]]))
  })

  it("reject a data type they cannot draw at", {
    # A float, but not one the bit manipulation can build.
    expect_error(nv_rnorm(dtype = "bf16", shape = 2L), "must be a 32- or 64-bit float data type")
    expect_error(nv_rnorm(dtype = "i32", shape = 2L), "must be a float data type")
    expect_error(nv_rbinom(4L, dtype = "bool"), "must be a numeric data type")
    expect_error(nv_sample_int(4L, n = 2L, dtype = "bool"), "must be a numeric data type")
  })
})

describe("nv_rnorm", {
  it("draws a sample of the requested shape and data type", {
    local_nv_seed(1L)
    out <- nv_rnorm(dtype = "f32", shape = c(2, 3))
    expect_dtype(out, "f32")
    expect_shape(out, c(2L, 3L))
    # an odd number of draws
    expect_shape(nv_rnorm(dtype = "f32", shape = c(3, 3)), c(3L, 3L))
    out <- nv_rnorm(dtype = "f64", shape = c(2L, 3L), mean = 10, sd = 9)
    expect_shape(out, c(2L, 3L))
    expect_dtype(out, "f64")
  })

  it("is NaN for an invalid mean or sd, like rnorm()", {
    local_nv_seed(1L)
    means <- c(0, Inf, NaN, 0, 0, 0, 3)
    sds <- c(1, 1, 1, -1, Inf, NaN, 0)
    out <- as.vector(nv_rnorm(7L, dtype = "f64", mean = nv_array(means), sd = nv_array(sds)))
    expect_equal(is.nan(out), suppressWarnings(is.nan(rnorm(7L, means, sds))))
    # an infinite mean or a zero sd is the mean
    expect_equal(out[c(2L, 7L)], c(Inf, 3))
  })

  it("accepts arrayish mean and sd", {
    local_nv_seed(1L)
    # an elementwise mean of the same shape as the sample
    means <- nv_array(matrix(c(-1000, 1000, -1000, 1000, -1000, 1000), nrow = 2))
    out <- nv_rnorm(c(2, 3), dtype = "f64", mean = means, sd = 1)
    values <- as_array(out)
    expect_shape(out, c(2L, 3L))
    # sd is 1, so each draw stays near its own mean
    expect_true(all(values[1, ] < -900))
    expect_true(all(values[2, ] > 900))

    # an elementwise sd
    sds <- nv_array(matrix(c(1e-6, 1e6, 1e-6, 1e6, 1e-6, 1e6), nrow = 2))
    spread <- as_array(nv_rnorm(c(2, 3), dtype = "f64", sd = sds))
    expect_true(all(abs(spread[1, ]) < 1))
    expect_true(all(abs(spread[2, ]) > 1))

    # mean/sd may be traced under jit
    f <- jit(function(m, sdev) nv_rnorm(c(2, 3), dtype = "f64", mean = m, sd = sdev))
    traced <- as_array(f(nv_scalar(1000, dtype = "f64"), nv_scalar(1, dtype = "f64")))
    expect_true(all(traced > 900))

    # an odd number of draws still reshapes correctly with arrayish mean
    odd_means <- nv_array(matrix(rep(c(-1000, 0, 1000), each = 3), nrow = 3))
    odd <- as_array(nv_rnorm(c(3, 3), dtype = "f64", mean = odd_means))
    expect_true(all(odd[, 1] < -900) && all(abs(odd[, 2]) < 100) && all(odd[, 3] > 900))
  })

  it("refuses a mean or sd of another shape than the sample's", {
    # also for a scalar sample, which would otherwise take the shape of `mean`/`sd`
    expect_error(
      nv_rnorm(c(2, 3), mean = nv_array(matrix(0, 2, 1))),
      "must be a scalar or have the shape of the sample"
    )
    expect_error(
      nv_rnorm(integer(), sd = nv_array(c(1, 2, 3))),
      "must be a scalar or have the shape of the sample"
    )
  })

  it("takes the sample's dtype from mean and sd", {
    draw <- function(...) dtype(nv_rnorm(2L, ...))

    # Neither brings a data type, so the sample falls back to the default float
    # rather than to whatever R stores its numbers as.
    expect_equal(draw(), default_float())
    expect_equal(draw(mean = 0L, sd = 1L), default_float())

    # Either one that has a data type gives the sample its own.
    expect_equal(draw(mean = nv_scalar(1, dtype = "f64")), as_dtype("f64"))
    expect_equal(draw(sd = nv_scalar(1, dtype = "f64")), as_dtype("f64"))
    expect_equal(
      draw(mean = nv_scalar(1, dtype = "f64"), sd = nv_scalar(1, dtype = "f32")),
      as_dtype("f64")
    )
    # An R value keeps the sample a float even where the other is an integer.
    expect_equal(draw(mean = nv_scalar(1L)), default_float())

    # `dtype` is the caller's word over the arguments', and is refused where the
    # sample could not hold them.
    expect_equal(draw(dtype = "f64"), as_dtype("f64"))
    expect_error(
      draw(dtype = "f32", mean = nv_scalar(1, dtype = "f64")),
      "Cannot bring `mean` to data type \"f32\""
    )

    # Arguments that agree on a data type the generator cannot draw at say so.
    expect_error(
      draw(mean = nv_scalar(1L), sd = nv_scalar(2L)),
      "must be a float data type"
    )
    expect_error(draw(mean = nv_scalar(1L), sd = nv_scalar(2L)), "Pass `dtype`")
  })
})

describe("nv_runif", {
  it("draws from the open interval", {
    local_nv_seed(1L)
    out <- nv_runif(dtype = "f32", shape = c(3, 4), min = -1, max = 1)
    expect_shape(out, c(3L, 4L))
    expect_dtype(out, "f32")
    values <- as_array(out)
    expect_true(all(values > -1 & values < 1))
  })

  it("returns min where min equals max, and advances the state regardless", {
    out <- with_nv_seed(1L, nv_runif(c(2, 3), min = 5, max = 5))
    # `min`/`max` may be traced, so the draw is made and the state advanced as
    # for any other interval.
    advanced <- with_nv_seed(1L, {
      nv_runif(c(2, 3), min = 5, max = 5)
      nv_get_random_state()
    })
    plain <- with_nv_seed(1L, {
      nv_runif(c(2, 3))
      nv_get_random_state()
    })
    expect_identical(advanced, plain)
    expect_shape(out, c(2L, 3L))
    expect_true(all(as.vector(out) == 5))
  })

  it("accepts arrayish min and max", {
    local_nv_seed(1L)
    # an elementwise interval of the same shape as the sample
    lower <- nv_array(matrix(c(0, 10, 100, 1000, 10000, 100000), nrow = 2), dtype = "f64")
    out <- nv_runif(c(2, 3), min = lower, max = lower + 1)
    values <- as_array(out)
    expect_shape(out, c(2L, 3L))
    expect_dtype(out, "f64")
    expect_true(all(values > as_array(lower) & values < as_array(lower) + 1))

    # a scalar bound combines with an array one, and the draws are those of the
    # standard uniform, scaled and shifted elementwise
    upper <- nv_array(matrix(1:6, nrow = 2), dtype = "f64")
    u <- with_nv_seed(1L, as_array(nv_runif(c(2, 3), dtype = "f64")))
    expect_equal(with_nv_seed(1L, as_array(nv_runif(c(2, 3), min = 0, max = upper))), u * 1:6)

    # min/max may be traced under jit
    f <- jit(function(a, b) nv_runif(c(2, 3), min = a, max = b))
    traced <- as_array(f(nv_scalar(1000, dtype = "f64"), nv_scalar(1001, dtype = "f64")))
    expect_true(all(traced > 1000 & traced < 1001))
  })

  it("refuses a min or max of another shape than the sample's", {
    expect_error(
      nv_runif(c(2, 3), max = nv_array(matrix(1, 2, 1))),
      "must be a scalar or have the shape of the sample"
    )
    expect_error(
      nv_runif(integer(), min = nv_array(c(0, 1))),
      "must be a scalar or have the shape of the sample"
    )
  })

  it("is NaN for an invalid interval, like runif()", {
    local_nv_seed(1L)
    lower <- c(0, 5, 1, -Inf, NaN, 10)
    upper <- c(1, 5, 0, 1, 1, Inf)
    out <- nv_runif(6L, dtype = "f64", min = nv_array(lower), max = nv_array(upper))
    expect_equal(is.nan(as.vector(out)), suppressWarnings(is.nan(runif(6L, lower, upper))))
    expect_equal(as.vector(out)[2L], 5)
  })

  it("differentiates with respect to min and max", {
    u <- with_nv_seed(1L, as.vector(nv_runif(c(2, 3), dtype = "f64")))
    g <- jit(gradient(function(a, b) sum(nv_runif(c(2, 3), min = a, max = b))))
    grads <- with_nv_seed(1L, g(nv_array(matrix(-2, 2, 3), dtype = "f64"), nv_array(matrix(3, 2, 3), dtype = "f64")))
    expect_equal(as.vector(grads[[1L]]), 1 - u)
    expect_equal(as.vector(grads[[2L]]), u)
  })
})

describe("nv_rbinom", {
  it("draws counts of successes of the requested shape and data type", {
    local_nv_seed(1L)
    out <- nv_rbinom(dtype = "i32", shape = c(2, 5))
    expect_shape(out, c(2L, 5L))
    expect_dtype(out, "i32")
    expect_true(all(as.vector(out) %in% c(0L, 1L)))

    out <- nv_rbinom(dtype = "f32", shape = 10L)
    expect_dtype(out, "f32")
    expect_shape(out, 10L)

    # a shape that is not a multiple of 8
    expect_shape(nv_rbinom(dtype = "i32", shape = c(3, 3)), c(3L, 3L))

    values <- as.vector(nv_rbinom(50L, size = 4L))
    expect_true(all(values >= 0L & values <= 4L))
  })

  it("draws zeros for zero trials, like rbinom()", {
    local_nv_seed(1L)
    expect_identical(as.vector(nv_rbinom(3L, size = 0L, dtype = "i32")), c(0L, 0L, 0L))
  })

  it("refuses a number of trials its data type cannot hold", {
    expect_error(nv_rbinom(5L, size = 300L, dtype = "ui8"), "must fit into the data type")
    expect_error(nv_rbinom(5L, size = 128L, dtype = "i8"), "must fit into the data type")
    local_nv_seed(1L)
    expect_dtype(nv_rbinom(5L, size = 255L, dtype = "ui8"), "ui8")
  })
})

describe("nv_sample_int", {
  it("draws the integers 1 to n", {
    local_nv_seed(1L)
    out <- nv_sample_int(n = 6L, shape = 10L)
    expect_shape(out, 10L)
    expect_dtype(out, default_int())
    values <- as.vector(out)
    expect_true(all(values >= 1L & values <= 6L))

    expect_shape(nv_sample_int(n = 4L, shape = c(2L, 3L)), c(2L, 3L))
    expect_dtype(nv_sample_int(n = 6L, shape = 4L, dtype = "i64"), "i64")
  })

  it("reaches the last integer without over-representing the first", {
    local_nv_seed(1L)
    values <- as.vector(nv_sample_int(n = 6L, shape = 5000L))
    expect_setequal(unique(values), 1:6)
    expect_true(all(abs(as.numeric(table(values)) / 5000 - 1 / 6) < 0.02))
  })

  it("always draws the one integer of a population of size one", {
    local_nv_seed(1L)
    expect_true(all(as.vector(nv_sample_int(n = 1L, shape = 20L)) == 1L))
  })

  it("refuses a population its data type cannot hold", {
    expect_error(nv_sample_int(n = 300L, shape = 4L, dtype = "ui8"), "must fit into the data type")
  })

  it("composes inside jit", {
    local_nv_seed(1L)
    # `n` is static, so it may be a literal in the traced body
    f <- jit(function() nv_sample_int(8L, 6L))
    values <- as.vector(f())
    expect_true(all(values >= 1L & values <= 6L))
  })
})

describe("nv_sample", {
  pop <- nv_array(c(10, 20, 30))

  it("draws from the population, at its data type", {
    local_nv_seed(1L)
    out <- nv_sample(x = pop, shape = 8L)
    expect_shape(out, 8L)
    expect_dtype(out, dtype(pop))
    expect_true(all(as.vector(out) %in% c(10, 20, 30)))
    expect_shape(nv_sample(x = pop, shape = c(2L, 3L)), c(2L, 3L))
  })

  it("reaches every element of the population", {
    local_nv_seed(1L)
    many <- as.vector(nv_sample(x = pop, shape = 500L))
    expect_setequal(unique(many), c(10, 20, 30))
  })

  it("takes a length-one population as the population, not as a count", {
    local_nv_seed(1L)
    expect_true(all(as.vector(nv_sample(x = nv_array(6), shape = 5L)) == 6))
  })

  it("refuses a population that is not 1-D or is empty", {
    expect_error(
      nv_sample(x = nv_array(matrix(1:6, nrow = 2)), shape = 3L),
      "must be a 1-D array"
    )
    expect_error(nv_sample(3L, x = nv_array(numeric(0), shape = 0L)), "must not be empty")
  })

  it("takes the population as a traced input", {
    local_nv_seed(1L)
    g <- jit(function(p) nv_sample(6L, p))
    expect_true(all(as.vector(g(pop)) %in% c(10, 20, 30)))
  })
})

describe("box_muller", {
  it("maps a uniform of exactly 0 to a finite sample", {
    for (dt in c("f32", "f64")) {
      u1 <- nv_array(c(0, 0.5), dtype = dt)
      u2 <- nv_array(c(0.125, 0), dtype = dt)
      z <- as.vector(as_array(box_muller(u1, u2)))
      expect_true(all(is.finite(z)))
      expect_equal(z, c(0, 0, 0, sqrt(-2 * log(0.5))), tolerance = 1e-6)
    }
  })
})
