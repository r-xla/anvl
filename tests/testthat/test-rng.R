describe("nv_set_seed", {
  it("makes the draws from the global state reproducible", {
    local_global_rng()
    nv_set_seed(42L)
    a <- as_array(nv_runif(3L))
    b <- as_array(nv_runif(3L))
    expect_false(identical(a, b))
    nv_set_seed(42L)
    expect_identical(as_array(nv_runif(3L)), a)
    expect_identical(as_array(nv_runif(3L)), b)
  })

  it("draws what an explicit state from the same seed draws", {
    local_global_rng()
    nv_set_seed(42L)
    s1 <- nv_runif(3L, nv_rng_state(42L))
    s2 <- nv_rnorm(c(2L, 2L), s1$state, mean = 1)
    expect_identical(as_array(nv_runif(3L)), as_array(s1$values))
    expect_identical(as_array(nv_rnorm(c(2L, 2L), mean = 1)), as_array(s2$values))
  })

  it("returns only the sample for every sampler", {
    local_global_rng()
    nv_set_seed(1L)
    expect_shape(nv_runif(c(2L, 3L)), c(2L, 3L))
    expect_shape(nv_rnorm(4L), 4L)
    expect_shape(nv_rbinom(3L, size = 2L), 3L)
    values <- as_array(nv_sample_int(5L, n = 3L))
    expect_true(all(values %in% 1:3))
    values <- as_array(nv_sample(5L, x = nv_array(c(10, 20))))
    expect_true(all(values %in% c(10, 20)))
  })

  it("seeds from base R's RNG when no seed is set", {
    local_global_rng()
    draw <- function() {
      globals$seed <- NULL
      globals$rng_state <- NULL
      withr::local_seed(3L)
      as_array(nv_runif(3L))
    }
    expect_identical(draw(), draw())
  })

  it("rejects a seed that is not an integer", {
    expect_error(nv_set_seed(1.5), "integerish")
  })
})

describe("the global RNG state in jit", {
  it("is threaded through every draw of a call and continued by the next", {
    local_global_rng()
    f <- jit(function(x) list(a = x + nv_runif(2L), b = nv_rnorm(2L)))
    nv_set_seed(42L)
    out1 <- f(nv_array(c(0, 0)))
    out2 <- f(nv_array(c(0, 0)))

    s1 <- nv_runif(2L, nv_rng_state(42L))
    s2 <- nv_rnorm(2L, s1$state)
    s3 <- nv_runif(2L, s2$state)
    s4 <- nv_rnorm(2L, s3$state)
    expect_identical(as_array(out1$a), as_array(s1$values))
    expect_identical(as_array(out1$b), as_array(s2$values))
    expect_identical(as_array(out2$a), as_array(s3$values))
    expect_identical(as_array(out2$b), as_array(s4$values))
    expect_identical(as_array(globals$rng_state), as_array(s4$state))
  })

  it("is shared between jitted functions and eager draws", {
    local_global_rng()
    f <- jit(function() nv_runif(2L))
    nv_set_seed(5L)
    a <- as_array(f())
    b <- as_array(nv_runif(2L))
    nv_set_seed(5L)
    expect_identical(as_array(nv_runif(2L)), a)
    expect_identical(as_array(f()), b)
  })

  it("leaves the global state alone in functions that do not draw from it", {
    local_global_rng()
    nv_set_seed(5L)
    nv_runif(1L)
    state <- globals$rng_state
    f <- jit(function(x) x + 1L)
    f(nv_array(1L))
    expect_identical(globals$rng_state, state)
  })

  it("is not available outside jit", {
    expect_error(trace_fn(function() nv_runif(2L), list()), "called through `jit\\(\\)`")
  })
})

describe("the global RNG state in sub-graphs", {
  # The values of `n` consecutive draws of `shape` from an explicit state.
  draws <- function(seed, n, shape = 1L) {
    state <- nv_rng_state(seed)
    lapply(seq_len(n), function(i) {
      res <- nv_runif(shape, state)
      state <<- res$state
      as_array(res$values)
    })
  }

  it("is carried through the body of nv_while", {
    local_global_rng()
    f <- jit(function() {
      out <- nv_while(
        list(i = 0L, total = nv_scalar(0)),
        function(i, total) i < 3L,
        function(i, total) list(i = i + 1L, total = total * 10 + nv_runif(integer()))
      )
      list(total = out$total, after = nv_runif(integer()))
    })
    nv_set_seed(1L)
    out <- f()
    expected <- draws(1L, 4L, integer())
    expect_equal(as_array(out$total), (expected[[1L]] * 10 + expected[[2L]]) * 10 + expected[[3L]], tolerance = 1e-6)
    expect_identical(as_array(out$after), expected[[4L]])
  })

  it("is carried through the body of nv_scan", {
    local_global_rng()
    f <- jit(function() {
      out <- nv_scan(
        init = list(nv_scalar(0)),
        body = function(carry, x) list(carry = carry, out = nv_runif(1L)),
        steps = 3L
      )
      list(out = out$out, after = nv_runif(1L))
    })
    nv_set_seed(1L)
    out <- f()
    expected <- draws(1L, 4L)
    expect_identical(as.vector(as_array(out$out)), unlist(expected[1:3]))
    expect_identical(as_array(out$after), expected[[4L]])
  })

  it("is threaded through the branch nv_if takes", {
    local_global_rng()
    f <- jit(function(pred) {
      list(
        branch = nv_if(pred, function() nv_runif(1L), function() nv_fill(-1, 1L)),
        after = nv_runif(1L)
      )
    })
    expected <- draws(1L, 2L)
    nv_set_seed(1L)
    out <- f(nv_scalar(TRUE))
    expect_identical(as_array(out$branch), expected[[1L]])
    expect_identical(as_array(out$after), expected[[2L]])
    nv_set_seed(1L)
    out <- f(nv_scalar(FALSE))
    expect_identical(as.vector(as_array(out$branch)), -1)
    expect_identical(as_array(out$after), expected[[1L]])
  })

  it("is threaded through a function differentiated by gradient()", {
    local_global_rng()
    f <- jit(function(x) {
      grad <- gradient(function(x) nv_sum(x * nv_runif(2L)))(x)
      list(grad = grad$x, after = nv_runif(1L))
    })
    nv_set_seed(1L)
    out <- f(nv_array(c(1, 1)))
    state <- nv_rng_state(1L)
    first <- nv_runif(2L, state)
    expect_identical(as_array(out$grad), as_array(first$values))
    expect_identical(as_array(out$after), as_array(nv_runif(1L, first$state)$values))
  })

  it("is not available in the condition of nv_while", {
    local_global_rng()
    f <- jit(function() {
      nv_while(list(i = 0L), function(i) nv_runif(integer()) < 0.5, function(i) list(i = i + 1L))
    })
    expect_error(f(), "cannot be drawn from here")
  })

  it("is not available in the function of a reduction", {
    local_global_rng()
    f <- jit(function(x) {
      nv_runif(1L)
      prim_reduce(x, nv_scalar(0), function(a, b) a + b + nv_runif(integer()), axes = 1L)
    })
    expect_error(f(nv_array(c(1, 2))), "cannot be drawn from here")
  })
})

describe("the global RNG state across devices", {
  dev0 <- nv_device("cpu:0")
  dev1 <- nv_device("cpu:1")

  it("follows the call to the device of its array inputs", {
    local_global_rng()
    f <- jit(function(x) x + nv_runif(3L))
    s1 <- nv_runif(3L, nv_rng_state(7L))
    s2 <- nv_runif(3L, s1$state)

    nv_set_seed(7L)
    out <- f(nv_array(c(0, 0, 0), device = dev0))
    expect_true(eq_device(device(globals$rng_state), dev0))
    expect_identical(as_array(out), as_array(s1$values))

    # the next call runs on another device and continues the sequence there
    out <- f(nv_array(c(0, 0, 0), device = dev1))
    expect_true(eq_device(device(out), dev1))
    expect_true(eq_device(device(globals$rng_state), dev1))
    expect_identical(as_array(out), as_array(s2$values))

    f(nv_array(c(0, 0, 0), device = dev0))
    expect_true(eq_device(device(globals$rng_state), dev0))
  })

  it("follows the device a jitted function is fixed to", {
    local_global_rng()
    f <- jit(function() nv_runif(3L), device = dev1)
    globals$rng_state <- nv_rng_state(7L, device = dev0)
    out <- f()
    expect_true(eq_device(device(out), dev1))
    expect_true(eq_device(device(globals$rng_state), dev1))
    expect_identical(as_array(out), as_array(nv_runif(3L, nv_rng_state(7L))$values))
  })

  it("follows the default device when a call has no array input", {
    local_global_rng()
    globals$rng_state <- nv_rng_state(7L, device = dev0)
    out <- with_default_device(dev1, nv_runif(3L))
    expect_true(eq_device(device(out), dev1))
    expect_true(eq_device(device(globals$rng_state), dev1))
    expect_identical(as_array(out), as_array(nv_runif(3L, nv_rng_state(7L))$values))
  })

  it("follows a call whose device comes from a closed-over array", {
    local_global_rng()
    x1 <- nv_array(c(0, 0, 0), device = dev1)
    f <- jit(function() x1 + nv_runif(3L))
    globals$rng_state <- nv_rng_state(7L, device = dev0)
    out <- f()
    expect_true(eq_device(device(out), dev1))
    expect_true(eq_device(device(globals$rng_state), dev1))
    expect_identical(as_array(out), as_array(nv_runif(3L, nv_rng_state(7L))$values))
  })

  it("follows a call whose device comes from a static argument", {
    local_global_rng()
    f <- jit(function(device) nv_fill(0, 3L, device = device) + nv_runif(3L), static = "device")
    s1 <- nv_runif(3L, nv_rng_state(7L))
    s2 <- nv_runif(3L, s1$state)
    globals$rng_state <- nv_rng_state(7L, device = dev0)
    out <- f(dev1)
    expect_true(eq_device(device(globals$rng_state), dev1))
    expect_identical(as_array(out), as_array(s1$values))
    out <- f(dev0)
    expect_true(eq_device(device(globals$rng_state), dev0))
    expect_identical(as_array(out), as_array(s2$values))
  })

  it("is seeded on the device of the call that first draws from it", {
    local_global_rng()
    nv_set_seed(7L)
    with_default_device(dev1, nv_runif(1L))
    expect_true(eq_device(device(globals$rng_state), dev1))
  })
})
