# The values of `n` consecutive eager draws of `shape` from the global state
# seeded with `seed`.
draws <- function(seed, n, shape = 1L) {
  with_nv_seed(seed, lapply(seq_len(n), function(i) as_array(nv_runif(shape))))
}

describe("nv_set_seed", {
  it("makes the draws from the global state reproducible", {
    local_nv_seed(NULL)
    nv_set_seed(42L)
    a <- as_array(nv_runif(3L))
    b <- as_array(nv_runif(3L))
    expect_false(identical(a, b))
    nv_set_seed(42L)
    expect_identical(as_array(nv_runif(3L)), a)
    expect_identical(as_array(nv_runif(3L)), b)
  })

  it("starts from the seed as the key and a zero counter", {
    local_nv_seed(-1L)
    expect_identical(as.numeric(as_array(globals$rng_state)), c(2^32 - 1, 0))
  })

  it("is not needed: an unset state is derived from base R's RNG", {
    draw <- function() {
      local_nv_seed(NULL)
      withr::local_seed(3L)
      list(as_array(nv_runif(3L)), as_array(nv_runif(3L)))
    }
    first <- draw()
    expect_identical(draw(), first)
    expect_false(identical(first[[1L]], first[[2L]]))
  })

  it("derives an unset state from base R's seed", {
    draw <- function(seed) {
      local_nv_seed(NULL)
      withr::local_seed(seed)
      as_array(nv_runif(3L))
    }
    expect_false(identical(draw(1L), draw(2L)))
  })

  it("derives an unset state from base R's RNG only once", {
    local_nv_seed(NULL)
    withr::local_seed(3L)
    a <- as_array(nv_runif(3L))
    # base R's seed no longer reseeds the state
    withr::local_seed(3L)
    expect_false(identical(as_array(nv_runif(3L)), a))
    stats::runif(1L)
    nv_set_seed(NULL)
    withr::local_seed(3L)
    expect_identical(as_array(nv_runif(3L)), a)
  })

  it("rejects a seed that is not an integer", {
    expect_error(nv_set_seed(1.5), "integerish")
  })

  it("is refused by a backend that cannot draw", {
    skip_if_no_quickr()
    local_backend("quickr")
    expect_error(nv_set_seed(1L), "does not support random draws")
  })

  it("cannot be called in a traced function", {
    f <- jit(function() {
      nv_set_seed(1L)
      nv_runif(1L)
    })
    expect_error(f(), "cannot be seeded in a function traced")
  })
})

describe("with_nv_seed", {
  it("seeds the global state for `code` and restores it afterwards", {
    local_nv_seed(5L)
    before <- as_array(globals$rng_state)
    a <- with_nv_seed(1L, as_array(nv_runif(2L)))
    expect_identical(as_array(globals$rng_state), before)
    expect_identical(with_nv_seed(1L, as_array(nv_runif(2L))), a)
  })
})

describe("nv_get_rng_state", {
  it("is NULL while the global state is not set", {
    local_nv_seed(NULL)
    expect_null(nv_get_rng_state())
  })

  it("returns a copy of the global state that later draws leave intact", {
    local_nv_seed(-1L)
    nv_runif(3L)
    state <- nv_get_rng_state()
    expected <- as.numeric(as_array(state))
    nv_runif(3L)
    expect_identical(as.numeric(as_array(state)), expected)
  })

  it("lets nv_set_rng_state() continue the sequence where it stopped", {
    local_nv_seed(7L)
    nv_runif(2L)
    state <- nv_get_rng_state()
    a <- as_array(nv_runif(3L))
    nv_set_rng_state(state)
    expect_identical(as_array(nv_runif(3L)), a)
    # the state passed is copied, so it can be used again
    nv_set_rng_state(state)
    expect_identical(as_array(nv_runif(3L)), a)
  })

  it("cannot be called in a traced function", {
    f <- jit(function() {
      nv_get_rng_state()
      nv_runif(1L)
    })
    expect_error(f(), "cannot be read in a function traced")
  })
})

describe("nv_set_rng_state", {
  it("unsets the global state with NULL", {
    local_nv_seed(1L)
    nv_set_rng_state(NULL)
    expect_null(nv_get_rng_state())
  })

  it("rejects what is not an RNG state", {
    expect_error(nv_set_rng_state(1L), "must be an RNG state returned by")
    expect_error(nv_set_rng_state(nv_array(c(1, 2), dtype = "f32")), "must be an RNG state")
    expect_error(nv_set_rng_state(nv_array(c(1, 2, 3), dtype = "ui64")), "must be an RNG state")
  })
})

describe("local_nv_seed", {
  it("restores the global state when the calling scope exits", {
    local_nv_seed(5L)
    before <- as_array(globals$rng_state)
    draw <- function() {
      local_nv_seed(1L)
      as_array(nv_runif(2L))
    }
    a <- draw()
    expect_identical(as_array(globals$rng_state), before)
    expect_identical(draw(), a)
  })
})

describe("the global RNG state in jit", {
  it("is threaded through every draw of a call and continued by the next", {
    local_nv_seed(NULL)
    f <- jit(function(x) list(a = x + nv_runif(2L), b = nv_rnorm(2L)))
    outs <- with_nv_seed(42L, {
      list(f(nv_array(c(0, 0))), f(nv_array(c(0, 0))), as_array(globals$rng_state))
    })

    # the same draws, one after the other outside of jit
    with_nv_seed(42L, {
      expect_identical(as_array(outs[[1L]]$a), as_array(nv_runif(2L)))
      expect_identical(as_array(outs[[1L]]$b), as_array(nv_rnorm(2L)))
      expect_identical(as_array(outs[[2L]]$a), as_array(nv_runif(2L)))
      expect_identical(as_array(outs[[2L]]$b), as_array(nv_rnorm(2L)))
      expect_identical(as_array(globals$rng_state), outs[[3L]])
    })
  })

  it("is taken by a function traced only once", {
    local_nv_seed(1L)
    n_traces <- 0L
    f <- jit(function() {
      n_traces <<- n_traces + 1L
      nv_runif(2L)
    })
    f()
    f()
    expect_identical(n_traces, 1L)
  })

  it("is shared between jitted functions and eager draws", {
    f <- jit(function() nv_runif(2L))
    expected <- draws(5L, 2L, 2L)
    expect_identical(with_nv_seed(5L, list(as_array(f()), as_array(nv_runif(2L)))), expected)
    expect_identical(with_nv_seed(5L, list(as_array(nv_runif(2L)), as_array(f()))), expected)
  })

  it("leaves the global state alone in functions that do not draw from it", {
    local_nv_seed(5L)
    nv_runif(1L)
    state <- globals$rng_state
    f <- jit(function(x) x + 1L)
    f(nv_array(1L))
    expect_identical(globals$rng_state, state)
  })

  it("is a side effect of a graph traced by trace_fn()", {
    graph <- trace_fn(nv_runif, list(1L))
    expect_length(graph$inputs, 0L)
    expect_length(graph$outputs, 1L)
    expect_named(graph$side_effects, "rng")
    state <- graph$side_effects$rng$input$aval
    expect_identical(as.character(dtype(state)), "ui64")
    expect_identical(shape(state), 2L)
    expect_identical(as.character(dtype(graph$side_effects$rng$output$aval)), "ui64")
  })

  it("is updated in place", {
    local_nv_seed(1L)
    before <- globals$rng_state
    nv_runif(1L)
    expect_error(as_array(before), "deleted or donated")
  })

  it("is left out of a graph traced by trace_fn() that does not draw", {
    graph <- trace_fn(function(x) x + 1, list(nv_array(1)))
    expect_length(graph$inputs, 1L)
    expect_length(graph$outputs, 1L)
    expect_length(graph$side_effects, 0L)
  })
})

describe("the global RNG state in sub-graphs", {
  it("is carried through the body of nv_while", {
    f <- jit(function() {
      out <- nv_while(
        list(i = 0L, total = nv_scalar(0)),
        function(i, total) i < 3L,
        function(i, total) list(i = i + 1L, total = total * 10 + nv_runif(integer()))
      )
      list(total = out$total, after = nv_runif(integer()))
    })
    expected <- draws(1L, 4L, integer())
    out <- with_nv_seed(1L, f())
    expect_equal(as_array(out$total), (expected[[1L]] * 10 + expected[[2L]]) * 10 + expected[[3L]], tolerance = 1e-6)
    expect_identical(as_array(out$after), expected[[4L]])
  })

  it("is carried through the body of nv_scan in reverse", {
    f <- jit(function(xs) {
      nv_scan(
        init = list(nv_scalar(0)),
        body = function(carry, x) list(carry = carry, out = x + nv_runif(1L)),
        xs = xs,
        reverse = TRUE
      )$out
    })
    expected <- unlist(draws(1L, 3L))
    out <- with_nv_seed(1L, f(nv_array(matrix(c(0, 10, 20), ncol = 1L))))
    # the body runs from the last step to the first, drawing in that order
    expect_equal(as.vector(as_array(out)), c(0, 10, 20) + rev(expected), tolerance = 1e-6)
  })

  it("is carried through the body of nv_scan", {
    f <- jit(function() {
      out <- nv_scan(
        init = list(nv_scalar(0)),
        body = function(carry, x) list(carry = carry, out = nv_runif(1L)),
        steps = 3L
      )
      list(out = out$out, after = nv_runif(1L))
    })
    expected <- draws(1L, 4L)
    out <- with_nv_seed(1L, f())
    expect_identical(as.vector(as_array(out$out)), unlist(expected[1:3]))
    expect_identical(as_array(out$after), expected[[4L]])
  })

  it("is threaded through nv_if when only the false branch draws", {
    f <- jit(function(pred) {
      list(branch = nv_if(pred, function() nv_fill(-1, 1L), function() nv_runif(1L)), after = nv_runif(1L))
    })
    expected <- draws(1L, 2L)
    out <- with_nv_seed(1L, f(FALSE))
    expect_identical(as_array(out$branch), expected[[1L]])
    expect_identical(as_array(out$after), expected[[2L]])
  })

  it("is threaded through the branch nv_if takes", {
    f <- jit(function(pred) {
      list(
        branch = nv_if(pred, function() nv_runif(1L), function() nv_fill(-1, 1L)),
        after = nv_runif(1L)
      )
    })
    expected <- draws(1L, 2L)
    out <- with_nv_seed(1L, f(nv_scalar(TRUE)))
    expect_identical(as_array(out$branch), expected[[1L]])
    expect_identical(as_array(out$after), expected[[2L]])
    out <- with_nv_seed(1L, f(nv_scalar(FALSE)))
    expect_identical(as.vector(as_array(out$branch)), -1)
    expect_identical(as_array(out$after), expected[[1L]])
  })

  it("is threaded through a function differentiated by gradient()", {
    f <- jit(function(x) {
      grad <- gradient(function(x) nv_sum(x * nv_runif(2L)))(x)
      list(grad = grad$x, after = nv_runif(1L))
    })
    out <- with_nv_seed(1L, f(nv_array(c(1, 1))))
    with_nv_seed(1L, {
      expect_identical(as_array(out$grad), as_array(nv_runif(2L)))
      expect_identical(as_array(out$after), as_array(nv_runif(1L)))
    })
  })

  it("is taken when only a sub-graph draws from it", {
    f <- jit(function() {
      nv_while(
        list(i = 0L, x = nv_fill(0, 1L)),
        function(i, x) i < 2L,
        function(i, x) list(i = i + 1L, x = nv_if(i < 1L, function() nv_runif(1L), function() x + nv_runif(1L)))
      )$x
    })
    expected <- draws(1L, 3L)
    with_nv_seed(1L, {
      expect_equal(as_array(f()), expected[[1L]] + expected[[2L]], tolerance = 1e-6)
      # the call advanced the global state by both draws
      expect_identical(as_array(nv_runif(1L)), expected[[3L]])
    })
  })

  it("is threaded through an nv_if nested in an nv_if", {
    f <- jit(function(p, q) {
      branch <- nv_if(
        p,
        function() nv_if(q, function() nv_runif(1L), function() nv_fill(-1, 1L)),
        function() nv_fill(-2, 1L)
      )
      list(branch = branch, after = nv_runif(1L))
    })
    expected <- draws(1L, 2L)
    out <- with_nv_seed(1L, f(TRUE, TRUE))
    expect_identical(as_array(out$branch), expected[[1L]])
    expect_identical(as_array(out$after), expected[[2L]])
    # where no branch on the way draws, the state is left for the next draw
    for (pq in list(c(TRUE, FALSE), c(FALSE, TRUE))) {
      out <- with_nv_seed(1L, f(pq[[1L]], pq[[2L]]))
      expect_identical(as_array(out$after), expected[[1L]])
    }
  })

  it("is carried through an nv_while nested in the body of nv_scan", {
    f <- jit(function() {
      out <- nv_scan(
        init = nv_scalar(0),
        body = function(carry, x) {
          inner <- nv_while(
            list(i = 0L, total = nv_scalar(0)),
            function(i, total) i < 2L,
            function(i, total) list(i = i + 1L, total = total * 10 + nv_runif(integer()))
          )$total
          list(carry = carry, out = inner)
        },
        steps = 2L
      )
      list(out = out$out, after = nv_runif(integer()))
    })
    expected <- unlist(draws(1L, 5L, integer()))
    out <- with_nv_seed(1L, f())
    expect_equal(
      as.vector(as_array(out$out)),
      c(expected[[1L]] * 10 + expected[[2L]], expected[[3L]] * 10 + expected[[4L]]),
      tolerance = 1e-6
    )
    expect_identical(as_array(out$after), expected[[5L]])
  })

  it("is not available in the condition of nv_while", {
    f <- jit(function() {
      nv_while(list(i = 0L), function(i) nv_runif(integer()) < 0.5, function(i) list(i = i + 1L))
    })
    expect_error(f(), "cannot be used here")
  })

  it("is not available in the function of a reduction", {
    f <- jit(function(x) {
      nv_runif(1L)
      prim_reduce(x, nv_scalar(0), function(a, b) a + b + nv_runif(integer()), axes = 1L)
    })
    expect_error(f(nv_array(c(1, 2))), "cannot be used here")
  })

  it("is not available in the update function of a scatter", {
    f <- jit(function(x) {
      prim_scatter(
        x,
        nv_array(matrix(1L, 1L, 1L)),
        nv_array(5),
        update_window_axes = integer(),
        inserted_window_axes = 1L,
        x_batching_axes = integer(),
        scatter_indices_batching_axes = integer(),
        scatter_axes_to_x_axes = 1L,
        index_vector_axis = 2L,
        update_fn = function(old, new) old + new + nv_runif(integer())
      )
    })
    expect_error(f(nv_array(c(1, 2))), "cannot be used here")
  })
})

describe("the global RNG state across devices", {
  dev0 <- nv_device("cpu:0")
  dev1 <- nv_device("cpu:1")

  it("follows the call to the device of its array inputs", {
    f <- jit(function(x) x + nv_runif(3L))
    expected <- draws(7L, 2L, 3L)

    local_nv_seed(7L)
    out <- f(nv_array(c(0, 0, 0), device = dev0))
    expect_true(eq_device(device(globals$rng_state), dev0))
    expect_identical(as_array(out), expected[[1L]])

    # the next call runs on another device and continues the sequence there
    out <- f(nv_array(c(0, 0, 0), device = dev1))
    expect_true(eq_device(device(out), dev1))
    expect_true(eq_device(device(globals$rng_state), dev1))
    expect_identical(as_array(out), expected[[2L]])

    f(nv_array(c(0, 0, 0), device = dev0))
    expect_true(eq_device(device(globals$rng_state), dev0))
  })

  it("follows the device a jitted function is fixed to", {
    local_nv_seed(NULL)
    f <- jit(function() nv_runif(3L), device = dev1)
    globals$rng_state <- rng_state_from_seed(7L, device = dev0)
    out <- f()
    expect_true(eq_device(device(out), dev1))
    expect_true(eq_device(device(globals$rng_state), dev1))
    expect_identical(as_array(out), draws(7L, 1L, 3L)[[1L]])
  })

  it("follows the default device when a call has no array input", {
    local_nv_seed(NULL)
    globals$rng_state <- rng_state_from_seed(7L, device = dev0)
    out <- with_default_device(dev1, nv_runif(3L))
    expect_true(eq_device(device(out), dev1))
    expect_true(eq_device(device(globals$rng_state), dev1))
    expect_identical(as_array(out), draws(7L, 1L, 3L)[[1L]])
  })

  it("follows a call whose device comes from a closed-over array", {
    local_nv_seed(NULL)
    x1 <- nv_array(c(0, 0, 0), device = dev1)
    f <- jit(function() x1 + nv_runif(3L))
    globals$rng_state <- rng_state_from_seed(7L, device = dev0)
    out <- f()
    expect_true(eq_device(device(out), dev1))
    expect_true(eq_device(device(globals$rng_state), dev1))
    expect_identical(as_array(out), draws(7L, 1L, 3L)[[1L]])
  })

  it("follows a call whose device comes from a static argument", {
    local_nv_seed(NULL)
    f <- jit(function(device) nv_fill(0, 3L, device = device) + nv_runif(3L), static = "device")
    expected <- draws(7L, 2L, 3L)
    globals$rng_state <- rng_state_from_seed(7L, device = dev0)
    out <- f(dev1)
    expect_true(eq_device(device(globals$rng_state), dev1))
    expect_identical(as_array(out), expected[[1L]])
    out <- f(dev0)
    expect_true(eq_device(device(globals$rng_state), dev0))
    expect_identical(as_array(out), expected[[2L]])
  })

  it("moves to the device of the call that draws from it", {
    local_nv_seed(NULL)
    globals$rng_state <- rng_state_from_seed(7L, device = dev0)
    with_default_device(dev1, nv_runif(1L))
    expect_true(eq_device(device(globals$rng_state), dev1))
  })

  it("is restored by nv_set_rng_state() from a state on another device", {
    local_nv_seed(NULL)
    globals$rng_state <- rng_state_from_seed(7L, device = dev1)
    state <- nv_get_rng_state()
    nv_set_rng_state(NULL)
    nv_set_rng_state(state)
    out <- with_default_device(dev0, nv_runif(3L))
    expect_true(eq_device(device(globals$rng_state), dev0))
    expect_identical(as_array(out), draws(7L, 1L, 3L)[[1L]])
  })
})
