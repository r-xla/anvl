# The values of `n` consecutive eager draws of `shape` from the global random
# state seeded with `seed`.
draws <- function(seed, n, shape = 1L) {
  with_nv_seed(seed, lapply(seq_len(n), function(i) as_array(nv_runif(shape))))
}

# The counter of the global random state, as long as it is below 2^32.
state_counter <- function() state_words()[[3L]]

# The values of `n` consecutive eager draws of `shape` from the random state
# `random_state`, which is left the global random state.
draws_from <- function(random_state, n, shape = 1L) {
  nv_set_random_state(random_state)
  lapply(seq_len(n), function(i) as_array(nv_runif(shape)))
}

# The random state that prim_fork_random_state() returns from the global random
# state seeded with `seed`.
forked_state <- function(seed) {
  do.call(new_random_state, unname(with_nv_seed(seed, prim_fork_random_state())))
}

# Leaves the global random state unset until the calling scope exits, so that
# the next draw derives it from base R's RNG.
local_unset_random_state <- function(envir = parent.frame()) {
  old <- nv_get_random_state()
  nv_set_random_state(NULL)
  withr::defer(nv_set_random_state(old), envir = envir)
}

describe("nv_set_seed", {
  it("makes the draws from the global random state reproducible", {
    local_nv_seed(NULL)
    nv_set_seed(42L)
    a <- as_array(nv_runif(3L))
    b <- as_array(nv_runif(3L))
    expect_false(identical(a, b))
    nv_set_seed(42L)
    expect_identical(as_array(nv_runif(3L)), a)
    expect_identical(as_array(nv_runif(3L)), b)
  })

  it("starts from the key JAX's key(seed) gives and a zero counter", {
    local_nv_seed(-1L)
    expect_identical(state_words(), c(0, 2^32 - 1, 0, 0))
  })

  it("is not needed: an unset random state is derived from base R's RNG", {
    draw <- function() {
      local_unset_random_state()
      withr::local_seed(3L)
      list(as_array(nv_runif(3L)), as_array(nv_runif(3L)))
    }
    first <- draw()
    expect_identical(draw(), first)
    expect_false(identical(first[[1L]], first[[2L]]))
  })

  it("derives an unset random state from base R's seed", {
    draw <- function(seed) {
      local_unset_random_state()
      withr::local_seed(seed)
      as_array(nv_runif(3L))
    }
    expect_false(identical(draw(1L), draw(2L)))
  })

  it("derives an unset random state from base R's RNG only once", {
    local_unset_random_state()
    withr::local_seed(3L)
    a <- as_array(nv_runif(3L))
    # base R's seed no longer reseeds the random state
    withr::local_seed(3L)
    expect_false(identical(as_array(nv_runif(3L)), a))
    stats::runif(1L)
    nv_set_random_state(NULL)
    withr::local_seed(3L)
    expect_identical(as_array(nv_runif(3L)), a)
  })

  it("derives the random state from base R's RNG at once for a NULL seed, as nv_random_state() does", {
    local_nv_seed(1L)
    withr::local_seed(3L)
    nv_set_seed(NULL)
    from_seed <- state_words()
    withr::local_seed(3L)
    expect_identical(random_state_words(nv_random_state()), from_seed)
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
  it("seeds the global random state for `code` and restores it afterwards", {
    local_nv_seed(5L)
    nv_runif(1L)
    before <- state_words()
    a <- with_nv_seed(1L, as_array(nv_runif(2L)))
    expect_identical(state_words(), before)
    expect_identical(with_nv_seed(1L, as_array(nv_runif(2L))), a)
  })
})

describe("local_nv_seed", {
  it("restores the global random state when the calling scope exits", {
    local_nv_seed(5L)
    before <- state_words()
    draw <- function() {
      local_nv_seed(1L)
      as_array(nv_runif(2L))
    }
    a <- draw()
    expect_identical(state_words(), before)
    expect_identical(draw(), a)
  })
})

describe("nv_get_random_state", {
  it("is NULL while the global random state is not set", {
    local_unset_random_state()
    expect_null(nv_get_random_state())
  })

  it("returns the global random state, its key and counter as arrays", {
    local_nv_seed(7L)
    nv_runif(3L)
    random_state <- nv_get_random_state()
    expect_s3_class(random_state, "AnvlRandomState")
    expect_dtype(random_state$random_key, "ui32")
    expect_dtype(random_state$random_counter, "ui64")
    expect_identical(state_words(random_state), c(0, 7, 1, 0))
    nv_runif(3L)
    # later draws leave it intact
    expect_identical(state_words(random_state), c(0, 7, 1, 0))
  })

  it("lets nv_set_random_state() continue the sequence where it stopped", {
    local_nv_seed(7L)
    nv_runif(2L)
    random_state <- nv_get_random_state()
    a <- as_array(nv_runif(3L))
    nv_set_random_state(random_state)
    expect_identical(as_array(nv_runif(3L)), a)
    nv_set_random_state(random_state)
    expect_identical(as_array(nv_runif(3L)), a)
  })

  it("survives a round trip through a file", {
    local_nv_seed(7L)
    nv_runif(2L)
    path <- withr::local_tempfile(fileext = ".rds")
    saveRDS(nv_get_random_state(), path)
    a <- as_array(nv_runif(3L))
    nv_set_random_state(readRDS(path))
    expect_identical(as_array(nv_runif(3L)), a)
  })

  it("prints its key and counter", {
    local_nv_seed(-1L)
    nv_runif(1L)
    expect_identical(
      format(nv_get_random_state()),
      "<AnvlRandomState> key 0x00000000ffffffff, counter 0x0000000000000001"
    )
  })

  it("cannot be called in a traced function", {
    f <- jit(function() {
      nv_get_random_state()
      nv_runif(1L)
    })
    expect_error(f(), "cannot be read in a function traced")
  })
})

describe("nv_set_random_state", {
  it("unsets the global random state with NULL", {
    local_nv_seed(1L)
    nv_set_random_state(NULL)
    expect_null(nv_get_random_state())
  })

  it("rejects what is not a random state", {
    expect_error(nv_set_random_state(c(1, 2, 3, 4)), "must be a random state")
    key <- nv_array(c(1, 2), dtype = "ui32")
    counter <- nv_scalar(0L, dtype = "ui64")
    # a list of a key and a counter is not one
    expect_error(nv_set_random_state(list(random_key = key, random_counter = counter)), "must be a random state")
    expect_error(nv_set_random_state(new_random_state(c(1, 2), counter)), "must be a random state")
    expect_error(nv_set_random_state(new_random_state(nv_array(1:2, dtype = "i32"), counter)), "must be a random state")
    expect_error(nv_set_random_state(new_random_state(key, nv_array(0L, dtype = "ui64"))), "must be a random state")
  })

  it("carries the counter into its high word", {
    local_nv_seed(1L)
    nv_set_random_state(random_state_of_words(c(1, 0, 2^32 - 1, 0)))
    a <- as_array(nv_runif(1L))
    expect_identical(state_words(), c(1, 0, 0, 1))
    b <- as_array(nv_runif(1L))
    expect_identical(state_words(), c(1, 0, 1, 1))
    # the program sees the carried counter too
    nv_set_random_state(random_state_of_words(c(1, 0, 2^32 - 1, 0)))
    expect_identical(as_array(jit(function() list(nv_runif(1L), nv_runif(1L)))()[[2L]]), b)
    expect_false(identical(a, b))
    # the high word of the counter reaches the hash
    expect_false(identical(b, draws(1L, 1L)[[1L]]))
  })
})

describe("a sampling call", {
  it("advances the counter by one, however many values it draws", {
    local_nv_seed(1L)
    for (shape in list(integer(), 1L, 1000L, c(3L, 5L))) {
      for (dtype in c("ui8", "ui16", "ui32", "ui64")) {
        before <- state_counter()
        prim_sample_bits(shape, dtype)
        expect_identical(state_counter(), before + 1)
      }
    }
  })

  it("advances the counter by the sampler's number of sampling calls", {
    local_nv_seed(1L)
    nv_rnorm(5L)
    expect_identical(state_counter(), 2)
    nv_runif(5L)
    nv_rbinom(5L, size = 3L)
    nv_sample_int(5L, n = 10L)
    nv_sample(5L, nv_array(1:3))
    expect_identical(state_counter(), 6)
  })

  it("does not move later draws by the size of its own", {
    a <- with_nv_seed(1L, {
      nv_runif(1L)
      as_array(nv_runif(5L))
    })
    b <- with_nv_seed(1L, {
      nv_runif(1000L)
      as_array(nv_runif(5L))
    })
    expect_identical(a, b)
  })
})

describe("global_random_state", {
  it("can be listed outside of a trace, as installing the package does", {
    expect_no_error(mget(ls(global_random_state, all.names = TRUE), envir = global_random_state))
  })
})

describe("the global random state in jit", {
  it("gives the draws of the same calls made eagerly", {
    f <- function() {
      list(nv_runif(1L), nv_runif(1L), nv_runif(1L))
    }
    eager <- with_nv_seed(1L, list(lapply(f(), as_array), state_words()))
    jitted <- with_nv_seed(1L, list(lapply(jit(f)(), as_array), state_words()))
    expect_identical(jitted, eager)
    expect_identical(eager[[2L]], c(0, 1, 3, 0))
  })

  it("gives the draws of the same calls made eagerly through nested jit()", {
    g <- function(x) list(a = x + nv_runif(2L), b = nv_rnorm(3L))
    f <- function(x) c(g(x), jit(g)(x))
    x <- nv_array(c(0, 0))
    eager <- with_nv_seed(4L, list(lapply(f(x), as_array), state_words()))
    jitted <- with_nv_seed(4L, list(lapply(jit(f)(x), as_array), state_words()))
    expect_identical(jitted, eager)
  })

  it("is continued by the next call", {
    f <- jit(function(x) list(a = x + nv_runif(2L), b = nv_rnorm(2L)))
    outs <- with_nv_seed(42L, list(f(nv_array(c(0, 0))), f(nv_array(c(0, 0)))))
    with_nv_seed(42L, {
      expect_identical(as_array(outs[[1L]]$a), as_array(nv_runif(2L)))
      expect_identical(as_array(outs[[1L]]$b), as_array(nv_rnorm(2L)))
      expect_identical(as_array(outs[[2L]]$a), as_array(nv_runif(2L)))
      expect_identical(as_array(outs[[2L]]$b), as_array(nv_rnorm(2L)))
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
    expect_identical(state_counter(), 2)
  })

  it("is shared between jitted functions and eager draws", {
    f <- jit(function() nv_runif(2L))
    expected <- draws(5L, 2L, 2L)
    expect_identical(with_nv_seed(5L, list(as_array(f()), as_array(nv_runif(2L)))), expected)
    expect_identical(with_nv_seed(5L, list(as_array(nv_runif(2L)), as_array(f()))), expected)
  })

  it("leaves the global random state alone in functions that do not draw from it", {
    local_nv_seed(5L)
    nv_runif(1L)
    before <- state_words()
    f <- jit(function(x) x + 1L)
    f(nv_array(1L))
    expect_identical(state_words(), before)
  })

  it("is a side effect of a graph traced by trace_fn(), with the draws the trace counted", {
    graph <- trace_fn(function() list(nv_runif(1L), nv_rnorm(1L)), list())
    expect_length(graph$inputs, 0L)
    expect_length(graph$outputs, 2L)
    expect_length(graph$side_effects, 1L)
    expect_true(identical(graph$side_effects[[1L]]$effect, global_random_state))
    input <- lapply(graph$side_effects[[1L]]$inputs, `[[`, "aval")
    expect_identical(
      lapply(input, function(aval) as.character(dtype(aval))),
      list(random_key = "ui32", random_counter = "ui64")
    )
    expect_identical(lapply(input, shape), list(random_key = 2L, random_counter = integer()))
    expect_identical(graph$side_effects[[1L]]$n, 3)
  })

  it("is an input of the program, its key and counter, which returns no random state", {
    graph <- trace_fn(nv_runif, list(2L))
    func <- stablehlo(graph)[[1L]]
    expect_length(func$inputs, 2L)
    expect_length(func$outputs, 1L)
    expect_null(func$inputs[[1L]]$alias)
  })

  it("is left out of a graph traced by trace_fn() that does not draw", {
    graph <- trace_fn(function(x) x + 1, list(nv_array(1)))
    expect_length(graph$inputs, 1L)
    expect_length(graph$outputs, 1L)
    expect_length(graph$side_effects, 0L)
  })
})

describe("the global random state in sub-graphs", {
  it("is forked by nv_while, which advances it by one", {
    f <- jit(function(n) {
      nv_while(
        list(i = 0L, total = nv_scalar(0)),
        function(i, total) i < n,
        function(i, total) list(i = i + 1L, total = total * 10 + nv_runif(integer()))
      )$total
    })
    for (n in c(0L, 1L, 3L)) {
      local_nv_seed(1L)
      f(n)
      expect_identical(state_counter(), 1)
    }
    # the body draws from the forked random state, one sampling call per
    # iteration
    expected <- local({
      local_nv_seed(NULL)
      draws_from(forked_state(1L), 3L, integer())
    })
    out <- with_nv_seed(1L, f(3L))
    expect_equal(as_array(out), (expected[[1L]] * 10 + expected[[2L]]) * 10 + expected[[3L]], tolerance = 1e-6)
  })

  it("is not forked by an nv_while whose body does not draw", {
    f <- jit(function() {
      nv_while(list(i = 0L), function(i) i < 3L, function(i) list(i = i + 1L))
    })
    local_nv_seed(1L)
    f()
    expect_identical(state_counter(), 0)
  })

  it("is carried through the body of nv_scan like an R loop", {
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

  it("is advanced by nv_if by the larger branch, whichever branch runs", {
    f <- jit(function(pred) {
      list(
        branch = nv_if(pred, function() nv_runif(1L), function() nv_rnorm(1L)),
        after = nv_runif(1L)
      )
    })
    for (pred in c(TRUE, FALSE)) {
      local_nv_seed(1L)
      f(pred)
      expect_identical(state_counter(), 3)
    }
    with_nv_seed(1L, {
      expected <- as_array(nv_runif(1L))
      nv_runif(1L)
      after <- as_array(nv_runif(1L))
    })
    out <- with_nv_seed(1L, f(TRUE))
    expect_identical(as_array(out$branch), expected)
    expect_identical(as_array(out$after), after)
    out <- with_nv_seed(1L, f(FALSE))
    expect_identical(as_array(out$branch), with_nv_seed(1L, as_array(nv_rnorm(1L))))
    expect_identical(as_array(out$after), after)
  })

  it("draws in nv_if what the branch draws eagerly", {
    f <- function() nv_runif(2L) + nv_rnorm(2L)
    g <- jit(function(pred) nv_if(pred, f, f))
    expect_identical(with_nv_seed(1L, as_array(g(TRUE))), with_nv_seed(1L, as_array(f())))
  })

  it("is advanced by an nv_if nested in an nv_if", {
    f <- jit(function(p, q) {
      branch <- nv_if(
        p,
        function() nv_if(q, function() nv_runif(1L), function() nv_fill(-1, 1L)),
        function() nv_fill(-2, 1L)
      )
      list(branch = branch, after = nv_runif(1L))
    })
    expected <- draws(1L, 2L)
    for (pq in list(c(TRUE, TRUE), c(TRUE, FALSE), c(FALSE, TRUE))) {
      out <- with_nv_seed(1L, f(pq[[1L]], pq[[2L]]))
      expect_identical(as_array(out$after), expected[[2L]])
    }
    out <- with_nv_seed(1L, f(TRUE, TRUE))
    expect_identical(as_array(out$branch), expected[[1L]])
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

  it("is used by prim_sample_bits() called in a function differentiated by gradient()", {
    f <- jit(function(x) {
      grad <- gradient(function(x) nv_sum(x * nv_convert(prim_sample_bits(3L, "ui8"), dtype = "f32")))(x)
      list(grad = grad$x, after = prim_sample_bits(1L, "ui8"))
    })
    out <- with_nv_seed(1L, f(nv_array(c(1, 1, 1))))
    with_nv_seed(1L, {
      expect_identical(as.numeric(as_array(out$grad)), as.numeric(as_array(prim_sample_bits(3L, "ui8"))))
      expect_identical(as_array(out$after), as_array(prim_sample_bits(1L, "ui8")))
    })
  })

  it("is drawn from once by value_and_gradient(), for the value and the gradient", {
    f <- jit(function(x) value_and_gradient(function(x) nv_sum(x * nv_runif(2L, dtype = "f64")))(x))
    x <- nv_array(c(2, 3), dtype = "f64")
    out <- with_nv_seed(1L, list(f(x), state_words()))
    u <- with_nv_seed(1L, as.vector(nv_runif(2L, dtype = "f64")))
    expect_equal(as.vector(out[[1L]]$grad$x), u)
    expect_equal(as.vector(out[[1L]]$value), sum(c(2, 3) * u))
    # one sampling call
    expect_identical(out[[2L]], c(0, 1, 1, 0))
  })

  it("is taken by the graph graph_gradient() builds from a graph that samples", {
    graph <- trace_fn(function(x) nv_sum(x * nv_runif(2L)), list(x = nv_aval("f32", 2L)))
    grad <- graph_gradient(graph, "x")
    state <- grad$side_effects[[1L]]
    expect_named(state$inputs, c("random_key", "random_counter"))
    expect_identical(state$n, 1)
    names <- vapply(grad$statements, function(call) call$primitive$name, character(1L))
    expect_identical(sum(names == "sample_bits"), 1L)
  })

  it("is forked by an nv_while nested in the body of nv_scan, once per step", {
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
    local_nv_seed(1L)
    f()
    expect_identical(state_counter(), 3)
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

describe("the global random state across devices", {
  dev0 <- nv_device("cpu:0")
  dev1 <- nv_device("cpu:1")

  it("gives the same draws on every device", {
    f <- jit(function(x) x + nv_runif(3L))
    expected <- draws(7L, 2L, 3L)
    local_nv_seed(7L)
    out <- f(nv_array(c(0, 0, 0), device = dev0))
    expect_identical(as_array(out), expected[[1L]])
    out <- f(nv_array(c(0, 0, 0), device = dev1))
    expect_true(eq_device(device(out), dev1))
    expect_identical(as_array(out), expected[[2L]])
  })

  it("follows the device a jitted function is fixed to", {
    f <- jit(function() nv_runif(3L), device = dev1)
    out <- with_nv_seed(7L, f())
    expect_true(eq_device(device(out), dev1))
    expect_identical(as_array(out), draws(7L, 1L, 3L)[[1L]])
  })

  it("follows the default device when a call has no array input", {
    out <- with_nv_seed(7L, with_default_device(dev1, nv_runif(3L)))
    expect_true(eq_device(device(out), dev1))
    expect_identical(as_array(out), draws(7L, 1L, 3L)[[1L]])
  })
})

describe("counter_plus", {
  it("advances the counter by 2^31 and more, with the carry into the high word", {
    words <- c(1, 2, 2^32 - 1, 7)
    # the advanced counters, low word first
    expected <- list(c(4, 8), c(2^31 + 4, 9), c(2^32 - 2, 2^21 + 7))
    ns <- c(5, 2^32 + 2^31 + 5, 2^53 - 1)
    state <- random_state_of_words(words)
    plus <- jit(counter_plus, static = "n")
    for (i in seq_along(ns)) {
      counter <- plus(state$random_counter, ns[[i]])
      expect_identical(state_words(new_random_state(state$random_key, counter)), c(1, 2, expected[[i]]))
    }
  })
})

describe("side_effect_advance", {
  it("advances a random state's counter by 2^31 and more, with the carry into the high word", {
    # the advanced counters, low word first
    expected <- list(c(4, 8), c(2^31 + 4, 9), c(2^32 - 2, 2^21 + 7))
    ns <- c(5, 2^32 + 2^31 + 5, 2^53 - 1)
    for (i in seq_along(ns)) {
      state <- random_state_of_words(c(1, 2, 2^32 - 1, 7))
      side_effect_advance(global_random_state, state, ns[[i]])
      expect_identical(state_words(state), c(1, 2, expected[[i]]))
    }
  })
})

describe("the global random state in a traced graph", {
  it("gives every sampling call the key and its own counter, computed from the one the graph takes", {
    graph <- trace_fn(function() list(prim_sample_bits(2L, "ui32"), prim_sample_bits(3L, "ui32")), list())
    expect_snapshot(graph)
  })

  it("forks the random state for nv_while and carries the counter through its body", {
    graph <- trace_fn(
      function(x) {
        nv_while(
          list(i = x),
          \(i) i < 3L,
          \(i) list(i = i + nv_convert(prim_sample_bits(integer(), "ui8"), "i32"))
        )
      },
      list(x = nv_scalar(0L, dtype = "i32"))
    )
    expect_snapshot(graph)
  })

  it("draws after an nv_while loop at the counter after its fork", {
    graph <- trace_fn(
      function(n) {
        out <- nv_while(list(i = nv_scalar(0L, dtype = "i32")), function(i) i < n, function(i) {
          prim_sample_bits(2L, "ui32")
          list(i = i + 1L)
        })
        list(out$i, prim_sample_bits(3L, "ui32"))
      },
      list(n = nv_aval("i32", integer()))
    )
    expect_snapshot(graph)
  })

  it("carries the random state through the steps of nv_scan by its counter", {
    graph <- trace_fn(
      function() {
        prim_sample_bits(2L, "ui32")
        nv_scan(
          init = nv_scalar(0L, dtype = "i32"),
          body = function(carry, x) {
            list(carry = carry, out = list(prim_sample_bits(1L, "ui32"), prim_sample_bits(1L, "ui32")))
          },
          steps = 3L
        )$out
      },
      list()
    )
    expect_snapshot(graph)
  })

  it("lets the branches of nv_if continue the counter of the random state", {
    graph <- trace_fn(
      function(p) {
        prim_sample_bits(2L, "ui32")
        nv_if(p, \() prim_sample_bits(1L, "ui32"), \() {
          prim_sample_bits(1L, "ui32")
          prim_sample_bits(1L, "ui32")
        })
      },
      list(p = nv_scalar(TRUE))
    )
    expect_snapshot(graph)
  })
})

describe("nv_random_state", {
  it("is a node of the argument tree, which prints it as a random state", {
    expect_match(format(pjrt::build_tree(list(rng = nv_random_state(1L)))), "AnvlRandomState")
  })

  it("gives the random state nv_set_seed() gives the global one", {
    local_nv_seed(42L)
    expect_identical(state_words(nv_random_state(42L)), state_words())
  })

  it("is advanced in place by the draws from it, which leave the global random state alone", {
    local_nv_seed(1L)
    rng <- nv_random_state(42L)
    a <- with_nv_rng_state(rng, as_array(nv_runif(3L)))
    b <- with_nv_rng_state(rng, as_array(nv_runif(3L)))
    expect_false(identical(a, b))
    expect_identical(state_words(rng), c(0, 42, 2, 0))
    expect_identical(state_words(), c(0, 1, 0, 0))
    # it draws what the global random state seeded alike draws
    expect_identical(with_nv_seed(42L, as_array(nv_runif(3L))), a)
  })

  it("is taken by a jitted function as an argument, which draws what the same code draws eagerly", {
    local_nv_seed(1L)
    draw <- function(rng) {
      list(with_nv_rng_state(rng, nv_rnorm(2L)), nv_runif(2L), with_nv_rng_state(rng, nv_sample_int(3L, 6L)))
    }
    eager_rng <- nv_random_state(5L)
    eager <- lapply(draw(eager_rng), as_array)
    jit_rng <- nv_random_state(5L)
    nv_set_seed(1L)
    compiled <- lapply(jit(draw)(jit_rng), as_array)
    expect_identical(compiled, eager)
    expect_identical(state_words(jit_rng), state_words(eager_rng))
    expect_identical(state_words(jit_rng)[[3L]], 3)
    # the global random state is advanced by the global draw only
    expect_identical(state_words()[[3L]], 1)
  })

  it("is drawn from in a loop body like the global random state", {
    rng <- nv_random_state(3L)
    f <- jit(
      function(rng) {
        local_nv_rng_state(rng)
        w <- nv_while(list(i = 0L, s = nv_scalar(0)), \(i, s) i < 3L, \(i, s) {
          list(i = i + 1L, s = s + nv_runif(integer()))
        })
        sc <- nv_scan(
          init = nv_scalar(0),
          body = \(carry, x) list(carry = carry + nv_runif(integer()), out = carry),
          steps = 2L
        )
        list(w = w$s, sc = sc$carry)
      }
    )
    f(rng)
    # a fork for the while loop, and a sampling call per step of the scan
    expect_identical(state_words(rng)[[3L]], 3)
  })

  it("compiles one program for every random state passed", {
    f <- jit(function(x, rng) with_nv_rng_state(rng, x + nv_runif(2L)))
    a <- as_array(f(nv_array(c(0, 0)), nv_random_state(1L)))
    b <- as_array(f(nv_array(c(0, 0)), nv_random_state(2L)))
    expect_false(identical(a, b))
    expect_identical(as_array(f(nv_array(c(0, 0)), nv_random_state(1L))), a)
    expect_identical(jit_cache_size(f), 1L)
  })

  it("compiles a program of its own apart from a list of the same arrays", {
    f <- jit(function(s) if (is_random_state(s)) with_nv_rng_state(s, nv_runif(1L)) else s$random_key)
    rng <- nv_random_state(1L)
    f(rng)
    expect_identical(
      as.numeric(as_array(f(list(random_key = rng$random_key, random_counter = rng$random_counter)))),
      c(0, 1)
    )
    expect_identical(jit_cache_size(f), 2L)
    f(nv_random_state(2L))
    expect_identical(jit_cache_size(f), 2L)
  })

  it("is traced as a side effect of its own per random state argument", {
    graph <- trace_fn(
      \(rng1, rng2) list(with_nv_rng_state(rng1, nv_runif(1L)), with_nv_rng_state(rng2, nv_runif(1L))),
      list(nv_random_state(1L), nv_random_state(2L))
    )
    # each random state is two inputs of the graph, its key and its counter,
    # and has a view of its own, which knows the leaf of its key
    expect_length(graph$inputs, 4L)
    expect_length(graph$side_effects, 2L)
    for (i in 1:2) {
      view <- graph$side_effects[[i]]
      expect_identical(view$effect$leaf, 2L * i - 1L)
      expect_true(is_traced_random_state(view$effect))
      expect_null(view$inputs)
      expect_identical(view$n, 1)
    }
    expect_false(identical(graph$side_effects[[1L]]$effect, graph$side_effects[[2L]]$effect))
    # each draw takes the key and the counter of its own random state
    draws <- Filter(function(call) call$primitive$name == "sample_bits", graph$statements)
    expect_length(draws, 2L)
    expect_identical(draws[[1L]]$inputs, graph$inputs[1:2])
    expect_identical(draws[[2L]]$inputs, graph$inputs[3:4])
    # the global random state is not used
    expect_false(any(vapply(
      graph$side_effects,
      function(view) identical(view$effect, global_random_state),
      logical(1L)
    )))
  })

  it("is taken by a jitted function also inside a list", {
    f <- jit(function(args) with_nv_rng_state(args$rng, args$x + nv_runif(2L)))
    rng <- nv_random_state(1L)
    out <- as_array(f(list(x = nv_array(c(0, 0)), rng = rng)))
    expect_identical(out, with_nv_rng_state(nv_random_state(1L), as_array(nv_runif(2L))))
    expect_identical(state_words(rng)[[3L]], 1)
  })

  it("is taken by a function gradient() differentiates", {
    f <- function(x, rng) with_nv_rng_state(rng, sum(x * nv_runif(2L)))
    u <- with_nv_rng_state(nv_random_state(1L), as_array(nv_runif(2L)))
    rng <- nv_random_state(1L)
    expect_equal(as_array(jit(gradient(f, wrt = "x"))(nv_array(c(1, 2)), rng)$x), u)
    expect_identical(state_words(rng)[[3L]], 1)
    rng <- nv_random_state(1L)
    g <- jit(function(x, rng) gradient(f, wrt = "x")(x, rng)$x)
    expect_equal(as_array(g(nv_array(c(1, 2)), rng)), u)
    expect_identical(state_words(rng)[[3L]], 1)
  })

  it("reads in a jitted function the key and counter it would read eagerly", {
    f <- function(s) {
      local_nv_rng_state(s)
      nv_runif(2L)
      before <- s$random_counter
      nv_runif(2L)
      list(before, s$random_counter, s$random_key)
    }
    eager <- lapply(f(nv_random_state(1L)), as_array)
    rng <- nv_random_state(1L)
    expect_identical(lapply(jit(f)(rng), as_array), eager)
    expect_identical(as.numeric(eager[[2L]]), 2)
    expect_identical(state_words(rng)[[3L]], 2)
  })

  it("reads in each step of nv_scan the counter after the draws so far", {
    f <- jit(function(s) {
      local_nv_rng_state(s)
      nv_scan(
        nv_scalar(0L, dtype = "ui64"),
        body = function(carry, x) {
          nv_runif(1L)
          list(carry = carry, out = s$random_counter)
        },
        steps = 3L
      )$out
    })
    expect_identical(as.numeric(as_array(f(nv_random_state(1L)))), c(1, 2, 3))
  })

  it("is not advanced by a loop body that only reads it", {
    f <- jit(function(s) {
      nv_while(list(i = 0L), \(i) i < 2L, \(i) list(i = i + 1L + 0L * nv_convert(s$random_counter, "i32")))$i
    })
    rng <- nv_random_state(1L)
    f(rng)
    expect_identical(state_words(rng)[[3L]], 0)
  })

  it("can be passed to several arguments only as different random states", {
    f <- jit(function(a, b) list(with_nv_rng_state(a, nv_runif(2L)), with_nv_rng_state(b, nv_runif(2L))))
    rng <- nv_random_state(1L)
    expect_error(f(rng, rng), "passed to the jitted function twice")
    # neither was advanced
    expect_identical(state_words(rng)[[3L]], 0)
    # two random states are drawn from and advanced separately
    a <- nv_random_state(1L)
    b <- nv_random_state(1L)
    out <- lapply(f(a, b), as_array)
    expect_identical(out[[1L]], out[[2L]])
    expect_identical(state_words(a), state_words(b))
    expect_identical(state_words(a)[[3L]], 1)
  })

  it("is forked once by an nv_while in a branch, when passed inside a list", {
    f <- jit(function(p, args) {
      nv_if(
        p,
        function() {
          local_nv_rng_state(args$rng)
          nv_while(list(i = 0L), \(i) i < 3L, \(i) {
            nv_runif(1L)
            list(i = i + 1L)
          })$i
        },
        function() nv_scalar(0L)
      )
    })
    rng <- nv_random_state(1L)
    local_nv_seed(1L)
    before <- state_words()
    invisible(f(TRUE, list(rng = rng)))
    # the fork of the loop, however often it ran, and nothing from the global
    # random state
    expect_identical(state_words(rng)[[3L]], 1)
    expect_identical(state_words(), before)
  })

  it("holds GraphValues and no descriptor once its trace is done", {
    graph <- trace_fn(\(rng) with_nv_rng_state(rng, nv_runif(1L)), list(nv_random_state(1L)))
    handle <- graph$side_effects[[1L]]$effect
    expect_true(is_graph_value(handle$key))
    expect_true(is_graph_value(handle$counter))
    expect_false(exists("desc", envir = handle, inherits = FALSE))
  })

  it("cannot be part of the state of nv_while() or nv_scan()", {
    expect_error(
      nv_while(list(i = 1L, rng = nv_random_state(1L)), \(i, rng) i <= 2L, \(i, rng) list(i = i + 1L, rng = rng)),
      "must not contain a side effect"
    )
    f <- jit(function(rng) {
      nv_scan(list(rng = rng), body = function(carry, x) list(carry = carry, out = NULL), steps = 2L)
    })
    expect_error(f(nv_random_state(1L)), "must not contain a side effect")
  })
})

describe("local_nv_rng_state", {
  it("makes a random state the default for the scope, advancing it in place", {
    local_nv_seed(5L)
    before <- state_words()
    rng <- nv_random_state(1L)
    out <- with_nv_rng_state(rng, as_array(nv_runif(2L)))
    expect_identical(out, with_nv_rng_state(nv_random_state(1L), as_array(nv_runif(2L))))
    expect_identical(state_words(rng)[[3L]], 1)
    # the global random state is restored and was not drawn from
    expect_identical(state_words(), before)
  })

  it("refuses NULL", {
    expect_error(local_nv_rng_state(NULL), "must be a random state")
  })

  it("lasts until the calling function returns", {
    rng <- nv_random_state(1L)
    f <- function() {
      local_nv_rng_state(rng)
      nv_runif(1L)
      nv_rnorm(1L)
    }
    local_nv_seed(5L)
    f()
    expect_identical(state_words(rng)[[3L]], 3)
    expect_identical(state_words()[[3L]], 0)
  })

  it("makes a random state argument the default in a jitted function, also in a loop body", {
    f <- jit(function(x, rng) {
      local_nv_rng_state(rng)
      w <- nv_while(list(i = 0L, s = x), \(i, s) i < 2L, \(i, s) list(i = i + 1L, s = s + nv_runif(integer())))
      w$s + nv_runif(1L)
    })
    rng <- nv_random_state(1L)
    local_nv_seed(5L)
    f(nv_scalar(0), rng)
    # the fork of the loop and the draw after it
    expect_identical(state_words(rng)[[3L]], 2)
    expect_identical(state_words()[[3L]], 0)
  })

  it("refuses a random state a traced function closes over", {
    rng <- nv_random_state(1L)
    f <- jit(function(x) {
      local_nv_rng_state(rng)
      x + nv_runif(1L)
    })
    expect_error(f(nv_scalar(0)), "must be a random state the jitted function takes as an argument")
  })

  it("makes a random state argument the default in a branch of nv_if() only", {
    f <- jit(function(p, rng) {
      nv_if(
        p,
        function() {
          local_nv_rng_state(rng)
          nv_runif(1L)
        },
        function() nv_runif(1L)
      )
    })
    rng <- nv_random_state(1L)
    local_nv_seed(5L)
    out <- f(nv_scalar(TRUE), rng)
    expect_identical(as_array(out), with_nv_rng_state(nv_random_state(1L), as_array(nv_runif(1L))))
    # each random state advances by the larger of the branches' draws from it
    expect_identical(state_words(rng)[[3L]], 1)
    expect_identical(state_counter(), 1)
  })

  it("makes a random state argument the default in a jitted function called in a branch", {
    step <- jit(function(x, rng) {
      local_nv_rng_state(rng)
      x + nv_runif(1L)
    })
    f <- jit(function(p, x, rng) nv_if(p, function() step(x, rng), function() x))
    rng <- nv_random_state(1L)
    local_nv_seed(5L)
    f(nv_scalar(TRUE), nv_array(0), rng)
    expect_identical(state_words(rng)[[3L]], 1)
    expect_identical(state_counter(), 0)
  })

  it("makes a random state argument the default in a function gradient() differentiates", {
    f <- jit(function(x, rng) {
      gradient(function(x) {
        local_nv_rng_state(rng)
        nv_sum(x * nv_runif(2L))
      })(x)$x
    })
    rng <- nv_random_state(1L)
    local_nv_seed(5L)
    out <- f(nv_array(c(1, 1)), rng)
    expect_identical(as_array(out), with_nv_rng_state(nv_random_state(1L), as_array(nv_runif(2L))))
    expect_identical(state_words(rng)[[3L]], 1)
    expect_identical(state_counter(), 0)
  })

  it("cannot be used in a loop body, nor in a function traced inside one", {
    in_body <- jit(function(rng) {
      nv_scan(
        nv_scalar(0),
        body = function(carry, x) {
          local_nv_rng_state(rng)
          list(carry = carry + nv_runif(integer()), out = carry)
        },
        steps = 2L
      )$carry
    })
    expect_error(in_body(nv_random_state(1L)), "cannot be used in the body of")
    in_branch_in_body <- jit(function(p, rng) {
      nv_while(list(i = 0L), \(i) i < 2L, \(i) {
        nv_if(
          p,
          function() {
            local_nv_rng_state(rng)
            nv_runif(integer())
          },
          function() nv_runif(integer())
        )
        list(i = i + 1L)
      })$i
    })
    expect_error(in_branch_in_body(nv_scalar(TRUE), nv_random_state(1L)), "cannot be used in the body of")
  })

  it("does not outlive a trace that fails", {
    f <- jit(function(rng) {
      local_nv_rng_state(rng)
      stop("failed")
    })
    expect_error(f(nv_random_state(1L)), "failed")
    local_nv_seed(1L)
    invisible(jit(function() nv_runif(1L))())
    expect_identical(state_counter(), 1)
  })
})
