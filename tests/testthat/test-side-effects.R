# A carried side effect counting the calls of the primitive `tick` it returns.
# Its value starts at `start` in the program (`from = "program"`) or comes from
# `store$value` on the host (`from = "host"`); the host receives the value a
# program leaves behind in `store$received`. Both are removed when `envir`
# exits.
local_counter <- function(from = "program", start = 0L, envir = parent.frame()) {
  store <- new.env(parent = emptyenv())
  store$value <- start
  register_side_effect(
    "counter",
    aval = AbstractArray(dtype = "i32", shape = integer()),
    get = if (from == "host") function() store$value,
    build = if (from == "program") function(func) hlo_scalar(start, dtype = "i32", func = func),
    receive = function(value) {
      store$received <- as_array(value)
    },
    backends = "pjrt"
  )
  withr::defer(
    {
      globals$side_effects$counter <- NULL
    },
    envir = envir
  )

  tick <- new_primitive(
    "tick",
    function() {
      graph_desc_add(self, list(), infer_fn = function(counter) list(value = counter, counter = counter))[[1L]]
    },
    side_effects = "counter",
    register = FALSE
  )
  tick[["stablehlo"]] <- function(counter) {
    list(counter, hlo_add(counter, hlo_scalar(1L, dtype = "i32", func = counter$func)))
  }
  list(store = store, tick = tick)
}

describe("register_side_effect", {
  it("registers the side effect under its name", {
    local_counter()
    expect_identical(side_effect("counter")$name, "counter")
  })

  it("takes the value either from the host or from the program", {
    expect_error(
      register_side_effect("x", AbstractArray(dtype = "i32", shape = integer()), backends = "pjrt"),
      "either from the host"
    )
  })

  it("advances only a counted side effect on the host", {
    expect_error(
      register_side_effect(
        "x",
        AbstractArray(dtype = "i32", shape = integer()),
        get = function() 0L,
        advance = function(n) NULL,
        backends = "pjrt"
      ),
      "only a counted side effect"
    )
  })
})

describe("new_primitive(side_effects = )", {
  it("refuses a side effect named like an argument of the primitive", {
    expect_error(
      new_primitive("clash", function(random_state) random_state, side_effects = "random_state", register = FALSE),
      "must not have an argument named like one of its side effects"
    )
  })
})

describe("a carried side effect in jit", {
  it("starts from the value the program builds and is received by the host", {
    x <- local_counter(start = 10L)
    f <- jit(function() list(x$tick(), x$tick()))
    expect_identical(lapply(f(), as_array), list(10L, 11L))
    expect_identical(x$store$received, 12L)
    # every call starts from the value the program builds
    expect_identical(lapply(f(), as_array), list(10L, 11L))
  })

  it("starts from the value the host supplies", {
    x <- local_counter(from = "host", start = 5L)
    f <- jit(function() x$tick())
    expect_identical(as_array(f()), 5L)
    expect_identical(x$store$received, 6L)
    x$store$value <- 20L
    expect_identical(as_array(f()), 20L)
  })

  it("is returned after the program's outputs and taken by no input", {
    x <- local_counter()
    graph <- trace_fn(function() x$tick(), list())
    func <- stablehlo(graph)[[1L]]
    expect_length(func$inputs, 0L)
    expect_length(func$outputs, 2L)
  })

  it("is handed back by the branch nv_if takes", {
    x <- local_counter()
    f <- jit(function(p) {
      nv_if(p, \() x$tick() + x$tick(), \() nv_scalar(-1L, dtype = "i32"))
      x$tick()
    })
    expect_identical(as_array(f(TRUE)), 2L)
    expect_identical(x$store$received, 3L)
    expect_identical(as_array(f(FALSE)), 0L)
    expect_identical(x$store$received, 1L)
  })

  it("is carried from one iteration of nv_while to the next", {
    x <- local_counter()
    f <- jit(function(n) {
      nv_while(list(i = 0L), function(i) i < n, function(i) {
        x$tick()
        list(i = i + 1L)
      })
      x$tick()
    })
    expect_identical(as_array(f(4L)), 4L)
    expect_identical(x$store$received, 5L)
  })

  it("is carried through the body of nv_scan together with the global random state", {
    x <- local_counter()
    local_nv_seed(1L)
    f <- jit(function() {
      nv_scan(
        init = nv_scalar(0),
        body = function(carry, x_) list(carry = carry + nv_runif(integer()), out = x$tick()),
        steps = 3L
      )$out
    })
    expect_identical(as.vector(as_array(f())), 0:2)
    expect_identical(x$store$received, 3L)
    expect_identical(unclass(nv_get_random_state())[[3L]], 3)
  })

  it("is threaded through a function differentiated by gradient()", {
    x <- local_counter()
    f <- jit(function(y) {
      grad <- gradient(function(y) nv_sum(y * nv_convert(x$tick(), dtype = "f32")))(y)
      list(grad = grad$y, after = x$tick())
    })
    out <- f(nv_array(c(1, 1)))
    expect_identical(as_array(out$after), 1L)
    expect_identical(x$store$received, 2L)
  })

  it("comes in registration order, whatever is used first", {
    x <- local_counter()
    graph <- trace_fn(function() list(x$tick(), nv_runif(1L)), list())
    expect_named(graph$side_effects, c("random_state", "counter"))
  })

  it("cannot be used in the function of a reduction", {
    x <- local_counter()
    expect_error(
      trace_fn(function() prim_reduce(nv_array(1:2), 0L, function(a, b) a + b + x$tick(), axes = 1L), list()),
      'The side effect "counter" cannot be used here'
    )
  })

  it("is refused by a backend that cannot run it", {
    skip_if_no_quickr()
    x <- local_counter()
    local_backend("quickr")
    f <- jit(function() x$tick())
    expect_error(f(), 'The quickr backend does not support the side effect "counter"')
  })
})
