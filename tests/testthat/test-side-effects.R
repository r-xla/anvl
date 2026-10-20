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
    avals = list(counter = AbstractArray(dtype = "i32", shape = integer())),
    get = if (from == "host") function() list(store$value),
    build = if (from == "program") {
      function(desc) {
        list(
          counter = graph_desc_add(
            prim_fill,
            list(),
            params = list(value = start, dtype = "i32", shape = integer()),
            infer_fn = infer_fill,
            desc = desc
          )[[1L]]
        )
      }
    },
    after_run = function(values, uses) {
      store$received <- as_array(values[[1L]])
    }
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
    expect_error(register_side_effect("x", AbstractArray(dtype = "i32", shape = integer())), "from the host")
  })
})

describe("new_primitive(side_effects = )", {
  it("refuses a side effect named like an argument of the primitive", {
    expect_error(
      new_primitive("clash", function(print_token) print_token, side_effects = "print_token", register = FALSE),
      "must not have an argument named like one of its side effects"
    )
  })
})

describe("a counted side effect in a trace", {
  it("is taken by every use one step further, and returned by none", {
    x <- local_steps()
    graph <- trace_fn(function() list(x$step(), x$step()), list())
    steps <- graph$side_effects$steps
    calls <- graph$statements
    # the first use takes the value the program takes, the second one more,
    # which is computed from it rather than handed on by the first
    expect_identical(calls[[1L]]$inputs, unname(steps$input))
    expect_identical(calls[[2L]]$inputs[[1L]], steps$input$steps)
    expect_identical(calls[[3L]]$inputs, list(calls[[2L]]$outputs[[1L]]))
    expect_length(calls[[1L]]$outputs, 1L)
    # the graph returns no value of it, but counts the uses it made
    expect_null(steps$output)
    expect_length(calls, 3L)
    expect_identical(steps$advance, 2)
  })

  it("continues its steps in the branches of nv_if and after the larger of them", {
    x <- local_steps()
    graph <- trace_fn(
      function(p) {
        x$step()
        nv_if(p, function() x$step(), function() {
          x$step()
          x$step()
        })
        x$step()
      },
      list(p = nv_aval("bool", integer()))
    )
    expect_identical(graph$side_effects$steps$advance, 4)
  })
})

describe("a counted side effect in jit", {
  it("is taken from the host and advanced there by the uses of a run", {
    x <- local_steps(start = 5L)
    f <- jit(function() list(x$step(), x$step()))
    expect_identical(lapply(f(), as_array), list(5L, 6L))
    expect_identical(x$store$value, 7L)
    expect_identical(lapply(f(), as_array), list(7L, 8L))
  })

  it("advances after the larger branch of nv_if, whichever runs", {
    x <- local_steps()
    f <- jit(function(p) {
      nv_if(p, \() x$step(), \() x$step() + x$step())
      x$step()
    })
    expect_identical(as_array(f(TRUE)), 2L)
    expect_identical(x$store$value, 3L)
  })

  it("is forked for the body of nv_while, which advances the fork", {
    x <- local_steps(start = 1L)
    f <- jit(function(n) {
      out <- nv_while(list(i = 0L, last = 0L), function(i, last) i < n, function(i, last) {
        list(i = i + 1L, last = x$step())
      })
      list(last = out$last, after = x$step())
    })
    out <- f(3L)
    # the fork takes 1 and starts the body from 100; the loop is one use
    expect_identical(as_array(out$last), 102L)
    expect_identical(as_array(out$after), 2L)
    expect_identical(x$store$value, 3L)
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

  it("is carried through the body of nv_scan together with a counted side effect", {
    x <- local_counter()
    s <- local_steps(start = 10L)
    f <- jit(function() {
      out <- nv_scan(
        init = nv_scalar(0L, dtype = "i32"),
        body = function(carry, x_) list(carry = carry + s$step(), out = x$tick()),
        steps = 3L
      )
      list(out = out$out, carry = out$carry, after = s$step())
    })
    out <- f()
    expect_identical(as.vector(as_array(out$out)), 0:2)
    expect_identical(as_array(out$carry), 33L)
    expect_identical(as_array(out$after), 13L)
    expect_identical(x$store$received, 3L)
    expect_identical(s$store$value, 14L)
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
    s <- local_steps()
    graph <- trace_fn(function() list(s$step(), x$tick()), list())
    expect_named(graph$side_effects, c("counter", "steps"))
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
