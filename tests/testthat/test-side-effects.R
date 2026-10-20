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

describe("a side effect in a trace", {
  it("is handed from use to use", {
    x <- local_counter(from = "host")
    graph <- trace_fn(function() list(x$tick(), x$tick()), list())
    counter <- graph$side_effects$counter
    calls <- graph$statements
    # the first use takes the value the program takes, the second the one the
    # first returns, which the graph returns to the host
    expect_identical(calls[[1L]]$inputs, unname(counter$input))
    expect_identical(calls[[2L]]$inputs, list(calls[[1L]]$outputs[[2L]]))
    expect_identical(counter$output, list(counter = calls[[2L]]$outputs[[2L]]))
  })

  it("is taken from the host and dropped by a program the host does not receive it from", {
    x <- local_counter(from = "host", received = FALSE)
    graph <- trace_fn(function() x$tick(), list())
    expect_length(graph$side_effects$counter$input, 1L)
    expect_null(graph$side_effects$counter$output)
    func <- stablehlo(graph)[[1L]]
    expect_length(func$inputs, 1L)
    expect_length(func$outputs, 1L)
  })

  it("is an input and an output of a loop body", {
    x <- local_counter()
    graph <- trace_fn(
      function(n) {
        nv_while(list(i = 0L), function(i) i < n, function(i) {
          x$tick()
          list(i = i + 1L)
        })
      },
      list(n = nv_aval("i32", integer()))
    )
    loop <- graph$statements[[length(graph$statements)]]
    body <- loop$params$body
    # the loop state `i`, the counter, then the captured `n`
    expect_length(body$inputs, 3L)
    expect_identical(body$inputs[[2L]]$label, "counter")
    # the loop state `i`, then the counter after the iteration's use
    expect_length(body$outputs, 2L)
    expect_identical(body$outputs[[2L]]$label, "counter")
    expect_length(body$side_effects, 0L)
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

  it("is carried through the body of nv_scan, after the user's carry", {
    x <- local_counter(start = 10L)
    f <- jit(function() {
      out <- nv_scan(
        init = nv_scalar(0L, dtype = "i32"),
        body = function(carry, x_) {
          value <- x$tick()
          list(carry = carry + value, out = value)
        },
        steps = 3L
      )
      list(out = out$out, carry = out$carry, after = x$tick())
    })
    out <- f()
    expect_identical(as.vector(as_array(out$out)), 10:12)
    expect_identical(as_array(out$carry), 33L)
    expect_identical(as_array(out$after), 13L)
    expect_identical(x$store$received, 14L)
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
    # the print token is registered when anvl is loaded, before the counter
    x <- local_counter()
    graph <- trace_fn(function() list(x$tick(), nv_print(1L, ordered = TRUE)), list())
    expect_named(graph$side_effects, c("print_token", "counter"))
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
