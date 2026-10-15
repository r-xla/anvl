# A side effect counting the calls of the primitive `tick` it returns, whose
# value lives in `store$value`. Both are removed when `envir` exits.
local_counter <- function(envir = parent.frame()) {
  store <- new.env(parent = emptyenv())
  store$value <- 0
  register_side_effect(
    "counter",
    aval = AbstractArray(dtype = "i32", shape = integer()),
    get = function() store$value,
    advance = function(n) {
      store$value <- store$value + n
    },
    advance_box = function(box, n) box + as.integer(n),
    fork = function(desc) cli_abort("A counter cannot be forked."),
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
})

describe("new_primitive(side_effects = )", {
  it("refuses a side effect named like an argument of the primitive", {
    expect_error(
      new_primitive("clash", function(random_state) random_state, side_effects = "random_state", register = FALSE),
      "must not have an argument named like one of its side effects"
    )
  })
})

describe("a side effect in jit", {
  it("is read before every call and advanced on the host after it", {
    x <- local_counter()
    f <- jit(function() x$tick())
    expect_identical(as_array(f()), 0L)
    expect_identical(x$store$value, 1)
    expect_identical(as_array(f()), 1L)
    expect_identical(x$store$value, 2)
  })

  it("is threaded through every use in a call", {
    x <- local_counter()
    f <- jit(function() list(x$tick(), x$tick()))
    expect_identical(lapply(f(), as_array), list(0L, 1L))
    expect_identical(x$store$value, 2)
  })

  it("is advanced by nv_if by the larger branch, whichever runs", {
    x <- local_counter()
    f <- jit(function(p) {
      list(nv_if(p, \() x$tick(), \() nv_scalar(-1L, dtype = "i32")), x$tick())
    })
    expect_identical(lapply(f(TRUE), as_array), list(0L, 1L))
    expect_identical(lapply(f(FALSE), as_array), list(-1L, 3L))
    expect_identical(x$store$value, 4)
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
    expect_identical(x$store$value, 3)
    expect_identical(unclass(nv_get_random_state())[[3L]], 3)
  })

  it("comes in registration order, whatever is used first", {
    x <- local_counter()
    graph <- trace_fn(function() list(x$tick(), nv_runif(1L)), list())
    expect_named(graph$side_effects, c("random_state", "counter"))
  })

  it("is not returned by the program", {
    x <- local_counter()
    graph <- trace_fn(function() x$tick(), list())
    func <- stablehlo(graph)[[1L]]
    expect_length(func$inputs, 1L)
    expect_length(func$outputs, 1L)
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
