# A side effect counting the calls of the primitive `tick` it returns, whose
# value lives in `store$value`. Both are removed when `envir` exits.
local_counter <- function(envir = parent.frame()) {
  store <- new.env(parent = emptyenv())
  register_side_effect(
    "counter",
    aval = AbstractArray(dtype = "i32", shape = integer()),
    getter = function() {
      if (is.null(store$value)) {
        store$value <- nv_scalar(0L, dtype = "i32")
      }
      store$value
    },
    setter = function(value) {
      store$value <- value
    },
    rules = list(pjrt = pjrt_state_slot)
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
      new_primitive("clash", function(rng) rng, side_effects = "rng", register = FALSE),
      "must not have an argument named like one of its side effects"
    )
  })
})

describe("a side effect in jit", {
  it("is read before every call and stored after it", {
    x <- local_counter()
    f <- jit(function() x$tick())
    expect_identical(as_array(f()), 0L)
    expect_identical(as_array(f()), 1L)
    expect_identical(as_array(x$store$value), 2L)
  })

  it("is threaded through every use in a call", {
    x <- local_counter()
    f <- jit(function() list(x$tick(), x$tick()))
    expect_identical(lapply(f(), as_array), list(0L, 1L))
    expect_identical(as_array(x$store$value), 2L)
  })

  it("is threaded through the branch nv_if takes", {
    x <- local_counter()
    f <- jit(function(p) nv_if(p, \() x$tick(), \() nv_scalar(-1L, dtype = "i32")))
    expect_identical(as_array(f(TRUE)), 0L)
    expect_identical(as_array(f(FALSE)), -1L)
    expect_identical(as_array(f(TRUE)), 1L)
  })

  it("is carried through the body of nv_scan together with the global RNG state", {
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
    expect_identical(as_array(x$store$value), 3L)
  })

  it("comes in registration order, whatever is used first", {
    x <- local_counter()
    graph <- trace_fn(function() list(x$tick(), nv_runif(1L)), list())
    expect_named(graph$side_effects, c("rng", "counter"))
  })

  it("cannot be used in the function of a reduction", {
    x <- local_counter()
    expect_error(
      trace_fn(function() prim_reduce(nv_array(1:2), 0L, function(a, b) a + b + x$tick(), axes = 1L), list()),
      'The side effect "counter" cannot be used here'
    )
  })

  it("is refused by a backend without a rule for it", {
    skip_if_no_quickr()
    x <- local_counter()
    local_backend("quickr")
    f <- jit(function() x$tick())
    expect_error(f(), 'The quickr backend does not support the side effect "counter"')
  })
})
