# A runtime side effect for these tests: an i32 tally on the host, which tick()
# adds to.
tally <- structure(new.env(parent = emptyenv()), class = c("TestTally", "RuntimeSideEffect"))
tally_host <- new.env(parent = emptyenv())
.S3method("side_effect_start", "TestTally", function(effect, desc) {
  inputs <- list(tally = GraphValue(AbstractArray(dtype = "i32", shape = integer()), label = "tally"))
  SideEffectView(effect, register_gvals(desc, inputs), inputs = inputs)
})
.S3method("side_effect_host_object", "TestTally", function(effect, args) tally_host)
.S3method("side_effect_receive", "TestTally", function(effect, object, value) {
  object$tally <- value$tally
  invisible(NULL)
})
tick <- function(by = 1L) {
  x <- desc_side_effect(current_descriptor(), tally)
  x$value <- list(tally = x$value$tally + by)
  invisible(NULL)
}
local_tally <- function(value = 0L) tally_host$tally <- nv_scalar(value, dtype = "i32")
tally_value <- function() as_array(tally_host$tally)

describe("desc_side_effect", {
  it("starts the program's view of the global random state from implicit inputs", {
    graph <- trace_fn(function() list(prim_sample_bits(2L, "ui32"), prim_sample_bits(3L, "ui32")), list())
    expect_length(graph$side_effects, 1L)
    view <- graph$side_effects[[1L]]
    expect_true(identical(view$effect, global_random_state))
    expect_named(view$inputs, c("random_key", "random_counter"))
    expect_identical(lapply(view$value, `[[`, "gnode"), view$inputs)
    expect_identical(view$n, 2)
    # a trace-time side effect hands back its count, not a value
    expect_null(view$outputs)
  })

  it("starts a loop body's view from fresh inputs, which the loop carries", {
    graph <- trace_fn(
      function() {
        nv_scan(
          nv_scalar(0L),
          body = function(carry, x) {
            prim_sample_bits(1L, "ui32")
            prim_sample_bits(1L, "ui32")
            list(carry = carry, out = carry)
          },
          steps = 3L
        )
      },
      list()
    )
    scan <- graph$statements[[length(graph$statements)]]
    # the body takes the key and the counter as more carry and returns them
    expect_identical(scan$params$n_carry, 3L)
    expect_identical(graph$side_effects[[1L]]$n, 6)
  })

  it("starts a branch's view from the enclosing view, continuing its count", {
    graph <- trace_fn(
      function(p) {
        prim_sample_bits(1L, "ui32")
        nv_if(p, function() prim_sample_bits(1L, "ui32"), function() {
          prim_sample_bits(1L, "ui32")
          prim_sample_bits(1L, "ui32")
        })
        prim_sample_bits(1L, "ui32")
      },
      list(p = nv_aval("bool", integer()))
    )
    expect_identical(graph$side_effects[[1L]]$n, 4)
  })

  it("refuses a side effect in a trace without a mode", {
    expect_error(
      trace_fn(
        function() {
          prim_reduce(nv_array(1:2), 0L, function(a, b) a + b + prim_sample_bits(integer(), "ui32"), axes = 1L)
        },
        list()
      ),
      "cannot be used here"
    )
  })
})

describe("desc_use_side_effect", {
  it("hands every use the counter after the uses before it, computed from the one the program takes", {
    graph <- trace_fn(function() list(prim_sample_bits(2L, "ui32"), prim_sample_bits(3L, "ui32")), list())
    inputs <- unname(graph$side_effects[[1L]]$inputs)
    calls <- graph$statements
    # the first draw splits at the counter the program takes, the second at one
    # more, which is computed from it rather than handed on by the first
    expect_identical(calls[[1L]]$inputs, inputs)
    expect_identical(calls[[2L]]$inputs[[1L]], inputs[[2L]])
    expect_identical(calls[[3L]]$inputs, list(inputs[[1L]], calls[[2L]]$outputs[[1L]]))
    expect_length(calls[[1L]]$outputs, 1L)
    expect_length(calls, 3L)
  })
})

describe("is_trace_time", {
  it("is TRUE for the handles of random states", {
    expect_true(is_trace_time(global_random_state))
  })
})

describe("side_effects_on_success", {
  it("is NULL for a graph without side effects", {
    expect_null(side_effects_on_success(trace_fn(function(x) x + 1, list(x = nv_aval("f32", integer())))))
  })

  it("refuses an object two views stand for before it advances any", {
    graph <- trace_fn(
      \(a, b) {
        list(with_nv_rng_state(a, prim_sample_bits(1L, "ui32")), with_nv_rng_state(b, prim_sample_bits(1L, "ui32")))
      },
      list(nv_random_state(1L), nv_random_state(2L))
    )
    rng <- nv_random_state(1L)
    on_success <- side_effects_on_success(graph)
    expect_error(on_success(list(), list(rng, rng)), "passed to the jitted function twice")
    expect_identical(state_words(rng)[[3L]], 0)
    on_success(list(), list(rng, nv_random_state(2L)))
    expect_identical(state_words(rng)[[3L]], 1)
  })

  it("keeps nothing of the trace", {
    graph <- trace_fn(function() prim_sample_bits(2L, "ui32"), list())
    state <- new.env()
    reg.finalizer(graph, function(e) {
      state$collected <- TRUE
    })
    on_success <- side_effects_on_success(graph)
    rm(graph)
    gc()
    expect_true(isTRUE(state$collected))
  })
})

describe("a runtime side effect", {
  it("is taken from the host and handed back after the run", {
    local_tally(5L)
    f <- jit(function(x) {
      tick()
      tick()
      x
    })
    f(nv_scalar(1))
    expect_identical(tally_value(), 7L)
    f(nv_scalar(1))
    expect_identical(tally_value(), 9L)
  })

  it("continues after nv_scan from the loop's result", {
    local_tally()
    f <- jit(function() {
      nv_scan(
        nv_scalar(0L),
        body = function(carry, x) {
          tick()
          list(carry = carry, out = carry)
        },
        steps = 4L
      )$carry
    })
    f()
    expect_identical(tally_value(), 4L)
  })

  it("continues after nv_while from the loop's result, known only at run time", {
    local_tally()
    f <- jit(function(n) {
      nv_while(list(i = 0L), \(i) i < n, \(i) {
        tick()
        list(i = i + 1L)
      })$i
    })
    f(nv_scalar(3L))
    expect_identical(tally_value(), 3L)
    f(nv_scalar(5L))
    expect_identical(tally_value(), 8L)
  })

  it("continues after nv_if from the branch that ran", {
    local_tally()
    f <- jit(function(p) {
      nv_if(
        p,
        function() {
          tick()
          tick()
          nv_scalar(1)
        },
        function() {
          tick()
          nv_scalar(2)
        }
      )
    })
    f(TRUE)
    expect_identical(tally_value(), 2L)
    f(FALSE)
    expect_identical(tally_value(), 3L)
  })

  it("is handed back unchanged by a branch that does not use it", {
    local_tally()
    g <- jit(function(p) {
      nv_if(
        p,
        function() {
          tick()
          nv_scalar(1)
        },
        function() nv_scalar(2)
      )
    })
    g(FALSE)
    expect_identical(tally_value(), 0L)
    g(TRUE)
    expect_identical(tally_value(), 1L)
  })

  it("reaches the program from a branch in a loop in a branch", {
    local_tally()
    f <- jit(function(p) {
      nv_if(
        p,
        function() {
          nv_scan(
            nv_scalar(0L),
            body = function(carry, x) {
              nv_if(
                TRUE,
                function() {
                  tick()
                  carry
                },
                function() carry
              )
              list(carry = carry, out = carry)
            },
            steps = 3L
          )$carry
        },
        function() nv_scalar(0L)
      )
    })
    f(TRUE)
    expect_identical(tally_value(), 3L)
  })

  it("continues after a function gradient() differentiates", {
    local_tally()
    f <- jit(gradient(function(x) {
      tick()
      sum(x * x)
    }))
    out <- f(nv_array(c(1, 2)))
    expect_identical(as.numeric(as_array(out$x)), c(2, 4))
    expect_identical(tally_value(), 1L)
  })

  it("is kept with its output by graph_gradient()", {
    local_tally()
    graph <- trace_fn(
      function(x) {
        tick()
        x * x
      },
      list(x = nv_aval("f32", integer()))
    )
    grad <- graph_gradient(graph, "x")
    expect_length(grad$side_effects, 1L)
    expect_length(grad$side_effects[[1L]]$outputs, 1L)
  })

  it("cannot be used in the function of a reduction", {
    local_tally()
    f <- jit(function() {
      prim_reduce(
        nv_array(1:2),
        0L,
        function(a, b) {
          tick()
          a + b
        },
        axes = 1L
      )
    })
    expect_error(f(), "cannot be used here")
  })

  it("is an input and an output of the printed graph", {
    local_tally()
    expect_snapshot(trace_fn(
      function(x) {
        tick()
        x + 1L
      },
      list(x = nv_aval("i32", integer()))
    ))
  })

  it("hands back its value where the random state hands back its count", {
    local_tally()
    f <- jit(function(p) {
      nv_scan(
        nv_scalar(0),
        body = function(carry, x) {
          tick()
          list(carry = carry + nv_runif(integer()), out = carry)
        },
        steps = 2L
      )
      nv_if(
        p,
        function() {
          tick()
          nv_runif(1L)
        },
        function() nv_runif(1L)
      )
    })
    local_nv_seed(1L)
    f(FALSE)
    # two ticks in the loop and none in the branch that ran
    expect_identical(tally_value(), 2L)
    # two draws in the loop and the larger branch's one
    expect_identical(state_words()[[3L]], 3)
  })

  it("is refused by the quickr backend", {
    skip_if_no_quickr()
    local_tally()
    local_backend("quickr")
    f <- jit(function() {
      tick()
      nv_scalar(1)
    })
    expect_error(f(), "does not support side effects")
  })
})

describe("a side effect on the quickr backend", {
  it("is refused", {
    skip_if_no_quickr()
    local_backend("quickr")
    expect_error(jit(function() prim_sample_bits(2L, "ui32"))(), "does not support side effects")
  })
})
