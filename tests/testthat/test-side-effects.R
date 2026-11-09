describe("desc_view", {
  it("starts the program's view of the global random state from implicit inputs", {
    graph <- trace_fn(function() list(prim_sample_bits(2L, "ui32"), prim_sample_bits(3L, "ui32")), list())
    expect_length(graph$side_effects, 1L)
    view <- graph$side_effects[[1L]]
    expect_true(identical(view$effect, global_rng_state_handle))
    expect_named(view$inputs, c("key", "counter"))
    expect_identical(lapply(view$value, `[[`, "gnode"), view$inputs)
    expect_identical(view$n, 2)
    # the program hands back its count, not a value
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

describe("is_side_effect", {
  it("is TRUE for the handles of random states, and not for the random states", {
    expect_true(is_side_effect(global_rng_state_handle))
    expect_false(is_side_effect(nv_rng_state(1L)))
  })
})

describe("side_effects_on_success", {
  it("is NULL for a graph without side effects", {
    expect_null(side_effects_on_success(trace_fn(function(x) x + 1, list(x = nv_aval("f32", integer())))))
  })

  it("refuses an object two views stand for before it advances any", {
    graph <- trace_with_rng_states(
      \(a, b) {
        list(with_nv_rng_state(a, prim_sample_bits(1L, "ui32")), with_nv_rng_state(b, prim_sample_bits(1L, "ui32")))
      },
      list(nv_rng_state(1L), nv_rng_state(2L))
    )
    rng <- nv_rng_state(1L)
    on_success <- side_effects_on_success(graph)
    expect_error(on_success(list(), list(rng, rng)), "passed to the jitted function twice")
    expect_identical(state_words(rng)[[3L]], 0)
    on_success(list(), list(rng, nv_rng_state(2L)))
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

describe("a side effect on the quickr backend", {
  it("is refused", {
    skip_if_no_quickr()
    local_backend("quickr")
    expect_error(jit(function() prim_sample_bits(2L, "ui32"))(), "does not support side effects")
  })
})
