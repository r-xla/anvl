test_that("jit: basic test", {
  f <- jit(function(x, y) {
    prim_add(x, y)
  })

  expect_equal(
    f(nv_scalar(1), nv_scalar(2)),
    nv_scalar(3)
  )
  expect_equal(
    f(nv_scalar(1), nv_scalar(2)),
    nv_scalar(3)
  )
})

test_that("jit: a constant", {
  x <- nv_scalar(1)
  f <- function(y) {
    prim_add(x, y)
  }
  f_jit <- jit(f)
  expect_equal(
    f_jit(nv_scalar(2)),
    nv_scalar(3)
  )
  x <- nv_scalar(2)
  # the constant is now saved in f_jit, so new x is not found
  cache_size(f_jit)
  expect_equal(
    f_jit(nv_scalar(2)),
    nv_scalar(3)
  )
  cache_size(f_jit)
})

test_that("jit basic test", {
  f <- function(x, y) {
    x + y
  }
  f_jit <- jit(f)

  expect_equal(
    f_jit(nv_array(1), nv_array(2)),
    nv_array(3)
  )
  # cache hit:
  expect_equal(
    f_jit(nv_array(1), nv_array(2)),
    nv_array(3)
  )
})

test_that("can return single array", {
  f <- jit(prim_add)
  expect_equal(
    f(
      nv_array(1.0),
      nv_array(-2.0)
    ),
    nv_array(-1.0)
  )
})

test_that("can return nested list", {
  f <- jit(\(x, y) {
    list(x, list(a = y))
  })
  x <- nv_array(1)
  y <- nv_array(2)
  expect_equal(
    f(x, y),
    list(x, list(a = y))
  )
})

test_that("can take in nested list", {
  f <- jit(\(x) {
    x$a
  })
  x <- nv_array(1)
  expect_equal(
    f(list(a = x)),
    x
  )
})

test_that("Can multiply values from lists", {
  f <- jit(function(x, y) {
    x[[1]] * y[[1]]
  })
  expect_equal(
    f(list(nv_scalar(2)), list(nv_scalar(3))),
    nv_scalar(6)
  )
})

test_that("multiple returns", {
  f <- jit(function(x, y) {
    list(
      prim_add(x, y),
      prim_mul(x, y)
    )
  })

  out <- f(
    nv_array(1.0),
    nv_array(2.0)
  )
  expect_equal(
    out[[1]],
    nv_array(3.0)
  )
  expect_equal(
    out[[2]],
    nv_array(2.0)
  )
})

test_that("jitted function has class JitFunction", {
  f_jit <- jit(function(x) x)
  expect_s3_class(f_jit, "JitFunction")
})

test_that("jit(jit(f)) works (#220)", {
  f_jit <- jit(function(x) x + nv_scalar(1L))
  f_jit2 <- jit(f_jit)
  expect_equal(f_jit2(nv_array(3L)), nv_array(4L))
})

test_that("jit returns R literals and arrays as AnvlArrays", {
  expect_equal(jit(\() 1L)(), nv_scalar(1L))
  expect_equal(jit(\() array(1L))(), nv_array(1L, shape = 1L))
  expect_equal(
    jit(\() array(c(1L, 2L, 3L)))(),
    nv_array(c(1L, 2L, 3L))
  )
})

test_that("keeps argument names", {
  f1 <- function(x, y) {
    x + y
  }
  f1_jit <- jit(f1)
  expect_equal(
    formals(f1_jit),
    formals(f1)
  )
  f1_jit(nv_array(1), nv_array(2))

  expect_equal(
    f1_jit(nv_array(1), nv_array(2)),
    nv_array(3)
  )
  f2 <- function(x, y = nv_array(1)) {
    x + y
  }
  f2_jit <- jit(f2)
  expect_equal(
    formals(f2_jit),
    formals(f2)
  )
  expect_equal(
    f2_jit(nv_array(1), nv_array(2)),
    nv_array(3)
  )
  f3 <- function(a, ...) {
    Reduce(`+`, list(...)) * a
  }
  f3_jit <- jit(f3)
  expect_equal(
    formals(f3_jit),
    formals(f3)
  )
  expect_equal(
    f3_jit(nv_array(2), nv_array(3), nv_array(4)),
    nv_array(14)
  )
})

test_that("can mark arguments as static ", {
  f <- jit(
    function(x, add_one) {
      if (add_one) {
        x + nv_array(1)
      } else {
        x
      }
    },
    static = "add_one"
    # TODO: Better error message ...
  )
  expect_equal(f(nv_array(1), TRUE), nv_array(2))
  expect_equal(f(nv_array(1), FALSE), nv_array(1))
})

test_that("static accepts integer positions", {
  body_fn <- function(x, add_one) {
    if (add_one) x + nv_array(1) else x
  }
  # Positional static arg resolves to the same name.
  f <- jit(body_fn, static = 2L)
  expect_equal(f(nv_array(1), TRUE), nv_array(2))
  expect_equal(f(nv_array(1), FALSE), nv_array(1))

  # Out-of-range index is an error.
  expect_error(jit(body_fn, static = 3L), "out of range")
  expect_error(jit(body_fn, static = 0L), "out of range")
})

test_that("static cannot be '...'", {
  f <- function(x, ...) x
  expect_error(jit(f, static = "..."), "must not contain")
  # Position pointing at `...` is also rejected.
  expect_error(jit(f, static = 2L), "must not contain")
})


test_that("jit: array return value is not wrapped in list", {
  f <- jit(prim_add)
  out <- f(nv_scalar(1.2), nv_scalar(-0.7))
  expect_equal(as_array(out), 1.2 + (-0.7), tolerance = 1e-6)
})

test_that("constants can be part of the program", {
  f <- jit(function(x) x + nv_scalar(1))
  expect_equal(f(nv_array(1)), nv_array(2))
})

test_that("Only constants in group generics", {
  f <- jit(function() {
    nv_scalar(1) + nv_scalar(2)
    #nv_add(nv_scalar(1), nv_scalar(2))
  })
  expect_equal(f(), nv_scalar(3))
})

test_that("... works (#19)", {
  expect_equal(
    jit(sum)(nv_array(1:10)),
    nv_scalar(55L, dtype = "i32")
  )

  f <- function(..., a) {
    a + sum(...)
  }
  expect_equal(
    jit(f)(a = nv_scalar(1L), nv_array(1:10)),
    nv_scalar(56L, dtype = "i32")
  )
})

test_that("good error message when passing AbstractArrays", {
  expect_error(
    jit(nv_negate)(nv_aval("f32", c(2, 2))),
    "invalid input `x`.*<AbstractArray>"
  )
})

test_that("jit_eval evaluates a scalar expression", {
  expect_equal(as_array(jit_eval(nv_scalar(1) + nv_scalar(2))), 3)
})

test_that("jit_eval does not modify calling environment", {
  x <- nv_array(1:2)
  jit_eval({
    x <- nv_array(3:4)
  })
  expect_equal(x, nv_array(1:2))
})

test_that("nested jit: jitted function can be called inside jit (#220)", {
  inner <- jit(function(x, y) x + y)
  outer <- jit(function(a, b) inner(a, b) * nv_scalar(2L))
  result <- outer(nv_array(3L), nv_array(4L))
  expect_equal(result, nv_array(14L))
})

test_that("hash for cache depends on in_tree (#122)", {
  f <- jit(
    \(...) {
      args <- list(...)
      args[[1]][[1L]][[1L]]
    }
  )
  expect_equal(cache_size(f), 0L)
  expect_equal(f(list(list(nv_scalar(1L)), nv_scalar(2L))), nv_scalar(1L))
  expect_equal(cache_size(f), 1L)
  expect_equal(f(list(list(nv_scalar(1L), nv_scalar(2L)))), nv_scalar(1L))
  expect_equal(cache_size(f), 2L)
})

describe("jit: device and backend handling", {
  it("backend = NULL, device = NULL uses default_backend()", {
    local_backend("pjrt")
    f <- jit(identity)
    expect_equal(backend(f), "pjrt")
    expect_equal(backend(f(1)), "pjrt")
  })

  it("backend = NULL, device = NULL follows default_backend() = 'quickr'", {
    skip_if_no_quickr()
    local_backend("quickr")
    f <- jit(identity)
    expect_equal(backend(f), "quickr")
    expect_equal(backend(f(1)), "quickr")
  })

  it("backend = 'pjrt', device = NULL uses pjrt", {
    local_backend("quickr")
    f <- jit(identity, backend = "pjrt")
    expect_equal(backend(f), "pjrt")
    expect_equal(backend(f(1)), "pjrt")
  })

  it("concrete device string", {
    f <- jit(identity, device = "cpu")
    expect_equal(backend(f), "pjrt")
    expect_equal(backend(f(1)), "pjrt")
  })

  it("concrete device object", {
    f <- jit(identity, device = pjrt::pjrt_device("cpu"))
    expect_equal(backend(f), "pjrt")
    expect_equal(backend(f(1)), "pjrt")
  })

  it("concrete device infers backend from device", {
    skip_if_no_quickr()
    local_backend("quickr")
    f <- jit(identity, device = pjrt::pjrt_device("cpu"))
    expect_equal(backend(f), "pjrt")
    expect_equal(backend(f(1)), "pjrt")
  })

  it("concrete device conflicts with mismatched backend", {
    skip_if_no_quickr()
    expect_error(
      jit(identity, backend = "quickr", device = pjrt::pjrt_device("cpu")),
      "Backend of requested device"
    )
  })

  it("constant's device can be defined via static argument", {
    f <- jit(function(x) nv_scalar(1, device = x), static = "x")
    expect_equal(device(f("cpu:0")), nv_device("cpu:0", "pjrt"))
    expect_equal(device(f("cpu:1")), nv_device("cpu:1", "pjrt"))
  })

  it("backend 'auto' works with pjrt and quickr input", {
    f <- jit(identity, backend = "auto")
    expect_equal(backend(f), "auto")
    # At call time, backend is picked from the input.
    expect_equal(backend(f(nv_scalar(1, backend = "pjrt"))), "pjrt")
    skip_if_no_quickr()
    expect_equal(backend(f(nv_scalar(1, backend = "quickr"))), "quickr")
  })

  it("backend 'auto' routes to quickr when all inputs are quickr", {
    skip_if_no_quickr()
    f <- jit(nv_add, backend = "auto")
    out <- f(nv_scalar(1, backend = "quickr"), nv_scalar(2, backend = "quickr"))
    expect_equal(backend(out), "quickr")
    expect_equal(as_array(out), 3)
  })

  it("backend 'auto' errs when call-time inputs use multiple backends", {
    skip_if_no_quickr()
    f <- jit(nv_add, backend = "auto")
    expect_error(
      f(nv_scalar(1, backend = "pjrt"), nv_scalar(2, backend = "quickr")),
      "multiple backends"
    )
  })

  it("cannot mix backends via closed-over constant", {
    # A closed-over constant from a different backend than the call-time input
    # must not silently compile on either backend.
    skip_if_no_quickr()
    const_q <- nv_scalar(1, backend = "quickr")
    f <- jit(function(x) x + const_q, backend = "pjrt")
    expect_error(
      f(nv_scalar(1, backend = "pjrt")),
      "Cannot compile a \"pjrt\" program"
    )
    const_x <- nv_scalar(1, backend = "pjrt")
    g <- jit(function(x) x + const_x, backend = "quickr")
    expect_error(
      g(nv_scalar(1, backend = "quickr")),
      "Cannot compile a \"quickr\" program"
    )
  })

  it("character device with backend = 'auto' is honored per chosen backend", {
    skip_if_no_quickr()
    f <- jit(identity, device = "cpu", backend = "auto")
    expect_equal(device(f(nv_scalar(1, backend = "pjrt"))), nv_device("cpu", "pjrt"))
    expect_equal(device(f(nv_scalar(1, backend = "quickr"))), nv_device("cpu", "quickr"))
  })

  it("concrete device with backend = 'auto' collapses to the device's backend", {
    skip_if_no_quickr()
    expect_error(
      jit(identity, device = nv_device("cpu", "quickr"), backend = "auto"),
      "Don't provide"
    )
  })

  it("device_arg caches separately per device value", {
    skip_if_no_quickr()
    f <- jit(
      function(dev) nv_scalar(1, device = dev),
      backend = "auto",
      device = device_arg("dev")
    )
    out_q <- f(nv_device("cpu", "quickr"))
    out_x <- f(nv_device("cpu", "pjrt"))
    expect_equal(backend(out_q), "quickr")
    expect_equal(backend(out_x), "pjrt")
  })

  it("converts constants with device specification to specified device", {
    f <- jit(function() nv_scalar(1), device = "cpu:1")
    expect_equal(device(f()), nv_device("cpu:1", "pjrt"))
  })

  it("device overrides found constant's device", {
    expect_equal(
      device(jit(\() nv_scalar(1, device = "cpu:1"), device = "cpu:0")()),
      nv_device("cpu:0", "pjrt")
    )
  })

  it("errs when finding inputs with different devices (when jit does not set concrete device)", {
    f <- function(x, y) x + y
    g <- jit(f)
    expect_error(
      g(nv_scalar(1, device = "cpu:0"), nv_scalar(2, device = "cpu:1")),
      "invalid input `y`.*different device"
    )
  })
  it("allocates scalar on default device when there is no AnvlArray to infer from", {
    g <- jit(function() 1, backend = "pjrt")
    expect_equal(device(g()), default_device("pjrt"))
  })
  it("uses specified device when input is R object", {
    g <- jit(function() 1, device = "cpu:1")
    expect_equal(device(g()), nv_device("cpu:1", "pjrt"))
  })
  it("errs when finding conflicting constants", {
    skip_if(!is_cuda())
    const <- nv_array(1:2, device = "cuda")
    f <- jit(function(x) {
      x * const
    })
    expect_error(
      f(nv_scalar(1, device = "cpu")),
      "more than one"
    )
  })
  it("works with different device IDs for 'pjrt' backend", {
    f <- jit(identity)
    expect_equal(
      device(f(nv_scalar(1, device = "cpu:0"))),
      nv_device("cpu:0", "pjrt")
    )
    skip_if(!is_cuda())
    expect_equal(
      device(f(nv_scalar(1, device = "cuda"))),
      nv_device("cuda", "pjrt")
    )
  })
  it("works with different devices with 'auto' backend", {
    f <- jit(nv_log, backend = "auto")
    expect_equal(
      device(f(nv_scalar(1, device = "cpu"))),
      nv_device("cpu", "pjrt")
    )
    skip_if(!is_cuda())
    expect_equal(
      device(f(nv_scalar(1, device = "cuda"))),
      nv_device("cuda", "pjrt")
    )
  })

  it("works when passing device as static arg for concrete backend", {
    f <- function(x) nv_scalar(1, device = x)
    g <- jit(f, backend = "pjrt", static = "x")
    expect_equal(device(g("cpu:0")), nv_device("cpu:0", "pjrt"))
    expect_equal(device(g("cpu:1")), nv_device("cpu:1", "pjrt"))
  })

  # device_arg
  it("device-arg works with concrete 'pjrt' backend", {
    local_backend("pjrt")
    f <- function(dev) nv_scalar(1, device = dev)
    expect_error(
      g <- jit(f, device = device_arg("dev"), backend = "pjrt"),
      "is only allowed"
    )
  })

  it("uses default backend when device_arg is character(1)", {
    local_backend("pjrt")
    f <- function(x) nv_scalar(1, device = x)
    g <- jit(f, device = device_arg("x"), backend = "auto")
    expect_equal(device(g("cpu:0")), nv_device("cpu:0", "pjrt"))
    expect_equal(device(g("cpu:1")), nv_device("cpu:1", "pjrt"))
    skip_if_no_quickr()
    expect_equal(device(g(nv_device("cpu", "quickr"))), nv_device("cpu", "quickr"))
  })

  it("device_arg works with backend = NULL (uses default backend)", {
    local_backend("pjrt")
    f <- jit(function(dev) 1L, device = device_arg("dev"), backend = NULL)
    expect_equal(device(f(nv_device("cpu:1", "pjrt"))), nv_device("cpu:1", "pjrt"))
  })

  it("device_arg works with 'auto' backend", {
    # device_arg is for JitFunctions that should work with any backend and infer device
    # as runtime arg.
    f <- jit(
      function(val, dev) {
        nv_array(val, device = dev)
      },
      device = device_arg("dev"),
      backend = "auto",
      static = c("val", "dev")
    )
    dev0 <- nv_device("cpu", "pjrt")
    expect_true(device(f(1, dev0)) == dev0)
    skip_if_no_quickr()
    dev1 <- nv_device("cpu", "quickr")
    expect_true(device(f(1, dev1)) == dev1)
  })

  it("literal's device can be defined via device_arg", {
    f <- jit(function(dev) 1L, device = device_arg("dev"))
    expect_equal(device(f("cpu:0")), nv_device("cpu:0", "pjrt"))
    expect_equal(device(f("cpu:1")), nv_device("cpu:1", "pjrt"))
  })
})

test_that("cache hit when using PJRTDevice", {
  # this used to be a bug before pjrt 0.2.0, because every PJRTDevice was a new external pointer
  # and hashtab hashes address of xptr

  # we don't need device_arg() as it is only for making jit backend-agnostic

  f <- jit(function(dev) nv_scalar(1, device = dev), static = "dev")
  dev0 <- nv_device("cpu", "pjrt")
  dev1 <- nv_device("cpu", "pjrt")
  f(dev = dev0)
  f(dev = dev1)
  expect_equal(cache_size(f), 1L)
})

test_that("static arguments with reference semantics are rejected", {
  e <- new.env()
  e$flag <- TRUE

  f <- jit(function(e, x) if (e$flag) x else -x, static = "e")
  expect_error(f(e, nv_array(1)), "reference semantics")

  # A reference class object is an environment underneath (so is an R6 object).
  gen <- methods::setRefClass("StaticRefCls", fields = list(flag = "logical"))
  g <- jit(function(o, x) if (o$flag) x else -x, static = "o")
  expect_error(g(gen$new(flag = TRUE), nv_array(1)), "reference semantics")

  # External pointer.
  h <- jit(function(p, x) x + 1, static = "p")
  expect_error(h(pjrt::pjrt_scalar(1), nv_array(1)), "reference semantics")

  # pjrt flattens a static list into one cache-key leaf per element, so an
  # environment nested in one goes stale just like a bare one.
  k <- jit(function(s, x) if (s$e$flag) x else -x, static = "s")
  expect_error(k(list(e = e), nv_array(1)), "reference semantics")
  # Also below a classed list and behind an unnamed element.
  expect_error(k(structure(list(e = e), class = "cfg"), nv_array(1)), "reference semantics")
  l <- jit(function(s, x) x + 1, static = "s")
  expect_error(l(list(1, e), nv_array(1)), "reference semantics")
})

test_that("static arguments with reference semantics are rejected (quickr)", {
  skip_if_no_quickr()
  local_backend("quickr")

  e <- new.env()
  e$flag <- TRUE
  f <- jit(function(e, x) if (e$flag) x else -x, static = "e")
  expect_error(f(e, nv_array(1)), "reference semantics")
})

test_that("static arguments without reference semantics are accepted", {
  # A function: keyed on its formals, body and environment.
  f <- jit(function(fn, x) fn(x), static = "fn")
  expect_equal(f(function(z) z + 1, nv_array(1)), nv_array(2))
  expect_equal(f(function(z) z * 3, nv_array(2)), nv_array(6))

  # A device is an external pointer, but an immutable interned one.
  g <- jit(function(dev) nv_scalar(1, device = dev), static = "dev")
  expect_equal(device(g(nv_device("cpu", "pjrt"))), nv_device("cpu", "pjrt"))

  # A plain list of values, and a formula (whose `.Environment` attribute is
  # metadata, not a value the trace reads).
  h <- jit(function(s, x) if (s$flag) x else -x, static = "s")
  expect_equal(h(list(flag = FALSE), nv_array(1)), nv_array(-1))
  k <- jit(function(s, x) x + 1, static = "s")
  expect_equal(k(y ~ x, nv_array(1)), nv_array(2))
})

test_that("rejecting a reference-semantics static names it helpfully", {
  e <- new.env()
  f <- jit(function(s, x) x + 1, static = "s")
  expect_snapshot(f(list(a = 1, opts = list(env = e)), nv_array(1)), error = TRUE)
})

# jit() driving pjrt's native dispatcher, end to end.
#
# These live here rather than in pjrt because they are an *integration* check:
# that anvl's real compile callback returns what pjrt's engine expects, that a
# real AnvlArray carries $data/$backend/$device where the engine looks for them,
# and that the wrapped outputs come back as anvl arrays. pjrt tests its
# dispatcher against hand-written fixtures; only anvl can test that the two
# actually agree.

skip_if_no_jit <- function() {
  testthat::skip_if_not(pjrt::plugins_downloaded())
}

# The dispatcher a jitted function dispatches through (stored in the closure
# environment of the function's fast entry).
jit_dispatcher <- function(f) {
  environment(attr(f, "jit_run_args"))$dispatcher
}
jit_size <- function(f) pjrt::dispatcher_size(jit_dispatcher(f))

arr_of <- function(res) as.numeric(tengen::as_array(res))

describe("jit: native dispatch", {
  it("dispatches, caches, and returns wrapped arrays", {
    skip_if_no_jit()
    f <- jit(function(x, y) x + y)
    x <- nv_array(c(1, 2, 3), dtype = "f32")
    y <- nv_array(c(10, 20, 30), dtype = "f32")

    r1 <- f(x, y)
    # The result is a fully wrapped array: the dispatcher built it natively.
    expect_s3_class(r1, "AnvlArray")
    expect_identical(as.character(tengen::dtype(r1)), "f32")
    expect_identical(tengen::shape(r1), 3L)
    expect_s3_class(r1$data, "PJRTBuffer")
    expect_s3_class(tengen::device(r1), "PJRTDevice")
    expect_identical(r1$backend, "pjrt")
    expect_equal(arr_of(r1), c(11, 22, 33))

    # A second call of the same signature is a cache hit...
    expect_equal(arr_of(f(x, y)), c(11, 22, 33))
    d <- jit_dispatcher(f)
    expect_s3_class(d, "Dispatcher")
    expect_equal(pjrt::dispatcher_size(d), 1L)

    # ...an output feeds straight back in as an input, without re-compiling...
    expect_equal(arr_of(f(r1, y)), c(21, 42, 63))
    expect_equal(pjrt::dispatcher_size(d), 1L)

    # ...and a new shape is a new cache entry.
    invisible(f(nv_array(1, dtype = "f32"), nv_array(2, dtype = "f32")))
    expect_equal(pjrt::dispatcher_size(d), 2L)

    # GC-correct: many dispatches with periodic gc(), then teardown.
    for (i in 1:300) {
      r <- f(x, y)
      if (i %% 100 == 0) {
        gc()
      }
      expect_equal(arr_of(r), c(11, 22, 33))
    }
  })

  it("preserves nested output structure and names", {
    skip_if_no_jit()
    f <- jit(function(x) list(sum = x + x, nested = list(sq = x * x)))
    res <- f(nv_array(c(2, 3), dtype = "f32"))
    expect_named(res, c("sum", "nested"))
    expect_named(res$nested, "sq")
    expect_equal(arr_of(res$sum), c(4, 6))
    expect_equal(arr_of(res$nested$sq), c(4, 9))
  })

  it("with static args compiles per static value", {
    skip_if_no_jit()
    f <- jit(function(x, flag) if (flag) x + 1 else x * 2, static = "flag")
    x <- nv_array(3, dtype = "f32")
    expect_equal(arr_of(f(x, TRUE)), 4)
    expect_equal(arr_of(f(x, FALSE)), 6)
    expect_equal(arr_of(f(x, TRUE)), 4) # hit
    expect_equal(jit_size(f), 2L)
  })

  it("a jitted call with no dynamic input dispatches on its statics alone", {
    skip_if_no_jit()
    # Zero dynamic leaves: the whole call is the static `n`, and the entry's
    # device comes from the compile callback rather than from an input.
    f <- jit(function(n) nv_eye(n), static = "n")
    expect_equal(tengen::as_array(f(2L)), diag(2))
    expect_equal(tengen::as_array(f(2L)), diag(2))
    expect_equal(jit_size(f), 1L)
  })

  it("uploads bare R literals and arrays", {
    skip_if_no_jit()
    f <- jit(function(x, y) x + y)
    x <- nv_array(c(1, 2), dtype = "f32")

    # A bare double literal is uploaded as a rank-0 f32 buffer per call; the
    # signature does not change, so the second call is a cache hit.
    expect_equal(arr_of(f(x, 5)), c(6, 7))
    expect_equal(arr_of(f(x, 50)), c(51, 52))
    expect_equal(jit_size(f), 1L)

    # kArray and kRData are *different* key material: bare R data has no dtype of
    # its own, so it cannot share an entry with an array that has one -- the two
    # compile to different programs, one taking an f32 input and one uploading the
    # R value at whatever dtype the trace decided.
    expect_equal(arr_of(f(x, nv_scalar(5, dtype = "f32"))), c(6, 7))
    expect_equal(jit_size(f), 2L)

    # A different R storage type is a different key again: `3L` is an R integer,
    # keyed apart from an R double even though both end up at f32 here.
    expect_equal(arr_of(f(x, 3L)), c(4, 5))
    expect_equal(jit_size(f), 3L)

    # An R array leaf uploads column-major, like pjrt_buffer().
    g <- jit(function(x) x)
    m <- matrix(c(1, 2, 3, 4), nrow = 2)
    expect_equal(tengen::as_array(g(m)), m)
  })

  it("every dtype is its own cache entry", {
    skip_if_no_jit()
    # Every dtype an AnvlDtype names -- which is every dtype tengen can build.
    dtypes <- c("bool", "i8", "i16", "i32", "i64", "ui8", "ui16", "ui32", "ui64", "f32", "f64")
    f <- jit(function(x) x)
    for (dt in dtypes) {
      invisible(f(nv_array(c(1, 2), dtype = dt)))
    }
    expect_equal(jit_size(f), length(dtypes))

    # Same dtype and shape, different values -> cache hit.
    g <- jit(function(x) x)
    invisible(g(nv_array(c(1, 2), dtype = "f64")))
    invisible(g(nv_array(c(7, 7), dtype = "f64")))
    expect_equal(jit_size(g), 1L)
  })

  it("invalid jit() inputs are rejected natively, naming the argument", {
    skip_if_no_jit()
    f <- jit(function(x, y) x + y)
    x <- nv_array(c(1, 2), dtype = "f32")
    expect_error(f(x, "nope"), "invalid input `y`.*<character> of length 1")
    expect_error(f(x, c(1, 2, 3)), "invalid input `y`.*<numeric> of length 3")
    expect_equal(jit_size(f), 0L) # rejected before any compile

    # A static argument must not be an AnvlArray: it would key the cache on its
    # contents, and be traced as an input execution never supplies.
    g <- jit(function(x, s) x + 1, static = "s")
    expect_error(g(x, x), "invalid static input `s`.*must not be an AnvlArray")
    expect_equal(jit_size(g), 0L)
  })

  it("rejects inputs spread across devices, naming the input", {
    skip_if_no_jit()
    skip_if(length(pjrt::devices(pjrt::pjrt_client("cpu"))) < 2L, "needs a second cpu device")
    f <- jit(function(x, y) x + y)
    x0 <- nv_array(c(1, 2), dtype = "f32", device = "cpu:0")
    y1 <- nv_array(c(3, 4), dtype = "f32", device = "cpu:1")

    # Without a fixed target device the first array's device is the call's, and a
    # conflicting input is an error -- caught natively, before the cache is
    # probed, so nothing is compiled.
    expect_error(f(x0, y1), "invalid input `y`.*different device")
    expect_equal(jit_size(f), 0L)
  })

  it("jit(device = ) fixes the entry's device and moves inputs to it", {
    skip_if_no_jit()
    skip_if(length(pjrt::devices(pjrt::pjrt_client("cpu"))) < 2L, "needs a second cpu device")
    f <- jit(function(x) x + 1, device = "cpu:0")
    x0 <- nv_array(c(1, 2), dtype = "f32", device = "cpu:0")
    res <- f(x0)
    expect_equal(arr_of(res), c(2, 3))
    # Devices are interned, so the wrapped output carries the very object.
    expect_identical(tengen::device(res), pjrt::pjrt_device("cpu:0"))

    # An input on another device is copied to the target rather than rejected,
    # and the device is not part of the key: one entry serves both.
    y1 <- nv_array(c(3, 4), dtype = "f32", device = "cpu:1")
    expect_equal(arr_of(f(y1)), c(4, 5))
    expect_equal(jit_size(f), 1L)
  })

  it("a jitted function with no array inputs keys on the default device", {
    skip_if_no_jit()
    f <- jit(function(n) n + 1)
    expect_equal(arr_of(f(41)), 42)
    expect_equal(arr_of(f(41)), 42)
    expect_equal(jit_size(f), 1L)
  })

  it("the quickr backend dispatches through the closure engine", {
    skip_if_no_quickr()
    with_backend("quickr", {
      f <- jit(function(x, y) x + y)
      x <- nv_array(c(1, 2), dtype = "f64")
      y <- nv_array(c(10, 20), dtype = "f64")
      r1 <- f(x, y)
      expect_s3_class(r1, "AnvlArray")
      expect_identical(r1$backend, "quickr")
      expect_equal(arr_of(r1), c(11, 22))
      invisible(f(x, y))
      expect_equal(jit_size(f), 1L)
    })
  })
})
