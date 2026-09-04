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

describe("jit: option validation", {
  it("rejects the removed `backend` argument, pointing at with_backend()", {
    expect_error(jit(identity, backend = "quickr"), "no .*backend.* argument")
    expect_error(jit(identity, backend = "quickr"), "with_backend")
  })

  it("rejects an option no backend takes, when the function is created", {
    expect_error(jit(identity, nonsense = 1), "No backend takes")
    expect_error(jit(identity, foo = 1, bar = 2), "No backend takes")
  })

  it("rejects an unnamed option", {
    expect_error(jit(identity, character(), 100L, NULL, TRUE), "must be a named backend option")
  })

  it("rejects an option another backend takes, when it is called", {
    # The backend is not known until the call, so this cannot be caught earlier.
    f <- jit(identity, unwrap = TRUE)
    expect_error(f(nv_array(1)), "pjrt.*does not support.*unwrap")
    skip_if_no_quickr()
    # quickr does take it: `unwrap` hands back a plain R array.
    expect_equal(with_backend("quickr", f(nv_array(1))), array(1))
    g <- jit(identity, donate = "x")
    expect_error(with_backend("quickr", g(nv_array(1))), "quickr.*does not support.*donate")
  })
})

describe("jit: backend and device handling", {
  it("runs on the backend in force when it is called", {
    f <- jit(identity)
    expect_equal(backend(f(1)), "pjrt")
    skip_if_no_quickr()
    expect_equal(with_backend("quickr", backend(f(1))), "quickr")
    expect_equal(backend(f(1)), "pjrt")
  })

  it("rejects an array of another backend", {
    skip_if_no_quickr()
    f <- jit(identity)
    x <- with_backend("quickr", nv_scalar(1))
    expect_error(f(x), "quickr")
    y <- nv_scalar(1)
    expect_error(with_backend("quickr", f(y)), "pjrt")
  })

  it("cannot mix backends via closed-over constant", {
    # A closed-over constant from a different backend than the call-time input
    # must not silently compile on either backend.
    skip_if_no_quickr()
    const_q <- with_backend("quickr", nv_scalar(1))
    f <- jit(function(x) x + const_q)
    expect_error(f(nv_scalar(1)), "Cannot compile a \"pjrt\" program")
    const_x <- nv_scalar(1)
    g <- jit(function(x) x + const_x)
    expect_error(with_backend("quickr", g(nv_scalar(1))), "Cannot compile a \"quickr\" program")
  })

  it("concrete device string", {
    f <- jit(identity, device = "cpu")
    expect_equal(device(f(1)), nv_device("cpu"))
  })

  it("concrete device object", {
    f <- jit(identity, device = pjrt::pjrt_device("cpu"))
    expect_equal(backend(f(1)), "pjrt")
  })

  it("rejects a device of another backend, and a non-device", {
    skip_if_no_quickr()
    f <- jit(identity, device = pjrt::pjrt_device("cpu"))
    expect_error(with_backend("quickr", f(1)), "backend in force")
    expect_error(jit(identity, device = 1L), "must be a device")
  })

  it("resolves a character device on the backend in force", {
    skip_if_no_quickr()
    f <- jit(identity, device = "cpu")
    expect_equal(device(f(nv_scalar(1))), nv_device("cpu"))
    with_backend("quickr", expect_equal(device(f(nv_scalar(1))), nv_device("cpu")))
  })

  it("constant's device can be defined via static argument", {
    f <- jit(function(x) nv_scalar(1, device = x), static = "x")
    expect_equal(device(f("cpu:0")), nv_device("cpu:0"))
    expect_equal(device(f("cpu:1")), nv_device("cpu:1"))
  })

  it("converts constants with device specification to specified device", {
    f <- jit(function() nv_scalar(1), device = "cpu:1")
    expect_equal(device(f()), nv_device("cpu:1"))
  })

  it("device overrides found constant's device", {
    expect_equal(
      device(jit(\() nv_scalar(1, device = "cpu:1"), device = "cpu:0")()),
      nv_device("cpu:0")
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
    g <- jit(function() 1)
    expect_equal(device(g()), default_device())
  })
  it("uses specified device when input is R object", {
    g <- jit(function() 1, device = "cpu:1")
    expect_equal(device(g()), nv_device("cpu:1"))
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
  it("works with different device IDs for the pjrt backend", {
    f <- jit(identity)
    expect_equal(
      device(f(nv_scalar(1, device = "cpu:0"))),
      nv_device("cpu:0")
    )
    skip_if(!is_cuda())
    expect_equal(
      device(f(nv_scalar(1, device = "cuda"))),
      nv_device("cuda")
    )
  })

  it("works when passing device as static arg", {
    f <- function(x) nv_scalar(1, device = x)
    g <- jit(f, static = "x")
    expect_equal(device(g("cpu:0")), nv_device("cpu:0"))
    expect_equal(device(g("cpu:1")), nv_device("cpu:1"))
  })

  # device_arg
  it("device_arg reads the device from a static argument", {
    f <- jit(function(dev) nv_scalar(1, device = dev), device = device_arg("dev"))
    expect_equal(device(f("cpu:0")), nv_device("cpu:0"))
    expect_equal(device(f(nv_device("cpu:1"))), nv_device("cpu:1"))
    skip_if_no_quickr()
    with_backend("quickr", expect_equal(device(f(nv_device("cpu"))), nv_device("cpu")))
  })

  it("device_arg rejects a device of another backend", {
    skip_if_no_quickr()
    f <- jit(function(dev) 1L, device = device_arg("dev"))
    dev_q <- with_backend("quickr", nv_device("cpu"))
    expect_error(f(dev_q), "backend in force")
  })

  it("device_arg works next to further static arguments", {
    f <- jit(
      function(val, dev) nv_array(val, device = dev),
      device = device_arg("dev"),
      static = c("val", "dev")
    )
    dev0 <- nv_device("cpu")
    expect_true(device(f(1, dev0)) == dev0)
  })

  it("literal's device can be defined via device_arg", {
    f <- jit(function(dev) 1L, device = device_arg("dev"))
    expect_equal(device(f("cpu:0")), nv_device("cpu:0"))
    expect_equal(device(f("cpu:1")), nv_device("cpu:1"))
  })
})

test_that("cache hit when using PJRTDevice", {
  # this used to be a bug before pjrt 0.2.0, because every PJRTDevice was a new external pointer
  # and hashtab hashes address of xptr

  f <- jit(function(dev) nv_scalar(1, device = dev), static = "dev")
  dev0 <- nv_device("cpu")
  dev1 <- nv_device("cpu")
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
  expect_equal(device(g(nv_device("cpu"))), nv_device("cpu"))

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
