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

test_that("jit returns R literals and arrays as ambiguous AnvlArrays", {
  expect_equal(jit(\() 1L)(), nv_scalar(1L, ambiguous = TRUE))
  expect_equal(jit(\() array(1L))(), nv_array(1L, shape = 1L, ambiguous = TRUE))
  expect_equal(
    jit(\() array(c(1L, 2L, 3L)))(),
    nv_array(c(1L, 2L, 3L), ambiguous = TRUE)
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
    local_backend("xla")
    f <- jit(identity)
    expect_equal(backend(f), "xla")
    expect_equal(backend(f(1)), "xla")
  })

  it("backend = NULL, device = NULL follows default_backend() = 'quickr'", {
    skip_if_no_quickr()
    local_backend("quickr")
    f <- jit(identity)
    expect_equal(backend(f), "quickr")
    expect_equal(backend(f(1)), "quickr")
  })

  it("backend = 'xla', device = NULL uses xla", {
    local_backend("quickr")
    f <- jit(identity, backend = "xla")
    expect_equal(backend(f), "xla")
    expect_equal(backend(f(1)), "xla")
  })

  it("concrete device string", {
    f <- jit(identity, device = "cpu")
    expect_equal(backend(f), "xla")
    expect_equal(backend(f(1)), "xla")
  })

  it("concrete device object", {
    f <- jit(identity, device = pjrt::pjrt_device("cpu"))
    expect_equal(backend(f), "xla")
    expect_equal(backend(f(1)), "xla")
  })

  it("concrete device infers backend from device", {
    skip_if_no_quickr()
    local_backend("quickr")
    f <- jit(identity, device = pjrt::pjrt_device("cpu"))
    expect_equal(backend(f), "xla")
    expect_equal(backend(f(1)), "xla")
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
    expect_equal(device(f("cpu:0")), nv_device("cpu:0", "xla"))
    expect_equal(device(f("cpu:1")), nv_device("cpu:1", "xla"))
  })

  it("backend 'auto' works with xla and quickr input", {
    f <- jit(identity, backend = "auto")
    expect_equal(backend(f), "auto")
    # At call time, backend is picked from the input.
    expect_equal(backend(f(nv_scalar(1, backend = "xla"))), "xla")
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
      f(nv_scalar(1, backend = "xla"), nv_scalar(2, backend = "quickr")),
      "multiple backends"
    )
  })

  it("cannot mix backends via closed-over constant", {
    # A closed-over constant from a different backend than the call-time input
    # must not silently compile on either backend.
    skip_if_no_quickr()
    const_q <- nv_scalar(1, backend = "quickr")
    f <- jit(function(x) x + const_q, backend = "xla")
    expect_error(
      f(nv_scalar(1, backend = "xla")),
      "Cannot compile a \"xla\" program"
    )
    const_x <- nv_scalar(1, backend = "xla")
    g <- jit(function(x) x + const_x, backend = "quickr")
    expect_error(
      g(nv_scalar(1, backend = "quickr")),
      "Cannot compile a \"quickr\" program"
    )
  })

  it("character device with backend = 'auto' is honored per chosen backend", {
    skip_if_no_quickr()
    f <- jit(identity, device = "cpu", backend = "auto")
    expect_equal(device(f(nv_scalar(1, backend = "xla"))), nv_device("cpu", "xla"))
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
    out_x <- f(nv_device("cpu", "xla"))
    expect_equal(backend(out_q), "quickr")
    expect_equal(backend(out_x), "xla")
  })

  it("converts constants with device specification to specified device", {
    f <- jit(function() nv_scalar(1), device = "cpu:1")
    expect_equal(device(f()), nv_device("cpu:1", "xla"))
  })

  it("device overrides found constant's device", {
    expect_equal(
      device(jit(\() nv_scalar(1, device = "cpu:1"), device = "cpu:0")()),
      nv_device("cpu:0", "xla")
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
    g <- jit(function() 1, backend = "xla")
    expect_equal(device(g()), default_device("xla"))
  })
  it("uses specified device when input is R object", {
    g <- jit(function() 1, device = "cpu:1")
    expect_equal(device(g()), nv_device("cpu:1", "xla"))
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
  it("works with different device IDs for 'xla' backend", {
    f <- jit(identity)
    expect_equal(
      device(f(nv_scalar(1, device = "cpu:0"))),
      nv_device("cpu:0", "xla")
    )
    skip_if(!is_cuda())
    expect_equal(
      device(f(nv_scalar(1, device = "cuda"))),
      nv_device("cuda", "xla")
    )
  })
  it("works with different devices with 'auto' backend", {
    f <- jit(nv_log, backend = "auto")
    expect_equal(
      device(f(nv_scalar(1, device = "cpu"))),
      nv_device("cpu", "xla")
    )
    skip_if(!is_cuda())
    expect_equal(
      device(f(nv_scalar(1, device = "cuda"))),
      nv_device("cuda", "xla")
    )
  })

  it("works when passing device as static arg for concrete backend", {
    f <- function(x) nv_scalar(1, device = x)
    g <- jit(f, backend = "xla", static = "x")
    expect_equal(device(g("cpu:0")), nv_device("cpu:0", "xla"))
    expect_equal(device(g("cpu:1")), nv_device("cpu:1", "xla"))
  })

  # device_arg
  it("device-arg works with concrete 'xla' backend", {
    local_backend("xla")
    f <- function(dev) nv_scalar(1, device = dev)
    expect_error(
      g <- jit(f, device = device_arg("dev"), backend = "xla"),
      "is only allowed"
    )
  })

  it("uses default backend when device_arg is character(1)", {
    local_backend("xla")
    f <- function(x) nv_scalar(1, device = x)
    g <- jit(f, device = device_arg("x"), backend = "auto")
    expect_equal(device(g("cpu:0")), nv_device("cpu:0", "xla"))
    expect_equal(device(g("cpu:1")), nv_device("cpu:1", "xla"))
    skip_if_no_quickr()
    expect_equal(device(g(nv_device("cpu", "quickr"))), nv_device("cpu", "quickr"))
  })

  it("device_arg works with backend = NULL (uses default backend)", {
    local_backend("xla")
    f <- jit(function(dev) 1L, device = device_arg("dev"), backend = NULL)
    expect_equal(device(f(nv_device("cpu:1", "xla"))), nv_device("cpu:1", "xla"))
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
    dev0 <- nv_device("cpu", "xla")
    expect_true(device(f(1, dev0)) == dev0)
    skip_if_no_quickr()
    dev1 <- nv_device("cpu", "quickr")
    expect_true(device(f(1, dev1)) == dev1)
  })

  it("literal's device can be defined via device_arg", {
    f <- jit(function(dev) 1L, device = device_arg("dev"))
    expect_equal(device(f("cpu:0")), nv_device("cpu:0", "xla"))
    expect_equal(device(f("cpu:1")), nv_device("cpu:1", "xla"))
  })
})

test_that("cache hit when using PJRTDevice", {
  # this used to be a bug before pjrt 0.2.0, because every PJRTDevice was a new external pointer
  # and hashtab hashes address of xptr

  # we don't need device_arg() as it is only for making jit backend-agnostic

  f <- jit(function(dev) nv_scalar(1, device = dev), static = "dev")
  dev0 <- nv_device("cpu", "xla")
  dev1 <- nv_device("cpu", "xla")
  f(dev = dev0)
  f(dev = dev1)
  expect_equal(cache_size(f), 1L)
})
