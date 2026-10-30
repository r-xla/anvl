describe("r_device()", {
  it("is the only device of the r backend", {
    dev <- r_device()
    expect_s3_class(dev, "RDevice")
    expect_true(is_device(dev))
    expect_equal(backend(dev), "r")
    expect_equal(format(dev), "RDevice(cpu)")
    expect_true(dev == r_device("cpu"))
    expect_error(r_device("cuda"), "cpu")
  })

  it("is not equal to the device of the quickr backend", {
    expect_false(r_device() == quickr_device())
    expect_false(eq_device(r_device(), quickr_device()))
  })
})

describe("the r backend", {
  it("holds an array as an R array", {
    local_backend("r")
    x <- nv_array(c(1, 2, 3))
    expect_equal(backend(x), "r")
    expect_dtype(x, "f64")
    expect_equal(device(x), r_device())
    expect_equal(platform(x), "cpu")
    expect_identical(as_array(x), array(c(1, 2, 3), dim = 3L))
    expect_identical(as_array(nv_empty(c(2L, 0L), dtype = "i32")), array(integer(), dim = c(2L, 0L)))
    expect_equal(backend(nv_array(1L, device = "cpu")), "r")
  })

  it("materializes R values at f64 and i32", {
    local_backend("r")
    expect_equal(default_dtypes(), list(float = as_dtype("f64"), int = as_dtype("i32")))
    expect_dtype(nv_array(1.5), "f64")
    expect_dtype(nv_array(1L), "i32")
    expect_dtype(jit(function(x) x + 1.5)(nv_array(1L)), "f64")
  })

  it("rejects a data type an R array cannot hold", {
    local_backend("r")
    expect_error(nv_array(1, dtype = "f32"), "\"r\" and \"quickr\" backends")
    expect_error(jit(function(x) nv_convert(x, "f32"))(nv_array(1)), "f32")
  })

  it("runs operations eagerly", {
    local_backend("r")
    x <- nv_array(c(1, 2, 3))
    out <- x * 2 + 1
    expect_equal(backend(out), "r")
    expect_equal(as_array(out), array(c(3, 5, 7), dim = 3L))
    expect_equal(as_array(sum(x)), 6)
  })

  it("runs a jitted function and returns arrays of the r backend", {
    local_backend("r")
    f <- jit(function(x, y) x %*% y)
    x <- nv_matrix(as.numeric(1:6), nrow = 2L)
    y <- nv_matrix(as.numeric(1:6), nrow = 3L)
    out <- f(x, y)
    expect_equal(backend(out), "r")
    expect_equal(as_array(out), matrix(as.numeric(1:6), nrow = 2L) %*% matrix(as.numeric(1:6), nrow = 3L))
    f(x, y)
    expect_identical(jit_cache_size(f), 1L)
  })

  it("preserves a nested output structure", {
    local_backend("r")
    f <- jit(function(x) list(flags = x > 1L, payload = list(shifted = x + 1L)))
    out <- f(nv_array(1:3))
    expect_dtype(out$flags, "bool")
    expect_dtype(out$payload$shifted, "i32")
    expect_identical(as_array(out$flags), array(c(FALSE, TRUE, TRUE), dim = 3L))
    expect_identical(as_array(out$payload$shifted), array(2:4, dim = 3L))
  })

  it("returns plain R arrays with unwrap = TRUE", {
    local_backend("r")
    f <- jit(function(x) list(a = x + 1L, b = sum(x)), unwrap = TRUE)
    out <- f(nv_array(1:3))
    expect_identical(out$a, array(2:4, dim = 3L))
    expect_identical(out$b, 6L)
  })

  it("runs on its device and rejects one of another backend", {
    local_backend("r")
    expect_equal(as_array(jit(function(x) x + 1, device = "cpu")(nv_array(1))), array(2))
    expect_equal(as_array(jit(function(x) x + 1, device = r_device())(nv_array(1))), array(2))
    expect_error(
      jit(function(x) x + 1, device = quickr_device())(nv_array(1)),
      "belongs to the .*quickr.* backend"
    )
  })

  it("is the only backend its arrays can run on", {
    x <- with_backend("r", nv_array(1))
    expect_error(jit(function(x) x)(x), "\"r\"")
    expect_error(with_backend("quickr", jit(function(x) x)(x)), "\"r\"")
  })

  it("names itself when it does not support a primitive", {
    local_backend("r")
    expect_error(
      jit(function(x) nv_popcnt(x))(nv_array(1L)),
      "The \"r\" backend does not support these primitives: popcnt"
    )
  })

  it("does not need quickr", {
    local_mocked_bindings(quickr_is_installed = function() FALSE)
    f <- jit(function(x) x * 2)
    expect_equal(with_backend("r", as_array(f(nv_array(1)))), array(2))
    expect_error(with_backend("quickr", f(nv_array(1))), "quickr")
  })
})
