# The data types an R double and an R integer commit to when nothing else
# decides one are a property of the backend (`default_dtypes()`), overridable
# per backend with options. They decide only what a value becomes when nothing
# else does: the yielding rule of `vignette("type-promotion")` is untouched.

describe("default_dtypes()", {
  it("reports the registered defaults of the backend in force", {
    expect_equal(default_dtypes(), list(float = as_dtype("f32"), int = as_dtype("i32")))
    expect_equal(with_backend("quickr", default_dtypes()), list(float = as_dtype("f64"), int = as_dtype("i32")))
    expect_error(with_backend("plain", default_dtypes()), "no default data types")
  })

  it("is overridden by the options, on every backend", {
    withr::local_options(anvl.default_float = "f64", anvl.default_int = "i64")
    expect_equal(default_dtypes(), list(float = as_dtype("f64"), int = as_dtype("i64")))
    expect_equal(with_backend("quickr", default_dtypes()), list(float = as_dtype("f64"), int = as_dtype("i64")))
  })

  it("validates an option when it is read", {
    withr::local_options(anvl.default_float = "nope")
    expect_error(default_dtypes(), "anvl.default_float")
    expect_error(nv_array(1.5), "anvl.default_float")
    withr::local_options(anvl.default_float = "f16")
    expect_error(default_dtypes(), "must be one of")
  })
})

describe("local_default_dtypes()", {
  it("sets the options for the scope", {
    local({
      local_default_dtypes(c(float = "f64", int = "i64"))
      expect_identical(getOption("anvl.default_float"), "f64")
      expect_identical(getOption("anvl.default_int"), "i64")
      expect_equal(default_dtypes(), list(float = as_dtype("f64"), int = as_dtype("i64")))
    })
    expect_null(getOption("anvl.default_float"))
    expect_equal(default_dtypes()$float, as_dtype("f32"))
  })

  it("accepts a DataType and leaves an unnamed category alone", {
    local_default_dtypes(c(int = "i64"))
    local_default_dtypes(list(float = as_dtype("f64")))
    expect_equal(default_dtypes(), list(float = as_dtype("f64"), int = as_dtype("i64")))
  })

  it("validates the value", {
    expect_error(local_default_dtypes(c(float = "i32")), "float.*must be one of")
    expect_error(local_default_dtypes(c(float = "f16")), "must be one of")
    expect_error(local_default_dtypes(c(float = "nope")), "must be one of")
    expect_error(local_default_dtypes(c(int = "f32")), "int.*must be one of")
    expect_error(local_default_dtypes(c(int = "i8")), "must be one of")
    expect_error(local_default_dtypes(c(int = "ui32")), "must be one of")
    expect_error(local_default_dtypes("f64"), "named list or character")
    expect_error(local_default_dtypes(c(double = "f64")), "named list or character")
    expect_error(local_default_dtypes(c(float = "f64", float = "f32")), "named list or character")
  })
})

describe("with_default_dtypes()", {
  it("scopes the change to the expression", {
    expect_equal(with_default_dtypes(c(float = "f64"), default_dtypes()$float), as_dtype("f64"))
    expect_equal(default_dtypes()$float, as_dtype("f32"))
    expect_equal(with_default_dtypes(c(int = "i64"), dtype(nv_array(1L))), as_dtype("i64"))
  })
})

describe("the default float", {
  it("decides what an R double is built at eagerly", {
    local_default_dtypes(c(float = "f64"))
    expect_equal(dtype(nv_array(1.5)), as_dtype("f64"))
    expect_equal(dtype(nv_scalar(1.5)), as_dtype("f64"))
    expect_equal(dtype(nv_array(matrix(c(1.5, 2.5, 3.5, 4.5), 2))), as_dtype("f64"))
    expect_equal(dtype(nv_fill(0, 3)), as_dtype("f64"))
    expect_equal(dtype(nv_seq(0, 1, steps = 3)), as_dtype("f64"))
    expect_equal(dtype(nv_eye(2)), as_dtype("f64"))
    state <- nv_rng_state(1L)
    expect_equal(dtype(nv_rnorm(3, state)[[2L]]), as_dtype("f64"))
    expect_equal(dtype(nv_runif(3, state)[[2L]]), as_dtype("f64"))
    expect_equal(peek_dtype(1.5), as_dtype("f64"))
    expect_error(dtype(1.5), "f64")
    # An explicit dtype still wins, and the other categories are untouched.
    expect_equal(dtype(nv_array(1.5, dtype = "f32")), as_dtype("f32"))
    expect_equal(dtype(nv_array(1L)), as_dtype("i32"))
    expect_equal(dtype(nv_array(TRUE)), as_dtype("bool"))
  })

  it("leaves data that is not an R value alone", {
    local_default_dtypes(c(float = "f64", int = "i64"))
    # A buffer already has its dtype; nv_minval() builds one from raw bytes.
    expect_equal(dtype(nv_scalar(pjrt::pjrt_scalar(1L, dtype = "i32"))), as_dtype("i32"))
    expect_equal(as.integer(nv_reduce_max(nv_array(1:3, dtype = "i32"))), 3L)
    expect_equal(as.integer(jit(function(x) nv_reduce_min(x))(nv_array(1:3, dtype = "i32"))), 1L)
  })

  it("decides what an R double commits to in a trace", {
    local_default_dtypes(c(float = "f64"))
    expect_equal(dtype(jit(function() 1.5)()), as_dtype("f64"))
    # An R argument is uploaded at the default.
    expect_equal(dtype(jit(function(x) x)(1.5)), as_dtype("f64"))
    # Constants built inside the trace as well.
    expect_equal(dtype(jit(function() nv_array(c(1, 2)))()), as_dtype("f64"))
    expect_equal(dtype(jit(function() nv_fill(0, 2))()), as_dtype("f64"))
    # And the all-R-values branch of promotion.
    expect_equal(dtype(jit(function(x, y) x + y)(1, 2)), as_dtype("f64"))
  })

  it("does not change the yielding rule", {
    local_default_dtypes(c(float = "f64"))
    expect_equal(dtype(nv_array(1, dtype = "f32") + 1.5), as_dtype("f32"))
    expect_equal(dtype(jit(function(x) x * 2)(nv_array(1, dtype = "f32"))), as_dtype("f32"))
    # Crossing a category takes the *default* of the other category.
    expect_equal(dtype(nv_array(1L, dtype = "i32") + 1.5), as_dtype("f64"))
  })

  it("keeps an R value exact", {
    local_default_dtypes(c(float = "f64"))
    expect_identical(as_array(jit(function(x) x / sqrt(2))(1)), 1 / sqrt(2))
    expect_identical(as_array(nv_scalar(1) / sqrt(2)), 1 / sqrt(2))
  })
})

describe("the default integer", {
  it("decides what an R integer is built at eagerly", {
    local_default_dtypes(c(int = "i64"))
    expect_equal(dtype(nv_array(1L)), as_dtype("i64"))
    expect_equal(dtype(nv_scalar(1L)), as_dtype("i64"))
    expect_equal(dtype(nv_seq(1, 3)), as_dtype("i64"))
    expect_equal(dtype(nv_fill(0L, 3)), as_dtype("i64"))
    state <- nv_rng_state(1L)
    expect_equal(dtype(nv_rbinom(3, state)[[2L]]), as_dtype("i64"))
    expect_equal(dtype(nv_sample_int(3, state, 6L)[[2L]]), as_dtype("i64"))
    expect_equal(peek_dtype(1L), as_dtype("i64"))
    expect_equal(dtype(nv_array(1.5)), as_dtype("f32"))
  })

  it("decides what an R integer commits to in a trace", {
    local_default_dtypes(c(int = "i64"))
    expect_equal(dtype(jit(function() 1L)()), as_dtype("i64"))
    expect_equal(dtype(jit(function(x) x)(1L)), as_dtype("i64"))
    expect_equal(dtype(jit(function() nv_seq(1, 3))()), as_dtype("i64"))
  })

  it("does not change the yielding rule", {
    local_default_dtypes(c(int = "i64"))
    expect_equal(dtype(nv_array(1L, dtype = "i32") + 1L), as_dtype("i32"))
    expect_equal(dtype(nv_array(1L, dtype = "i8") * 2L), as_dtype("i8"))
    expect_equal(dtype(nv_array(TRUE) + 1L), as_dtype("i64"))
  })
})

describe("a compiled program", {
  it("is keyed on the defaults it was compiled under", {
    n_traced <- 0L
    f <- jit(function(x) {
      n_traced <<- n_traced + 1L
      x + 1.5
    })
    x <- nv_array(1L, dtype = "i32") # the literal decides the float
    expect_equal(dtype(f(x)), as_dtype("f32"))
    expect_equal(n_traced, 1L)
    expect_equal(dtype(f(x)), as_dtype("f32"))
    expect_equal(n_traced, 1L)
    with_default_dtypes(c(float = "f64"), expect_equal(dtype(f(x)), as_dtype("f64")))
    expect_equal(n_traced, 2L)
    # Back to f32: the first entry is served, not the f64 one.
    expect_equal(dtype(f(x)), as_dtype("f32"))
    expect_equal(n_traced, 2L)
    with_default_dtypes(c(float = "f64"), expect_equal(dtype(f(x)), as_dtype("f64")))
    expect_equal(n_traced, 2L)

    # A literal-only program as well.
    g <- jit(function() 1.5)
    expect_equal(dtype(g()), as_dtype("f32"))
    with_default_dtypes(c(float = "f64"), expect_equal(dtype(g()), as_dtype("f64")))
    expect_equal(dtype(g()), as_dtype("f32"))
  })

  it("runs on, and is pinned to, the backend in force when it is called", {
    skip_if_no_quickr()
    f <- jit(function() 1.5)
    expect_equal(dtype(f()), as_dtype("f32"))
    expect_equal(with_backend("quickr", dtype(f())), as_dtype("f64"))
    expect_equal(dtype(f()), as_dtype("f32"))
    # An override applies on every backend.
    local_default_dtypes(c(int = "i64"))
    g <- jit(function() 1L)
    expect_equal(dtype(g()), as_dtype("i64"))
    expect_equal(with_backend("quickr", dtype(nv_array(1L))), as_dtype("i64"))
  })

  it("keeps one cache per backend", {
    skip_if_no_quickr()
    n_traced <- 0L
    f <- jit(function(x) {
      n_traced <<- n_traced + 1L
      x + 1.5
    })
    expect_equal(dtype(f(nv_array(1L))), as_dtype("f32"))
    expect_equal(with_backend("quickr", dtype(f(nv_array(1L)))), as_dtype("f64"))
    expect_equal(n_traced, 2L)
    expect_equal(dtype(f(nv_array(1L))), as_dtype("f32"))
    expect_equal(with_backend("quickr", dtype(f(nv_array(1L)))), as_dtype("f64"))
    expect_equal(n_traced, 2L)
  })
})

describe("eager code", {
  it("reads the same default the operation runs with", {
    skip_if_no_quickr()
    # A plain R helper decides a promotion eagerly, between dispatches. The
    # default it reads is the one of the backend in force, which is also the
    # backend the operation then runs on.
    promote <- function(x) as_anvl_arrays(x, 1.5, .promote = promote_common())[[2L]]
    expect_equal(dtype(promote(nv_array(1L, dtype = "i32"))), as_dtype("f32"))
    with_backend("quickr", {
      expect_equal(dtype(promote(nv_array(1L, dtype = "i32"))), as_dtype("f64"))
      expect_equal(peek_dtype(1.5), as_dtype("f64"))
      expect_equal(dtype(nv_fill(0, 3)), as_dtype("f64"))
    })
  })

  it("rejects an array of another backend instead of guessing a default", {
    skip_if_no_quickr()
    x <- with_backend("quickr", nv_array(1L))
    expect_error(x + 1.5, "quickr")
    expect_error(nv_fill_like(x, 0), "backend in force")
  })
})

describe("the quickr backend", {
  it("keys a compiled program on the defaults as well", {
    skip_if_no_quickr()
    local_backend("quickr")
    n_traced <- 0L
    f <- jit(function(x) {
      n_traced <<- n_traced + 1L
      x + 1.5
    })
    x <- nv_array(1L)
    expect_equal(dtype(f(x)), as_dtype("f64"))
    with_default_dtypes(c(float = "f32"), expect_equal(dtype(f(x)), as_dtype("f32")))
    expect_equal(n_traced, 2L)
    expect_equal(dtype(f(x)), as_dtype("f64"))
    expect_equal(n_traced, 2L)
  })

  it("commits an R double to f64 everywhere", {
    skip_if_no_quickr()
    local_backend("quickr")
    expect_equal(dtype(nv_array(1.5)), as_dtype("f64"))
    expect_equal(peek_dtype(1.5), as_dtype("f64"))
    expect_equal(dtype(jit(function() 1.5)()), as_dtype("f64"))
    expect_equal(dtype(jit(function(x) x + 1.5)(nv_array(1L))), as_dtype("f64"))
    expect_equal(dtype(jit(function(x) x)(1.5)), as_dtype("f64"))
    expect_equal(dtype(nv_array(1L)), as_dtype("i32"))
  })
})
