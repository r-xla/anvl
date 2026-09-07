# The data types an R double and an R integer commit to when nothing else
# decides one are registered per backend (`default_dtypes()`) and overridden on
# every backend by two global options. They decide only what a value becomes
# when nothing else does: the yielding rule of `vignette("type-promotion")` is
# untouched.

describe("default_dtypes()", {
  it("reports the registered defaults of the backend in force", {
    local_registered_default_dtypes()
    expect_equal(default_dtypes(), list(float = as_dtype("f32"), int = as_dtype("i32")))
    expect_equal(with_backend("quickr", default_dtypes()), list(float = as_dtype("f64"), int = as_dtype("i32")))
    expect_error(with_backend("plain", default_dtypes()), "names no usable backend")
  })

  it("is overridden by the options, on every backend", {
    withr::local_options(anvl.default_dtypes = c(float = "f64", int = "i64"))
    expect_equal(default_dtypes(), list(float = as_dtype("f64"), int = as_dtype("i64")))
    expect_equal(with_backend("quickr", default_dtypes()), list(float = as_dtype("f64"), int = as_dtype("i64")))
  })

  it("validates an option when it is read", {
    withr::local_options(anvl.default_dtypes = c(float = "nope"))
    expect_error(default_dtypes(), "anvl.default_dtypes")
    expect_error(nv_array(1.5), "anvl.default_dtypes")
    withr::local_options(anvl.default_dtypes = c(float = "f16"))
    expect_error(default_dtypes(), "must be one of")
    withr::local_options(anvl.default_dtypes = "f64")
    expect_error(default_dtypes(), "named list or character vector")
  })
})

describe("local_default_dtypes()", {
  it("sets the options for the scope", {
    local_registered_default_dtypes()
    local({
      local_default_dtypes(c(float = "f64", int = "i64"))
      expect_identical(getOption("anvl.default_dtypes"), c(float = "f64", int = "i64"))
      expect_equal(default_dtypes(), list(float = as_dtype("f64"), int = as_dtype("i64")))
    })
    expect_null(getOption("anvl.default_dtypes"))
    expect_equal(default_dtypes()$float, as_dtype("f32"))
  })

  it("accepts a DataType and leaves an unnamed category alone", {
    # The setters merge, so naming one category does not disturb the other --
    # both in sequence and nested.
    local_default_dtypes(c(int = "i64"))
    local_default_dtypes(list(float = as_dtype("f64")))
    expect_equal(default_dtypes(), list(float = as_dtype("f64"), int = as_dtype("i64")))
    expect_equal(
      with_default_dtypes(c(float = "f32"), default_dtypes()),
      list(float = as_dtype("f32"), int = as_dtype("i64"))
    )
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
    expect_error(local_default_dtypes(c(float = "f64", float = "f32")), "more than once")
  })
})

describe("with_default_dtypes()", {
  it("scopes the change to the expression", {
    before <- default_float()
    expect_equal(with_default_dtypes(c(float = "f64"), default_float()), as_dtype("f64"))
    expect_equal(default_float(), before)
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
    expect_equal(dtype(nv_array(1L)), default_int())
    expect_equal(dtype(nv_array(TRUE)), as_dtype("bool"))
  })

  it("does not warn about staging through a narrower data type", {
    # Staging is only worth a warning when it widens past the data type the
    # value would have taken anyway. An R integer stages through `i32`, which
    # under an `i64` default is narrower than its own default.
    local_default_dtypes(c(int = "i64"))
    expect_no_warning(nv_array(1L, dtype = "i8") * 2L)
    expect_no_warning(jit(function(x) nv_convert(x, "f64"))(1L))
    # An R double staged through `f64` under an `f32` default still warns.
    with_default_dtypes(
      c(float = "f32"),
      expect_warning(nv_convert(1.5, "i32"), class = "anvl_staging_widens_warning")
    )
    local_default_dtypes(c(float = "f64"))
    expect_no_warning(nv_convert(1.5, "i32"))
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
    expect_equal(dtype(nv_array(1.5)), default_float())
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

  it("decides the data type of the indices an operation returns", {
    x <- nv_array(c(3, 1, 4, 1, 5))
    local_default_dtypes(c(int = "i64"))
    i64 <- as_dtype("i64")
    expect_equal(dtype(nv_argmax(x)), i64)
    expect_equal(dtype(nv_argmin(x)), i64)
    expect_equal(dtype(nv_argsort(x)), i64)
    expect_equal(dtype(nv_cummax(x, with_indices = TRUE)$indices), i64)
    expect_equal(dtype(nv_cummin(x, with_indices = TRUE)$indices), i64)
    # `hlo_top_k` fixes its indices at i32, so these are converted.
    expect_equal(dtype(nv_top_k(x, k = 2L, with_indices = TRUE)$indices), i64)
    # And in a trace, where the program is keyed on the defaults.
    expect_equal(dtype(jit(function(x) nv_argmax(x))(x)), i64)
    expect_equal(dtype(jit(function(x) nv_argsort(x))(x)), i64)
    expect_equal(dtype(jit(function(x) nv_cummin(x, with_indices = TRUE)$indices)(x)), i64)
    expect_equal(dtype(jit(function(x) nv_top_k(x, k = 2L, with_indices = TRUE)$indices)(x)), i64)
  })

  it("does not change the indices themselves", {
    x <- nv_array(c(3, 1, 4, 1, 5))
    at_i32 <- list(
      argmax = as_array(nv_argmax(x)),
      argsort = as_array(nv_argsort(x)),
      cummax = as_array(nv_cummax(x, with_indices = TRUE)$indices),
      top_k = as_array(nv_top_k(x, k = 2L, with_indices = TRUE)$indices)
    )
    local_default_dtypes(c(int = "i64"))
    expect_equal(as_array(nv_argmax(x)), at_i32$argmax)
    expect_equal(as_array(nv_argsort(x)), at_i32$argsort)
    expect_equal(as_array(nv_cummax(x, with_indices = TRUE)$indices), at_i32$cummax)
    expect_equal(as_array(nv_top_k(x, k = 2L, with_indices = TRUE)$indices), at_i32$top_k)
  })

  it("does not disturb a gradient that scatters through those indices", {
    # `prim_top_k` and `prim_cummax` route their reverse rule through
    # `prim_scatter` with the forward indices, so a wider index data type has
    # to survive the scatter.
    x <- nv_array(c(3, 1, 4, 1, 5, 9))
    f <- function(x) nv_reduce_sum(nv_top_k(x, k = 3L))
    g <- function(x) nv_reduce_sum(nv_cummax(x))
    at_i32 <- list(
      top_k = as_array(jit(gradient(f))(x)[[1L]]),
      cummax = as_array(jit(gradient(g))(x)[[1L]])
    )
    local_default_dtypes(c(int = "i64"))
    expect_equal(as_array(jit(gradient(f))(x)[[1L]]), at_i32$top_k)
    expect_equal(as_array(jit(gradient(g))(x)[[1L]]), at_i32$cummax)
  })

  it("decides the data type of an LU decomposition's pivots", {
    # LAPACK's getrf writes 32-bit pivots, so `pivots` and `permutation` are
    # converted after the custom call rather than produced at the default.
    a <- nv_matrix(c(4, 3, 6, 3, 2, 8, 1, 5, 7), nrow = 3, dtype = "f64")
    at_i32 <- lapply(nv_lu(a)[c("pivots", "permutation")], as_array)
    local_default_dtypes(c(int = "i64"))
    factored <- nv_lu(a)
    expect_equal(dtype(factored$pivots), as_dtype("i64"))
    expect_equal(dtype(factored$permutation), as_dtype("i64"))
    expect_equal(as_array(factored$pivots), at_i32$pivots)
    expect_equal(as_array(factored$permutation), at_i32$permutation)
  })
})

describe("a compiled program", {
  it("is keyed on the defaults it was compiled under", {
    # The baseline has to differ from the override applied below, or there is
    # only ever one program and nothing to key.
    local_registered_default_dtypes()
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
    local_registered_default_dtypes()
    f <- jit(function() 1.5)
    expect_equal(dtype(f()), as_dtype("f32"))
    expect_equal(with_backend("quickr", dtype(f())), as_dtype("f64"))
    expect_equal(dtype(f()), as_dtype("f32"))
    # An override applies on every backend that can represent it.
    local_default_dtypes(c(int = "i64"))
    g <- jit(function() 1L)
    expect_equal(dtype(g()), as_dtype("i64"))
    # quickr has no `i64`, so there it is an error rather than a silent
    # fallback to something the backend does support.
    expect_error(with_backend("quickr", nv_array(1L)), "quickr")
  })

  it("resolves a constant that names a device like any other", {
    # A constant that names a device takes the eager path, but it is still part
    # of the trace and must read the defaults the rest of the trace reads --
    # the baseline, and an override over it.
    local_default_dtypes(c(float = "f64"))
    f <- jit(function() {
      dev <- nv_device("cpu")
      list(
        plain = nv_array(1.5),
        with_device = nv_array(1.5, device = dev),
        scoped_plain = with_default_dtypes(c(float = "f32"), nv_array(1.5)),
        scoped_device = with_default_dtypes(c(float = "f32"), nv_array(1.5, device = dev))
      )
    })
    out <- f()
    expect_equal(dtype(out$plain), as_dtype("f64"))
    expect_equal(dtype(out$with_device), as_dtype("f64"))
    expect_equal(dtype(out$scoped_plain), as_dtype("f32"))
    expect_equal(dtype(out$scoped_device), as_dtype("f32"))
  })

  it("keeps one cache per backend", {
    skip_if_no_quickr()
    local_registered_default_dtypes()
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
    expect_equal(dtype(promote(nv_array(1L, dtype = "i32"))), default_float())
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

describe("a scoped override inside a jitted body", {
  it("applies to the values built in its scope, and only there", {
    # A trace's outputs are arrays, so the data types are recorded as a side
    # effect of tracing rather than returned.
    seen <- character()
    note <- function(x) {
      seen <<- c(seen, as.character(dtype(x)))
      x
    }
    f <- jit(function(x) {
      with_default_dtypes(c(float = "f64"), {
        note(nv_array(1.5))
        note(nv_fill(0, 3))
        note(x + 1.5)
        note(nv_eye(2))
        note(nv_seq(0, 1, steps = 3))
      })
      note(nv_array(1.5))
      x
    })
    invisible(f(nv_array(1L, dtype = "i32")))
    expect_equal(seen, c(rep("f64", 5L), as.character(default_float())))
  })

  it("reaches a helper that builds its own literals", {
    helper <- function(x) x * 2 + 0.5
    f <- jit(function(x) list(lo = helper(x), hi = with_default_dtypes(c(float = "f64"), helper(x))))
    out <- f(nv_array(1L, dtype = "i32"))
    expect_equal(dtype(out$lo), default_float())
    expect_equal(dtype(out$hi), as_dtype("f64"))
  })

  it("does not reach a bare R value handed out of the scope", {
    # The value has committed to nothing inside the scope, so it takes the
    # default where it is used -- the per-operation rule, not a special case.
    expect_equal(dtype(jit(function() with_default_dtypes(c(float = "f64"), 1.5))()), default_float())
  })

  it("takes the trace's baseline, not the backend in force", {
    skip_if_no_quickr()
    # A program is compiled for one backend, so switching inside the body
    # cannot change what its R values commit to.
    expect_equal(dtype(jit(function() with_backend("quickr", nv_array(1.5)))()), default_float())
  })

  it("does not change what the program is keyed on", {
    local_registered_default_dtypes()
    n_traced <- 0L
    f <- jit(function(x) {
      n_traced <<- n_traced + 1L
      with_default_dtypes(c(float = "f64"), x + 1.5)
    })
    x <- nv_array(1L, dtype = "i32")
    expect_equal(dtype(f(x)), as_dtype("f64"))
    expect_equal(n_traced, 1L)
    # The scoped region is `f64` either way, but the baseline still keys the
    # cache, so a different ambient default is a different program.
    with_default_dtypes(c(float = "f64"), expect_equal(dtype(f(x)), as_dtype("f64")))
    expect_equal(n_traced, 2L)
    expect_equal(dtype(f(x)), as_dtype("f64"))
    expect_equal(n_traced, 2L)
  })
})

describe("the quickr backend", {
  it("rejects a float default it cannot represent", {
    skip_if_no_quickr()
    local_backend("quickr")
    # quickr has no single precision, so `f32` cannot be honoured; it is an
    # error rather than a double labelled `f32`, which is the mislabelling the
    # data type system exists to prevent.
    with_default_dtypes(c(float = "f32"), {
      expect_error(nv_array(1.5), "quickr")
      # A constant of a trace is captured backend-agnostically, so this one is
      # only caught where the program is lowered.
      expect_error(jit(function() nv_array(1.5))(), "quickr")
    })
    expect_error(nv_array(1.5, dtype = "f32"), "quickr")
    expect_error(jit(function(x) nv_convert(x, "f32"))(nv_array(1, dtype = "f64")), "quickr")
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
