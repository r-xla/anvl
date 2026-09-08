# The data types an R double and an R integer commit to when nothing else
# decides one are registered per backend (`default_dtypes()`) and overridden by
# the `anvl.default_dtypes` option, for every backend or per backend. They
# decide only what a value becomes
# when nothing else does: the yielding rule of `vignette("type-promotion")` is
# untouched.

describe("default_dtypes()", {
  it("reports the registered defaults of the backend in force", {
    local_registered_default_dtypes()
    expect_equal(default_dtypes(), list(float = as_dtype("f32"), int = as_dtype("i32")))
    expect_equal(with_backend("quickr", default_dtypes()), list(float = as_dtype("f64"), int = as_dtype("i32")))
    # `active_backend()` takes the option as given, so a backend that is not
    # registered is reported where its defaults are read.
    expect_error(with_backend("plain", default_dtypes()), "no default data types")
  })

  it("is overridden by an option value that names no backend, on every backend", {
    withr::local_options(anvl.default_dtypes = c(float = "f64", int = "i64"))
    expect_equal(default_dtypes(), list(float = as_dtype("f64"), int = as_dtype("i64")))
    expect_equal(with_backend("quickr", default_dtypes()), list(float = as_dtype("f64"), int = as_dtype("i64")))
  })

  it("takes an option value as it is, reading the names it knows", {
    # Nothing about the option is validated. A value or a name anvl cannot
    # read is simply not a default, and a data type it can read is used as
    # named -- one that does not fit fails where the data is allocated or the
    # program compiled.
    registered <- list(float = as_dtype("f32"), int = as_dtype("i32"))
    withr::local_options(anvl.default_dtypes = "f64")
    expect_equal(default_dtypes(), registered)
    withr::local_options(anvl.default_dtypes = c(nope = "f64"))
    expect_equal(default_dtypes(), registered)
    withr::local_options(anvl.default_dtypes = c(float = "f16"))
    expect_equal(default_float(), as_dtype("f16"))
    # A name that is not a data type at all still fails where it is converted.
    withr::local_options(anvl.default_dtypes = c(float = "nope"))
    expect_error(default_dtypes(), "Unsupported dtype")
  })
})

describe("local_default_dtypes()", {
  it("sets the options for the scope", {
    local_registered_default_dtypes()
    local({
      local_default_dtypes(c(float = "f64", int = "i64"))
      # The setter writes an entry for the backend in force, so the option's
      # value is per backend even when only one was ever named.
      expect_identical(getOption("anvl.default_dtypes"), list(pjrt = c(float = "f64", int = "i64")))
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

  it("takes what it is given as it is", {
    # A data type of the other category, or one no backend supports, is taken
    # as given: it commits wherever the default is read and fails there.
    local_default_dtypes(c(float = "i32"))
    expect_equal(default_float(), as_dtype("i32"))
    expect_error(local_default_dtypes(c(float = "nope")), "Unsupported dtype")
  })
})

describe("with_default_dtypes()", {
  it("scopes the change to the expression", {
    before <- default_float()
    expect_equal(with_default_dtypes(c(float = "f64"), default_float()), as_dtype("f64"))
    expect_equal(default_float(), before)
    expect_equal(with_default_dtypes(c(int = "i64"), default_int()), as_dtype("i64"))
  })

  it("nests, each setter merging into the one around it", {
    # Cleared so the assertions below rest on the setters alone: a suite-wide
    # override would supply the category a non-merging setter drops.
    local_registered_default_dtypes()
    # A setter reads the option already in force, so an inner one naming the
    # other category adds to the outer override instead of replacing it --
    # whichever order they come in.
    expect_equal(
      with_default_dtypes(c(float = "f64"), with_default_dtypes(c(int = "i64"), default_dtypes())),
      list(float = as_dtype("f64"), int = as_dtype("i64"))
    )
    expect_equal(
      with_default_dtypes(c(int = "i64"), with_default_dtypes(c(float = "f64"), default_dtypes())),
      list(float = as_dtype("f64"), int = as_dtype("i64"))
    )
    # Both naming the same category: the innermost wins, and the outer one is
    # back in force once it exits.
    expect_equal(
      with_default_dtypes(c(float = "f64"), {
        c(
          inner = with_default_dtypes(c(float = "f32"), as.character(default_float())),
          after = as.character(default_float())
        )
      }),
      c(inner = "f32", after = "f64")
    )
  })
})

describe("an override of one backend", {
  # The registered defaults are a property of the backend, so an override is
  # one too: the setters change the backend in force, and the option's value
  # may name a backend per entry.
  it("is where a setter that names no backend goes", {
    local_registered_default_dtypes()
    local_default_dtypes(c(int = "i64"))
    expect_equal(default_int(), as_dtype("i64"))
    expect_equal(with_backend("quickr", default_int()), as_dtype("i32"))
  })

  it("can be set for a backend that is not in force", {
    local_registered_default_dtypes()
    local_backend("quickr")
    local_default_dtypes(c(int = "i64"), backend = "pjrt")
    expect_equal(default_int(), as_dtype("i32"))
    expect_equal(with_backend("pjrt", default_int()), as_dtype("i64"))
  })

  it("is read off an entry of the option that names it", {
    withr::local_options(anvl.default_dtypes = list(pjrt = c(int = "i64")))
    expect_equal(default_int(), as_dtype("i64"))
    expect_equal(with_backend("quickr", default_int()), as_dtype("i32"))
  })

  it("wins over a value that names no backend", {
    # `float` / `int` at the top of the option apply to every backend; an
    # entry named after one is that backend's own.
    withr::local_options(anvl.default_dtypes = list(int = "i64", pjrt = c(int = "i32")))
    expect_equal(default_int(), as_dtype("i32"))
    expect_equal(with_backend("quickr", default_int()), as_dtype("i64"))
  })

  it("leaves the other backends alone when a setter merges into it", {
    withr::local_options(anvl.default_dtypes = c(float = "f64"))
    local_default_dtypes(c(int = "i64"))
    expect_equal(default_dtypes(), list(float = as_dtype("f64"), int = as_dtype("i64")))
    expect_equal(
      with_backend("quickr", default_dtypes()),
      list(float = as_dtype("f64"), int = as_dtype("i32"))
    )
  })

  it("is not narrowed by what that backend can represent", {
    # The data types a backend supports are its own business: an override it
    # cannot honour is reported as set, and only fails where the data is
    # allocated or the program compiled (see `test-quickr-optional.R`).
    # quickr has no `i64`, named or not.
    local_default_dtypes(c(int = "i64"), backend = "quickr")
    expect_equal(with_backend("quickr", default_int()), as_dtype("i64"))
    withr::local_options(anvl.default_dtypes = c(int = "i64"))
    expect_equal(with_backend("quickr", default_int()), as_dtype("i64"))
  })

  it("cannot be set for another backend from inside a trace", {
    # A program is compiled for one backend, so an override filed under another
    # could not reach it: an error rather than a silent no-op.
    local({
      desc <- local_descriptor()
      expect_error(
        local_default_dtypes(c(int = "i64"), backend = "quickr"),
        "compiled for the .*pjrt.* backend"
      )
      # The trace's own backend is fine, named or not.
      expect_error(local_default_dtypes(c(int = "i64"), backend = "pjrt"), NA)
      expect_error(local_default_dtypes(c(int = "i64")), NA)
    })
  })

  it("is passed over when it names a backend that does not exist", {
    withr::local_options(anvl.default_dtypes = list(pjrtt = c(int = "i64")))
    expect_equal(default_int(), as_dtype("i32"))
  })

  it("is what a compiled program of that backend is keyed on", {
    # The dispatcher reads the pair through this resolver on every call, so it
    # is what an entry of that backend's cache is filed under -- and what the
    # trace is then pinned to.
    local_registered_default_dtypes()
    pjrt_key <- default_dtypes_context("pjrt")
    quickr_key <- default_dtypes_context("quickr")
    expect_identical(pjrt_key(), c(float = "f32", int = "i32"))
    expect_identical(quickr_key(), c(float = "f64", int = "i32"))
    withr::local_options(anvl.default_dtypes = list(pjrt = c(int = "i64")))
    expect_identical(pjrt_key(), c(float = "f32", int = "i64"))
    expect_identical(quickr_key(), c(float = "f64", int = "i32"))
  })
})

describe("with_dtypes()", {
  it("converts the arguments, the body's defaults and the result", {
    local_registered_default_dtypes()
    add_f64 <- with_dtypes(nv_add, c(float = "f64"))
    out <- add_f64(nv_array(1, dtype = "f32"), 2.5)
    expect_equal(dtype(out), as_dtype("f64"))
    # The `f32` operand is converted before the call, so the R value meets an
    # `f64` array and the sum keeps every digit of 2.5.
    expect_equal(as.vector(out), 3.5)
    # A category the wrapper does not name is untouched, in the arguments and
    # in the result.
    expect_equal(dtype(add_f64(nv_array(1L, dtype = "i32"), 2L)), as_dtype("i32"))
    expect_equal(dtype(add_f64(nv_array(TRUE), TRUE)), as_dtype("bool"))
    # Inside the body the defaults are the ones the wrapper names.
    expect_equal(dtype(with_dtypes(function() nv_fill(0, 2), c(float = "f64"))()), as_dtype("f64"))
    expect_equal(default_float(), as_dtype("f32"))
  })

  it("converts every output of a jitted function and leaves a static argument alone", {
    local_registered_default_dtypes()
    f <- jit(function(x, n) list(a = x + 1, b = nv_fill(0, n)), static = "n")
    g <- with_dtypes(f, c(float = "f64", int = "i64"))
    out <- g(nv_array(c(1, 2), dtype = "f32"), 2L)
    expect_equal(lapply(out, dtype), list(a = as_dtype("f64"), b = as_dtype("f64")))
    expect_equal(as.vector(out$a), c(2, 3))
  })

  it("converts an unsigned array without setting a default for it", {
    local_registered_default_dtypes()
    # `uint` is a conversion category only, so a wrapper naming just it leaves
    # the defaults of the call alone.
    f <- with_dtypes(function(x) list(x = x, filled = nv_fill(0, 2)), c(uint = "ui32"))
    out <- f(nv_array(1L, dtype = "ui8"))
    expect_equal(dtype(out$x), as_dtype("ui32"))
    expect_equal(dtype(out$filled), as_dtype("f32"))
    # Named alongside the others it converts as they do.
    g <- with_dtypes(nv_add, c(float = "f64", uint = "ui32"))
    expect_equal(dtype(g(nv_array(1L, dtype = "ui8"), 2L)), as_dtype("ui32"))
    expect_equal(dtype(g(nv_array(1, dtype = "f32"), 2)), as_dtype("f64"))
  })

  it("keeps the signature of the function it wraps", {
    local_registered_default_dtypes()
    f <- function(x, n = 2, ...) list(x = x, filled = nv_fill(0, n))
    g <- with_dtypes(f, c(float = "f64"))
    expect_identical(formals(g), formals(f))
    # An argument the wrapper is not given is left out, so `f`'s own default
    # decides -- and `...` reaches `f` as it would without the wrapper.
    expect_equal(shape(g(nv_array(1, dtype = "f32"))$filled), 2L)
    expect_equal(shape(g(nv_array(1, dtype = "f32"), 3)$filled), 3L)
    expect_equal(
      dtype(with_dtypes(function(...) nv_add(...), c(float = "f64"))(
        nv_array(1, dtype = "f32"),
        2
      )),
      as_dtype("f64")
    )
  })

  it("walks a structured argument and result as a tree", {
    local_registered_default_dtypes()
    f <- with_dtypes(
      function(pair, scale) {
        list(sum = pair$a + pair$b, scaled = list(pair$a * scale))
      },
      c(float = "f64")
    )
    out <- f(list(a = nv_array(1, dtype = "f32"), b = nv_array(2, dtype = "f32")), 2)
    expect_named(out, c("sum", "scaled"))
    expect_equal(dtype(out$sum), as_dtype("f64"))
    expect_equal(dtype(out$scaled[[1L]]), as_dtype("f64"))
    expect_equal(as.vector(out$sum), 3)
  })

  it("jits a jitted function again with its own configuration", {
    local_registered_default_dtypes()
    f <- jit(function(x, flag) if (flag) x + 1 else x * 2, static = "flag", cache_size = 7L, donate = "x")
    g <- with_dtypes(f, c(float = "f64"))
    # The conversions are part of the compiled program, so the result is a
    # `JitFunction` again -- with the configuration `f` was built with.
    expect_s3_class(g, "JitFunction")
    cfg <- jit_config(g)
    expect_identical(cfg$static, "flag")
    expect_identical(cfg$cache_size, 7L)
    expect_identical(cfg$dots, list(donate = "x"))
    # The static argument still selects the branch, and stays an R value.
    expect_equal(as.vector(g(nv_array(3, dtype = "f32"), TRUE)), 4)
    expect_equal(as.vector(g(nv_array(3, dtype = "f32"), FALSE)), 6)
    expect_equal(dtype(g(nv_array(3, dtype = "f32"), TRUE)), as_dtype("f64"))
  })

  it("rejects anything but a mapping of the data type categories", {
    expect_error(with_dtypes(1, c(float = "f64")), "must be a function")
    expect_error(with_dtypes(nv_add, "f64"), "must map the data type categories")
    expect_error(with_dtypes(nv_add, c(flaot = "f64")), "must map the data type categories")
    expect_error(with_dtypes(nv_add, c(bool = "bool")), "must map the data type categories")
    expect_error(with_dtypes(nv_add, character()), "must map the data type categories")
  })
})
