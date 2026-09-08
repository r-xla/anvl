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
