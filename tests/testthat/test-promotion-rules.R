test_that("common_dtype_of: single argument", {
  expect_equal(common_dtype_of(AbstractArray("i32", Shape(c(1, 2)))), as_dtype("i32"))
  expect_equal(common_dtype_of(RData(integer(), "double")), as_dtype("f32"))
})

test_that("common_dtype_of: two typed arguments", {
  check <- function(dt1, dt2, expected) {
    s1 <- AbstractArray(dt1, Shape(c(1, 2)))
    s2 <- AbstractArray(dt2, Shape(c(2, 1)))
    expect_equal(common_dtype_of(s1, s2), as_dtype(expected))
    expect_equal(common_dtype_of(s2, s1), as_dtype(expected))
  }

  check("i32", "i32", "i32")
  check("f32", "i32", "f32")
  check("f32", "f64", "f64")
  check("ui32", "i32", "i64")
})

test_that("common_dtype_of: an R value yields to a typed one", {
  check <- function(data, dt, expected) {
    rd <- RData(integer(), typeof(data))
    known <- AbstractArray(dt, Shape(c(2, 1)))
    expect_equal(common_dtype_of(rd, known), as_dtype(expected))
    expect_equal(common_dtype_of(known, rd), as_dtype(expected))
  }

  # It takes the dtype it meets, within its own category ...
  check(1L, "i8", "i8")
  check(1L, "i64", "i64")
  check(1.5, "f64", "f64")
  check(1.5, "f32", "f32")
  # ... crosses to the other category when that is what it meets ...
  check(1L, "f64", "f64")
  # ... but a float R value never becomes an integer, and nothing becomes a bool.
  check(1.5, "i32", "f32")
  check(1L, "bool", "i32")
  check(TRUE, "i32", "i32")
})

test_that("common_dtype_of: R values among themselves take their defaults", {
  check <- function(d1, d2, expected) {
    r1 <- RData(integer(), typeof(d1))
    r2 <- RData(integer(), typeof(d2))
    expect_equal(common_dtype_of(r1, r2), as_dtype(expected))
    expect_equal(common_dtype_of(r2, r1), as_dtype(expected))
  }
  check(1L, 2L, "i32")
  check(1.5, 2.5, "f32")
  check(1L, 2.5, "f32")
  check(TRUE, FALSE, "bool")
  check(TRUE, 1L, "i32")
  check(TRUE, 1.5, "f32")
})

test_that("common_dtype_of: multiple arguments", {
  i32 <- AbstractArray("i32", Shape(1))
  f32 <- AbstractArray("f32", Shape(2))
  f64 <- AbstractArray("f64", Shape(3))

  expect_equal(common_dtype_of(i32, f32, f64), as_dtype("f64"))
  expect_equal(common_dtype_of(f64, f32, i32), as_dtype("f64"))
  expect_equal(common_dtype_of(i32, i32, i32), as_dtype("i32"))
  # An R value in the middle still yields to the typed ones around it.
  expect_equal(common_dtype_of(RData(integer(), "integer"), AbstractArray("i64", Shape(2))), as_dtype("i64"))
  expect_equal(
    common_dtype_of(RData(integer(), "double"), i32, AbstractArray("f64", Shape(1))),
    as_dtype("f64")
  )
})

test_that("common_dtype_of: error on no arguments", {
  expect_error(common_dtype_of(), "No arguments provided")
})

test_that("promote_dt_known", {
  check <- function(dt1, dt2, dt3) {
    expect_equal(
      promote_dt_known(as_dtype(dt1), as_dtype(dt2)),
      as_dtype(dt3)
    )
    expect_equal(
      promote_dt_known(as_dtype(dt2), as_dtype(dt1)),
      as_dtype(dt3)
    )
  }

  check("f64", "f64", "f64")
  check("i32", "i32", "i32")
  check("bool", "bool", "bool")

  # floats dominate
  check("f64", "f32", "f64")
  check("f64", "i32", "f64")
  check("f32", "i32", "f32")
  check("f32", "bool", "f32")

  # signed ints
  check("i32", "i16", "i32")
  check("i64", "i32", "i64")
  check("i64", "i16", "i64")
  check("i64", "bool", "i64")
  # against unsigned ints
  check("i32", "ui8", "i32")
  check("i32", "ui32", "i64")
  check("i64", "ui64", "i64")
  # unsigned vs unsigned
  check("ui64", "ui32", "ui64")
})

test_that("promote_dt_rdata", {
  check <- function(rdtype, known, z) {
    expect_equal(
      promote_dt_rdata(as_dtype(rdtype), as_dtype(known)),
      as_dtype(z)
    )
  }
  # An R value commits to i32, f32 or bool, and yields from there.
  check("i32", "i32", "i32")
  check("bool", "bool", "bool")
  check("f32", "f32", "f32")

  check("i32", "i8", "i8")
  check("i32", "i64", "i64")
  check("i32", "bool", "i32")

  check("f32", "f64", "f64")
  check("f32", "i32", "f32")

  check("bool", "f32", "f32")
  check("bool", "i32", "i32")
})

test_that("common_dtype is the promotion of two known dtypes", {
  expect_equal(common_dtype("i32", "f32"), as_dtype("f32"))
  expect_equal(common_dtype("i32", "i64"), as_dtype("i64"))
  expect_equal(common_dtype("f64", "f32"), as_dtype("f64"))
})

test_that("a rule that cannot place an argument says which one", {
  # The diagnosis is only useful if it points at the operand to change, which
  # is what the multi-operand calls need it for.
  x <- nv_array(c(1, 2), dtype = "f64")
  expect_error(nv_pad(x, 0L, 1L, 1L), "`padding_value` is an R integer")
  expect_error(prim_pad(1.5, 1L, 0L, 0L, 0L), "`x` is an R double and `padding_value` is an R integer")
  expect_error(as_anvl_arrays(1.5, 1L, .promote = promote_rdata_common()), "argument 1 is an R double")
  expect_error(as_anvl_arrays(v = 1.5, .promote = promote_dtype("i32")), "Cannot bring `v`")
  # `coerce` is an argument of the rule, not of the function the user called, so
  # it is not offered as a way out here.
  err <- tryCatch(as_anvl_arrays(v = 1.5, .promote = promote_dtype("i32")), error = identity)
  expect_false(any(grepl("coerce", conditionMessage(err), fixed = TRUE)))
  coerced <- suppressWarnings(as_anvl_arrays(v = 1.5, .promote = promote_dtype("i32", coerce = TRUE)))
  expect_equal(dtype(coerced$v), as_dtype("i32"))
})

test_that("common_dtype_of: a fallback settles what R values alone commit to", {
  rint <- RData(integer(), "integer")
  rdbl <- RData(integer(), "double")
  rlgl <- RData(integer(), "logical")

  # With nothing to read a dtype off, the fallback replaces the default the R
  # values would take on their own.
  expect_equal(common_dtype_of(rint, rint, .fallback = "f32"), as_dtype("f32"))
  expect_equal(common_dtype_of(rdbl, rdbl, .fallback = "f64"), as_dtype("f64"))
  expect_equal(common_dtype_of(rlgl, .fallback = "i8"), as_dtype("i8"))
  # ... and no fallback leaves them their default.
  expect_equal(common_dtype_of(rint, rint), as_dtype("i32"))

  # The R values yield to it as they do to any dtype, so one in a lower
  # category leaves them where they are.
  expect_equal(common_dtype_of(rdbl, .fallback = "i32"), as_dtype("f32"))
  expect_equal(common_dtype_of(rint, .fallback = "bool"), as_dtype("i32"))

  # An argument that brings a dtype of its own claims them instead: the
  # fallback is ignored, whatever it named.
  expect_equal(
    common_dtype_of(AbstractArray("i32", Shape(2)), rint, .fallback = "f64"),
    as_dtype("i32")
  )
  expect_equal(
    common_dtype_of(AbstractArray("f64", Shape(2)), rdbl, .fallback = "f32"),
    as_dtype("f64")
  )
})

test_that("promote_common(fallback = ) realizes R values at the fallback", {
  # Nothing brings a dtype: every argument is built at the fallback.
  args <- as_anvl_arrays(1, 2L, .promote = promote_common(fallback = "f64"))
  expect_equal(dtype(args[[1L]]), as_dtype("f64"))
  expect_equal(dtype(args[[2L]]), as_dtype("f64"))

  # An argument that has one wins over the fallback.
  args <- as_anvl_arrays(nv_array(1L), 2L, .promote = promote_common(fallback = "f64"))
  expect_equal(dtype(args[[1L]]), as_dtype("i32"))
  expect_equal(dtype(args[[2L]]), as_dtype("i32"))

  # `on` still restricts which arguments the rule covers.
  args <- as_anvl_arrays(
    x = 1,
    y = 2,
    .promote = promote_common(on = "x", fallback = "f64")
  )
  expect_equal(dtype(args$x), as_dtype("f64"))
  expect_equal(dtype(args$y), as_dtype("f32"))

  expect_equal(format(promote_common(fallback = "f64")), "<promote_common(fallback f64)>")
  expect_equal(format(promote_common()), "<promote_common>")
})

test_that("promote_rdata_common() moves the R values and nothing else", {
  # The common data type of inputs that may not be converted is the one they
  # already share, and the R values are realized at it.
  args <- as_anvl_arrays(nv_array(1, dtype = "f64"), 1.5, .promote = promote_rdata_common())
  expect_equal(dtype(args[[1L]]), as_dtype("f64"))
  expect_equal(dtype(args[[2L]]), as_dtype("f64"))

  # `on` restricts which arguments have to agree; the rest keep their own.
  args <- as_anvl_arrays(
    x = nv_array(1L, dtype = "i8"),
    y = 1L,
    z = 2,
    .promote = promote_rdata_common(on = c("x", "y"))
  )
  expect_equal(dtype(args$y), as_dtype("i8"))
  expect_equal(dtype(args$z), as_dtype("f32"))

  # Several data types among the covered arguments have no common one to reach
  # without converting one of them, which this rule does not do.
  expect_error(
    as_anvl_arrays(
      a = nv_array(1, dtype = "f32"),
      b = nv_array(1, dtype = "f64"),
      .promote = promote_rdata_common()
    ),
    "no common data type"
  )
  # ... and the error names them, rather than leaving the mismatch to whatever
  # consumes the arguments.
  err <- tryCatch(
    as_anvl_arrays(
      a = nv_array(1, dtype = "f32"),
      b = nv_array(1, dtype = "f64"),
      .promote = promote_rdata_common()
    ),
    error = identity
  )
  expect_match(conditionMessage(err), "`a` is `f32` and `b` is `f64`", fixed = TRUE)

  # An argument the rule leaves out may still disagree.
  args <- as_anvl_arrays(
    x = nv_array(1, dtype = "f32"),
    y = nv_array(1, dtype = "f64"),
    .promote = promote_rdata_common(on = "x")
  )
  expect_equal(dtype(args$y), as_dtype("f64"))

  expect_equal(format(promote_rdata_common()), "<promote_rdata_common>")
})

test_that("a promotion rule is a function of the call's arguments", {
  # A rule anvl knows nothing about, written the way a package would: every
  # argument at the widest float in the call, and never below f32.
  widest_float <- function(args) {
    widths <- vapply(
      args,
      function(a) {
        dt <- peek_dtype(to_abstract(a))
        if (is_dtype_float(dt)) dtype_width(dt) else 0L
      },
      integer(1L)
    )
    rep(list(as_dtype(paste0("f", max(c(32L, widths))))), length(args))
  }

  out <- as_anvl_arrays(nv_array(1L), 2.5, .promote = widest_float)
  expect_equal(lapply(out, dtype), list(as_dtype("f32"), as_dtype("f32")))
  out <- as_anvl_arrays(nv_array(1L), 2.5, nv_array(1, dtype = "f64"), .promote = widest_float)
  expect_equal(unique(lapply(out, dtype)), list(as_dtype("f64")))

  # `NULL` leaves an argument where it is.
  keeps_second <- function(args) list(as_dtype("f64"), NULL)
  out <- as_anvl_arrays(nv_array(1L), nv_array(1L, dtype = "i8"), .promote = keeps_second)
  expect_equal(lapply(out, dtype), list(as_dtype("f64"), as_dtype("i8")))

  # It sees the arguments as the caller passed them, R values uncommitted.
  seen <- NULL
  spy <- function(args) {
    seen <<- args
    vector("list", length(args))
  }
  as_anvl_arrays(nv_array(1L), 2.5, .promote = spy)
  expect_true(is_anvl_array(seen[[1L]]))
  expect_identical(seen[[2L]], 2.5)

  # ... and composes with anvl's own through promote_grouped(), once it says
  # which arguments it covers. A bare function cannot be grouped: the group has
  # to know that before it calls anything.
  expect_error(
    promote_grouped(spy, promote_common(on = c("a", "b"))),
    "takes promotion rules"
  )
  mine <- promotion_rule(
    function(args) c(rep(list(as_dtype("f64")), 2L), vector("list", 2L)),
    "mine",
    on = c("x", "y")
  )
  out <- as_anvl_arrays(
    x = nv_array(1L),
    y = 2.5,
    a = nv_array(1L, dtype = "i8"),
    b = 2L,
    .promote = promote_grouped(mine, promote_common(on = c("a", "b")))
  )
  expect_equal(
    lapply(out, dtype),
    list(x = as_dtype("f64"), y = as_dtype("f64"), a = as_dtype("i8"), b = as_dtype("i8"))
  )
})

test_that("a rule that does not answer per argument is reported against the rule", {
  expect_error(
    as_anvl_arrays(nv_array(1L), 1.5, .promote = function(args) "f64"),
    "must answer with one data type per argument"
  )
  expect_error(
    as_anvl_arrays(nv_array(1L), 1.5, .promote = function(args) list(as_dtype("f64"))),
    "must answer with one data type per argument"
  )
  expect_error(
    as_anvl_arrays(nv_array(1L), 1.5, .promote = function(args) list("f64", "f64")),
    "must answer with one data type per argument"
  )
})

test_that("promote_grouped() refuses groups that could overlap", {
  # Checked where the group is built, against what the rules say they cover,
  # rather than on the first call that reaches it.
  expect_error(
    promote_grouped(promote_common(on = c("x", "y")), promote_common(on = c("y", "z"))),
    "covers the same argument"
  )
  expect_error(
    promote_grouped(promote_common(on = 1:2), promote_common(on = 2:3)),
    "covers the same argument"
  )
  # A rule that names no `on` covers any argument, so it can only stand alone.
  expect_error(
    promote_grouped(promote_common(), promote_common(on = "x")),
    "must say which arguments it covers"
  )
  expect_s3_class(promote_grouped(promote_common()), "PromotionRule")

  # A group is a rule, so groups nest -- and the check sees through them.
  expect_s3_class(
    promote_grouped(promote_grouped(promote_common(on = "x")), promote_common(on = "y")),
    "PromotionRule"
  )
  expect_error(
    promote_grouped(promote_grouped(promote_common(on = "x")), promote_common(on = "x")),
    "covers the same argument"
  )

  # Only a PromotionRule can be grouped; a bare function has nothing to declare.
  expect_error(promote_grouped(function(args) list()), "takes promotion rules")
  expect_error(promote_grouped(), "takes promotion rules")
})

test_that("promotion_rule() builds a rule that prints and groups like the built-in ones", {
  mine <- promotion_rule(
    function(args) list(as_dtype("f64"), as_dtype("f64"), NULL),
    "mine",
    on = c("x", "y")
  )
  expect_s3_class(mine, "PromotionRule")
  # A rule of your own names its own kind, and shows what it covers.
  expect_equal(format(mine), "<promote_mine on \"x\", \"y\">")

  # It groups with anvl's own, and a group declares what its rules cover
  # together -- which is what lets groups nest.
  grouped <- promote_grouped(mine, promote_common(on = "z"))
  expect_equal(attr(grouped, "spec")$on, c("x", "y", "z"))
  out <- as_anvl_arrays(x = 1L, y = 2L, z = 3.5, .promote = grouped)
  expect_equal(lapply(out, dtype), list(x = as_dtype("f64"), y = as_dtype("f64"), z = as_dtype("f32")))

  # `on` is a promise the rule has to keep: one that places an argument outside
  # it is still caught when the rules actually answer.
  bad <- promotion_rule(function(args) rep(list(as_dtype("f64")), length(args)), "bad", on = "x")
  expect_error(
    as_anvl_arrays(x = 1L, z = 2L, .promote = promote_grouped(bad, promote_common(on = "z"))),
    "covers the same argument"
  )
})
