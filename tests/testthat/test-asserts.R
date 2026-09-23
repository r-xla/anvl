describe("assert_shapevec()", {
  it("reports a malformed whole number the way `assert_int_param()` does", {
    # One wording per mistake: a shape resolved in a `prim_*()` wrapper and an
    # axis vector checked in an inference rule go through the same helper.
    expect_snapshot(error = TRUE, assert_shapevec(c(1L, NA_integer_), var_name = "shape"))
    expect_snapshot(error = TRUE, assert_shapevec(Inf, var_name = "shape"))
    expect_snapshot(error = TRUE, assert_shapevec(integer(), min_len = 1L, var_name = "shape"))
  })

  it("keeps the checks only a shape needs, and names the argument for them", {
    # `var_name` is forced before `x` is rebound: `caller_arg()` deparses
    # whatever `x` holds when it is first read, so an unforced default would
    # name the value (`-2L`) rather than the argument.
    expect_snapshot(error = TRUE, assert_shapevec(-2L, var_name = "shape"))
    expect_snapshot(error = TRUE, assert_shapevec(rep(2147483647L, 3L), var_name = "shape"))
    # `NULL` is the empty set of axes to `assert_int_param()`, but not a shape.
    expect_snapshot(error = TRUE, assert_shapevec(NULL, var_name = "shape"))
  })
})

describe("assert_linalg_matrix()", {
  it("gives every linear algebra primitive one wording for one mistake", {
    # `prim_chol()` used to check its operand in the wrapper as well, so the
    # same mistake read differently there than in the rules.
    msg <- function(f, x) conditionMessage(tryCatch(f(x), error = identity))

    # A constraint all five share: the operand has to be a float.
    int_square <- nv_array(matrix(1:4, 2L, 2L))
    expect_equal(msg(prim_chol, int_square), msg(prim_qr, int_square))
    expect_equal(msg(prim_chol, int_square), msg(prim_eigh, int_square))
    expect_match(msg(prim_chol, int_square), "must have a float data type", fixed = TRUE)

    empty <- nv_array(numeric(), shape = c(0L, 0L), dtype = "f32")
    expect_equal(msg(prim_chol, empty), msg(prim_qr, empty))
    expect_match(msg(prim_chol, empty), "zero-sized axis", fixed = TRUE)
  })
})

describe("assert_fill_value()", {
  it("reports a NaN as the value it is", {
    # `{.obj_type_friendly}` calls a NaN "a numeric `NA`", which beside the
    # value itself used to read as three things ("a numeric `NA` NaN").
    expect_snapshot(error = TRUE, assert_fill_value(NaN, "i32", arg = "value"))
  })

  it("still describes a value of the wrong type by its type", {
    expect_snapshot(error = TRUE, assert_fill_value("a", "f32", arg = "value"))
    expect_snapshot(error = TRUE, assert_fill_value(1.5, "i32", arg = "value"))
  })
})
