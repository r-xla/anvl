describe("assert_shapevec()", {
  it("ends every complaint with a full stop, as the rest of the set does", {
    expect_snapshot(error = TRUE, assert_shapevec(c(1L, NA_integer_), var_name = "shape"))
    expect_snapshot(error = TRUE, assert_shapevec(Inf, var_name = "shape"))
    expect_snapshot(error = TRUE, assert_shapevec(integer(), min_len = 1L, var_name = "shape"))
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
