# assert_shapevec() / reports a malformed whole number the way `assert_int_param()` does

    Code
      assert_shapevec(c(1L, NA_integer_), var_name = "shape")
    Condition
      Error in `assert_int_param()`:
      ! `shape` must not contain missing values.
      x Got c(1, NA).

---

    Code
      assert_shapevec(Inf, var_name = "shape")
    Condition
      Error in `assert_int_param()`:
      ! `shape` must contain whole numbers in the integer range.
      x Got Inf.

---

    Code
      assert_shapevec(integer(), min_len = 1L, var_name = "shape")
    Condition
      Error in `assert_int_param()`:
      ! `shape` must have at least 1 entry.
      x Got integer(0).

# assert_shapevec() / keeps the checks only a shape needs, and names the argument for them

    Code
      assert_shapevec(-2L, var_name = "shape")
    Condition
      Error in `assert_shapevec()`:
      ! `shape` must not contain a negative axis size.
      x Got -2.

---

    Code
      assert_shapevec(rep(2147483647L, 3L), var_name = "shape")
    Condition
      Error in `assert_shapevec()`:
      ! `shape` must describe an array with fewer than 2^63 elements.
      x Got c(2147483647, 2147483647, 2147483647).

---

    Code
      assert_shapevec(NULL, var_name = "shape")
    Condition
      Error in `assert_shape_not_null()`:
      ! `shape` must be a whole number vector.
      x Got NULL.

# assert_fill_value() / reports a NaN as the value it is

    Code
      assert_fill_value(NaN, "i32", arg = "value")
    Condition
      Error in `assert_fill_value()`:
      ! `value` must be a whole number to be built at data type "i32".
      x Got NaN.

# assert_fill_value() / still describes a value of the wrong type by its type

    Code
      assert_fill_value("a", "f32", arg = "value")
    Condition
      Error in `assert_fill_value()`:
      ! `value` must be a number to be built at data type "f32".
      x Got a string.

---

    Code
      assert_fill_value(1.5, "i32", arg = "value")
    Condition
      Error in `assert_fill_value()`:
      ! `value` must be a whole number to be built at data type "i32".
      x Got a number 1.5.

