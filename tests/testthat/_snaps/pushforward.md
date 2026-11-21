# pushforward / refuses a tangent for an array that is not a float

    Code
      jit(function(x, n) pushforward(f, x, n)(n = n))(f64(1), nv_array(1L, dtype = "i32"))
    Condition
      Error:
      ! Can only pass a tangent for a float array.
      x `n` is an array of data type "i32".
      i Pass `NULL` in its place.

# pushforward / refuses a tangent of another data type, shape or structure than its argument

    Code
      jit(function(x) pushforward(f, x)(nv_array(1, dtype = "f32")))(f64(1))
    Condition
      Error:
      ! The tangent of `x` must have the data type and shape of the argument.
      x The argument is "f64[1]", the tangent "f32[1]".

---

    Code
      jit(function(x) pushforward(f, x)(f64(1, 2)))(f64(1))
    Condition
      Error:
      ! The tangent of `x` must have the data type and shape of the argument.
      x The argument is "f64[1]", the tangent "f64[2]".

