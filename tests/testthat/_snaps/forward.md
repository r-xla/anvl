# jvp / refuses tangents not named by the arguments it differentiates with respect to

    Code
      jit(jvp(f, wrt = "x"))(f64(1), f64(2), tangents = list(y = f64(1)))
    Condition
      Error in `jvp()`:
      ! `tangents` must be a list with one tangent per argument differentiated with respect to, named by it.
      i Expected the names `x`.
      x Got `y`.

---

    Code
      jit(jvp(f))(f64(1), f64(2), tangents = list(x = f64(1)))
    Condition
      Error in `jvp()`:
      ! `tangents` must be a list with one tangent per argument differentiated with respect to, named by it.
      i Expected the names `x` and `y`.
      x Got `x`.

---

    Code
      jit(jvp(f, wrt = "x"))(f64(1), f64(2), tangents = f64(1))
    Condition
      Error in `jvp()`:
      ! `tangents` must be a list with one tangent per argument differentiated with respect to, named by it.
      i Expected the names `x`.
      x Got a single array.

# jvp / refuses to differentiate with respect to an argument that is not a float array

    Code
      jit(jvp(f))(nv_array(1L, dtype = "i32"), tangents = list(x = nv_array(1L,
        dtype = "i32")))
    Condition
      Error in `check_wrt_arrayish()`:
      ! Can only differentiate with respect to float arrays.
      x Got "i32".

# jvp / refuses a tangent of another data type or shape than its argument

    Code
      jit(jvp(f))(f64(1), tangents = list(x = nv_array(1, dtype = "f32")))
    Condition
      Error in `jvp()`:
      ! The tangent of `x` must have the data type and shape of the argument.
      x The argument is "f64[1]", the tangent "f32[1]".

---

    Code
      jit(jvp(f))(f64(1), tangents = list(x = f64(1, 2)))
    Condition
      Error in `jvp()`:
      ! The tangent of `x` must have the data type and shape of the argument.
      x The argument is "f64[1]", the tangent "f64[2]".

