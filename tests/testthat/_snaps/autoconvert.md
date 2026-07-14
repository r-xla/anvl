# jit: bare vector without dim errors

    Code
      f(c(1, 2, 3))
    Condition
      Error:
      ! invalid input `x`: expected an AnvlArray, a length-1 atomic scalar, or an is.array() value; got <numeric> of length 3

# jit: non-array/non-scalar leaves (e.g. character) error

    Code
      f("hello")
    Condition
      Error:
      ! invalid input `x`: expected an AnvlArray, a length-1 atomic scalar, or an is.array() value; got <character> of length 1

# xla: bare vector errors

    Code
      f_compiled(c(1, 2, 3))
    Condition
      Error in `check_jit_input()`:
      ! Attempted to autoconvert `x` to an <AnvlArray>.
      i Expected an <AnvlArray>, a length-1 atomic scalar, or an `is.array()` value.
      x Got <numeric> of length 3.

# quickr: bare vector errors

    Code
      f(c(1, 2, 3))
    Condition
      Error:
      ! invalid input `x`: expected an AnvlArray, a length-1 atomic scalar, or an is.array() value; got <numeric> of length 3

# jit: error shows path for nested list element

    Code
      f(list(list(a = "abc")))
    Condition
      Error:
      ! invalid input `l[[1]]$a`: expected an AnvlArray, a length-1 atomic scalar, or an is.array() value; got <character> of length 1

# jit: error shows path for unnamed nested element

    Code
      f(list("bad", nv_scalar(1)))
    Condition
      Error:
      ! invalid input `pair[[1]]`: expected an AnvlArray, a length-1 atomic scalar, or an is.array() value; got <character> of length 1

# xla: error shows path for nested list element

    Code
      f_compiled(list("bad", nv_scalar(1)))
    Condition
      Error in `check_jit_input()`:
      ! Attempted to autoconvert `pair[[1]]` to an <AnvlArray>.
      i Expected an <AnvlArray>, a length-1 atomic scalar, or an `is.array()` value.
      x Got <character> of length 1.

