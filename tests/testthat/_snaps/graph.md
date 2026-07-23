# GraphLiteral

    Code
      gl
    Output
      GraphLiteral(1, i32?, ()) 

# error handling

    Code
      jit(prim_ceil)(nv_array(1:4))
    Condition
      Error in `prim_ceil()`:
      ! `x` must have dtype FloatType.
      x Got i32.

---

    Code
      jit(prim_transpose, static = "permutation")(nv_array(1:4, shape = c(2, 2)),
      permutation = c(2, 2))
    Condition
      Error in `resolve_axes()`:
      ! `permutation` must not contain duplicate axes.
      x Got 2 and 2.

# error handling: stablehlo errors use anvl's terminology

    Code
      jit(prim_add)(nv_array(1:4), nv_array(c(1, 2, 3, 4)))
    Condition
      Error in `prim_add()`:
      ! `lhs` and `rhs` must have the same array type.
      x Got array<4xi32> and array<4xf32>.

# can print GraphLiteral if it holds scalar array

    Code
      GraphLiteral(LiteralArray(nv_scalar(1L), dtype = "i32", shape = integer(),
      ambiguous = TRUE))
    Output
      GraphLiteral(1, i32?, ()) 

