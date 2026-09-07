# GraphLiteral

    Code
      gl
    Output
      GraphLiteral(1, i32, (integer(0))) 

# error handling

    Code
      jit(prim_ceil)(nv_array(1:4))
    Condition
      Error in `prim_ceil()`:
      ! `x` must have dtype float.
      x Got i32.

---

    Code
      jit(prim_transpose, static = "permutation")(nv_array(1:4, shape = c(2, 2)),
      permutation = c(2, 2))
    Condition
      Error in `resolve_axes()`:
      ! `permutation` must be between 1 and 1, or between -1 and -1 to count from the end.
      x Got 2 and 2.

# error handling: stablehlo errors use anvl's terminology

    Code
      jit(prim_add)(nv_array(1:4), nv_array(1:6))
    Condition
      Error in `prim_add()`:
      ! `lhs` and `rhs` must have the same array type.
      x Got array<4xi32> and array<6xi32>.

# can print GraphLiteral if it holds scalar array

    Code
      GraphLiteral(LiteralArray(nv_scalar(1L), dtype = "i32", shape = integer()))
    Output
      GraphLiteral(1, i32, (integer(0))) 

# how an R value is built into a graph / builds a closed-over R array used twice as one constant

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: f32[c(2, 2)]
        Constants:
          %c1: f32[c(2, 2)]
        Body:
          %1: f32[c(2, 2)] = add(%x1, %c1)
          %2: f32[c(2, 2)] = add(%1, %c1)
        Outputs:
          %2: f32[c(2, 2)] 

