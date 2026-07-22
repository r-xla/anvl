# GraphLiteral

    Code
      gl
    Output
      GraphLiteral(1, i32?, ()) 

# error handling

    Code
      jit(prim_ceil)(nv_array(1:4))
    Condition
      Error:
      ! 'to_array_terminology' is not an exported object from 'namespace:stablehlo'

---

    Code
      jit(prim_transpose, static = "permutation")(nv_array(1:4, shape = c(2, 2)),
      permutation = c(2, 2))
    Condition
      Error:
      ! 'to_array_terminology' is not an exported object from 'namespace:stablehlo'

# error handling: stablehlo errors speak of arrays, not tensors

    Code
      jit(prim_add)(nv_array(1:4), nv_array(c(1, 2, 3, 4)))
    Condition
      Error:
      ! 'to_array_terminology' is not an exported object from 'namespace:stablehlo'

# can print GraphLiteral if it holds scalar array

    Code
      GraphLiteral(LiteralArray(nv_scalar(1L), dtype = "i32", shape = integer(),
      ambiguous = TRUE))
    Output
      GraphLiteral(1, i32?, ()) 

