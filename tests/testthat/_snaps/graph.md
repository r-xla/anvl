# GraphLiteral

    Code
      gl
    Output
      GraphLiteral(1, i32, ()) 

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
      GraphLiteral(LiteralArray(nv_scalar(1L), dtype = "i32", shape = integer()))
    Output
      GraphLiteral(1, i32, ()) 

# how an R value is built into a graph / builds an R scalar as an inlined literal and an R array as a constant

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: f32[2, 2]
        Constants:
          %c1: f32[2, 2]
        Body:
          %1: f32[2, 2] = broadcast_in_axes [shape = c(2, 2), broadcast_axes = <any>] (2:f32)
          %2: f32[2, 2] = mul(%x1, %1)
          %3: f32[2, 2] = add(%x1, %c1)
          %4: f32[2, 2] = add(%2, %3)
        Outputs:
          %4: f32[2, 2] 

# how an R value is built into a graph / builds a closed-over R array used twice as one constant

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: f32[2, 2]
        Constants:
          %c1: f32[2, 2]
        Body:
          %1: f32[2, 2] = add(%x1, %c1)
          %2: f32[2, 2] = add(%1, %c1)
        Outputs:
          %2: f32[2, 2] 

# how an R value is built into a graph / converts inside the program when the value crosses its category

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: i32[]
        Body:
          %1: i32[] = convert [dtype = i32] (1.5:f64)
          %2: i32[] = add(%x1, %1)
        Outputs:
          %2: i32[] 

# how an R value is built into a graph / uploads an R argument used at two data types once and converts

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: f32[]
          %x2: f64[] <- double
        Body:
          %1: f32[] = convert [dtype = f32] (%x2)
          %2: f32[] = mul(%x1, %1)
          %3: f64[] = convert [dtype = f64] (%2)
          %4: f64[] = add(%3, %x2)
        Outputs:
          %4: f64[] 

# how an R value is built into a graph / keeps an R argument open under an inlined gradient

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: f32[] <- double
        Constants:
          %c1: f32[]
        Body:
          %1: f32[] = mul(%x1, %x1)
          %2: f32[] = mul(%c1, %x1)
          %3: f32[] = mul(%c1, %x1)
          %4: f32[] = add(%2, %3)
        Outputs:
          %4: f32[] 

