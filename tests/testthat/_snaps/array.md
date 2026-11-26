# array

    Code
      x
    Output
      AnvlArray
       1
       2
       3
       4
      [ CPUi32{4,1} ] 

# nv_array()'s shape / refuses NULL rather than inferring the shape or building a scalar

    Code
      nv_array(x, shape = dim(x))
    Condition
      Error in `nv_array()`:
      ! `shape` must be "auto" or a whole number vector.
      x Got NULL.
      i Leave it at "auto" to infer it, or use `shp()` for the shape of a scalar.

# nv_array()'s shape / names the constructor, not a helper, in its errors

    Code
      nv_array(1:6, shape = c(2, -3))
    Condition
      Error in `nv_array()`:
      ! `shape` must not contain a negative axis size.
      x Got c(2, -3).

---

    Code
      nv_empty(c(2.5, 3), dtype = "f32")
    Condition
      Error in `nv_empty()`:
      ! `shape` must contain whole numbers.
      x Got c(2.5, 3).

# nv_scalar

    Code
      x
    Output
      AnvlArray
       1
      [ CPUf32{} ] 

# AbstractArray

    Code
      x
    Output
      AbstractArray(dtype=f32, shape=2x3) 

# ConcreteArray

    Code
      x
    Output
      ConcreteArray
       1 3 5
       2 4 6
      [ CPUf32{2,3} ] 

# stablehlo dtype is printed

    Code
      nv_array(TRUE)
    Output
      AnvlArray
       1
      [ CPUbool{1} ] 

# shp() / refuses NULL, so the dim() of a vector does not become the shape of a scalar

    Code
      shp(2, NULL)
    Condition
      Error in `shp()`:
      ! Axis sizes must be whole numbers.
      x Argument 2 is NULL.
      i Call `shp()` without arguments for the shape of a scalar.

# shp() / refuses what is not an axis size

    Code
      shp(-1)
    Condition
      Error in `shp()`:
      ! `...` must not contain a negative axis size.
      x Got -1.

