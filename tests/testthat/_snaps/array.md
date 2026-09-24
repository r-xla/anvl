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

