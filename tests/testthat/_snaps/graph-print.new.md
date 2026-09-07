# sub-graphs (if)

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: i1[integer(0)]
        Constants:
          %c1: f32[integer(0)]
          %c2: f32[integer(0)]
        Body:
          %1: f32[integer(0)] = if [true_graph = graph[0 -> 1], false_graph = graph[0 -> 1]] (%x1)
        Outputs:
          %1: f32[integer(0)] 

# params

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: i32[10]
        Body:
          %1: i32[integer(0)] = reduce_max [axes = 1, drop = TRUE] (%x1)
        Outputs:
          %1: i32[integer(0)] 

# an input the caller supplies as bare R data names its R type

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: f64[integer(0)]
          %x2: f64[integer(0)] <- double
        Body:
          %1: f64[integer(0)] = add(%x1, %x2)
        Outputs:
          %1: f64[integer(0)] 

