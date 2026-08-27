# literals

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: f32[]
        Body:
          %1: f32[] = convert [dtype = f32] (1:i32)
          %2: f32[] = mul(%x1, %1)
        Outputs:
          %2: f32[] 

---

    Code
      graph
    Output
      <AnvlGraph>
        Inputs: (none)
        Body:
          %1: f32[2, 1] = fill [value = 1, dtype = f32, shape = c(2, 1)] ()
        Outputs:
          %1: f32[2, 1] 

# constants

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: f32[]
        Constants:
          %c1: f32[]
        Body:
          %1: f32[] = add(%x1, %c1)
        Outputs:
          %1: f32[] 

# sub-graphs (if)

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: i1[]
        Constants:
          %c1: f32[]
          %c2: f32[]
        Body:
          %1: f32[] = if [true_graph = graph[0 -> 1], false_graph = graph[0 -> 1]] (%x1)
        Outputs:
          %1: f32[] 

# sub-graphs (while)

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: f32[]
        Constants:
          %c1: f32[]
          %c2: f32[]
        Body:
          %1: f32[] = while [cond_graph = graph[1 -> 1], body_graph = graph[1 -> 1]] (%c2)
        Outputs:
          %1: f32[] 

# params

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: i32[10]
        Body:
          %1: i32[] = reduce_max [axes = 1, drop = TRUE] (%x1)
        Outputs:
          %1: i32[] 

