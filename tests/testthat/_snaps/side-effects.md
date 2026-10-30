# a runtime side effect / is an input and an output of the printed graph

    Code
      trace_fn(function(x) {
        tick()
        x + 1L
      }, list(x = nv_aval("i32", integer())))
    Output
      <AnvlGraph> (%x1: i32[], %tally1: i32[]) {
        %1: i32[] = add(%tally1, 1:i32)
        %2: i32[] = add(%x1, 1:i32)
        return (%2, %1)
      }

