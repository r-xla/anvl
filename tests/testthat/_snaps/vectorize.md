# vectorize / traces batched sample_bits with randomness independent

    Code
      graph
    Output
      <AnvlGraph> (%x1: f32[3], %key1: ui32[2], %counter1: ui64[]) {
        %1: ui64[3] = iota [axis = 1, shape = 3, dtype = "ui64", start = 0] ()
        %key2: ui32[2,3] = broadcast_in_axes [
          shape = c(2, 3), broadcast_axes = 1
        ] (%key1)
        %counter2: ui64[3] = add(%counter1, %1)
        %2: ui32[2,3] = sample_bits [shape = 2, dtype = "ui32"] (%key2, %counter2)
        %counter3: ui64[] = add(%counter1, 3:ui64)
        %3: ui32[2] = sample_bits [shape = 2, dtype = "ui32"] (%key1, %counter3)
        return (%2, %3)
      } then {
        %counter1 += 4
      }

# vectorize / traces shared sample_bits with randomness common

    Code
      graph
    Output
      <AnvlGraph> (%x1: f32[3], %key1: ui32[2], %counter1: ui64[]) {
        %1: ui32[2] = sample_bits [shape = 2, dtype = "ui32"] (%key1, %counter1)
        %2: ui32[2,3] = broadcast_in_axes [shape = c(2, 3), broadcast_axes = 1] (%1)
        %counter2: ui64[] = add(%counter1, 1:ui64)
        %3: ui32[2] = sample_bits [shape = 2, dtype = "ui32"] (%key1, %counter2)
        return (%2, %3)
      } then {
        %counter1 += 2
      }

