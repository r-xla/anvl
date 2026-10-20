# the global random state in a traced graph / gives every sampling call the key and its own counter, computed from the one the graph takes

    Code
      graph
    Output
      <AnvlGraph> (%key1: ui32[2], %counter1: ui64[]) {
        %1: ui32[2] = sample_bits [shape = 2, dtype = "ui32"] (%key1, %counter1)
        %counter2: ui64[] = add(%counter1, 1:ui64)
        %2: ui32[3] = sample_bits [shape = 3, dtype = "ui32"] (%key1, %counter2)
        return (%1, %2)
      }

# the global random state in a traced graph / forks the random state for nv_while and carries the counter through its body

    Code
      graph
    Output
      <AnvlGraph> (%x1: i32[], %key1: ui32[2], %counter1: ui64[]) {
        (%key2, %counter2): (ui32[2], ui64[]) = fork_random_state(%key1, %counter1)
        (%1, %key3, %counter3): (i32[], ui32[2], ui64[]) = while [
          cond = (%x2: i32[], %key4: ui32[2], %counter4: ui64[]) {
            %2: bool[] = lt(%x2, 3:i32)
            return %2
          },
          body = (%x3: i32[], %key5: ui32[2], %counter5: ui64[]) {
            %3: ui8[] = sample_bits [
              shape = integer(0), dtype = "ui8"
            ] (%key5, %counter5)
            %4: i32[] = convert [dtype = i32] (%3)
            %5: i32[] = add(%x3, %4)
            %counter6: ui64[] = add(%counter5, 1:ui64)
            return (%5, %key5, %counter6)
          },
          n_captures = 0
        ] (%x1, %key2, %counter2)
        return %1
      }

# the global random state in a traced graph / draws after an nv_while loop at the counter after its fork

    Code
      graph
    Output
      <AnvlGraph> [%c1: i32[]] (%x1: i32[], %key1: ui32[2], %counter1: ui64[]) {
        (%key2, %counter2): (ui32[2], ui64[]) = fork_random_state(%key1, %counter1)
        (%1, %key3, %counter3): (i32[], ui32[2], ui64[]) = while [
          cond = (%x2: i32[], %key4: ui32[2], %counter5: ui64[], %x3: i32[]) {
            %3: bool[] = lt(%x2, %x3)
            return %3
          },
          body = (%x4: i32[], %key5: ui32[2], %counter6: ui64[], %x5: i32[]) {
            %4: ui32[2] = sample_bits [shape = 2, dtype = "ui32"] (%key5, %counter6)
            %5: i32[] = add(%x4, 1:i32)
            %counter7: ui64[] = add(%counter6, 1:ui64)
            return (%5, %key5, %counter7)
          },
          n_captures = 1
        ] (%c1, %key2, %counter2, %x1)
        %counter4: ui64[] = add(%counter1, 1:ui64)
        %2: ui32[3] = sample_bits [shape = 3, dtype = "ui32"] (%key1, %counter4)
        return (%1, %2)
      }

# the global random state in a traced graph / carries the random state through the steps of nv_scan by its counter

    Code
      graph
    Output
      <AnvlGraph> [%c1: i32[]] (%key1: ui32[2], %counter1: ui64[]) {
        %1: ui32[2] = sample_bits [shape = 2, dtype = "ui32"] (%key1, %counter1)
        %counter2: ui64[] = add(%counter1, 1:ui64)
        (%2, %key2, %counter3, %3, %4): (
          i32[], ui32[2], ui64[], ui32[3,1], ui32[3,1]
        ) = scan [
          body = (%x1: i32[], %key3: ui32[2], %counter4: ui64[]) {
            %5: ui32[1] = sample_bits [shape = 1, dtype = "ui32"] (%key3, %counter4)
            %counter5: ui64[] = add(%counter4, 1:ui64)
            %6: ui32[1] = sample_bits [shape = 1, dtype = "ui32"] (%key3, %counter5)
            %counter6: ui64[] = add(%counter4, 2:ui64)
            return (%x1, %key3, %counter6, %5, %6)
          },
          steps = 3, reverse = FALSE, n_carry = 3, n_xs = 0, n_captures = 0
        ] (%c1, %key1, %counter2)
        return (%3, %4)
      }

# the global random state in a traced graph / lets the branches of nv_if continue the counter of the random state

    Code
      graph
    Output
      <AnvlGraph> (%x1: bool[], %key1: ui32[2], %counter1: ui64[]) {
        %1: ui32[2] = sample_bits [shape = 2, dtype = "ui32"] (%key1, %counter1)
        %2: ui32[1] = if [
          true = (%counter2: ui64[], %key2: ui32[2]) {
            %counter3: ui64[] = add(%counter2, 1:ui64)
            %3: ui32[1] = sample_bits [shape = 1, dtype = "ui32"] (%key2, %counter3)
            return %3
          },
          false = (%counter4: ui64[], %key3: ui32[2]) {
            %counter5: ui64[] = add(%counter4, 1:ui64)
            %4: ui32[1] = sample_bits [shape = 1, dtype = "ui32"] (%key3, %counter5)
            %counter6: ui64[] = add(%counter4, 2:ui64)
            %5: ui32[1] = sample_bits [shape = 1, dtype = "ui32"] (%key3, %counter6)
            return %5
          },
          n_captures = 2
        ] (%x1, %counter1, %key1)
        return %2
      }

