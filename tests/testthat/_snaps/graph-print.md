# format_param() / prints NULL and atomic vectors in R syntax

    Code
      format_param(NULL)
    Output
      [1] "NULL"
    Code
      format_param(1L)
    Output
      [1] "1"
    Code
      format_param(1.5)
    Output
      [1] "1.5"
    Code
      format_param(TRUE)
    Output
      [1] "TRUE"
    Code
      format_param("abc")
    Output
      [1] "\"abc\""
    Code
      format_param(c(1L, 2L, 3L))
    Output
      [1] "c(1, 2, 3)"
    Code
      format_param(c(1, 1e+06))
    Output
      [1] "c(1, 1e+06)"
    Code
      format_param(0.1234567890123)
    Output
      [1] "0.1234568"
    Code
      format_param(0.1234567890123, digits = 17)
    Output
      [1] "0.12345678901230001"
    Code
      format_param(c("a", "b"))
    Output
      [1] "c(\"a\", \"b\")"
    Code
      format_param(c(TRUE, FALSE))
    Output
      [1] "c(TRUE, FALSE)"
    Code
      format_param(integer())
    Output
      [1] "integer(0)"
    Code
      format_param(character())
    Output
      [1] "character(0)"

# format_param() / prints lists as list() calls

    Code
      format_param(list())
    Output
      [1] "list()"
    Code
      format_param(list(1, 2))
    Output
      [1] "list(1, 2)"
    Code
      format_param(list(a = 1, b = 2))
    Output
      [1] "list(a = 1, b = 2)"
    Code
      format_param(list(1, b = 2))
    Output
      [1] "list(1, b = 2)"
    Code
      format_param(list(a = NULL, b = 1))
    Output
      [1] "list(a = NULL, b = 1)"
    Code
      format_param(list(a = list(b = c(2, 3, 4))))
    Output
      [1] "list(a = list(b = c(2, 3, 4)))"

# format_param() / prints a data type under its anvl name

    Code
      format_param(as_dtype("f32"))
    Output
      [1] "f32"
    Code
      format_param(as_dtype("bool"))
    Output
      [1] "bool"

# format_param() / shows an array's value only when it holds one element

    Code
      format_param(nv_scalar(2.5, dtype = "f32"))
    Output
      [1] "2.5:f32"
    Code
      format_param(nv_scalar(TRUE))
    Output
      [1] "TRUE:bool"
    Code
      format_param(nv_array(array(1.5, dim = c(1, 1)), dtype = "f32"))
    Output
      [1] "1.5:f32[1,1]"
    Code
      format_param(nv_array(matrix(1:6, nrow = 2), dtype = "i32"))
    Output
      [1] "i32[2,3]"

# format_param() / prints a graph in full

    Code
      cat(format_param(g))
    Output
      (%x1: f32[], %x2: i32[3]) {
        %1: f32[3] = convert [dtype = f32] (%x2)
        %2: f32[3] = broadcast_in_axes [shape = 3, broadcast_axes = integer(0)] (%x1)
        %3: f32[3] = add(%2, %1)
        return %3
      }

# format_param() / names an unknown object by its class instead of deparsing it

    Code
      format_param(function(x) x)
    Output
      [1] "<function>"
    Code
      format_param(new.env())
    Output
      [1] "<environment>"
    Code
      format_param(quote(x + y))
    Output
      [1] "<call>"
    Code
      format_param(structure(list(secret = "hidden"), class = "SomeObject"))
    Output
      [1] "<SomeObject>"

# format_param_parts() / prefixes the elements that have a name

    Code
      format_param_parts(list(axis = 1, drop = TRUE))
    Output
      [1] "axis = 1"    "drop = TRUE"
    Code
      format_param_parts(list(1, 2))
    Output
      [1] "1" "2"
    Code
      format_param_parts(list(1, drop = TRUE))
    Output
      [1] "1"           "drop = TRUE"

# format.PrimitiveCall() / renders its params the way a graph body does

    Code
      cat(format(graph$calls[[1L]]))
    Output
      reduce_max(i32[10]) [axes = 1, drop = TRUE] -> i32[]

# format.PrimitiveCall() / leaves out the bracket list of a call that carries no params

    Code
      cat(format(graph$calls[[1L]]))
    Output
      add(f32[], f32[]) -> f32[]

# format.PrimitiveCall() / keeps a sub-graph param to its signature, having no graph to name it against

    Code
      cat(format(call))
    Output
      while(f32[]) [cond_graph = (f32[]) -> bool[], body_graph = (f32[]) -> f32[]] -> f32[]

# format.AnvlGraph() / shows literals, constants, params, captures and nested sub-graphs

    Code
      nested_graph()
    Output
      <AnvlGraph> [%c1: f32[], %c2: f32[]] (%x1: f32[]) {
        %1: f32[] = convert [dtype = f32] (2:i32)
        %2: f32[] = mul(%x1, %1)
        %3: f32[] = while [
          cond_graph = [%x1] (%x2: f32[]) {
            %5: bool[] = less(%x2, %x1)
            return %5
          },
          body_graph = [%x1, %2, %c1] (%x3: f32[]) {
            %6: bool[] = less(%x3, %2)
            %7: f32[] = if [
              true_graph = [%x3, %2] () {
                %8: f32[] = add(%x3, %2)
                return %8
              },
              false_graph = [%x3, %2, %c1] () {
                %9: f32[] = add(%x3, %c1)
                return %9
              }
            ] (%6)
            return %7
          }
        ] (%c2)
        %4: f32[2,1] = broadcast_in_axes [
          shape = c(2, 1), broadcast_axes = integer(0)
        ] (%3)
        return %4
      }

# format.AnvlGraph() / leaves out the bracket list of a graph that captures nothing

    Code
      graph
    Output
      <AnvlGraph> (%x1: f32[]) {
        %1: f32[] = while [
          cond_graph = (%x2: f32[]) {
            %2: bool[] = less(%x2, 9:f32)
            return %2
          },
          body_graph = (%x3: f32[]) {
            %3: f32[] = add(%x3, 1:f32)
            return %3
          }
        ] (%x1)
        return %1
      }

# format.AnvlGraph() / has nothing between the signature and the return when there are no calls

    Code
      trace_fn(identity, list(x = nv_scalar(1, dtype = "f32")))
    Output
      <AnvlGraph> (%x1: f32[]) {
        return %x1
      }

# format.AnvlGraph() / gives a sub-graph param its own rows and fills the short ones around it

    Code
      cat(format(graph, width = 80L))
    Output
      <AnvlGraph> (%x1: f32[6]) {
        %1: f32[] = reduce [
          axes = 1, drop = TRUE,
          reductor_graph = (%x2: f32[], %x3: f32[]) {
            %2: f32[] = add(%x2, %x3)
            return %2
          }
        ] (%x1, 0:f32)
        return %1
      }

# format.AnvlGraph() / returns every output of a graph that has more than one

    Code
      graph
    Output
      <AnvlGraph> (%x1: f32[2]) {
        %1: f32[2] = broadcast_in_axes [
          shape = 2, broadcast_axes = integer(0)
        ] (1:f32)
        %2: f32[2] = add(%x1, %1)
        %3: f32[2] = broadcast_in_axes [
          shape = 2, broadcast_axes = integer(0)
        ] (2:f32)
        %4: f32[2] = mul(%x1, %3)
        return (%2, %4)
      }

# format.AnvlGraph() / names the fill an optimization pass makes of a constant

    Code
      graph
    Output
      <AnvlGraph> (%x1: f32[]) {
        %1: f32[] = add(%x1, %2)
        %2: f32[] = fill [value = 7:f32, dtype = f32, shape = integer(0)] ()
        return %1
      }

# format.AnvlGraph() / names the R type of an input the caller supplies as bare R data

    Code
      graph
    Output
      <AnvlGraph> (%x1: f64[], %x2: f64[] <- double, %x3: i32[2] <- integer) {
        %1: f64[] = add(%x1, %x2)
        %2: f64[2] = convert [dtype = f64] (%x3)
        %3: f64[2] = broadcast_in_axes [shape = 2, broadcast_axes = integer(0)] (%1)
        %4: f64[2] = add(%3, %2)
        return %4
      }

# format.AnvlGraph() / breaks a call line too long for the width into filled param rows

    Code
      cat(format(gather_graph(), width = 80L))
    Output
      <AnvlGraph> [%c1: i32[1]] (%x1: f32[3,4], %x2: i32[2]) {
        %1: i32[2,1] = broadcast_in_axes [shape = c(2, 1), broadcast_axes = 1] (%x2)
        %2: i32[2,1] = broadcast_in_axes [shape = c(2, 1), broadcast_axes = 2] (%c1)
        %3: i32[2,2] = concatenate [axis = 2] (%1, %2)
        %4: f32[2,4] = gather [
          slice_sizes = c(1, 4), offset_axes = 2, collapsed_slice_axes = 1,
          x_batching_axes = integer(0), start_indices_batching_axes = integer(0),
          start_index_map = c(1, 2), index_vector_axis = 2,
          indices_are_sorted = FALSE, unique_indices = FALSE
        ] (%x1, %3)
        return %4
      }

# format.AnvlGraph() / breaks a call's operand list, which is as much a list as its params

    Code
      cat(body_section(graph, width = 80L), sep = "\n")
    Output
        %1: f32[90] = concatenate [axis = 1] (
          %x1, %x2, %x3, %x4, %x5, %x6, %x7, %x8, %x9, %x10, %x11, %x12, %x13, %x14,
          %x15, %x16, %x17, %x18, %x19, %x20, %x21, %x22, %x23, %x24, %x25, %x26,
          %x27, %x28, %x29, %x30
        )

# format.AnvlGraph() / breaks the ids and the types of a call with more outputs than fit

    Code
      cat(body_section(graph, width = 80L), sep = "\n")
    Output
        (%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12): (
          f32[3], f32[3], f32[3], f32[3], f32[3], f32[3], f32[3], f32[3], f32[3],
          f32[3], f32[3], f32[3]
        ) = sort [axis = 1, descending = FALSE, is_stable = FALSE] (
          %x1, %x2, %x3, %x4, %x5, %x6, %x7, %x8, %x9, %x10, %x11, %x12
        )

# format.AnvlGraph() / breaks a list of one element, which is a list like any other

    Code
      cat(format(graph, width = 40L))
    Output
      <AnvlGraph> (%x1: f32[2,3]) {
        %1: f32[3,2] = reshape [
          shape = c(3, 2)
        ] (%x1)
        return %1
      }

# format.GraphDescriptor() / prints the graph a trace has built so far

    Code
      descriptor
    Output
      <GraphDescriptor> (%x1: f32[]) {
        %1: f32[] = add(%x1, 1:f32)
        return %1
      }

