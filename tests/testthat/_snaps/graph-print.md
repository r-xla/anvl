# literals

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: f32[]
        Body:
          %1: f32[] = convert [dtype = f32] (1:i32[])
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
          %x1: bool[]
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

# format_param_parts: a call with no parameters has no parts

    Code
      format_param_parts(NULL)
    Output
      character(0)
    Code
      format_param_parts(list())
    Output
      character(0)

# format_param_value: atomic scalars

    Code
      format_param_value(1L)
    Output
      [1] "1"
    Code
      format_param_value(1.5)
    Output
      [1] "1.5"
    Code
      format_param_value(TRUE)
    Output
      [1] "TRUE"
    Code
      format_param_value("abc")
    Output
      [1] "\"abc\""

# format_param_value: atomic vectors

    Code
      format_param_value(c(1L, 2L, 3L))
    Output
      [1] "c(1, 2, 3)"
    Code
      format_param_value(c("a", "b"))
    Output
      [1] "c(\"a\", \"b\")"
    Code
      format_param_value(c(TRUE, FALSE))
    Output
      [1] "c(TRUE, FALSE)"

# format_param_value: empty atomic vectors show typeof(0)

    Code
      format_param_value(integer())
    Output
      [1] "integer(0)"
    Code
      format_param_value(character())
    Output
      [1] "character(0)"
    Code
      format_param_value(logical())
    Output
      [1] "logical(0)"

# format_param_value: lists

    Code
      format_param_value(list(1, 2))
    Output
      [1] "[1, 2]"
    Code
      format_param_value(list(a = 1, b = 2))
    Output
      [1] "[a = 1, b = 2]"

# format_param_value: NULL nested in a list is printed as NULL

    Code
      format_param_value(list(NULL, 1))
    Output
      [1] "[NULL, 1]"
    Code
      format_param_value(list(a = NULL, b = 1))
    Output
      [1] "[a = NULL, b = 1]"

# format_param_value: nested lists

    Code
      format_param_value(list(list(x = 1), 2))
    Output
      [1] "[[x = 1], 2]"
    Code
      format_param_value(list(list(c(1, 2, 3))))
    Output
      [1] "[[c(1, 2, 3)]]"
    Code
      format_param_value(list(a = list(b = c(2, 3, 4))))
    Output
      [1] "[a = [b = c(2, 3, 4)]]"

# format_param_value: dtype prints under its anvl name

    Code
      format_param_value(as_dtype("f32"))
    Output
      [1] "f32"
    Code
      format_param_value(as_dtype("i32"))
    Output
      [1] "i32"

# format_param_value: graph is summarized by input/output count

    Code
      format_param_value(g)
    Output
      [1] "graph[1 -> 1]"

# an input the caller supplies as bare R data names its R type

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: f64[]
          %x2: f64[] <- double
        Body:
          %1: f64[] = add(%x1, %x2)
        Outputs:
          %1: f64[] 

---

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: f32[]
          %x2: i32[2] <- integer
        Body:
          %1: f32[2] = convert [dtype = f32] (%x2)
          %2: f32[2] = broadcast_in_axes [shape = 2, broadcast_axes = integer(0)] (%x1)
          %3: f32[2] = add(%2, %1)
        Outputs:
          %3: f32[2] 

# format_param_value: named atomic vectors keep their names

    Code
      format_param_value(c(a = 1, b = 2))
    Output
      [1] "c(a = 1, b = 2)"
    Code
      format_param_value(c(a = 1L))
    Output
      [1] "c(a = 1)"

# format_param_value: character values are escaped

    Code
      format_param_value("a\"b")
    Output
      [1] "\"a\\\"b\""
    Code
      format_param_value("a\\b")
    Output
      [1] "\"a\\\\b\""

# format_param_value: a one-element array parameter prints its value

    Code
      format_param_value(nv_scalar(1, dtype = "f32"))
    Output
      [1] "1:f32[]"
    Code
      format_param_value(nv_array(1, shape = c(1, 1), dtype = "f32"))
    Output
      [1] "1:f32[1, 1]"
    Code
      format_param_value(nv_array(c(1, 2, 3), dtype = "f32"))
    Output
      [1] "f32[3]"

# a folded constant prints its value in the `fill` it becomes

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: f32[]
        Body:
          %1: f32[] = mul(%x1, %2)
          %2: f32[] = fill [value = 2:f32[], dtype = f32, shape = integer(0)] ()
        Outputs:
          %1: f32[] 

# format_param_value: a partially named vector names only what has a name

    Code
      format_param_value(c(a = 1, 2))
    Output
      [1] "c(a = 1, 2)"
    Code
      format_param_value(stats::setNames(c(1, 2), c("", "b")))
    Output
      [1] "c(1, b = 2)"
    Code
      format_param_value(stats::setNames(1, ""))
    Output
      [1] "1"

# format_param_value: a partially named list names only what has a name

    Code
      format_param_value(list(a = 1, 2))
    Output
      [1] "[a = 1, 2]"
    Code
      format_param_parts(list(a = 1, 2))
    Output
      [1] "a = 1" "2"    

# format_param_value: each element of a vector is formatted on its own

    Code
      format_param_value(c(1, 2.5))
    Output
      [1] "c(1, 2.5)"
    Code
      format_param_value(c(1, 1e+10))
    Output
      [1] "c(1, 1e+10)"
    Code
      format_param_value(c(0.1, 1e-20))
    Output
      [1] "c(0.1, 1e-20)"

# a call whose parameters do not fit the width fills them over further lines

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: i64[]
        Constants:
          %c1: f32[3]
        Body:
          %1: i64[1] = reshape [shape = 1] (%x1)
          %2: i64[1] = concatenate [axis = 1] (%1)
          %3: f32[] = gather [slice_sizes = 1, offset_axes = integer(0), collapsed_slice_axes = 1,
            x_batching_axes = integer(0), start_indices_batching_axes = integer(0), start_index_map = 1,
            index_vector_axis = 1, indices_are_sorted = TRUE, unique_indices = TRUE] (%c1, %2)
        Outputs:
          %3: f32[] 

