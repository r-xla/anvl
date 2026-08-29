# literals

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: f32[]
        Body:
          %1: f32[] = convert [dtype = f32, ambiguous = FALSE] (1:i32?)
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
          %1: f32[2, 1] = fill [value = 1, dtype = f32, shape = c(2, 1), ambiguous = FALSE] ()
        Outputs:
          %1: f32[2, 1] 

# ambiguity is printed via ?

    Code
      graph
    Output
      <AnvlGraph>
        Inputs:
          %x1: i1[]
        Body:
          %1: f32?[] = convert [dtype = f32, ambiguous = TRUE] (%x1)
          %2: f32?[] = mul(%1, 1:f32?)
        Outputs:
          %2: f32?[] 

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

# format_param: empty cases collapse to empty string

    Code
      format_param(NULL)
    Output
      [1] ""
    Code
      format_param(list())
    Output
      [1] ""

# format_param: atomic scalars

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

# format_param: atomic vectors

    Code
      format_param(c(1L, 2L, 3L))
    Output
      [1] "c(1, 2, 3)"
    Code
      format_param(c("a", "b"))
    Output
      [1] "c(\"a\", \"b\")"
    Code
      format_param(c(TRUE, FALSE))
    Output
      [1] "c( TRUE, FALSE)"

# format_param: empty atomic vectors show typeof(0)

    Code
      format_param(integer())
    Output
      [1] "integer(0)"
    Code
      format_param(character())
    Output
      [1] "character(0)"
    Code
      format_param(logical())
    Output
      [1] "logical(0)"

# format_param: lists

    Code
      format_param(list(1, 2))
    Output
      [1] "[1, 2]"
    Code
      format_param(list(a = 1, b = 2))
    Output
      [1] "[a = 1, b = 2]"

# format_param: NULL nested in a list is printed as NULL

    Code
      format_param(list(NULL, 1))
    Output
      [1] "[NULL, 1]"
    Code
      format_param(list(a = NULL, b = 1))
    Output
      [1] "[a = NULL, b = 1]"

# format_param: nested lists

    Code
      format_param(list(list(x = 1), 2))
    Output
      [1] "[[x = 1], 2]"
    Code
      format_param(list(list(c(1, 2, 3))))
    Output
      [1] "[[c(1, 2, 3)]]"
    Code
      format_param(list(a = list(b = c(2, 3, 4))))
    Output
      [1] "[a = [b = c(2, 3, 4)]]"

# format_param: dtype uses repr

    Code
      format_param(as_dtype("f32"))
    Output
      [1] "f32"
    Code
      format_param(as_dtype("i32"))
    Output
      [1] "i32"

# format_param: graph is summarized by input/output count

    Code
      format_param(g)
    Output
      [1] "graph[1 -> 1]"

