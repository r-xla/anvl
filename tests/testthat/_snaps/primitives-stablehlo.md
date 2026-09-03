# prim_print

    Code
      out <<- f(x)
    Output
      AnvlArray
       1
       2
       3
      [ f32{3} ]

# prim_print shows the R type where a value has no data type yet

    Code
      invisible(g(1))
    Output
      AnvlArray
       1
      [ double{} printed at f32 ]

---

    Code
      invisible(h(matrix(1:4, nrow = 2)))
    Output
      AnvlArray
       1 3
       2 4
      [ integer{2,2} printed at i32 ]

