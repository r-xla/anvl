# nv_custom_call / passes attributes and returns the operand of a side-effect call

    Code
      out <- nv_custom_call("print_tensor", x, attrs = list(print_header = "my array",
        print_footer = "----"))
    Output
      my array
       1
       2
       3
      ----

