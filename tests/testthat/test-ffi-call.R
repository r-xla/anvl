# These exercise nv_ffi_call() against the handlers {pjrt} registers when it
# loads, so no compiler is needed here.

describe("nv_ffi_call", {
  it("calls a multi-result handler", {
    A <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
    out <- nv_ffi_call(
      "eigh",
      A,
      # eigh returns (vectors, values) and wants column-major buffers
      output_types = list(vt("f64", c(2, 2)), vt("f64", 2)),
      operand_layouts = list(c(0L, 1L)),
      result_layouts = list(c(0L, 1L), 0L)
    )
    expect_length(out, 2L)
    ref <- nv_eigh(A)
    expect_equal(as_array(out[[1L]]), as_array(ref$vectors))
    expect_equal(as_array(out[[2L]]), as_array(ref$values))
  })

  it("unwraps a single result", {
    A <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
    packed <- nv_ffi_call(
      "geqrf",
      A,
      output_types = list(vt("f64", c(2, 2)), vt("f64", 2)),
      operand_layouts = list(c(0L, 1L)),
      result_layouts = list(c(0L, 1L), 0L)
    )
    # a bare ValueType (not a list of them) comes back as a bare array
    Q <- nv_ffi_call(
      "orgqr",
      packed[[1L]],
      packed[[2L]],
      output_types = vt("f64", c(2, 2)),
      operand_layouts = list(c(0L, 1L), 0L),
      result_layouts = list(c(0L, 1L))
    )
    expect_s3_class(Q, "AnvlArray")
    expect_equal(abs(as_array(Q)), abs(as_array(nv_qr(A)$Q)))
  })

  it("passes attributes and returns the operand of a side-effect call", {
    x <- nv_array(c(1, 2, 3), dtype = "f32")
    expect_snapshot(
      out <- nv_ffi_call(
        "print_tensor",
        x,
        attrs = list(print_header = "my array", print_footer = "----")
      )
    )
    expect_equal(as_array(out), as_array(x))
  })

  it("defaults to row-major layouts", {
    x <- nv_array(matrix(1:6, nrow = 2), dtype = "f32")
    out <- nv_ffi_call("print_tensor", x, attrs = list(print_footer = ""))
    expect_equal(as_array(out), as_array(x))
  })

  it("works inside jit()", {
    f <- jit(function(A) {
      nv_ffi_call(
        "eigh",
        A,
        output_types = list(vt("f64", c(2, 2)), vt("f64", 2)),
        operand_layouts = list(c(0L, 1L)),
        result_layouts = list(c(0L, 1L), 0L)
      )[[2L]]
    })
    A <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
    expect_equal(as_array(f(A)), as_array(nv_eigh(A)$values))
  })

  it("recompiles when the declared output types change", {
    A <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
    call_eigh <- function(value_shape) {
      nv_ffi_call(
        "eigh",
        A,
        output_types = list(vt("f64", c(2, 2)), vt("f64", value_shape)),
        operand_layouts = list(c(0L, 1L)),
        result_layouts = list(c(0L, 1L), 0L)
      )
    }
    expect_equal(shape(call_eigh(2L)[[2L]]), 2L)
    # the output types are not derivable from the input, so they have to be
    # part of the compilation cache key
    expect_equal(shape(call_eigh(1L)[[2L]]), 1L)
  })

  it("reports an unregistered target", {
    x <- nv_array(c(1, 2), dtype = "f64")
    expect_error(
      nv_ffi_call("no_such_handler", x, output_types = vt("f64", 2)),
      "no_such_handler"
    )
  })

  it("validates its arguments", {
    x <- nv_array(c(1, 2), dtype = "f64")
    expect_error(nv_ffi_call("t", x, output_types = "nope"), "Must be of type")
    expect_error(
      nv_ffi_call("t", x, output_types = vt("f64", 2), operand_layouts = list()),
      "one entry per operand"
    )
    expect_error(
      nv_ffi_call("t", x, output_types = vt("f64", 2), result_layouts = list()),
      "one entry per result"
    )
    expect_error(
      nv_ffi_call("t", x, output_types = vt("f64", 2), aliases = c(1L, 2L)),
      "one entry per result"
    )
    expect_error(
      nv_ffi_call("t", x, output_types = vt("f64", 2), aliases = 7L),
      "between 1 and 1"
    )
    expect_error(nv_ffi_call("t"), "needs at least one")
    expect_error(
      nv_ffi_call("t", x, attrs = list(bad = list(1))),
      "non-empty atomic vector"
    )
  })
})
