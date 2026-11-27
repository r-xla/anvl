# These exercise nv_custom_call() against the handlers {pjrt} registers when it
# loads, so no compiler is needed here.

describe("nv_custom_call", {
  it("calls a multi-result handler", {
    A <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
    out <- nv_custom_call(
      "eigh",
      A,
      # eigh returns (vectors, values) and, being LAPACK, wants column-major
      # buffers: R's own order, the default layout
      output_types = list(AbstractArray("f64", c(2, 2)), AbstractArray("f64", 2))
    )
    expect_length(out, 2L)
    ref <- nv_eigh(A)
    expect_equal(as_array(out[[1L]]), as_array(ref$vectors))
    expect_equal(as_array(out[[2L]]), as_array(ref$values))
  })

  it("unwraps a single result", {
    A <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
    packed <- nv_custom_call(
      "geqrf",
      A,
      output_types = list(AbstractArray("f64", c(2, 2)), AbstractArray("f64", 2))
    )
    # a bare AbstractArray (not a list of them) comes back as a bare array
    Q <- nv_custom_call(
      "orgqr",
      packed[[1L]],
      packed[[2L]],
      output_types = AbstractArray("f64", c(2, 2))
    )
    expect_s3_class(Q, "AnvlArray")
    expect_equal(abs(as_array(Q)), abs(as_array(nv_qr(A)$Q)))
  })

  it("passes attributes and returns the operand of a side-effect call", {
    x <- nv_array(c(1, 2, 3), dtype = "f32")
    expect_snapshot(
      out <- nv_custom_call(
        "print_tensor",
        x,
        attrs = list(print_header = "my array", print_footer = "----")
      )
    )
    expect_equal(as_array(out), as_array(x))
  })

  it("hands the handler the layout it asks for", {
    # eigh reads one triangle, so a matrix whose triangles differ tells the
    # default column-major layout from a row-major one, which transposes it
    A <- nv_matrix(c(2, 5, 1, 2), nrow = 2, dtype = "f64")
    eigh_values <- function(layout) {
      nv_custom_call(
        "eigh",
        A,
        output_types = list(AbstractArray("f64", c(2, 2)), AbstractArray("f64", 2)),
        operand_layouts = layout
      )[[2L]]
    }
    col_major <- as_array(eigh_values(NULL))
    row_major <- as_array(eigh_values(list(2:1)))
    expect_equal(col_major, as_array(nv_eigh(A)$values))
    expect_equal(row_major, as_array(nv_eigh(nv_transpose(A))$values))
    expect_false(isTRUE(all.equal(col_major, row_major)))
  })

  it("works inside jit()", {
    f <- jit(function(A) {
      nv_custom_call(
        "eigh",
        A,
        output_types = list(AbstractArray("f64", c(2, 2)), AbstractArray("f64", 2))
      )[[2L]]
    })
    A <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
    expect_equal(as_array(f(A)), as_array(nv_eigh(A)$values))
  })

  it("recompiles when the declared output types change", {
    A <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
    call_eigh <- function(value_shape) {
      nv_custom_call(
        "eigh",
        A,
        output_types = list(AbstractArray("f64", c(2, 2)), AbstractArray("f64", value_shape))
      )
    }
    expect_equal(shape(call_eigh(2L)[[2L]]), 2L)
    # the output types are not derivable from the input, so they have to be
    # part of the compilation cache key: a program served from the cache would
    # return 2 values, a new one hands the handler a buffer it rejects
    expect_error(call_eigh(1L), "eigh requires W")
  })

  it("takes layouts per platform", {
    A <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
    out <- nv_custom_call(
      "eigh",
      A,
      output_types = list(AbstractArray("f64", c(2, 2)), AbstractArray("f64", 2)),
      # eigh wants column-major on both platforms; the point here is that the
      # per-platform spelling resolves to the right entry
      operand_layouts = list(cpu = list(1:2), cuda = list(1:2)),
      result_layouts = list(cpu = list(1:2, 1L), cuda = list(1:2, 1L))
    )
    expect_equal(as_array(out[[2L]]), as_array(nv_eigh(A)$values))
  })

  it("resolves per-platform layouts at lowering time", {
    # R axes, minor to major, become XLA dimensions of the reversed shape
    spec <- list(cpu = list(2:1), cuda = list(1:2))
    local_platform("cuda")
    expect_equal(custom_call_layouts(spec, 2L, "operand_layouts"), list(c(1L, 0L)))
    local_platform("cpu")
    expect_equal(custom_call_layouts(spec, 2L, "operand_layouts"), list(c(0L, 1L)))
    # a uniform spec applies on every platform
    expect_equal(custom_call_layouts(list(2:1), 2L, "x"), list(c(0L, 1L)))
    # no spec is R's column-major order, XLA's default layout
    expect_equal(custom_call_layouts(NULL, c(2L, 1L, 0L), "x"), list(c(1L, 0L), 0L, integer()))

    local_platform("cuda")
    expect_error(
      custom_call_layouts(list(cpu = list(1L)), 1L, "operand_layouts"),
      "no entry for platform"
    )
  })

  it("reports an unregistered target", {
    x <- nv_array(c(1, 2), dtype = "f64")
    expect_error(
      nv_custom_call("no_such_handler", x, output_types = AbstractArray("f64", 2)),
      "no_such_handler"
    )
  })

  it("validates its arguments", {
    x <- nv_array(c(1, 2), dtype = "f64")
    expect_error(nv_custom_call("t", x, output_types = "nope"), "Must be of type")
    expect_error(
      nv_custom_call("t", x, output_types = AbstractArray("f64", 2), operand_layouts = list()),
      "one entry per operand"
    )
    expect_error(
      nv_custom_call("t", x, output_types = AbstractArray("f64", 2), result_layouts = list()),
      "one entry per result"
    )
    expect_error(
      nv_custom_call(
        "t",
        x,
        output_types = AbstractArray("f64", 2),
        operand_layouts = list(cpu = list(1L, 1L), cuda = list(1L))
      ),
      "for platform \"cpu\" must have one entry per operand"
    )
    # XLA's 0-based dimensions are not R axes
    expect_error(
      nv_custom_call(
        "t",
        nv_matrix(c(1, 2, 3, 4), nrow = 2),
        output_types = AbstractArray("f64", 2),
        operand_layouts = list(c(0L, 1L))
      ),
      "must order the 2 axes of operand 1"
    )
    expect_error(
      nv_custom_call("t", x, output_types = AbstractArray("f64", 2), aliases = c(1L, 2L)),
      "one entry per result"
    )
    expect_error(
      nv_custom_call("t", x, output_types = AbstractArray("f64", 2), aliases = 7L),
      "between 1 and 1"
    )
    expect_error(nv_custom_call("t"), "needs at least one")
    expect_error(
      nv_custom_call("t", x, attrs = list(bad = list(1))),
      "non-empty atomic vector"
    )
  })
})
