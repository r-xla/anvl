# These exercise nv_custom_call() against the handlers {pjrt} registers when it
# loads, so no compiler is needed here.

describe("nv_custom_call", {
  it("calls a multi-result handler", {
    A <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
    out <- nv_custom_call(
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
    packed <- nv_custom_call(
      "geqrf",
      A,
      output_types = list(vt("f64", c(2, 2)), vt("f64", 2)),
      operand_layouts = list(c(0L, 1L)),
      result_layouts = list(c(0L, 1L), 0L)
    )
    # a bare ValueType (not a list of them) comes back as a bare array
    Q <- nv_custom_call(
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
      out <- nv_custom_call(
        "print_tensor",
        x,
        attrs = list(print_header = "my array", print_footer = "----")
      )
    )
    expect_equal(as_array(out), as_array(x))
  })

  it("defaults to row-major layouts", {
    x <- nv_array(matrix(1:6, nrow = 2), dtype = "f32")
    out <- nv_custom_call("print_tensor", x, attrs = list(print_footer = ""))
    expect_equal(as_array(out), as_array(x))
  })

  it("works inside jit()", {
    f <- jit(function(A) {
      nv_custom_call(
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
      nv_custom_call(
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

  it("takes layouts per platform", {
    A <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
    out <- nv_custom_call(
      "eigh",
      A,
      output_types = list(vt("f64", c(2, 2)), vt("f64", 2)),
      # eigh wants column-major on both platforms; the point here is that the
      # per-platform spelling resolves to the right entry
      operand_layouts = list(
        cpu = list(c(0L, 1L)),
        cuda = list(c(0L, 1L))
      ),
      result_layouts = list(
        cpu = list(c(0L, 1L), 0L),
        cuda = list(c(0L, 1L), 0L)
      )
    )
    expect_equal(as_array(out[[2L]]), as_array(nv_eigh(A)$values))
  })

  it("resolves per-platform layouts at lowering time", {
    spec <- list(cpu = list(c(1L, 0L)), cuda = list(c(0L, 1L)))
    local_platform("cuda")
    expect_equal(custom_call_layouts(spec, "operand_layouts"), list(c(0L, 1L)))
    local_platform("cpu")
    expect_equal(custom_call_layouts(spec, "operand_layouts"), list(c(1L, 0L)))
    # a uniform spec is passed through untouched
    expect_equal(custom_call_layouts(list(c(1L, 0L)), "x"), list(c(1L, 0L)))
    expect_null(custom_call_layouts(NULL, "x"))

    local_platform("cuda")
    expect_error(
      custom_call_layouts(list(cpu = list(0L)), "operand_layouts"),
      "no entry for platform"
    )
  })

  it("reports an unregistered target", {
    x <- nv_array(c(1, 2), dtype = "f64")
    expect_error(
      nv_custom_call("no_such_handler", x, output_types = vt("f64", 2)),
      "no_such_handler"
    )
  })

  it("validates its arguments", {
    x <- nv_array(c(1, 2), dtype = "f64")
    expect_error(nv_custom_call("t", x, output_types = "nope"), "Must be of type")
    expect_error(
      nv_custom_call("t", x, output_types = vt("f64", 2), operand_layouts = list()),
      "one entry per operand"
    )
    expect_error(
      nv_custom_call("t", x, output_types = vt("f64", 2), result_layouts = list()),
      "one entry per result"
    )
    expect_error(
      nv_custom_call(
        "t",
        x,
        output_types = vt("f64", 2),
        operand_layouts = list(cpu = list(0L, 0L), cuda = list(0L))
      ),
      "for platform \"cpu\" must have one entry per operand"
    )
    expect_error(
      nv_custom_call("t", x, output_types = vt("f64", 2), aliases = c(1L, 2L)),
      "one entry per result"
    )
    expect_error(
      nv_custom_call("t", x, output_types = vt("f64", 2), aliases = 7L),
      "between 1 and 1"
    )
    expect_error(nv_custom_call("t"), "needs at least one")
    expect_error(nv_custom_call(list("a"), x, output_types = vt("f64", 2)), "named by platform")
    expect_error(nv_custom_call(list(cpu = 1), x, output_types = vt("f64", 2)), "named by platform")
    expect_error(
      nv_custom_call("t", x, attrs = list(bad = list(1))),
      "non-empty atomic vector"
    )
  })
})

scale_module <- pjrt::pjrt_cuda_module(
  r"(
template <typename T>
__global__ void scale(const T *x, T *out, T a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a * x[i];
}
)",
  kernels = c("scale<float>", "scale<double>")
)

nv_scale <- function(x, a) {
  n <- as.integer(prod(shape(x)))
  ctype <- switch(as.character(dtype(x)), f32 = "float", f64 = "double")
  a <- if (ctype == "float") pjrt::pjrt_cuda_scalar(a, "f32") else a
  nv_custom_call(
    cuda_kernel(
      scale_module,
      sprintf("scale<%s>", ctype),
      grid = ceiling(n / 128),
      block = 128L,
      scalars = list(a, n)
    ),
    x,
    output_types = vt(dtype(x), shape(x))
  )
}

describe("cuda_kernel", {
  it("describes a launch", {
    k <- cuda_kernel(scale_module, "scale<float>", grid = c(2, 3), block = 64L, scalars = list(1L))
    expect_s3_class(k, "AnvlCudaKernel")
    expect_identical(k$attrs$kernel, "scale<float>")
    expect_identical(k$attrs$grid_y, 3L)
    expect_output(print(k), "grid \\(2, 3, 1\\), block \\(64, 1, 1\\)")
  })

  it("lowers to the pjrt_cuda_kernel custom call", {
    k <- cuda_kernel(scale_module, "scale<float>", grid = 1L, block = 32L, scalars = list(4L))
    local_platform("cuda")
    expect_identical(custom_call_target(k), k)
    expect_identical(custom_call_target(list(cpu = "a", cuda = k)), k)
    local_platform("cpu")
    expect_identical(custom_call_target(list(cpu = "a", cuda = k)), "a")
    expect_error(custom_call_target(k), "runs only on CUDA")
    expect_error(custom_call_target(list(cuda = k)), "no entry for platform \"cpu\"")
  })

  it("picks the target of the platform the program is compiled for", {
    skip_if(is_cuda())
    k <- cuda_kernel(scale_module, "scale<float>", grid = 1L, block = 32L, scalars = list(1, 1L))
    x <- nv_array(c(1, 2, 3), dtype = "f32")
    expect_output(
      nv_custom_call(list(cpu = "print_tensor", cuda = k), x, attrs = list(print_header = "on the cpu")),
      "on the cpu"
    )
    expect_error(nv_scale(x, 2), "runs only on CUDA")
  })

  it("runs a kernel, eagerly and under jit()", {
    skip_if(!is_cuda())
    x <- nv_array(c(1, 2, 3), dtype = "f32")
    expect_equal(as.vector(as_array(nv_scale(x, 2))), c(2, 4, 6))
    f <- jit(function(x) nv_sum(nv_scale(x, 0.5)))
    expect_equal(as_array(f(nv_array(c(2, 4), dtype = "f64"))), 3)
    expect_equal(as_array(f(nv_array(c(2, 4), dtype = "f32"))), 3)
  })

  it("takes no attrs, unless they are for the handlers of other platforms", {
    skip_if(!is_cuda())
    k <- cuda_kernel(
      scale_module,
      "scale<float>",
      grid = 1L,
      block = 32L,
      scalars = list(pjrt::pjrt_cuda_scalar(3, "f32"), 1L)
    )
    x <- nv_array(2, dtype = "f32")
    expect_error(
      nv_custom_call(k, x, output_types = vt("f32", 1L), attrs = list(a = 1L)),
      "not to CUDA kernels"
    )
    out <- nv_custom_call(list(cpu = "some_handler", cuda = k), x, output_types = vt("f32", 1L), attrs = list(a = 1L))
    expect_equal(as.vector(as_array(out)), 6)
  })
})
