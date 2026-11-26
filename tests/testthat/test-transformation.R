describe("transformation_fn", {
  it("does not let a formal of f shadow what it uses", {
    f <- function(x, impl, args) sum(x * x)
    g <- transformation_fn(f, function(args) names(args))
    expect_identical(g(1, impl = 2, args = 3), c("x", "impl", "args"))
  })

  it("evaluates the arguments in the caller", {
    g <- transformation_fn(function(x) x, function(args) args$x)
    h <- function() {
      y <- 5
      g(x = y + 1)
    }
    expect_identical(h(), 6)
  })
})

describe("carry_closure", {
  # a loop body over the carry (a, b, c) and a value w after it: a depends on
  # w, b on a, and c only on itself
  avals <- list(a = nv_aval("f64", integer()), b = nv_aval("f64", integer()), c = nv_aval("f64", integer()))
  body <- trace_fn(
    function(a, b, c, w) list(a = a * w, b = b + a, c = c + 1),
    c(avals, list(w = nv_aval("f64", integer())))
  )

  it("flags what the body makes depend on a flagged value, until that adds no more", {
    flags <- carry_closure(body, c(FALSE, FALSE, FALSE), TRUE, graph_requiring_outputs)
    expect_equal(flags, c(TRUE, TRUE, FALSE))
  })

  it("keeps what enters flagged", {
    flags <- carry_closure(body, c(FALSE, FALSE, TRUE), FALSE, graph_requiring_outputs)
    expect_equal(flags, c(FALSE, FALSE, TRUE))
  })
})
