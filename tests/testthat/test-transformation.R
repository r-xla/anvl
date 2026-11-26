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
