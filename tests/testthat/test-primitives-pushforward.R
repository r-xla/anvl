# The pushforward rules are checked automatically: every primitive with a
# pushforward rule has an entry in `pushforward_cases`, and
# `check_pushforward_rule()` runs the same checks on each of them against
# references that do not use the pushforward rules. torch has no forward mode to
# compare with.

# Samplers for the values a case calls its primitive with, as a
# `function(shape)` returning an R array, or a number for the scalar shape.
unif <- function(lower, upper) {
  function(shape) shaped(runif(prod(shape), lower, upper), shape)
}
# Away from the integers and the half-integers, where a step function jumps.
off_step <- function(lower, upper) {
  function(shape) shaped(sample(lower:upper, prod(shape), replace = TRUE) + runif(prod(shape), 0.1, 0.4), shape)
}
coin <- function() {
  function(shape) shaped(runif(prod(shape)) < 0.5, shape)
}
fixed <- function(value) {
  function(shape) shaped(rep(value, prod(shape)), shape)
}
shaped <- function(values, shape) {
  if (length(shape)) array(values, shape) else values
}

# `x`, an R array or number, as an array of data type `dtype`.
anvl_array <- function(x, dtype) {
  if (is.null(dim(x))) nv_scalar(x, dtype = dtype) else nv_array(x, dtype = dtype)
}

# A call of a primitive: its arguments, as samplers -- a float sampler gives an
# array of the data type under test, a logical one a boolean array -- the
# arguments to differentiate with respect to, the arguments that are scalars,
# which the primitive broadcasts to the shape of the others, and static
# parameters.
pushforward_case <- function(args, wrt = names(args), scalar = character(), params = list()) {
  list(args = args, wrt = wrt, scalar = scalar, params = params)
}

binary_case <- function(lhs = unif(-2, 2), rhs = unif(-2, 2)) {
  pushforward_case(list(lhs = lhs, rhs = rhs))
}

pushforward_cases <- list(
  add = binary_case(),
  sub = binary_case(),
  mul = binary_case(),
  div = binary_case(rhs = unif(0.5, 2)),
  remainder = pushforward_case(list(x = unif(-5, 5), y = unif(0.5, 2))),
  pow = pushforward_case(list(x = unif(0.5, 2), y = unif(-2, 2))),
  negate = pushforward_case(list(x = unif(-2, 2))),
  abs = pushforward_case(list(x = unif(-2, 2))),
  pmax = binary_case(),
  pmin = binary_case(),
  clamp = pushforward_case(list(x = unif(-2, 2), min = fixed(-1), max = fixed(1)), wrt = "x"),
  ifelse = pushforward_case(list(test = coin(), yes = unif(-2, 2), no = unif(-2, 2)), wrt = c("yes", "no")),
  exp = pushforward_case(list(x = unif(-2, 2))),
  expm1 = pushforward_case(list(x = unif(-2, 2))),
  log = pushforward_case(list(x = unif(0.5, 3))),
  log1p = pushforward_case(list(x = unif(-0.5, 2))),
  plogis = pushforward_case(list(x = unif(-3, 3))),
  sqrt = pushforward_case(list(x = unif(0.5, 3))),
  rsqrt = pushforward_case(list(x = unif(0.5, 3))),
  cbrt = pushforward_case(list(x = unif(0.5, 3))),
  sin = pushforward_case(list(x = unif(-3, 3))),
  cos = pushforward_case(list(x = unif(-3, 3))),
  tan = pushforward_case(list(x = unif(-1, 1))),
  asin = pushforward_case(list(x = unif(-0.9, 0.9))),
  acos = pushforward_case(list(x = unif(-0.9, 0.9))),
  atan = pushforward_case(list(x = unif(-3, 3))),
  atan2 = pushforward_case(list(y = unif(-2, 2), x = unif(-2, 2))),
  sinh = pushforward_case(list(x = unif(-2, 2))),
  cosh = pushforward_case(list(x = unif(-2, 2))),
  tanh = pushforward_case(list(x = unif(-2, 2))),
  asinh = pushforward_case(list(x = unif(-2, 2))),
  acosh = pushforward_case(list(x = unif(1.5, 3))),
  atanh = pushforward_case(list(x = unif(-0.9, 0.9))),
  lgamma = pushforward_case(list(x = unif(0.5, 3))),
  digamma = pushforward_case(list(x = unif(0.5, 3))),
  psigamma = pushforward_case(list(x = unif(0.5, 3), deriv = fixed(1)), wrt = "x"),
  erf = pushforward_case(list(x = unif(-2, 2))),
  erfc = pushforward_case(list(x = unif(-2, 2))),
  erf_inv = pushforward_case(list(x = unif(-0.9, 0.9))),
  floor = pushforward_case(list(x = off_step(-3L, 2L))),
  ceiling = pushforward_case(list(x = off_step(-3L, 2L))),
  round = pushforward_case(list(x = off_step(-3L, 2L))),
  sign = pushforward_case(list(x = unif(-2, 2))),
  eq = binary_case(),
  ne = binary_case(),
  gt = binary_case(),
  ge = binary_case(),
  lt = binary_case(),
  le = binary_case(),
  is_finite = pushforward_case(list(x = unif(-2, 2)))
)

# Calls with a scalar operand, named `<primitive>/<scalar operands>`. The
# primitive broadcasts it, so its cotangent is summed and its tangent
# broadcast.
pushforward_scalar_cases <- list(
  "add/rhs" = pushforward_case(list(lhs = unif(-2, 2), rhs = unif(-2, 2)), scalar = "rhs"),
  "sub/lhs" = pushforward_case(list(lhs = unif(-2, 2), rhs = unif(-2, 2)), scalar = "lhs"),
  "mul/lhs" = pushforward_case(list(lhs = unif(-2, 2), rhs = unif(-2, 2)), scalar = "lhs"),
  "div/rhs" = pushforward_case(list(lhs = unif(-2, 2), rhs = unif(0.5, 2)), scalar = "rhs"),
  "pow/y" = pushforward_case(list(x = unif(0.5, 2), y = unif(-2, 2)), scalar = "y"),
  "remainder/y" = pushforward_case(list(x = unif(-5, 5), y = unif(0.5, 2)), scalar = "y"),
  "pmax/rhs" = pushforward_case(list(lhs = unif(-2, 2), rhs = unif(-2, 2)), scalar = "rhs"),
  "atan2/x" = pushforward_case(list(y = unif(-2, 2), x = unif(-2, 2)), scalar = "x"),
  "psigamma/deriv" = pushforward_case(list(x = unif(0.5, 3), deriv = fixed(1)), wrt = "x", scalar = "deriv"),
  "clamp/x" = pushforward_case(list(x = unif(-2, 2), min = fixed(-1), max = fixed(1)), wrt = "x", scalar = "x"),
  "ifelse/no" = pushforward_case(
    list(test = coin(), yes = unif(-2, 2), no = unif(-2, 2)),
    wrt = c("yes", "no"),
    scalar = "no"
  ),
  "eq/rhs" = pushforward_case(list(lhs = unif(-2, 2), rhs = unif(-2, 2)), scalar = "rhs")
)

# `body(args)`, for `args` a named list, as a function with one formal per name.
with_formals <- function(arg_names, body) {
  f <- function() body(mget(arg_names, envir = environment()))
  formals(f) <- stats::setNames(rep(list(quote(expr = )), length(arg_names)), arg_names)
  f
}

# The tangent of `fn` at the arguments `x` along `t`, a list with a direction
# for some of them.
tangent_along <- function(fn, x, t) {
  do.call(do.call(pushforward, c(list(fn), x)), t)
}

# The central difference of `fn` at the arguments `x` in the direction `t`, which
# names a subset of them. `fn` returns an array or a list of arrays.
central_difference <- function(fn, x, t, h) {
  at <- function(step) {
    x[names(t)] <- Map(\(a, d) a + step * d, x[names(t)], t)
    do.call(fn, x)
  }
  hi <- at(h)
  lo <- at(-h)
  if (is_graph_box(hi)) (hi - lo) / (2 * h) else Map(\(a, b) (a - b) / (2 * h), hi, lo)
}

# Checks the pushforward rule of `prim_<name>` on `case`, at f64:
#   - the tangent along each argument in `wrt` alone, and along all of them at
#     once, against a central difference;
#   - the tangent against the pullback: <u, J t> must equal <J^T u, t>;
#   - forward-over-reverse: the tangent of the gradient against a central
#     difference of the gradient, which needs a pushforward rule for whatever
#     the reverse rule computes with -- not yet with a scalar operand, whose
#     cotangent the reverse rule sums with `prim_sum()`;
#   - at f32, that the tangent keeps the data type.
# An output that is not a float has no derivative, and so no tangent.
check_pushforward_rule <- function(name, case) {
  prim <- getFromNamespace(paste0("prim_", name), "anvl")
  shape <- c(2L, 3L)
  args <- Map(\(sample, nm) sample(if (nm %in% case$scalar) integer() else shape), case$args, names(case$args))
  arrays_at <- function(dtype) {
    lapply(args, \(a) anvl_array(a, dtype = if (is.logical(a)) "bool" else dtype))
  }
  directions <- lapply(case$wrt, \(nm) shaped(rnorm(length(args[[nm]])), dim(args[[nm]])))
  names(directions) <- case$wrt

  f <- with_formals(names(args), \(a) do.call(prim, c(a, case$params)))
  x <- arrays_at("f64")
  t <- lapply(directions, anvl_array, dtype = "f64")
  out_dtype <- dtype(trace_fn(f, x)$outputs[[1L]])

  if (!is_dtype_float(out_dtype)) {
    jit(function(x, t) {
      expect_null(tangent_along(f, x, t))
      x[[1L]]
    })(x, t)
    return(invisible())
  }

  h <- 1e-5
  u <- nv_array(array(rnorm(prod(shape)), shape), dtype = "f64")
  run <- jit(function(x, t, u) {
    along <- c(lapply(names(t), \(nm) t[nm]), list(t))
    loss <- with_formals(names(x), \(a) sum(do.call(f, a) * u))
    grad <- gradient(loss, wrt = names(t))
    # `f` of the arguments in `wrt` alone, the others held at `x`
    f_wrt <- with_formals(names(t), function(a) {
      args <- x
      args[names(a)] <- a
      do.call(f, args)
    })
    jt <- tangent_along(f, x, t)
    jtu <- do.call(pullback, c(list(f_wrt), x[names(t)]))(u)
    c(
      list(
        tangent = lapply(along, \(d) tangent_along(f, x, d)),
        fd = lapply(along, \(d) central_difference(f, x, d, h)),
        u_jt = sum(u * jt),
        jtu_t = Reduce(`+`, Map(\(nm) sum(jtu[[nm]] * t[[nm]]), names(t)))
      ),
      if (!length(case$scalar)) {
        list(hvp = tangent_along(grad, x, t), hvp_fd = central_difference(grad, x, t, h))
      }
    )
  })
  res <- run(x, t, u)

  labels <- c(paste("along", names(t)), "along all")
  for (i in seq_along(res$tangent)) {
    expect_equal(dtype(res$tangent[[i]]), out_dtype)
    expect_equal(as_array(res$tangent[[i]]), as_array(res$fd[[i]]), tolerance = 1e-6, label = labels[[i]])
  }
  expect_equal(as_array(res$u_jt), as_array(res$jtu_t), tolerance = 1e-10)
  for (nm in names(res$hvp)) {
    expect_equal(as_array(res$hvp[[nm]]), as_array(res$hvp_fd[[nm]]), tolerance = 1e-5, label = paste("hvp", nm))
  }

  tangent32 <- jit(\(x, t) tangent_along(f, x, t))(arrays_at("f32"), lapply(directions, anvl_array, dtype = "f32"))
  expect_equal(dtype(tangent32), as_dtype("f32"))
  expect_equal(as_array(tangent32), as_array(res$tangent[[length(res$tangent)]]), tolerance = 1e-4)
}

describe("pushforward rules", {
  it("have a case for every primitive with a pushforward rule", {
    nms <- names(asNamespace("anvl"))
    prims <- Filter(\(p) inherits(p, "AnvlPrimitive"), mget(nms[startsWith(nms, "prim_")], asNamespace("anvl")))
    with_rule <- sub("^prim_", "", names(Filter(\(p) !is.null(p[["pushforward"]]), prims)))
    expect_setequal(names(pushforward_cases), with_rule)
  })

  for (name in names(pushforward_cases)) {
    it(sprintf("of prim_%s agree with finite differences and the pullback", name), {
      withr::local_seed(1L)
      check_pushforward_rule(name, pushforward_cases[[name]])
    })
  }

  for (name in names(pushforward_scalar_cases)) {
    prim_name <- sub("/.*", "", name)
    scalar <- sub(".*/", "", name)
    it(sprintf("of prim_%s agree with finite differences and the pullback for a scalar %s", prim_name, scalar), {
      withr::local_seed(1L)
      check_pushforward_rule(prim_name, pushforward_scalar_cases[[name]])
    })
  }
})
