# The forward rules are checked automatically: every primitive with a forward
# rule has an entry in `forward_cases`, and `check_forward_rule()` runs the same
# checks on each of them against references that do not use the forward rules.
# torch has no forward mode to compare with.

# Samplers for the values a case calls its primitive with, as a
# `function(shape)` returning an R array.
unif <- function(lower, upper) {
  function(shape) array(runif(prod(shape), lower, upper), shape)
}
# Away from the integers and the half-integers, where a step function jumps.
off_step <- function(lower, upper) {
  function(shape) array(sample(lower:upper, prod(shape), replace = TRUE) + runif(prod(shape), 0.1, 0.4), shape)
}
coin <- function() {
  function(shape) array(runif(prod(shape)) < 0.5, shape)
}
fixed <- function(value) {
  function(shape) array(value, shape)
}

# A call of a primitive: its arguments, as samplers -- a float sampler gives an
# array of the data type under test, a logical one a boolean array -- the
# arguments to differentiate with respect to, and static parameters.
forward_case <- function(args, wrt = names(args), params = list()) {
  list(args = args, wrt = wrt, params = params)
}

binary_case <- function(lhs = unif(-2, 2), rhs = unif(-2, 2)) {
  forward_case(list(lhs = lhs, rhs = rhs))
}

forward_cases <- list(
  add = binary_case(),
  sub = binary_case(),
  mul = binary_case(),
  div = binary_case(rhs = unif(0.5, 2)),
  remainder = forward_case(list(x = unif(-5, 5), y = unif(0.5, 2))),
  pow = forward_case(list(x = unif(0.5, 2), y = unif(-2, 2))),
  negate = forward_case(list(x = unif(-2, 2))),
  abs = forward_case(list(x = unif(-2, 2))),
  pmax = binary_case(),
  pmin = binary_case(),
  clamp = forward_case(list(x = unif(-2, 2), min = fixed(-1), max = fixed(1)), wrt = "x"),
  ifelse = forward_case(list(test = coin(), yes = unif(-2, 2), no = unif(-2, 2)), wrt = c("yes", "no")),
  exp = forward_case(list(x = unif(-2, 2))),
  expm1 = forward_case(list(x = unif(-2, 2))),
  log = forward_case(list(x = unif(0.5, 3))),
  log1p = forward_case(list(x = unif(-0.5, 2))),
  plogis = forward_case(list(x = unif(-3, 3))),
  sqrt = forward_case(list(x = unif(0.5, 3))),
  rsqrt = forward_case(list(x = unif(0.5, 3))),
  cbrt = forward_case(list(x = unif(0.5, 3))),
  sin = forward_case(list(x = unif(-3, 3))),
  cos = forward_case(list(x = unif(-3, 3))),
  tan = forward_case(list(x = unif(-1, 1))),
  asin = forward_case(list(x = unif(-0.9, 0.9))),
  acos = forward_case(list(x = unif(-0.9, 0.9))),
  atan = forward_case(list(x = unif(-3, 3))),
  atan2 = forward_case(list(y = unif(-2, 2), x = unif(-2, 2))),
  sinh = forward_case(list(x = unif(-2, 2))),
  cosh = forward_case(list(x = unif(-2, 2))),
  tanh = forward_case(list(x = unif(-2, 2))),
  asinh = forward_case(list(x = unif(-2, 2))),
  acosh = forward_case(list(x = unif(1.5, 3))),
  atanh = forward_case(list(x = unif(-0.9, 0.9))),
  lgamma = forward_case(list(x = unif(0.5, 3))),
  digamma = forward_case(list(x = unif(0.5, 3))),
  psigamma = forward_case(list(x = unif(0.5, 3), deriv = fixed(1)), wrt = "x"),
  erf = forward_case(list(x = unif(-2, 2))),
  erfc = forward_case(list(x = unif(-2, 2))),
  erf_inv = forward_case(list(x = unif(-0.9, 0.9))),
  floor = forward_case(list(x = off_step(-3L, 2L))),
  ceiling = forward_case(list(x = off_step(-3L, 2L))),
  round = forward_case(list(x = off_step(-3L, 2L))),
  sign = forward_case(list(x = unif(-2, 2))),
  eq = binary_case(),
  ne = binary_case(),
  gt = binary_case(),
  ge = binary_case(),
  lt = binary_case(),
  le = binary_case(),
  is_finite = forward_case(list(x = unif(-2, 2)))
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
  do.call(jvp(fn, wrt = names(t)), c(x, list(tangents = t)))
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

# Checks the forward rule of `prim_<name>` on `case`, at f64:
#   - the tangent along each argument in `wrt` alone, and along all of them at
#     once, against a central difference;
#   - the tangent against the reverse rule: <u, J t> must equal <J^T u, t>;
#   - forward-over-reverse: the tangent of the gradient against a central
#     difference of the gradient, which needs a forward rule for whatever the
#     reverse rule computes with;
#   - at f32, that the tangent keeps the data type.
# An output that is not a float has no derivative, so its tangent must be zero.
check_forward_rule <- function(name, case) {
  prim <- getFromNamespace(paste0("prim_", name), "anvl")
  shape <- c(2L, 3L)
  args <- lapply(case$args, \(sample) sample(shape))
  arrays_at <- function(dtype) {
    lapply(args, \(a) nv_array(a, dtype = if (is.logical(a)) "bool" else dtype))
  }
  directions <- lapply(case$wrt, \(nm) array(rnorm(prod(shape)), shape))
  names(directions) <- case$wrt

  f <- with_formals(names(args), \(a) do.call(prim, c(a, case$params)))
  x <- arrays_at("f64")
  t <- lapply(directions, nv_array, dtype = "f64")
  out_dtype <- dtype(trace_fn(f, x)$outputs[[1L]])

  if (!is_dtype_float(out_dtype)) {
    tangent <- jit(\(x, t) tangent_along(f, x, t))(x, t)
    expect_equal(dtype(tangent), out_dtype)
    expect_false(any(as.logical(as_array(tangent))))
    return(invisible())
  }

  h <- 1e-5
  u <- nv_array(array(rnorm(prod(shape)), shape), dtype = "f64")
  run <- jit(function(x, t, u) {
    along <- c(lapply(names(t), \(nm) t[nm]), list(t))
    loss <- with_formals(names(x), \(a) sum(do.call(f, a) * u))
    grad <- gradient(loss, wrt = names(t))
    jt <- tangent_along(f, x, t)
    jt_grad <- do.call(grad, x)
    list(
      jvp = lapply(along, \(d) tangent_along(f, x, d)),
      fd = lapply(along, \(d) central_difference(f, x, d, h)),
      u_jt = sum(u * jt),
      jt_u_t = Reduce(`+`, Map(\(g, d) sum(g * d), jt_grad, t)),
      hvp = tangent_along(grad, x, t),
      hvp_fd = central_difference(grad, x, t, h)
    )
  })
  res <- run(x, t, u)

  labels <- c(paste("along", names(t)), "along all")
  for (i in seq_along(res$jvp)) {
    expect_equal(dtype(res$jvp[[i]]), out_dtype)
    expect_equal(as_array(res$jvp[[i]]), as_array(res$fd[[i]]), tolerance = 1e-6, label = labels[[i]])
  }
  expect_equal(as_array(res$u_jt), as_array(res$jt_u_t), tolerance = 1e-10)
  for (nm in names(t)) {
    expect_equal(as_array(res$hvp[[nm]]), as_array(res$hvp_fd[[nm]]), tolerance = 1e-5, label = paste("hvp", nm))
  }

  tangent32 <- jit(\(x, t) tangent_along(f, x, t))(arrays_at("f32"), lapply(directions, nv_array, dtype = "f32"))
  expect_equal(dtype(tangent32), as_dtype("f32"))
  expect_equal(as_array(tangent32), as_array(res$jvp[[length(res$jvp)]]), tolerance = 1e-4)
}

describe("forward rules", {
  it("have a case for every primitive with a forward rule", {
    nms <- names(asNamespace("anvl"))
    prims <- Filter(\(p) inherits(p, "AnvlPrimitive"), mget(nms[startsWith(nms, "prim_")], asNamespace("anvl")))
    with_rule <- sub("^prim_", "", names(Filter(\(p) !is.null(p[["forward"]]), prims)))
    expect_setequal(names(forward_cases), with_rule)
  })

  for (name in names(forward_cases)) {
    it(sprintf("of prim_%s agree with finite differences and the reverse rule", name), {
      withr::local_seed(1L)
      check_forward_rule(name, forward_cases[[name]])
    })
  }
})
