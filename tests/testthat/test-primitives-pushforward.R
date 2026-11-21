# A primitive tagged "elementwise" gets its pushforward rule from its pullback
# rule (`pushforward_from_pullback()`), which the pullback tests cover, so only
# the derivation itself is checked here, on a few primitives that take each of
# its paths. Every other pushforward rule has an entry in `pushforward_cases`.
# `check_pushforward_rule()` runs the same checks on each against references
# that do not use the pushforward rules. torch has no forward mode to compare
# with.

# Samplers for the values a case calls its primitive with, as a
# `function(shape)` returning an R array, or a number for the scalar shape.
unif <- function(lower, upper) {
  function(shape) shaped(runif(prod(shape), lower, upper), shape)
}
coin <- function() {
  function(shape) shaped(runif(prod(shape)) < 0.5, shape)
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

# The pushforward rules of their own.
pushforward_cases <- list(
  # at f64 a conversion to f64 is the identity, at f32 one that changes the
  # data type
  convert = pushforward_case(list(x = unif(-2, 2)), params = list(dtype = "f64"))
)

# Pushforward rules derived from the pullback rule, named
# `<primitive>[/<scalar operand>]`: an operand alone and several whose tangents
# add up, a scalar operand whose tangent is broadcast, an operand without a
# tangent, and a result that is not a float.
derived_cases <- list(
  sin = pushforward_case(list(x = unif(-3, 3))),
  mul = binary_case(),
  "mul/lhs" = pushforward_case(list(lhs = unif(-2, 2), rhs = unif(-2, 2)), scalar = "lhs"),
  ifelse = pushforward_case(list(test = coin(), yes = unif(-2, 2), no = unif(-2, 2)), wrt = c("yes", "no")),
  eq = binary_case()
)

# Whether the pushforward rule of `prim` is the one `pushforward_from_pullback()`
# derives from its pullback rule.
derived_from_pullback <- function(prim) {
  identical(body(prim[["pushforward"]]), body(pushforward_from_pullback(prim[["pullback"]])))
}

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
#     cotangent the pullback rule sums with `prim_sum()`;
#   - at f32, that the tangent has the data type of the result.
# An output that is not a float has no derivative, and so no tangent.
check_pushforward_rule <- function(name, case) {
  prim <- getFromNamespace(paste0("prim_", name), "anvl")
  shape <- c(2L, 3L)
  args <- Map(\(sample, nm) sample(if (nm %in% case$scalar) integer() else shape), case$args, names(case$args))
  arrays_at <- function(dtype) {
    lapply(args, \(a) {
      anvl_array(
        a,
        dtype = if (is.logical(a)) {
          "bool"
        } else if (is.integer(a)) {
          "i32"
        } else {
          dtype
        }
      )
    })
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

  x32 <- arrays_at("f32")
  tangent32 <- jit(\(x, t) tangent_along(f, x, t))(x32, lapply(directions, anvl_array, dtype = "f32"))
  expect_equal(dtype(tangent32), dtype(trace_fn(f, x32)$outputs[[1L]]))
  expect_equal(as_array(tangent32), as_array(res$tangent[[length(res$tangent)]]), tolerance = 1e-4)
}

describe("pushforward rules", {
  it("have a case for every primitive with a rule of its own", {
    nms <- names(asNamespace("anvl"))
    prims <- Filter(\(p) inherits(p, "AnvlPrimitive"), mget(nms[startsWith(nms, "prim_")], asNamespace("anvl")))
    own <- Filter(\(p) !is.null(p[["pushforward"]]) && !derived_from_pullback(p), prims)
    expect_setequal(names(pushforward_cases), sub("^prim_", "", names(own)))
  })

  for (name in names(pushforward_cases)) {
    it(sprintf("of prim_%s agree with finite differences and the pullback", name), {
      withr::local_seed(1L)
      check_pushforward_rule(name, pushforward_cases[[name]])
    })
  }

  it("of prim_convert give no tangent to an integer result, nor one to the float result of an integer", {
    x <- nv_array(c(1.5, 2.5), dtype = "f64")
    tangent <- "none yet"
    jit(function(x) {
      tangent <<- pushforward(function(x) prim_convert(x, "i32"), x)(x)
      x
    })(x)
    expect_null(tangent)
    from_int <- jit(function(i) pushforward(function(i) prim_convert(i, "f64"), i)(NULL))
    expect_equal(as.vector(as_array(from_int(nv_array(1:2, dtype = "i32")))), c(0, 0))
  })
})

describe("pushforward_from_pullback", {
  it("gives every elementwise primitive a pushforward rule", {
    for (prim in primitives_tagged("elementwise")) {
      expect_false(is.null(prim[["pushforward"]]), label = primitive_def(prim)$name)
    }
  })

  for (name in names(derived_cases)) {
    it(sprintf("derives a pushforward rule that agrees with finite differences and the pullback for %s", name), {
      withr::local_seed(1L)
      check_pushforward_rule(sub("/.*", "", name), derived_cases[[name]])
    })
  }
})
