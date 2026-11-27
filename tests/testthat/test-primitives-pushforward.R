# A primitive tagged "elementwise" gets its pushforward rule from its pullback
# rule (`pushforward_from_pullback()`), which the pullback tests cover, and one
# tagged "linear" applies itself to the tangents (`pushforward_linear()`), so
# only each derivation is checked here, on one primitive as an example. Every
# other pushforward rule has an entry in `pushforward_cases`.
# `check_pushforward_rule()` runs the same checks on each against references
# that do not use the pushforward rules. torch has no forward mode to compare
# with.

# Samplers for the values a case calls its primitive with, as a
# `function(shape)` returning an R array, or a number for the scalar shape.
unif <- function(lower, upper) {
  function(shape) shaped(runif(prod(shape), lower, upper), shape)
}
shaped <- function(values, shape) {
  if (length(shape)) array(values, shape) else values
}

# `x`, an R array or number, as an array of data type `dtype`.
anvl_array <- function(x, dtype) {
  if (is.null(dim(x))) nv_scalar(x, dtype = dtype) else nv_array(x, dtype = dtype)
}

# A call of a primitive: its arguments, as samplers of arrays of the data type
# under test, all of which are differentiated, the arguments that are scalars,
# which the primitive broadcasts to the shape of the others, static parameters,
# and, where the primitive does not take the arguments as they are, a `call`
# that calls it with them.
pushforward_case <- function(args, scalar = character(), params = list(), call = NULL) {
  list(args = args, scalar = scalar, params = params, call = call)
}

# The pushforward rules of their own.
pushforward_cases <- list(
  # at f64 a conversion to f64 is the identity, at f32 one that changes the
  # data type
  convert = pushforward_case(list(x = unif(-2, 2)), params = list(dtype = "f64")),
  # the barrier takes its operands as a list, and returns one
  optimization_barrier = pushforward_case(
    list(x = unif(-2, 2)),
    call = function(x) prim_optimization_barrier(list(x))[[1L]]
  )
)

# Whether the pushforward rule of `prim` is derived from its tags: from its
# pullback rule (`pushforward_from_pullback()`), or as that of a linear
# primitive (`pushforward_linear()`).
derived_from_tags <- function(prim) {
  rule <- body(prim[["pushforward"]])
  identical(rule, body(pushforward_from_pullback(prim[["pullback"]]))) ||
    identical(rule, body(pushforward_linear(prim)))
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

# `fn` of the arguments `t` names alone, the others held at `x`.
fixing <- function(fn, x, t) {
  with_formals(names(t), function(a) {
    args <- x
    args[names(a)] <- a
    do.call(fn, args)
  })
}

# Checks the pushforward rule of `prim_<name>` on `case`, at f64, against the
# pullback, which does not use the pushforward rules:
#   - the tangent along each argument alone, and along all of them at
#     once: element i of J t is <J^T e_i, t>, with J^T e_i the pullback of the
#     i-th unit cotangent;
#   - forward-over-reverse against reverse-over-reverse: the tangent of the
#     gradient of <u, f> along t is its Hessian times t, and as the Hessian is
#     symmetric, so is the pullback of the gradient along t;
#   - at f32, that the tangent has the data type of the result.
check_pushforward_rule <- function(name, case) {
  prim <- getFromNamespace(paste0("prim_", name), "anvl")
  shape <- c(2L, 3L)
  args <- Map(\(sample, nm) sample(if (nm %in% case$scalar) integer() else shape), case$args, names(case$args))
  arrays_at <- function(dtype) lapply(args, anvl_array, dtype = dtype)
  directions <- lapply(args, \(a) shaped(rnorm(length(a)), dim(a)))

  f <- with_formals(names(args), \(a) do.call(case$call %||% prim, c(a, case$params)))
  x <- arrays_at("f64")
  t <- lapply(directions, anvl_array, dtype = "f64")
  out <- trace_fn(f, x)$outputs[[1L]]
  out_dtype <- dtype(out)

  n_out <- prod(shape(out))
  as_out <- \(values) anvl_array(shaped(values, shape(out)), dtype = "f64")
  units <- lapply(seq_len(n_out), \(i) as_out(replace(numeric(n_out), i, 1)))
  u <- as_out(rnorm(n_out))
  run <- jit(function(x, t, u, units) {
    along <- c(lapply(names(t), \(nm) t[nm]), list(t))
    # the rows of the Jacobian, J^T e_i
    rows <- lapply(units, do.call(pullback, c(list(fixing(f, x, t)), x[names(t)])))
    # element i of J d is <J^T e_i, d>
    reference <- function(d) {
      lapply(rows, \(row) Reduce(`+`, Map(\(nm) sum(row[[nm]] * d[[nm]]), names(d))))
    }
    loss <- with_formals(names(x), \(a) sum(do.call(f, a) * u))
    grad <- gradient(loss, wrt = names(t))
    list(
      tangent = lapply(along, \(d) tangent_along(f, x, d)),
      reference = lapply(along, reference),
      hvp = tangent_along(grad, x, t),
      hvp_reverse = do.call(pullback, c(list(fixing(grad, x, t)), x[names(t)]))(t)
    )
  })
  res <- run(x, t, u, units)

  labels <- c(paste("along", names(t)), "along all")
  for (i in seq_along(res$tangent)) {
    expect_equal(dtype(res$tangent[[i]]), out_dtype)
    expect_equal(
      as.vector(as_array(res$tangent[[i]])),
      vapply(res$reference[[i]], as_array, numeric(1L)),
      tolerance = 1e-10,
      label = labels[[i]]
    )
  }
  for (nm in names(t)) {
    expect_equal(as_array(res$hvp[[nm]]), as_array(res$hvp_reverse[[nm]]), tolerance = 1e-10, label = paste("hvp", nm))
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
    own <- Filter(\(p) !is.null(p[["pushforward"]]) && !derived_from_tags(p), prims)
    expect_setequal(names(pushforward_cases), sub("^prim_", "", names(own)))
  })

  for (name in names(pushforward_cases)) {
    it(sprintf("of prim_%s agree with the pullback", name), {
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

  # `mul` with a scalar operand: the tangents of two operands add up, and that
  # of the scalar one is broadcast
  it("derives a pushforward rule that agrees with the pullback, e.g. for prim_mul", {
    withr::local_seed(1L)
    check_pushforward_rule("mul", pushforward_case(list(lhs = unif(-2, 2), rhs = unif(-2, 2)), scalar = "lhs"))
  })
})

describe("pushforward_linear", {
  # two float operands: along either alone, the other one's tangent is zeros
  it("derives a pushforward rule that agrees with the pullback, e.g. for prim_concatenate", {
    withr::local_seed(1L)
    check_pushforward_rule(
      "concatenate",
      pushforward_case(list(x = unif(-2, 2), y = unif(-2, 2)), params = list(axis = 1L))
    )
  })
})
