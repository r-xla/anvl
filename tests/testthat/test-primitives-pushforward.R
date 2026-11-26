# A primitive tagged "elementwise" gets its pushforward rule from its pullback
# rule (`pushforward_from_pullback()`), which the pullback tests cover, so only
# the derivation itself is checked here, on a few primitives that take each of
# its paths. Every other pushforward rule has an entry in `pushforward_cases`,
# or, for a primitive that takes functions, entries in `subgraph_cases`.
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
ints <- function(lower, upper) {
  function(shape) shaped(sample(lower:upper, prod(shape), replace = TRUE), shape)
}
always <- function(value) {
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
# array of the data type under test, a logical one a boolean array, an integer
# one an `i32` array -- the arguments to differentiate with respect to, the
# arguments that are scalars, which the primitive broadcasts to the shape of
# the others, the shapes of the other arguments where they are not `c(2, 3)`,
# static parameters, a function `fn` of the arguments, by name, that calls the
# primitive in place of calling it with them, whether to check
# forward-over-reverse, and the tolerance of the comparison with finite
# differences.
pushforward_case <- function(
  args,
  wrt = names(args),
  scalar = character(),
  shapes = list(),
  params = list(),
  fn = NULL,
  hvp = TRUE,
  tolerance = 1e-6
) {
  list(
    args = args,
    wrt = wrt,
    scalar = scalar,
    shapes = shapes,
    params = params,
    fn = fn,
    hvp = hvp,
    tolerance = tolerance
  )
}

# The results of a primitive as a list, with one entry per output.
as_outputs <- function(r) {
  if (is.list(r) && !is_arrayish(r)) r else list(r)
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

# The pushforward rules of the primitives that take functions, named
# `<primitive>/<what the case covers>`, each called through `fn`. What its
# functions use is closed over, the arguments of `fn` among it.
subgraph_cases <- list(
  # a branch returning a value it closes over as it is, and a result that is
  # not a float
  "if/true" = pushforward_case(
    list(p = always(TRUE), x = unif(-2, 2), y = unif(-2, 2)),
    wrt = c("x", "y"),
    scalar = "p",
    fn = function(p, x, y) nv_if(p, \() list(x * y, sin(x), x < y), \() list(cos(y), x, x > y))
  ),
  "if/false" = pushforward_case(
    list(p = always(FALSE), x = unif(-2, 2), y = unif(-2, 2)),
    wrt = c("x", "y"),
    scalar = "p",
    fn = function(p, x, y) nv_if(p, \() list(x * y, sin(x), x < y), \() list(cos(y), x, x > y))
  ),
  "if/nested" = pushforward_case(
    list(p = always(TRUE), q = always(FALSE), x = unif(-2, 2), y = unif(-2, 2)),
    wrt = c("x", "y"),
    scalar = c("p", "q"),
    fn = function(p, q, x, y) nv_if(p, \() nv_if(q, \() x * y, \() sin(x) * y), \() x)
  ),
  "if/scan" = pushforward_case(
    list(p = always(TRUE), h = unif(-1, 1), xs = unif(-1, 1)),
    wrt = c("h", "xs"),
    scalar = "p",
    shapes = list(h = 2L),
    fn = function(p, h, xs) {
      nv_if(p, \() nv_scan(h, xs, \(h, x) list(carry = sin(h) * x, out = NULL))$carry, \() h)
    }
  ),
  # the carry, `xs` and a value the body closes over, and outputs of every step
  scan = pushforward_case(
    list(h = unif(-1, 1), w = unif(-1, 1), xs = unif(-1, 1)),
    shapes = list(h = 2L, w = 2L),
    fn = function(h, w, xs) {
      r <- nv_scan(h, xs, \(h, x) list(carry = sin(h) * w + x, out = h * x))
      list(r$carry, r$out)
    }
  ),
  "scan/reverse" = pushforward_case(
    list(h = unif(-1, 1), w = unif(-1, 1), xs = unif(-1, 1)),
    shapes = list(h = 2L, w = 2L),
    fn = function(h, w, xs) {
      r <- nv_scan(h, xs, \(h, x) list(carry = sin(h) * w + x, out = h * x), reverse = TRUE)
      list(r$carry, r$out)
    }
  ),
  # `b` enters without a tangent and gains one, the counters `i` and `t`
  # never have one, and neither does the integer output
  "scan/carry" = pushforward_case(
    list(a = unif(-1, 1), w = unif(-1, 1)),
    shapes = list(a = 2L, w = 2L),
    fn = function(a, w) {
      init <- list(a = a, b = nv_fill(0, 2L, dtype = dtype(a)), t = nv_fill(0, integer(), dtype = dtype(a)), i = 0L)
      body <- function(c, x) {
        list(carry = list(a = c$a * w, b = c$b + sin(c$a), t = c$t + 1, i = c$i + 1L), out = list(c$a * c$t, c$i))
      }
      r <- nv_scan(init, body = body, steps = 3L)
      list(r$carry$a, r$carry$b, r$carry$t, r$out[[1L]], r$out[[2L]])
    }
  ),
  # the outputs of the steps have tangents, the carry has none
  "scan/outputs" = pushforward_case(
    list(xs = unif(-1, 1)),
    fn = function(xs) {
      body <- function(c, x) list(carry = c + 1, out = sin(x) * c)
      nv_scan(nv_fill(1, integer(), dtype = dtype(xs)), xs, body)$out
    }
  ),
  "scan/nested" = pushforward_case(
    list(h = unif(-1, 1), w = unif(-1, 1)),
    shapes = list(h = 2L, w = 2L),
    fn = function(h, w) {
      inner <- function(c) nv_scan(c, body = \(d, x) list(carry = sin(d) * w + c, out = NULL), steps = 2L)$carry
      r <- nv_scan(h, body = \(c, x) list(carry = inner(c), out = c * c), steps = 2L)
      list(r$carry, r$out)
    }
  ),
  # the predicate differs from step to step
  "scan/if" = pushforward_case(
    list(h = unif(-1, 1), w = unif(-1, 1), p = coin(), v = unif(-1, 1)),
    wrt = c("h", "w", "v"),
    shapes = list(h = 2L, w = 2L, p = 3L),
    fn = function(h, w, p, v) {
      body <- function(c, x) {
        list(carry = nv_if(x$p, \() c * w + x$v, \() sin(c) - x$v), out = NULL)
      }
      nv_scan(h, list(p = p, v = v), body)$carry
    }
  ),
  # forward mode differentiates a while loop, which reverse mode cannot: no
  # pullback to compare with
  "while" = pushforward_case(
    list(y = unif(-1, 1), w = unif(-1, 1)),
    shapes = list(y = 2L, w = 2L),
    fn = function(y, w) nv_while(list(i = 0L, y = y), \(i, y) i < 3L, \(i, y) list(i = i + 1L, y = sin(y) * w + y))$y
  ),
  # `b` enters without a tangent and gains one, the counters `i` and `t`
  # never have one
  "while/state" = pushforward_case(
    list(a = unif(-1, 1), w = unif(-1, 1)),
    shapes = list(a = 2L, w = 2L),
    fn = function(a, w) {
      init <- list(i = 0L, t = nv_fill(0, integer(), dtype = dtype(a)), a = a, b = nv_fill(0, 2L, dtype = dtype(a)))
      r <- nv_while(init, \(i, t, a, b) i < 3L, \(i, t, a, b) list(i = i + 1L, t = t + 1, a = a * w, b = b + t * a))
      list(r$t, r$a, r$b)
    }
  ),
  "while/scan" = pushforward_case(
    list(y = unif(-1, 1), xs = unif(-1, 1)),
    shapes = list(y = 2L),
    fn = function(y, xs) {
      body <- function(i, y) list(i = i + 1L, y = nv_scan(y, xs, \(c, x) list(carry = sin(c) * x, out = NULL))$carry)
      nv_while(list(i = 0L, y = y), \(i, y) i < 2L, body)$y
    }
  ),
  "while/nested" = pushforward_case(
    list(y = unif(-1, 1), w = unif(-1, 1)),
    shapes = list(y = 2L, w = 2L),
    fn = function(y, w) {
      inner <- function(y) nv_while(list(j = 0L, z = y), \(j, z) j < 2L, \(j, z) list(j = j + 1L, z = sin(z) + w))$z
      nv_while(list(i = 0L, y = y), \(i, y) i < 2L, \(i, y) list(i = i + 1L, y = inner(y) * y))$y
    }
  ),
  # a rule that reads the result, so that forward-over-reverse goes through
  # the forward pass
  custom_gradient = pushforward_case(
    list(x = unif(-1, 1), y = unif(-1, 1)),
    fn = function(x, y) {
      prim_custom_gradient(
        list(x = x, y = y),
        fn = \(x, y) exp(x) * y,
        rule = \(x, y, out, grad, required) list(x = grad * out, y = grad * exp(x))
      )
    }
  ),
  # x^n by a while loop that `fn` runs
  "custom_gradient/while" = pushforward_case(
    list(x = unif(0.5, 2), n = always(3L)),
    wrt = "x",
    scalar = "n",
    fn = function(x, n) {
      prim_custom_gradient(
        list(x = x, n = n),
        fn = \(x, n) nv_while(list(i = 1L, y = x), \(i, y) i < n, \(i, y) list(i = i + 1L, y = y * x))$y,
        rule = \(x, n, out, grad, required) list(x = grad * n * out / x)
      )
    }
  )
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

# Checks the pushforward rule of `prim_<name>` on `case`, at f64, on each of
# its float outputs:
#   - the tangent along each argument in `wrt` alone, and along all of them at
#     once, against a central difference;
#   - the tangent against the pullback, if the primitive has a pullback rule:
#     <u, J t> must equal <J^T u, t>;
#   - forward-over-reverse, unless `case$hvp` is `FALSE`: the tangent of the
#     gradient against a central difference of the gradient, which needs a
#     pushforward rule for whatever the pullback rule computes with -- not yet
#     with a scalar operand that is differentiated, whose cotangent the
#     pullback rule sums with `prim_sum()`;
#   - at f32, that the tangent has the data type of the output.
# An output that is not a float has no derivative, and so no tangent.
check_pushforward_rule <- function(name, case) {
  prim <- getFromNamespace(paste0("prim_", name), "anvl")
  shape_of <- function(nm) if (nm %in% case$scalar) integer() else case$shapes[[nm]] %||% c(2L, 3L)
  args <- Map(\(sample, nm) sample(shape_of(nm)), case$args, names(case$args))
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

  f <- case$fn %||% with_formals(names(args), \(a) do.call(prim, c(a, case$params)))
  x <- arrays_at("f64")
  t <- lapply(directions, anvl_array, dtype = "f64")
  outs <- trace_fn(f, x)$outputs
  is_float <- vapply(outs, \(o) is_dtype_float(dtype(o)), logical(1L))

  if (!any(is_float)) {
    jit(function(x, t) {
      tangents <- as_outputs(tangent_along(f, x, t))
      expect_true(all(vapply(tangents, is.null, logical(1L))))
      x[[1L]]
    })(x, t)
    return(invisible())
  }

  # `f` with its float outputs alone, as a list
  g <- with_formals(names(args), \(a) as_outputs(do.call(f, a))[is_float])
  h <- 1e-5
  u <- lapply(outs[is_float], \(o) anvl_array(shaped(rnorm(prod(shape(o))), shape(o)), dtype = "f64"))
  has_pullback <- !is.null(prim[["pullback"]])
  check_hvp <- has_pullback && case$hvp && !any(case$scalar %in% case$wrt)
  run <- jit(function(x, t, u) {
    along <- c(lapply(names(t), \(nm) t[nm]), list(t))
    # `g` of the arguments in `wrt` alone, the others held at `x`
    g_wrt <- with_formals(names(t), function(a) {
      args <- x
      args[names(a)] <- a
      do.call(g, args)
    })
    jt <- tangent_along(g, x, t)
    c(
      list(
        tangent = lapply(along, \(d) tangent_along(g, x, d)),
        fd = lapply(along, \(d) central_difference(g, x, d, h))
      ),
      if (has_pullback) {
        jtu <- do.call(pullback, c(list(g_wrt), x[names(t)]))(u)
        list(
          u_jt = Reduce(`+`, Map(\(a, b) sum(a * b), u, jt)),
          jtu_t = Reduce(`+`, Map(\(nm) sum(jtu[[nm]] * t[[nm]]), names(t)))
        )
      },
      if (check_hvp) {
        loss <- with_formals(names(x), \(a) Reduce(`+`, Map(\(o, w) sum(o * w), do.call(g, a), u)))
        grad <- gradient(loss, wrt = names(t))
        list(hvp = tangent_along(grad, x, t), hvp_fd = central_difference(grad, x, t, h))
      }
    )
  })
  res <- run(x, t, u)

  labels <- c(paste("along", names(t)), "along all")
  out_dtypes <- lapply(outs[is_float], dtype)
  for (i in seq_along(res$tangent)) {
    for (k in seq_along(out_dtypes)) {
      label <- if (length(out_dtypes) > 1L) sprintf("%s, output %d", labels[[i]], k) else labels[[i]]
      expect_equal(dtype(res$tangent[[i]][[k]]), out_dtypes[[k]], label = label)
      expect_equal(
        as_array(res$tangent[[i]][[k]]),
        as_array(res$fd[[i]][[k]]),
        tolerance = case$tolerance,
        label = label
      )
    }
  }
  if (has_pullback) {
    expect_equal(as_array(res$u_jt), as_array(res$jtu_t), tolerance = 1e-10)
  }
  for (nm in names(res$hvp)) {
    expect_equal(as_array(res$hvp[[nm]]), as_array(res$hvp_fd[[nm]]), tolerance = 1e-5, label = paste("hvp", nm))
  }

  x32 <- arrays_at("f32")
  tangent32 <- jit(\(x, t) tangent_along(g, x, t))(x32, lapply(directions, anvl_array, dtype = "f32"))
  outs32 <- trace_fn(g, x32)$outputs
  for (k in seq_along(tangent32)) {
    expect_equal(dtype(tangent32[[k]]), dtype(outs32[[k]]))
    expect_equal(as_array(tangent32[[k]]), as_array(res$tangent[[length(res$tangent)]][[k]]), tolerance = 1e-4)
  }
}

describe("pushforward rules", {
  it("have a case for every primitive with a rule of its own", {
    nms <- names(asNamespace("anvl"))
    prims <- Filter(\(p) inherits(p, "AnvlPrimitive"), mget(nms[startsWith(nms, "prim_")], asNamespace("anvl")))
    own <- Filter(\(p) !is.null(p[["pushforward"]]) && !derived_from_pullback(p), prims)
    cased <- c(names(pushforward_cases), unique(sub("/.*", "", names(subgraph_cases))))
    expect_setequal(cased, sub("^prim_", "", names(own)))
  })

  for (name in names(pushforward_cases)) {
    it(sprintf("of prim_%s agree with finite differences and the pullback", name), {
      withr::local_seed(1L)
      check_pushforward_rule(name, pushforward_cases[[name]])
    })
  }

  for (name in names(subgraph_cases)) {
    it(sprintf("of %s agree with finite differences and the pullback", name), {
      withr::local_seed(1L)
      check_pushforward_rule(sub("/.*", "", name), subgraph_cases[[name]])
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

# The number of times a print ran in `expr`, which is invisible, so that its
# value does not show: each one shows the value 7 on a line of its own.
prints_of_seven <- function(expr) sum(trimws(capture.output(expr)) == "7")

# The calls of the primitive `name` among the statements of `graph`.
calls_of <- function(graph, name) {
  Filter(\(call) call$primitive$name == name, graph$statements)
}

describe("prim_if", {
  x <- nv_array(c(1, 2, 3), dtype = "f64")
  v <- c(1, -1, 2)
  true_ <- nv_scalar(TRUE)

  it("pushes the tangents through the random numbers the branch drew", {
    f <- function(p, x) {
      a <- nv_if(
        p,
        function() {
          z <- nv_rnorm(integer(), dtype = "f64")
          list(value = x * z, draw = z)
        },
        function() list(value = x, draw = nv_scalar(0, "f64"))
      )
      last <- nv_rnorm(integer(), dtype = "f64")
      list(value = a$value * last, draw = a$draw, last = last)
    }
    run <- with_nv_seed(1L, jit(f)(true_, x))
    push <- jit(function(p, x, v) pushforward(function(x) f(p, x)$value, x)(v))
    tangent <- with_nv_seed(1L, push(true_, x, nv_array(v, dtype = "f64")))
    expect_equal(as.numeric(tangent), as.numeric(run$draw) * v * as.numeric(run$last))
  })

  it("returns no tangent of an output that no tangent reaches in either branch", {
    f <- function(p, x, y) nv_if(p, \() list(x * 2, y), \() list(x, y * 3))
    avals <- list(p = nv_aval("bool", integer()), x = nv_aval("f64", 3L), y = nv_aval("f64", 3L))
    graph <- trace_fn(function(p, x, y) pushforward(\(x) f(p, x, y), x)(x), avals)
    pushed <- calls_of(graph, "if")[[2L]]
    expect_length(pushed$outputs, 1L)
  })
})

describe("prim_scan", {
  x <- nv_array(c(1, 2, 3), dtype = "f64")
  v <- c(1, -1, 2)

  it("pushes the tangents through the random numbers the body drew", {
    f <- function(x) {
      body <- function(carry, xs) {
        z <- nv_rnorm(integer(), dtype = "f64")
        list(carry = list(a = carry$a + xs$v * z), out = z)
      }
      r <- prim_scan(list(a = nv_scalar(0, "f64")), list(v = x), body, steps = 3L)
      last <- nv_rnorm(integer(), dtype = "f64")
      list(value = r$carry$a * last, draws = r$out, last = last)
    }
    run <- with_nv_seed(1L, jit(f)(x))
    push <- jit(function(x, v) pushforward(function(x) f(x)$value, x)(v))
    tangent <- with_nv_seed(1L, push(x, nv_array(v, dtype = "f64")))
    expect_equal(as.numeric(tangent), sum(as.numeric(run$draws) * v) * as.numeric(run$last))
  })

  it("carries no tangent of a value that no tangent reaches", {
    # a float and an integer counter, and the random state the body draws from
    f <- function(a) {
      body <- function(carry, x) {
        z <- nv_rnorm(integer(), dtype = "f64")
        list(carry = list(a = carry$a * z, t = carry$t + 1, i = carry$i + 1L), out = NULL)
      }
      prim_scan(list(a = a, t = nv_scalar(0, "f64"), i = 0L), list(), body, steps = 3L)$carry$a
    }
    avals <- list(a = nv_aval("f64", 3L), v = nv_aval("f64", 3L))
    graph <- trace_fn(function(a, v) pushforward(f, a)(v), avals)
    scans <- calls_of(graph, "scan")
    expect_length(scans, 2L)
    expect_equal(scans[[2L]]$params$n_carry, scans[[1L]]$params$n_carry + 1L)
  })

  it("passes the tangent of the carry through a scan of zero steps", {
    f <- function(x) nv_scan(x, body = \(c, x) list(carry = sin(c), out = c), steps = 0L)
    out <- jit(function(x, v) pushforward(f, x)(v))(x, nv_array(v, dtype = "f64"))
    expect_equal(as.numeric(out$carry), v)
    expect_equal(shape(out$out), c(3L, 0L))
  })

  it("prints once per step", {
    f <- function(x) {
      body <- function(carry, xs) {
        prim_print(nv_scalar(7L))
        list(carry = carry * xs, out = NULL)
      }
      nv_scan(nv_scalar(1, "f64"), x, body)$carry
    }
    push <- jit(function(x, v) pushforward(f, x)(v))
    expect_equal(prints_of_seven(invisible(push(x, nv_array(v, dtype = "f64")))), 3L)
  })

  it("pushes tangents through more than once, e.g. for the columns of a Jacobian", {
    # the cumulative product, whose Jacobian is lower triangular
    f <- function(x) nv_scan(nv_scalar(1, "f64"), x, \(c, x) list(carry = c * x, out = c * x))$out
    jacobian <- jit(function(x) {
      push <- pushforward(f, x)
      eye <- nv_eye(3L, dtype = "f64")
      columns <- lapply(1:3, \(j) nv_reshape(push(eye[, j]), c(3L, 1L)))
      do.call(nv_concatenate, c(columns, list(axis = 2L)))
    })
    expected <- rbind(c(1, 0, 0), c(2, 1, 0), c(6, 3, 2))
    expect_equal(as_array(jacobian(x)), expected)
  })

  it("composes with gradient() in either order", {
    # x^3 by a scan, whose second derivative is 6 x
    cube <- function(x) nv_scan(x, body = \(c, i) list(carry = c * x, out = NULL), steps = 2L)$carry
    one <- function() nv_scalar(1, "f64")
    d1_rev <- function(x) gradient(cube)(x)$x
    d1_fwd <- function(x) pushforward(cube, x)(one())
    x <- nv_scalar(0.7, "f64")
    expect_equal(as.numeric(jit(function(x) pushforward(d1_rev, x)(one()))(x)), 6 * 0.7)
    expect_equal(as.numeric(jit(gradient(d1_fwd))(x)$x), 6 * 0.7)
    expect_equal(as.numeric(jit(function(x) pushforward(d1_fwd, x)(one()))(x)), 6 * 0.7)
  })
})

describe("prim_while", {
  x <- nv_array(c(1, 2, 3), dtype = "f64")
  v <- c(1, -1, 2)

  it("pushes the tangents through the random numbers the body drew", {
    # The value is x times the sum of the draws, so its tangent along v is
    # that sum times v.
    f <- function(x) {
      body <- function(i, y, s) {
        z <- nv_rnorm(integer(), dtype = "f64")
        list(i = i + 1L, y = y + x * z, s = s + z)
      }
      init <- list(i = 0L, y = nv_fill(0, 3L, dtype = "f64"), s = nv_scalar(0, "f64"))
      r <- nv_while(init, \(i, y, s) i < 3L, body)
      last <- nv_rnorm(integer(), dtype = "f64")
      list(value = r$y * last, sum = r$s, last = last)
    }
    run <- with_nv_seed(1L, jit(f)(x))
    push <- jit(function(x, v) pushforward(function(x) f(x)$value, x)(v))
    tangent <- with_nv_seed(1L, push(x, nv_array(v, dtype = "f64")))
    expect_equal(as.numeric(tangent), as.numeric(run$sum) * v * as.numeric(run$last))
  })

  it("differentiates a loop whose number of iterations depends on the state", {
    # y0 x^k for the first k with y0 x^k >= 10: here k = 6
    f <- function(y0, x) nv_while(list(y = y0), \(y) y < 10, \(y) list(y = y * x))$y
    push <- jit(function(y0, x, dy0, dx) pushforward(f, y0, x)(dy0, dx))
    one <- nv_scalar(1, "f64")
    zero <- nv_scalar(0, "f64")
    y0 <- nv_scalar(1, "f64")
    x <- nv_scalar(1.5, "f64")
    expect_equal(as.numeric(push(y0, x, one, zero)), 1.5^6)
    expect_equal(as.numeric(push(y0, x, zero, one)), 6 * 1.5^5)
  })

  it("carries no tangent of a value that no tangent reaches", {
    f <- function(a) {
      body <- function(i, t, a) list(i = i + 1L, t = t + 1, a = a * nv_rnorm(integer(), dtype = "f64"))
      nv_while(list(i = 0L, t = nv_scalar(0, "f64"), a = a), \(i, t, a) i < 3L, body)$a
    }
    avals <- list(a = nv_aval("f64", 3L), v = nv_aval("f64", 3L))
    graph <- trace_fn(function(a, v) pushforward(f, a)(v), avals)
    loops <- calls_of(graph, "while")
    expect_length(loops, 2L)
    expect_equal(length(loops[[2L]]$inputs), length(loops[[1L]]$inputs) + 1L)
  })

  it("prints once per iteration", {
    f <- function(x) {
      body <- function(i, y) {
        prim_print(nv_scalar(7L))
        list(i = i + 1L, y = y * x)
      }
      nv_while(list(i = 0L, y = x), \(i, y) i < 3L, body)$y
    }
    push <- jit(function(x, v) pushforward(f, x)(v))
    expect_equal(prints_of_seven(invisible(push(x, nv_array(v, dtype = "f64")))), 3L)
  })

  it("composes with itself into second derivatives", {
    cube <- function(x) nv_while(list(i = 1L, y = x), \(i, y) i < 3L, \(i, y) list(i = i + 1L, y = y * x))$y
    one <- function() nv_scalar(1, "f64")
    d1 <- function(x) pushforward(cube, x)(one())
    expect_equal(as.numeric(jit(function(x) pushforward(d1, x)(one()))(nv_scalar(0.7, "f64"))), 6 * 0.7)
  })
})

describe("prim_custom_gradient", {
  x <- nv_array(c(1, 2, 3), dtype = "f64")
  v <- c(1, -1, 2)

  it("pushes the tangents through fn, as the rule only replaces reverse mode", {
    # a straight-through rule, which does not give the derivative of fn
    f <- function(x) {
      prim_custom_gradient(list(x = x), fn = \(x) x * x, rule = \(x, out, grad, required) list(x = grad))
    }
    out <- jit(function(x, v) pushforward(f, x)(v))(x, nv_array(v, dtype = "f64"))
    expect_equal(as.numeric(out), 2 * c(1, 2, 3) * v)
  })

  it("pushes the tangent of a value fn closes over through it", {
    f <- function(x, w) {
      prim_custom_gradient(list(x = x), fn = \(x) x * w, rule = \(x, out, grad, required) list(x = grad * w))
    }
    out <- jit(function(x, w, v) pushforward(f, x, w)(w = v))(x, nv_scalar(2, "f64"), nv_scalar(1, "f64"))
    expect_equal(as.numeric(out), c(1, 2, 3))
  })

  it("gives the Hessian through the gradient of a rule with a forward pass", {
    # x^n with x^(n - 1) as the residual: the second derivative is n (n - 1) x^(n - 2)
    pow_fwd <- function(x, n) {
      prim_custom_gradient(
        list(x = x, n = n),
        fn = \(x, n) nv_while(list(i = 1L, y = x), \(i, y) i < n, \(i, y) list(i = i + 1L, y = y * x))$y,
        forward = function(x, n, required) {
          p <- x^(n - 1L)
          list(out = p * x, residuals = list(p = p))
        },
        rule = \(x, n, out, grad, required, residuals) list(x = grad * n * residuals$p)
      )
    }
    grad <- function(x) gradient(\(x) pow_fwd(x, 3L))(x)$x
    hvp <- jit(function(x, v) pushforward(grad, x)(v))
    expect_equal(as.numeric(hvp(nv_scalar(2, "f64"), nv_scalar(1, "f64"))), 12)
  })
})
