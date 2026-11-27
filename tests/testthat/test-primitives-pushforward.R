# A primitive tagged "elementwise" gets its pushforward rule from its pullback
# rule (`pushforward_from_pullback()`), which the pullback tests cover, one
# tagged "linear" applies itself to the tangents (`pushforward_linear()`), and
# one tagged "bilinear" applies itself to each tangent in place of its operand
# (`pushforward_bilinear()`), so only each derivation is checked here, on one
# primitive as an example. Every other pushforward rule has an entry in
# `pushforward_cases`, or, for a primitive that takes functions, entries in
# `subgraph_cases`.
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
fixed <- function(values) {
  function(shape) shaped(values, shape)
}
normal <- function() {
  function(shape) shaped(rnorm(prod(shape)), shape)
}
shaped <- function(values, shape) {
  if (length(shape)) array(values, shape) else values
}

# A sampler of a batch of matrices -- the first two axes of the shape form the
# matrix, any trailing ones are batch axes -- each drawn by `draw(rows, cols)`.
matrices <- function(draw) {
  function(shape) {
    each <- lapply(seq_len(prod(shape[-(1:2)])), \(i) draw(shape[[1L]], shape[[2L]]))
    array(unlist(each), shape)
  }
}
symmetric <- function() {
  matrices(\(n, cols) {
    x <- matrix(rnorm(n * n), n)
    x + t(x)
  })
}
positive_definite <- function() {
  matrices(\(n, cols) {
    x <- matrix(rnorm(n * n), n)
    tcrossprod(x) + diag(n)
  })
}
# the triangles of which are well-conditioned
diagonally_dominant <- function() {
  matrices(\(n, cols) matrix(runif(n * n, -1, 1), n) + diag(n + 1, n))
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
# samplers of the directions to differentiate in where they are not `normal()`
# -- for an argument whose tangents the primitive restricts, such as to
# symmetric matrices -- static parameters, a function `fn` of the arguments,
# by name, that calls the primitive in place of calling it with them, a
# function `reference` that computes the same as `fn` without the primitive,
# for a primitive without a pullback rule, and whether to check
# forward-over-reverse.
pushforward_case <- function(
  args,
  wrt = names(args),
  scalar = character(),
  shapes = list(),
  directions = list(),
  params = list(),
  fn = NULL,
  reference = NULL,
  hvp = TRUE
) {
  list(
    args = args,
    wrt = wrt,
    scalar = scalar,
    shapes = shapes,
    directions = directions,
    params = params,
    fn = fn,
    reference = reference,
    hvp = hvp
  )
}

f64 <- function(...) nv_array(c(...), dtype = "f64")

# The results of a primitive as an unnamed list, with one entry per output.
as_outputs <- function(r) {
  if (is.list(r) && !is_arrayish(r)) unname(r) else list(r)
}

# `prim_scatter()` of rows of `update` into the rows of `x`, a 5x3, that
# `indices` gives; an index of 9 is out of bounds.
scatter_rows_case <- function(indices, update_fn, unique_indices = FALSE) {
  pushforward_case(
    list(x = unif(-2, 2), scatter_indices = fixed(indices), update = unif(-2, 2)),
    wrt = c("x", "update"),
    shapes = list(x = c(5L, 3L), scatter_indices = c(length(indices), 1L), update = c(length(indices), 3L)),
    params = list(
      update_window_axes = 2L,
      inserted_window_axes = 1L,
      x_batching_axes = integer(),
      scatter_indices_batching_axes = integer(),
      scatter_axes_to_x_axes = 1L,
      index_vector_axis = 2L,
      unique_indices = unique_indices,
      update_fn = update_fn
    )
  )
}

# The pushforward rules of their own, named `<primitive>[/<variant>]`.
pushforward_cases <- list(
  # at f64 a conversion to f64 is the identity, at f32 one that changes the
  # data type
  convert = pushforward_case(list(x = unif(-2, 2)), params = list(dtype = "f64")),
  print = pushforward_case(list(x = unif(-2, 2))),
  # an output that is not a float among those that are
  optimization_barrier = pushforward_case(
    list(x = unif(-2, 2), i = ints(-3L, 3L), y = unif(-2, 2)),
    wrt = c("x", "y"),
    shapes = list(y = 4L),
    fn = function(x, i, y) prim_optimization_barrier(list(x = x, i = i, y = y))
  ),
  which_max = pushforward_case(list(x = unif(-2, 2)), params = list(axis = 1L)),
  which_min = pushforward_case(list(x = unif(-2, 2)), params = list(axis = 2L)),
  bitcast_convert = pushforward_case(list(x = unif(-2, 2)), params = list(dtype = "i64")),
  max = pushforward_case(list(x = unif(-2, 2)), shapes = list(x = c(3L, 4L)), params = list(axes = 2L)),
  "max/keep" = pushforward_case(
    list(x = unif(-2, 2)),
    shapes = list(x = c(3L, 4L)),
    params = list(axes = 1L, drop = FALSE)
  ),
  min = pushforward_case(list(x = unif(-2, 2)), params = list(axes = c(1L, 2L))),
  # the pullback rule computes with prim_cumprod, which has no pushforward rule
  prod = pushforward_case(
    list(x = unif(0.5, 1.5)),
    shapes = list(x = c(3L, 4L)),
    params = list(axes = 1L),
    hvp = FALSE
  ),
  "prod/zero" = pushforward_case(
    list(x = fixed(c(0.5, 0, 1.5, 0.8, 1.2, 0, 0.9, 1.1, 0, 0.7, 1.3, 0.6))),
    shapes = list(x = c(3L, 4L)),
    params = list(axes = 2L),
    hvp = FALSE
  ),
  cummax = pushforward_case(list(x = unif(-2, 2)), shapes = list(x = c(4L, 3L)), params = list(axis = 1L)),
  cummin = pushforward_case(list(x = unif(-2, 2)), shapes = list(x = c(3L, 4L)), params = list(axis = 2L)),
  sort = pushforward_case(list(x = unif(-2, 2)), fn = function(x) prim_sort(list(x), axis = 2L)),
  # the key and its payloads, one of them an integer without a tangent
  "sort/payloads" = pushforward_case(
    list(key = unif(-2, 2), payload = unif(-2, 2), index = ints(1L, 3L)),
    wrt = c("key", "payload"),
    fn = function(key, payload, index) prim_sort(list(key, payload, index), axis = 1L, decreasing = TRUE)
  ),
  top_k = pushforward_case(list(x = unif(-2, 2)), shapes = list(x = c(5L, 3L)), params = list(k = 3L, indices = TRUE)),
  "top_k/values" = pushforward_case(
    list(x = unif(-2, 2)),
    shapes = list(x = c(5L, 3L)),
    params = list(k = 2L, indices = FALSE)
  ),
  scatter = scatter_rows_case(c(4L, 2L), "replace", unique_indices = TRUE),
  "scatter/replace" = scatter_rows_case(c(4L, 2L, 4L, 9L), "replace"),
  "scatter/add" = scatter_rows_case(c(4L, 2L, 4L, 9L), "add"),
  "scatter/sub" = scatter_rows_case(c(4L, 2L, 4L), "sub"),
  "scatter/mul" = scatter_rows_case(c(4L, 2L, 9L), "mul", unique_indices = TRUE),
  "scatter/max" = scatter_rows_case(c(4L, 2L, 4L, 9L), "max"),
  "scatter/min" = scatter_rows_case(c(4L, 2L, 4L), "min"),
  # The pushforward rule of `prim_chol()` takes the factor as a function of the
  # symmetric matrices, as its pullback rule does, so the directions are
  # symmetric.
  chol = pushforward_case(
    list(x = positive_definite()),
    shapes = list(x = c(3L, 3L, 2L)),
    directions = list(x = symmetric()),
    params = list(lower = TRUE)
  ),
  "chol/upper" = pushforward_case(
    list(x = positive_definite()),
    shapes = list(x = c(3L, 3L)),
    directions = list(x = symmetric()),
    params = list(lower = FALSE)
  ),
  triangular_solve = pushforward_case(
    list(a = diagonally_dominant(), b = unif(-2, 2)),
    shapes = list(a = c(3L, 3L, 2L), b = c(3L, 2L, 2L)),
    params = list(left_side = TRUE, lower = TRUE, unit_diagonal = FALSE, transpose_a = FALSE)
  ),
  "triangular_solve/right" = pushforward_case(
    list(a = diagonally_dominant(), b = unif(-2, 2)),
    shapes = list(a = c(3L, 3L), b = c(2L, 3L)),
    params = list(left_side = FALSE, lower = FALSE, unit_diagonal = TRUE, transpose_a = TRUE)
  ),
  "triangular_solve/upper" = pushforward_case(
    list(a = diagonally_dominant(), b = unif(-2, 2)),
    shapes = list(a = c(3L, 3L), b = c(3L, 2L)),
    params = list(left_side = TRUE, lower = FALSE, unit_diagonal = FALSE, transpose_a = TRUE)
  ),
  lu = pushforward_case(list(x = unif(-2, 2)), shapes = list(x = c(3L, 3L, 2L))),
  "lu/tall" = pushforward_case(list(x = unif(-2, 2)), shapes = list(x = c(4L, 2L))),
  "lu/wide" = pushforward_case(list(x = unif(-2, 2)), shapes = list(x = c(2L, 4L)))
)

# The pushforward rules of the primitives that take functions, named
# `<primitive>/<what the case covers>`, each called through `fn`. What its
# functions use is closed over, the arguments of `fn` among it. A while loop,
# which has no pullback rule, is checked against the same loop unrolled.
subgraph_cases <- list(
  # a branch returning a value it closes over as it is, and a result that is
  # not a float
  "if/true" = pushforward_case(
    list(p = fixed(TRUE), x = unif(-2, 2), y = unif(-2, 2)),
    wrt = c("x", "y"),
    scalar = "p",
    fn = function(p, x, y) nv_if(p, \() list(x * y, sin(x), x < y), \() list(cos(y), x, x > y))
  ),
  "if/false" = pushforward_case(
    list(p = fixed(FALSE), x = unif(-2, 2), y = unif(-2, 2)),
    wrt = c("x", "y"),
    scalar = "p",
    fn = function(p, x, y) nv_if(p, \() list(x * y, sin(x), x < y), \() list(cos(y), x, x > y))
  ),
  "if/nested" = pushforward_case(
    list(p = fixed(TRUE), q = fixed(FALSE), x = unif(-2, 2), y = unif(-2, 2)),
    wrt = c("x", "y"),
    scalar = c("p", "q"),
    fn = function(p, q, x, y) nv_if(p, \() nv_if(q, \() x * y, \() sin(x) * y), \() x)
  ),
  "if/scan" = pushforward_case(
    list(p = fixed(TRUE), h = unif(-1, 1), xs = unif(-1, 1)),
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
  "while" = pushforward_case(
    list(y = unif(-1, 1), w = unif(-1, 1)),
    shapes = list(y = 2L, w = 2L),
    fn = function(y, w) nv_while(list(i = 0L, y = y), \(i, y) i < 3L, \(i, y) list(i = i + 1L, y = sin(y) * w + y))$y,
    reference = function(y, w) {
      for (i in 1:3) {
        y <- sin(y) * w + y
      }
      y
    }
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
    },
    reference = function(a, w) {
      t <- nv_fill(0, integer(), dtype = dtype(a))
      b <- nv_fill(0, 2L, dtype = dtype(a))
      for (i in 1:3) {
        b <- b + t * a
        t <- t + 1
        a <- a * w
      }
      list(t, a, b)
    }
  ),
  "while/scan" = pushforward_case(
    list(y = unif(-1, 1), xs = unif(-1, 1)),
    shapes = list(y = 2L),
    fn = function(y, xs) {
      body <- function(i, y) list(i = i + 1L, y = nv_scan(y, xs, \(c, x) list(carry = sin(c) * x, out = NULL))$carry)
      nv_while(list(i = 0L, y = y), \(i, y) i < 2L, body)$y
    },
    reference = function(y, xs) {
      for (i in 1:2) {
        y <- nv_scan(y, xs, \(c, x) list(carry = sin(c) * x, out = NULL))$carry
      }
      y
    }
  ),
  "while/nested" = pushforward_case(
    list(y = unif(-1, 1), w = unif(-1, 1)),
    shapes = list(y = 2L, w = 2L),
    fn = function(y, w) {
      inner <- function(y) nv_while(list(j = 0L, z = y), \(j, z) j < 2L, \(j, z) list(j = j + 1L, z = sin(z) + w))$z
      nv_while(list(i = 0L, y = y), \(i, y) i < 2L, \(i, y) list(i = i + 1L, y = inner(y) * y))$y
    },
    reference = function(y, w) {
      for (i in 1:2) {
        z <- y
        for (j in 1:2) {
          z <- sin(z) + w
        }
        y <- z * y
      }
      y
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
    list(x = unif(0.5, 2), n = fixed(3L)),
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

# Whether the pushforward rule of `prim` is derived from its tags: from its
# pullback rule (`pushforward_from_pullback()`), or as that of a linear
# (`pushforward_linear()`) or a bilinear primitive (`pushforward_bilinear()`).
derived_from_tags <- function(prim) {
  rule <- body(prim[["pushforward"]])
  identical(rule, body(pushforward_from_pullback(prim[["pullback"]]))) ||
    identical(rule, body(pushforward_linear(prim))) ||
    identical(rule, body(pushforward_bilinear(prim)))
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

# Checks the pushforward rule of `prim_<name>` on `case`, at f64, on each of
# its float outputs, against the pullback of `case$reference`, or of the
# primitive itself, which does not use the pushforward rules:
#   - the tangent along each argument in `wrt` alone, and along all of them at
#     once: element i of J t is <J^T e_i, t>, with J^T e_i the pullback of the
#     i-th unit cotangent;
#   - forward-over-reverse against reverse-over-reverse, if the primitive has a
#     pullback rule and `case$hvp` is not `FALSE`: the tangent of the gradient
#     of <u, f> along t is its Hessian times t, and as the Hessian is
#     symmetric, so is the pullback of the gradient along t. That needs both
#     rules for whatever the pullback rule computes with;
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
  directions <- lapply(case$wrt, \(nm) (case$directions[[nm]] %||% normal())(shape_of(nm)))
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

  # `fn` with its float outputs alone, as an unnamed list
  float_outputs <- function(fn) with_formals(names(args), \(a) as_outputs(do.call(fn, a))[is_float])
  g <- float_outputs(f)
  reference <- float_outputs(case$reference %||% f)
  out_shapes <- lapply(outs[is_float], shape)
  as_cotangent <- \(values) Map(\(v, s) anvl_array(shaped(v, s), dtype = "f64"), values, out_shapes)
  zeros <- lapply(out_shapes, \(s) numeric(prod(s)))
  # one unit cotangent per element of each float output
  unit_output <- rep(seq_along(out_shapes), lengths(zeros))
  units <- Map(
    \(k, i) as_cotangent(replace(zeros, k, list(replace(zeros[[k]], i, 1)))),
    unit_output,
    sequence(lengths(zeros))
  )
  u <- as_cotangent(lapply(zeros, \(z) rnorm(length(z))))
  check_hvp <- !is.null(prim[["pullback"]]) && case$hvp
  run <- jit(function(x, t, u, units) {
    along <- c(lapply(names(t), \(nm) t[nm]), list(t))
    # the rows of the Jacobian, J^T e_i
    rows <- lapply(units, do.call(pullback, c(list(fixing(reference, x, t)), x[names(t)])))
    # element i of J d is <J^T e_i, d>
    reference_along <- function(d) {
      lapply(rows, \(row) Reduce(`+`, Map(\(nm) sum(row[[nm]] * d[[nm]]), names(d))))
    }
    c(
      list(
        tangent = lapply(along, \(d) tangent_along(g, x, d)),
        reference = lapply(along, reference_along)
      ),
      if (check_hvp) {
        loss <- with_formals(names(x), \(a) Reduce(`+`, Map(\(o, w) sum(o * w), do.call(g, a), u)))
        grad <- gradient(loss, wrt = names(t))
        list(
          hvp = tangent_along(grad, x, t),
          hvp_reverse = do.call(pullback, c(list(fixing(grad, x, t)), x[names(t)]))(t)
        )
      }
    )
  })
  res <- run(x, t, u, units)

  labels <- c(paste("along", names(t)), "along all")
  out_dtypes <- lapply(outs[is_float], dtype)
  for (i in seq_along(res$tangent)) {
    for (k in seq_along(out_dtypes)) {
      label <- if (length(out_dtypes) > 1L) sprintf("%s, output %d", labels[[i]], k) else labels[[i]]
      expect_equal(dtype(res$tangent[[i]][[k]]), out_dtypes[[k]], label = label)
      expect_equal(
        as.vector(as_array(res$tangent[[i]][[k]])),
        vapply(res$reference[[i]][unit_output == k], as_array, numeric(1L)),
        tolerance = 1e-10,
        label = label
      )
    }
  }
  for (nm in names(res$hvp)) {
    expect_equal(as_array(res$hvp[[nm]]), as_array(res$hvp_reverse[[nm]]), tolerance = 1e-10, label = paste("hvp", nm))
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
    own <- Filter(\(p) !is.null(p[["pushforward"]]) && !derived_from_tags(p), prims)
    cased <- unique(sub("/.*", "", c(names(pushforward_cases), names(subgraph_cases))))
    expect_setequal(cased, sub("^prim_", "", names(own)))
  })

  for (name in names(pushforward_cases)) {
    it(sprintf("of prim_%s agree with the pullback", name), {
      withr::local_seed(1L)
      # `prim_print()` prints at every run
      utils::capture.output(check_pushforward_rule(sub("/.*", "", name), pushforward_cases[[name]]))
    })
  }

  for (name in names(subgraph_cases)) {
    it(sprintf("of %s agree with the pullback", name), {
      withr::local_seed(1L)
      check_pushforward_rule(sub("/.*", "", name), subgraph_cases[[name]])
    })
  }

  it("of prim_scatter differentiate an update_fn given by name, with respect to the update of a multiplying one only with unique indices", {
    scatter <- function(x, update, update_fn) {
      prim_scatter(
        x,
        nv_array(matrix(c(2L, 2L)), dtype = "i32"),
        update,
        update_window_axes = integer(),
        inserted_window_axes = 1L,
        x_batching_axes = integer(),
        scatter_indices_batching_axes = integer(),
        scatter_axes_to_x_axes = 1L,
        index_vector_axis = 2L,
        update_fn = update_fn
      )
    }
    push <- function(update_fn, ...) {
      jit(function(x, update) {
        pushforward(\(x, update) scatter(x, update, update_fn), x, update)(...)
      })(nv_array(c(1, 2, 3), dtype = "f64"), nv_array(c(10, 20), dtype = "f64"))
    }
    expect_error(push(\(old, new) old + new, f64(1, 1, 1)), "not implemented for an `update_fn` that is a function")
    expect_error(push("mul", NULL, f64(1, 1)), "needs `unique_indices = TRUE`")
    # the tangent of `x` alone needs no unique indices: it is multiplied by
    # both updates to its position
    expect_equal(push("mul", f64(1, 1, 1)), f64(1, 200, 1))
  })

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

  it("of prim_chol count a tangent by its symmetric part, as its pullback rule does", {
    withr::local_seed(1L)
    x <- nv_array(positive_definite()(c(3L, 3L)), dtype = "f64")
    dx <- normal()(c(3L, 3L))
    for (lower in c(TRUE, FALSE)) {
      along <- jit(function(x, dx) pushforward(\(x) prim_chol(x, lower = lower), x)(dx))
      expected <- as_array(along(x, nv_array((dx + t(dx)) / 2, dtype = "f64")))
      expect_equal(as_array(along(x, nv_array(dx, dtype = "f64"))), expected)
    }
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

describe("pushforward_bilinear", {
  it("derives a pushforward rule that agrees with the pullback, e.g. for prim_dot_general", {
    withr::local_seed(1L)
    check_pushforward_rule(
      "dot_general",
      pushforward_case(
        list(lhs = unif(-2, 2), rhs = unif(-2, 2)),
        shapes = list(rhs = c(3L, 4L)),
        params = list(contracting_axes = list(2L, 1L), batching_axes = list(integer(), integer()))
      )
    )
  })
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
