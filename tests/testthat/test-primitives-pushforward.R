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
ints <- function(lower, upper) {
  function(shape) shaped(sample(lower:upper, prod(shape), replace = TRUE), shape)
}
fixed <- function(values) {
  function(shape) shaped(values, shape)
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
# static parameters, whether to check forward-over-reverse, the tolerance of
# the comparison with finite differences, and, for a primitive that does not
# take the arguments as such, the function that calls it with them, a named
# list, in place of the parameters.
pushforward_case <- function(
  args,
  wrt = names(args),
  scalar = character(),
  shapes = list(),
  params = list(),
  hvp = TRUE,
  tolerance = 1e-6,
  call = NULL
) {
  list(
    args = args,
    wrt = wrt,
    scalar = scalar,
    shapes = shapes,
    params = params,
    hvp = hvp,
    tolerance = tolerance,
    call = call
  )
}

f64 <- function(...) nv_array(c(...), dtype = "f64")

# The results of a primitive as an unnamed list, with one entry per output.
as_outputs <- function(r) {
  if (is.list(r) && !is_arrayish(r)) unname(r) else list(r)
}

binary_case <- function(lhs = unif(-2, 2), rhs = unif(-2, 2)) {
  pushforward_case(list(lhs = lhs, rhs = rhs))
}

# `prim_convolution()` of a batch of 1-d signals `x`, with axes (batch, feature,
# spatial), and a `kernel` with axes (output feature, input feature, spatial),
# strided and padded.
conv1d_params <- list(
  x_batch_axis = 1L,
  x_feature_axis = 2L,
  x_spatial_axes = 3L,
  kernel_input_feature_axis = 2L,
  kernel_output_feature_axis = 1L,
  kernel_spatial_axes = 3L,
  output_batch_axis = 1L,
  output_feature_axis = 2L,
  output_spatial_axes = 3L,
  window_strides = 2L,
  padding = matrix(1L, 1L, 2L),
  x_dilation = 1L,
  kernel_dilation = 1L
)

# `prim_scatter()` of rows of `update` into the rows of `x`, a 5x3, that
# `indices` gives; an index of 9 is out of bounds.
scatter_rows_case <- function(indices, update_fn, unique_indices = FALSE, hvp = TRUE) {
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
    ),
    hvp = hvp
  )
}

# The pushforward rules of their own, named `<primitive>[/<variant>]`.
pushforward_cases <- list(
  # at f64 a conversion to f64 is the identity, at f32 one that changes the
  # data type
  convert = pushforward_case(list(x = unif(-2, 2)), params = list(dtype = "f64")),
  dot_general = pushforward_case(
    list(lhs = unif(-2, 2), rhs = unif(-2, 2)),
    shapes = list(rhs = c(3L, 4L)),
    params = list(contracting_axes = list(2L, 1L), batching_axes = list(integer(), integer()))
  ),
  "dot_general/batched" = pushforward_case(
    list(lhs = unif(-2, 2), rhs = unif(-2, 2)),
    shapes = list(lhs = c(2L, 3L, 2L), rhs = c(3L, 4L, 2L)),
    params = list(contracting_axes = list(2L, 1L), batching_axes = list(3L, 3L))
  ),
  convolution = pushforward_case(
    list(x = unif(-2, 2), kernel = unif(-2, 2)),
    shapes = list(x = c(2L, 3L, 6L), kernel = c(4L, 3L, 3L)),
    params = conv1d_params,
    # prim_rev has no pushforward rule yet
    hvp = FALSE
  ),
  # prim_broadcast_in_axes and prim_sum have no pushforward rule yet
  max = pushforward_case(list(x = unif(-2, 2)), shapes = list(x = c(3L, 4L)), params = list(axes = 2L), hvp = FALSE),
  "max/keep" = pushforward_case(
    list(x = unif(-2, 2)),
    shapes = list(x = c(3L, 4L)),
    params = list(axes = 1L, drop = FALSE),
    hvp = FALSE
  ),
  min = pushforward_case(list(x = unif(-2, 2)), params = list(axes = c(1L, 2L)), hvp = FALSE),
  # prim_transpose, prim_reshape, prim_static_slice and others have no
  # pushforward rule yet
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
  sort = pushforward_case(list(x = unif(-2, 2)), call = \(a) prim_sort(unname(a), axis = 2L)),
  # the key and its payloads, one of them an integer without a tangent
  "sort/payloads" = pushforward_case(
    list(key = unif(-2, 2), payload = unif(-2, 2), index = ints(1L, 3L)),
    wrt = c("key", "payload"),
    call = \(a) prim_sort(unname(a), axis = 1L, decreasing = TRUE)
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
  # prim_gather has no pushforward rule yet
  "scatter/mul" = scatter_rows_case(c(4L, 2L, 9L), "mul", unique_indices = TRUE, hvp = FALSE),
  "scatter/max" = scatter_rows_case(c(4L, 2L, 4L, 9L), "max", hvp = FALSE),
  "scatter/min" = scatter_rows_case(c(4L, 2L, 4L), "min", hvp = FALSE)
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

# Checks the pushforward rule of `prim_<name>` on `case`, at f64, on each of
# its float outputs:
#   - the tangent along each argument in `wrt` alone, and along all of them at
#     once, against a central difference;
#   - the tangent against the pullback, if the primitive has a pullback rule:
#     <u, J t> must equal <J^T u, t>;
#   - forward-over-reverse, unless `case$hvp` is `FALSE`: the tangent of the
#     gradient against a central difference of the gradient, which needs a
#     pushforward rule for whatever the pullback rule computes with -- not yet
#     with a scalar operand, whose cotangent the pullback rule sums with
#     `prim_sum()`;
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

  f <- with_formals(names(args), case$call %||% \(a) do.call(prim, c(a, case$params)))
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
  check_hvp <- has_pullback && case$hvp && !length(case$scalar)
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
    expect_setequal(unique(sub("/.*", "", names(pushforward_cases))), sub("^prim_", "", names(own)))
  })

  for (name in names(pushforward_cases)) {
    it(sprintf("of prim_%s agree with finite differences and the pullback", name), {
      withr::local_seed(1L)
      check_pushforward_rule(sub("/.*", "", name), pushforward_cases[[name]])
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
