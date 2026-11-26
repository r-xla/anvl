#' @title Jacobians
#' @description
#' Return a new function that computes the Jacobian of `f`: the derivative of
#' every element of its value with respect to every element of its arguments.
#' `f` can return any value, an array or a (nested) list of arrays. The
#' returned function has the same signature as `f`.
#'
#' `mode` says how the Jacobian is computed. `"backward"` pulls back a unit
#' cotangent for every element of the value (see [`pullback()`]), and
#' `"forward"` pushes forward a unit tangent for every element of the
#' arguments in `wrt` (see [`pushforward()`]), all of them at once with
#' [`vectorize()`]. So `"backward"` is the cheaper one for a value with fewer
#' elements than the arguments, and `"forward"` for one with more.
#' @param f (`function`)\cr
#'   Function to differentiate. It can return any (nested) list of arrays.
#' @param wrt (`character` | `integer` | `NULL`)\cr
#'   Names or positions of the arguments to differentiate with respect to, as
#'   for [`gradient()`]: float arrays, or lists of them, passed as arrays with
#'   a data type. If `NULL` (the default), all arguments.
#' @param mode (`character(1)`)\cr
#'   `"backward"` (the default) or `"forward"`: how to compute the Jacobian,
#'   see Description.
#' @return (`function`)\cr
#'   Has the same formals as `f` and must be called inside [`jit()`]. It
#'   returns the value of `f` with each of its float arrays `y` replaced by the
#'   derivatives of `y`: a named `list` with an entry per argument of `f` (or
#'   per `wrt` entry), structured like that argument, as [`gradient()`]
#'   returns it. In place of each array `a` of the argument, it holds the
#'   derivative of `y` with respect to `a`, of shape `c(shape(y), shape(a))`
#'   and the data type of `y`: its element at `[i, j]` is the derivative of
#'   `y[i]` with respect to `a[j]`, for `i` and `j` indices into the axes of
#'   `y` and `a`. An array of the value that is not a float has no
#'   derivative, and gets `NULL`.
#' @seealso `r roxy_article("autodiff")`, [`hessian()`], [`gradient()`],
#'   [`pullback()`], [`pushforward()`]
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' f <- function(x) x^2L
#' x <- nv_array(c(1, 2, 3))
#' # 2 * x on the diagonal
#' jit(jacobian(f))(x)
#' jit(jacobian(f, mode = "forward"))(x)
#'
#' # a block for each argument
#' g <- function(x, y) x * sum(y)
#' jit(jacobian(g))(x, nv_array(c(1, 2)))
#'
#' # a value with several parts, each with its derivatives
#' h <- function(x) list(square = x^2L, total = sum(x))
#' jit(jacobian(h))(x)
jacobian <- function(f, wrt = NULL, mode = "backward") {
  assert_function(f)
  wrt <- resolve_transformation_args(f, wrt, "wrt")
  assert_choice(mode, c("backward", "forward"))
  jacobian_fn(f, wrt, mode, "jacobian")
}

#' @title Hessians
#' @description
#' Return a new function that computes the Hessian of `f`, a function that
#' returns a single float scalar: its second derivatives with respect to
#' every two elements of its arguments. It is the [`jacobian()`] of the
#' [`gradient()`] of `f`, forward-over-reverse: the gradient in backward mode,
#' and its Jacobian in forward mode, vectorized over a unit tangent for every
#' element of the arguments. The returned function has the same signature as
#' `f`.
#' @inheritParams jacobian
#' @param f (`function`)\cr
#'   Function to differentiate. Must return a single float scalar.
#' @return (`function`)\cr
#'   Has the same formals as `f` and must be called inside [`jit()`]. It
#'   returns a named `list` with an entry per argument of `f` (or per `wrt`
#'   entry), structured like that argument, as [`gradient()`] returns it, with
#'   each array of it replaced by that same `list` again: the second
#'   derivatives of `f` with respect to the arrays `a` and `b` are at
#'   `[[a]][[b]]`, of shape `c(shape(a), shape(b))`.
#' @seealso `r roxy_article("autodiff")`, [`jacobian()`], [`gradient()`]
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' f <- function(x) sum(x^3L)
#' x <- nv_array(c(1, 2, 3))
#' # 6 * x on the diagonal
#' jit(hessian(f))(x)
#'
#' # the second derivatives with respect to two arguments
#' g <- function(x, y) sum(x^2L * y)
#' jit(hessian(g))(x, nv_array(c(4, 5, 6)))
hessian <- function(f, wrt = NULL) {
  assert_function(f)
  wrt <- resolve_transformation_args(f, wrt, "wrt")
  jacobian_fn(gradient(f, wrt), wrt, "forward", "hessian")
}

# The function `jacobian()` or `hessian()` (named by `fn`) returns for `f`.
jacobian_fn <- function(f, wrt, mode, fn) {
  transformation_fn(f, function(args) {
    assert_in_trace(fn)
    trace_jacobian(f, args, wrt, mode, fn)
  })
}

# The Jacobian of `f` at the arguments `args` of the call, with respect to the
# arguments `wrt` names, in the mode `mode` (see `jacobian()`).
trace_jacobian <- function(f, args, wrt, mode, fn) {
  prep <- prepare_gradient_args(args, wrt)
  wrt_leaves <- prep$args_flat[prep$is_wrt_flat]
  traced <- if (mode == "backward") trace_pullback(f, prep, fn)$flat else trace_pushforward(f, prep, fn)$flat
  is_float <- vapply(traced$value, \(v) is_dtype_float(dtype(v)), logical(1L))
  # one list per float array of the value, with a block per array in `wrt`
  blocks <- if (mode == "backward") {
    jacobian_backward(traced$pullback, traced$value, is_float, wrt_leaves)
  } else {
    jacobian_forward(traced$pushforward, is_float, wrt_leaves)
  }
  derivatives <- vector("list", length(is_float))
  derivatives[is_float] <- lapply(blocks, function(blocks_y) {
    leaves <- vector("list", length(prep$args_flat))
    leaves[prep$is_wrt_flat] <- blocks_y
    jac <- unflatten(prep$in_tree, leaves)
    if (is.null(wrt)) jac else jac[names(jac) %in% wrt]
  })
  unflatten(traced$out_tree, derivatives)
}

# The blocks of the Jacobian in reverse mode: the pullback `pull` (see
# `trace_pullback()`) of a unit cotangent of every element of the float arrays
# of the value `values` (those `is_float` selects), vectorized over them. One
# list per float array of the value, with a block per array of `wrt_leaves`.
jacobian_backward <- function(pull, values, is_float, wrt_leaves) {
  ys <- values[is_float]
  # nothing to map over
  if (!length(ys)) {
    return(list())
  }
  # each of shape c(n, shape(a)), for n unit cotangents
  pulled <- vectorize(
    function(basis) {
      cts <- vector("list", length(values))
      cts[is_float] <- basis
      pull(cts)
    },
    out_axes = 1L
  )(unit_basis(ys))
  sizes <- basis_sizes(ys)
  Map(
    function(y, size, offset) {
      Map(
        function(p, a) {
          block <- prim_reshape(slice_axis(p, 1L, offset, size), c(shape(y), shape(a)))
          if (dtype(block) == dtype(y)) block else prim_convert(block, dtype(y))
        },
        pulled,
        wrt_leaves
      )
    },
    ys,
    sizes,
    basis_offsets(sizes)
  )
}

# The blocks of the Jacobian in forward mode: the pushforward `push` (see
# `trace_pushforward()`) of a unit tangent of every element of the arrays
# `wrt_leaves`, vectorized over them. One list per float array of the value
# (those `is_float` selects), with a block per array of `wrt_leaves`.
jacobian_forward <- function(push, is_float, wrt_leaves) {
  # nothing to map over
  if (!length(wrt_leaves)) {
    return(rep(list(list()), sum(is_float)))
  }
  # each of shape c(shape(y), n), for n unit tangents
  pushed <- vectorize(function(basis) push(basis)[is_float])(unit_basis(wrt_leaves))
  sizes <- basis_sizes(wrt_leaves)
  lapply(pushed, function(p) {
    k <- naxes(p)
    Map(
      function(a, size, offset) prim_reshape(slice_axis(p, k, offset, size), c(shape(p)[-k], shape(a))),
      wrt_leaves,
      sizes,
      basis_offsets(sizes)
    )
  })
}

# `x` sliced along its axis `axis` to the `size` elements after the first
# `offset`.
slice_axis <- function(x, axis, offset, size) {
  start <- rep(1L, naxes(x))
  end <- shape(x)
  start[[axis]] <- offset + 1L
  end[[axis]] <- offset + size
  prim_static_slice(x, start, end, rep(1L, naxes(x)))
}

# The unit vectors of the elements of the arrays `xs`, taken one after the
# other in column-major order: one array per array of `xs`, of its shape and
# data type with a trailing axis of all the elements, whose entry k along that
# axis is 1 at the k-th element and 0 elsewhere.
unit_basis <- function(xs) {
  sizes <- basis_sizes(xs)
  total <- sum(sizes)
  Map(
    function(x, size, offset) {
      grid <- c(size, total)
      one_hot <- prim_eq(
        prim_iota(1L, grid, default_int(), start = offset + 1L),
        prim_iota(2L, grid, default_int())
      )
      prim_reshape(prim_convert(one_hot, dtype(x)), c(shape(x), total))
    },
    xs,
    sizes,
    basis_offsets(sizes)
  )
}

# The number of elements of each of the arrays `xs`.
basis_sizes <- function(xs) {
  vapply(xs, \(x) as.integer(prod(shape(x))), integer(1L))
}

# The number of elements before each array, of `sizes` elements each.
basis_offsets <- function(sizes) {
  cumsum(c(0L, sizes))[seq_along(sizes)]
}
