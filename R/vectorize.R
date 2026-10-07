#' @title Vectorize a Function
#' @description
#' Return a new function that maps `f` over an axis of its arguments, like
#' [base::Vectorize()]. Unlike `Vectorize()`, it does not loop: the returned
#' function traces `f` once and turns every operation of it into one that works
#' on the whole batch, so `f` can be written for a single instance and still
#' run as one program over all of them.
#'
#' The arguments in `args` are mapped over: their axis `axis` is the batch axis,
#' and `f` sees one slice of it at a time, with that axis dropped. Every leaf of a
#' nested argument is mapped over. The other arguments are passed to each
#' application of `f` unchanged. Every output of `f` is stacked along `axis`.
#' @param f (`function`)\cr
#'   Function to vectorize.
#' @param args (`character` | `integer` | `NULL`)\cr
#'   Names or positions of the arguments to map over, like `vectorize.args` of
#'   [base::Vectorize()]. If `NULL` (the default), every argument that is an
#'   array is mapped over, and plain R values -- such as static arguments of
#'   the enclosing [`jit()`] -- are passed on unchanged.
#' @param axis (`integer(1)`)\cr
#'   The axis of the arguments in `args` that is mapped over, and the axis of
#'   the outputs along which the results are stacked. All arguments in `args`
#'   must have the same size along it, and every output must have room for it:
#'   an output with `n` axes per slice can be stacked along axes `1` to `n + 1`.
#'   For another layout, transpose the arguments or results.
#' @return (`function`)\cr
#'   Has the same formals as `f` and must be called inside [`jit()`]. It returns
#'   what `f` returns, with every array gaining the batch axis at `axis`.
#' @section Supported Primitives:
#' A primitive applied to a value that is mapped over needs a `vectorize` rule
#' (see [`rule_vectorize()`]); calling one that has none raises an error.
#' Values that are not mapped over can go through any primitive. Without a rule
#' so far are control flow ([`prim_if()`], [`prim_while()`], [`prim_scan()`]),
#' [`prim_reduce()`], [`prim_rng_bit_generator()`], [`prim_convolution()`] and
#' the decompositions [`prim_qr()`], [`prim_lu()`], [`prim_svd()`] and
#' [`prim_eigh()`].
#' @seealso [`rule_vectorize()`], [`gradient()`]
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' # f is written for a single vector
#' f <- function(x, y) sum(x * y)
#' x <- nv_array(matrix(1:6, nrow = 3), dtype = "f32")
#'
#' # one dot product per row of x, against the same y
#' jit(vectorize(f, args = "x"))(x, nv_array(c(1, 2), dtype = "f32"))
#'
#' # map over the columns instead: each column divided by its sum
#' jit(vectorize(function(x) x / sum(x), axis = 2L))(x)
vectorize <- function(f, args = NULL, axis = 1L) {
  assert_function(f)
  mapped_args <- resolve_transformation_args(f, args, "args")
  axis <- as.integer(checkmate::assert_int(axis, lower = 1L))
  transformation_fn(f, function(call_args) {
    assert_in_trace("vectorize")
    vectorize_call(f, call_args, mapped_args, axis)
  })
}

# Traces `f` on one slice of the arguments `mapped_args` names and replays the
# graph into the current descriptor, batched over `axis`.
vectorize_call <- function(f, call_args, mapped_args, axis) {
  missing_args <- setdiff(mapped_args, names(call_args))
  if (length(missing_args)) {
    cli_abort("Cannot map over {.arg {missing_args}}: {cli::qty(length(missing_args))}{?it was/they were} not passed.")
  }
  args_flat <- flatten(call_args)
  in_tree <- build_tree(call_args)
  is_mapped <- if (is.null(mapped_args)) {
    vapply(args_flat, \(x) is_graph_box(x) || is_anvl_array(x), logical(1L))
  } else {
    pjrt::tree_leaf_mask(in_tree, mapped_args)
  }
  # The argument each flat leaf belongs to, for messages.
  sizes <- pjrt::tree_child_sizes(in_tree)
  leaf_args <- rep(pjrt::tree_child_names(in_tree) %||% rep("", length(sizes)), times = sizes)

  size <- NULL
  trace_args <- args_flat
  for (i in which(is_mapped)) {
    x <- args_flat[[i]]
    if (!is_arrayish(x)) {
      cli_abort(c(
        "Can only map over arrays.",
        x = "{.arg {leaf_args[[i]]}} is {.cls {class(x)[1L]}}."
      ))
    }
    aval <- to_abstract(x)
    x_shape <- shape(aval)
    if (length(x_shape) < axis) {
      cli_abort(c(
        "Every argument mapped over must have an axis {.val {axis}}.",
        x = "{.arg {leaf_args[[i]]}} has shape {shape_repr(x_shape)}.",
        i = "Name the arguments to map over with {.arg args}; an argument of the enclosing {.fn jit} that is not static is an array here."
      ))
    }
    if (is.null(size)) {
      size <- x_shape[[axis]]
    } else if (x_shape[[axis]] != size) {
      cli_abort(c(
        "Every argument mapped over must have the same size along axis {.val {axis}}.",
        x = "{.arg {leaf_args[[i]]}} has size {.val {x_shape[[axis]]}}, an earlier one {.val {size}}."
      ))
    }
    trace_args[[i]] <- if (is_rdata(aval)) {
      RData(shape = x_shape[-axis], r_type = aval$r_type)
    } else {
      AbstractArray(dtype = aval$dtype, shape = x_shape[-axis])
    }
  }
  if (is.null(size)) {
    cli_abort("{.fn vectorize} needs at least one argument to map over.")
  }

  graph <- trace_fn(f, args_flat = trace_args, in_tree = in_tree)
  is_input <- !graph$is_static_flat
  operands <- Map(
    function(box, mapped) if (mapped) move_axis(box, axis, 1L) else box,
    graph_operands(graph, args_flat),
    is_mapped[is_input]
  )

  outs <- graph_vectorize(graph, operands, is_mapped[is_input], size)
  outs <- lapply(seq_along(outs), function(j) {
    out <- outs[[j]]
    if (naxes(out) < axis) {
      cli_abort(c(
        "Every output must have room for the batch axis at axis {.val {axis}}.",
        x = "Output {.val {j}} has shape {shape_repr(shape(out)[-1L])}, so it can be stacked along axes {.val {1L}} to {.val {naxes(out)}} only.",
        i = "Stack along an axis it has room for, and transpose the result if needed."
      ))
    }
    move_axis(out, from = 1L, to = axis)
  })
  unflatten(graph$out_tree, outs)
}

# Replays `graph` into the current descriptor as a function of `inputs` (boxes,
# one per graph input), where those that `batched` flags carry a leading batch
# axis of size `size`. A call none of whose inputs is batched is replayed as it
# is; every other call goes through its primitive's vectorize rule, and its
# outputs are batched. Returns the boxes of the outputs, all batched.
graph_vectorize <- function(graph, inputs, batched, size) {
  desc <- current_descriptor()
  boxes <- hashtab()
  is_batched <- hashtab()
  for (i in seq_along(graph$inputs)) {
    boxes[[graph$inputs[[i]]]] <- inputs[[i]]
    is_batched[[graph$inputs[[i]]]] <- batched[[i]]
  }
  const_boxes <- register_consts(desc, graph$constants)
  for (i in seq_along(const_boxes)) {
    boxes[[graph$constants[[i]]]] <- const_boxes[[i]]
  }
  box_of <- function(g) {
    if (is_graph_literal(g)) {
      return(desc$gval_to_box[[g]] %||% GraphBox(g, desc))
    }
    boxes[[g]]
  }
  batched_of <- function(g) !is_graph_literal(g) && isTRUE(is_batched[[g]])

  for (call in graph$statements) {
    in_boxes <- lapply(call$inputs, box_of)
    in_batched <- vapply(call$inputs, batched_of, logical(1L))
    if (!any(in_batched)) {
      out_boxes <- replay_statement(desc, call, lapply(in_boxes, \(b) b$gnode))
    } else {
      out_boxes <- apply_vectorize_rule(call$primitive, in_boxes, in_batched, call$params, size)
      check_vectorized_outputs(call, out_boxes, size)
    }
    for (j in seq_along(call$outputs)) {
      boxes[[call$outputs[[j]]]] <- out_boxes[[j]]
      is_batched[[call$outputs[[j]]]] <- any(in_batched)
    }
  }

  lapply(graph$outputs, function(g) batch_operand(box_of(g), batched_of(g), size))
}

apply_vectorize_rule <- function(primitive, inputs, batched, params, size) {
  rule <- primitive[["vectorize"]]
  if (is.null(rule)) {
    cli_abort(c(
      "{.fn vectorize} does not support {.fn {paste0('prim_', primitive$name)}} yet.",
      i = "Only values that are not mapped over can go through it."
    ))
  }
  if (!is.null(rule$fn)) {
    out <- rule$fn(inputs, batched, params, size)
    return(if (is_graph_box(out)) list(out) else out)
  }
  prim_fn <- primitive$fn
  name <- paste0("prim_", primitive$name)
  if (is.null(prim_fn) || !all(names(params) %in% formalArgs(prim_fn))) {
    cli_abort(c(
      "The vectorize rule of {.fn {name}} declares its parameters, but cannot call it with them.",
      i = "A declared rule calls the primitive {.fn new_primitive} made with its parameters as arguments of the same names; otherwise, give it a rule of its own with {.code rule_vectorize(fn = )}."
    ))
  }

  for (i in rule$unbatched) {
    if (batched[[i]]) {
      cli_abort(
        "{.fn vectorize} cannot map over operand {i} of {.fn {name}}."
      )
    }
  }
  operands <- batch_operands(inputs, batched, size, scalar = rule$scalar, unbatched = rule$unbatched)
  for (nm in names(rule$params)) {
    params[[nm]] <- rule$params[[nm]](params[[nm]], size)
  }
  out <- do.call(prim_fn, c(operands, params))
  if (is_graph_box(out)) list(out) else out
}

# A vectorize rule must return what the call returned, each with the batch axis
# in front.
check_vectorized_outputs <- function(call, outputs, size) {
  name <- paste0("prim_", call$primitive$name)
  if (length(outputs) != length(call$outputs)) {
    cli_abort(
      "Internal error: the vectorize rule of {.fn {name}} returned {length(outputs)} output{?s}, not {length(call$outputs)}."
    )
  }
  for (j in seq_along(outputs)) {
    want <- call$outputs[[j]]$aval
    want_shape <- c(size, shape(want))
    got <- outputs[[j]]
    if (!identical(as.integer(shape(got)), as.integer(want_shape)) || dtype(got) != dtype(want)) {
      cli_abort(c(
        "Internal error: the vectorize rule of {.fn {name}} returned a wrong output {j}.",
        x = "Expected {.val {as.character(dtype(want))}} of shape {shape_repr(want_shape)}, got {.val {as.character(dtype(got))}} of shape {shape_repr(shape(got))}."
      ))
    }
  }
}

#' @title Vectorize Rule
#' @description
#' Construct the rule [`vectorize()`] uses for a primitive applied to values
#' that are mapped over. Inside the rule, such values carry the batch axis as
#' their first axis.
#'
#' Most primitives do not need code of their own: they work on an extra
#' leading axis already, and only the parameters that name axes or sizes
#' change. Declaring how each of them changes is enough, via `params`. The
#' rule then gives every operand the batch axis -- broadcasting the ones that
#' are not mapped over -- and applies the primitive with the changed
#' parameters, so all of its outputs are batched. A parameter `params` does not
#' name stays as it is.
#'
#' For a primitive that this does not fit, pass `fn` instead, with the
#' signature `function(inputs, batched, params, size)`: `inputs` are the
#' operands, `batched` says which of them carry the batch axis, and `size` is
#' its size. It returns the outputs, each with the batch axis first.
#'
#' A declared rule calls the primitive with its parameters as arguments, so it
#' only fits a primitive made with [`new_primitive()`] whose parameters are
#' named like the arguments of its function.
#' @param fn (`NULL` | `function`)\cr
#'   A rule of its own. If given, the other arguments must be left at their
#'   defaults.
#' @param params (named `list()`)\cr
#'   How each parameter that refers to the axes of the operands changes, one
#'   of the kinds in [`param_axes()`].
#' @param scalar (`integer()`)\cr
#'   Positions of the operands that may be scalars while the others are not,
#'   e.g. the bounds of [`prim_clamp()`]. Such an operand stays a scalar when it
#'   is not mapped over.
#' @param unbatched (`integer()`)\cr
#'   Positions of the operands that cannot be mapped over, e.g. the padding
#'   value of [`prim_pad()`]. They are passed on unchanged.
#' @return (`anvl_rule_vectorize`)\cr
#'   A rule to assign to `prim_<name>[["vectorize"]]`.
#' @seealso [`vectorize()`], [`param_axes()`]
#' @export
#' @examples
#' # the rule of prim_transpose()
#' rule_vectorize(params = list(perm = param_axis_map()))
rule_vectorize <- function(fn = NULL, params = list(), scalar = integer(), unbatched = integer()) {
  checkmate::assert_function(fn, null.ok = TRUE)
  checkmate::assert_list(params, types = "anvl_param_kind", names = "unique")
  checkmate::assert_integerish(scalar, lower = 1L)
  checkmate::assert_integerish(unbatched, lower = 1L)
  if (!is.null(fn) && (length(params) || length(scalar) || length(unbatched))) {
    cli_abort("Provide either {.arg fn} or {.arg params}, {.arg scalar} and {.arg unbatched}.")
  }
  structure(
    list(fn = fn, params = params, scalar = as.integer(scalar), unbatched = as.integer(unbatched)),
    class = "anvl_rule_vectorize"
  )
}

#' @title Parameter Kinds
#' @description
#' How a parameter of a primitive changes when its operands gain a leading
#' batch axis of size `size`, for [`rule_vectorize()`]:
#'
#' * `param_axes()`: axis indices, which move one axis further (`axes + 1L`).
#' * `param_axis_map()`: one axis per axis of an operand, such as a
#'   permutation; the batch axis maps to the batch axis (`c(1L, axes + 1L)`).
#' * `param_shape()`: a shape, which gains the batch axis (`c(size, shape)`).
#' * `param_per_axis()`: one entry per axis of an operand, with `value` for the
#'   batch axis.
#' @param value (`integer(1)` | `function`)\cr
#'   The entry for the batch axis, or a function of `size` that returns it.
#' @return (`anvl_param_kind`)\cr
#'   A function of the parameter's value and `size` that returns its new value.
#' @export
#' @examples
#' param_axes()(c(1L, 3L), size = 5L)
#' param_axis_map()(c(2L, 1L), size = 5L)
#' param_shape()(c(2L, 3L), size = 5L)
#' param_per_axis(0L)(c(1L, 2L), size = 5L)
param_axes <- function() {
  param_kind(function(x, size) x + 1L)
}

#' @rdname param_axes
#' @export
param_axis_map <- function() {
  param_kind(function(x, size) c(1L, x + 1L))
}

#' @rdname param_axes
#' @export
param_shape <- function() {
  param_kind(function(x, size) c(size, x))
}

#' @rdname param_axes
#' @export
param_per_axis <- function(value) {
  if (!is.function(value)) {
    checkmate::assert_int(value)
  }
  first <- if (is.function(value)) value else function(size) value
  param_kind(function(x, size) c(as.integer(first(size)), x))
}

param_kind <- function(fn) {
  structure(fn, class = "anvl_param_kind")
}

# `x` with a leading batch axis of size `size`: as it is if it already has one,
# broadcast otherwise.
batch_operand <- function(x, batched, size) {
  if (batched) {
    return(x)
  }
  prim_broadcast_in_axes(x, c(size, shape(x)), seq_len(naxes(x)) + 1L)
}

# The operands of a primitive whose operands share one shape, each given the
# batch axis -- except those at the positions `unbatched` names, and a scalar at
# a position `scalar` names when the others are not scalars: such an operand is
# broadcast to the others' shape if it is batched, and left a scalar otherwise.
batch_operands <- function(inputs, batched, size, scalar = integer(), unbatched = integer()) {
  inner_shape <- function(i) {
    s <- shape(inputs[[i]])
    if (batched[[i]]) s[-1L] else s
  }
  main <- setdiff(seq_along(inputs), c(scalar, unbatched))
  full <- if (length(main)) inner_shape(main[[1L]])
  lapply(seq_along(inputs), function(i) {
    x <- inputs[[i]]
    if (i %in% unbatched) {
      return(x)
    }
    if (i %in% scalar && !identical(inner_shape(i), full)) {
      if (!batched[[i]]) {
        return(x)
      }
      return(prim_broadcast_in_axes(x, c(size, full), 1L))
    }
    batch_operand(x, batched[[i]], size)
  })
}

# `x` with its axis `from` moved to position `to`.
move_axis <- function(x, from, to) {
  if (from == to) {
    return(x)
  }
  perm <- seq_len(naxes(x))[-from]
  perm <- append(perm, from, after = to - 1L)
  prim_transpose(x, perm)
}
