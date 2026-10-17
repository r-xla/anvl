#' @title Vectorize a Function
#' @description
#' Return a new function that maps `f` over an axis of its arguments, like
#' [base::Vectorize()]. Unlike `Vectorize()`, it does not call `f` once per
#' instance: the returned function traces `f` once and turns every operation of
#' it into one that works on the whole batch, so `f` can be written for a
#' single instance and still run as one program over all of them.
#'
#' `in_axes` says which arguments are mapped over, and along which axis: `f`
#' sees one slice of such an argument at a time, with that axis dropped. Every
#' leaf of a nested argument is mapped over along the same axis. The other
#' arguments are passed to each application of `f` unchanged. `out_axes` says
#' along which axis each output of `f` is stacked.
#' @param f (`function`)\cr
#'   Function to vectorize.
#' @param in_axes (`integer(1)` | named `list()` | named `integer()`)\cr
#'   The arguments to map over and their batch axes. A single axis maps over
#'   every argument that is an array, along that axis. That includes every
#'   argument of the enclosing [`jit()`] that is not static, R data among them;
#'   other plain R values -- such as static arguments, or an R array written
#'   inside the traced function -- are passed on unchanged.
#'   A named list or vector gives the axis of each argument to map over, by
#'   its name; arguments it leaves out or gives `NULL` are passed on
#'   unchanged. A nested argument takes one axis for all of its leaves. All
#'   arguments mapped over must have the same size along their batch axes.
#' @param out_axes (`integer(1)` | `list()`)\cr
#'   The axes along which the outputs of `f` are stacked. A single axis applies
#'   to every output. A list follows the structure of what `f` returns, down to
#'   any depth, matching named elements by name: an entry applies to every
#'   output below it. An output with `n`
#'   axes per slice can be stacked along axes `1` to `n + 1`. `NULL` returns an
#'   output as it is, which works for one that does not depend on the
#'   arguments mapped over.
#' @return (`function`)\cr
#'   Has the same formals as `f` and must be called inside [`jit()`]. It returns
#'   what `f` returns, with the batch axis added to every output where
#'   `out_axes` puts it; an output whose entry in `out_axes` is `NULL` is
#'   returned as it is.
#' @section Supported Primitives:
#' A primitive applied to a value that is mapped over needs a `vectorize` rule
#' (see [`rule_vectorize()`]); calling one that has none raises an error.
#' Values that are not mapped over can go through any primitive. Without a rule
#' so far are the decompositions [`prim_qr()`], [`prim_lu()`], [`prim_svd()`]
#' and [`prim_eigh()`]. Some operands cannot be mapped over yet: the start
#' indices of [`prim_dynamic_slice()`] and [`prim_dynamic_update_slice()`], and
#' the padding value of [`prim_pad()`].
#'
#' A few primitives are applied to one slice after the other, in a
#' [`prim_scan()`] over the batch, instead of to the whole batch at once:
#' [`prim_rng_bit_generator()`] mapped over its state, [`prim_reduce()`] mapped
#' over its initial value, and a grouped [`prim_convolution()`] mapped over its
#' kernel. A [`prim_if()`] whose predicate is mapped over evaluates each
#' branch that some slice takes for all slices, and selects between their
#' results; a slice that does not take a branch runs it on the inputs of one
#' that does, so that its gradient stays free of `NaN`s from inputs the branch
#' is not meant for. Gradients then match those of the slices one at a time,
#' except where the slice whose inputs are borrowed has an infinite derivative
#' itself: an argument that is not mapped over then gets `NaN` instead of
#' `Inf`. A [`prim_while()`] whose condition is mapped over runs
#' until it is `FALSE` for every slice, applying the body to the slices that
#' are done too and discarding the result. Side effects such as
#' [`prim_print()`] then also happen for slices that do not take that path.
#' @seealso [`rule_vectorize()`], [`gradient()`], `r roxy_article("vectorize")`
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' # f is written for a single vector
#' f <- function(x, y) sum(x * y)
#' x <- nv_array(matrix(1:6, nrow = 3L), dtype = "f32")
#'
#' # one dot product per row of x, against the same y
#' jit(vectorize(f, in_axes = list(x = 1L)))(x, nv_array(c(1, 2), dtype = "f32"))
#'
#' # map over the columns instead: each column divided by its sum
#' jit(vectorize(function(x) x / sum(x), in_axes = 2L, out_axes = 2L))(x)
vectorize <- function(f, in_axes = 1L, out_axes = 1L) {
  assert_function(f)
  in_axes <- check_in_axes(f, in_axes)
  check_out_axes(out_axes)
  transformation_fn(f, function(call_args) {
    assert_in_trace("vectorize")
    vectorize_call(f, call_args, in_axes, out_axes)
  })
}

# `in_axes` of `vectorize()` as an axis or a named list of axes, without the
# entries that are `NULL`.
check_in_axes <- function(f, in_axes) {
  if (!is.list(in_axes)) {
    if (is.null(names(in_axes))) {
      return(as.integer(checkmate::assert_int(in_axes, lower = 1L)))
    }
    in_axes <- as.list(in_axes)
  }
  if (!length(in_axes)) {
    cli_abort("{.arg in_axes} must name at least one argument to map over.")
  }
  if (!checkmate::test_names(names(in_axes), type = "unique")) {
    cli_abort(
      "{.arg in_axes} must be a single axis or a list of them named by the arguments of {.arg f}, not by their positions."
    )
  }
  resolve_transformation_args(f, names(in_axes), "in_axes")
  for (nm in names(in_axes)) {
    checkmate::assert_int(in_axes[[nm]], lower = 1L, null.ok = TRUE, .var.name = sprintf("in_axes$%s", nm))
  }
  in_axes <- Filter(Negate(is.null), in_axes)
  if (!length(in_axes)) {
    cli_abort("{.arg in_axes} must name at least one argument to map over.")
  }
  lapply(in_axes, as.integer)
}

check_out_axes <- function(out_axes) {
  if (!is.list(out_axes) && (length(out_axes) > 1L || !is.null(names(out_axes)))) {
    cli_abort("{.arg out_axes} must be a single axis or a list of them, not a vector.")
  }
  if (is.list(out_axes)) {
    lapply(out_axes, check_out_axes)
  } else {
    checkmate::assert_int(out_axes, lower = 1L, null.ok = TRUE, .var.name = "out_axes")
  }
  invisible(out_axes)
}

# Traces `f` on one slice of the arguments `in_axes` maps over and replays the
# graph into the current descriptor, batched along a leading axis, which then
# moves to where `out_axes` puts it.
vectorize_call <- function(f, call_args, in_axes, out_axes) {
  args_flat <- flatten(call_args)
  in_tree <- build_tree(call_args)
  # The argument each flat leaf belongs to.
  sizes <- pjrt::tree_child_sizes(in_tree)
  arg_names <- pjrt::tree_child_names(in_tree) %||% rep("", length(sizes))
  # An argument passed through `...` has no name, so messages give its position.
  arg_names[!nzchar(arg_names)] <- paste("argument", which(!nzchar(arg_names)))
  leaf_args <- rep(arg_names, times = sizes)
  leaf_axes <- if (is.list(in_axes)) {
    missing_args <- setdiff(names(in_axes), names(call_args))
    if (length(missing_args)) {
      cli_abort(
        "Cannot map over {.arg {missing_args}}: {cli::qty(length(missing_args))}{?it was/they were} not passed."
      )
    }
    lapply(leaf_args, function(nm) if (nm %in% names(in_axes)) in_axes[[nm]])
  } else {
    lapply(args_flat, function(x) if (is_graph_box(x) || is_anvl_array(x)) in_axes)
  }
  is_mapped <- !vapply(leaf_axes, is.null, logical(1L))

  size <- NULL
  trace_args <- args_flat
  for (i in which(is_mapped)) {
    x <- args_flat[[i]]
    axis <- leaf_axes[[i]]
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
        "Every argument mapped over must have its batch axis.",
        x = "{.arg {leaf_args[[i]]}} has shape {shape_repr(x_shape)}, so it has no axis {.val {axis}}.",
        i = "Name the arguments to map over with {.arg in_axes}; an argument of the enclosing {.fn jit} that is not static is an array here."
      ))
    }
    if (is.null(size)) {
      size <- x_shape[[axis]]
    } else if (x_shape[[axis]] != size) {
      cli_abort(c(
        "Every argument mapped over must have the same size along its batch axis.",
        x = "{.arg {leaf_args[[i]]}} has size {.val {x_shape[[axis]]}} along axis {.val {axis}}, an earlier one {.val {size}}."
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
    function(box, axis) if (is.null(axis)) box else move_axis(box, axis, 1L),
    graph_operands(graph, args_flat),
    leaf_axes[is_input]
  )

  res <- graph_vectorize(graph, operands, is_mapped[is_input], size)
  outs <- Map(
    function(out, batched, axis, j) {
      path <- pjrt::tree_path(graph$out_tree, j)
      what <- if (nzchar(path)) sprintf("Output `%s`", path) else sprintf("Output %d", j)
      if (is.null(axis)) {
        if (batched) {
          cli_abort(c(
            "Can only return an output as it is if it does not depend on the arguments mapped over.",
            x = "{what} does, but {.arg out_axes} is {.code NULL} for it."
          ))
        }
        return(out)
      }
      out <- batch_operand(out, batched, size)
      if (naxes(out) < axis) {
        cli_abort(c(
          "Every output must have room for its batch axis.",
          x = "{what} has shape {shape_repr(shape(out)[-1L])}, so it can be stacked along axes {.val {1L}} to {.val {naxes(out)}} only, not {.val {axis}}.",
          i = "Stack it along an axis it has room for with {.arg out_axes}."
        ))
      }
      move_axis(out, from = 1L, to = axis)
    },
    res$outputs,
    res$batched,
    expand_out_axes(out_axes, graph$out_tree, length(res$outputs)),
    seq_along(res$outputs)
  )
  unflatten(graph$out_tree, outs)
}

# `out_axes` (see `vectorize()`) as one entry per flat output of a function
# whose output has the structure `out_tree` and `n` leaves.
expand_out_axes <- function(out_axes, out_tree, n) {
  if (!is.list(out_axes)) {
    return(rep(list(out_axes), n))
  }
  axes <- vector("list", n)
  walk <- function(spec, node, path) {
    if (!is.list(spec)) {
      for (j in unlist(flatten(node))) {
        axes[j] <<- list(spec)
      }
      return()
    }
    if (is.list(node) && all(nzchar(names(node) %||% "")) && setequal(names(spec), names(node))) {
      spec <- spec[names(node)]
    }
    same <- is.list(node) && length(spec) == length(node) && identical(names(spec), names(node))
    if (!same) {
      where <- if (length(path)) sprintf("at `%s`", paste(path, collapse = "$")) else "at the top"
      cli_abort(c(
        "{.arg out_axes} must be a single axis or a list with the structure of what {.arg f} returns.",
        x = "It does not match the output {where}."
      ))
    }
    keys <- names(node) %||% as.character(seq_along(node))
    Map(function(s, nd, k) walk(s, nd, c(path, k)), spec, node, keys)
  }
  walk(out_axes, unflatten(out_tree, as.list(seq_len(n))), character())
  axes
}

# Replays `graph` into the current descriptor as a function of `inputs` (boxes,
# one per graph input), where those that `batched` flags carry a leading batch
# axis of size `size`. A call none of whose inputs is batched is replayed as it
# is; every other call goes through its primitive's vectorize rule, and its
# outputs are batched where the rule says so (see `partly_batched()`). Returns
# the boxes of the outputs as `outputs`, and which of them are batched as
# `batched`.
graph_vectorize <- function(graph, inputs, batched, size) {
  desc <- current_descriptor()
  boxes <- hashtab()
  is_batched <- hashtab()
  for (i in seq_along(graph$inputs)) {
    # An input may be a box of an enclosing descriptor, which this one then
    # captures.
    boxes[[graph$inputs[[i]]]] <- maybe_box_arrayish(inputs[[i]], desc)
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
      # Fresh outputs: a graph can be replayed more than once, e.g. the
      # condition of a `prim_while()` into both of its new sub-graphs.
      outputs <- lapply(call$outputs, \(g) GraphValue(g$aval))
      out_boxes <- replay_statement(desc, call, lapply(in_boxes, \(b) b$gnode), outputs)
      out_batched <- rep(FALSE, length(outputs))
    } else {
      res <- apply_vectorize_rule(call$primitive, in_boxes, in_batched, call$params, size)
      check_vectorized_outputs(call, res$outputs, res$batched, size)
      out_boxes <- res$outputs
      out_batched <- res$batched
    }
    for (j in seq_along(call$outputs)) {
      boxes[[call$outputs[[j]]]] <- out_boxes[[j]]
      is_batched[[call$outputs[[j]]]] <- any(in_batched) && out_batched[[j]]
    }
  }

  list(
    outputs = lapply(graph$outputs, box_of),
    batched = vapply(graph$outputs, batched_of, logical(1L))
  )
}

# The outputs of `graph_vectorize()`, those `out_batched` flags with the batch
# axis -- broadcast if the graph did not give it to them -- and the others
# without, which they must then come out as.
graph_vectorize_as <- function(graph, inputs, batched, size, out_batched) {
  res <- graph_vectorize(graph, inputs, batched, size)
  Map(
    function(out, is_batched, want) {
      if (want) {
        return(batch_operand(out, is_batched, size))
      }
      if (is_batched) {
        cli_abort("Internal error: an output that must not be batched is.")
      }
      out
    },
    res$outputs,
    res$batched,
    out_batched
  )
}

# Which outputs of `graph` `graph_vectorize()` may batch, given which of its
# inputs `batched` flags, without replaying it: those that depend on a batched
# input. A rule may leave some of them unbatched still, which
# `graph_vectorize_as()` then broadcasts where needed.
graph_batched_outputs <- function(graph, batched) {
  is_batched <- hashtab()
  for (i in seq_along(graph$inputs)) {
    is_batched[[graph$inputs[[i]]]] <- batched[[i]]
  }
  batched_of <- function(g) !is_graph_literal(g) && isTRUE(is_batched[[g]])
  for (call in graph$statements) {
    b <- any(vapply(call$inputs, batched_of, logical(1L)))
    for (out in call$outputs) {
      is_batched[[out]] <- b
    }
  }
  vapply(graph$outputs, batched_of, logical(1L))
}

# Which of the values a loop carries are batched: those that enter batched, and
# those its body batches from the batched ones among the carry and `other` (the
# inputs of `body` after the carry), until that adds no more.
carry_batched <- function(body, carry, other) {
  n <- length(carry)
  repeat {
    out <- graph_batched_outputs(body, c(carry, other))[seq_len(n)]
    if (!any(out & !carry)) {
      return(carry)
    }
    carry <- carry | out
  }
}

apply_vectorize_rule <- function(primitive, inputs, batched, params, size) {
  rule <- primitive[["vectorize"]]
  if (is.null(rule)) {
    cli_abort(c(
      "{.fn vectorize} does not support {.fn {paste0('prim_', primitive$name)}} yet.",
      i = "Only values that are not mapped over can go through it."
    ))
  }
  name <- paste0("prim_", primitive$name)
  if (!is.null(rule$fn)) {
    out <- rule$fn(inputs, batched, params, size)
    out_batched <- attr(out, "batched")
    out <- if (is_graph_box(out)) list(out) else unclass(out)
    attr(out, "batched") <- NULL
    if (is.null(out_batched)) {
      out_batched <- rep(TRUE, length(out))
    } else if (!checkmate::test_logical(out_batched, any.missing = FALSE, len = length(out))) {
      cli_abort(c(
        "The vectorize rule of {.fn {name}} returned an invalid attribute {.field batched}.",
        i = "It must be a logical vector with one entry per output, without missing values."
      ))
    }
    return(list(outputs = out, batched = out_batched))
  }
  prim_fn <- primitive$fn
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
  out <- if (is_graph_box(out)) list(out) else out
  list(outputs = out, batched = rep(TRUE, length(out)))
}

# The outputs of a vectorize rule, of which only those `batched` flags carry
# the batch axis.
partly_batched <- function(outputs, batched) {
  structure(outputs, batched = batched)
}

# A vectorize rule must return what the call returned, each with the batch axis
# in front unless `batched` says otherwise.
check_vectorized_outputs <- function(call, outputs, batched, size) {
  name <- paste0("prim_", call$primitive$name)
  if (length(outputs) != length(call$outputs)) {
    cli_abort(
      "The vectorize rule of {.fn {name}} returned {length(outputs)} output{?s}, not {length(call$outputs)}."
    )
  }
  for (j in seq_along(outputs)) {
    want <- call$outputs[[j]]$aval
    want_shape <- if (batched[[j]]) c(size, shape(want)) else shape(want)
    got <- outputs[[j]]
    if (!identical(as.integer(shape(got)), as.integer(want_shape)) || dtype(got) != dtype(want)) {
      cli_abort(c(
        "The vectorize rule of {.fn {name}} returned a wrong output {j}.",
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
#' operands, `batched` says which of them carry the batch axis, `params` are
#' the parameters of the call, and `size` is the size of the batch axis. It
#' returns the list of outputs, or a single output as it is, each with the
#' batch axis first. An output
#' that does not depend on the values mapped over may be returned without it,
#' if the list of outputs says so in its attribute `batched`, a logical vector
#' with one entry per output (set it last: [c()] and [lapply()] drop it). Only
#' an output that is the same for every slice may be marked `FALSE`: one that
#' differs between slices would silently give every slice the same value.
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

# `x`, whose batch axis is first, with that axis folded into its axis `axis`
# (counted without the batch axis). Along the merged axis, the elements of
# `axis` run fastest, so each slice of the batch is one contiguous block.
merge_batch_axis <- function(x, axis) {
  x <- move_axis(x, 1L, axis + 1L)
  s <- shape(x)
  prim_reshape(x, c(s[seq_len(axis - 1L)], s[[axis]] * s[[axis + 1L]], s[-seq_len(axis + 1L)]))
}

# Undoes `merge_batch_axis()`: splits a batch axis of size `size` off the axis
# `axis` of `x` and moves it in front.
split_batch_axis <- function(x, axis, size) {
  s <- shape(x)
  x <- prim_reshape(x, c(s[seq_len(axis - 1L)], s[[axis]] %/% size, size, s[-seq_len(axis)]))
  move_axis(x, axis + 1L, 1L)
}

# The rule of a primitive that cannot work on the whole batch at once: applies
# `apply`, a function of the operands of one slice returning its outputs, to
# one slice after the other, in a `prim_scan()` over the batch axis.
vectorize_by_loop <- function(apply, inputs, batched, size) {
  body <- function(carry, x) {
    operands <- inputs
    operands[batched] <- x
    list(carry = carry, out = apply(operands))
  }
  prim_scan(list(0L), inputs[batched], body, steps = size)$out
}

# `x` without a gradient: its reverse rule gives `x` a zero cotangent, whatever
# reaches it. A float goes through a bitcast to the unsigned integer of its
# width and back, which XLA removes; any other value has no gradient anyway.
no_gradient <- function(x) {
  if (!is_dtype_float(dtype(x))) {
    return(x)
  }
  bits <- prim_bitcast_convert(x, paste0("ui", dtype_width(dtype(x))))
  prim_bitcast_convert(bits, dtype(x))
}
