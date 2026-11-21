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
#' along which axis each output of `f` is stacked. By default, both are the last
#' axis: the slices of a matrix are its columns, and the results of `f` are
#' stacked as columns.
#'
#' `randomness` says what the random draws of `f` (see [nv_set_seed()]) give
#' the slices, as PyTorch's `torch.func.vmap()` does. By default, every slice
#' draws numbers of its own: the same ones a loop over the slices, applying `f`
#' to one after the other, would draw, so `vectorize()` changes how fast a
#' simulation runs, not its results. With `"same"`, every slice draws the same
#' numbers, the ones a single application of `f` draws, and the random state
#' advances as after that application.
#' @param f (`function`)\cr
#'   Function to vectorize.
#' @param in_axes (`integer(1)` | named `list()` | named `integer()`)\cr
#'   The arguments to map over and their batch axes. Negative axes count from
#'   the end, so the default, `-1L`, is the last axis of every argument mapped
#'   over. A single axis maps over
#'   every argument that is an array, along that axis. That includes every
#'   argument of the enclosing [`jit()`] that is not static, R data among them;
#'   other values -- such as static arguments, a random state, or an R array
#'   written inside the traced function -- are passed on unchanged.
#'   A named list or vector gives the axis of each argument to map over, by
#'   its name; arguments it leaves out or gives `NULL` are passed on
#'   unchanged. A nested argument takes one axis for all of its leaves. All
#'   arguments mapped over must have the same size along their batch axes.
#' @param out_axes (`integer(1)` | `list()`)\cr
#'   The axes along which the outputs of `f` are stacked, by default the last
#'   one. A single axis applies
#'   to every output. A list follows the structure of what `f` returns, down to
#'   any depth, matching named elements by name: an entry applies to every
#'   output below it. An output with `n`
#'   axes per slice can be stacked along axes `1` to `n + 1`, or `-(n + 1)` to
#'   `-1` counting from the end. `NULL` returns an
#'   output as it is, which works for one that does not depend on the
#'   arguments mapped over.
#' @param randomness (`character(1)`)\cr
#'   Whether the slices draw `"different"` (the default) or the `"same"`
#'   random numbers.
#' @return (`function`)\cr
#'   Has the same formals as `f` and must be called inside [`jit()`]. It returns
#'   what `f` returns, with the batch axis added to every output where
#'   `out_axes` puts it; an output whose entry in `out_axes` is `NULL` is
#'   returned as it is.
#' @section Supported Primitives:
#' A primitive applied to a value that is mapped over needs a `vectorize` rule
#' (see [`rule_vectorize()`]); calling one that has none raises an error.
#' Values that are not mapped over can go through any primitive. Without a rule
#' so far are [`prim_gather()`], [`prim_scatter()`], [`prim_chol()`],
#' [`prim_triangular_solve()`], control flow ([`prim_if()`], [`prim_while()`],
#' [`prim_scan()`]), [`prim_reduce()`],
#' [`prim_convolution()`] and the decompositions [`prim_qr()`], [`prim_lu()`],
#' [`prim_svd()`] and [`prim_eigh()`].
#' Some operands cannot be mapped over yet: the start indices of
#' [`prim_dynamic_slice()`] and [`prim_dynamic_update_slice()`], and the
#' padding value of [`prim_pad()`].
#' @seealso [`rule_vectorize()`], [`gradient()`], `r roxy_article("vectorize")`
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' # f is written for a single vector
#' f <- function(x, y) sum(x * y)
#' x <- nv_array(matrix(1:6, nrow = 2L), dtype = "f32")
#'
#' # one dot product per column of x, against the same y
#' jit(vectorize(f, in_axes = list(x = -1L)))(x, nv_array(c(1, 2), dtype = "f32"))
#'
#' # map over the rows instead: each row divided by its sum
#' jit(vectorize(function(x) x / sum(x), in_axes = 1L, out_axes = 1L))(x)
#'
#' # every column gets noise of its own, the noise a loop over the columns
#' # would draw
#' noisy <- function(x) x + nv_rnorm(shape = 2L)
#' nv_set_seed(1L)
#' jit(vectorize(noisy))(x)
#' # or the same noise for every column
#' jit(vectorize(noisy, randomness = "same"))(x)
vectorize <- function(f, in_axes = -1L, out_axes = -1L, randomness = "different") {
  assert_function(f)
  in_axes <- check_in_axes(f, in_axes)
  check_out_axes(out_axes)
  assert_choice(randomness, c("different", "same"))
  transformation_fn(f, function(call_args) {
    assert_in_trace("vectorize")
    vectorize_call(f, call_args, in_axes, out_axes, randomness)
  })
}

# `in_axes` of `vectorize()` as an axis or a named list of axes, without the
# entries that are `NULL`.
check_in_axes <- function(f, in_axes) {
  if (!is.list(in_axes)) {
    if (is.null(names(in_axes))) {
      return(check_batch_axis(in_axes, "in_axes"))
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
  in_axes <- Map(
    function(axis, nm) check_batch_axis(axis, sprintf("in_axes$%s", nm), null_ok = TRUE),
    in_axes,
    names(in_axes)
  )
  in_axes <- Filter(Negate(is.null), in_axes)
  if (!length(in_axes)) {
    cli_abort("{.arg in_axes} must name at least one argument to map over.")
  }
  in_axes
}

check_out_axes <- function(out_axes) {
  if (!is.list(out_axes) && (length(out_axes) > 1L || !is.null(names(out_axes)))) {
    cli_abort("{.arg out_axes} must be a single axis or a list of them, not a vector.")
  }
  if (is.list(out_axes)) {
    lapply(out_axes, check_out_axes)
  } else {
    check_batch_axis(out_axes, "out_axes", null_ok = TRUE)
  }
  invisible(out_axes)
}

# A batch axis of `in_axes` or `out_axes`, the argument `arg`, as an integer: a
# whole number other than 0, negative ones counting from the end.
check_batch_axis <- function(axis, arg, null_ok = FALSE) {
  checkmate::assert_int(axis, null.ok = null_ok, .var.name = arg)
  if (is.null(axis)) {
    return(NULL)
  }
  if (axis == 0L) {
    cli_abort("{.arg {arg}} must not be 0: axes count from 1, or from -1 at the end.")
  }
  as.integer(axis)
}

# The position of the axis `axis` (see `check_batch_axis()`) among `n` axes, or
# `NA` if there is no such axis.
batch_axis_position <- function(axis, n) {
  pos <- if (axis < 0L) n + 1L + axis else axis
  if (pos < 1L || pos > n) NA_integer_ else pos
}

# Traces `f` on one slice of the arguments `in_axes` maps over and replays the
# graph into the current descriptor, batched along a trailing axis, which then
# moves to where `out_axes` puts it. A random state among the arguments is a
# leaf, which `f` receives as it is.
vectorize_call <- function(f, call_args, in_axes, out_axes, randomness) {
  args_flat <- flatten(call_args, is_leaf = is_rng_state)
  in_tree <- build_tree(call_args, is_leaf = is_rng_state)
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
    pos <- batch_axis_position(axis, length(x_shape))
    if (is.na(pos)) {
      cli_abort(c(
        "Every argument mapped over must have its batch axis.",
        x = "{.arg {leaf_args[[i]]}} has shape {shape_repr(x_shape)}, so it has no axis {.val {axis}}.",
        i = "Name the arguments to map over with {.arg in_axes}; an argument of the enclosing {.fn jit} that is not static is an array here."
      ))
    }
    axis <- pos
    leaf_axes[[i]] <- pos
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

  # `f` takes the side effects it uses as fresh inputs, which are bound to the
  # values the slices start from once its uses are counted.
  enclosing <- current_descriptor()
  desc <- local_descriptor(side_effects_mode = "vectorize")
  graph <- trace_fn(f, args_flat = trace_args, in_tree = in_tree, desc = desc)
  is_input <- !graph$is_static_flat
  operands <- Map(
    function(box, axis) if (is.null(axis)) box else move_axis(box, axis, naxes(box)),
    graph_operands(graph, args_flat),
    leaf_axes[is_input]
  )
  side_effects <- desc_bind_vectorized(enclosing, graph, size, randomness)

  res <- graph_vectorize(
    graph,
    c(operands, side_effects$operands),
    c(is_mapped[is_input], side_effects$batched),
    size
  )
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
      n <- naxes(out)
      pos <- batch_axis_position(axis, n)
      if (is.na(pos)) {
        cli_abort(c(
          "Every output must have room for its batch axis.",
          x = "{what} has shape {shape_repr(shape(out)[-n])}, so it can be stacked along axes {.val {1L}} to {.val {n}} (or {.val {-n}} to {.val {-1L}}) only, not {.val {axis}}.",
          i = "Stack it along an axis it has room for with {.arg out_axes}."
        ))
      }
      move_axis(out, from = n, to = pos)
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
# one per graph input), where those that `batched` flags carry a trailing batch
# axis of size `size`. A call none of whose inputs is batched is replayed as it
# is; every other call goes through its primitive's vectorize rule, and its
# outputs are batched where the rule says so. Returns
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
  operands <- batch_operands(inputs, batched, size, unbatched = rule$unbatched)
  for (nm in names(rule$params)) {
    params[[nm]] <- rule$params[[nm]](params[[nm]], size)
  }
  out <- do.call(prim_fn, c(operands, params))
  out <- if (is_graph_box(out)) list(out) else out
  list(outputs = out, batched = rep(TRUE, length(out)))
}

# A vectorize rule must return what the call returned, each with the batch axis
# last unless `batched` says otherwise.
check_vectorized_outputs <- function(call, outputs, batched, size) {
  name <- paste0("prim_", call$primitive$name)
  if (length(outputs) != length(call$outputs)) {
    cli_abort(
      "The vectorize rule of {.fn {name}} returned {length(outputs)} output{?s}, not {length(call$outputs)}."
    )
  }
  for (j in seq_along(outputs)) {
    want <- call$outputs[[j]]$aval
    want_shape <- if (batched[[j]]) c(shape(want), size) else shape(want)
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
#' their last axis.
#'
#' Most primitives do not need code of their own: they work on an extra
#' trailing axis already, and a parameter that names an axis still names the
#' same one. Only the parameters with an entry per axis, or a shape, change.
#' Declaring how each of them changes is enough, via `params`. The
#' rule then gives every operand the batch axis -- broadcasting the ones that
#' are not mapped over, except a scalar operand of an element-wise primitive,
#' which stays a scalar -- and applies the primitive with the changed
#' parameters, so all of its outputs are batched. A parameter `params` does not
#' name stays as it is.
#'
#' For a primitive that this does not fit, pass `fn` instead, with the
#' signature `function(inputs, batched, params, size)`: `inputs` are the
#' operands, `batched` says which of them carry the batch axis, `params` are
#' the parameters of the call, and `size` is the size of the batch axis. It
#' returns the list of outputs, or a single output as it is, each with the
#' batch axis last. An output
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
#'   How each parameter with an entry per axis of the operands, or a shape,
#'   changes, one of the kinds in [`param_shape()`].
#' @param unbatched (`integer()`)\cr
#'   Positions of the operands that cannot be mapped over, e.g. the padding
#'   value of [`prim_pad()`]. They are passed on unchanged.
#' @return (`anvl_rule_vectorize`)\cr
#'   A rule to assign to `prim_<name>[["vectorize"]]`.
#' @seealso [`vectorize()`], [`param_shape()`]
#' @export
#' @examples
#' # the rule of prim_transpose()
#' rule_vectorize(params = list(perm = param_axis_map()))
rule_vectorize <- function(fn = NULL, params = list(), unbatched = integer()) {
  checkmate::assert_function(fn, null.ok = TRUE)
  checkmate::assert_list(params, types = "anvl_param_kind", names = "unique")
  checkmate::assert_integerish(unbatched, lower = 1L)
  if (!is.null(fn) && (length(params) || length(unbatched))) {
    cli_abort("Provide either {.arg fn} or {.arg params} and {.arg unbatched}.")
  }
  structure(
    list(fn = fn, params = params, unbatched = as.integer(unbatched)),
    class = "anvl_rule_vectorize"
  )
}

#' @title Parameter Kinds
#' @description
#' How a parameter of a primitive changes when its operands gain a trailing
#' batch axis of size `size`, for [`rule_vectorize()`]. A parameter that names
#' axes needs none of these: the batch axis comes after all of them, so they
#' keep their positions.
#'
#' * `param_shape()`: a shape, which gains the batch axis (`c(shape, size)`).
#' * `param_axis_map()`: one axis per axis of an operand, of an output with as
#'   many axes, such as a permutation; the batch axis maps to the batch axis
#'   (`c(axes, length(axes) + 1L)`).
#' * `param_per_axis()`: one entry per axis of an operand, with `value` for the
#'   batch axis.
#' @param value (`integer(1)` | `function`)\cr
#'   The entry for the batch axis, or a function of `size` that returns it.
#' @return (`anvl_param_kind`)\cr
#'   A function of the parameter's value and `size` that returns its new value.
#' @export
#' @examples
#' param_shape()(c(2L, 3L), size = 5L)
#' param_axis_map()(c(2L, 1L), size = 5L)
#' param_per_axis(0L)(c(1L, 2L), size = 5L)
param_shape <- function() {
  param_kind(function(x, size) c(x, size))
}

#' @rdname param_shape
#' @export
param_axis_map <- function() {
  param_kind(function(x, size) c(x, length(x) + 1L))
}

#' @rdname param_shape
#' @export
param_per_axis <- function(value) {
  if (!is.function(value)) {
    checkmate::assert_int(value)
  }
  last <- if (is.function(value)) value else function(size) value
  param_kind(function(x, size) c(x, as.integer(last(size))))
}

param_kind <- function(fn) {
  structure(fn, class = "anvl_param_kind")
}

# `x` with a trailing batch axis of size `size`: as it is if it already has one,
# broadcast otherwise.
batch_operand <- function(x, batched, size) {
  if (batched) {
    return(x)
  }
  prim_broadcast_in_axes(x, c(shape(x), size), seq_len(naxes(x)))
}

# The operands of a primitive whose operands share one shape, each given the
# batch axis -- except those at the positions `unbatched` names, and a scalar
# operand of an element-wise primitive, which stands for an array of the
# others' shape: such an operand is broadcast to that shape if it is batched,
# and left a scalar otherwise.
batch_operands <- function(inputs, batched, size, unbatched = integer()) {
  inner_shapes <- lapply(seq_along(inputs), function(i) {
    s <- shape(inputs[[i]])
    if (batched[[i]]) s[-length(s)] else s
  })
  arrays <- Filter(length, inner_shapes[setdiff(seq_along(inputs), unbatched)])
  full <- if (length(arrays)) arrays[[1L]] else integer()
  lapply(seq_along(inputs), function(i) {
    x <- inputs[[i]]
    if (i %in% unbatched) {
      return(x)
    }
    if (!length(inner_shapes[[i]]) && length(full)) {
      if (!batched[[i]]) {
        return(x)
      }
      return(prim_broadcast_in_axes(x, c(full, size), length(full) + 1L))
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
