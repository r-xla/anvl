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
#'   Has the same formals as `f` and can be called directly or inside [`jit()`]. It returns
#'   what `f` returns, with the batch axis added to every output where
#'   `out_axes` puts it; an output whose entry in `out_axes` is `NULL` is
#'   returned as it is.
#' @seealso [`rule_vectorize()`], [`gradient()`], `r roxy_article("vectorize")`
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' # f is written for a single vector
#' f <- function(x, y) sum(x * y)
#' x <- nv_array(matrix(1:6, nrow = 2L), dtype = "f32")
#'
#' # one dot product per column of x, against the same y
#' vectorize(f, in_axes = list(x = -1L))(x, nv_array(c(1, 2), dtype = "f32"))
#'
#' # map over the rows instead: each row divided by its sum
#' jit(vectorize(function(x) x / sum(x), in_axes = 1L, out_axes = 1L))(x)
#'
#' # every column gets noise of its own, the noise a loop over the columns
#' # would draw
#' noisy <- function(x) x + nv_runif(shape = 2L)
#' nv_set_seed(1L)
#' jit(vectorize(noisy))(x)
#' # or the same noise for every column
#' jit(vectorize(noisy, randomness = "same"))(x)
vectorize <- function(f, in_axes = -1L, out_axes = -1L, randomness = "different") {
  assert_function(f)
  in_axes <- check_in_axes(f, in_axes)
  check_out_axes(out_axes)
  assert_choice(randomness, c("different", "same"))
  transformation_fn(
    f,
    function(call_args) {
      # errors name the call of the vectorized function
      call <- rlang::caller_env()
      vectorize_call(f, call_args, in_axes, out_axes, randomness, call = call)
    },
    eager = TRUE
  )
}

# `in_axes` of `vectorize()` as an axis or a named list of axes, without the
# entries that are `NULL`. Errors name `call`.
check_in_axes <- function(f, in_axes, call = rlang::caller_env()) {
  if (!is.list(in_axes)) {
    if (is.null(names(in_axes))) {
      return(check_batch_axis(in_axes, "in_axes", call = call))
    }
    in_axes <- as.list(in_axes)
  }
  if (!length(in_axes)) {
    cli_abort("{.arg in_axes} must name at least one argument to map over.", call = call)
  }
  nms <- names(in_axes)
  if (!checkmate::test_names(nms, type = "named")) {
    cli_abort(
      "{.arg in_axes} must be a single axis or a list of them named by the arguments of {.arg f}, not by their positions.",
      call = call
    )
  }
  if (anyDuplicated(nms)) {
    cli_abort(
      c(
        "{.arg in_axes} must name every argument at most once.",
        x = "It names {.arg {unique(nms[duplicated(nms)])}} more than once."
      ),
      call = call
    )
  }
  resolve_transformation_args(f, nms, "in_axes", call = call)
  in_axes <- Map(
    function(axis, nm) check_batch_axis(axis, sprintf("in_axes$%s", nm), null_ok = TRUE, call = call),
    in_axes,
    nms
  )
  in_axes <- Filter(Negate(is.null), in_axes)
  if (!length(in_axes)) {
    cli_abort("{.arg in_axes} must name at least one argument to map over.", call = call)
  }
  in_axes
}

# Errors name `call`.
check_out_axes <- function(out_axes, call = rlang::caller_env()) {
  if (!is.list(out_axes) && (length(out_axes) > 1L || !is.null(names(out_axes)))) {
    cli_abort("{.arg out_axes} must be a single axis or a list of them, not a vector.", call = call)
  }
  if (is.list(out_axes)) {
    lapply(out_axes, check_out_axes, call = call)
  } else {
    check_batch_axis(out_axes, "out_axes", null_ok = TRUE, call = call)
  }
  invisible(out_axes)
}

# A batch axis of `in_axes` or `out_axes`, the argument `arg`, as an integer: a
# whole number other than 0, negative ones counting from the end. Errors name
# `call`.
check_batch_axis <- function(axis, arg, null_ok = FALSE, call = rlang::caller_env()) {
  problem <- checkmate::check_int(axis, null.ok = null_ok)
  if (!isTRUE(problem)) {
    cli_abort("{.arg {arg}} is invalid: {problem}", call = call)
  }
  if (is.null(axis)) {
    return(NULL)
  }
  if (axis == 0L) {
    cli_abort("{.arg {arg}} must not be 0: axes count from 1, or from -1 at the end.", call = call)
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
# leaf, which `f` receives as it is. Errors name `call`.
vectorize_call <- function(f, call_args, in_axes, out_axes, randomness, call) {
  flat <- flatten_args(call_args)
  mapped <- resolve_mapped_leaves(flat, in_axes, call)
  is_mapped <- !vapply(mapped$axes, is.null, logical(1L))

  # `f` takes the side effects it uses as fresh inputs, which are bound to the
  # values the slices start from once its uses are counted.
  enclosing <- current_descriptor()
  desc <- local_descriptor(side_effects_mode = "vectorize")
  graph <- trace_fn(f, args_flat = mapped$slice_args, in_tree = flat$in_tree, desc = desc)
  is_input <- !graph$is_static_flat
  operands <- Map(
    function(box, axis) if (is.null(axis)) box else move_axis(box, axis, naxes(box)),
    graph_operands(graph, flat$args_flat),
    mapped$axes[is_input]
  )
  side_effects <- desc_bind_vectorized(enclosing, graph, mapped$size, randomness)
  res <- graph_vectorize(
    graph,
    c(operands, side_effects$operands),
    batched = c(is_mapped[is_input], side_effects$batched),
    is_state = c(rep(FALSE, length(operands)), rep(TRUE, length(side_effects$operands))),
    size = mapped$size,
    call = call
  )

  outs <- Map(
    function(out, batched, axis, j) {
      path <- pjrt::tree_path(graph$out_tree, j)
      what <- if (nzchar(path)) sprintf("Output `%s`", path) else sprintf("Output %d", j)
      place_batch_axis(out, batched, axis, mapped$size, what, call)
    },
    res$outputs,
    res$batched,
    expand_out_axes(out_axes, graph$out_tree, length(res$outputs), call),
    seq_along(res$outputs)
  )
  unflatten(graph$out_tree, outs)
}

# The leaves of the arguments `flat` (see `flatten_args()`) that `in_axes` maps
# over: the position of each one's batch axis, `NULL` for a leaf that is not
# mapped over, as `axes`; the size they agree on along it, as `size`; and the
# leaves to trace `f` with, those mapped over as the abstract array of one
# slice, as `slice_args`. Errors name `call`.
resolve_mapped_leaves <- function(flat, in_axes, call) {
  args_flat <- flat$args_flat
  # the argument each leaf belongs to
  sizes <- pjrt::tree_child_sizes(flat$in_tree)
  arg_names <- pjrt::tree_child_names(flat$in_tree) %||% rep("", length(sizes))
  missing_args <- if (is.list(in_axes)) setdiff(names(in_axes), arg_names)
  if (length(missing_args)) {
    cli_abort(
      "Cannot map over {.arg {missing_args}}: {cli::qty(length(missing_args))}{?it was/they were} not passed.",
      call = call
    )
  }
  # An argument passed through `...` has no name, so messages give its position.
  arg_names[!nzchar(arg_names)] <- paste("argument", which(!nzchar(arg_names)))
  leaf_args <- rep(arg_names, times = sizes)
  axes <- if (is.list(in_axes)) {
    lapply(leaf_args, function(nm) if (nm %in% names(in_axes)) in_axes[[nm]])
  } else {
    lapply(args_flat, function(x) if (is_graph_box(x) || is_anvl_array(x)) in_axes)
  }

  size <- NULL
  slice_args <- args_flat
  for (i in which(!vapply(axes, is.null, logical(1L)))) {
    x <- args_flat[[i]]
    if (!is_arrayish(x)) {
      cli_abort(
        c(
          "Can only map over arrays.",
          x = "{.arg {leaf_args[[i]]}} is {.cls {class(x)[1L]}}."
        ),
        call = call
      )
    }
    aval <- to_abstract(x)
    x_shape <- shape(aval)
    axis <- batch_axis_position(axes[[i]], length(x_shape))
    if (is.na(axis)) {
      cli_abort(
        c(
          "Every argument mapped over must have its batch axis.",
          x = "{.arg {leaf_args[[i]]}} has shape {shape_repr(x_shape)}, so it has no axis {.val {axes[[i]]}}.",
          i = "Name the arguments to map over with {.arg in_axes}; an argument of the enclosing {.fn jit} that is not static is an array here."
        ),
        call = call
      )
    }
    axes[[i]] <- axis
    if (is.null(size)) {
      size <- x_shape[[axis]]
    } else if (x_shape[[axis]] != size) {
      cli_abort(
        c(
          "Every argument mapped over must have the same size along its batch axis.",
          x = "{.arg {leaf_args[[i]]}} has size {.val {x_shape[[axis]]}} along axis {.val {axis}}, an earlier one {.val {size}}."
        ),
        call = call
      )
    }
    slice_args[[i]] <- if (is_rdata(aval)) {
      RData(shape = x_shape[-axis], r_type = aval$r_type)
    } else {
      AbstractArray(dtype = aval$dtype, shape = x_shape[-axis])
    }
  }
  if (is.null(size)) {
    cli_abort("{.fn vectorize} needs at least one argument to map over.", call = call)
  }
  list(axes = axes, size = size, slice_args = slice_args)
}

# The output `out` of the vectorized function -- with a trailing batch axis of
# size `size` if it is `batched`, as one of every slice otherwise -- with its
# batch axis where the entry `axis` of `out_axes` puts it, or as it is for an
# entry `NULL`. `what` names the output for the messages. Errors name `call`.
place_batch_axis <- function(out, batched, axis, size, what, call) {
  if (is.null(axis)) {
    if (batched) {
      cli_abort(
        c(
          "Can only return an output as it is if it does not depend on the arguments mapped over.",
          x = "{what} does, but {.arg out_axes} is {.code NULL} for it."
        ),
        call = call
      )
    }
    return(out)
  }
  slice_shape <- if (batched) shape(out)[-naxes(out)] else shape(out)
  n <- length(slice_shape) + 1L
  pos <- batch_axis_position(axis, n)
  if (is.na(pos)) {
    cli_abort(
      c(
        "Every output must have room for its batch axis.",
        x = "{what} has shape {shape_repr(slice_shape)}, so it can be stacked along axes {.val {1L}} to {.val {n}} (or {.val {-n}} to {.val {-1L}}) only, not {.val {axis}}.",
        i = "Stack it along an axis it has room for with {.arg out_axes}."
      ),
      call = call
    )
  }
  if (batched) move_axis(out, from = n, to = pos) else add_batch_axis(out, size, pos)
}

# `out_axes` (see `vectorize()`) as one entry per flat output of a function
# whose output has the structure `out_tree` and `n` leaves. Errors name `call`.
expand_out_axes <- function(out_axes, out_tree, n, call) {
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
      cli_abort(
        c(
          "{.arg out_axes} must be a single axis or a list with the structure of what {.arg f} returns.",
          x = "It does not match the output {where}."
        ),
        call = call
      )
    }
    keys <- names(node) %||% as.character(seq_along(node))
    Map(function(s, nd, k) walk(s, nd, c(path, k)), spec, node, keys)
  }
  walk(out_axes, unflatten(out_tree, as.list(seq_len(n))), character())
  axes
}

# Replays `graph` into the current descriptor as a function of `inputs` (boxes,
# one per graph input), where those that `batched` flags carry a trailing batch
# axis of size `size`, and those that `is_state` flags are the random states the
# slices start from. A call none of whose inputs is batched is replayed as it
# is; every other call goes through its primitive's vectorize rule, and its
# outputs are batched where the rule says so. Returns the boxes of the outputs
# as `outputs`, and which of them are batched as `batched`. Errors name `call`.
graph_vectorize <- function(graph, inputs, batched, is_state, size, call) {
  desc <- current_descriptor()
  trans <- bind_graph(graph, desc, inputs)$trans
  is_batched <- hashtab()
  # the values that are batched only because they derive from the random states
  from_state <- hashtab()
  for (i in seq_along(graph$inputs)) {
    is_batched[[graph$inputs[[i]]]] <- batched[[i]]
    from_state[[graph$inputs[[i]]]] <- is_state[[i]]
  }
  box_of <- function(g) register_gval(desc, if (is_graph_literal(g)) g else trans[[g]] %||% g)
  batched_of <- function(g) !is_graph_literal(g) && isTRUE(is_batched[[g]])

  for (stmt in graph$statements) {
    in_boxes <- lapply(stmt$inputs, box_of)
    in_batched <- vapply(stmt$inputs, batched_of, logical(1L))
    state_only <- FALSE
    if (!any(in_batched)) {
      # Fresh outputs, so that the replay shares no value with `graph`, which
      # stays a graph of its own.
      outputs <- lapply(stmt$outputs, \(g) GraphValue(g$aval))
      replay_statement(desc, stmt, lapply(in_boxes, \(b) b$gnode), outputs)
      out_batched <- rep(FALSE, length(outputs))
    } else {
      state_only <- all(vapply(stmt$inputs[in_batched], \(g) isTRUE(from_state[[g]]), logical(1L)))
      out_boxes <- apply_vectorize_rule(stmt$primitive, in_boxes, in_batched, stmt$params, size, state_only, call)
      check_vectorized_outputs(stmt, out_boxes, size, call)
      outputs <- lapply(out_boxes, \(b) b$gnode)
      out_batched <- rep(TRUE, length(outputs))
    }
    for (j in seq_along(stmt$outputs)) {
      trans[[stmt$outputs[[j]]]] <- outputs[[j]]
      is_batched[[stmt$outputs[[j]]]] <- out_batched[[j]]
      from_state[[stmt$outputs[[j]]]] <- state_only
    }
  }

  list(
    outputs = lapply(graph$outputs, box_of),
    batched = vapply(graph$outputs, batched_of, logical(1L))
  )
}

# The outputs of the primitive `primitive` applied to `inputs` with the
# parameters `params`, by its vectorize rule, each with a trailing batch axis of
# size `size`. The inputs that `batched` flags carry it; `state_only` says that
# they do only because they derive from the random states, for the message when
# there is no rule. Errors name `call`.
apply_vectorize_rule <- function(primitive, inputs, batched, params, size, state_only, call) {
  rule <- primitive[["vectorize"]]
  name <- paste0("prim_", primitive$name)
  if (is.null(rule)) {
    abort_no_vectorize_rule(primitive, state_only, call)
  }
  if (!is.null(rule$fn)) {
    out <- rule$fn(inputs, batched, params, size)
    return(if (is_graph_box(out)) list(out) else out)
  }
  prim_fn <- primitive$fn
  if (is.null(prim_fn) || !all(names(params) %in% formalArgs(prim_fn))) {
    cli_abort(
      c(
        "The vectorize rule of {.fn {name}} declares its parameters, but cannot call it with them.",
        i = "A declared rule calls the primitive {.fn new_primitive} made with its parameters as arguments of the same names; otherwise, give it a rule of its own with {.code rule_vectorize(fn = )}."
      ),
      call = call
    )
  }

  mapped <- intersect(rule$unbatched, which(batched))
  if (length(mapped)) {
    cli_abort("{.fn vectorize} cannot map over operand {mapped[[1L]]} of {.fn {name}}.", call = call)
  }
  operands <- batch_operands(inputs, batched, size, unbatched = rule$unbatched)
  declared <- names(rule$params)
  params[declared] <- Map(function(kind, value) kind(value, size), rule$params, params[declared])
  out <- do.call(prim_fn, c(operands, params))
  if (is_graph_box(out)) list(out) else out
}

# The error for a call of `primitive`, which has no vectorize rule, with a
# batched input (see `apply_vectorize_rule()`).
abort_no_vectorize_rule <- function(primitive, state_only, call) {
  if (!state_only) {
    cli_abort(
      c(
        "{.fn vectorize} does not support {.fn {paste0('prim_', primitive$name)}} yet.",
        i = "Only values that are not mapped over can go through it."
      ),
      call = call
    )
  }
  # The fork of the random state is no function of anvl's API: the body of an
  # `nv_while()` that draws forks it.
  where <- if (primitive$name == "fork_rng_state") {
    cli::format_inline("an {.fn nv_while} loop that draws")
  } else {
    cli::format_inline("{.fn {paste0('prim_', primitive$name)}}")
  }
  cli_abort(
    c(
      "{.fn vectorize} cannot give every slice random numbers of its own in {where} yet.",
      i = "Draw the same numbers for every slice with {.code randomness = \"same\"}."
    ),
    call = call
  )
}

# A vectorize rule must return what the statement `stmt` returned, each with
# the batch axis last. Errors name `call`.
check_vectorized_outputs <- function(stmt, outputs, size, call) {
  name <- paste0("prim_", stmt$primitive$name)
  if (length(outputs) != length(stmt$outputs)) {
    cli_abort(
      "The vectorize rule of {.fn {name}} returned {length(outputs)} output{?s}, not {length(stmt$outputs)}.",
      call = call
    )
  }
  for (j in seq_along(outputs)) {
    want <- stmt$outputs[[j]]$aval
    want_shape <- c(shape(want), size)
    got <- outputs[[j]]
    if (!identical(as.integer(shape(got)), as.integer(want_shape)) || dtype(got) != dtype(want)) {
      cli_abort(
        c(
          "The vectorize rule of {.fn {name}} returned a wrong output {j}.",
          x = "Expected {.val {as.character(dtype(want))}} of shape {shape_repr(want_shape)}, got {.val {as.character(dtype(got))}} of shape {shape_repr(shape(got))}."
        ),
        call = call
      )
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
#' are not mapped over -- and applies the primitive with the changed
#' parameters, so all of its outputs are batched. A scalar operand among
#' operands that are not scalars stays a scalar if it is not mapped over, and
#' is broadcast to their shape if it is; an operand that must stay as it is
#' belongs in `unbatched`. A parameter `params` does not name stays as it is.
#'
#' For a primitive that this does not fit, pass `fn` instead, with the
#' signature `function(inputs, batched, params, size)`: `inputs` are the
#' operands, `batched` says which of them carry the batch axis, `params` are
#' the parameters of the call, and `size` is the size of the batch axis. It
#' returns the list of outputs, or a single output as it is, each with the
#' batch axis last, also one that is the same for every slice.
#'
#' A declared rule calls the primitive with its parameters as arguments, so it
#' only fits a primitive made with [`new_primitive()`] whose parameters are
#' named like the arguments of its function.
#' @param fn (`NULL` | `function`)\cr
#'   A rule of its own. If given, the other arguments must be left at their
#'   defaults.
#' @param params (named `list()`)\cr
#'   How each parameter with an entry per axis of the operands, or a shape,
#'   changes, one of the kinds in [`vec_shape()`].
#' @param unbatched (`integer()`)\cr
#'   Positions of the operands that cannot be mapped over, e.g. the padding
#'   value of [`prim_pad()`]. They are passed on unchanged.
#' @return (`anvl_rule_vectorize`)\cr
#'   A rule to assign to `prim_<name>[["vectorize"]]`.
#' @seealso [`vectorize()`], [`vec_shape()`]
#' @export
#' @examples
#' # the rule of prim_transpose()
#' rule_vectorize(params = list(perm = vec_axis_map()))
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
#' * `vec_shape()`: a shape, which gains the batch axis (`c(shape, size)`).
#' * `vec_axis_map()`: for each axis of the output, the operand axis it takes,
#'   such as a permutation; the batch axis maps to itself
#'   (`c(axes, length(axes) + 1L)`).
#' * `vec_per_axis()`: one entry per axis of an operand, with `value` for the
#'   batch axis.
#' @param value (`integer(1)` | `function`)\cr
#'   The entry for the batch axis, or a function of `size` that returns it.
#' @return (`anvl_param_kind`)\cr
#'   A function of the parameter's value and `size` that returns its new value.
#' @export
#' @examples
#' vec_shape()(c(2L, 3L), size = 5L)
#' vec_axis_map()(c(2L, 1L), size = 5L)
#' vec_per_axis(0L)(c(1L, 2L), size = 5L)
vec_shape <- function() {
  param_kind(function(x, size) c(x, size))
}

#' @rdname vec_shape
#' @export
vec_axis_map <- function() {
  param_kind(function(x, size) c(x, length(x) + 1L))
}

#' @rdname vec_shape
#' @export
vec_per_axis <- function(value) {
  if (!is.function(value)) {
    checkmate::assert_int(value)
  }
  last <- if (is.function(value)) value else function(size) value
  param_kind(function(x, size) c(x, as.integer(last(size))))
}

param_kind <- function(fn) {
  structure(fn, class = "anvl_param_kind")
}

# `x` with a batch axis of size `size` broadcast in at position `axis`, by
# default after its last axis.
add_batch_axis <- function(x, size, axis = naxes(x) + 1L) {
  prim_broadcast_in_axes(x, append(shape(x), size, after = axis - 1L), seq_len(naxes(x) + 1L)[-axis])
}

# `x` with a trailing batch axis of size `size`: as it is if it is `batched`,
# broadcast otherwise.
batch_operand <- function(x, batched, size) {
  if (batched) x else add_batch_axis(x, size)
}

# The operands of a primitive whose operands share one shape, each given the
# batch axis -- except those at the positions `unbatched` names, and a scalar
# operand among ones that are not scalars, which stands for an array of their
# shape: such an operand is broadcast to that shape if it is batched, and left
# a scalar otherwise.
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
