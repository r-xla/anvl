#' @title Vectorize a Function
#' @description
#' Return a new function that maps `f` over an axis of its arguments.
#' Instead of looping over the slices, the function is rewritten to operate
#' on the whole batch at once.
#'
#' This is analogous to `vmap` found in other array computing
#' frameworks and is also in the spirit of [base::Vectorize()].
#'
#' See `r roxy_article("vectorize")` for more details.
#'
#' @param f (`function`)\cr
#'   Function to vectorize.
#' @param in_axes (`integer(1)` | named `list()`)\cr
#'   Specify the batch axis.
#'   If `in_axes` is a single integer, it maps over every argument that is an
#'   array; other values, such as R numbers, are passed to every application
#'   as they are.
#'   A scalar is never mapped over, as it has no axis: every application gets it
#'   as it is, whether it is an R value or an array, e.g. one that an enclosing
#'   [`jit()`] made of an R value.
#'   If it is a named list, it must be compatible with the argument tree of the function.
#'   The values of the tree must either be integers, indicating the batch axis (negative
#'   values count from the end), or `NULL`, indicating no batching.
#'   Values are applied to all leaves of the node.
#'   Arguments or named elements the list leaves out are not batched either.
#' @param out_axes (`integer(1)` | `list()`)\cr
#'   Along which axis to stack the outputs of `f`. Specified like `in_axes`,
#'   but a tree follows the structure of what `f` returns. `NULL` returns an
#'   output as it is, which only works for one that does not depend on the
#'   arguments mapped over.
#' @param randomness (`character(1)`)\cr
#'   Whether the slices draw `"different"` (the default) or the `"same"`
#'   random numbers.
#' @return (`function`)\cr
#'   Has the same formals as `f` and must be called inside [`jit()`]. It returns
#'   what `f` returns, with the batch axis added to every output where
#'   `out_axes` puts it; an output whose entry in `out_axes` is `NULL` is
#'   returned as it is.
#' @seealso [`rule_vectorize()`], [`gradient()`], `r roxy_article("vectorize")`
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' # f is written for a single vector
#' f <- function(x, y) sum(x * y)
#' x <- nv_array(matrix(1:6, nrow = 2L), dtype = "f32")
#' y <- nv_array(c(1, 2), dtype = "f32")
#'
#' # one dot product per column of x, against the same y
#' jit(vectorize(f, in_axes = list(x = -1L)))(x, y)
#'
#' # map over the rows instead: each row divided by its sum
#' jit(vectorize(function(x) x / sum(x), in_axes = 1L, out_axes = 1L))(x)
#'
#' # a nested list in in_axes maps over a part of a nested argument: the
#' # columns of params$w, while params$b is the same for every column
#' affine <- function(params, y) sum(params$w * y) + params$b
#' params <- list(w = x, b = nv_scalar(10, dtype = "f32"))
#' jit(vectorize(affine, in_axes = list(params = list(w = -1L))))(params, y)
#'
#' # out_axes as a list: stack the dot products, and return the norm of y,
#' # which does not depend on the columns, as it is
#' dot_norm <- function(x, y) list(dot = sum(x * y), norm = sqrt(sum(y^2)))
#' jit(vectorize(
#'   dot_norm,
#'   in_axes = list(x = -1L),
#'   out_axes = list(dot = -1L, norm = NULL)
#' ))(x, y)
#'
#' # vectorize() composes with other transformations, e.g. one gradient of a
#' # loss per column of x
#' loss <- function(w, x) sum((x * w)^2)
#' jit(vectorize(gradient(loss, wrt = "w"), in_axes = list(x = -1L)))(y, x)
#'
#' # and with itself: every column of x against every column of x, which is
#' # t(x) %*% x
#' f_all <- vectorize(vectorize(f, in_axes = list(x = -1L)), in_axes = list(y = -1L))
#' jit(f_all)(x, x)
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
    # errors name the call of the vectorized function
    call <- rlang::caller_env()
    assert_in_trace("vectorize", call = call)
    vectorize_call(f, call_args, in_axes, out_axes, randomness, call = call)
  })
}

# `in_axes` of `vectorize()` as an axis or a list of axes named by arguments,
# each nested to any depth (see `check_in_axes_tree()`), without the entries
# that are `NULL`. Errors name `call`.
check_in_axes <- function(f, in_axes, call = rlang::caller_env()) {
  if (!is.list(in_axes)) {
    if (!is.null(names(in_axes))) {
      cli_abort(
        "{.arg in_axes} must be a single axis or a list of them named by the arguments of {.arg f}, not a vector.",
        call = call
      )
    }
    return(check_batch_axis(in_axes, "in_axes", call = call))
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
    function(axes, nm) check_in_axes_tree(axes, sprintf("in_axes$%s", nm), call = call),
    in_axes,
    nms
  )
  in_axes <- Filter(Negate(is.null), in_axes)
  if (!length(unlist(in_axes))) {
    cli_abort("{.arg in_axes} must name at least one argument to map over.", call = call)
  }
  in_axes
}

# The entry `axes` of `in_axes`, the argument `arg`, as a batch axis (see
# `check_batch_axis()`), `NULL`, or a list of them, nested to any depth, that
# names no element twice. Errors name `call`.
check_in_axes_tree <- function(axes, arg, call) {
  if (!is.list(axes)) {
    if (!is.null(names(axes))) {
      cli_abort("{.arg {arg}} must be a single axis or a list of them, not a vector.", call = call)
    }
    return(check_batch_axis(axes, arg, null_ok = TRUE, call = call))
  }
  nms <- names(axes) %||% character(length(axes))
  if (anyDuplicated(nms[nzchar(nms)])) {
    cli_abort("{.arg {arg}} must name every element at most once.", call = call)
  }
  paths <- ifelse(nzchar(nms), sprintf("%s$%s", arg, nms), sprintf("%s[[%d]]", arg, seq_along(axes)))
  Map(function(x, path) check_in_axes_tree(x, path, call = call), axes, paths)
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

# Binds the side effects of `graph`, the function `vectorize()` traced on one
# slice, which took them as fresh inputs, in the trace `desc`, which applies it
# to `size` slices: their inputs become inputs of `graph`, after its own.
# Returns the operands for them and which of them are batched. With
# `randomness = "different"`, the slices use a side effect one after the other,
# as a loop over them would, and the views of `desc` continue after all of
# them; with `"same"`, every slice starts from the value `desc` is at, and the
# views continue after one.
desc_bind_vectorized <- function(desc, graph, size, randomness) {
  views <- graph$side_effects
  graph$inputs <- c(graph$inputs, views_values(views, "inputs"))
  graph$side_effects <- list()
  bound <- lapply(views, function(body) {
    x <- desc_view(desc, body$effect)
    start <- view_current(x)[names(body$inputs)]
    if (randomness == "same") {
      x$n <- x$n + body$n
      return(list(operands = start, batched = rep(FALSE, length(start))))
    }
    x$n <- x$n + size * body$n
    stacked <- side_effect_stack(body$effect, start, body$n, size)[names(body$inputs)]
    list(operands = stacked, batched = rep(TRUE, length(stacked)))
  })
  list(
    operands = unname(unlist(lapply(bound, `[[`, "operands"), recursive = FALSE)),
    batched = unlist(lapply(bound, `[[`, "batched")) %||% logical()
  )
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
  missing_args <- if (is.list(in_axes)) setdiff(names(in_axes), pjrt::tree_child_names(flat$in_tree))
  if (length(missing_args)) {
    cli_abort(
      "Cannot map over {.arg {missing_args}}: {cli::qty(length(missing_args))}{?it was/they were} not passed.",
      call = call
    )
  }
  # Messages name each leaf by its path. An argument passed through `...` has no
  # name, so they give its position.
  leaf_args <- vapply(seq_along(args_flat), function(i) pjrt::tree_path(flat$in_tree, i), character(1L))
  leaf_args <- sub("^\\[\\[([0-9]+)\\]\\]", "argument \\1", leaf_args)
  axes <- if (is.list(in_axes)) {
    expand_axes(in_axes, flat$in_tree, length(args_flat), partial = TRUE, arg = "in_axes", part = "argument", call)
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
    # A scalar has no axis to map over, so every slice gets it as it is,
    # whether it is an R value or an array.
    if (!length(x_shape)) {
      axes[i] <- list(NULL)
      next
    }
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
    cli_abort(
      c(
        "{.fn vectorize} needs at least one argument to map over.",
        i = "A scalar is not mapped over: every slice gets it as it is."
      ),
      call = call
    )
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
  expand_axes(out_axes, out_tree, n, partial = FALSE, arg = "out_axes", part = "output", call = call)
}

# The axis that `spec` -- the argument `arg` of `vectorize()`, `in_axes` or
# `out_axes` -- gives each of the `n` leaves of the tree `tree`, as a list. A
# single axis applies to every leaf, a list follows the structure of `tree`,
# matching named elements by name and others by position, and an axis in it
# applies to every leaf below it. With `partial`, a list may leave out named
# elements, whose leaves get `NULL`. A mismatch is an error about the `part`
# of `f` -- argument or output -- where it occurs. Errors name `call`.
expand_axes <- function(spec, tree, n, partial, arg, part, call) {
  axes <- vector("list", n)
  of <- if (part == "output") "what {.arg f} returns" else "the arguments of {.arg f}"
  abort_mismatch <- function(path) {
    where <- if (length(path)) sprintf("at `%s`", paste(path, collapse = "$")) else "at the top"
    cli_abort(
      c(
        sprintf("{.arg %s} must be a single axis or a list with the structure of %s.", arg, of),
        x = "It does not match the {part} {where}."
      ),
      call = call
    )
  }
  walk <- function(spec, node, path) {
    if (!is.list(spec)) {
      for (j in unlist(flatten(node))) {
        axes[j] <<- list(spec)
      }
      return()
    }
    if (!is.list(node)) {
      abort_mismatch(path)
    }
    keys <- names(node)
    if (!is.null(names(spec)) && all(nzchar(names(spec)))) {
      pos <- match(names(spec), keys)
      ok <- !anyNA(pos) && !anyDuplicated(pos) && (partial || length(pos) == length(node))
    } else {
      pos <- seq_along(spec)
      ok <- length(spec) == length(node) && identical(names(spec), keys)
    }
    if (!ok) {
      abort_mismatch(path)
    }
    keys <- keys %||% character(length(node))
    keys[!nzchar(keys)] <- which(!nzchar(keys))
    for (k in seq_along(spec)) {
      walk(spec[[k]], node[[pos[[k]]]], c(path, keys[[pos[[k]]]]))
    }
  }
  walk(spec, unflatten(tree, as.list(seq_len(n))), character())
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
#' Many primitives (such as elementwise ops) just work as is.
#' For others, some parameters need to be adjusted to account for the added batch axis.
#' This includes shape parameters of axis mappings.
#' The rule is then responsible for rewriting these parameters so that they
#' work correctly with the added batch axis.
#' For the most common cases, see [`vec_shape()`] where it is enough
#' to just mark the parameter with the appropriate kind via the `params` argument.
#' For more complex rules it is necessary to provide the `fn` argument.
#'
#' @param fn (`NULL` | `function`)\cr
#'   A rule of its own, for a primitive for which declaring `params` and `unbatched`
#'   is not enough. If given, the other arguments must be left at their
#'   defaults. It is called as `fn(inputs, batched, params, size)` for every
#'   call of the primitive with at least one operand that is mapped over, where
#'   * `inputs` is the list of operands, in the order of the primitive's
#'     arguments: one that is mapped over with the batch axis as its last axis,
#'     the others as they are,
#'   * `batched` is a `logical()` saying which of `inputs` are mapped over,
#'   * `params` is the named list of the parameters of the call, and
#'   * `size` is the size of the batch axis.
#'
#'   It returns the outputs of the primitive applied to every slice, as a list
#'   in the order of the primitive's outputs, or a single output as it is.
#'   Every output has the batch axis as its last axis, also one that is the
#'   same for every slice.
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
