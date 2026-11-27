check_wrt_arrayish <- function(args_flat, is_wrt_flat) {
  for (arg in args_flat[is_wrt_flat]) {
    if (!is_arrayish(arg)) {
      cli_abort(c(
        "Cannot compute gradient with respect to non-array argument.",
        x = "Got {.cls {class(arg)}}"
      ))
    }

    dt <- peek_dtype(arg)
    if (!is_dtype_float(dt)) {
      # `repr()` on a data type gives stablehlo's spelling (`i1` for a
      # boolean); the pages speak anvl's, which `as.character()` gives.
      cli_abort(c(
        "Can only compute gradient with respect to float arrays.",
        x = "Got {.val {as.character(dt)}}."
      ))
    }

    # A value with no data type of its own cannot be differentiated with
    # respect to: the gradient comes back at whatever data type the forward
    # pass happened to settle the value at, so the answer would depend on how
    # the rest of the body used it rather than on what the caller passed.
    # Materializing it here would only hide that behind the default.
    if (has_no_dtype(arg)) {
      cli_abort(c(
        "Cannot compute gradient with respect to a value that has no data type.",
        x = "It is an R {peek_r_type(arg)}, which takes its data type from the way the function body uses it (see {.code ?RData}).", # nolint
        i = "Give it one first, e.g. {.code nv_array(x, dtype = \"f32\")} or an explicit {.code nv_array(x, dtype = \"f64\")}, so the gradient's data type is the caller's choice." # nolint
      ))
    }
  }
}

# The R storage type behind a value that has no data type yet, for messages.
peek_r_type <- function(x) {
  to_abstract(x)$r_type %||% "value"
}

# The leaves of the arguments `args` and their tree (see `flatten_args()`), and
# which leaves belong to the arguments `wrt` names, as `is_wrt_flat`.
prepare_gradient_args <- function(args, wrt) {
  prep <- flatten_args(args)
  prep$is_wrt_flat <- if (!is.null(wrt)) {
    pjrt::tree_leaf_mask(prep$in_tree, wrt)
  } else {
    rep(TRUE, length(prep$args_flat))
  }
  check_wrt_arrayish(prep$args_flat, prep$is_wrt_flat)
  prep
}

#' @title Pullback Rule
#' @description
#' Construct a pullback rule for a primitive, the rule that reverse-mode
#' automatic differentiation uses. Provide exactly one of `backward` and
#' `forward`.
#'
#' Pass `backward` when the primitive's forward statement can run unmodified, which
#' covers most use cases. It has the signature
#' `function(inputs, outputs, grads, params, required)` and returns a `list`
#' with one entry per input: that input's gradient, or `NULL` where
#' `required` says it is not needed.
#'
#' Pass `forward` when a slightly different forward pass enables a more
#' efficient backward pass. It has the signature
#' `function(inputs, params, required)`, where `required` says which inputs
#' need a gradient, and returns `list(outputs = , backward = )`: the forward
#' results and a closure with the signature of `backward` above, which can use
#' intermediate values of the forward pass via lexical scoping.
#'
#' @param backward (`NULL` | `function`)\cr
#'   Backward hook for the default case.
#' @param forward (`NULL` | `function`)\cr
#'   Alternative forward hook that returns both the outputs and a backward
#'   closure.
#' @return (`anvl_rule_pullback`)
#' @examples
#' # the rule of prim_negate()
#' rule_pullback(function(inputs, outputs, grads, params, required) {
#'   list(if (required[[1L]]) prim_negate(grads[[1L]]))
#' })
#' @seealso [`graph_gradient()`]
#' @export
rule_pullback <- function(backward = NULL, forward = NULL) {
  if (is.null(backward) == is.null(forward)) {
    cli_abort("Provide exactly one of {.arg backward} or {.arg forward}.")
  }
  if (!is.null(backward)) {
    checkmate::assert_function(backward)
  } else {
    checkmate::assert_function(forward)
  }
  structure(
    list(forward = forward, backward = backward),
    class = "anvl_rule_pullback"
  )
}

#' @title Transform a Graph to Its Gradient
#' @description
#' Low-level graph transformation that transforms a graph into its gradient.
#' The function `f` represented by `graph` must return a single
#' float scalar. The resulting graph computes the gradients of that scalar with respect
#' to the inputs specified by `wrt`.
#'
#' @details
#' To support alternative forward passes for more efficient backward passes, we
#' replay and possibly rewrite the graph into a new descriptor.
#' Afterwards, we traverse it backwards and call the pullback rules where necessary.
#'
#' See [`rule_pullback()`] for more information.
#'
#' [`gradient()`] and [`value_and_gradient()`] differentiate the same way, but
#' into the trace they are called in rather than into a graph of its own;
#' prefer them unless you need to operate on graphs directly.
#' @param graph ([`AnvlGraph`])\cr
#'   The graph to transform. Must produce a single scalar float output.
#' @param wrt (`NULL` | `character()`)\cr
#'   Names of the graph inputs to differentiate with respect to. `NULL` differentiates with respect
#'   to all inputs.
#' @return ([`AnvlGraph`])\cr
#'   Its outputs are the requested gradients.
#' @seealso [`gradient()`], [`value_and_gradient()`], [`rule_pullback()`]
#' @export
#' @examples
#' graph <- trace_fn(prim_mul, list(nv_aval("f32", integer()), nv_aval("f32", integer())))
#' graph
#' graph_gradient(graph, "lhs")
graph_gradient <- function(graph, wrt) {
  # `graph` may be a promise of a trace, which must not run inside `desc`.
  force(graph)
  desc <- local_descriptor()
  # The gradient takes the side effects of `graph` as `graph` does, through the
  # same inputs, and makes the same uses of them.
  desc$side_effects <- lapply(graph$side_effects, function(x) {
    value <- lapply(x$value, function(box) register_gval(desc, box$gnode))
    SideEffectView(x$effect, value, n = x$n, inputs = x$inputs)
  })
  out <- validate_gradient_output(graph$outputs)
  seed <- gradient_seed(desc, out)
  grad <- graph_pullback(graph, compute_requirements(graph, wrt))$pullback(list(seed))
  desc$outputs <- lapply(grad, \(box) box$gnode)
  desc$in_tree <- graph$in_tree
  desc$is_static_flat <- graph$is_static_flat
  desc$static_args_flat <- graph$static_args_flat
  desc$out_tree <- gradient_out_tree(graph, wrt)
  descriptor_to_graph(desc)
}

# Replays the forward pass of `graph` into the current descriptor, as a
# function of `inputs` -- the boxes its inputs are bound to -- or, where
# `inputs` is `NULL`, of inputs of the descriptor's own (`graph_gradient()`).
# `reqs` says which values require a gradient, as `compute_requirements()`
# returns it. Returns the boxes of its outputs as `value`, and its `pullback`:
# a function that takes a cotangent per output, `NULL` for a zero one, records
# the backward pass into the descriptor current when it is called, and returns
# the cotangents of the inputs `reqs$requires_grad` selects. It can be called
# more than once.
graph_pullback <- function(graph, reqs, inputs = NULL) {
  rebuilt <- rebuild_forward_into(graph, current_descriptor(), inputs, reqs$required_env)
  targets <- graph$inputs[reqs$requires_grad]
  list(
    value = lapply(graph$outputs, rebuilt$box_of),
    pullback = function(out_cts) pull_back(graph, rebuilt$backwards, reqs$required_env, targets, out_cts)
  )
}

# The tree the gradients of `graph`'s inputs come back in: that of the inputs
# `wrt` names, or of all of them where `wrt` is `NULL`.
gradient_out_tree <- function(graph, wrt) {
  if (!is.null(wrt)) {
    pjrt::tree_filter_by_names(graph$in_tree, wrt)
  } else {
    graph$in_tree
  }
}

validate_gradient_output <- function(out_gvals) {
  if (length(out_gvals) != 1L) {
    cli_abort("gradient can only be computed for functions that return a single output")
  }
  out <- out_gvals[[1L]]
  if (!identical(shape(out$aval), integer())) {
    cli_abort("gradient can only be computed for functions that return a scalar")
  }
  dt <- out$aval$dtype
  if (!is_dtype_float(dt)) {
    cli_abort(c(
      x = "gradient can only be computed for functions that return float scalar",
      i = "Got dtype={.field {as.character(dt)}}"
    ))
  }
  out
}

# The cotangent reverse mode starts from for the scalar output `out`: its
# derivative with respect to itself, a literal 1 at its data type. StableHLO
# has no literals at the 16-bit floats, so those are refused.
gradient_seed <- function(desc, out) {
  dtype <- out$aval$dtype
  if (as.character(dtype) %in% c("f16", "bf16")) {
    cli_abort(c(
      "Can't differentiate a function that returns a {.val {as.character(dtype)}} array.",
      i = "Convert the output to {.val f32} or {.val f64}."
    ))
  }
  get_box_or_register_const(desc, GraphLiteral(LiteralArray(1L, shape = integer(), dtype = dtype)))
}

# Determine which gvals will require a gradient.
# Returns:
#   - required_env: hashtab(gval -> logical), used by phase 2 to skip calls
#     that don't contribute to a `wrt` input.
#   - requires_grad: logical vector aligned with `graph$inputs`, used by
#     phase 3 to pick which inputs to emit gradients for.
compute_requirements <- function(graph, wrt) {
  requires_grad_all <- if (is.null(wrt)) {
    rep(TRUE, tree_size(graph$in_tree))
  } else {
    pjrt::tree_leaf_mask(graph$in_tree, wrt)
  }
  # `in_tree` may include static (non-array) args not present in
  # `graph$inputs`; filter them out. `gradient()` already rejects static
  # `wrt` entries, so this drop is safe.
  is_static <- graph$is_static_flat
  if (!is.null(is_static) && any(requires_grad_all & is_static)) {
    # The argument each flat leaf belongs to: the top-level child's name,
    # repeated once per leaf beneath it.
    sizes <- pjrt::tree_child_sizes(graph$in_tree)
    nms <- pjrt::tree_child_names(graph$in_tree) %||% rep("", length(sizes))
    flat_argnames <- rep(nms, times = sizes)
    bad <- unique(flat_argnames[requires_grad_all & is_static])
    cli_abort(c(
      "Cannot compute gradient with respect to {.arg {bad}}.",
      x = "{cli::qty(length(bad))}{?It was/They were} passed as {?a plain R value/plain R values}",
      i = "{cli::qty(length(bad))}Pass {?it/them} as an {.cls AnvlArray}."
    ))
  }
  requires_grad <- if (is.null(is_static)) {
    requires_grad_all
  } else {
    requires_grad_all[!is_static]
  }

  required_env <- hashtab()
  for (i in seq_along(graph$inputs)) {
    required_env[[graph$inputs[[i]]]] <- requires_grad[[i]]
  }
  for (const in graph$constants) {
    required_env[[const]] <- FALSE
  }
  required_env <- propagate_requirements(graph, required_env)

  list(required_env = required_env, requires_grad = requires_grad)
}

# The backward pass of a sub-graph whose forward `rebuild_forward_into()`
# replayed: seeds `graph`'s outputs with `out_grads`, runs the pullback rules
# `backwards` in the current descriptor, and returns the cotangent of each
# value in `targets` -- a zero where the target did not reach any output.
pull_back <- function(graph, backwards, required_env, targets, out_grads) {
  grad_env <- hashtab()
  for (i in seq_along(graph$outputs)) {
    out <- graph$outputs[[i]]
    # Only an output that requires a gradient is seeded, so that no value
    # the pullback rules leave out gains a cotangent.
    if (!is_graph_literal(out) && isTRUE(required_env[[out]]) && !is.null(out_grads[[i]])) {
      # The same value can be returned more than once, e.g. a branch
      # `\() list(s, s)`, or a branch returning one of its inputs in two
      # places. Its cotangents accumulate, as for any other reuse.
      grad_env[[out]] <- if (is.null(grad_env[[out]])) out_grads[[i]] else prim_add(grad_env[[out]], out_grads[[i]])
    }
  }

  grad_env <- run_backward_pass(graph, backwards, required_env, grad_env)
  lapply(targets, function(target) {
    grad_env[[target]] %||% zeros(target$aval$dtype, shape(target$aval))
  })
}

# Splits the reverse of a sub-graph -- a branch of `prim_if()`, the body of
# `prim_scan()` -- into two closed graphs, so that its backward pass does not
# have to rerun the forward one, as JAX's partial evaluation of a `cond` or a
# `scan` does:
#   - fwd: `graph`'s forward pass, taking the same inputs, returning its
#     outputs and then the residuals -- the values computed by the forward
#     pass that `bwd` reads.
#   - bwd: taking a cotangent per output of `graph`, then `graph`'s inputs,
#     then the residuals, and returning the cotangents of
#     `graph$inputs[needed]`.
#   - residual_avals: the avals of the residuals.
split_vjp <- function(graph, needed) {
  targets <- graph$inputs[needed]
  required_env <- requirements_from(graph, targets)

  desc_fwd <- local_descriptor()
  rebuilt <- rebuild_forward_into(graph, desc_fwd, required_env = required_env)
  bwd <- pull_back_graph(graph, rebuilt$backwards, required_env, targets)

  # What the backward pass read of the forward one, it captured. An input of
  # `graph` it reads as an input of its own; the rest are the residuals. An
  # array it closed over stays a constant.
  map <- hashtab()
  inputs <- lapply(graph$inputs, function(g) {
    map[[g]] <- GraphValue(aval = g$aval, label = g$label)
  })
  captured <- Filter(\(g) !is_concrete_array(g$aval), bwd$constants)
  residuals <- Filter(\(g) is.null(map[[g]]), captured)
  for (g in residuals) {
    # The branch is pure, so what its backward pass reads was computed by its
    # forward pass.
    if (is.null(desc_fwd$gval_to_box[[g]])) {
      cli_abort("Internal error: the backward pass of a sub-graph read a value from outside it.")
    }
    map[[g]] <- GraphValue(aval = g$aval, label = g$label)
  }
  substitute_gnodes(bwd, map)
  bwd$inputs <- c(bwd$inputs, inputs, lapply(residuals, \(g) map[[g]]))
  bwd$constants <- Filter(\(g) is_concrete_array(g$aval), bwd$constants)

  outputs <- lapply(graph$outputs, \(g) if (is_graph_literal(g)) g else rebuilt$trans[[g]] %||% g)
  desc_fwd$outputs <- c(outputs, residuals)
  list(
    fwd = descriptor_to_graph(desc_fwd),
    bwd = bwd,
    residual_avals = lapply(residuals, \(g) g$aval)
  )
}

# The backward pass of `split_vjp()`, traced into a graph of its own
# whose inputs are the cotangents of `graph`'s outputs. The values of the
# forward pass the pullback rules read become its constants.
pull_back_graph <- function(graph, backwards, required_env, targets) {
  desc <- local_descriptor()
  out_grads <- lapply(graph$outputs, function(out) {
    register_input(desc, GraphValue(aval = AbstractArray(dtype = out$aval$dtype, shape = out$aval$shape)))
  })
  cts <- pull_back(graph, backwards, required_env, targets, out_grads)
  desc$outputs <- lapply(cts, \(ct) maybe_box_arrayish(ct, desc)$gnode)
  descriptor_to_graph(desc)
}

# `compute_requirements()` reads the set to differentiate with respect to off
# the graph's argument names; a sub-graph has no arguments, so its targets are
# named directly.
requirements_from <- function(graph, targets) {
  required_env <- hashtab()
  for (gval in c(graph$inputs, graph$constants)) {
    required_env[[gval]] <- FALSE
  }
  for (target in targets) {
    required_env[[target]] <- TRUE
  }
  propagate_requirements(graph, required_env)
}

# Forward propagation over `graph`'s calls, starting from the seeded
# `required_env`: a call's float outputs require a gradient exactly when one of
# its operands does. An integer or boolean output never does -- it has no
# derivative, and the pullback rules give such values a zero -- so a random state
# or a loop counter computed alongside a float does not drag the calls it feeds
# into the backward pass. Literals are inlined constants and never do either.
propagate_requirements <- function(graph, required_env) {
  for (call in graph$statements) {
    requires <- any(vapply(
      call$inputs,
      function(x) !is_graph_literal(x) && isTRUE(required_env[[x]]),
      logical(1L)
    ))
    for (out_node in call$outputs) {
      required_env[[out_node]] <- requires && is_dtype_float(out_node$aval$dtype)
    }
  }
  required_env
}

# Replays `graph`'s forward pass into `desc`, call by call. `split_vjp()` uses
# it to rebuild a sub-graph -- a branch of `prim_if()`, the body of
# `prim_scan()` -- `graph_apply()` to rerun a closed graph, and
# `graph_pullback()` and `graph_pushforward()` to replay what `pullback()` and
# `pushforward()` traced into the descriptor they are called in.
#
# Returns:
#   - trans: hashtab(original gval -> new gval) for every replaced output,
#     every input `inputs` binds, and every constant whose array `desc` already
#     holds under a GraphValue of its own. Other constants and literals are not
#     added (they fall through).
#   - backwards: ordered list that needs to be traversed in reverse for the
#     backward pass.
#   - translate: function(original gval) -> the gval `desc` holds for it.
#   - box_of: function(original gval) -> the box `desc` holds for it.
#
# `inputs` are the boxes the graph's inputs are bound to -- the arguments
# passed to the gradient function -- with `NULL` for one the graph does not
# read. Without them, the graph's inputs become inputs of `desc`.
#
# `fresh = TRUE` gives every replayed call new output gvals, so that `trans`
# maps every output the graph computes, not only the replaced ones.
#
# With `required_env`, a call none of whose operands requires a gradient keeps
# its plain forward even where its rule has a replacement: the backward pass
# skips it, so what the replacement keeps for it -- an if's residuals -- would
# be computed for nothing.
rebuild_forward_into <- function(graph, desc, inputs = NULL, required_env = NULL, fresh = FALSE) {
  if (is.null(inputs)) {
    register_inputs(desc, graph$inputs)
  }
  # Existing GraphValues are reused where possible to minimize cloning.
  # An alternative forward pass replaces the outputs of its statement, which
  # later statements may read, so every input is looked up in `trans`. An
  # input passed as `NULL` must not be read: it would fall through to a
  # GraphValue `desc` never defined.
  bound <- bind_graph(graph, desc, inputs)
  trans <- bound$trans
  translate_gnode <- function(g) {
    if (is_graph_literal(g)) {
      return(g)
    }
    if (isTRUE(bound$unbound[[g]])) {
      cli_abort("Internal error: a replayed graph read an input that was passed as `NULL`.")
    }
    trans[[g]] %||% g
  }
  # The box for a translated gval, minted on demand (a literal has none until
  # it is used as a call input).
  box_for <- function(g) register_gval(desc, translate_gnode(g))

  # One entry per statement, which run_backward_pass() traverses in reverse.
  backwards <- vector("list", length(graph$statements))

  for (i in seq_along(graph$statements)) {
    call <- graph$statements[[i]]
    rule <- call$primitive[["pullback"]]

    needs_grad <- is.null(required_env) ||
      any(vapply(call$inputs, \(x) !is_graph_literal(x) && isTRUE(required_env[[x]]), logical(1L)))

    if (is.null(rule) || is.null(rule$forward) || !needs_grad) {
      # No rule, a backward-only rule, or a call the backward pass skips: the
      # forward computation is unchanged, so the original output gvals are
      # reused directly, unless `fresh` asks for new ones. The statement is
      # rebuilt to read its inputs through `trans`, in case an upstream
      # alternative forward replaced one of them.
      new_inputs <- lapply(call$inputs, translate_gnode)
      new_outputs <- call$outputs
      if (fresh) {
        new_outputs <- lapply(call$outputs, \(g) GraphValue(aval = g$aval))
        for (j in seq_along(call$outputs)) {
          trans[[call$outputs[[j]]]] <- new_outputs[[j]]
        }
      }
      replay_statement(desc, call, new_inputs, new_outputs)

      # Without a pullback rule `backwards[[i]]` stays NULL. `run_backward_pass`
      # treats that as "skip if no input requires grad, otherwise abort":
      # primitives like `prim_fill` whose inputs are all static parameters
      # never reach the abort branch, so they don't need a pullback rule, and
      # neither does a skipped call whose rule only has a replacement forward.
      if (!is.null(rule$backward)) {
        backwards[[i]] <- list(
          fn = rule$backward,
          inputs = lapply(new_inputs, GraphBox, desc = desc),
          outputs = lapply(new_outputs, GraphBox, desc = desc),
          params = call$params
        )
      }
    } else {
      # Alternative-forward path
      # Here, new GraphValue outputs are generated and subsequent GraphStatements that
      # referenced the old ones need to be rewired
      input_boxes <- lapply(call$inputs, box_for)
      required <- vapply(
        call$inputs,
        \(x) is.null(required_env) || (!is_graph_literal(x) && isTRUE(required_env[[x]])),
        logical(1L)
      )
      fwd_result <- rule$forward(input_boxes, call$params, required)
      for (j in seq_along(call$outputs)) {
        trans[[call$outputs[[j]]]] <- fwd_result$outputs[[j]]$gnode
      }
      backwards[[i]] <- list(
        fn = fwd_result$backward,
        inputs = input_boxes,
        outputs = fwd_result$outputs,
        params = call$params
      )
    }
  }

  list(trans = trans, backwards = backwards, translate = translate_gnode, box_of = box_for)
}

# Replays `graph` into the current descriptor as a function of `inputs`
# (boxes, one per graph input, or `NULL` for an input the graph does not read)
# and returns the boxes of its outputs, e.g. a branch of `prim_if()`'s forward
# pass that `split_vjp()` prepared. No backward pass follows, so every call
# keeps its plain forward: an empty `required_env` says that nothing requires a
# gradient. Every call gets new output gvals, so that a graph applied more than
# once in one trace never defines the same value twice.
graph_apply <- function(graph, inputs) {
  rebuilt <- rebuild_forward_into(graph, current_descriptor(), inputs, required_env = hashtab(), fresh = TRUE)
  lapply(graph$outputs, rebuilt$box_of)
}

# Walk statements in reverse, invoking each statement's backward to accumulate
# gradients keyed by the *original* graph's gvals.
run_backward_pass <- function(graph, backwards, required_env, grad_env) {
  add_or_init <- function(grad1, grad2) {
    if (is.null(grad1)) {
      return(grad2)
    }
    if (is.null(grad2)) {
      cli_abort("Internal error: a pullback rule returned no gradient for an input that requires one.")
    }
    prim_add(grad1, grad2)
  }

  for (i in rev(seq_along(graph$statements))) {
    call <- graph$statements[[i]]
    input_required <- vapply(
      call$inputs,
      function(x) required_env[[x]] %||% FALSE,
      logical(1L)
    )
    # A call none of whose outputs requires a gradient contributes none, even
    # where an operand requires one -- e.g. a loop that only counts.
    output_required <- vapply(call$outputs, \(x) isTRUE(required_env[[x]]), logical(1L))
    if (!any(input_required) || !any(output_required)) {
      next
    }

    output_grads <- lapply(call$outputs, \(output) {
      # Output grad may be NULL if there is dead code.
      grad_env[[output]] %||%
        zeros(dtype(output), shape(output))
    })

    bwd <- backwards[[i]]
    if (is.null(bwd)) {
      cli_abort(c(
        "No pullback rule for primitive {.field {call$primitive$name}}.",
        i = "Cannot compute gradient through this primitive."
      ))
    }

    # An input that requires no gradient gets none: the rule returns `NULL`
    # for it, and nothing is accumulated.
    input_grads <- bwd$fn(bwd$inputs, bwd$outputs, output_grads, bwd$params, input_required)
    for (j in which(input_required)) {
      input_gval <- call$inputs[[j]]
      ct <- input_grads[[j]]
      # Elementwise primitives broadcast a scalar operand, so a gradient of the
      # wrong shape would otherwise be accumulated without complaint. The rule
      # may be a user's own, so this is not reported as an internal error.
      if (!is.null(ct) && !identical(shape(ct), shape(input_gval))) {
        cli_abort(c(
          "The pullback rule of primitive {.field {call$primitive$name}} returned a gradient of the wrong shape for input {j}.", # nolint
          x = "Expected {shape_repr(shape(input_gval))}, got {shape_repr(shape(ct))}."
        ))
      }
      grad_env[[input_gval]] <- add_or_init(grad_env[[input_gval]], ct)
    }
  }

  grad_env
}

#' @title Gradients
#' @description
#' Return a new function that computes the gradient of `f` via reverse-mode
#' automatic differentiation. `f` must return a single float scalar. The
#' returned function has the same signature as `f`.
#'
#' * `gradient()` returns only the gradients, structured like the inputs (or
#'   the subset selected by `wrt`).
#' * `value_and_gradient()` returns both the output of `f` and its gradients,
#'   computed in a single forward and reverse pass.
#' @param f (`function`)\cr
#'   Function to differentiate. Must return a single scalar float array.
#' @param wrt (`character` | `integer` | `NULL`)\cr
#'   Names or positions of the arguments to compute the gradient with respect to.
#'   Only float arrays can be included; static arguments must not appear in
#'   `wrt`. At call time, an argument in `wrt` must be an array with a data
#'   type, not an R value such as `3`, since the data type of its gradient
#'   would otherwise be undetermined.
#'   If `NULL` (the default), the gradient is computed with respect to all
#'   arguments (which must all be arrayish in that case).
#' @return (`function`)\cr
#'   Has the same formals as `f` and must be called inside [`jit()`].
#'   For `gradient()`, it returns a named `list` of gradients, one per
#'   argument of `f` (or per `wrt` entry), each structured like that argument.
#'   For `value_and_gradient()`, it returns `list(value = , grad = )`: the
#'   return value of `f`, and that same `list` of gradients.
#' @seealso `r roxy_article("autodiff")`, [`graph_gradient()`] for the
#'   low-level graph transformation.
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' f <- function(x, y) sum(x * y)
#' g <- jit(gradient(f))
#' g(nv_array(c(1, 2), dtype = "f32"), nv_array(c(3, 4), dtype = "f32"))
#'
#' # differentiate with respect to a single argument
#' g_x <- jit(gradient(f, wrt = "x"))
#' g_x(nv_array(c(1, 2), dtype = "f32"), nv_array(c(3, 4), dtype = "f32"))
#'
#' # static (non-array) arguments are passed through but cannot be in wrt
#' f2 <- function(x, power) sum(x^power)
#' g2 <- jit(gradient(f2, wrt = "x"), static = "power")
#' g2(nv_array(c(1, 2, 3), dtype = "f32"), power = 2L)
#'
#' # an argument in `wrt` must be passed as an array, not as an R value
#' g3 <- jit(gradient(function(x) x^2L))
#' g3(nv_scalar(3))
#' try(g3(3))
#'
#' # the value of `f` together with its gradient
#' loss_fn <- function(x) sum(x^2L)
#' vg <- jit(value_and_gradient(loss_fn))
#' result <- vg(nv_array(c(3, 4), dtype = "f32"))
#' result$value
#' result$grad
gradient <- function(f, wrt = NULL) {
  assert_function(f)
  wrt <- resolve_transformation_args(f, wrt, "wrt")
  transformation_fn(f, function(args) trace_value_and_grad(f, args, wrt, "gradient")$grad)
}

#' @rdname gradient
#' @export
value_and_gradient <- function(f, wrt = NULL) {
  assert_function(f)
  wrt <- resolve_transformation_args(f, wrt, "wrt")
  transformation_fn(f, function(args) trace_value_and_grad(f, args, wrt, "value_and_gradient"))
}

# gradient() / value_and_gradient() (named by `fn`) at the call's arguments:
# the pullback of `f` with respect to the arguments `wrt` names, applied to a 1
# for its value.
trace_value_and_grad <- function(f, args, wrt, fn, call = rlang::caller_env()) {
  prep <- prepare_gradient_args(args, wrt)
  assert_in_trace(fn, call = call)
  out <- trace_pullback(f, prep, fn)
  value <- validate_gradient_output(lapply(flatten(out$value), \(box) box$gnode))
  seed <- gradient_seed(current_descriptor(), value)
  grad <- out$pullback(unflatten(build_tree(out$value), list(seed)))
  list(value = out$value, grad = if (is.null(wrt)) grad else grad[names(grad) %in% wrt])
}

#' @title Pullbacks
#' @description
#' Evaluate `.f` at the arguments `...` and return its pullback: the function
#' that takes a cotangent of the result and returns the corresponding cotangent
#' back to the arguments, by reverse-mode automatic differentiation. That is
#' the cotangent times the Jacobian of `.f`, the vector-Jacobian product.
#'
#' For \eqn{f: \mathbb{R}^n \to \mathbb{R}^m}{f: R^n -> R^m}, whose Jacobian
#' \eqn{J_f(x)}{J_f(x)} at \eqn{x}{x} is the \eqn{m \times n}{m x n} matrix of
#' the derivatives \eqn{\partial f_i / \partial x_j}{d f_i / d x_j}, the
#' pullback at \eqn{x}{x} is the linear map
#' \deqn{w \mapsto J_f(x)^\top w}{w -> t(J_f(x)) w}
#' from a cotangent \eqn{w \in \mathbb{R}^m}{w in R^m} of the result to the
#' gradient of the weighted sum \eqn{\sum_i w_i f_i(x)}{sum_i w_i f_i(x)}. With
#' the \eqn{i}{i}-th unit vector as \eqn{w}{w}, it gives the \eqn{i}{i}-th row
#' of the Jacobian.
#'
#' * `pullback()` returns only the pullback.
#' * `value_and_pullback()` returns both the result of `.f` and its pullback.
#'
#' `.f` runs once, when `pullback()` is called. The pullback can be called more
#' than once, e.g. with every unit vector to build a Jacobian row by row, as
#' [`jacobian()`] does.
#' @param .f (`function`)\cr
#'   Function to differentiate. It can return any (nested) list of arrays.
#' @param ... (any)\cr
#'   The arguments to evaluate `.f` at, matched to its formals as in a call of
#'   `.f`.
#' @return
#'   For `pullback()`, the pullback (`function`); for `value_and_pullback()`,
#'   `list(value = , pullback = )`: the return value of `.f`, and its pullback.
#'
#'   The pullback takes one argument, `cotangent`, structured like the value
#'   of `.f`, with an array of the same shape and data type for each float
#'   array of it; an R value takes the data type of its array. `NULL` in place
#'   of an array -- or of a whole part of the value -- stands for zeros. An
#'   array of the value that is not a float has no cotangent, and gets `NULL`.
#'
#'   The pullback returns a named `list` with the cotangent of each argument,
#'   structured like that argument, as [`gradient()`] does. An argument `.f`
#'   takes through its `...` keeps the name it was passed under: name it in
#'   the call to read its cotangent by name. One that is not a float array --
#'   an integer or boolean array, an R value such as `3`, a random state -- is
#'   held fixed, and gets `NULL` in place of a cotangent, as does each such
#'   part of a list.
#'
#'   Like [`gradient()`], `pullback()` must be called inside [`jit()`]. The
#'   pullback must be called in the same function, or in a function traced
#'   inside it, such as a branch of [`nv_if()`].
#' @seealso `r roxy_article("autodiff")`, [`gradient()`]
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' f <- function(x, y) x * y
#' x <- nv_array(c(1, 2, 3))
#' y <- nv_array(c(4, 5, 6))
#' # cotangent * y and cotangent * x
#' g <- jit(function(x, y, ct) pullback(f, x, y)(ct))
#' g(x, y, nv_array(c(1, 0, 1)))
#'
#' # the value of `f` together with its pullback
#' h <- jit(function(x, y, ct) {
#'   out <- value_and_pullback(f, x, y)
#'   list(value = out$value, x = out$pullback(ct)$x)
#' })
#' h(x, y, nv_array(c(1, 0, 1)))
#'
#' # a result with several parts: `NULL` is the cotangent of one that does not
#' # matter
#' stats <- function(x) list(total = sum(x), squares = x^2L)
#' dtotal <- jit(function(x) pullback(stats, x)(list(total = 1, squares = NULL)))
#' dtotal(x)
pullback <- function(.f, ...) {
  assert_function(.f)
  trace_pullback(.f, differentiated_args(.f, list(...)), "pullback")$pullback
}

#' @rdname pullback
#' @export
value_and_pullback <- function(.f, ...) {
  assert_function(.f)
  trace_pullback(.f, differentiated_args(.f, list(...)), "value_and_pullback")[c("value", "pullback")]
}

# The pullback of `f` for `pullback()`, `value_and_pullback()`, `gradient()` or
# `jacobian()`, named by `fn`, at the arguments `prep` holds (see
# `prepare_gradient_args()`): `f` traced into a graph whose forward pass is
# replayed into the enclosing trace, and the function that records its
# backward pass there. It differentiates the leaves `prep$is_wrt_flat`
# selects, and returns `NULL` for the others. `flat` holds the same on the
# leaves: the boxes of the value, its tree `out_tree`, and a pullback from a
# cotangent per leaf of the value (`NULL` for a zero) to the cotangents of the
# leaves `prep$is_wrt_flat` selects.
trace_pullback <- function(f, prep, fn, call = rlang::caller_env()) {
  traced <- trace_replayed(f, prep, fn, call = call, replay = function(graph, operands) {
    # A leaf that is no array is static, and no input of the graph.
    requires_grad <- prep$is_wrt_flat[!graph$is_static_flat]
    reqs <- list(
      required_env = requirements_from(graph, graph$inputs[requires_grad]),
      requires_grad = requires_grad
    )
    graph_pullback(graph, reqs, operands)
  })
  graph <- traced$graph
  value <- unflatten(graph$out_tree, traced$res$value)
  pull_flat <- function(cts) {
    assert_called_within(traced$enclosing, "pullback", fn)
    traced$res$pullback(cts)
  }
  pull <- function(cotangent) {
    assert_called_within(traced$enclosing, "pullback", fn)
    cts <- flatten_cotangent(value, cotangent, graph$out_tree, call = rlang::current_env())
    pulled <- vector("list", length(prep$args_flat))
    pulled[prep$is_wrt_flat] <- pull_flat(cts)
    unflatten(prep$in_tree, pulled)
  }
  list(
    value = value,
    pullback = pull,
    flat = list(value = traced$res$value, out_tree = graph$out_tree, pullback = pull_flat)
  )
}

# Whether `x`, a leaf of the arguments of `pullback()`, is a float array with a
# data type of its own. An R value takes its data type from the way the body
# uses it, so the data type of its cotangent would not be the caller's choice.
is_float_array <- function(x) {
  is_arrayish(x) && !has_no_dtype(x) && is_dtype_float(dtype(x))
}

# Whether `desc` is `ancestor` or a trace opened inside it.
descriptor_within <- function(desc, ancestor) {
  while (!is.null(desc)) {
    if (identical(desc, ancestor)) {
      return(TRUE)
    }
    desc <- desc$parent
  }
  FALSE
}

# The cotangent `cotangent` of `value`, the result of the function
# `pullback()` differentiates, whose structure `out_tree` is, as a list with an
# entry per array of `value`: a box of the current trace at the array's data
# type and shape, or `NULL` for a zero one. Errors name `call`.
flatten_cotangent <- function(value, cotangent, out_tree, call) {
  Map(
    function(x, ct, i) {
      if (is.null(ct)) {
        return(NULL)
      }
      path <- pjrt::tree_path(out_tree, i)
      at <- if (nzchar(path)) cli::format_inline(" at {.code {path}}") else ""
      if (!is_dtype_float(dtype(x))) {
        cli_abort(
          c(
            "The value of {.arg .f} has no cotangent{at}.",
            x = "It is {.val {as.character(dtype(x))}}, not a float.",
            i = "Pass {.code NULL} there."
          ),
          call = call
        )
      }
      check_seed(x, ct, kind = "cotangent", where = at, of = "value", call = call)
    },
    flatten(value),
    leaves_like(value, cotangent, call, "{.arg cotangent} must be structured like the value of {.arg .f}."),
    seq_len(tree_size(out_tree))
  )
}
