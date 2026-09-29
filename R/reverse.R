check_wrt_arrayish <- function(args_flat, is_wrt_flat) {
  for (i in seq_along(args_flat)) {
    if (is_wrt_flat[[i]]) {
      if (!is_arrayish(args_flat[[i]])) {
        cli_abort(c(
          "Cannot compute gradient with respect to non-array argument.",
          x = "Got {.cls {class(args_flat[[i]])}}"
        ))
      }

      if (!is_dtype_float(peek_dtype(args_flat[[i]]))) {
        # `repr()` on a data type gives stablehlo's spelling (`i1` for a
        # boolean); the pages speak anvl's, which `as.character()` gives.
        cli_abort(c(
          "Can only compute gradient with respect to float arrays.",
          x = "Got {.val {as.character(peek_dtype(args_flat[[i]]))}}."
        ))
      }

      # A value with no data type of its own cannot be differentiated with
      # respect to: the gradient comes back at whatever data type the forward
      # pass happened to settle the value at, so the answer would depend on how
      # the rest of the body used it rather than on what the caller passed.
      # Materializing it here would only hide that behind the default.
      if (has_no_dtype(args_flat[[i]])) {
        cli_abort(c(
          "Cannot compute gradient with respect to a value that has no data type.",
          x = "It is an R {peek_r_type(args_flat[[i]])}, which takes its data type from the way the function body uses it (see {.code ?RData}).", # nolint
          i = "Give it one first, e.g. {.code nv_array(x, dtype = \"f32\")} or an explicit {.code nv_array(x, dtype = \"f64\")}, so the gradient's data type is the caller's choice." # nolint
        ))
      }
    }
  }
}

# The R storage type behind a value that has no data type yet, for messages.
peek_r_type <- function(x) {
  to_abstract(x)$r_type %||% "value"
}

prepare_gradient_args <- function(args, wrt) {
  args_flat <- flatten(args)
  in_tree <- build_tree(args)
  is_wrt_flat <- if (!is.null(wrt)) {
    pjrt::tree_leaf_mask(in_tree, wrt)
  } else {
    rep(TRUE, length(args_flat))
  }
  check_wrt_arrayish(args_flat, is_wrt_flat)
  list(args_flat = args_flat, in_tree = in_tree)
}

#' @title Reverse Rule
#' @description
#' Construct a reverse-mode autodiff rule for a primitive. Provide exactly one
#' of `backward` and `forward`.
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
#' @return (`anvl_rule_reverse`)
#' @examples
#' # the rule of prim_negate()
#' rule_reverse(function(inputs, outputs, grads, params, required) {
#'   list(if (required[[1L]]) prim_negate(grads[[1L]]))
#' })
#' @seealso [`transform_gradient()`]
#' @export
rule_reverse <- function(backward = NULL, forward = NULL) {
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
    class = "anvl_rule_reverse"
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
#' Afterwards, we traverse it backwards and call the gradient rules where necessary.
#'
#' See [`rule_reverse()`] for more information.
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
#' @seealso [`gradient()`], [`value_and_gradient()`], [`rule_reverse()`]
#' @export
#' @examples
#' graph <- trace_fn(prim_mul, list(nv_aval("f32", integer()), nv_aval("f32", integer())))
#' graph
#' transform_gradient(graph, "lhs")
transform_gradient <- function(graph, wrt) {
  # `graph` may be a promise of a trace, which must not run inside `desc`.
  force(graph)
  desc <- local_descriptor()
  res <- graph_value_and_grad(graph, wrt)
  desc$outputs <- lapply(res$grad, \(box) box$gnode)
  desc$in_tree <- graph$in_tree
  desc$is_static_flat <- graph$is_static_flat
  desc$static_args_flat <- graph$static_args_flat
  desc$out_tree <- gradient_out_tree(graph, wrt)
  descriptor_to_graph(desc)
}

# Differentiates `graph` in the descriptor currently being traced. Returns the
# boxes of the graph's outputs (`value`) and of the gradients of the inputs
# `wrt` names (`grad`), in the order of those inputs.
#
# To support alternative forward passes for more efficient backward passes, the
# forward pass is replayed -- and possibly rewritten -- into the descriptor
# (phase 1). Afterwards it is traversed backwards, calling the gradient rules
# where necessary (phase 2).
#
# `inputs` binds the graph's inputs to boxes of the descriptor, as for
# `graph_apply()`: this is how `gradient()` differentiates the closed graph it
# traced where it is called, as JAX evaluates a jaxpr into the enclosing trace.
# `NULL` registers the graph's own inputs, for a graph differentiated on its
# own.
graph_value_and_grad <- function(graph, wrt, inputs = NULL) {
  desc <- current_descriptor()
  out <- validate_gradient_output(graph$outputs)
  reqs <- compute_requirements(graph, wrt)

  rebuilt <- rebuild_forward_into(graph, desc, inputs, reqs$required_env)

  grad_env <- hashtab()
  grad_env[[out]] <- get_box_or_register_const(desc, nv_scalar(1L, dtype = out$aval$dtype))
  grad_env <- run_backward_pass(graph, rebuilt$backwards, reqs$required_env, grad_env)

  box_of <- function(g) desc$gval_to_box[[g]] %||% GraphBox(g, desc)
  list(
    value = lapply(graph$outputs, function(g) box_of(if (is_graph_literal(g)) g else rebuilt$trans[[g]] %||% g)),
    grad = lapply(collect_input_grads(graph, desc, grad_env, reqs$requires_grad), box_of)
  )
}

# The tree the gradients of `graph`'s inputs come back in: that of the inputs
# `wrt` names, or of all of them.
gradient_out_tree <- function(graph, wrt) {
  if (length(wrt)) {
    pjrt::tree_filter_by_names(graph$in_tree, wrt)
  } else {
    graph$in_tree
  }
}

# The boxes of the current descriptor that the inputs of `graph` -- traced by
# `gradient()` from `args_flat` -- are bound to, one per argument that is not
# static. An R value the descriptor has not given a data type yet materializes
# at the one the traced body settled it at, which the descriptor's own finalize
# then takes into account like any other use of it there.
gradient_operands <- function(graph, args_flat) {
  desc <- current_descriptor()
  Map(
    function(x, input) {
      if (is_rdata_box(x)) {
        x <- materialize_rdata(x, input$aval$dtype)
      }
      maybe_box_arrayish(x, desc)
    },
    args_flat[!graph$is_static_flat],
    graph$inputs
  )
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

# Determine which gvals will require a gradient.
# Returns:
#   - required_env: hashtab(gval -> logical), used by phase 2 to skip calls
#     that don't contribute to a `wrt` input.
#   - requires_grad: logical vector aligned with `graph$inputs`, used by
#     phase 3 to pick which inputs to emit gradients for.
compute_requirements <- function(graph, wrt) {
  requires_grad_all <- if (is.null(wrt) || length(wrt) == 0L) {
    rep(TRUE, tree_size(graph$in_tree))
  } else {
    pjrt::tree_leaf_mask(graph$in_tree, wrt)
  }
  # `in_tree` may include static (non-array) args not present in
  # `graph$inputs`; filter them out. `gradient()` already rejects static
  # `wrt` entries, so this drop is safe.
  is_static <- graph$is_static_flat
  if (!is.null(is_static) && any(requires_grad_all & is_static)) {
    # pjrt dropped flat_names(); the per-leaf top-level group name is the
    # top-level child's name repeated once per leaf beneath it.
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
  for (i in seq_along(graph$constants)) {
    required_env[[graph$constants[[i]]]] <- FALSE
  }
  required_env <- propagate_requirements(graph, required_env)

  list(required_env = required_env, requires_grad = requires_grad)
}

# The backward pass of a sub-graph whose forward `rebuild_forward_into()`
# replayed: seeds `graph`'s outputs with `out_grads`, runs the reverse rules
# `backwards` in the current descriptor, and returns the cotangent of each
# value in `targets` -- a zero where the target did not reach any output.
pull_back <- function(graph, backwards, required_env, targets, out_grads) {
  grad_env <- hashtab()
  for (i in seq_along(graph$outputs)) {
    out <- graph$outputs[[i]]
    # Only an output that requires a gradient is seeded, so that no value
    # the reverse rules leave out gains a cotangent.
    if (!is_graph_literal(out) && isTRUE(required_env[[out]]) && !is.null(out_grads[[i]])) {
      # An output repeated in the list accumulates, as any other reuse does.
      grad_env[[out]] <- if (is.null(grad_env[[out]])) out_grads[[i]] else prim_add(grad_env[[out]], out_grads[[i]])
    }
  }

  grad_env <- run_backward_pass(graph, backwards, required_env, grad_env)
  lapply(targets, function(target) {
    grad_env[[target]] %||% zeros(target$aval$dtype, shape(target$aval))
  })
}

# Splits the reverse of a sub-graph into two closed graphs, so that its
# backward pass does not have to rerun the forward one, as JAX's partial
# evaluation of a `cond` does:
#   - fwd: `graph`'s forward pass, taking the same inputs, returning its
#     outputs and then the residuals -- the values computed by the forward
#     pass that `bwd` reads.
#   - bwd: taking a cotangent per output of `graph`, then `graph`'s inputs,
#     then the residuals, and returning the cotangents of
#     `graph$inputs[needed]`.
#   - residuals: the avals of the residuals.
split_vjp <- function(graph, needed) {
  targets <- graph$inputs[needed]
  required_env <- requirements_from(graph, targets)

  desc_fwd <- local_descriptor()
  rebuilt <- rebuild_forward_into(graph, desc_fwd, required_env = required_env)
  bwd <- trace_pull_back(graph, rebuilt$backwards, required_env, targets)

  # What the backward pass read of the forward one, it captured. An input of
  # `graph` it reads as an input of its own; the rest are the residuals. An
  # array it closed over stays a constant.
  map <- hashtab()
  inputs <- lapply(graph$inputs, function(g) {
    map[[g]] <- GraphValue(aval = g$aval)
  })
  captured <- Filter(\(g) !is_concrete_array(g$aval), bwd$constants)
  residuals <- Filter(\(g) is.null(map[[g]]), captured)
  for (g in residuals) {
    map[[g]] <- GraphValue(aval = g$aval)
    # A no-op for a value of the forward pass; one from further out is
    # captured by it like any other.
    get_box_or_register_const(desc_fwd, g)
  }
  substitute_gnodes(bwd, map)
  bwd$inputs <- c(bwd$inputs, inputs, lapply(residuals, \(g) map[[g]]))
  bwd$constants <- Filter(\(g) is_concrete_array(g$aval), bwd$constants)

  outputs <- lapply(graph$outputs, \(g) if (is_graph_literal(g)) g else rebuilt$trans[[g]] %||% g)
  desc_fwd$outputs <- c(outputs, residuals)
  list(
    fwd = descriptor_to_graph(desc_fwd),
    bwd = bwd,
    residuals = lapply(residuals, \(g) g$aval)
  )
}

# The backward pass of `split_vjp()`, traced into a graph of its own
# whose inputs are the cotangents of `graph`'s outputs. The values of the
# forward pass the reverse rules read become its constants.
trace_pull_back <- function(graph, backwards, required_env, targets) {
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
# derivative, and the reverse rules give such values a zero -- so an RNG state
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
# it to rebuild a branch of `prim_if()`, `graph_apply()` to rerun a closed
# graph, `graph_value_and_grad()` to replay what `gradient()` traced into the
# descriptor it is called in.
#
# Returns:
#   - trans: hashtab(original gval -> new gval) for every replaced output,
#     every input `inputs` binds, and every constant whose array `desc` already
#     holds under a GraphValue of its own. Other constants and literals are not
#     added (they fall through).
#   - backwards: ordered list that needs to be traversed in reverse for the
#     backward pass.
#
# With `required_env`, a call none of whose operands requires a gradient keeps
# its plain forward even where its rule has a replacement: the backward pass
# skips it, so what the replacement keeps for it -- an if's residuals -- would
# be computed for nothing.
rebuild_forward_into <- function(graph, desc, inputs = NULL, required_env = NULL) {
  # consts and inputs keep their identity, only GraphValues created by GraphStatements
  # get new identifier -- unless `inputs` binds the inputs to boxes of `desc`,
  # in which case the graph is replayed as a function of them.
  if (is.null(inputs)) {
    register_inputs(desc, graph$inputs)
  } else {
    # A box of an enclosing trace -- an operand of the call the sub-graph
    # belongs to -- is captured by `desc` like any other value from outside.
    inputs <- lapply(inputs, maybe_box_arrayish, desc = desc)
  }

  # Existing GraphValues are reused where possible to minimize cloning.
  # If an alternative forward pass is called, this possibly invalidates
  # inputs to subsequent GraphStatements, so we have to look up the translated gnode every time
  # (we could actually delay this lookup until
  trans <- hashtab()
  translate_gnode <- function(g) {
    if (is_graph_literal(g)) {
      return(g)
    }
    trans[[g]] %||% g
  }
  for (i in seq_along(inputs)) {
    trans[[graph$inputs[[i]]]] <- inputs[[i]]$gnode
  }
  # An array `desc` already holds is read from the GraphValue it has for it.
  for (const in graph$constants) {
    box <- get_box_or_register_const(desc, const)
    if (!identical(box$gnode, const)) {
      trans[[const]] <- box$gnode
    }
  }
  # Get/create the box for a translated gval. Literals reach this branch
  # only when used as a call input; mint a box on demand (GraphBox has value semantics)
  box_for <- function(g) {
    new_g <- translate_gnode(g)
    box <- desc$gval_to_box[[new_g]]
    if (is.null(box)) {
      box <- GraphBox(new_g, desc)
      desc$gval_to_box[[new_g]] <- box
    }
    box
  }

  # We store the backward rules in a list so we can just traverse it backwards afterwards

  # backwards be longer than graph$statements, but never shorter
  backwards <- vector("list", length(graph$statements))

  for (i in seq_along(graph$statements)) {
    call <- graph$statements[[i]]
    rule <- call$primitive[["reverse"]]

    needs_grad <- is.null(required_env) ||
      any(vapply(call$inputs, \(x) !is_graph_literal(x) && isTRUE(required_env[[x]]), logical(1L)))

    if (is.null(rule) || is.null(rule$forward) || !needs_grad) {
      # No rule, a backward-only rule, or a call the backward pass skips: the
      # forward computation is unchanged,
      # so we can reuse the original output gvals directly. Only mint a new
      # GraphStatement if an upstream alt-forward replaced one of our inputs;
      # otherwise share the original statement object verbatim.
      new_inputs <- lapply(call$inputs, translate_gnode)
      new_call <- GraphStatement(call$primitive, new_inputs, call$params, call$outputs)

      desc$statements$add(new_call)
      register_gvals(desc, call$outputs)

      # Without a backward rule `backwards[[i]]` stays NULL. `run_backward_pass`
      # treats that as "skip if no input requires grad, otherwise abort":
      # primitives like `prim_fill` whose inputs are all static parameters
      # never reach the abort branch, so they don't need a reverse rule, and
      # neither does a skipped call whose rule only has a replacement forward.
      if (!is.null(rule$backward)) {
        backwards[[i]] <- list(
          fn = rule$backward,
          inputs = lapply(new_call$inputs, GraphBox, desc = desc),
          outputs = lapply(call$outputs, GraphBox, desc = desc),
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

  list(trans = trans, backwards = backwards)
}

# Replays `graph` into the current descriptor as a function of `inputs`
# (boxes, one per graph input) and returns the boxes of its outputs, e.g. a
# branch of `prim_if()`'s forward pass that `split_vjp()` prepared. No backward
# pass follows, so every call keeps its plain forward: an empty `required_env`
# says that nothing requires a gradient.
graph_apply <- function(graph, inputs) {
  desc <- current_descriptor()
  rebuilt <- rebuild_forward_into(graph, desc, inputs, required_env = hashtab())
  lapply(graph$outputs, function(out) {
    g <- if (is_graph_literal(out)) out else rebuilt$trans[[out]] %||% out
    desc$gval_to_box[[g]] %||% GraphBox(g, desc)
  })
}

# Walk statements in reverse, invoking each statement's backward to accumulate
# gradients keyed by the *original* graph's gvals.
run_backward_pass <- function(graph, backwards, required_env, grad_env) {
  add_or_init <- function(grad1, grad2) {
    if (is.null(grad1)) {
      return(grad2)
    }
    if (is.null(grad2)) {
      cli_abort("Internal error: a reverse rule returned no gradient for an input that requires one.")
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
        "No reverse rule for primitive {.field {call$primitive$name}}.",
        i = "Cannot compute gradient through this primitive."
      ))
    }

    # An input that requires no gradient gets none: the rule returns `NULL`
    # for it, and nothing is accumulated.
    input_grads <- bwd$fn(bwd$inputs, bwd$outputs, output_grads, bwd$params, input_required)
    for (j in which(input_required)) {
      input_gval <- call$inputs[[j]]
      grad_env[[input_gval]] <- add_or_init(grad_env[[input_gval]], input_grads[[j]])
    }
  }

  grad_env
}

# For each input the user asked to differentiate w.r.t., return the
# accumulated gradient gnode -- or a zero of matching shape if the input
# never reached the loss.
collect_input_grads <- function(graph, desc, grad_env, requires_grad) {
  input_grads <- list()
  for (i in seq_along(graph$inputs)) {
    if (!requires_grad[[i]]) {
      next
    }
    input <- graph$inputs[[i]]
    x <- grad_env[[input]] %||%
      {
        const <- get_box_or_register_const(
          desc,
          nv_scalar(0L, dtype = input$aval$dtype)
        )
        nv_broadcast_to(const, shape(input$aval))
      }
    input_grads <- c(input_grads, list(x$gnode))
  }
  input_grads
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
#' @seealso `r roxy_article("autodiff")`, [`transform_gradient()`] for the
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
  wrt <- resolve_arg_names(f, wrt, "wrt")
  if (!is.null(wrt) && !all(wrt %in% formalArgs(f))) {
    cli_abort("wrt must be a subset of the formal arguments of f")
  }
  f_gradient <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    prep <- prepare_gradient_args(args, wrt)

    if (is.null(current_descriptor(silent = TRUE))) {
      cli_abort(c(
        "{.fn gradient} can only be called inside a {.fn jit}-compiled function.",
        i = "Wrap the result of {.fn gradient} in {.fn jit}, e.g. {.code jit(gradient(f))}."
      ))
    }
    fwd_graph <- trace_fn(f, args_flat = prep$args_flat, in_tree = prep$in_tree)
    res <- graph_value_and_grad(fwd_graph, wrt, gradient_operands(fwd_graph, prep$args_flat))
    unflatten(gradient_out_tree(fwd_graph, wrt), res$grad)
  }
  formals(f_gradient) <- formals2(f)
  return(f_gradient)
}

#' @rdname gradient
#' @export
value_and_gradient <- function(f, wrt = NULL) {
  assert_function(f)
  wrt <- resolve_arg_names(f, wrt, "wrt")
  if (!is.null(wrt) && !all(wrt %in% formalArgs(f))) {
    cli_abort("wrt must be a subset of the formal arguments of f")
  }
  f_value_and_grad <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    prep <- prepare_gradient_args(args, wrt)

    if (is.null(current_descriptor(silent = TRUE))) {
      cli_abort(c(
        "{.fn value_and_gradient} can only be called inside a {.fn jit}-compiled function.",
        i = "Wrap the result of {.fn value_and_gradient} in {.fn jit}, e.g. {.code jit(value_and_gradient(f))}."
      ))
    }
    fwd_graph <- trace_fn(f, args_flat = prep$args_flat, in_tree = prep$in_tree)
    res <- graph_value_and_grad(fwd_graph, wrt, gradient_operands(fwd_graph, prep$args_flat))
    list(
      value = unflatten(fwd_graph$out_tree, res$value),
      grad = unflatten(gradient_out_tree(fwd_graph, wrt), res$grad)
    )
  }
  formals(f_value_and_grad) <- formals2(f)
  f_value_and_grad
}
