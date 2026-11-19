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

# A random state among the arguments is a leaf, which the traced function
# receives as it is, so that its draws continue those of the enclosing trace.
prepare_gradient_args <- function(args, wrt) {
  args_flat <- flatten(args, is_leaf = is_rng_state)
  in_tree <- build_tree(args, is_leaf = is_rng_state)
  is_wrt_flat <- if (!is.null(wrt)) {
    pjrt::tree_leaf_mask(in_tree, wrt)
  } else {
    rep(TRUE, length(args_flat))
  }
  check_wrt_arrayish(args_flat, is_wrt_flat)
  list(args_flat = args_flat, in_tree = in_tree, is_wrt_flat = is_wrt_flat)
}

#' @title Reverse Rule
#' @description
#' Construct a reverse-mode autodiff rule for a primitive. Provide exactly one
#' of `backward` and `forward`.
#'
#' Pass `backward` when the primitive's forward statement can run unmodified, which
#' covers most use cases. It has the signature
#' `function(inputs, outputs, grads, params, required)` and returns a `list`
#' with one entry per input: that input's gradient, a [`SymbolicZero()`] where
#' that gradient is zero, or `NULL` where `required` says it is not needed.
#'
#' Zero gradients are symbolic: a [`SymbolicZero()`] stands for zeros that
#' nothing computes. A call none of whose outputs received a gradient is
#' skipped, so `grads` holds a `SymbolicZero` only for some of the outputs of a
#' primitive with several.
#'
#' Pass `forward` when a slightly different forward pass enables a more
#' efficient backward pass. It has the signature
#' `function(inputs, params, required)`, where `required` says which inputs
#' need a gradient, and returns `list(outputs = , backward = )`: the forward
#' results and a closure with the signature of `backward` above, which can use
#' intermediate values of the forward pass via lexical scoping.
#'
#' A rule with a `forward` pass can also be the `rule` of
#' [prim_custom_gradient()].
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
#'
#' # the rule of prim_floor(), whose derivative is zero
#' rule_reverse(function(inputs, outputs, grads, params, required) {
#'   list(SymbolicZero(inputs[[1L]]))
#' })
#' @seealso [`graph_gradient()`]
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

#' @title Symbolic Zero
#' @description
#' The zero gradient of a value in a reverse rule (see [`rule_reverse()`]): it
#' stands for an array of zeros that nothing computes. A value whose gradient
#' is a symbolic zero adds nothing to the backward pass, so, unlike an array of
#' zeros, it cannot become a `NaN` where it meets an infinite derivative, such
#' as that of `exp()` at 1000.
#' @param x (arrayish | [`AbstractArray`] | `any`)\cr
#'   For `SymbolicZero()`, the value whose gradient is zero, or its abstract
#'   array. For `is_symbolic_zero()`, the object to test.
#' @return `SymbolicZero()` returns a `SymbolicZero`: a list holding `aval`, the
#'   [`AbstractArray`] of the zeros it stands for. `is_symbolic_zero()` returns
#'   a `logical(1)`.
#' @examples
#' z <- SymbolicZero(nv_aval("f32", c(2L, 3L)))
#' z
#' is_symbolic_zero(z)
#' @export
SymbolicZero <- function(x) {
  structure(list(aval = to_abstract(x, pure = TRUE)), class = "SymbolicZero")
}

#' @rdname SymbolicZero
#' @export
is_symbolic_zero <- function(x) {
  inherits(x, "SymbolicZero")
}

#' @export
format.SymbolicZero <- function(x, ...) {
  sprintf("SymbolicZero(%s)", format(x$aval))
}

#' @export
print.SymbolicZero <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

# The array of zeros `ct` stands for if it is a symbolic zero, else `ct`.
instantiate_cotangent <- function(ct) {
  if (is_symbolic_zero(ct)) zeros(ct$aval$dtype, shape(ct$aval)) else ct
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
  seed <- get_box_or_register_const(desc, nv_scalar(1L, dtype = out$aval$dtype))
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
# a function that takes a cotangent per output, a `SymbolicZero` for a zero
# one, records the backward pass into the descriptor current when it is
# called, and returns the cotangents of the inputs `reqs$requires_grad`
# selects -- zeros where no cotangent reached the input. It can be called more
# than once.
graph_pullback <- function(graph, reqs, inputs = NULL) {
  desc <- current_descriptor()
  rebuilt <- rebuild_forward_into(graph, desc, inputs, reqs$required_env)

  box_of <- function(g) desc$gval_to_box[[g]] %||% GraphBox(g, desc)
  replayed <- function(g) box_of(if (is_graph_literal(g)) g else rebuilt$trans[[g]] %||% g)
  targets <- graph$inputs[reqs$requires_grad]
  list(
    value = lapply(graph$outputs, replayed),
    pullback = function(out_cts) {
      cts <- pull_back(graph, rebuilt$backwards, reqs$required_env, targets, out_cts)
      lapply(cts, instantiate_cotangent)
    }
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
# replayed: seeds `graph`'s outputs with `out_cts`, runs the reverse rules
# `backwards` in the current descriptor, and returns the cotangent of each
# value in `targets` -- a `SymbolicZero` where no cotangent reached the target.
pull_back <- function(graph, backwards, required_env, targets, out_cts) {
  grad_env <- seed_cotangents(graph, required_env, out_cts)
  grad_env <- run_backward_pass(graph, backwards, required_env, grad_env)
  lapply(targets, \(target) grad_env[[target]] %||% SymbolicZero(target$aval))
}

# The cotangents `out_cts`, one per output of `graph`, as the hashtab(gval ->
# box) `run_backward_pass()` starts from. Only an output that requires a
# gradient and whose cotangent is no symbolic zero is seeded, so that no value
# the reverse rules leave out gains a cotangent.
seed_cotangents <- function(graph, required_env, out_cts) {
  grad_env <- hashtab()
  for (i in seq_along(graph$outputs)) {
    out <- graph$outputs[[i]]
    ct <- out_cts[[i]]
    if (is_symbolic_zero(ct) || is_graph_literal(out) || !isTRUE(required_env[[out]])) {
      next
    }
    # The same value can be returned more than once, e.g. a branch
    # `\() list(s, s)`, or a branch returning one of its inputs in two places.
    # Its cotangents accumulate, as for any other reuse.
    grad_env[[out]] <- if (is.null(grad_env[[out]])) ct else prim_add(grad_env[[out]], ct)
  }
  grad_env
}

# Splits the reverse of a sub-graph -- a branch of `prim_if()`, the body of
# `prim_scan()` -- into closed graphs, so that its backward pass does not have
# to rerun the forward one, as JAX's partial evaluation of a `cond` or a `scan`
# does:
#   - fwd: `graph`'s forward pass, taking the same inputs, returning its
#     outputs and then the residuals -- the values computed by the forward
#     pass that the backward pass reads.
#   - bwd_for: a function of `seeded`, which says for each output of `graph`
#     whether it receives a cotangent, returning the backward pass for that
#     case: `graph`, a graph taking a cotangent per output of `graph` (one that
#     is not seeded is not read, and can be passed as `NULL`), then `graph`'s
#     inputs, then the residuals, and returning the cotangents of
#     `graph$inputs[needed]`; and `zero`, which of those are zeros because no
#     cotangent reached the input. An output left out of the backward pass
#     leaves out what only it needed: a zero cotangent is not pushed through
#     the reverse rules, where it could meet an infinity and turn into a NaN.
#   - residual_avals: the avals of the residuals.
# The residuals are those of the backward pass where every output receives a
# cotangent, which reads what any other case does.
split_vjp <- function(graph, needed) {
  targets <- graph$inputs[needed]
  required_env <- requirements_from(graph, targets)

  desc_fwd <- local_descriptor()
  rebuilt <- rebuild_forward_into(graph, desc_fwd, required_env = required_env)
  pull <- function(seeded) {
    pull_back_graph(graph, rebuilt$backwards, required_env, targets, seeded)
  }
  all_seeded <- rep(TRUE, length(graph$outputs))
  full <- pull(all_seeded)

  # What the backward pass read of the forward one, it captured. An input of
  # `graph` it reads as an input of its own; the rest are the residuals. An
  # array it closed over stays a constant.
  is_input <- hashtab()
  for (g in graph$inputs) {
    is_input[[g]] <- TRUE
  }
  captured <- Filter(\(g) !is_concrete_array(g$aval), full$graph$constants)
  residuals <- Filter(\(g) is.null(is_input[[g]]), captured)
  for (g in residuals) {
    # The branch is pure, so what its backward pass reads was computed by its
    # forward pass.
    if (is.null(desc_fwd$gval_to_box[[g]])) {
      cli_abort("Internal error: the backward pass of a sub-graph read a value from outside it.")
    }
  }

  close_bwd <- function(pulled) {
    bwd <- pulled$graph
    map <- hashtab()
    fresh <- function(g) map[[g]] <- GraphValue(aval = g$aval, label = g$label)
    inputs <- c(lapply(graph$inputs, fresh), lapply(residuals, fresh))
    reads_unknown <- vapply(bwd$constants, \(g) !is_concrete_array(g$aval) && is.null(map[[g]]), logical(1L))
    if (any(reads_unknown)) {
      cli_abort("Internal error: the backward pass of a sub-graph read a value that is no residual.")
    }
    substitute_gnodes(bwd, map)
    bwd$inputs <- c(bwd$inputs, inputs)
    bwd$constants <- Filter(\(g) is_concrete_array(g$aval), bwd$constants)
    list(graph = bwd, zero = pulled$zero)
  }
  cache <- new.env(parent = emptyenv())
  key <- function(seeded) paste(as.integer(seeded), collapse = "")
  cache[[key(all_seeded)]] <- close_bwd(full)
  bwd_for <- function(seeded) {
    cache[[key(seeded)]] %||% (cache[[key(seeded)]] <- close_bwd(pull(seeded)))
  }

  outputs <- lapply(graph$outputs, \(g) if (is_graph_literal(g)) g else rebuilt$trans[[g]] %||% g)
  desc_fwd$outputs <- c(outputs, residuals)
  list(
    fwd = descriptor_to_graph(desc_fwd),
    bwd_for = bwd_for,
    residual_avals = lapply(residuals, \(g) g$aval)
  )
}

# The backward pass of a sub-graph whose forward `rebuild_forward_into()`
# replayed, for `split_vjp()`: traced into a graph of its own whose inputs are
# the cotangents of `graph`'s outputs, of which those `seeded` says are seeded,
# and whose outputs are the cotangents of `targets` -- zeros where no cotangent
# reached the target, which `zero` says. The values of the forward pass the
# reverse rules read become its constants.
pull_back_graph <- function(graph, backwards, required_env, targets, seeded) {
  desc <- local_descriptor()
  out_cts <- Map(
    function(out, seed) {
      aval <- AbstractArray(dtype = out$aval$dtype, shape = out$aval$shape)
      ct <- register_input(desc, GraphValue(aval = aval))
      if (seed) ct else SymbolicZero(aval)
    },
    graph$outputs,
    seeded
  )
  cts <- pull_back(graph, backwards, required_env, targets, out_cts)
  desc$outputs <- lapply(cts, \(ct) maybe_box_arrayish(instantiate_cotangent(ct), desc)$gnode)
  list(graph = descriptor_to_graph(desc), zero = vapply(cts, is_symbolic_zero, logical(1L)))
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
# derivative, and the reverse rules give such values a zero -- so a random state
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
# `prim_scan()` -- `graph_apply()` to rerun a closed graph,
# `graph_pullback()` to replay what `pullback()` traced into the descriptor
# it is called in.
#
# Returns:
#   - trans: hashtab(original gval -> new gval) for every replaced output,
#     every input `inputs` binds, and every constant whose array `desc` already
#     holds under a GraphValue of its own. Other constants and literals are not
#     added (they fall through).
#   - backwards: ordered list that needs to be traversed in reverse for the
#     backward pass.
#   - translate: function(original gval) -> the gval `desc` holds for it.
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
  } else {
    # A box of an enclosing trace -- an operand of the call the sub-graph
    # belongs to -- is captured by `desc` like any other value from outside.
    inputs <- lapply(inputs, \(x) if (!is.null(x)) maybe_box_arrayish(x, desc = desc))
  }

  # Existing GraphValues are reused where possible to minimize cloning.
  # An alternative forward pass replaces the outputs of its statement, which
  # later statements may read, so every input is looked up in `trans`.
  trans <- hashtab()
  translate_gnode <- function(g) {
    if (is_graph_literal(g)) {
      return(g)
    }
    if (isTRUE(unbound[[g]])) {
      cli_abort("Internal error: a replayed graph read an input that was passed as `NULL`.")
    }
    trans[[g]] %||% g
  }
  # The graph is pure, so its inputs are GraphValues of its own: bind each to
  # the value passed for it. One passed as `NULL` must not be read: it would
  # fall through to a GraphValue `desc` never defined.
  unbound <- hashtab()
  for (i in seq_along(inputs)) {
    if (is.null(inputs[[i]])) {
      unbound[[graph$inputs[[i]]]] <- TRUE
    } else {
      trans[[graph$inputs[[i]]]] <- inputs[[i]]$gnode
    }
  }
  # An array `desc` already holds is read from the GraphValue it has for it.
  const_boxes <- register_consts(desc, graph$constants)
  for (i in seq_along(const_boxes)) {
    if (!identical(const_boxes[[i]]$gnode, graph$constants[[i]])) {
      trans[[graph$constants[[i]]]] <- const_boxes[[i]]$gnode
    }
  }
  # The box for a translated gval, minted on demand (a literal has none until
  # it is used as a call input).
  box_for <- function(g) register_gval(desc, translate_gnode(g))

  # One entry per statement, which run_backward_pass() traverses in reverse.
  backwards <- vector("list", length(graph$statements))

  for (i in seq_along(graph$statements)) {
    call <- graph$statements[[i]]
    rule <- call$primitive[["reverse"]]

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
      new_call <- GraphStatement(call$primitive, new_inputs, call$params, new_outputs)

      desc$statements$add(new_call)
      register_gvals(desc, new_outputs)

      # Without a backward rule `backwards[[i]]` stays NULL. `run_backward_pass`
      # treats that as "skip if no input requires grad, otherwise abort":
      # primitives like `prim_fill` whose inputs are all static parameters
      # never reach the abort branch, so they don't need a reverse rule, and
      # neither does a skipped call whose rule only has a replacement forward.
      if (!is.null(rule$backward)) {
        backwards[[i]] <- list(
          fn = rule$backward,
          inputs = lapply(new_call$inputs, GraphBox, desc = desc),
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

  list(trans = trans, backwards = backwards, translate = translate_gnode)
}

# Replays `graph` into the current descriptor as a function of `inputs`
# (boxes, one per graph input, or `NULL` for an input the graph does not read)
# and returns the boxes of its outputs, e.g. a branch of `prim_if()`'s forward
# pass that `split_vjp()` prepared. No backward pass follows, so every call
# keeps its plain forward: an empty `required_env` says that nothing requires a
# gradient. Every call gets new output gvals, so that a graph applied more than
# once in one trace never defines the same value twice.
graph_apply <- function(graph, inputs) {
  desc <- current_descriptor()
  rebuilt <- rebuild_forward_into(graph, desc, inputs, required_env = hashtab(), fresh = TRUE)
  lapply(graph$outputs, function(out) {
    g <- rebuilt$translate(out)
    desc$gval_to_box[[g]] %||% GraphBox(g, desc)
  })
}

# Walk statements in reverse, invoking each statement's backward to accumulate
# cotangents keyed by the *original* graph's gvals, in `grad_env`.
#
# The cotangents are symbolic zeros: a value without an entry in `grad_env` has
# a zero cotangent that nothing computes. A call none of whose outputs has a
# cotangent contributes none and is skipped -- its output fed no output of the
# graph, or only something whose derivative is zero, such as a comparison, a
# loop that only counts, or `prim_floor()`. Differentiating it anyway would
# multiply zeros into its derivative, which is a NaN wherever that derivative
# is infinite, e.g. that of `exp()` at 1000. A rule receives a `SymbolicZero`
# for an output without a cotangent -- only a call with more than one output
# can have one -- and returns one for an input whose cotangent is zero.
run_backward_pass <- function(graph, backwards, required_env, grad_env) {
  for (i in rev(seq_along(graph$statements))) {
    call <- graph$statements[[i]]
    output_grads <- lapply(call$outputs, \(output) grad_env[[output]])
    received <- !vapply(output_grads, is.null, logical(1L))
    if (!any(received)) {
      next
    }
    output_grads[!received] <- lapply(call$outputs[!received], \(output) SymbolicZero(output$aval))
    input_required <- vapply(
      call$inputs,
      function(x) required_env[[x]] %||% FALSE,
      logical(1L)
    )
    if (!any(input_required)) {
      next
    }

    bwd <- backwards[[i]]
    if (is.null(bwd)) {
      cli_abort(c(
        "No reverse rule for primitive {.field {call$primitive$name}}.",
        i = "Cannot compute gradient through this primitive."
      ))
    }

    # An input that requires no gradient gets none: the rule returns `NULL`
    # for it, and nothing is accumulated, as for a symbolic zero.
    input_grads <- bwd$fn(bwd$inputs, bwd$outputs, output_grads, bwd$params, input_required)
    for (j in which(input_required)) {
      ct <- input_grads[[j]]
      if (is.null(ct)) {
        cli_abort(c(
          "The reverse rule of primitive {.field {call$primitive$name}} returned `NULL` for input {j}, which requires a gradient.", # nolint
          i = "Return {.code SymbolicZero(x)} for a gradient that is zero."
        ))
      }
      if (is_symbolic_zero(ct)) {
        next
      }
      input_gval <- call$inputs[[j]]
      grad_env[[input_gval]] <- if (is.null(grad_env[[input_gval]])) ct else prim_add(grad_env[[input_gval]], ct)
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
  wrt <- resolve_wrt(f, wrt)
  f_gradient <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    trace_value_and_grad(f, args, wrt, "gradient")$grad
  }
  formals(f_gradient) <- formals2(f)
  f_gradient
}

#' @rdname gradient
#' @export
value_and_gradient <- function(f, wrt = NULL) {
  assert_function(f)
  wrt <- resolve_wrt(f, wrt)
  f_value_and_grad <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    trace_value_and_grad(f, args, wrt, "value_and_gradient")
  }
  formals(f_value_and_grad) <- formals2(f)
  f_value_and_grad
}

# The `wrt` of gradient() / value_and_gradient() as names of `f`'s formals.
resolve_wrt <- function(f, wrt, call = rlang::caller_env()) {
  wrt <- resolve_arg_names(f, wrt, "wrt")
  if (!is.null(wrt) && !all(wrt %in% formalArgs(f))) {
    cli_abort("wrt must be a subset of the formal arguments of f", call = call)
  }
  wrt
}

# gradient() / value_and_gradient() (named by `fn`) at the call's arguments:
# the pullback of `f` with respect to the arguments `wrt` names, applied to a 1
# for its value.
trace_value_and_grad <- function(f, args, wrt, fn, call = rlang::caller_env()) {
  prep <- prepare_gradient_args(args, wrt)
  if (is.null(current_descriptor(silent = TRUE))) {
    cli_abort(
      c(
        "{.fn {fn}} can only be called inside a {.fn jit}-compiled function.",
        i = "Wrap the result of {.fn {fn}} in {.fn jit}, e.g. {.code jit({fn}(f))}."
      ),
      call = call
    )
  }
  out <- trace_pullback(f, prep, fn)
  value <- validate_gradient_output(lapply(flatten(out$value), \(box) box$gnode))
  seed <- get_box_or_register_const(current_descriptor(), nv_scalar(1L, dtype = value$aval$dtype))
  grad <- out$pullback(unflatten(build_tree(out$value), list(seed)))
  list(value = out$value, grad = if (is.null(wrt)) grad else grad[names(grad) %in% wrt])
}

#' @title Pullbacks
#' @description
#' Evaluate `.f` at the arguments `...` and return its pullback: the function
#' that takes a cotangent of the result and returns the corresponding cotangent
#' back to the arguments, by reverse-mode automatic differentiation. That is
#' the cotangent times the Jacobian of `.f`, the vector-Jacobian product.
#' Where [`gradient()`] differentiates a function that returns a single float
#' scalar, a pullback differentiates one with any result: it gives the
#' gradient of the sum of the result's elements, each weighted by its
#' cotangent.
#'
#' * `pullback()` returns only the pullback.
#' * `value_and_pullback()` returns both the result of `.f` and its pullback.
#'
#' `.f` runs once, when `pullback()` is called. The pullback can be called more
#' than once, e.g. with every unit vector to build a Jacobian row by row.
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
  trace_pullback(.f, pullback_args(.f, list(...)), "pullback")$pullback
}

#' @rdname pullback
#' @export
value_and_pullback <- function(.f, ...) {
  assert_function(.f)
  trace_pullback(.f, pullback_args(.f, list(...)), "value_and_pullback")
}

# The arguments `args` of `pullback()`, matched to the formals of `f`, as
# `prepare_gradient_args()` returns them: every float array among them is
# differentiated.
pullback_args <- function(f, args) {
  args <- match_args_to_formals(f, args)
  args_flat <- flatten(args, is_leaf = is_rng_state)
  list(
    args_flat = args_flat,
    in_tree = build_tree(args, is_leaf = is_rng_state),
    is_wrt_flat = vapply(args_flat, is_float_array, logical(1L))
  )
}

# The pullback of `f` for `pullback()`, `value_and_pullback()` or
# `gradient()`, named by `fn`, at the arguments `prep` holds (see
# `prepare_gradient_args()`): `f` traced into a graph whose forward pass is
# replayed into the enclosing trace, and the function that records its
# backward pass there. It differentiates the leaves `prep$is_wrt_flat`
# selects, and returns `NULL` for the others.
trace_pullback <- function(f, prep, fn, call = rlang::caller_env()) {
  enclosing <- current_descriptor(silent = TRUE)
  if (is.null(enclosing)) {
    cli_abort(
      c(
        "{.fn {fn}} can only be called inside a {.fn jit}-compiled function.",
        i = "Call it in the function you pass to {.fn jit}."
      ),
      call = call
    )
  }
  # The arguments are expressions of the enclosing trace, which must be
  # recorded there before `f`'s trace opens.
  force(prep)
  # A side effect `f` uses closes over the enclosing trace's value, and its
  # uses continue the enclosing trace's.
  desc <- local_descriptor(side_effects_mode = "capture")
  graph <- trace_fn(f, args_flat = prep$args_flat, in_tree = prep$in_tree, desc = desc)
  # A leaf that is no array is static, and no input of the graph.
  requires_grad <- prep$is_wrt_flat[!graph$is_static_flat]
  reqs <- list(
    required_env = requirements_from(graph, graph$inputs[requires_grad]),
    requires_grad = requires_grad
  )
  res <- graph_pullback(graph, reqs, gradient_operands(graph, prep$args_flat))
  desc_end_capture(enclosing, list(graph))

  value <- unflatten(graph$out_tree, res$value)
  pull <- function(cotangent) {
    if (!descriptor_within(current_descriptor(silent = TRUE), enclosing)) {
      cli_abort(c(
        "A pullback can only be called in the function {.fn {fn}} was called in.",
        i = "Or in a function traced inside it, such as a branch of {.fn nv_if}."
      ))
    }
    cts <- flatten_cotangent(value, cotangent, graph$out_tree, call = rlang::current_env())
    pulled <- vector("list", length(prep$args_flat))
    pulled[prep$is_wrt_flat] <- res$pullback(cts)
    unflatten(prep$in_tree, pulled)
  }
  list(value = value, pullback = pull)
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
# type and shape, or a `SymbolicZero` where the cotangent is `NULL`. Errors
# name `call`.
flatten_cotangent <- function(value, cotangent, out_tree, call) {
  Map(
    function(x, ct, i) {
      if (is.null(ct)) {
        return(SymbolicZero(x))
      }
      path <- pjrt::tree_path(out_tree, i)
      at <- if (nzchar(path)) " at {.code {path}}" else ""
      dt <- dtype(x)
      if (!is_dtype_float(dt)) {
        cli_abort(
          c(
            paste0("The value of {.arg .f} has no cotangent", at, "."),
            x = "It is {.val {as.character(dt)}}, not a float.",
            i = "Pass {.code NULL} there."
          ),
          call = call
        )
      }
      if (has_no_dtype(ct)) {
        ct <- materialize_at(ct, dt)
      } else if (!is_arrayish(ct)) {
        cli_abort(
          c(
            paste0("The cotangent", at, " must be an array."),
            x = "Got {.cls {class(ct)}}."
          ),
          call = call
        )
      }
      ct <- maybe_box_arrayish(ct)
      if (dtype(ct) != dt || !identical(shape(ct), shape(x))) {
        cli_abort(
          c(
            paste0("The cotangent", at, " must have the data type and shape of the value."),
            x = "The value is {.val {format_aval_short(to_abstract(x))}}, the cotangent {.val {format_aval_short(to_abstract(ct))}}." # nolint
          ),
          call = call
        )
      }
      ct
    },
    flatten(value),
    cotangent_leaves(value, cotangent, call),
    seq_len(tree_size(out_tree))
  )
}

# The leaves of `cotangent` aligned with those of `value`, `NULL` for each leaf
# of a part of `value` whose cotangent is `NULL`. Errors name `call`.
cotangent_leaves <- function(value, cotangent, call) {
  if (is.null(cotangent)) {
    return(vector("list", length(flatten(value))))
  }
  if (build_tree(cotangent) == build_tree(value)) {
    return(flatten(cotangent))
  }
  same_node <- rlang::is_bare_list(value) &&
    rlang::is_bare_list(cotangent) &&
    length(cotangent) == length(value) &&
    identical(names(cotangent), names(value))
  if (!same_node) {
    cli_abort(
      c(
        "{.arg cotangent} must be structured like the value of {.arg .f}.",
        i = "{.code NULL} can stand in for any part of it."
      ),
      call = call
    )
  }
  do.call(c, unname(Map(cotangent_leaves, value, cotangent, list(call)))) %||% list()
}
