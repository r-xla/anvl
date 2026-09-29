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
#' efficient backward pass. It has the signature `function(inputs, params)`
#' and returns `list(outputs = , backward = )`: the forward results and a
#' closure with the signature of `backward` above, which can use intermediate
#' values of the forward pass via lexical scoping.
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
#' This is the building block used by [`gradient()`] and [`value_and_gradient()`]; prefer
#' those higher-level wrappers unless you need to operate on graphs directly.
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
  transform_gradient_impl(graph, wrt)$graph
}

# Internal worker. Returns list(graph, fwd_translation) where fwd_translation
# is a hashtab mapping each original forward gval (call output) to its cloned
# counterpart in `graph`. `value_and_gradient` uses it to translate the
# original forward outputs to gvals that exist in the gradient graph;
# `gradient()` deliberately discards it.
transform_gradient_impl <- function(graph, wrt) {
  out <- validate_gradient_output(graph$outputs)
  reqs <- compute_requirements(graph, wrt)

  # Phase 1 -- rebuild the forward into a fresh descriptor. For each statement
  # either clone it verbatim (default-reverse / no rule) or hand off to the
  # general-form rule so it can emit its own forward primitives.
  rebuilt <- rebuild_forward_pass(graph)
  desc <- rebuilt$desc

  # Phase 2 -- run backwards in reverse statement order.
  grad_env <- run_backward_pass(
    graph,
    desc,
    rebuilt$backwards,
    reqs$required_env,
    out
  )

  # Phase 3 -- collect gradients for the inputs we differentiate w.r.t.
  desc$outputs <- collect_input_grads(graph, desc, grad_env, reqs$requires_grad)
  desc$in_tree <- graph$in_tree
  desc$is_static_flat <- graph$is_static_flat
  desc$static_args_flat <- graph$static_args_flat
  desc$out_tree <- if (length(wrt)) {
    pjrt::tree_filter_by_names(graph$in_tree, wrt)
  } else {
    graph$in_tree
  }

  list(
    graph = descriptor_to_graph(desc),
    fwd_translation = rebuilt$trans
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
  # Forward propagate: a statement's outputs require grad iff any input does.
  # Literals are inlined constants and never require grad.
  for (call in graph$statements) {
    any_input_requires <- any(vapply(
      call$inputs,
      function(x) {
        if (is_graph_literal(x)) {
          return(FALSE)
        }
        required_env[[x]]
      },
      logical(1L)
    ))
    for (out_node in call$outputs) {
      required_env[[out_node]] <- any_input_requires
    }
  }

  list(required_env = required_env, requires_grad = requires_grad)
}

# Set up a fresh descriptor, seed it with `graph`'s inputs/constants, and
# rebuild the forward call by call. The descriptor is created via
# `local_descriptor(envir = envir)` so its lifetime is tied to the caller's
# frame -- it stays the current descriptor after this function returns, so
# subsequent phases (and any prim_* emits inside backward closures) land in
# it.
# Returns:
#   - desc: the fresh descriptor.
#   - trans: hashtab(original gval -> new gval) for every replaced output.
#     Inputs, constants, and literals are not added (they fall through).
#   - backwards: ordered list that needs to be traversed in reverse for the backward pass.
rebuild_forward_pass <- function(graph, envir = parent.frame()) {
  desc <- local_descriptor(envir = envir)

  # Inputs and constants keep their identity; only the outputs of statements
  # may be replaced.
  register_inputs(desc, graph$inputs)
  register_consts(desc, graph$constants)

  # Existing GraphValues are reused where possible to minimize cloning.
  # An alternative forward pass replaces the outputs of its statement, which
  # later statements may read, so every input is looked up in `trans`.
  trans <- hashtab()
  translate_gnode <- function(g) {
    if (is_graph_literal(g)) {
      return(g)
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
    rule <- call$primitive[["reverse"]]

    if (is.null(rule) || is.null(rule$forward)) {
      # No rule, or backward-only rule: the forward computation is unchanged,
      # so the original output gvals are reused directly. The statement is
      # rebuilt only to read its inputs through `trans`, in case an upstream
      # alternative forward replaced one of them.
      new_inputs <- lapply(call$inputs, translate_gnode)
      new_call <- GraphStatement(call$primitive, new_inputs, call$params, call$outputs)

      desc$statements$add(new_call)
      register_gvals(desc, call$outputs)

      # If `rule` is NULL `backwards[[i]]` stays NULL. `run_backward_pass`
      # treats that as "skip if no input requires grad, otherwise abort":
      # primitives like `prim_fill` whose inputs are all static parameters
      # never reach the abort branch, so they don't need a reverse rule.
      if (!is.null(rule)) {
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
      fwd_result <- rule$forward(input_boxes, call$params)
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

  list(desc = desc, trans = trans, backwards = backwards)
}

# Walk statements in reverse, invoking each statement's backward to accumulate
# gradients keyed by the *original* graph's gvals.
run_backward_pass <- function(graph, desc, backwards, required_env, out) {
  grad_env <- hashtab()
  grad_env[[out]] <- get_box_or_register_const(
    desc,
    nv_scalar(1L, dtype = out$aval$dtype)
  )

  add_or_init <- function(grad1, grad2) {
    if (is.null(grad1)) {
      return(grad2)
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
    if (!any(input_required)) {
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

    # input_grads[!input_required] is a list of NULLs and is silently
    # skipped by add_or_init below.
    input_grads <- bwd$fn(bwd$inputs, bwd$outputs, output_grads, bwd$params, input_required)
    for (j in seq_along(call$inputs)) {
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
  lapply(graph$inputs[requires_grad], function(input) {
    x <- grad_env[[input]] %||%
      {
        const <- get_box_or_register_const(
          desc,
          nv_scalar(0L, dtype = input$aval$dtype)
        )
        nv_broadcast_to(const, shape(input$aval))
      }
    x$gnode
  })
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
  wrt <- resolve_wrt(f, wrt)
  f_gradient <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    fwd <- trace_gradient_forward(f, args, wrt, "gradient")
    grad_graph <- transform_gradient(fwd$graph, wrt)
    # the enclosing descriptor is modified in place
    inline_graph_into_desc(fwd$parent_desc, grad_graph)
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
    fwd <- trace_gradient_forward(f, args, wrt, "value_and_gradient")
    fwd_graph <- fwd$graph
    res <- transform_gradient_impl(fwd_graph, wrt)
    grad_graph <- res$graph
    trans <- res$fwd_translation

    fwd_outputs <- lapply(fwd_graph$outputs, \(g) trans[[g]] %||% g)
    combined_graph <- grad_graph
    combined_graph$outputs <- c(fwd_outputs, grad_graph$outputs)

    combined_graph$out_tree <- pjrt::tree_concat(
      list(fwd_graph$out_tree, grad_graph$out_tree),
      names = c("value", "grad")
    )
    inline_graph_into_desc(fwd$parent_desc, combined_graph)
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

# The forward pass of gradient() / value_and_gradient() (named by `fn`): `f`
# traced at the call's arguments into a graph to be inlined into the enclosing
# trace, whose descriptor is returned alongside it.
trace_gradient_forward <- function(f, args, wrt, fn, call = rlang::caller_env()) {
  prep <- prepare_gradient_args(args, wrt)
  parent_desc <- current_descriptor(silent = TRUE)
  if (is.null(parent_desc)) {
    cli_abort(
      c(
        "{.fn {fn}} can only be called inside a {.fn jit}-compiled function.",
        i = "Wrap the result of {.fn {fn}} in {.fn jit}, e.g. {.code jit({fn}(f))}."
      ),
      call = call
    )
  }
  graph <- trace_fn(f, args_flat = prep$args_flat, in_tree = prep$in_tree, mode = "inline")
  list(graph = graph, parent_desc = parent_desc)
}
