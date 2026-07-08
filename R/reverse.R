check_wrt_arrayish <- function(args_flat, is_wrt_flat) {
  for (i in seq_along(args_flat)) {
    if (is_wrt_flat[[i]]) {
      if (!is_arrayish(args_flat[[i]])) {
        cli_abort(c(
          "Cannot compute gradient with respect to non-array argument.",
          x = "Got {.cls {class(args_flat[[i]])}}"
        ))
      }

      if (!inherits(dtype_abstract(args_flat[[i]]), "FloatType")) {
        cli_abort(c(
          "Can only compute gradient with respect to float arrays.",
          x = "Got {repr(dtype_abstract(args_flat[[i]]))}"
        ))
      }
    }
  }
}

prepare_gradient_args <- function(args, wrt) {
  args_flat <- flatten(args)
  if (!is.null(wrt)) {
    in_tree <- build_tree(mark_some(args, wrt))
    is_wrt_flat <- in_tree$marked
    in_tree$marked <- NULL
    class(in_tree) <- c("ListNode", "Node")
  } else {
    in_tree <- build_tree(args)
    is_wrt_flat <- rep(TRUE, length(args_flat))
  }
  check_wrt_arrayish(args_flat, is_wrt_flat)
  list(args_flat = args_flat, in_tree = in_tree)
}

#' @title Reverse Rule
#' @description
#' Construct a reverse-mode autodiff rule for a primitive.
#' The `backward` argument should be provided if the `forward` call for
#' the primitive should run un-modified.
#' This covers most use-cases.
#' The `backward` argument should have this signature:
#' `function(inputs, outputs, grads, params, required) -> list(input_grads)`.
#'
#' In some scenarios, it can be beneficial to perform a slightly different forward pass
#' to enable a more efficient backward pass.
#' In this case, pass the `forward` argument.
#' It should return a list containing the results from the forward pass, as well as
#' closure that has the same signature as the one above.
#' It can make use of intermediate values computed in the forward pass via lexical scoping.
#'
#' @param backward (`function`)\cr
#'   Backward hook for default case.
#' @param forward (`function`)\cr
#'   Alternative-forward hook that returns both primals and backward closure.
#' @return An `anvl_rule_reverse` object.
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

#' @title Transform a graph to its gradient
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
#' @param wrt (`character`)\cr
#'   Names of the graph inputs to differentiate with respect to.
#' @return An [`AnvlGraph`] whose outputs are the requested gradients.
#' @seealso [`gradient()`], [`value_and_gradient()`], [`rule_reverse()`]
#' @export
#' @examples
#' graph <- trace_fn(prim_mul, list(nv_aval("f32", c()), nv_aval("f32", c())))
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

  # Phase 1 -- rebuild the forward into a fresh descriptor. For each call
  # either clone it verbatim (default-reverse / no rule) or hand off to the
  # general-form rule so it can emit its own forward primitives.
  rebuilt <- rebuild_forward_pass(graph)
  desc <- rebuilt$desc

  # Phase 2 -- run backwards in reverse call order.
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
    filter_list_node(graph$in_tree, wrt)
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
  if (!(dt == as_dtype("f32") || dt == as_dtype("f64"))) {
    cli_abort(c(
      x = "gradient can only be computed for functions that return float scalar",
      i = "Got dtype={.field {repr(dt)}}"
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
  requires_grad_all <- flat_mask_from_names(graph$in_tree, wrt)
  # `in_tree` may include static (non-array) args not present in
  # `graph$inputs`; filter them out. `gradient()` already rejects static
  # `wrt` entries, so this drop is safe.
  is_static <- graph$is_static_flat
  if (!is.null(is_static) && any(requires_grad_all & is_static)) {
    flat_argnames <- rep(
      graph$in_tree$names,
      times = vapply(graph$in_tree$nodes, tree_size, integer(1L))
    )
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
  # Forward propagate: a call's outputs require grad iff any input does.
  # Literals are inlined constants and never require grad.
  for (call in graph$calls) {
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

  # consts and inputs keep their identity, only GraphValues created by PrimitiveCalls
  # get new identifier
  register_inputs(desc, graph$inputs)
  register_consts(desc, graph$constants)

  # Existing GraphValues are reused where possible to minimize cloning.
  # If an alternative forward pass is called, this possibly invalidates
  # inputs to subsequent PrimitiveCalls, so we have to look up the translated gnode every time
  # (we could actually delay this lookup until
  trans <- hashtab()
  translate_gnode <- function(g) {
    if (is_graph_literal(g)) {
      return(g)
    }
    trans[[g]] %||% g
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

  # backwards be longer than graph$calls, but never shorter
  backwards <- vector("list", length(graph$calls))

  for (i in seq_along(graph$calls)) {
    call <- graph$calls[[i]]
    rule <- call$primitive[["reverse"]]

    if (is.null(rule) || is.null(rule$forward)) {
      # No rule, or backward-only rule: the forward computation is unchanged,
      # so we can reuse the original output gvals directly. Only mint a new
      # PrimitiveCall if an upstream alt-forward replaced one of our inputs;
      # otherwise share the original call object verbatim.
      new_inputs <- lapply(call$inputs, translate_gnode)
      new_call <- PrimitiveCall(call$primitive, new_inputs, call$params, call$outputs)

      desc$calls$add(new_call)
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
      # Here, new GraphValue outputs are generated and subsequent PrimitiveCalls that
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

# Walk calls in reverse, invoking each call's backward to accumulate
# gradients keyed by the *original* graph's gvals.
run_backward_pass <- function(graph, desc, backwards, required_env, out) {
  grad_env <- hashtab()
  grad_env[[out]] <- get_box_or_register_const(
    desc,
    nv_scalar(1L, dtype = out$aval$dtype, ambiguous = out$aval$ambiguous)
  )

  add_or_init <- function(grad1, grad2) {
    if (is.null(grad1)) {
      return(grad2)
    }
    prim_add(grad1, grad2)
  }

  for (i in rev(seq_along(graph$calls))) {
    call <- graph$calls[[i]]
    input_required <- vapply(
      call$inputs,
      function(x) required_env[[x]] %||% FALSE,
      logical(1L)
    )
    if (!any(input_required)) {
      next
    }

    output_grads <- lapply(call$outputs, \(output) {
      grad <- grad_env[[output]]
      if (is.null(grad)) {
        # Output grad may be NULL if there is dead code.
        prim_fill(0L, dtype = dtype(output), shape = shape(output))
      } else {
        grad
      }
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
  input_grads <- list()
  for (i in seq_along(graph$inputs)) {
    if (!requires_grad[[i]]) {
      next
    }
    input <- graph$inputs[[i]]
    grad <- grad_env[[input]]
    x <- if (is.null(grad)) {
      const <- get_box_or_register_const(
        desc,
        nv_scalar(0L, dtype = input$aval$dtype, ambiguous = input$aval$ambiguous)
      )
      nv_broadcast_to(const, shape(input$aval))
    } else {
      grad
    }
    # Match the input's ambiguity flag.
    x$gnode$aval$ambiguous <- input$aval$ambiguous
    input_grads <- c(input_grads, list(x$gnode))
  }
  input_grads
}


#' @title Gradient
#' @description
#' Returns a new function that computes the gradient of `f` via reverse-mode automatic
#' differentiation. `f` must return a single float scalar. The returned function has the
#' same signature as `f` and returns the gradients in the same structure as the inputs
#' (or the subset selected by `wrt`).
#' @param f (`function`)\cr
#'   Function to differentiate. Arguments can be arrayish ([`AnvlArray`]) or
#'   static (non-array) values. Must return a single scalar float array.
#' @param wrt (`character` | `integer` | `NULL`)\cr
#'   Names or positions of the arguments to compute the gradient with respect to.
#'   Only arrayish (float array) arguments can be included; static arguments
#'   must not appear in `wrt`.
#'   If `NULL` (the default), the gradient is computed with respect to all
#'   arguments (which must all be arrayish in that case).
#' @return `function`
#' @seealso [`value_and_gradient()`] to get both the output and gradients,
#'   [`transform_gradient()`] for the low-level graph transformation.
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' f <- function(x, y) sum(x * y)
#' g <- jit(gradient(f))
#' g(nv_array(c(1, 2), dtype = "f32"), nv_array(c(3, 4), dtype = "f32"))
#'
#' # Differentiate with respect to a single argument
#' g_x <- jit(gradient(f, wrt = "x"))
#' g_x(nv_array(c(1, 2), dtype = "f32"), nv_array(c(3, 4), dtype = "f32"))
#'
#' # Static (non-array) arguments are passed through but cannot be in wrt
#' f2 <- function(x, power) sum(x^power)
#' g2 <- jit(gradient(f2, wrt = "x"), static = "power")
#' g2(nv_array(c(1, 2, 3), dtype = "f32"), power = 2L)
gradient <- function(f, wrt = NULL) {
  wrt <- resolve_arg_names(f, wrt, "wrt")
  if (!is.null(wrt) && !all(wrt %in% formalArgs(f))) {
    cli_abort("wrt must be a subset of the formal arguments of f")
  }
  f_gradient <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    prep <- prepare_gradient_args(args, wrt)

    parent_desc <- .current_descriptor(silent = TRUE)
    if (is.null(parent_desc)) {
      cli_abort(c(
        "{.fn gradient} can only be called inside a {.fn jit}-compiled function.",
        i = "Wrap the result of {.fn gradient} in {.fn jit}, e.g. {.code jit(gradient(f))}."
      ))
    }
    fwd_graph <- trace_fn(
      f,
      args_flat = prep$args_flat,
      in_tree = prep$in_tree,
      mode = "inline"
    )
    grad_graph <- transform_gradient(fwd_graph, wrt)
    # parent_desc is modified in place
    inline_graph_into_desc(parent_desc, grad_graph)
  }
  formals(f_gradient) <- formals2(f)
  return(f_gradient)
}

#' @title Value and Gradient
#' @description
#' Returns a new function that computes both the output of `f` and its gradient in a
#' single forward+reverse pass. The result is a named list with elements `value` (the
#' original return value of `f`) and `grad` (the gradients, structured like the inputs or
#' the `wrt` subset).
#' @inheritParams gradient
#' @return A function with the same formals as `f` that returns
#'   `list(value = ..., grad = ...)`.
#' @seealso [`gradient()`]
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' loss_fn <- function(x) sum(x^2L)
#' vg <- jit(value_and_gradient(loss_fn))
#' result <- vg(nv_array(c(3, 4), dtype = "f32"))
#' result$value
#' result$grad
value_and_gradient <- function(f, wrt = NULL) {
  wrt <- resolve_arg_names(f, wrt, "wrt")
  if (!is.null(wrt) && !all(wrt %in% formalArgs(f))) {
    cli_abort("wrt must be a subset of the formal arguments of f")
  }
  f_value_and_grad <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    prep <- prepare_gradient_args(args, wrt)

    parent_desc <- .current_descriptor(silent = TRUE)
    if (is.null(parent_desc)) {
      cli_abort(c(
        "{.fn value_and_gradient} can only be called inside a {.fn jit}-compiled function.",
        i = "Wrap the result of {.fn value_and_gradient} in {.fn jit}, e.g. {.code jit(value_and_gradient(f))}."
      ))
    }
    fwd_graph <- trace_fn(
      f,
      args_flat = prep$args_flat,
      in_tree = prep$in_tree,
      mode = "inline"
    )
    res <- transform_gradient_impl(fwd_graph, wrt)
    grad_graph <- res$graph
    trans <- res$fwd_translation

    fwd_outputs <- lapply(fwd_graph$outputs, \(g) trans[[g]] %||% g)
    combined_graph <- grad_graph
    combined_graph$outputs <- c(fwd_outputs, grad_graph$outputs)

    counter <- new_counter()
    value_tree <- reindex_tree(fwd_graph$out_tree, counter)
    grad_tree <- reindex_tree(grad_graph$out_tree, counter)
    combined_graph$out_tree <- ListNode(
      list(value_tree, grad_tree),
      names = c("value", "grad")
    )
    inline_graph_into_desc(parent_desc, combined_graph)
  }
  formals(f_value_and_grad) <- formals2(f)
  f_value_and_grad
}
