# Helpers shared by the transformations of a function -- `gradient()`,
# `pullback()`, `pushforward()`, `vectorize()` -- that trace it into a graph of
# its own and replay that graph into the trace they are called in.

# The leaves of the arguments `args` of a transformation, as `args_flat`, and
# their tree, as `in_tree`. A random state among them is a leaf, which the
# transformed function receives as it is, so that its draws continue those of
# the enclosing trace.
flatten_args <- function(args) {
  list(
    args_flat = flatten(args, is_leaf = is_rng_state),
    in_tree = build_tree(args, is_leaf = is_rng_state)
  )
}

# `x` (see `resolve_arg_names()`) for the argument `arg` of a transformation of
# `f` such as `gradient()`, which must name formal arguments of `f`.
resolve_transformation_args <- function(f, x, arg, call = rlang::caller_env()) {
  x <- resolve_arg_names(f, x, arg)
  if (!is.null(x) && !all(x %in% formalArgs(f))) {
    cli_abort("{.arg {arg}} must be a subset of the formal arguments of {.arg f}.", call = call)
  }
  x
}

# The function a transformation of `f` such as `gradient()` returns: it has the
# formals of `f` and hands the arguments it was called with, evaluated, to
# `impl`. The formals of `f` live in its frame, so everything it uses is looked
# up past that frame, where no formal can shadow it.
transformation_fn <- function(f, impl) {
  force(impl)
  out <- function() {
    base::get("impl", envir = base::parent.env(base::environment()))(
      base::lapply(base::as.list(base::match.call())[-1L], base::eval, envir = base::parent.frame())
    )
  }
  formals(out) <- formals2(f)
  out
}

# The boxes of the current descriptor that the inputs of `graph` -- traced by
# a transformation such as `gradient()` from `args_flat` -- are bound to, one
# per argument that is not static. An R value the descriptor has not given a
# data type yet materializes at the one the traced body settled it at, which the
# descriptor's own finalize then takes into account like any other use of it
# there; a bare R value the transformation traced as an input does the same.
graph_operands <- function(graph, args_flat) {
  desc <- current_descriptor()
  Map(
    function(x, input) {
      if (is_rdata_box(x)) {
        x <- materialize_rdata(x, input$aval$dtype)
      } else if (is_valid_r(x)) {
        x <- build_r_at(x, input$aval$dtype, desc)
      }
      maybe_box_arrayish(x, desc)
    },
    args_flat[!graph$is_static_flat],
    graph$inputs
  )
}

# A transformation such as `gradient()`, named `name`, works on the trace the
# function it returns is called in. Errors name `call`, the call of that
# function.
assert_in_trace <- function(name, call = rlang::caller_env()) {
  if (is.null(current_descriptor(silent = TRUE))) {
    cli_abort(
      c(
        "{.fn {name}} can only be called inside a {.fn jit}-compiled function.",
        i = "Wrap the result of {.fn {name}} in {.fn jit}, e.g. {.code jit({name}(f))}."
      ),
      call = call
    )
  }
}

# The arguments `args` of `pullback()` or `pushforward()`, matched to the
# formals of `f`, as `prepare_gradient_args()` returns them: every float array
# among them is differentiated.
differentiated_args <- function(f, args) {
  prep <- flatten_args(match_args_to_formals(f, args))
  prep$is_wrt_flat <- vapply(prep$args_flat, is_float_array, logical(1L))
  prep
}

# `f` traced for `fn`, one of the transformations, at the arguments `prep`
# holds into a graph of its own, which `replay(graph, operands)` replays into
# the enclosing trace as a function of `operands`, the boxes its inputs are
# bound to there. Returns the graph, what `replay` returned, and the
# descriptor of the enclosing trace. Errors name `call`.
trace_replayed <- function(f, prep, fn, replay, call) {
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
  res <- replay(graph, graph_operands(graph, prep$args_flat))
  desc_end_capture(enclosing, list(graph))
  list(graph = graph, res = res, enclosing = enclosing)
}

# Refuses to call a pullback or a pushforward (`what`) that `fn` made in the
# trace `enclosing` outside that trace and the traces opened inside it.
assert_called_within <- function(enclosing, what, fn) {
  if (!descriptor_within(current_descriptor(silent = TRUE), enclosing)) {
    cli_abort(c(
      "A {what} can only be called in the function {.fn {fn}} was called in.",
      i = "Or in a function traced inside it, such as a branch of {.fn nv_if}."
    ))
  }
}

# The leaves of `x` -- a tangent of arguments or a cotangent of a value --
# aligned with those of `like`, `NULL` for each leaf of a part of `like` whose
# counterpart in `x` is `NULL`. `is_leaf` is passed to `flatten()`. Errors name
# `call`, with the headline `mismatch` for an `x` structured unlike `like`.
leaves_like <- function(like, x, call, mismatch, is_leaf = NULL) {
  if (is.null(x)) {
    return(vector("list", length(flatten(like, is_leaf = is_leaf))))
  }
  if (build_tree(x, is_leaf = is_leaf) == build_tree(like, is_leaf = is_leaf)) {
    return(flatten(x, is_leaf = is_leaf))
  }
  same_node <- rlang::is_bare_list(like) &&
    rlang::is_bare_list(x) &&
    length(x) == length(like) &&
    identical(names(x), names(like))
  if (!same_node) {
    cli_abort(c(mismatch, i = "{.code NULL} can stand in for any part of it."), call = call)
  }
  leaves <- Map(leaves_like, like, x, MoreArgs = list(call = call, mismatch = mismatch, is_leaf = is_leaf))
  do.call(c, unname(leaves)) %||% list()
}

# `x`, the tangent or cotangent (`kind`) of the float array `primal`, boxed in
# the current trace after checking that the two fit together; an R value takes
# the data type of `primal`. `where` (formatted) says which part of the
# arguments or of the value -- `of` -- it belongs to. Errors name `call`.
check_seed <- function(primal, x, kind, where, of, call) {
  dt <- dtype(primal)
  if (has_no_dtype(x)) {
    x <- materialize_at(x, dt)
  } else if (!is_arrayish(x)) {
    cli_abort(c("The {kind}{where} must be an array.", x = "Got {.cls {class(x)}}."), call = call)
  }
  x <- maybe_box_arrayish(x)
  if (dtype(x) != dt || !identical(shape(x), shape(primal))) {
    cli_abort(
      c(
        "The {kind}{where} must have the data type and shape of the {of}.",
        x = "The {of} is {.val {format_aval_short(to_abstract(primal))}}, the {kind} {.val {format_aval_short(to_abstract(x))}}." # nolint
      ),
      call = call
    )
  }
  x
}
