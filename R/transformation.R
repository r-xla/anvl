# Helpers shared by the transformations of a function -- `gradient()`,
# `pullback()`, `vectorize()` -- that trace it into a graph of its own and
# replay that graph into the trace they are called in.

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
