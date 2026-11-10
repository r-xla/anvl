#' @include rng.R
NULL

# The random state in a trace
#
# A random state a trace draws from is a side effect (see R/side-effects.R),
# whose handle (`new_rng_state_handle()`) says which random state it is:
# - the global random state, `global_rng_state_handle`;
# - a random state the traced function closes over, a handle of the random
#   state R holds;
# - a random state passed to the jitted function as an argument, a handle that
#   knows the GraphValues of its key and its counter, which the trace computes
#   from its words among its inputs, and its place among the random states of
#   the arguments.
# The program takes the key and the counter of the first two as implicit inputs,
# and the words of the third among its inputs, which it computes them from.
# After the run, R advances each random state by the draws the program made from
# it.
#
# What the traced function holds of a random state is an `AnvlRngState`: the one
# R holds, which it closes over, or, for one it received as an argument, one of
# its key and counter in the trace (`received_rng_state()`).
#
# `local_nv_rng_state()` makes a random state the default of the trace under
# way, and of the traces opened in it (`active_rng_state()`).

# The handle of a random state in a trace (see the top of this file): the global
# random state's with no fields, one the traced function closes over with
# `object`, the random state R holds, and one passed as an argument with `key`,
# `counter` and `ordinal`. A handle is a list, which `identical()` compares by
# its fields, and never changes.
new_rng_state_handle <- function(...) {
  structure(list(...), class = c("AnvlRngStateHandle", "SideEffect"))
}

# The global random state in every trace, which the samplers draw from unless
# they are given another.
global_rng_state_handle <- new_rng_state_handle()

# The handle a draw from the random state `x` in a trace uses: that of a random
# state the traced function received, or a handle of the random state R holds,
# which the function closes over.
rng_state_handle <- function(x) {
  x$handle %||% new_rng_state_handle(object = x)
}

# Whether `x`, a random state, is one of a trace's key and counter, which only
# the trace can read.
is_traced_rng_state <- function(x) {
  is_graph_box(x$key)
}

# The random state a jitted function receives for one among its arguments, of
# whose key and counter the trace under way was given the 32-bit words, `key`
# and `counter`, boxes of bare R data (see `AnvlRngState()`). The key is its
# words, and the counter the bits of its low and high word, in the order
# `prim_bitcast_convert()` reads them. Its handle knows its place among the
# random states of the arguments, in the order the trace receives them, which is
# the order `pjrt::flatten()` meets them in (see `rng_state_argument()`).
received_rng_state <- function(key, counter) {
  desc <- current_descriptor()
  # the counter's words are `ui32`, like the key's
  words_dtype <- rng_state_dtype("key")
  value <- rng_state_value(
    materialize_rdata(key, words_dtype),
    prim_bitcast_convert(materialize_rdata(counter, words_dtype), rng_state_dtype("counter"))
  )
  handle <- new_rng_state_handle(
    key = value$key$gnode,
    counter = value$counter$gnode,
    ordinal = length(desc$rng_state_args) + 1L
  )
  desc$rng_state_args <- c(desc$rng_state_args, list(handle))
  x <- new.env(parent = emptyenv())
  x$key <- value$key
  x$counter <- value$counter
  x$handle <- handle
  structure(x, class = "AnvlRngState")
}

# A random state is a node of a tree (see `pjrt::register_tree_node()`) of its
# key and its counter: a jitted function takes them among its arguments, and a
# trace receives them (see `AnvlRngState()`). Registered when anvl is loaded.
register_rng_state_node <- function() {
  pjrt::register_tree_node(
    "AnvlRngState",
    flatten = function(x) list(list(key = x$key, counter = x$counter), NULL),
    unflatten = function(children, aux) AnvlRngState(children$key, children$counter),
    repr = function(children, aux) sprintf("AnvlRngState(%s)", paste(children, collapse = ", "))
  )
}

# The value of a random state in a trace, of the boxes `key` and `counter`,
# labelled for printing.
rng_state_value <- function(key, counter) {
  list(key = label_box(key, "key"), counter = label_box(counter, "counter"))
}

# `box`, with its node labelled `label` (see `GraphValue()`) unless it has a
# label already.
label_box <- function(box, label) {
  if (!is_graph_literal(box$gnode) && is.null(box$gnode$label)) {
    box$gnode$label <- label
  }
  box
}

# The handle of the random state a sampling call draws from: the one
# `local_nv_rng_state()` made the default in the trace under way or a trace it
# is opened in, and otherwise the global one.
active_rng_state <- function() {
  desc <- current_descriptor(silent = TRUE)
  while (!is.null(desc)) {
    if (!is.null(desc$default_rng_state)) {
      return(desc$default_rng_state)
    }
    desc <- desc$parent
  }
  global_rng_state_handle
}

# Whether the trace `desc` is the body of a loop, or is opened inside one.
in_loop_body <- function(desc) {
  while (!is.null(desc)) {
    if (identical(desc$side_effects_mode, "loop")) {
      return(TRUE)
    }
    desc <- desc$parent
  }
  FALSE
}

# ---- The side effect protocol (see R/side-effects.R) --------------------------

# The program takes the key and the counter of the global random state and of
# one the traced function closes over as implicit inputs, and those of a random
# state passed as an argument from among its inputs, which it must be among.
#' @export
side_effect_start.AnvlRngStateHandle <- function(effect, desc) {
  if (is.null(effect$key)) {
    inputs <- lapply(rng_state_fields, function(field) {
      GraphValue(AbstractArray(dtype = field$dtype, shape = field$shape), label = field$label)
    })
    return(SideEffectView(effect, register_gvals(desc, inputs), inputs = inputs))
  }
  if (!any(vapply(desc$rng_state_args, identical, logical(1L), effect))) {
    cli_abort(c(
      "This random state cannot be drawn from here.",
      i = "It was passed to a function traced inside the jitted function rather than to the jitted function."
    ))
  }
  SideEffectView(effect, rng_state_value(register_gval(desc, effect$key), register_gval(desc, effect$counter)))
}

#' @export
side_effect_host_object.AnvlRngStateHandle <- function(effect, args) {
  if (!is.null(effect$ordinal)) {
    rng_state_argument(args, effect$ordinal)
  } else {
    effect$object %||% global_rng_state()
  }
}

#' @export
side_effect_input.AnvlRngStateHandle <- function(effect, object, name) {
  rng_state_buffer(object, name)
}

# A draw splits at the counter plus the draws before it, so that no draw waits
# for another.
#' @export
side_effect_after.AnvlRngStateHandle <- function(effect, value, n) {
  rng_state_value(value$key, counter_plus(value$counter, n))
}

# The fork (`prim_fork_rng_state()`) gives a child random state, which a loop
# body advances as often as it runs.
#' @export
side_effect_fork.AnvlRngStateHandle <- function(effect, value) {
  out <- graph_desc_add(prim_fork_rng_state, value, infer_fn = infer_fork_rng_state)
  rng_state_value(out[[1L]], out[[2L]])
}

# After a run, R advances the counter by the draws the program made.
#' @export
side_effect_advance.AnvlRngStateHandle <- function(effect, object, n) {
  object$counter <- array(counter_words_plus(object$counter, n), dim = 2L)
  invisible(NULL)
}

#' @export
side_effect_format.AnvlRngStateHandle <- function(effect, ids, n) {
  sprintf("%s += %s", ids[["counter"]], format(n))
}

#' @export
format.AnvlRngStateHandle <- function(x, ...) {
  "<AnvlRngStateHandle>"
}

#' @export
print.AnvlRngStateHandle <- function(x, ...) {
  cat(format(x), "\n", sep = "")
  invisible(x)
}

# The counter of the 32-bit words `counter`, its low and high word, plus `n` (a
# whole R double below 2^53), wrapping around at 2^64. `n` is added as its low
# and its high 32 bits, so that every sum stays below 2^53, exact as a double.
counter_words_plus <- function(counter, n) {
  low <- counter[[1L]] + n %% 2^32
  high <- counter[[2L]] + n %/% 2^32 + low %/% 2^32
  c(low %% 2^32, high %% 2^32)
}

# The random state that is the `ordinal`-th random state among the arguments
# `args` of a call, in the order `pjrt::flatten()` meets them in.
rng_state_argument <- function(args, ordinal) {
  states <- Filter(is_rng_state, flatten(args, is_leaf = is_rng_state))
  if (ordinal > length(states)) {
    cli_abort("The random state the jitted function drew from is no longer among its arguments.")
  }
  states[[ordinal]]
}

# The `ui64` counter `counter` plus `n` (a whole R double below 2^53) in a
# trace, wrapping around at 2^64. `n` enters as R integer literals, which keep
# the counter's data type where a double would make the sum a float: its low 31
# bits, and the rest shifted into place.
counter_plus <- function(counter, n) {
  if (n == 0) {
    return(counter)
  }
  counter <- counter + as.integer(n %% 2^31)
  if (n >= 2^31) {
    counter <- counter + nv_shift_left(nv_fill(as.integer(n %/% 2^31), integer(), dtype = "ui64"), 31L)
  }
  counter
}
