#' @include rng.R
NULL

# The random state in a trace
#
# A random state a trace draws from is a side effect (see R/side-effects.R),
# whose handle (`new_random_state_handle()`) says which random state it is:
# - the global random state, `global_random_state_handle`;
# - a random state the traced function closes over, a handle of the random
#   state R holds;
# - a random state passed to the jitted function as an argument, a handle that
#   knows the GraphValues of its key and its counter, which the trace computes
#   from its words among its inputs, and its place among the random states of
#   the arguments.
# The program takes the key and the counter of the first two as implicit inputs,
# and the words of the third as one of its inputs, which it computes them from.
# After the run, R advances each random state by the draws the program made from
# it.
#
# What the traced function holds of a random state is an `AnvlRandomState` too:
# the one R holds, which it closes over, or, for one it received as an argument,
# one of the handle alone (`traced_random_state()`). Either reads its key and
# its counter where the trace is, after the draws so far
# (`random_state_current()`).
#
# `local_nv_rng_state()` makes a random state the default of the trace under
# way, and of the traces opened in it (`active_random_state()`).

# The handle of a random state in a trace (see the top of this file): the global
# random state's with no fields, one the traced function closes over with
# `object`, the random state R holds, and one passed as an argument with `key`,
# `counter` and `ordinal`. A handle is a list, which `identical()` compares by
# its fields, and never changes.
new_random_state_handle <- function(...) {
  structure(list(...), class = c("AnvlRandomStateHandle", "SideEffect"))
}

# The global random state in every trace, which the samplers draw from unless
# they are given another.
global_random_state_handle <- new_random_state_handle()

# The handle a draw from the random state `x` in a trace uses: that of a random
# state the traced function received, or a handle of the random state R holds,
# which the function closes over.
random_state_handle <- function(x) {
  x$handle %||% new_random_state_handle(object = x)
}

# Whether `x`, a random state, is one a traced function received for an
# argument, which only the trace can read.
is_traced_random_state <- function(x) {
  !is.null(x$handle)
}

# The random state of the handle `handle` that a traced function receives.
traced_random_state <- function(handle) {
  x <- new.env(parent = emptyenv())
  x$handle <- handle
  for (name in names(random_state_fields)) {
    makeActiveBinding(name, random_state_binding(x, name), x)
  }
  structure(x, class = "AnvlRandomState")
}

# The random state a jitted function receives for one among its arguments, of
# whose words the trace under way was given the box `words`, of bare R data (see
# `register_random_state_node()`). The program takes them at `f64`, which holds
# them exactly, and converts them to `ui32` (see `build_r_staged()`), from which
# it computes the key and the counter: the counter is the bits of its low and
# its high word, in the order `prim_bitcast_convert()` reads them, as
# `random_state_buffer()` lays them out. Its handle knows its place among the
# random states of the arguments, in the order the trace rebuilds them, which is
# the order `pjrt::flatten()` meets them in (see `random_state_argument()`).
argument_random_state <- function(words) {
  desc <- current_descriptor()
  words <- materialize_rdata(words, random_state_dtype("random_key"))
  key <- prim_static_slice(words, 1L, 2L, 1L)
  counter <- prim_bitcast_convert(prim_static_slice(words, 3L, 4L, 1L), random_state_dtype("random_counter"))
  handle <- new_random_state_handle(
    key = key$gnode,
    counter = counter$gnode,
    ordinal = length(desc$random_state_args) + 1L
  )
  desc$random_state_args <- c(desc$random_state_args, list(handle))
  traced_random_state(handle)
}

# A random state is a node of a tree (see `pjrt::register_tree_node()`). Outside
# of a trace, its one child is its words, as bare R data: a jitted function
# takes a random state among its arguments as one input, which the call uploads
# to the device of its other inputs, and is traced with an
# `argument_random_state()` of it. In a trace, its one child is its handle,
# which is no array: there a random state is not a value, and a function the
# trace opens, such as the one `gradient()` differentiates, receives it as it
# is. Registered when anvl is loaded.
register_random_state_node <- function() {
  pjrt::register_tree_node(
    "AnvlRandomState",
    # a random state has no aux data: its children are all of it
    flatten = function(x) {
      found <- globals$found_random_states
      if (!is.null(found)) {
        found(x)
        return(list(list(), NULL))
      }
      if (is_traced_random_state(x) || currently_tracing()) {
        return(list(list(handle = random_state_handle(x)), NULL))
      }
      # an array: pjrt's dispatcher takes no other bare R data of length 4
      list(list(words = array(x$words, dim = 4L)), NULL)
    },
    unflatten = function(children, aux) {
      if (!is.null(children$handle)) {
        traced_random_state(children$handle)
      } else if (is_graph_box(children$words)) {
        argument_random_state(children$words)
      } else {
        # words a trace takes for a static value, which `jit()` refuses for a
        # random state (see `check_static_args()`)
        random_state_of_words(as.vector(children$words))
      }
    },
    repr = function(children, aux) sprintf("AnvlRandomState(%s)", paste(children, collapse = ", "))
  )
}

# The key and the counter of the random state of the handle `handle` in the
# trace under way: those of its view there, the counter after the trace's draws.
random_state_current <- function(handle) {
  desc <- current_descriptor(silent = TRUE)
  if (is.null(desc)) {
    cli_abort("The random state of a traced function cannot be read after its trace.")
  }
  view_current(desc_view(desc, handle))
}

# The value of a random state in a trace, of the boxes `key` and `counter`,
# labelled for printing.
random_state_value <- function(key, counter) {
  list(random_key = label_box(key, "key"), random_counter = label_box(counter, "counter"))
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
active_random_state <- function() {
  desc <- current_descriptor(silent = TRUE)
  while (!is.null(desc)) {
    if (!is.null(desc$default_random_state)) {
      return(desc$default_random_state)
    }
    desc <- desc$parent
  }
  global_random_state_handle
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
side_effect_start.AnvlRandomStateHandle <- function(effect, desc) {
  if (is.null(effect$ordinal)) {
    inputs <- lapply(random_state_fields, function(field) {
      GraphValue(AbstractArray(dtype = field$dtype, shape = field$shape), label = field$label)
    })
    return(SideEffectView(effect, register_gvals(desc, inputs), inputs = inputs))
  }
  if (!any(vapply(desc$random_state_args, identical, logical(1L), effect))) {
    cli_abort(c(
      "This random state cannot be drawn from here.",
      i = "It was rebuilt by a trace inside the jitted function, such as a loop body, rather than passed to it."
    ))
  }
  SideEffectView(effect, random_state_value(register_gval(desc, effect$key), register_gval(desc, effect$counter)))
}

#' @export
side_effect_host_object.AnvlRandomStateHandle <- function(effect, args) {
  if (!is.null(effect$ordinal)) {
    random_state_argument(args, effect$ordinal)
  } else {
    effect$object %||% global_random_state()
  }
}

#' @export
side_effect_input.AnvlRandomStateHandle <- function(effect, object, name) {
  random_state_buffer(object, name)
}

# A draw splits at the counter plus the draws before it, so that no draw waits
# for another.
#' @export
side_effect_after.AnvlRandomStateHandle <- function(effect, value, n) {
  random_state_value(value$random_key, counter_plus(value$random_counter, n))
}

# The fork: a call of `prim_fork_random_state()` that gives a child random
# state, which a loop body advances as often as it runs.
#' @export
side_effect_fork.AnvlRandomStateHandle <- function(effect, value) {
  out <- graph_desc_add(prim_fork_random_state, value, infer_fn = infer_fork_random_state)
  random_state_value(out[[1L]], out[[2L]])
}

# After a run, R advances the counter by the draws the program made.
#' @export
side_effect_advance.AnvlRandomStateHandle <- function(effect, object, n) {
  object$words[3:4] <- counter_words_plus(object$words[3:4], n)
  invisible(NULL)
}

#' @export
side_effect_format.AnvlRandomStateHandle <- function(effect, ids, n) {
  sprintf("%s += %s", ids[["random_counter"]], format(n))
}

#' @export
format.AnvlRandomStateHandle <- function(x, ...) {
  "<AnvlRandomStateHandle>"
}

#' @export
print.AnvlRandomStateHandle <- function(x, ...) {
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
# `args` of a call, in the order `pjrt::flatten()` meets them in, which records
# the random states it meets rather than flattening them.
random_state_argument <- function(args, ordinal) {
  found <- new.env(parent = emptyenv())
  found$states <- list()
  old <- globals$found_random_states
  globals$found_random_states <- function(x) found$states <- c(found$states, list(x))
  on.exit(assign("found_random_states", old, envir = globals))
  pjrt::flatten(args)
  if (ordinal > length(found$states)) {
    cli_abort("The random state the jitted function drew from is no longer among its arguments.")
  }
  found$states[[ordinal]]
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
