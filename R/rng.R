#' @include aaa.R
#' @include side-effects.R
NULL

#' @title Set the Global Random State
#' @description
#' Seeds the global random state, analogous to base R's [set.seed()].
#' Calling
#' it is optional: until the global random state is first drawn from, it is not
#' set, and the first draw derives it from base R's RNG, so a [set.seed()]
#' before that makes the draws reproducible too.
#' Once anvl's random state is initialized, calling `set.seed()` no longer affects it.
#'
#' `nv_set_seed(seed)` is `nv_set_random_state(nv_random_state(seed))`, and
#' `local_nv_seed()` and `with_nv_seed()` are [local_nv_rng_state()] and
#' [with_nv_rng_state()] of `nv_random_state(seed)`.
#'
#' See `r roxy_article("random-numbers")` for more details.
#' @param seed (`integer(1)` | `NULL`)\cr
#'   The seed, a 32-bit integer, or `NULL` to derive the global random state
#'   from base R's RNG at once.
#' @param random_state (`AnvlRandomState` | `NULL`)\cr
#'   A random state returned by `nv_get_random_state()`. `NULL` unsets the
#'   global random state, so that the next draw derives it from base R's RNG.
#' @param code (any)\cr
#'   An expression to evaluate with the seeded global random state.
#' @param envir (`environment`)\cr
#'   The scope whose exit restores the global random state.
#' @return `nv_set_seed()`, `local_nv_seed()` and `nv_set_random_state()`
#'   return `NULL`, invisibly; `with_nv_seed()` returns the result of
#'   evaluating `code`. `nv_get_random_state()` returns the global random state,
#'   an `AnvlRandomState`, or `NULL` while it is not set.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' nv_set_seed(42L)
#' nv_runif(3L)
#' nv_runif(3L)
#'
#' # setting the seed again repeats the sequence
#' nv_set_seed(42L)
#' nv_runif(3L)
#'
#' # the counter counts sampling calls, not values
#' nv_get_random_state()
#'
#' # an unset random state is derived from base R's RNG on the next draw
#' nv_set_random_state(NULL)
#' set.seed(1)
#' nv_runif(3L)
#'
#' # a jitted function draws what the same code run eagerly draws
#' f <- function() nv_rnorm(2L) + nv_runif(2L)
#' with_nv_seed(1L, f())
#' with_nv_seed(1L, jit(f)())
#'
#' # save the random state and continue from it later
#' random_state <- nv_get_random_state()
#' nv_runif(2L)
#' nv_set_random_state(random_state)
#' nv_runif(2L)
#' @export
nv_set_seed <- function(seed) {
  abort_if_tracing("seeded")
  nv_set_random_state(nv_random_state(seed))
}

#' @rdname nv_set_seed
#' @export
nv_get_random_state <- function() {
  abort_if_tracing("read")
  abort_if_no_rng()
  state <- globals$random_state
  if (!is.null(state)) copy_random_state(state)
}

#' @rdname nv_set_seed
#' @export
nv_set_random_state <- function(random_state) {
  abort_if_tracing("set")
  abort_if_no_rng()
  globals$random_state <- if (!is.null(random_state)) copy_random_state(assert_random_state(random_state))
  invisible(NULL)
}

#' @export
format.AnvlRandomState <- function(x, ...) {
  # a 64-bit value of its high and low 32-bit word
  hex <- function(high, low) paste0("0x", format_hex32(high), format_hex32(low))
  w <- random_state_words(x)
  sprintf("<AnvlRandomState> key %s, counter %s", hex(w[[1L]], w[[2L]]), hex(w[[4L]], w[[3L]]))
}

#' @export
print.AnvlRandomState <- function(x, ...) {
  cat(format(x), "\n", sep = "")
  invisible(x)
}

# The 32-bit word `x`, a whole R double in [0, 2^32), as 8 hexadecimal digits.
format_hex32 <- function(x) {
  paste0(sprintf("%04x", as.integer(x %/% 65536)), sprintf("%04x", as.integer(x %% 65536)))
}

abort_if_tracing <- function(what) {
  if (currently_tracing()) {
    cli_abort(c(
      "The global random state cannot be {what} in a function traced by {.fn jit}.",
      i = "Do it before calling the function."
    ))
  }
}

abort_if_no_rng <- function() {
  backend <- active_backend()
  if (backend != "pjrt") {
    cli_abort("The {backend} backend does not support random draws, so it has no global random state.")
  }
}

# `x` if it is a random state: an `AnvlRandomState` of a `ui32` key of shape 2
# and a `ui64` scalar counter.
assert_random_state <- function(x, arg = "random_state") {
  is_array_of <- function(a, dtype, shape) is_anvl_array(a) && dtype(a) == dtype && identical(shape(a), shape)
  if (
    !is_random_state(x) ||
      !is_array_of(x$random_key, "ui32", 2L) ||
      !is_array_of(x$random_counter, "ui64", integer())
  ) {
    cli_abort(c(
      "{.arg {arg}} must be a random state, as {.fn nv_random_state} or {.fn nv_get_random_state} return.",
      i = "A random state is a {.cls ui32} key of shape 2 and a {.cls ui64} scalar counter."
    ))
  }
  x
}

is_random_state <- function(x) {
  inherits(x, c("AnvlRandomState", "AnvlTracedRandomState"))
}

# A random state of the key `key` and the counter `counter`: an environment, so
# that the draws from it advance it in place.
new_random_state <- function(key, counter) {
  env <- new.env(parent = emptyenv())
  env$random_key <- key
  env$random_counter <- counter
  structure(env, class = "AnvlRandomState")
}

copy_random_state <- function(x) {
  new_random_state(x$random_key, x$random_counter)
}

# The random state of the 32-bit words `words` (whole R doubles in [0, 2^32)):
# the key's words, in the order of JAX's key data, then the counter's low and
# high word.
random_state_of_words <- function(words) {
  # a bitcast to `ui64` reads the first word as the low one
  new_random_state(
    nv_array(words[1:2], dtype = "ui32"),
    nv_bitcast_convert(nv_array(words[3:4], dtype = "ui32"), dtype = "ui64")
  )
}

# The 32-bit words of the random state `x`, as `random_state_of_words()` takes
# them.
random_state_words <- function(x) {
  counter <- nv_bitcast_convert(x$random_counter, dtype = "ui32")
  c(as.numeric(as_array(x$random_key)), as.numeric(as_array(counter)))
}

#' @title Create a Random State
#' @description
#' Creates a random state of its own, which the samplers draw from in place of
#' the global random state (see [nv_set_seed()]) within [with_nv_rng_state()]
#' or after [local_nv_rng_state()]. Like the global random state, it is a key
#' and a counter of the ThreeFry-2x32 generator, and every sampling call that
#' draws from it advances it in place, so two draws from the same random state
#' differ.
#'
#' A function compiled with [jit()] can take a random state as an argument,
#' also inside a list, and make it the samplers' default with
#' [local_nv_rng_state()] or [with_nv_rng_state()]: the compiled program takes
#' its key and counter as inputs, like those of any other argument, and after
#' the call the random state is advanced in place by the number of sampling
#' calls the program made from it, as the global random state is. Its `random_key` and `random_counter` read in the jitted function what
#' they read eagerly, the counter after the draws so far; the random state
#' itself is not a value there, which could be returned or carried through a
#' loop. A jitted function cannot draw from a random state it closes over, and
#' the same random state must not be passed to it twice.
#' @param seed (`integer(1)` | `NULL`)\cr
#'   The seed, a 32-bit integer, which gives the same random state as
#'   [nv_set_seed()] gives the global random state. `NULL` derives it from
#'   base R's RNG.
#' @return (`AnvlRandomState`)\cr
#'   A random state of the key that `seed` gives and the counter 0.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' rng <- nv_random_state(42L)
#' with_nv_rng_state(rng, nv_runif(3L))
#' # the random state was advanced, so the next draw differs
#' with_nv_rng_state(rng, nv_runif(3L))
#' rng
#'
#' # a jitted function takes a random state as an argument
#' f <- jit(function(x, rng) with_nv_rng_state(rng, x + nv_rnorm(2L)))
#' f(nv_array(c(1, 2)), nv_random_state(1L))
#' @export
nv_random_state <- function(seed = NULL) {
  abort_if_no_rng()
  if (is.null(seed)) random_state_from_r() else random_state_from_seed(assert_int(seed, coerce = TRUE))
}

#' @title Draw From a Random State for a Scope
#' @description
#' `local_nv_rng_state()` and `with_nv_rng_state()` make the random state
#' `rng_state` the one the samplers draw from, for the calling scope or for
#' `code` only, as [local_nv_seed()] and [with_nv_seed()] do with a seed. The
#' draws advance `rng_state` itself, in place, and the random state from before
#' is restored afterwards.
#'
#' Outside of [jit()], `rng_state` is the global random state for the scope.
#' In a function traced by [jit()], `rng_state` must be a random state the
#' jitted function takes as an argument: the samplers in the scope draw from it,
#' also in the bodies of loops, and the jitted function advances it after the
#' call. A random state the
#' function closes over is an error, as the function could neither take its
#' current state nor advance it. It can be called in a branch of [nv_if()] or a
#' function [gradient()] differentiates, such as a jitted function called
#' there, but not in the body of [nv_while()] or [nv_scan()], nor in a function
#' traced inside one: called before the loop, it reaches the loop body.
#' @param rng_state (`AnvlRandomState`)\cr
#'   The random state to draw from, as [nv_random_state()] returns.
#' @param code (any)\cr
#'   An expression to evaluate drawing from `rng_state`.
#' @param envir (`environment`)\cr
#'   The scope whose exit restores the default from before.
#' @return `local_nv_rng_state()` returns `NULL`, invisibly; `with_nv_rng_state()`
#'   returns the result of evaluating `code`.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' rng <- nv_random_state(1L)
#' with_nv_rng_state(rng, list(nv_runif(2L), nv_rnorm(2L)))
#' # both samplers drew from `rng`, three sampling calls in all
#' rng
#'
#' # in a jitted function, from a random state it takes as an argument
#' step <- jit(function(x, rng) {
#'   local_nv_rng_state(rng)
#'   x + nv_rnorm(2L) + nv_runif(2L)
#' })
#' step(nv_array(c(0, 0)), rng)
#' rng
#' @export
local_nv_rng_state <- function(rng_state, envir = parent.frame()) {
  if (currently_tracing()) {
    if (!is_traced_random_state(rng_state)) {
      cli_abort(c(
        "{.arg rng_state} must be a random state the jitted function takes as an argument.",
        i = "A random state the function closes over cannot be drawn from in it."
      ))
    }
    # a loop body draws from the default set before the loop
    if (current_descriptor(silent = TRUE)$in_loop_body) {
      cli_abort(c(
        "{.fn local_nv_rng_state} cannot be used in the body of {.fn nv_while} or {.fn nv_scan}.",
        i = "Call it before the loop: the default it sets there reaches the loop body."
      ))
    }
    # the default of the trace under way and the traces opened in it, for the
    # calling scope
    old <- globals$default_traced_random_state
    globals$default_traced_random_state <- rng_state
    withr::defer(assign("default_traced_random_state", old, envir = globals), envir = envir)
  } else {
    abort_if_no_rng()
    assert_random_state(rng_state, "rng_state")
    old <- globals$random_state
    globals$random_state <- rng_state
    withr::defer(assign("random_state", old, envir = globals), envir = envir)
  }
  invisible(NULL)
}

#' @rdname local_nv_rng_state
#' @export
with_nv_rng_state <- function(rng_state, code) {
  local_nv_rng_state(rng_state)
  code
}

# A random state is a node of a tree (see `pjrt::register_tree_node()`), whose
# children are its key and its counter: a jitted function takes a random state
# among its arguments as two inputs, and is traced with a
# `AnvlTracedRandomState` of them. Registered when anvl is loaded.
register_random_state_node <- function() {
  pjrt::register_tree_node(
    "AnvlRandomState",
    # a random state has no aux data: its key and counter are all of it
    flatten = function(x) list(list(random_key = x$random_key, random_counter = x$random_counter), NULL),
    unflatten = function(children, aux) {
      if (is_graph_box(children$random_key)) {
        traced_random_state(children$random_key$gnode, children$random_counter$gnode)
      } else {
        new_random_state(children$random_key, children$random_counter)
      }
    },
    repr = function(children, aux) sprintf("AnvlRandomState(%s)", paste(children, collapse = ", "))
  )
}

# The global random state as R holds it, outside of any program, which the
# first draw from it derives from base R's RNG.
stored_random_state <- function() {
  if (is.null(globals$random_state)) {
    globals$random_state <- random_state_from_r()
  }
  globals$random_state
}

# ---- The random state in a trace ----------------------------------------------

# A random state as a traced function takes it: the handle of a trace-time side
# effect (see R/side-effects.R), which identifies it to the samplers. The
# `random_key` and `random_counter` of a random state passed as an argument read
# the key and the counter where the trace that reads them is, after its draws so
# far, as those of a random state read eagerly do. The global random state's
# handle has none, so that it can be stored with the package, and is read with
# `nv_get_random_state()` outside of a trace.
# - `key`, `counter`: the GraphValues of the key and the counter among the
#   inputs of the outermost trace, `NULL` for the global random state;
# - `leaf`: the position of the key among the call's leaves, which
#   `side_effect_start()` records.
traced_random_state <- function(key = NULL, counter = NULL) {
  x <- new.env(parent = emptyenv())
  x$key <- key
  x$counter <- counter
  x$leaf <- NULL
  if (!is.null(key)) {
    makeActiveBinding("random_key", function() random_state_current(x)$random_key, x)
    makeActiveBinding("random_counter", function() random_state_current(x)$random_counter, x)
  }
  structure(x, class = c("AnvlTracedRandomState", "TraceTimeSideEffect"))
}

is_traced_random_state <- function(x) {
  inherits(x, "AnvlTracedRandomState")
}

#' @export
format.AnvlTracedRandomState <- function(x, ...) {
  "<AnvlTracedRandomState>"
}

#' @export
print.AnvlTracedRandomState <- function(x, ...) {
  cat(format(x), "\n", sep = "")
  invisible(x)
}

# The global random state in every trace, which the samplers draw from unless
# they are given another.
global_random_state <- traced_random_state()

# The key and the counter of the traced random state `x` in the trace under
# way: those of its view there, the counter after the trace's draws.
random_state_current <- function(x) {
  desc <- current_descriptor(silent = TRUE)
  if (is.null(desc)) {
    cli_abort("A random state of a traced function cannot be read after its trace.")
  }
  side_effect_current(desc_side_effect(desc, x))
}

# `box`, with its node labelled `label` (see `GraphValue()`) unless it has a
# label already.
# The value of a random state in a trace, of the boxes `key` and `counter`,
# labelled for printing.
random_state_value <- function(key, counter) {
  list(random_key = label_box(key, "key"), random_counter = label_box(counter, "counter"))
}

label_box <- function(box, label) {
  if (!is_graph_literal(box$gnode) && is.null(box$gnode$label)) {
    box$gnode$label <- label
  }
  box
}

# Which random state a sampling call draws from: the one `local_nv_rng_state()`
# made the default in the trace under way, and otherwise the global one.
random_state_id <- function() {
  globals$default_traced_random_state %||% global_random_state
}

abort_random_state_not_argument <- function() {
  cli_abort(c(
    "A random state can only be drawn from by a jitted function it is passed to.",
    i = "Pass it to the jitted function as an argument rather than closing over it."
  ))
}

# The side effect protocol (see R/side-effects.R).

# The outermost trace takes the global random state's key and counter as
# implicit inputs, and a random state passed as an argument from among its
# inputs, which it must be among.
#' @export
side_effect_start.AnvlTracedRandomState <- function(effect, desc) {
  if (identical(effect, global_random_state)) {
    inputs <- list(
      random_key = GraphValue(AbstractArray(dtype = "ui32", shape = 2L), label = "key"),
      random_counter = GraphValue(AbstractArray(dtype = "ui64", shape = integer()), label = "counter")
    )
    return(SideEffectView(effect, register_gvals(desc, inputs), inputs = inputs))
  }
  input <- which(vapply(desc$inputs, identical, logical(1L), effect$key))
  if (length(input) != 1L) {
    abort_random_state_not_argument()
  }
  effect$leaf <- which(!desc$is_static_flat)[[input]]
  SideEffectView(effect, random_state_value(register_gval(desc, effect$key), register_gval(desc, effect$counter)))
}

#' @export
side_effect_host_object.AnvlTracedRandomState <- function(effect, args) {
  if (is.null(effect$leaf)) stored_random_state() else random_state_argument(args, effect$leaf)
}

# A draw splits at the counter plus the draws before it, so that no draw waits
# for another.
#' @export
side_effect_at.AnvlTracedRandomState <- function(effect, value, n) {
  random_state_value(value$random_key, counter_plus(value$random_counter, n))
}

# The fork: a call of `prim_fork_random_state()` that gives a child random
# state, which a loop body advances as often as it runs.
#' @export
side_effect_fork.AnvlTracedRandomState <- function(effect, value) {
  out <- graph_desc_add(prim_fork_random_state, value, infer_fn = infer_fork_random_state)
  random_state_value(out[[1L]], out[[2L]])
}

# After a run, the counter is advanced by the draws the program made.
#' @export
side_effect_advance.AnvlTracedRandomState <- function(effect, object, n) {
  counter <- object$random_counter
  object$random_counter <- counter + nv_scalar(n, dtype = "ui64", device = device(counter))
  invisible(NULL)
}

#' @export
side_effect_format.AnvlTracedRandomState <- function(effect, ids, n) {
  sprintf("%s += %s", ids[["random_counter"]], format(n))
}

# The random state the leaf `leaf` of the arguments `args` of a call is the key
# of, finding it as `pjrt::flatten()` does: a bare list is recursed into, `NULL`
# has no leaf, a random state has two, and anything else one.
random_state_argument <- function(args, leaf) {
  # `list(n, found)`: the leaves of `x`, whose first is `n + 1`, end at `n`, or
  # `x` holds the random state `found`
  walk <- function(x, n) {
    if (is.null(x)) {
      return(list(n = n))
    }
    if (is_random_state(x)) {
      return(list(n = n + 2L, found = if (n + 1L == leaf) x))
    }
    if (is.list(x) && !is.object(x)) {
      for (el in x) {
        res <- walk(el, n)
        if (!is.null(res$found)) {
          return(res)
        }
        n <- res$n
      }
      return(list(n = n))
    }
    list(n = n + 1L)
  }
  walk(args, 0L)$found
}

#' @rdname nv_set_seed
#' @export
local_nv_seed <- function(seed, envir = parent.frame()) {
  abort_if_tracing("seeded")
  local_nv_rng_state(nv_random_state(seed), envir = envir)
}

#' @rdname nv_set_seed
#' @export
with_nv_seed <- function(seed, code) {
  local_nv_seed(seed)
  code
}

# The random state of the key words `key` (two whole R doubles in [0, 2^32),
# in the order of JAX's key data) and the counter `0`.
random_state_of_key <- function(key) {
  new_random_state(nv_array(key, dtype = "ui32"), nv_scalar(0L, dtype = "ui64"))
}

# A fresh random state from two uniform draws of base R's RNG, each giving one
# 32-bit word of the key. A uniform draw of R's default generator has 32
# random bits.
random_state_from_r <- function() {
  random_state_of_key(floor(stats::runif(2L) * 2^32))
}

# The random state of the integer `seed`: the key of JAX's `key(seed)`, the
# words 0 and `seed` read as an unsigned 32-bit integer.
random_state_from_seed <- function(seed) {
  random_state_of_key(c(0, seed %% 2^32))
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
