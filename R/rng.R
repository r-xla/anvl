#' @include aaa.R
#' @include side-effects.R
NULL

# The random state: the user API, and a random state as R holds it. How a trace
# draws from one is in R/rng-trace.R.

#' @title Set the Global Random State
#' @description
#' Seeds the global random state, analogous to base R's [set.seed()].
#' Calling
#' it is optional: until the global random state is first drawn from, it is not
#' set, and the first draw derives it from base R's RNG, so a [set.seed()]
#' before that makes the draws reproducible too.
#' Once anvl's random state is initialized, calling `set.seed()` no longer affects it.
#'
#' `nv_set_seed(seed)` is `nv_set_rng_state(nv_rng_state(seed))`, and
#' `local_nv_seed()` and `with_nv_seed()` are [local_nv_rng_state()] and
#' [with_nv_rng_state()] of `nv_rng_state(seed)`.
#'
#' See `r roxy_article("random-numbers")` for more details.
#' @param seed (`integer(1)`)\cr
#'   The seed, a 32-bit integer.
#' @param rng_state (`AnvlRngState` | `NULL`)\cr
#'   A random state returned by `nv_get_rng_state()`, of which the global
#'   random state becomes a copy. `NULL` unsets the global random state, so
#'   that the next draw derives it from base R's RNG.
#' @param code (any)\cr
#'   An expression to evaluate with the seeded global random state.
#' @param envir (`environment`)\cr
#'   The scope whose exit restores the global random state.
#' @return `nv_set_seed()`, `local_nv_seed()` and `nv_set_rng_state()`
#'   return `NULL`, invisibly; `with_nv_seed()` returns the result of
#'   evaluating `code`. `nv_get_rng_state()` returns a copy of the global
#'   random state, an `AnvlRngState`, or `NULL` while it is not set.
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
#' nv_get_rng_state()
#'
#' # an unset random state is derived from base R's RNG on the next draw
#' nv_set_rng_state(NULL)
#' set.seed(1)
#' nv_runif(3L)
#'
#' # a jitted function draws what the same code run eagerly draws
#' f <- function() nv_rnorm(2L) + nv_runif(2L)
#' with_nv_seed(1L, f())
#' with_nv_seed(1L, jit(f)())
#'
#' # save the random state and continue from it later
#' random_state <- nv_get_rng_state()
#' nv_runif(2L)
#' nv_set_rng_state(random_state)
#' nv_runif(2L)
#' @export
nv_set_seed <- function(seed) {
  abort_if_tracing("seeded")
  nv_set_rng_state(nv_rng_state(seed))
}

#' @rdname nv_set_seed
#' @export
nv_get_rng_state <- function() {
  abort_if_tracing("read")
  abort_if_no_rng()
  state <- globals$random_state
  # `NULL` visibly, which `if` alone would return invisibly
  if (is.null(state)) NULL else random_state_of_words(state$words)
}

#' @rdname nv_set_seed
#' @export
nv_set_rng_state <- function(rng_state) {
  abort_if_tracing("set")
  abort_if_no_rng()
  globals$random_state <- if (!is.null(rng_state)) random_state_of_words(assert_random_state(rng_state)$words)
  invisible(NULL)
}

#' @rdname nv_set_seed
#' @export
local_nv_seed <- function(seed, envir = parent.frame()) {
  abort_if_tracing("seeded")
  local_nv_rng_state(nv_rng_state(seed), envir = envir)
}

#' @rdname nv_set_seed
#' @export
with_nv_seed <- function(seed, code) {
  local_nv_seed(seed)
  code
}

#' @export
format.AnvlRngState <- function(x, ...) {
  if (is_traced_random_state(x)) {
    return("<AnvlRngState> of a traced function")
  }
  # a 64-bit value of its high and low 32-bit word
  hex <- function(high, low) paste0("0x", format_hex32(high), format_hex32(low))
  w <- x$words
  sprintf("<AnvlRngState> key %s, counter %s", hex(w[[1L]], w[[2L]]), hex(w[[4L]], w[[3L]]))
}

#' @export
print.AnvlRngState <- function(x, ...) {
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

# `x` if it is a random state: outside of a trace, one R holds the words of, and
# in a trace also one the traced function received (see `traced_random_state()`).
assert_random_state <- function(x, arg = "rng_state") {
  if (!is_random_state(x)) {
    cli_abort("{.arg {arg}} must be a random state, as {.fn nv_rng_state} or {.fn nv_get_rng_state} return.")
  }
  if (is_traced_random_state(x) && !currently_tracing()) {
    cli_abort("{.arg {arg}} is the random state of a traced function, which cannot be used after its trace.")
  }
  x
}

is_random_state <- function(x) {
  inherits(x, "AnvlRngState")
}

# ---- A random state on the host -----------------------------------------------

# The random state of the 32-bit words `words` (whole R doubles in [0, 2^32)):
# the key's words, in the order of JAX's key data, then the counter's low and
# high word. It is an environment, so that the draws from it advance it in
# place, and R holds its words, so that advancing it runs no program and reading
# it waits for none. A jitted function takes its words, as bare R data that the
# call uploads (see `register_random_state_node()`), so a donation takes them
# from no random state.
random_state_of_words <- function(words) {
  x <- new.env(parent = emptyenv())
  x$words <- words
  structure(x, class = "AnvlRngState")
}

# The key and the counter of a random state, by name: their words among its
# words (see `random_state_of_words()`), their data type and shape, and the
# label of their node in a printed graph.
random_state_fields <- list(
  key = list(words = 1:2, dtype = "ui32", shape = 2L, label = "key"),
  counter = list(words = 3:4, dtype = "ui64", shape = integer(), label = "counter")
)

# The random state of the arrays `key` (`ui32`, shape 2) and `counter` (`ui64`
# scalar), e.g. the ones `prim_fork_random_state()` returns.
random_state_of_arrays <- function(key, counter) {
  random_state_of_words(bytes_words(c(as_raw(key), as_raw(counter))))
}

# The buffer of the field `name` of the random state `x`, which R holds the
# words of, on the default device, from their bytes: a `ui64` reads the first of
# two words as the low one.
random_state_buffer <- function(x, name) {
  field <- random_state_fields[[name]]
  pjrt_buffer(
    words_bytes(x$words[field$words]),
    dtype = field$dtype,
    shape = field$shape,
    device = default_device("pjrt"),
    row_major = FALSE
  )
}

# The data type of the field `name` of a random state, which `as_dtype()` builds
# once: it takes longer than the rest of a read.
random_state_dtype <- function(name) {
  dtype <- globals$random_state_dtypes[[name]]
  if (is.null(dtype)) {
    dtype <- as_dtype(random_state_fields[[name]]$dtype)
    globals$random_state_dtypes[[name]] <- dtype
  }
  dtype
}

# The bytes of the 32-bit words `words` (whole R doubles in [0, 2^32)), each
# little-endian, as PJRT reads them on the host; `bytes_words()` reads them
# back.
words_bytes <- function(words) {
  as.raw(rep(words, each = 4L) %/% 256^(0:3) %% 256)
}

bytes_words <- function(bytes) {
  colSums(matrix(as.numeric(bytes), nrow = 4L) * 256^(0:3))
}

# The global random state as R holds it, outside of any program, which the
# first draw from it derives from base R's RNG.
global_random_state <- function() {
  if (is.null(globals$random_state)) {
    globals$random_state <- random_state_from_r()
  }
  globals$random_state
}

# The random state of the key words `key` (two whole R doubles in [0, 2^32),
# in the order of JAX's key data) and the counter `0`.
random_state_of_key <- function(key) {
  random_state_of_words(c(key, 0, 0))
}

# A fresh random state from two uniform draws of base R's RNG, each giving one
# 32-bit word of the key. A uniform draw of R's default generator has 32
# random bits.
random_state_from_r <- function() {
  random_state_of_key(floor(stats::runif(2L) * 2^32))
}

# The random state of the 32-bit integer `seed`: the key of JAX's `key(seed)`
# with `jax_enable_x64` off, JAX's default, the words 0 and `seed` read as an
# unsigned 32-bit integer.
random_state_from_seed <- function(seed) {
  random_state_of_key(c(0, seed %% 2^32))
}

#' @title Create a Random State
#' @description
#' Creates a random state of its own, which the samplers draw from in place of
#' the global random state (see [nv_set_seed()]) within [with_nv_rng_state()]
#' or after [local_nv_rng_state()]. Like the global random state, it is a key
#' and a counter of the ThreeFry-2x32 generator, and every sampling call that
#' draws from it advances it in place, so two draws from the same random state
#' differ. R holds the key and the counter, so advancing a random state runs no
#' program.
#'
#' A function compiled with [jit()] can draw from a random state it takes as an
#' argument, also inside a list, or one it closes over, which it makes the
#' samplers' default with [local_nv_rng_state()] or [with_nv_rng_state()]. The
#' compiled program takes the key and the counter of an argument like other R
#' data among the arguments, on the device of the call, and those of a random
#' state it closes over as it takes the global random state's, from the random
#' state it closed over when it was traced. After the call, the random state is
#' advanced in place by the number of sampling calls the program made from it,
#' as the global random state is. In the jitted function, a random state is an
#' `AnvlRngState` too, but it is not a value there, which could be returned
#' or carried through a loop. A jitted function cannot draw from the same
#' random state as two different ones, such as one passed to it twice, which
#' is an error, and cannot create one.
#' @param seed (`integer(1)`)\cr
#'   The seed, a 32-bit integer, which gives the same random state as
#'   [nv_set_seed()] gives the global random state.
#' @return (`AnvlRngState`)\cr
#'   A random state of the key that `seed` gives and the counter 0.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' rng <- nv_rng_state(42L)
#' with_nv_rng_state(rng, nv_runif(3L))
#' # the random state was advanced, so the next draw differs
#' with_nv_rng_state(rng, nv_runif(3L))
#' rng
#'
#' # a jitted function takes a random state as an argument
#' f <- jit(function(x, rng) with_nv_rng_state(rng, x + nv_rnorm(2L)))
#' f(nv_array(c(1, 2)), nv_rng_state(1L))
#' @export
nv_rng_state <- function(seed) {
  # one created while tracing would be closed over, and its draws carry on from
  # call to call where the same code run eagerly starts afresh
  if (currently_tracing()) {
    cli_abort(c(
      "A random state cannot be created in a function traced by {.fn jit}.",
      i = "Create it before calling the function, and pass it to the function or close over it."
    ))
  }
  abort_if_no_rng()
  random_state_from_seed(assert_int(seed, coerce = TRUE))
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
#' In a function traced by [jit()], the samplers in the scope draw from
#' `rng_state`, also in the bodies of loops, and the call advances it
#' afterwards: a random state the jitted function takes as an argument, or one
#' it closes over (see [nv_rng_state()]). It can be called in a branch of
#' [nv_if()] or a function [gradient()] differentiates, such as a jitted
#' function called there, but not in the body of [nv_while()] or [nv_scan()],
#' nor in a function traced inside one: called before the loop, it reaches the
#' loop body. These functions trace the functions they are given also when they
#' are called outside of [jit()].
#' @param rng_state (`AnvlRngState`)\cr
#'   The random state to draw from, as [nv_rng_state()] returns.
#' @param code (any)\cr
#'   An expression to evaluate drawing from `rng_state`.
#' @param envir (`environment`)\cr
#'   The scope whose exit restores the default from before.
#' @return `local_nv_rng_state()` returns `NULL`, invisibly; `with_nv_rng_state()`
#'   returns the result of evaluating `code`.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' rng <- nv_rng_state(1L)
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
  abort_if_no_rng()
  assert_random_state(rng_state, "rng_state")
  desc <- current_descriptor(silent = TRUE)
  if (!is.null(desc)) {
    # a loop body draws from the default set before the loop
    if (in_loop_body(desc)) {
      cli_abort(c(
        "{.fn local_nv_rng_state} cannot be used in the body of {.fn nv_while} or {.fn nv_scan}.",
        i = "Call it before the loop: the default it sets there reaches the loop body."
      ))
    }
    # the default of the trace under way and the traces opened in it, for the
    # calling scope
    old <- desc$default_random_state
    desc$default_random_state <- random_state_handle(rng_state)
    withr::defer(assign("default_random_state", old, envir = desc), envir = envir)
  } else {
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
