#' @include aaa.R
#' @include side-effects.R
NULL

# The random state: the user API, and a random state as R holds it. How a trace
# draws from one is in R/rng-trace.R.

#' @title Set the Global Random State
#' @description
#' `nv_set_seed()` seeds the global random state, like base R's [set.seed()].
#' Until it is set, its first use derives it from base R's RNG.
#' `nv_get_rng_state()` and `nv_set_rng_state()` save and restore it, and
#' `local_nv_seed()` and `with_nv_seed()` seed it for a scope.
#'
#' See `r roxy_article("random-numbers")` for more details.
#' @param seed (`integer(1)`)\cr
#'   The seed, a 32-bit integer.
#' @param rng_state (`AnvlRngState`)\cr
#'   A random state, as `nv_get_rng_state()` returns.
#' @param code (any)\cr
#'   An expression to evaluate with the seeded global random state.
#' @param envir (`environment`)\cr
#'   The scope whose exit restores the global random state.
#' @return `nv_set_seed()`, `local_nv_seed()` and `nv_set_rng_state()`
#'   return `NULL`, invisibly; `with_nv_seed()` returns the result of
#'   evaluating `code`. `nv_get_rng_state()` returns a copy of the global
#'   random state, an `AnvlRngState`.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' nv_set_seed(42L)
#' nv_runif(3L)
#' nv_set_seed(42L)
#' nv_runif(3L)
#'
#' # save the random state and continue from it later
#' rng_state <- nv_get_rng_state()
#' nv_runif(2L)
#' nv_set_rng_state(rng_state)
#' nv_runif(2L)
#'
#' with_nv_seed(1L, nv_runif(2L))
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
  rng_state_copy(global_rng_state())
}

#' @rdname nv_set_seed
#' @export
nv_set_rng_state <- function(rng_state) {
  abort_if_tracing("set")
  abort_if_no_rng()
  globals$rng_state <- rng_state_copy(assert_rng_state(rng_state))
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
  if (is_traced_rng_state(x)) {
    return("<AnvlRngState> of a traced function")
  }
  # a 64-bit value of its high and low 32-bit word
  hex <- function(high, low) paste0("0x", format_hex32(high), format_hex32(low))
  w <- rng_state_words(x)
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
# in a trace also one of the trace's key and counter (see `AnvlRngState()`).
assert_rng_state <- function(x, arg = "rng_state") {
  if (!is_rng_state(x)) {
    cli_abort("{.arg {arg}} must be a random state, as {.fn nv_rng_state} or {.fn nv_get_rng_state} return.")
  }
  if (is_traced_rng_state(x) && !currently_tracing()) {
    cli_abort("{.arg {arg}} is the random state of a traced function, which cannot be used after its trace.")
  }
  x
}

is_rng_state <- function(x) {
  inherits(x, "AnvlRngState")
}

# ---- A random state -----------------------------------------------------------

# The random state of the key `key` and the counter `counter`. A trace and a
# jitted function see them as a `ui32` array of shape 2 and a `ui64` scalar,
# which they are in a trace: boxes of it. R holds them as their 32-bit words
# (whole R doubles in [0, 2^32)), each an array of shape 2: the key's in the
# order of JAX's key data, the counter's low and high one. A jitted function
# takes those words as bare R data, which the call uploads to its device, and
# its trace receives them as the key and the counter (see
# `received_rng_state()`). It is an environment, so that the draws from it
# advance it in place, and R holds its words, so that advancing it runs no
# program and reading it waits for none.
AnvlRngState <- function(key, counter) {
  if (is_rdata_box(key)) {
    return(received_rng_state(key, counter))
  }
  x <- new.env(parent = emptyenv())
  x$key <- key
  x$counter <- counter
  structure(x, class = "AnvlRngState")
}

# A random state R holds, of the same key and counter as `x`, which the draws
# from `x` leave intact.
rng_state_copy <- function(x) {
  AnvlRngState(x$key, x$counter)
}

# The 32-bit words of the random state `x`, which R holds: the key's, then the
# counter's low and high one.
rng_state_words <- function(x) {
  c(as.vector(x$key), as.vector(x$counter))
}

# The random state of the 32-bit words `words` (see `rng_state_words()`).
rng_state_of_words <- function(words) {
  AnvlRngState(array(words[1:2], dim = 2L), array(words[3:4], dim = 2L))
}

# The key and the counter of a random state, by name: their data type and shape
# in a trace, and the label of their node in a printed graph.
rng_state_fields <- list(
  key = list(dtype = "ui32", shape = 2L, label = "key"),
  counter = list(dtype = "ui64", shape = integer(), label = "counter")
)

# The random state of the arrays `key` (`ui32`, shape 2) and `counter` (`ui64`
# scalar), e.g. the ones `prim_fork_rng_state()` returns.
rng_state_of_arrays <- function(key, counter) {
  rng_state_of_words(bytes_words(c(as_raw(key), as_raw(counter))))
}

# The buffer of the field `name` of the random state `x`, which R holds the
# words of, on the default device, from their bytes: a `ui64` reads the first of
# two words as the low one.
rng_state_buffer <- function(x, name) {
  field <- rng_state_fields[[name]]
  pjrt_buffer(
    words_bytes(as.vector(x[[name]])),
    dtype = field$dtype,
    shape = field$shape,
    device = default_device("pjrt"),
    row_major = FALSE
  )
}

# The data type of the field `name` of a random state, which `as_dtype()` builds
# once: it takes longer than the rest of a read.
rng_state_dtype <- function(name) {
  dtype <- globals$rng_state_dtypes[[name]]
  if (is.null(dtype)) {
    dtype <- as_dtype(rng_state_fields[[name]]$dtype)
    globals$rng_state_dtypes[[name]] <- dtype
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

# The global random state as R holds it, outside of any program, which its
# first use derives from base R's RNG.
global_rng_state <- function() {
  if (is.null(globals$rng_state)) {
    globals$rng_state <- rng_state_from_r()
  }
  globals$rng_state
}

# The random state of the key words `key` (two whole R doubles in [0, 2^32),
# in the order of JAX's key data) and the counter `0`.
rng_state_of_key <- function(key) {
  rng_state_of_words(c(key, 0, 0))
}

# A fresh random state from two uniform draws of base R's RNG, each giving one
# 32-bit word of the key. A uniform draw of R's default generator has 32
# random bits.
rng_state_from_r <- function() {
  rng_state_of_key(floor(stats::runif(2L) * 2^32))
}

# The random state of the 32-bit integer `seed`
rng_state_from_seed <- function(seed) {
  rng_state_of_key(c(0, seed %% 2^32))
}

#' @title Create a Random State
#' @description
#' Creates a random state of its own, which the samplers draw from in place of
#' the global random state within [with_nv_rng_state()] or after
#' [local_nv_rng_state()]. Draws advance it in place. A function compiled with
#' [jit()] can draw from a random state it takes as an argument or closes over,
#' but cannot create or return one.
#'
#' See `r roxy_article("random-numbers")` for more details.
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
  rng_state_from_seed(assert_int(seed, coerce = TRUE))
}

#' @title Draw From a Random State for a Scope
#' @description
#' `local_nv_rng_state()` and `with_nv_rng_state()` make `rng_state` the random
#' state the samplers draw from, for the calling scope or for `code` only, and
#' restore the one from before afterwards. The draws advance `rng_state` in
#' place. In a function traced by [jit()], they cannot be called in the body of
#' [nv_while()] or [nv_scan()]: called before the loop, they reach its body.
#'
#' See `r roxy_article("random-numbers")` for more details.
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
  assert_rng_state(rng_state, "rng_state")
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
    old <- desc$default_rng_state
    desc$default_rng_state <- rng_state_handle(rng_state)
    withr::defer(assign("default_rng_state", old, envir = desc), envir = envir)
  } else {
    old <- globals$rng_state
    globals$rng_state <- rng_state
    withr::defer(assign("rng_state", old, envir = globals), envir = envir)
  }
  invisible(NULL)
}

#' @rdname local_nv_rng_state
#' @export
with_nv_rng_state <- function(rng_state, code) {
  local_nv_rng_state(rng_state)
  code
}
