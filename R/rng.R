#' @include aaa.R
#' @include side-effects.R
NULL

#' @title Set the Global Random State
#' @description
#' Seeds the global random state, analogous to base R's [set.seed()]. Calling
#' it is optional: until the global random state is first drawn from, it is not
#' set, and the first draw derives it from base R's RNG, so a [set.seed()]
#' before that makes the draws reproducible too.
#'
#' All random draws in anvl -- the samplers [nv_runif()], [nv_rnorm()],
#' [nv_rbinom()], [nv_sample_int()] and [nv_sample()], and the primitive
#' [prim_sample_bits()] they build on -- draw from the global random state. It
#' is a key and a counter of the ThreeFry-2x32 generator. Every call of
#' [prim_sample_bits()], a *sampling call*, splits it once: it draws from a
#' fresh key derived from the key and the counter, and advances the counter by
#' one, however many values it drew. A sampler advances the counter by its
#' number of sampling calls: [nv_rnorm()] by two, the others by one.
#'
#' A [jit()]-compiled function makes the same sampling calls as the same code
#' run without [jit()], in the same order, so it draws the same values: the
#' compiled program takes the global random state as an input, and after the
#' call the global random state is advanced by the number of sampling calls the
#' program makes, which is known when the function is traced.
#'
#' A draw in a loop body gives a new sample on every iteration. How [nv_scan()],
#' [nv_if()] and [nv_while()] advance the global random state, so that the
#' number of sampling calls of a program is known when it is traced, is
#' explained in the vignette "Random Numbers". Sampling is not possible in the
#' condition of [nv_while()] or in the functions of a reduction or a scatter.
#'
#' @section Seeding from base R:
#' When the global random state is not set, the next draw derives it from base
#' R's RNG, consuming two of its uniform draws. This happens only once: after
#' that, the global random state is advanced by anvl's draws alone, and
#' **calling [set.seed()] again does not reseed it** -- call `nv_set_seed()` for
#' that. `nv_set_seed(NULL)` unsets the global random state, so that the next
#' draw derives it from base R's RNG again.
#'
#' `local_nv_seed()` and `with_nv_seed()` seed the global random state for the
#' calling scope or for `code` only, like [withr::local_seed()] and
#' [withr::with_seed()] do for base R's RNG: the global random state from
#' before is restored afterwards.
#'
#' The global random state is seeded when R runs the call, so these functions
#' cannot be called in a function traced by [jit()], where they would seed it
#' once, while tracing, rather than on every call.
#'
#' @section Saving and restoring the random state:
#' A seed restarts a sequence from its beginning. To continue one where it
#' stopped, e.g. when resuming a model fit, save the random state itself:
#' `nv_get_random_state()` returns the global random state, and
#' `nv_set_random_state()` makes a saved one the global random state again. A
#' random state is an `AnvlRandomState`, a list of two arrays: `random_key`, the
#' key, a `"ui32"` array of its two 32-bit words as JAX holds them, and
#' `random_counter`, the counter, a `"ui64"` scalar. [nv_save()] can write it to
#' a file and [nv_read()] read it back. `NULL` is the random state that is not
#' set. The counter wraps around at \eqn{2^{64}}, where its draws repeat, so
#' restore a random state that `nv_get_random_state()` returned rather than one
#' made up.
#'
#' @section Devices:
#' The global random state is held in arrays, which a call that draws from it
#' takes as inputs (copied to its device if they are on another one). The generator is built from integer
#' arithmetic, so a seed gives the same random bits whatever devices the draws
#' run on. The
#' samples computed from them, e.g. by [nv_rnorm()], can differ in the last
#' digits between platforms, as their floating point functions do.
#' @param seed (`integer(1)` | `NULL`)\cr
#'   The seed, a 32-bit integer, or `NULL` to derive the global random state
#'   from base R's RNG.
#' @param random_state (`AnvlRandomState` | `NULL`)\cr
#'   A random state returned by `nv_get_random_state()`.
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
#' nv_set_seed(NULL)
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
  abort_if_no_rng()
  globals$random_state <- if (!is.null(seed)) random_state_from_seed(assert_int(seed, coerce = TRUE))
  invisible(NULL)
}
# REVIEW: Unify the *_seed and *_random_state functions (?)
# ANSWER: I think so: a saved random state is just a seed that continues a
# sequence instead of starting one. I would let nv_set_seed(), local_nv_seed()
# and with_nv_seed() take an integer seed, an `AnvlRandomState` or `NULL`,
# drop nv_set_random_state(), and keep nv_get_random_state() to save the
# state (the counterpart of reading `.Random.seed` in base R).

#' @rdname nv_set_seed
#' @export
nv_get_random_state <- function() {
  abort_if_tracing("read")
  abort_if_no_rng()
  state <- globals$random_state
  if (!is.null(state)) structure(state, class = "AnvlRandomState")
}

#' @rdname nv_set_seed
#' @export
nv_set_random_state <- function(random_state) {
  abort_if_tracing("set")
  abort_if_no_rng()
  globals$random_state <- if (!is.null(random_state)) unclass(assert_random_state(random_state))
  invisible(NULL)
}

#' @export
format.AnvlRandomState <- function(x, ...) {
  # a 64-bit value of its high and low 32-bit word
  hex <- function(high, low) paste0("0x", format_hex32(high), format_hex32(low))
  # the key's first word is its high word, as in JAX; a bitcast counter gives
  # its low word first
  key <- as.numeric(as_array(x$random_key))
  counter <- as.numeric(as_array(nv_bitcast_convert(x$random_counter, dtype = "ui32")))
  sprintf("<AnvlRandomState> key %s, counter %s", hex(key[[1L]], key[[2L]]), hex(counter[[2L]], counter[[1L]]))
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

# `x` if it is a random state: an `AnvlRandomState`, or a list like it of a
# `ui32` key of shape 2 and a `ui64` scalar counter (as `nv_read()` gives one
# back that `nv_save()` saved).
assert_random_state <- function(x) {
  is_array_of <- function(a, dtype, shape) is_anvl_array(a) && dtype(a) == dtype && identical(shape(a), shape)
  if (
    !is.list(x) ||
      !setequal(names(x), c("random_key", "random_counter")) ||
      !is_array_of(x$random_key, "ui32", 2L) ||
      !is_array_of(x$random_counter, "ui64", integer())
  ) {
    cli_abort(c(
      "{.arg random_state} must be a random state returned by {.fn nv_get_random_state}.",
      i = "A random state is a {.cls ui32} key of shape 2 and a {.cls ui64} scalar counter."
    ))
  }
  structure(x[c("random_key", "random_counter")], class = "AnvlRandomState")
}

#' @rdname nv_set_seed
#' @export
local_nv_seed <- function(seed, envir = parent.frame()) {
  state <- globals$random_state
  nv_set_seed(seed)
  withr::defer(assign("random_state", state, envir = globals), envir = envir)
  invisible(NULL)
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
  list(random_key = nv_array(key, dtype = "ui32"), random_counter = nv_scalar(0L, dtype = "ui64"))
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

# The values `values` of the random state (see its registration below), its
# key and counter, with the counter advanced by `n` steps (below 2^53), traced
# into the current trace.
random_state_at <- function(values, n) {
  if (n == 0) {
    return(values)
  }
  list(random_key = values$random_key, random_counter = counter_plus(values$random_counter, n))
}

# The `ui64` counter `counter` plus `n` (a whole R double below 2^53), wrapping
# around at 2^64. `n` enters as R integers: its low 31 bits, and the rest
# shifted into place.
counter_plus <- function(counter, n) {
  counter <- counter + as.integer(n %% 2^31)
  if (n >= 2^31) {
    counter <- counter + nv_shift_left(nv_fill(as.integer(n %/% 2^31), integer(), dtype = "ui64"), 31L)
  }
  counter
}

# The global random state's counter after `n` uses, computed by anvl.
counter_advanced <- jit(counter_plus, static = "n")

# The global random state as a side effect, which `prim_sample_bits()` and
# `prim_fork_random_state()` use. It lives in `globals$random_state`, its key,
# a `ui32[2]` array (the words of JAX's key data), and its counter, a `ui64` scalar array;
# reading it while it is not set derives it from base R's RNG. A program takes
# both as inputs and returns neither: after a run, the counter is advanced by
# the uses the program made.
register_side_effect(
  "random_state",
  avals = list(
    random_key = AbstractArray(dtype = "ui32", shape = 2L),
    random_counter = AbstractArray(dtype = "ui64", shape = integer())
  ),
  get = function() {
    if (is.null(globals$random_state)) {
      globals$random_state <- random_state_from_r()
    }
    unname(globals$random_state)
  },
  counted = list(at = random_state_at, fork = fork_random_state),
  after_run = function(values, uses) {
    globals$random_state$random_counter <- counter_advanced(globals$random_state$random_counter, uses)
  },
  labels = c("key", "counter")
)
