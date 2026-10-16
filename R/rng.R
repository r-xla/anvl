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
#' call returns, the global random state is advanced by the number of sampling
#' calls the program makes, which is known when the function is traced.
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
#' random state is an `AnvlRandomState`: plain R values, the key and the
#' counter, each a 64-bit unsigned integer stored as its two 32-bit words, so
#' [saveRDS()] can write it to a file. `NULL` is the random state that is not
#' set. The counter wraps around at \eqn{2^{64}}, where its draws repeat, so
#' restore a random state that `nv_get_random_state()` returned rather than one
#' made up.
#'
#' @section Devices:
#' The global random state is held by R and copied to the device of each call
#' that draws from it, and the generator is built from integer arithmetic, so a
#' seed gives the same random bits whatever devices the draws run on. The
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

#' @rdname nv_set_seed
#' @export
nv_get_random_state <- function() {
  abort_if_tracing("read")
  abort_if_no_rng()
  words <- globals$random_state
  if (!is.null(words)) structure(words, class = "AnvlRandomState")
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
  hex <- function(lo, hi) paste0("0x", format_hex32(hi), format_hex32(lo))
  sprintf("<AnvlRandomState> key %s, counter %s", hex(x[[1L]], x[[2L]]), hex(x[[3L]], x[[4L]]))
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

# `x` if it is a random state: an `AnvlRandomState` of four 32-bit words.
assert_random_state <- function(x) {
  if (!inherits(x, "AnvlRandomState")) {
    cli_abort(c(
      "{.arg random_state} must be a random state returned by {.fn nv_get_random_state}.",
      x = "Got {.cls {class(x)[[1L]]}}."
    ))
  }
  words <- unclass(x)
  if (!is.double(words) || length(words) != 4L || anyNA(words) || any(words < 0 | words >= 2^32 | words %% 1 != 0)) {
    cli_abort("{.arg random_state} must hold four 32-bit words.")
  }
  x
}

#' @rdname nv_set_seed
#' @export
local_nv_seed <- function(seed, envir = parent.frame()) {
  words <- globals$random_state
  nv_set_seed(seed)
  withr::defer(assign("random_state", words, envir = globals), envir = envir)
  invisible(NULL)
}

#' @rdname nv_set_seed
#' @export
with_nv_seed <- function(seed, code) {
  local_nv_seed(seed)
  code
}

# A fresh random state from two uniform draws of base R's RNG, each giving one
# 32-bit word of the key, and the counter `0`. A uniform draw of R's default
# generator has 32 random bits.
random_state_from_r <- function() {
  c(floor(stats::runif(2L) * 2^32), 0, 0)
}

# The random state of the integer `seed`: the key `seed` read as an unsigned
# 32-bit integer, and the counter `0`.
random_state_from_seed <- function(seed) {
  c(seed %% 2^32, 0, 0, 0)
}

# The random state `words` with its counter advanced by `n` steps, wrapping
# around at 2^64.
random_state_advanced <- function(words, n) {
  lo <- words[[3L]] + n %% 2^32
  c(words[[1L]], words[[2L]], lo %% 2^32, (words[[4L]] + n %/% 2^32 + lo %/% 2^32) %% 2^32)
}

# The random state `box`, a `ui32[4]` GraphBox, with its counter advanced by `n`
# steps (below 2^53), traced into the current trace.
random_state_advance_box <- function(box, n) {
  words <- nv_convert(nv_static_slice(box, 3L, 4L, 1L), dtype = "ui64")
  counter <- nv_or(
    nv_static_slice(words, 1L, 1L, 1L),
    nv_shift_left(nv_static_slice(words, 2L, 2L, 1L), 32L)
  )
  # `n` enters as two R integers, its low 30 bits and the rest
  high <- nv_shift_left(nv_fill(as.integer(n %/% 2^30), 1L, dtype = "ui64"), 30L)
  counter <- counter + as.integer(n %% 2^30) + high
  # the low word is cleared of the high one first, so that the conversion is
  # exact
  nv_concatenate(
    nv_static_slice(box, 1L, 2L, 1L),
    nv_convert(nv_shift_right_logical(nv_shift_left(counter, 32L), 32L), dtype = "ui32"),
    nv_convert(nv_shift_right_logical(counter, 32L), dtype = "ui32")
  )
}

# The global random state as a side effect, which `prim_sample_bits()` and
# `prim_fork_random_state()` use. Its value lives in `globals$random_state`, an
# R double vector of the four 32-bit words of the key and the counter, low word
# first; reading it while it is not set derives it from base R's RNG. A program
# takes it from the host as a `ui32[4]` input. It is counted, so that the host
# advances it by the steps a program advanced it, and no program returns it.
register_side_effect(
  "random_state",
  aval = AbstractArray(dtype = "ui32", shape = 4L),
  get = function() {
    if (is.null(globals$random_state)) {
      globals$random_state <- random_state_from_r()
    }
    globals$random_state
  },
  counted = list(
    advance_box = random_state_advance_box,
    fork = fork_random_state,
    advance = function(n) {
      globals$random_state <- random_state_advanced(globals$random_state, n)
    }
  )
)
