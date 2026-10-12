#' @include aaa.R
#' @include side-effects.R
#' @include backend-pjrt.R
NULL

#' @title Set the Global RNG Seed
#' @description
#' Seeds the global RNG state, analogous to base R's [set.seed()]. Calling it
#' is optional: until the global state is first drawn from, it is not set, and
#' the first draw derives it from base R's RNG, so a [set.seed()] before that
#' makes the draws reproducible too.
#'
#' All random draws in anvl -- the samplers [nv_runif()], [nv_rnorm()],
#' [nv_rbinom()], [nv_sample_int()] and [nv_sample()], and the primitive
#' [prim_sample_bits()] they build on -- draw from one global RNG state and
#' advance it. A [jit()]-compiled function that draws from it takes the global
#' state as a hidden input and hands the advanced state back as a hidden
#' output, so every draw inside one call advances the same state, and
#' consecutive calls continue where the previous one stopped. How the draws are
#' split into compiled functions does not matter: drawing in one jitted
#' function gives the same values as drawing one after the other outside of
#' [jit()].
#'
#' The global state is also threaded through the branches of [nv_if()], the
#' bodies of [nv_while()] and [nv_scan()], and functions differentiated by
#' [gradient()], so a draw in a loop body gives a new sample on every
#' iteration. It cannot be drawn from in the condition of [nv_while()], nor in
#' the functions of a reduction or a scatter, as these
#' cannot hand an advanced state back.
#'
#' @section Seeding from base R:
#' When the global state is not set, the next draw derives it from base R's
#' RNG, consuming two of its uniform draws. This happens only once: after that,
#' the global state is advanced by anvl's draws alone, and **calling
#' [set.seed()] again does not reseed it** -- call `nv_set_seed()` for that.
#' `nv_set_seed(NULL)` unsets the global state, so that the next draw derives
#' it from base R's RNG again.
#'
#' `local_nv_seed()` and `with_nv_seed()` seed the global state for the calling
#' scope or for `code` only, like [withr::local_seed()] and
#' [withr::with_seed()] do for base R's RNG: the global state from before is
#' restored afterwards.
#'
#' The global state is seeded when R runs the call, so these functions cannot be
#' called in a function traced by [jit()], where they would seed it once, while
#' tracing, rather than on every call.
#'
#' @section Saving and restoring the state:
#' A seed restarts a sequence from its beginning. To continue one where it
#' stopped, e.g. when resuming a model fit, save the state itself:
#' `nv_get_rng_state()` returns a copy of the global state, and
#' `nv_set_rng_state()` makes a copy of a saved state the global state again.
#' The state is a `ui64` [`AnvlArray`] of shape `2`, the key and the counter of
#' the generator (see [prim_sample_bits()]), so [nv_save()] can write it to a
#' file. `NULL` is the state that is not set. Both functions copy the state, as
#' the draws update the global state in place.
#'
#' @section Devices:
#' The global state lives on the device of the call that last drew from it,
#' and a call that runs on another device -- however that device is decided --
#' copies it there. A seed therefore gives the same sequence whatever devices
#' the draws run on.
#' @param seed (`integer(1)` | `NULL`)\cr
#'   The seed, or `NULL` to derive the global state from base R's RNG.
#' @param state ([`AnvlArray`] | `NULL`)\cr
#'   A state returned by `nv_get_rng_state()`.
#' @param code (any)\cr
#'   An expression to evaluate with the seeded global state.
#' @param envir (`environment`)\cr
#'   The scope whose exit restores the global state.
#' @return `nv_set_seed()`, `local_nv_seed()` and `nv_set_rng_state()` return
#'   `NULL`, invisibly; `with_nv_seed()` returns the result of evaluating
#'   `code`.
#'   `nv_get_rng_state()` returns a copy of the global state, a `ui64`
#'   [`AnvlArray`] of shape `2`, or `NULL` while it is not set.
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
#' # an unset state is derived from base R's RNG on the next draw
#' nv_set_seed(NULL)
#' set.seed(1)
#' nv_runif(3L)
#'
#' # draws inside a jitted function share the global state
#' f <- jit(function() nv_rnorm(2L) + nv_runif(2L))
#' f()
#'
#' # seed for one expression only
#' with_nv_seed(1L, nv_runif(2L))
#' with_nv_seed(1L, nv_runif(2L))
#'
#' # save the state and continue from it later
#' state <- nv_get_rng_state()
#' nv_runif(2L)
#' nv_set_rng_state(state)
#' nv_runif(2L)
#' @export
nv_set_seed <- function(seed) {
  abort_if_tracing("seeded")
  globals$rng_state <- if (!is.null(seed)) rng_state_from_seed(assert_int(seed, coerce = TRUE))
  invisible(NULL)
}

#' @rdname nv_set_seed
#' @export
nv_get_rng_state <- function() {
  abort_if_tracing("read")
  state <- globals$rng_state
  if (!is.null(state)) rng_state_copy(state)
}

#' @rdname nv_set_seed
#' @export
nv_set_rng_state <- function(state) {
  abort_if_tracing("set")
  globals$rng_state <- if (!is.null(state)) rng_state_copy(assert_rng_state(state))
  invisible(NULL)
}


abort_if_tracing <- function(what) {
  if (currently_tracing()) {
    cli_abort(c(
      "The global RNG state cannot be {what} in a function traced by {.fn jit}.",
      i = "Do it before calling the function."
    ))
  }
}

# `state` if it is an RNG state: a `ui64` array of shape `2`.
assert_rng_state <- function(state) {
  if (!is_anvl_array(state)) {
    cli_abort(c(
      "{.arg state} must be an RNG state returned by {.fn nv_get_rng_state}.",
      x = "Got {.cls {class(state)[[1L]]}}."
    ))
  }
  if (!identical(as.character(dtype(state)), "ui64") || !identical(shape(state), 2L)) {
    cli_abort(c(
      "{.arg state} must be an RNG state: a {.val ui64} array of shape {.val {2L}}.",
      x = "Got a {.val {as.character(dtype(state))}} array of shape {.val {shape(state)}}."
    ))
  }
  state
}

# A copy of the RNG state `state` that no draw invalidates: a draw updates the
# global state in place, consuming the array it held.
rng_state_copy <- jit(function(state) state)

#' @rdname nv_set_seed
#' @export
local_nv_seed <- function(seed, envir = parent.frame()) {
  state <- globals$rng_state
  nv_set_seed(seed)
  withr::defer(assign("rng_state", state, envir = globals), envir = envir)
  invisible(NULL)
}

#' @rdname nv_set_seed
#' @export
with_nv_seed <- function(seed, code) {
  local_nv_seed(seed)
  code
}

# The global RNG state as a side effect, which `prim_sample_bits()` uses. Its
# value lives in `globals$rng_state`; reading it while it is not set derives it
# from base R's RNG.
local({
  aval <- AbstractArray(dtype = "ui64", shape = 2L)
  getter <- function() {
    if (is.null(globals$rng_state)) {
      globals$rng_state <- rng_state_from_r()
    }
    globals$rng_state
  }
  setter <- function(value) {
    globals$rng_state <- value
  }
  register_side_effect("rng", aval, getter, setter, rules = list(pjrt = pjrt_state_slot(aval, getter, setter)))
})

# A fresh RNG state from two uniform draws of base R's RNG: a key of 53 bits,
# 32 from the first draw and 21 from the second, and the counter `0`. A
# uniform draw of R's default generator has 32 random bits.
rng_state_from_r <- function() {
  u <- stats::runif(2L)
  key <- floor(u[[1L]] * 2^32) + floor(u[[2L]] * 2^21) * 2^32
  nv_array(c(key, 0), dtype = "ui64")
}

# The RNG state of the integer `seed`, a `ui64[2]` array on `device`: the key,
# `seed` read as an unsigned 32-bit integer, and the counter `0`.
rng_state_from_seed <- function(seed, device = NULL) {
  nv_array(c(seed %% 2^32, 0), dtype = "ui64", device = device)
}
