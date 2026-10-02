#' @include aaa.R
NULL

#' @title Set the Global RNG Seed
#' @description
#' Seeds the global RNG state, analogous to base R's [set.seed()].
#'
#' The samplers ([nv_runif()], [nv_rnorm()], [nv_rbinom()], [nv_sample_int()]
#' and [nv_sample()]) draw from the global state when they are called without a
#' `state`, and then return only the sample. A [jit()]-compiled function that
#' draws from it takes the global state as a hidden input and hands the
#' advanced state back as a hidden output, so every draw inside one call
#' advances the same state, and consecutive calls continue where the previous
#' one stopped.
#'
#' Until a seed is set, the global state is seeded from base R's RNG on its
#' first use, so [set.seed()] also makes it reproducible.
#'
#' The global state is also threaded through the branches of [nv_if()], the
#' bodies of [nv_while()] and [nv_scan()], and functions differentiated by
#' [gradient()], so a draw in a loop body gives a new sample on every
#' iteration. It cannot be drawn from in the condition of [nv_while()], nor in
#' the functions of a reduction, a scatter or a sort comparator; pass a `state`
#' explicitly there.
#'
#' @section Devices:
#' The global state lives on the device of the call that last drew from it,
#' and a call that runs on another device -- however that device is decided --
#' copies it there. A seed therefore gives the same sequence whatever devices
#' the draws run on.
#' @param seed (`integer(1)`)\cr
#'   The seed.
#' @return `NULL`, invisibly.
#' @family rng
#' @seealso [nv_rng_state()] to create an explicit RNG state.
#' @examplesIf pjrt::plugins_downloaded()
#' nv_set_seed(42L)
#' nv_runif(3L)
#' nv_runif(3L)
#'
#' # setting the seed again repeats the sequence
#' nv_set_seed(42L)
#' nv_runif(3L)
#'
#' # draws inside a jitted function share the global state
#' f <- jit(function() nv_rnorm(2L) + nv_runif(2L))
#' f()
#' @export
nv_set_seed <- function(seed) {
  globals$seed <- assert_int(seed, coerce = TRUE)
  globals$rng_state <- NULL
  invisible(NULL)
}

# The name of the hidden argument through which a jitted function receives the
# global RNG state.
RNG_STATE_ARG <- ".anvl_rng_state"

# The state a sampler draws from: `state` itself, or the global RNG state when
# it is `NULL`. Called in the sampler's traced body; `rng_state_out()` hands the
# result back.
#
# The global state is the one of the current trace: at the root of a jit call
# the hidden input `global_rng_fn()` registered, in a sub-graph the one its
# higher-order primitive threads into it (see `rng_thread()`). A root without
# one signals `anvl_global_rng_needed`, which the jit wrapper takes as its cue to
# call again with the state (see `jit()`); with it, a sub-graph that has none is
# one that cannot pass the state on.
rng_state_in <- function(state) {
  if (!is.null(state)) {
    return(state)
  }
  desc <- current_descriptor()
  if (!is.null(desc$rng_state)) {
    return(desc$rng_state)
  }
  stash <- globals[["DESCRIPTOR_STASH"]]
  root <- if (length(stash)) stash[[1L]] else desc
  if (is.null(root$rng_state)) {
    cli_abort(
      c(
        "The global RNG state can only be drawn from in a function called through {.fn jit}.",
        i = "Pass {.arg state} explicitly, e.g. one created by {.fn nv_rng_state}."
      ),
      class = "anvl_global_rng_needed"
    )
  }
  cli_abort(c(
    "The global RNG state cannot be drawn from here.",
    i = "It cannot be drawn from in the condition of {.fn nv_while}, nor in the function of a reduction, a scatter or a sort comparator.",
    i = "Pass {.arg state} explicitly, e.g. one created by {.fn nv_rng_state}."
  ))
}

# What a sampler returns: `list(state, values)` for an explicit `state`; for
# the global one (`state` is `NULL`), `new_state` becomes the global state of
# the trace and only `values` is returned.
rng_state_out <- function(state, new_state, values) {
  if (!is.null(state)) {
    return(list(state = new_state, values = values))
  }
  desc <- current_descriptor()
  desc$rng_state <- new_state
  values
}

# Wraps `f`, traced into a (sub-)graph, so that it starts from the global RNG
# state `state` and returns `list(out, state)`: its own output and the global
# state it leaves behind. `state` may be a box of an enclosing trace, which the
# sub-graph then closes over. `NULL` (the enclosing trace has no global state)
# leaves `f` as it is.
rng_thread <- function(f, state) {
  force(f)
  if (is.null(state)) {
    return(f)
  }
  function(...) {
    desc <- current_descriptor()
    desc$rng_state <- state
    out <- f(...)
    list(out, desc$rng_state)
  }
}

# Wraps `f` to take the global RNG state as the hidden argument
# `RNG_STATE_ARG` rather than pass it on to `f`, and to return `list(out,
# state)` like `rng_thread()`.
rng_arg_fn <- function(f) {
  force(f)
  function(...) {
    args <- list(...)
    state <- args[[RNG_STATE_ARG]]
    args[[RNG_STATE_ARG]] <- NULL
    do.call(rng_thread(f, state), args)
  }
}

# The function to trace for a jit call: `f`, taking the global RNG state as a
# hidden input and returning it after its output when the call `args` carry
# it.
global_rng_fn <- function(f, args) {
  if (RNG_STATE_ARG %in% names(args)) rng_arg_fn(f) else f
}

# Calls `run` (a backend's fast entry) with the global RNG state as the hidden
# argument, and stores the state the call returns as the new global state. The
# dispatcher copies the state to the call's device (see `dispatcher()`'s
# `follow`), so it is passed wherever it lives.
global_rng_dispatch <- function(run, args, backend) {
  args[[RNG_STATE_ARG]] <- global_rng_state(backend)
  out <- run(args)
  globals$rng_state <- out[[2L]]
  out[[1L]]
}

# The global RNG state as an array of `backend`, seeding it if it is not set.
global_rng_state <- function(backend) {
  state <- globals$rng_state
  if (is.null(state)) {
    seed <- globals$seed %||% sample.int(.Machine$integer.max, 1L)
    state <- nv_rng_state(seed)
  } else if (backend(state) != backend) {
    state <- nv_array(as_raw(state), dtype = dtype(state), shape = shape(state))
  }
  globals$rng_state <- state
  state
}

# `prim_while()`'s condition and body for a loop state that carries the global
# RNG state as `RNG_STATE_ARG`: the body draws from it and puts the state it
# leaves behind back into the loop state it returns; the condition drops it, so
# it cannot draw.
rng_while_body <- function(body) {
  # forced: the caller rebinds `body` to the wrapper
  force(body)
  function(...) {
    res <- rng_arg_fn(body)(...)
    c(res[[1L]], stats::setNames(list(res[[2L]]), RNG_STATE_ARG))
  }
}

rng_while_cond <- function(cond) {
  # forced: the caller rebinds `cond` to the wrapper
  force(cond)
  function(...) {
    args <- list(...)
    args[[RNG_STATE_ARG]] <- NULL
    do.call(cond, args)
  }
}
