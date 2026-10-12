## ---------------------------------------------------------------------------
## Shared by the three uniform specs. Not a spec itself (leading underscore).
##
## Why these two intervals, for every uniform function:
##
##   unit  [0, 1] is the default, and its width of exactly 1 makes every
##         division and every log(width) exact. That is the point of including
##         it -- it isolates everything *except* the arithmetic on the width --
##         but it also means a clean `unit` result says nothing at all about
##         the divisions, which is why it is never run alone.
##
##   wide  [-pi, 2*pi] is an ordinary interval a caller would really pass,
##         whose limits and width of 3*pi are representable in neither f32 nor
##         f64. It is what actually exercises the division and the logarithm.
##
## Note what is *not* here. Degenerate, reversed and infinite limits are
## combinatorial rather than numerical: they have a handful of distinct cases
## and no interesting interior, so they belong in the unit tests in
## tests/testthat/test-api-distributions.R, not in a 2^32-sample sweep.
##
## The limits are passed as bare R doubles. Earlier revisions of these drivers
## wrapped them in nv_scalar(dtype = "f64") to dodge a literal-precision bug
## that pinned every uniform sweep to a 5.9e-7 error floor. That bug is gone
## with the RData promotion framework -- a bare literal is now built into the
## graph at the dtype its use site needs -- so the workaround is removed, and
## these sweeps are what will hold it removed.
## ---------------------------------------------------------------------------

UNIF_INTERVALS <- list(
  unit = list(min = 0, max = 1),
  wide = list(min = -pi, max = 2 * pi)
)

## The JAX virtualenv at the root of the sibling checkouts. Anything already
## pinned by RETICULATE_PYTHON wins, so an environment that already carries jax
## is left alone.
jax_init <- local({
  done <- FALSE
  function() {
    if (done) {
      return(invisible(TRUE))
    }
    library(reticulate)
    venv <- normalizePath(file.path(here(), "..", "..", "..", "py-benchmarks", ".venv"), mustWork = FALSE)
    if (!nzchar(Sys.getenv("RETICULATE_PYTHON")) && dir.exists(venv)) {
      use_virtualenv(venv, required = TRUE)
    }
    ## JAX's 64-bit mode is off by default and must be set before any array
    ## exists, or every f64 cell below silently truncates to f32.
    import("jax")$config$update("jax_enable_x64", TRUE)
    done <<- TRUE
    invisible(TRUE)
  }
})

jax_dtype <- function(dtype) {
  jnp <- reticulate::import("jax.numpy", convert = FALSE)
  if (dtype == "f32") jnp$float32 else jnp$float64
}
