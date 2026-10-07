#' @section Order of prints:
#' A print is not ordered with respect to other prints. The value handed back
#' is `x` itself rather than a result of the print, so nothing depends on the
#' print, and its only input is `x`. Within one compiled program, XLA may
#' therefore run it at any point after `x` is computed, and two prints may
#' appear in an order other than the one in the code, even when one printed
#' value is computed from the other. A print is never dropped, as it is marked
#' as having a side effect.
#'
#' Ordering the prints would need a token threaded from one print to the next,
#' as JAX does for `jax.debug.print(..., ordered = True)`; anvl does not do that.
