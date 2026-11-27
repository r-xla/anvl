#' @section Random State:
#' The draws come from the active random state: the global random state (see
#' [nv_set_seed()]), or the random state that [with_nv_rng_state()] or
#' [local_nv_rng_state()] make the default for a scope. They advance it in
#' place.
