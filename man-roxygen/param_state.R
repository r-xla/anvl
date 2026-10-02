#' @param state (`NULL` | [`arrayish`])\cr
#'   RNG state: a 1-D array of two `ui64` elements, as [nv_rng_state()]
#'   returns. The data type and length are fixed by the generator, not by the
#'   default data types, and the returned `state` has them too.
#'   The default (`NULL`) draws from the global RNG state instead (see
#'   [nv_set_seed()]) and returns only the sample.
