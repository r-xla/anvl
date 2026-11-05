#' @param init (`named list()`)\cr
#'   Named list of initial state values, a tree in the sense of pjrt's
#'   [`RTree`][pjrt::build_tree]. Each leaf becomes a parameter of the loop's
#'   sub-graphs. An R value is carried at the data type `body` gives it back
#'   at, or else at the narrowest that holds every one its uses ask for, as
#'   described in `r roxy_article("type-promotion")`.
