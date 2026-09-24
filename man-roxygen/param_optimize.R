#' @param optimize (`logical(1)` | `character()`)\cr
#'   Which graph optimization passes to run on the traced graph before returning
#'   it. `TRUE` runs all passes, `FALSE` (default) runs none, and a character
#'   vector selects a subset by name. The available passes are:
#'   * `"inline_scalars"`: replace scalar-shaped constants with inline literals.
#'   * `"remove_unused_constants"`: drop constants not referenced by the graph.
#'
#'   [`jit()`] always traces with all passes enabled.
