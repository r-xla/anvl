#' @section Order of prints:
#' Prints are not ordered by default: the compiler may run the prints of a
#' program in any order that respects the data they print, so their output may
#' come in a different order than the prints in the code. With `ordered = TRUE`
#' a print is ordered with respect to every other ordered print of the program.
#' Note that this can have some performance implications due to restricting
#' the order of execution.
