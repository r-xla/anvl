#' @section Order of prints:
#' Prints are not ordered by default: the compiler may run the prints of a
#' program in any order that respects the data they print, so their output may
#' come in a different order than the prints in the code. With `ordered = TRUE`
#' a print is ordered with respect to every other ordered print of the program:
#' each takes a token from the one before it and hands it on to the next, so
#' they run in the order of the code, iteration by iteration in a loop.
#' Prints in different calls are always printed in the order of the calls.
