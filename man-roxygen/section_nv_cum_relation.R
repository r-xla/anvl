#' @section Relation to base R:
#' [base::<%= cum_base_fn %>()] flattens a multi-axis array before
#' accumulating, while `<%= cum_nv_name %>()` (and `<%= cum_base_fn %>()` on
#' an anvl array) accumulates along a single axis, the last one by default.
#' Flatten with [nv_flatten()] first if you want a single running sequence.
#' The two orders still differ: anvl arrays are row-major (C order), so the
#' flattened sequence iterates the last axis fastest, whereas base R uses
#' column-major (Fortran) order. They agree on 1-D inputs.
