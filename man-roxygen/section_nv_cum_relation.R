#' @section Relation to base R:
#' `<%= cum_nv_name %>()` with `axis = NULL` and [base::<%= cum_base_fn %>()]
#' both flatten first, but in different orders -- anvl arrays are row-major (C
#' order), base R is column-major (Fortran) -- so for a multi-axis input the
#' two give different running values. They agree on 1-D inputs.
