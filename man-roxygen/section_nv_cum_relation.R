#' @section Relation to base R:
#' Both `<%= cum_nv_name %>()` (with `axis = NULL`) and [base::<%= cum_base_fn %>()]
#' flatten a multi-dimensional input to 1-D before accumulating, but the
#' flatten order differs: anvl arrays are row-major (C order), so the
#' flattened sequence iterates the last axis fastest, whereas base R uses
#' column-major (Fortran) order. The two agree on 1-D inputs.
