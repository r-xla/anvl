#' <% .kind <- if (grepl("^(TRUE|FALSE)$", trimws(ex_lhs))) "logical" else if (grepl("L$", trimws(ex_lhs))) "integer" else "double" %>
#' <% .default <- c(logical = "bool", integer = "i32", double = "f32")[[.kind]] %>
#' <% .other <- if (.kind == "double") "1L" else "1" %>
#' <% .other_kind <- if (.kind == "double") "an R integer" else "an R double" %>
#' <% .target <- if (grepl("^f", ex_dtype)) "a float" else if (ex_dtype == "bool") "a boolean" else "an integer" %>
#' @examplesIf pjrt::plugins_downloaded()
#' # two R values: both built at <%= .default %>, an R <%= .kind %>'s default data type
#' <%= ex_fn %>(<%= ex_lhs %>, <%= ex_rhs %>)
#'
#' # the R value is built at the array's data type instead
#' <%= ex_fn %>(<%= ex_lhs %>, nv_scalar(<%= ex_rhs %>, "<%= ex_dtype2 %>"))
#'
#' # <%= .other_kind %> cannot be built at <%= .target %> data type
#' try(<%= ex_fn %>(<%= .other %>, nv_scalar(<%= ex_rhs %>, "<%= ex_dtype %>")))
#'
#' # neither operand is converted to meet the other
#' try(<%= ex_fn %>(nv_scalar(<%= ex_lhs %>, "<%= ex_dtype %>"), nv_scalar(<%= ex_rhs %>, "<%= ex_dtype2 %>")))
