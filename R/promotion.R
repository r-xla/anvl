#' @title Type Promotion Rules
#' @description
#' Computes the common dtype for a set of abstract types, respecting whether a type is ambiguous or not.
#' A type is ambiguous if it comes from a literal (like 1 or 1.0) or was promoted
#' to an ambiguous type.
#' Promoting to an ambiguous type can happen in scenarios like `x + 1.2`, where `x` is a bool or an int.
#'
#' @param lhs_dtype ([`tengen::DataType`])\cr
#'   The left-hand side type.
#' @param rhs_dtype ([`tengen::DataType`])\cr
#'   The right-hand side type.
#' @param lhs_ambiguous (`logical(1)`)\cr
#'   Whether the left-hand side type is ambiguous.
#' @param rhs_ambiguous (`logical(1)`)\cr
#'   Whether the right-hand side type is ambiguous.
#' @return (`list(dtype = [`tengen::DataType`], ambiguous = `logical(1)`)\cr
#' @export
common_dtype <- function(lhs_dtype, rhs_dtype, lhs_ambiguous = FALSE, rhs_ambiguous = FALSE) {
  lhs_dtype <- as_dtype(lhs_dtype)
  rhs_dtype <- as_dtype(rhs_dtype)
  # because ambiguous types can't be unsigned types, we can just use the normal promotion rules
  if (lhs_ambiguous && rhs_ambiguous) {
    dt <- promote_dt_ambiguous(lhs_dtype, rhs_dtype)
    return(list(dtype = dt, ambiguous = TRUE))
  } else if (lhs_ambiguous) {
    dt <- promote_dt_ambiguous_to_known(lhs_dtype, rhs_dtype)
    return(list(dtype = dt, ambiguous = (dt == lhs_dtype) && (dt != rhs_dtype)))
  } else if (rhs_ambiguous) {
    dt <- promote_dt_ambiguous_to_known(rhs_dtype, lhs_dtype)
    return(list(dtype = dt, ambiguous = (dt == rhs_dtype) && (dt != lhs_dtype)))
  } else {
    dt <- promote_dt_known(lhs_dtype, rhs_dtype)
    return(list(dtype = dt, ambiguous = FALSE))
  }
}

# Like common_dtype, but for multiple arguments and also determines whether the result type is ambiguous.
# For internal use
common_type_info <- function(...) {
  args <- list(...)
  if (length(args) == 0L) {
    cli_abort("No arguments provided")
  } else if (length(args) == 1L) {
    arg <- to_abstract(args[[1L]])
    return(list(dtype(arg), arg$ambiguous))
  }
  init <- to_abstract(args[[1L]])
  cdt <- dtype(init)
  cdt_ambiguous <- init$ambiguous
  for (arg in args[-1L]) {
    arg <- to_abstract(arg)
    out <- common_dtype(cdt, dtype(arg), cdt_ambiguous, arg$ambiguous)
    cdt <- out[[1L]]
    cdt_ambiguous <- out[[2L]]
  }
  list(dtype = cdt, ambiguous = cdt_ambiguous)
}


promote_dt_ambiguous <- function(adtype1, adtype2) {
  promote_dt_known(adtype1, adtype2)
}

promote_dt_ambiguous_to_known <- function(adtype, dtype) {
  # there are only two cases where we cast a known type to an amibugous type:
  # 1. the ambiguous type is a float and the known type is not
  # 2. the known type is a bool but ambiguous type is not
  if (is_dtype_float(adtype) && !is_dtype_float(dtype)) {
    return(adtype)
  }
  if (!is_dtype_bool(adtype) && is_dtype_bool(dtype)) {
    return(adtype)
  }
  return(dtype)
}

promote_dt_known <- function(dt1, dt2) {
  if (dt1 == dt2) {
    return(dt1)
  }
  if (is_dtype_bool(dt1)) {
    return(dt2)
  }
  if (is_dtype_bool(dt2)) {
    return(dt1)
  }
  if (is_dtype_float(dt1)) {
    if (is_dtype_float(dt2)) {
      return(as_dtype(paste0("f", max(dtype_bits(dt1), dtype_bits(dt2)))))
    }
    # bools and integers are cast to the float
    return(dt1)
  }
  if (is_dtype_float(dt2)) {
    return(dt2)
  }
  if (is_dtype_int(dt1)) {
    if (is_dtype_int(dt2)) {
      return(as_dtype(paste0("i", max(dtype_bits(dt1), dtype_bits(dt2)))))
    }
    if (dtype_bits(dt2) < dtype_bits(dt1)) {
      # the int can hold the unsigned int
      return(dt1)
    }
    # int can't hold the unsigned int
    # we use signed int, but increase bits of unsigned int
    # this can lead to overflows then we have uint64 but this can't be avoided
    return(as_dtype(paste0("i", min(64L, dtype_bits(dt2) * 2L))))
  }
  if (is_dtype_int(dt2)) {
    if (is_dtype_uint(dt1)) {
      if (dtype_bits(dt2) > dtype_bits(dt1)) {
        return(dt2)
      }
      return(as_dtype(paste0("i", min(64L, dtype_bits(dt1) * 2L))))
    }
    cli_abort("internal error")
  }
  # both are unsigned
  as_dtype(paste0("ui", max(dtype_bits(dt1), dtype_bits(dt2))))
}

default_dtype <- function(x) {
  if (is.integer(x)) {
    as_dtype("i32")
  } else if (is.double(x)) {
    as_dtype("f32")
  } else if (is.logical(x)) {
    as_dtype("bool")
  } else {
    cli_abort("No default type for: {.class class(x)[1L]}")
  }
}

promotable_to <- function(from, to) {
  if (identical(from, to)) {
    return(TRUE)
  }
  dt <- common_dtype(from, to)
  if (dt$dtype != to) {
    return(FALSE)
  }
  TRUE
}
