#' @title Type Promotion Rules
#' @description
#' Computes the common data type of two data types, following the rules
#' described in `vignette("type-promotion")`.
#'
#' R values entering a program have no data type of their own (see
#' [`RDataArray`]) and are not promoted with this: they *take* the dtype of
#' whatever they are combined with, and commit to a default only when nothing
#' claims them.
#'
#' @param lhs_dtype ([`tengen::DataType`])\cr
#'   The left-hand side type.
#' @param rhs_dtype ([`tengen::DataType`])\cr
#'   The right-hand side type.
#' @return ([`tengen::DataType`])
#' @examples
#' common_dtype("i32", "f32")
#' common_dtype("i32", "i64")
#' @export
common_dtype <- function(lhs_dtype, rhs_dtype) {
  promote_dt_known(as_dtype(lhs_dtype), as_dtype(rhs_dtype))
}

# The common dtype of several arrayish values, the one every operand of an
# operation is brought to. An R value yields: it takes the dtype of the values
# it meets, and contributes only the dtype it would commit to when it meets
# nothing but other R values.
# For internal use.
common_type_info <- function(...) {
  args <- list(...)
  if (length(args) == 0L) {
    cli_abort("No arguments provided")
  }
  cdt <- NULL
  cdt_is_rdata <- TRUE
  for (arg in args) {
    aval <- to_abstract(arg)
    is_rdata <- is_rdata_array(aval)
    dt <- if (is_rdata) aval$default_dtype else aval$dtype
    if (is.null(cdt)) {
      cdt <- dt
      cdt_is_rdata <- is_rdata
      next
    }
    if (cdt_is_rdata && is_rdata) {
      cdt <- promote_dt_known(cdt, dt)
    } else if (cdt_is_rdata) {
      cdt <- promote_dt_rdata(cdt, dt)
      cdt_is_rdata <- FALSE
    } else if (is_rdata) {
      cdt <- promote_dt_rdata(dt, cdt)
    } else {
      cdt <- promote_dt_known(cdt, dt)
    }
  }
  cdt
}


# What an R value becomes when it meets a value that has a dtype. It yields --
# an R double meeting an `f16` array becomes `f16`, not the default float --
# except where the dtype cannot hold it: a value from a float R type meeting an
# integer dtype stays a float, and anything meeting `bool` stays itself.
# `rdtype` is the dtype the R value would commit to on its own.
promote_dt_rdata <- function(rdtype, dtype) {
  if (is_dtype_float(rdtype) && !is_dtype_float(dtype)) {
    return(rdtype)
  }
  if (!is_dtype_bool(rdtype) && is_dtype_bool(dtype)) {
    return(rdtype)
  }
  dtype
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
      return(as_dtype(paste0("f", max(dtype_width(dt1), dtype_width(dt2)))))
    }
    # bools and integers are cast to the float
    return(dt1)
  }
  if (is_dtype_float(dt2)) {
    return(dt2)
  }
  if (is_dtype_int(dt1)) {
    if (is_dtype_int(dt2)) {
      return(as_dtype(paste0("i", max(dtype_width(dt1), dtype_width(dt2)))))
    }
    if (dtype_width(dt2) < dtype_width(dt1)) {
      # the int can hold the unsigned int
      return(dt1)
    }
    # int can't hold the unsigned int
    # we use signed int, but increase bits of unsigned int
    # this can lead to overflows then we have uint64 but this can't be avoided
    return(as_dtype(paste0("i", min(64L, dtype_width(dt2) * 2L))))
  }
  if (is_dtype_int(dt2)) {
    if (is_dtype_uint(dt1)) {
      if (dtype_width(dt2) > dtype_width(dt1)) {
        return(dt2)
      }
      return(as_dtype(paste0("i", min(64L, dtype_width(dt1) * 2L))))
    }
    cli_abort("internal error")
  }
  # both are unsigned
  as_dtype(paste0("ui", max(dtype_width(dt1), dtype_width(dt2))))
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

# The dtype an R value of this storage type commits to when nothing in the
# program tells it what it is. The single place that decision is made.
default_dtype_r <- function(r_type) {
  switch(
    r_type,
    double = as_dtype("f32"),
    integer = as_dtype("i32"),
    logical = as_dtype("bool"),
    cli_abort("No default type for R type {.val {r_type}}")
  )
}

promotable_to <- function(from, to) {
  if (identical(from, to)) {
    return(TRUE)
  }
  common_dtype(from, to) == to
}
