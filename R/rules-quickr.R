#' @keywords internal
NULL

# Helpers ----------------------------------------------------------------------

quickr_user_arg_names <- function(n) {
  n <- as.integer(n)
  if (!n) {
    return(character())
  }
  paste0("x", seq_len(n))
}

quickr_dtype_info <- function(dt_chr) {
  dt_chr <- as.character(dt_chr)

  # FIXME: hack
  if (dt_chr %in% c("f64", "f32")) {
    return(list(ctor = "double", zero = 0.0, scalar_cast = as.double))
  }
  if (dt_chr == "i32") {
    return(list(ctor = "integer", zero = 0L, scalar_cast = as.integer))
  }
  if (dt_chr %in% "bool") {
    return(list(ctor = "logical", zero = FALSE, scalar_cast = as.logical))
  }

  cli_abort(paste0(
    "Unsupported dtype for quickr lowering: {.val {dt_chr}}. ",
    "Supported dtypes are: {.val f64}, {.val i32}, {.val pred}."
  ))
}

quickr_dtype_to_r_ctor <- function(dt_chr) {
  quickr_dtype_info(dt_chr)$ctor
}

quickr_aval_type_call <- function(aval) {
  dt_chr <- as.character(dtype(aval))
  ctor <- quickr_dtype_to_r_ctor(dt_chr)
  sh <- shape(aval)
  if (length(sh) > 5L) {
    cli_abort("quickr lowering currently supports arrays up to rank 5")
  }
  dims <- if (!length(sh)) 1L else sh
  as.call(c(list(as.name(ctor)), as.list(dims)))
}

quickr_declare_stmt <- function(arg_names, arg_avals) {
  type_calls <- Map(
    function(nm, aval) {
      arg <- list(quickr_aval_type_call(aval))
      names(arg) <- nm
      as.call(c(list(as.name("type")), arg))
    },
    arg_names,
    arg_avals
  )
  as.call(c(list(as.name("declare")), type_calls))
}

quickr_base_has_declare <- function() {
  exists("declare", envir = baseenv(), inherits = FALSE)
}

quickr_scalar_cast <- function(value, dt_chr) {
  quickr_dtype_info(dt_chr)$scalar_cast(value)
}

quickr_zero_literal_for <- function(aval) {
  quickr_dtype_info(dtype(aval))$zero
}

quickr_emit_assign <- function(lhs_sym, rhs_expr) {
  list(rlang::call2("<-", lhs_sym, rhs_expr))
}

quickr_emit_full_like <- function(out_sym, value_expr, shape_out, out_aval) {
  if (!length(shape_out)) {
    return(quickr_emit_assign(out_sym, value_expr))
  }
  if (length(shape_out) > 5L) {
    cli_abort("constant/broadcast: only arrays up to rank 5 are currently supported")
  }

  quickr_emit_assign(
    out_sym,
    rlang::call2("array", value_expr, dim = as.integer(shape_out))
  )
}

quickr_emit_known_empty <- function(out_sym, out_aval) {
  shape_out <- as.integer(shape(out_aval))
  if (!length(shape_out) || !isTRUE(any(shape_out == 0L))) {
    return(NULL)
  }

  quickr_emit_full_like(out_sym, quickr_zero_literal_for(out_aval), shape_out, out_aval)
}

quickr_subscript <- function(expr, idxs, drop = NULL) {
  if (!length(idxs)) {
    return(expr)
  }
  if (!is.null(drop)) {
    idxs <- c(idxs, list(drop = drop))
  }
  as.call(c(list(as.name("[")), list(expr), idxs))
}

quickr_row_major_loop <- function(idxs, shp, inner) {
  body <- inner
  for (d in rev(seq_along(idxs))) {
    body <- as.call(list(
      as.name("for"),
      idxs[[d]],
      rlang::call2("seq_len", as.integer(shp[[d]])),
      body
    ))
  }
  body
}

quickr_alloc_zero <- function(shape, aval) {
  shape <- as.integer(shape)
  rank <- length(shape)
  if (!rank) {
    cli_abort("Internal error: quickr_alloc_zero() requires rank >= 1") # nocov
  }
  if (rank > 5L) {
    cli_abort("Internal error: quickr_alloc_zero() supports rank <= 5") # nocov
  }

  info <- quickr_dtype_info(dtype(aval))
  out_zero <- info$zero
  ctor <- info$ctor

  if (rank == 1L) {
    rlang::call2(ctor, shape[[1L]])
  } else if (rank == 2L) {
    rlang::call2("matrix", out_zero, nrow = shape[[1L]], ncol = shape[[2L]])
  } else {
    rlang::call2("array", out_zero, dim = shape)
  }
}

quickr_alloc_full <- function(shape, value_expr) {
  shape <- as.integer(shape)
  rank <- length(shape)
  if (!rank) {
    cli_abort("Internal error: quickr_alloc_full() requires rank >= 1") # nocov
  }
  if (rank > 5L) {
    cli_abort("Internal error: quickr_alloc_full() supports rank <= 5") # nocov
  }

  if (rank == 1L) {
    rlang::call2("array", value_expr, dim = shape[[1L]])
  } else if (rank == 2L) {
    rlang::call2("matrix", value_expr, nrow = shape[[1L]], ncol = shape[[2L]])
  } else {
    rlang::call2("array", value_expr, dim = shape)
  }
}

quickr_emit_convert <- function(out_sym, operand_expr, shape_in, in_aval, out_aval, ctx = NULL) {
  shape_in <- as.integer(shape_in)
  rank <- length(shape_in)
  dt_out <- as.character(dtype(out_aval))
  dt_in <- as.character(dtype(in_aval))

  # Ensure we fail early with a consistent "Unsupported dtype for quickr lowering"
  # error, and avoid duplicating dtype validation in each emitter branch.
  quickr_dtype_info(dt_out)

  # quickr's r2f rejects `NaN` / `Inf` as scalar literals (its
  # `is_scalar_atomic` check returns FALSE for them), so a literal-NaN
  # convert like `as.double(NaN)` won't compile. Construct these values at
  # runtime instead, from a non-literal zero — see `.quickr_emit_special_float`.
  if (rank == 0L && .is_special_float_literal(operand_expr)) {
    return(.quickr_emit_special_float(out_sym, operand_expr, dt_out, ctx))
  }

  cast_expr <- function(expr) {
    if (dt_out == "f64") {
      return(rlang::call2("as.double", expr))
    }

    if (dt_out == "i32") {
      if (dt_in == "i32") {
        return(expr)
      }
      # StableHLO convert truncates toward zero; as.integer() matches this.
      return(rlang::call2("as.integer", expr))
    }

    if (dt_out %in% "bool") {
      if (dt_in %in% "bool") {
        return(expr)
      }
      # StableHLO convert to pred behaves like "x != 0".
      return(rlang::call2("!=", expr, 0))
    }
  }

  if (rank == 0L) {
    return(quickr_emit_assign(out_sym, cast_expr(operand_expr)))
  }

  casted <- cast_expr(operand_expr)
  if (rank == 1L) {
    return(quickr_emit_assign(out_sym, casted))
  }
  quickr_emit_assign(out_sym, rlang::call2("array", casted, dim = shape_in))
}

.is_special_float_literal <- function(x) {
  is.numeric(x) && length(x) == 1L && (is.nan(x) || is.infinite(x))
}

# Emit two statements that bind `out_sym` to the runtime-computed NaN / ±Inf
# value (`0/0`, `1/0`, `-1/0`) via a fresh zero temp. quickr rejects NaN/Inf
# as scalar literals but compiles arithmetic on declared zero variables.
# The zero temp must be a `new_tmp_sym()` from ctx so quickr's symbol
# tracker recognizes it.
.quickr_emit_special_float <- function(out_sym, val, dt_out, ctx) {
  if (is.null(ctx) || is.null(ctx$new_tmp_sym)) {
    cli_abort("Internal error: NaN/Inf scalar emit requires ctx$new_tmp_sym")
  }
  ctor <- quickr_dtype_info(dt_out)$ctor # "double" or similar
  zero_sym <- ctx$new_tmp_sym()
  numerator <- if (is.nan(val)) {
    zero_sym
  } else if (val > 0) {
    1.0
  } else {
    -1.0
  }
  cast <- function(e) rlang::call2(if (ctor == "double") "as.double" else ctor, e)
  list(
    rlang::call2("<-", zero_sym, cast(0)),
    rlang::call2("<-", out_sym, rlang::call2("/", numerator, zero_sym))
  )
}

quickr_emit_truncating_i32 <- function(out_sym, operand_expr, shape_out) {
  shape_out <- as.integer(shape_out)
  rank <- length(shape_out)
  casted <- rlang::call2("as.integer", operand_expr)

  if (rank <= 1L) {
    return(quickr_emit_assign(out_sym, casted))
  }

  quickr_emit_assign(out_sym, rlang::call2("array", casted, dim = shape_out))
}

quickr_emit_select <- function(out_sym, pred_expr, true_expr, false_expr, pred_shape, out_aval) {
  pred_shape <- as.integer(pred_shape)
  out_shape <- as.integer(shape(out_aval))
  rank <- length(out_shape)

  if (!length(pred_shape)) {
    return(list(
      rlang::call2("<-", out_sym, false_expr),
      as.call(list(as.name("if"), pred_expr, rlang::call2("<-", out_sym, true_expr)))
    ))
  }

  if (!identical(pred_shape, out_shape)) {
    cli_abort("select: predicate shape must be scalar or match the output shape")
  }

  init <- quickr_emit_assign(out_sym, false_expr)
  if (rank == 0L || isTRUE(any(out_shape == 0L))) {
    return(init)
  }

  idx_syms <- lapply(seq_len(rank), function(d) as.name(paste0("i_", as.character(out_sym), "_", d)))
  pred_at <- quickr_subscript(pred_expr, idx_syms)
  out_at <- quickr_subscript(out_sym, idx_syms)
  true_at <- quickr_subscript(true_expr, idx_syms)
  assign_true <- rlang::call2("<-", out_at, true_at)
  body <- as.call(list(as.name("if"), pred_at, assign_true))

  c(init, list(quickr_row_major_loop(idx_syms, out_shape, body)))
}

quickr_stable_expm1_expr <- function(x_expr) {
  poly <- rlang::call2(
    "*",
    x_expr,
    rlang::call2(
      "+",
      1.0,
      rlang::call2(
        "*",
        x_expr,
        rlang::call2(
          "+",
          0.5,
          rlang::call2(
            "*",
            x_expr,
            rlang::call2(
              "+",
              1.0 / 6.0,
              rlang::call2(
                "*",
                x_expr,
                rlang::call2(
                  "+",
                  1.0 / 24.0,
                  rlang::call2("*", x_expr, 1.0 / 120.0)
                )
              )
            )
          )
        )
      )
    )
  )

  rlang::call2(
    "ifelse",
    rlang::call2("<", rlang::call2("abs", x_expr), 1e-5),
    poly,
    rlang::call2("-", rlang::call2("exp", x_expr), 1.0)
  )
}

quickr_stable_log1p_expr <- function(x_expr) {
  poly <- rlang::call2(
    "*",
    x_expr,
    rlang::call2(
      "+",
      1.0,
      rlang::call2(
        "*",
        x_expr,
        rlang::call2(
          "+",
          -0.5,
          rlang::call2(
            "*",
            x_expr,
            rlang::call2(
              "+",
              1.0 / 3.0,
              rlang::call2(
                "*",
                x_expr,
                rlang::call2(
                  "+",
                  -0.25,
                  rlang::call2("*", x_expr, 0.2)
                )
              )
            )
          )
        )
      )
    )
  )

  rlang::call2(
    "ifelse",
    rlang::call2("<", rlang::call2("abs", x_expr), 1e-5),
    poly,
    rlang::call2("log", rlang::call2("+", 1.0, x_expr))
  )
}

quickr_emit_iota <- function(out_sym, dim, start, shape_out, out_aval) {
  dim <- as.integer(dim)
  shape_out <- as.integer(shape_out)
  rank <- length(shape_out)
  dt_chr <- as.character(dtype(out_aval))

  if (!rank || rank > 5L) {
    cli_abort("iota: only arrays of rank 1..5 are supported by quickr lowering")
  }
  if (dim < 1L || dim > rank) {
    cli_abort("iota: invalid {.arg dim}: {dim}")
  }

  start_expr <- quickr_scalar_cast(start, dt_chr)

  if (rank == 1L) {
    idx <- rlang::call2("seq_len", shape_out[[1L]])
    expr <- rlang::call2("+", start_expr, rlang::call2("-", idx, 1L))
    return(quickr_emit_assign(out_sym, expr))
  }

  # Assign whole slices along `dim`, looping only over the other dimensions.
  n <- as.integer(shape_out[[dim]])
  idx <- rlang::call2("seq_len", n)
  vals <- rlang::call2("+", start_expr, rlang::call2("-", idx, 1L))

  aligned_shape <- rep.int(1L, rank)
  aligned_shape[[dim]] <- n
  aligned_sym <- as.name(paste0("aligned_", as.character(out_sym)))
  aligned_stmt <- rlang::call2("<-", aligned_sym, rlang::call2("array", vals, dim = as.integer(aligned_shape)))

  alloc_out <- quickr_alloc_zero(shape_out, out_aval)
  out_stmt <- rlang::call2("<-", out_sym, alloc_out)

  if (isTRUE(any(as.integer(shape_out) == 0L))) {
    return(list(aligned_stmt, out_stmt))
  }

  # Don't loop over extent-1 dimensions: their index is always 1L.
  loop_dims <- setdiff(which(as.integer(shape_out) > 1L), dim)

  idxs_lhs <- vector("list", rank)
  idxs_rhs <- vector("list", rank)
  for (d in seq_len(rank)) {
    if (d == dim) {
      idxs_lhs[[d]] <- rlang::call2("seq_len", n)
      idxs_rhs[[d]] <- rlang::call2("seq_len", n)
      next
    }

    if (d %in% loop_dims) {
      idxs_lhs[[d]] <- as.name(paste0("i_", as.character(out_sym), "_", d))
    } else {
      idxs_lhs[[d]] <- 1L
    }
    idxs_rhs[[d]] <- 1L
  }

  rhs <- quickr_subscript(aligned_sym, idxs_rhs)
  inner <- rlang::call2("<-", quickr_subscript(out_sym, idxs_lhs), rhs)

  body <- if (length(loop_dims)) {
    loop_syms <- lapply(loop_dims, function(d) as.name(paste0("i_", as.character(out_sym), "_", d)))
    quickr_row_major_loop(loop_syms, shape_out[loop_dims], inner)
  } else {
    inner
  }

  list(aligned_stmt, out_stmt, body)
}

quickr_emit_reverse <- function(out_sym, operand_expr, shape_in, dims, out_aval) {
  shape_in <- as.integer(shape_in)
  dims <- sort(unique(as.integer(dims)))
  rank <- length(shape_in)

  if (rank == 0L) {
    cli_abort("reverse: scalar operands are not supported by quickr lowering") # nocov
  }
  if (rank > 5L) {
    cli_abort("reverse: only arrays up to rank 5 are supported")
  }
  if (length(dims) && (min(dims) < 1L || max(dims) > rank)) {
    cli_abort("reverse: invalid {.arg dims}: {dims}")
  }

  idxs <- lapply(seq_len(rank), function(d) {
    if (d %in% dims) {
      # Avoid `n:1` when `n == 0`: `0:1` is non-empty in R and selects the wrong
      # elements (and quickr lowers it to an invalid Fortran slice).
      n <- as.integer(shape_in[[d]])
      rlang::call2("+", rlang::call2("-", n, rlang::call2("seq_len", n)), 1L)
    } else {
      rlang::call2("seq_len", shape_in[[d]])
    }
  })

  quickr_emit_assign(out_sym, quickr_subscript(operand_expr, idxs, drop = if (rank > 1L) FALSE else NULL))
}

quickr_emit_concatenate <- function(out_sym, operands_expr, operands_shape, dimension, out_aval) {
  dimension <- as.integer(dimension)
  rank <- length(shape(out_aval))
  if (!rank || rank > 5L) {
    cli_abort("concatenate: only arrays of rank 1..5 are supported by quickr lowering")
  }
  if (dimension < 1L || dimension > rank) {
    cli_abort("concatenate: invalid {.arg dimension}: {dimension}")
  }

  operands_shape <- lapply(operands_shape, as.integer)

  out_shape <- as.integer(shape(out_aval))
  alloc_out <- quickr_alloc_zero(out_shape, out_aval)

  stmts <- list(rlang::call2("<-", out_sym, alloc_out))

  offset <- 0L
  for (k in seq_along(operands_expr)) {
    shp <- operands_shape[[k]]
    dim_n <- as.integer(shp[[dimension]])

    # Avoid `a:b` when `b < a`: `:` is never empty in R and quickr lowers it to
    # an invalid Fortran slice.
    if (dim_n == 0L) {
      next
    }

    dim_start <- offset + 1L
    dim_end <- offset + dim_n

    out_idxs <- lapply(seq_len(rank), function(d) {
      if (d == dimension) {
        as.call(list(as.name(":"), dim_start, dim_end))
      } else if (as.integer(shp[[d]]) == 1L) {
        # Use a length-1 range to preserve that dimension.
        as.call(list(as.name(":"), 1L, 1L))
      } else {
        rlang::call2("seq_len", shp[[d]])
      }
    })

    stmts <- c(
      stmts,
      list(rlang::call2(
        "<-",
        quickr_subscript(out_sym, out_idxs),
        operands_expr[[k]]
      ))
    )
    offset <- offset + dim_n
  }

  stmts
}

quickr_emit_static_slice <- function(
  out_sym,
  operand_expr,
  start_indices,
  limit_indices,
  strides,
  shape_out,
  out_aval
) {
  start_indices <- as.integer(start_indices)
  limit_indices <- as.integer(limit_indices)
  strides <- as.integer(strides)
  shape_out <- as.integer(shape_out)

  rank <- length(shape_out)
  if (rank > 5L) {
    cli_abort("static_slice: only arrays up to rank 5 are supported")
  }
  stopifnot(length(start_indices) == rank)
  stopifnot(length(limit_indices) == rank)
  stopifnot(length(strides) == rank)

  if (rank == 0L) {
    return(quickr_emit_assign(out_sym, operand_expr))
  }

  idxs <- Map(
    function(start, stride, n) {
      rlang::call2(
        "+",
        start,
        rlang::call2(
          "*",
          rlang::call2("-", rlang::call2("seq_len", n), 1L),
          stride
        )
      )
    },
    as.list(start_indices),
    as.list(strides),
    as.list(shape_out)
  )

  quickr_emit_assign(out_sym, quickr_subscript(operand_expr, idxs, drop = if (rank > 1L) FALSE else NULL))
}

quickr_clamp_scalar <- function(x, lower, upper) {
  rlang::call2("min", upper, rlang::call2("max", x, lower))
}

quickr_dynamic_slice_idxs <- function(start_indices_expr, shape_in, slice_sizes) {
  Map(
    function(start_expr, n_in, n_slice) {
      upper <- as.integer(n_in - n_slice + 1L)
      start <- quickr_clamp_scalar(start_expr, 1L, upper)
      rlang::call2("+", start, rlang::call2("-", rlang::call2("seq_len", n_slice), 1L))
    },
    start_indices_expr,
    as.list(shape_in),
    as.list(slice_sizes)
  )
}

quickr_emit_dynamic_slice <- function(out_sym, operand_expr, start_indices_expr, shape_in, slice_sizes, out_aval) {
  shape_in <- as.integer(shape_in)
  slice_sizes <- as.integer(slice_sizes)
  rank <- length(shape_in)

  if (rank == 0L) {
    stopifnot(!length(start_indices_expr))
    return(quickr_emit_assign(out_sym, operand_expr))
  }
  if (rank > 5L) {
    cli_abort("dynamic_slice: only arrays up to rank 5 are supported")
  }
  stopifnot(length(start_indices_expr) == rank)
  stopifnot(length(slice_sizes) == rank)

  shape_out <- as.integer(shape(out_aval))
  stopifnot(identical(shape_out, slice_sizes))

  idxs <- quickr_dynamic_slice_idxs(start_indices_expr, shape_in, slice_sizes)

  quickr_emit_assign(
    out_sym,
    quickr_subscript(operand_expr, idxs, drop = if (rank > 1L) FALSE else NULL)
  )
}

quickr_emit_dyn_update_slice <- function(
  out_sym,
  operand_expr,
  update_expr,
  start_indices_expr,
  shape_in,
  shape_update,
  out_aval
) {
  shape_in <- as.integer(shape_in)
  shape_update <- as.integer(shape_update)
  rank <- length(shape_in)

  if (rank == 0L) {
    stopifnot(!length(start_indices_expr))
    return(quickr_emit_assign(out_sym, update_expr))
  }
  if (rank > 5L) {
    cli_abort("dynamic_update_slice: only arrays up to rank 5 are supported")
  }
  stopifnot(length(start_indices_expr) == rank)
  stopifnot(length(shape_update) == rank)

  idxs <- quickr_dynamic_slice_idxs(start_indices_expr, shape_in, shape_update)

  list(
    rlang::call2("<-", out_sym, operand_expr),
    rlang::call2(
      "<-",
      quickr_subscript(out_sym, idxs),
      update_expr
    )
  )
}

quickr_emit_pad <- function(
  out_sym,
  operand_expr,
  padding_value_expr,
  edge_padding_low,
  edge_padding_high,
  interior_padding,
  shape_in,
  out_aval
) {
  edge_padding_low <- as.integer(edge_padding_low)
  edge_padding_high <- as.integer(edge_padding_high)
  interior_padding <- as.integer(interior_padding)
  shape_in <- as.integer(shape_in)
  rank <- length(shape_in)

  if (rank == 0L) {
    stopifnot(!length(edge_padding_low))
    stopifnot(!length(edge_padding_high))
    stopifnot(!length(interior_padding))
    return(quickr_emit_assign(out_sym, operand_expr))
  }
  if (rank > 5L) {
    cli_abort("pad: only arrays up to rank 5 are supported")
  }
  stopifnot(length(edge_padding_low) == rank)
  stopifnot(length(edge_padding_high) == rank)
  stopifnot(length(interior_padding) == rank)

  out_shape <- as.integer(shape(out_aval))
  inferred_shape <- edge_padding_low + edge_padding_high + shape_in + pmax(shape_in - 1L, 0L) * interior_padding
  stopifnot(identical(out_shape, inferred_shape))

  alloc_out <- quickr_alloc_full(out_shape, padding_value_expr)

  idx_expr <- function(start, len, stride = 1L, full_len = NULL) {
    if (len <= 0L) {
      return(integer())
    }
    if (len == 1L) {
      return(start)
    }
    if (!is.null(full_len) && identical(start, 1L) && identical(len, full_len)) {
      return(rlang::call2("seq_len", full_len))
    }
    if (identical(stride, 1L)) {
      return(as.call(list(as.name(":"), start, start + len - 1L)))
    }
    rlang::call2(
      "+",
      start,
      rlang::call2(
        "*",
        rlang::call2("-", rlang::call2("seq_len", len), 1L),
        stride
      )
    )
  }

  idx_map <- Map(
    function(low, interior, n, out_n) {
      stride <- interior + 1L
      crop_low <- max(-low, 0L)
      start_offset <- if (!crop_low) 0L else (crop_low + stride - 1L) %/% stride
      output_start <- low + 1L + start_offset * stride

      if (output_start > out_n || start_offset >= n) {
        return(list(len = 0L, input = integer(), output = integer()))
      }

      len <- min(
        n - start_offset,
        ((out_n - output_start) %/% stride) + 1L
      )

      input_start <- start_offset + 1L
      list(
        len = len,
        input = idx_expr(input_start, len, full_len = n),
        output = idx_expr(output_start, len, stride = stride)
      )
    },
    as.list(edge_padding_low),
    as.list(interior_padding),
    as.list(shape_in),
    as.list(out_shape)
  )

  if (isTRUE(any(vapply(idx_map, \(x) !x$len, logical(1L))))) {
    return(quickr_emit_assign(out_sym, alloc_out))
  }

  in_idxs <- lapply(idx_map, `[[`, "input")
  out_idxs <- lapply(idx_map, `[[`, "output")
  rhs_expr <- quickr_subscript(operand_expr, in_idxs)

  list(
    rlang::call2("<-", out_sym, alloc_out),
    rlang::call2(
      "<-",
      quickr_subscript(out_sym, out_idxs),
      rhs_expr
    )
  )
}

quickr_emit_gather <- function(
  out_sym,
  operand_expr,
  start_indices_expr,
  shape_operand,
  shape_start_indices,
  slice_sizes,
  offset_dims,
  collapsed_slice_dims,
  operand_batching_dims,
  start_indices_batching_dims,
  start_index_map,
  index_vector_dim,
  out_aval
) {
  shape_operand <- as.integer(shape_operand)
  shape_start_indices <- as.integer(shape_start_indices)
  slice_sizes <- as.integer(slice_sizes)
  offset_dims <- as.integer(offset_dims)
  collapsed_slice_dims <- sort(unique(as.integer(collapsed_slice_dims)))
  operand_batching_dims <- as.integer(operand_batching_dims)
  start_indices_batching_dims <- as.integer(start_indices_batching_dims)
  start_index_map <- as.integer(start_index_map)
  index_vector_dim <- as.integer(index_vector_dim)

  op_rank <- length(shape_operand)
  si_rank <- length(shape_start_indices)
  out_shape <- as.integer(shape(out_aval))
  out_rank <- length(out_shape)

  if (!op_rank) {
    cli_abort("gather: scalar operands are not supported by quickr lowering")
  }
  if (op_rank > 5L || si_rank > 5L || out_rank > 5L) {
    cli_abort("gather: only arrays up to rank 5 are supported")
  }
  if (length(operand_batching_dims) || length(start_indices_batching_dims)) {
    cli_abort("gather: batching dims are not supported by quickr lowering")
  }
  if (!identical(index_vector_dim, si_rank)) {
    cli_abort("gather: only index_vector_dim on the last dimension is supported by quickr lowering")
  }
  if (si_rank == 0L) {
    cli_abort("gather: start_indices must have rank >= 1")
  }
  if (!identical(length(slice_sizes), op_rank)) {
    cli_abort("gather: slice_sizes must have length equal to operand rank")
  }
  index_vector_size <- as.integer(shape_start_indices[[si_rank]])
  if (!identical(as.integer(length(start_index_map)), as.integer(index_vector_size))) {
    cli_abort("gather: start_index_map length must match start_indices index vector size")
  }
  if (length(start_index_map) && (min(start_index_map) < 1L || max(start_index_map) > op_rank)) {
    cli_abort("gather: invalid start_index_map: {start_index_map}")
  }
  if (length(unique(start_index_map)) != length(start_index_map)) {
    cli_abort("gather: start_index_map must not contain duplicates")
  }
  if (length(collapsed_slice_dims) && (min(collapsed_slice_dims) < 1L || max(collapsed_slice_dims) > op_rank)) {
    cli_abort("gather: invalid collapsed_slice_dims: {collapsed_slice_dims}")
  }

  slice_dims <- setdiff(seq_len(op_rank), collapsed_slice_dims)
  if (!identical(length(offset_dims), length(slice_dims))) {
    cli_abort("gather: offset_dims must have length {length(slice_dims)}")
  }
  if (length(offset_dims) && (min(offset_dims) < 1L || max(offset_dims) > out_rank)) {
    cli_abort("gather: invalid offset_dims: {offset_dims}")
  }

  batch_out_dims <- setdiff(seq_len(out_rank), offset_dims)
  expected_batch_rank <- si_rank - 1L
  if (!identical(length(batch_out_dims), expected_batch_rank)) {
    cli_abort("gather: output batch rank does not match start_indices batch rank")
  }

  alloc_out <- if (out_rank == 0L) {
    NULL
  } else {
    quickr_alloc_zero(out_shape, out_aval)
  }

  out_idxs <- lapply(seq_len(out_rank), function(d) as.name(paste0("i_", as.character(out_sym), "_", d)))
  batch_idxs <- if (length(batch_out_dims)) out_idxs[batch_out_dims] else list()

  start_at_component <- function(k) {
    k <- as.integer(k)
    idxs <- c(batch_idxs, list(k))
    quickr_subscript(start_indices_expr, idxs)
  }

  start_sym <- vector("list", op_rank)
  start_stmts <- list()
  for (k in seq_along(start_index_map)) {
    d <- start_index_map[[k]]
    sym <- as.name(paste0("s_", as.character(out_sym), "_", d))
    start_sym[[d]] <- sym
    upper <- as.integer(shape_operand[[d]] - slice_sizes[[d]] + 1L)
    start_stmts <- c(
      start_stmts,
      list(rlang::call2("<-", sym, start_at_component(k))),
      list(rlang::call2("<-", sym, quickr_clamp_scalar(sym, 1L, upper)))
    )
  }

  operand_idxs <- lapply(seq_len(op_rank), function(d) {
    s <- start_sym[[d]] %||% 1L
    if (d %in% collapsed_slice_dims) {
      return(s)
    }
    k <- match(d, slice_dims)
    od <- offset_dims[[k]]
    rlang::call2("+", s, rlang::call2("-", out_idxs[[od]], 1L))
  })

  if (out_rank == 0L) {
    stmts <- c(
      start_stmts,
      quickr_emit_assign(out_sym, quickr_subscript(operand_expr, operand_idxs))
    )
    return(stmts)
  }

  elem_assign <- rlang::call2(
    "<-",
    quickr_subscript(out_sym, out_idxs),
    quickr_subscript(operand_expr, operand_idxs)
  )
  inner <- as.call(c(list(as.name("{")), c(start_stmts, list(elem_assign))))

  list(
    rlang::call2("<-", out_sym, alloc_out),
    quickr_row_major_loop(out_idxs, out_shape, inner)
  )
}

quickr_expr_of_node <- function(node, node_expr) {
  if (is_graph_literal(node)) {
    return(quickr_scalar_cast(node$aval$data, as.character(dtype(node))))
  }
  expr <- node_expr[[node]]
  stopifnot(!is.null(expr))
  expr
}


# Primitive emission ------------------------------------------------------------

quickr_emit_dot_general <- function(
  out_sym,
  lhs_expr,
  rhs_expr,
  lhs_shape,
  rhs_shape,
  out_shape,
  out_aval,
  contracting_dims,
  batching_dims
) {
  lhs_shape <- as.integer(lhs_shape)
  rhs_shape <- as.integer(rhs_shape)
  out_shape <- as.integer(out_shape)

  lhs_rank <- length(lhs_shape)
  rhs_rank <- length(rhs_shape)
  out_rank <- length(out_shape)

  if (lhs_rank > 5L || rhs_rank > 5L || out_rank > 5L) {
    cli_abort("dot_general: only arrays up to rank 5 are supported")
  }

  cd_lhs <- as.integer(contracting_dims[[1L]])
  cd_rhs <- as.integer(contracting_dims[[2L]])
  bd_lhs <- as.integer(batching_dims[[1L]])
  bd_rhs <- as.integer(batching_dims[[2L]])

  free_lhs <- setdiff(seq_len(lhs_rank), c(bd_lhs, cd_lhs))
  free_rhs <- setdiff(seq_len(rhs_rank), c(bd_rhs, cd_rhs))

  for_loop <- function(var_sym, upper, body) {
    as.call(list(
      as.name("for"),
      var_sym,
      rlang::call2("seq_len", as.integer(upper)),
      body
    ))
  }

  nbatch <- length(bd_lhs)
  nfree_lhs <- length(free_lhs)

  out_idxs <- lapply(seq_len(out_rank), function(d) as.name(paste0("i_", as.character(out_sym), "_", d)))
  k_idxs <- lapply(seq_along(cd_lhs), function(i) as.name(paste0("k_", as.character(out_sym), "_", i)))
  acc <- as.name(paste0("acc_", as.character(out_sym)))

  lhs_idxs <- vector("list", lhs_rank)
  for (d in seq_len(lhs_rank)) {
    if (d %in% bd_lhs) {
      lhs_idxs[[d]] <- out_idxs[[match(d, bd_lhs)]]
    } else if (d %in% free_lhs) {
      lhs_idxs[[d]] <- out_idxs[[nbatch + match(d, free_lhs)]]
    } else {
      lhs_idxs[[d]] <- k_idxs[[match(d, cd_lhs)]]
    }
  }

  rhs_idxs <- vector("list", rhs_rank)
  for (d in seq_len(rhs_rank)) {
    if (d %in% bd_rhs) {
      rhs_idxs[[d]] <- out_idxs[[match(d, bd_rhs)]]
    } else if (d %in% free_rhs) {
      rhs_idxs[[d]] <- out_idxs[[nbatch + nfree_lhs + match(d, free_rhs)]]
    } else {
      rhs_idxs[[d]] <- k_idxs[[match(d, cd_rhs)]]
    }
  }

  lhs_at <- quickr_subscript(lhs_expr, lhs_idxs)
  rhs_at <- quickr_subscript(rhs_expr, rhs_idxs)
  prod_at <- rlang::call2("*", lhs_at, rhs_at)
  zero <- quickr_zero_literal_for(out_aval)

  if (!length(cd_lhs)) {
    if (out_rank == 0L) {
      return(quickr_emit_assign(out_sym, prod_at))
    }

    alloc <- quickr_alloc_zero(out_shape, out_aval)
    assign_elem <- rlang::call2("<-", quickr_subscript(out_sym, out_idxs), prod_at)
    return(list(rlang::call2("<-", out_sym, alloc), quickr_row_major_loop(out_idxs, out_shape, assign_elem)))
  }

  update_acc <- rlang::call2("<-", acc, rlang::call2("+", acc, prod_at))
  contract_body <- update_acc
  for (i in seq_along(k_idxs)) {
    contract_body <- for_loop(k_idxs[[i]], lhs_shape[[cd_lhs[[i]]]], contract_body)
  }

  if (out_rank == 0L) {
    return(list(
      rlang::call2("<-", acc, zero),
      contract_body,
      rlang::call2("<-", out_sym, acc)
    ))
  }

  alloc <- quickr_alloc_zero(out_shape, out_aval)
  assign_out <- rlang::call2("<-", quickr_subscript(out_sym, out_idxs), acc)
  elem_body <- as.call(c(list(as.name("{")), list(rlang::call2("<-", acc, zero), contract_body, assign_out)))

  list(rlang::call2("<-", out_sym, alloc), quickr_row_major_loop(out_idxs, out_shape, elem_body))
}

quickr_emit_transpose <- function(out_sym, operand_expr, permutation, out_shape, out_aval) {
  if (length(out_shape) != 2L || length(permutation) != 2L) {
    cli_abort("transpose: only rank-2 arrays are supported")
  }
  if (identical(permutation, c(1L, 2L))) {
    return(quickr_emit_assign(out_sym, operand_expr))
  }
  stopifnot(identical(permutation, c(2L, 1L)))
  quickr_emit_assign(out_sym, rlang::call2("t", operand_expr))
}

quickr_emit_broadcast_in_dim <- function(out_sym, operand_expr, shape_in, shape_out, broadcast_dimensions, out_aval) {
  broadcast_dimensions <- as.integer(broadcast_dimensions)
  if (identical(shape_in, shape_out)) {
    return(quickr_emit_assign(out_sym, operand_expr))
  }

  rank_in <- length(shape_in)
  rank_out <- length(shape_out)

  if (rank_in == 0L) {
    return(quickr_emit_full_like(out_sym, operand_expr, shape_out, out_aval))
  }

  if (rank_in > 5L || rank_out > 5L) {
    cli_abort("broadcast_in_dim: only arrays up to rank 5 are supported")
  }
  aligned_shape <- rep.int(1L, rank_out)
  for (d_in in seq_len(rank_in)) {
    aligned_shape[[broadcast_dimensions[[d_in]]]] <- as.integer(shape_in[[d_in]])
  }

  bcast_dims <- which((as.integer(aligned_shape) == 1L) & (as.integer(shape_out) != 1L))
  if (!length(bcast_dims)) {
    if (rank_in == rank_out && identical(broadcast_dimensions, seq_len(rank_out))) {
      return(quickr_emit_assign(out_sym, operand_expr))
    }
    return(quickr_emit_assign(out_sym, rlang::call2("array", operand_expr, dim = as.integer(shape_out))))
  }

  aligned_sym <- as.name(paste0("aligned_", as.character(out_sym)))
  aligned_expr <- if (rank_in == rank_out && identical(broadcast_dimensions, seq_len(rank_out))) {
    operand_expr
  } else {
    rlang::call2("array", operand_expr, dim = as.integer(aligned_shape))
  }
  aligned_stmt <- rlang::call2("<-", aligned_sym, aligned_expr)

  out_zero <- quickr_zero_literal_for(out_aval)
  alloc_out <- rlang::call2("array", out_zero, dim = as.integer(shape_out))
  out_stmt <- rlang::call2("<-", out_sym, alloc_out)

  idxs_lhs <- vector("list", rank_out)
  idxs_rhs <- vector("list", rank_out)
  for (d in seq_len(rank_out)) {
    if (d %in% bcast_dims) {
      idxs_lhs[[d]] <- as.name(paste0("i_", as.character(out_sym), "_", d))
      idxs_rhs[[d]] <- 1L
    } else if (as.integer(shape_out[[d]]) == 1L) {
      idxs_lhs[[d]] <- 1L
      idxs_rhs[[d]] <- 1L
    } else {
      idxs_lhs[[d]] <- rlang::call2("seq_len", as.integer(shape_out[[d]]))
      idxs_rhs[[d]] <- rlang::call2("seq_len", as.integer(aligned_shape[[d]]))
    }
  }

  rhs <- quickr_subscript(aligned_sym, idxs_rhs)
  inner <- rlang::call2("<-", quickr_subscript(out_sym, idxs_lhs), rhs)
  loop_syms <- lapply(bcast_dims, function(d) as.name(paste0("i_", as.character(out_sym), "_", d)))
  body <- quickr_row_major_loop(loop_syms, shape_out[bcast_dims], inner)
  list(aligned_stmt, out_stmt, body)
}

quickr_emit_reduce2_axis_loop <- function(
  out_sym,
  m,
  n,
  dims,
  drop,
  alloc_stmts,
  init_acc_expr,
  inner_start,
  update_builder
) {
  dims <- as.integer(dims)
  ii <- as.name(paste0("i_", as.character(out_sym)))
  jj <- as.name(paste0("j_", as.character(out_sym)))
  acc <- as.name(paste0("acc_", as.character(out_sym)))

  outer_sym <- if (identical(dims, 2L)) ii else jj
  outer_n <- if (identical(dims, 2L)) m else n
  inner_sym <- if (identical(dims, 2L)) jj else ii
  inner_n <- if (identical(dims, 2L)) n else m

  assign_out <- if (identical(dims, 2L)) {
    if (isTRUE(drop)) {
      rlang::call2("<-", rlang::call2("[", out_sym, ii), acc)
    } else {
      rlang::call2("<-", rlang::call2("[", out_sym, ii, 1L), acc)
    }
  } else {
    if (isTRUE(drop)) {
      rlang::call2("<-", rlang::call2("[", out_sym, jj), acc)
    } else {
      rlang::call2("<-", rlang::call2("[", out_sym, 1L, jj), acc)
    }
  }

  body_parts <- list(rlang::call2("<-", acc, init_acc_expr))
  if (as.integer(inner_n) >= as.integer(inner_start)) {
    body_parts <- c(
      body_parts,
      list(as.call(list(
        as.name("for"),
        inner_sym,
        as.call(list(as.name(":"), as.integer(inner_start), as.integer(inner_n))),
        update_builder(ii, jj, acc)
      )))
    )
  }
  body_parts <- c(body_parts, list(assign_out))
  outer_body <- as.call(c(list(as.name("{")), body_parts))

  c(
    alloc_stmts,
    list(as.call(list(
      as.name("for"),
      outer_sym,
      rlang::call2("seq_len", as.integer(outer_n)),
      outer_body
    )))
  )
}

quickr_emit_reduce <- function(kind, out_sym, operand_expr, shape_in, dims, drop, out_aval) {
  kind <- as.character(kind)
  shape_in <- as.integer(shape_in)
  dims <- sort(unique(as.integer(dims)))
  rank <- length(shape_in)
  dt_out <- as.character(dtype(out_aval))
  if (!kind %in% c("sum", "prod", "max", "min")) {
    cli_abort("Internal error: unknown reduction kind: {.val {kind}}")
  }
  if (kind %in% c("max", "min") && length(dims) && isTRUE(any(shape_in[dims] == 0L, na.rm = TRUE))) {
    cli_abort("{kind}: reductions over empty dimensions are not supported by quickr lowering")
  }
  if (kind %in% c("sum", "prod")) {
    init_acc_scalar <- if (kind == "sum") {
      quickr_zero_literal_for(out_aval)
    } else if (dt_out == "f64") {
      1.0
    } else {
      1L
    }
    if (as.character(dtype(out_aval)) %in% "bool") {
      cli_abort("{kind}: pred reductions are not supported by quickr lowering")
    }
  } else {
    init_acc_scalar <- NULL
  }

  if (rank == 0L) {
    if (length(dims)) {
      cli_abort("{kind}: scalar reduction dims must be empty")
    }
    return(quickr_emit_assign(out_sym, operand_expr))
  }

  if (rank == 1L) {
    if (!length(dims)) {
      return(quickr_emit_assign(out_sym, operand_expr))
    }
    if (!identical(dims, 1L)) {
      cli_abort("{kind}: unsupported reduction dims for rank-1 array")
    }
    if (kind == "prod" && dt_out == "i32") {
      return(quickr_emit_truncating_i32(out_sym, rlang::call2("prod", operand_expr), shape(out_aval)))
    }
    if (isTRUE(drop)) {
      return(quickr_emit_assign(out_sym, rlang::call2(kind, operand_expr)))
    }
    return(quickr_emit_assign(
      out_sym,
      rlang::call2("array", rlang::call2(kind, operand_expr), dim = 1L)
    ))
  }

  if (rank == 2L) {
    m <- as.integer(shape_in[[1L]])
    n <- as.integer(shape_in[[2L]])
    ctor <- quickr_dtype_to_r_ctor(as.character(dtype(out_aval)))

    if (identical(dims, c(1L, 2L))) {
      if (kind == "prod" && dt_out == "i32") {
        return(quickr_emit_truncating_i32(out_sym, rlang::call2("prod", operand_expr), shape(out_aval)))
      }
      if (isTRUE(drop)) {
        return(quickr_emit_assign(out_sym, rlang::call2(kind, operand_expr)))
      }
      return(quickr_emit_assign(
        out_sym,
        rlang::call2("matrix", rlang::call2(kind, operand_expr), nrow = 1L, ncol = 1L)
      ))
    }

    update <- switch(
      kind,
      sum = function(ii, jj, acc) {
        rlang::call2("<-", acc, rlang::call2("+", acc, rlang::call2("[", operand_expr, ii, jj)))
      },
      prod = function(ii, jj, acc) {
        rlang::call2("<-", acc, rlang::call2("*", acc, rlang::call2("[", operand_expr, ii, jj)))
      },
      max = function(ii, jj, acc) {
        rlang::call2("<-", acc, rlang::call2("max", acc, rlang::call2("[", operand_expr, ii, jj)))
      },
      min = function(ii, jj, acc) {
        rlang::call2("<-", acc, rlang::call2("min", acc, rlang::call2("[", operand_expr, ii, jj)))
      }
    )

    if (identical(dims, 2L)) {
      init_acc_expr <- if (kind %in% c("max", "min")) {
        ii <- as.name(paste0("i_", as.character(out_sym)))
        rlang::call2("[", operand_expr, ii, 1L)
      } else {
        init_acc_scalar
      }
      inner_start <- if (kind %in% c("max", "min")) 2L else 1L
      alloc <- if (isTRUE(drop)) {
        quickr_emit_assign(out_sym, rlang::call2(ctor, m))
      } else {
        quickr_emit_assign(out_sym, rlang::call2("matrix", quickr_zero_literal_for(out_aval), nrow = m, ncol = 1L))
      }
      return(quickr_emit_reduce2_axis_loop(out_sym, m, n, 2L, drop, alloc, init_acc_expr, inner_start, update))
    }

    if (identical(dims, 1L)) {
      init_acc_expr <- if (kind %in% c("max", "min")) {
        jj <- as.name(paste0("j_", as.character(out_sym)))
        rlang::call2("[", operand_expr, 1L, jj)
      } else {
        init_acc_scalar
      }
      inner_start <- if (kind %in% c("max", "min")) 2L else 1L
      alloc <- if (isTRUE(drop)) {
        quickr_emit_assign(out_sym, rlang::call2(ctor, n))
      } else {
        quickr_emit_assign(out_sym, rlang::call2("matrix", quickr_zero_literal_for(out_aval), nrow = 1L, ncol = n))
      }
      return(quickr_emit_reduce2_axis_loop(out_sym, m, n, 1L, drop, alloc, init_acc_expr, inner_start, update))
    }

    cli_abort("{kind}: unsupported reduction dims for rank-2 array")
  }

  if (!length(dims)) {
    return(quickr_emit_assign(out_sym, operand_expr))
  }
  if (!identical(dims, seq_len(rank))) {
    cli_abort("{kind}: for rank > 2, only full reductions (dims = seq_len(rank)) are supported")
  }

  if (kind == "prod" && dt_out == "i32") {
    return(quickr_emit_truncating_i32(out_sym, rlang::call2("prod", operand_expr), shape(out_aval)))
  }

  if (isTRUE(drop)) {
    return(quickr_emit_assign(out_sym, rlang::call2(kind, operand_expr)))
  }

  quickr_emit_assign(
    out_sym,
    rlang::call2("array", rlang::call2(kind, operand_expr), dim = rep(1L, rank))
  )
}

quickr_emit_reduce_boolean <- function(kind, out_sym, operand_expr, shape_in, dims, drop, out_aval) {
  kind <- as.character(kind)
  shape_in <- as.integer(shape_in)
  dims <- sort(unique(as.integer(dims)))
  rank <- length(shape_in)

  if (!kind %in% c("any", "all")) {
    cli_abort("Internal error: unknown boolean reduction kind: {.val {kind}}")
  }

  if (rank == 0L) {
    if (length(dims)) {
      cli_abort("{kind}: scalar reduction dims must be empty")
    }
    return(quickr_emit_assign(out_sym, operand_expr))
  }

  if (!length(dims)) {
    return(quickr_emit_assign(out_sym, operand_expr))
  }

  reduce_call <- function(x) rlang::call2(kind, x)

  if (rank == 1L) {
    if (!identical(dims, 1L)) {
      cli_abort("{kind}: unsupported reduction dims for rank-1 array")
    }

    reduced <- reduce_call(operand_expr)
    if (isTRUE(drop)) {
      return(quickr_emit_assign(out_sym, reduced))
    }
    return(quickr_emit_assign(out_sym, rlang::call2("array", reduced, dim = 1L)))
  }

  if (rank == 2L) {
    m <- as.integer(shape_in[[1L]])
    n <- as.integer(shape_in[[2L]])

    if (identical(dims, c(1L, 2L))) {
      reduced <- reduce_call(operand_expr)
      if (isTRUE(drop)) {
        return(quickr_emit_assign(out_sym, reduced))
      }
      return(quickr_emit_assign(out_sym, rlang::call2("matrix", reduced, nrow = 1L, ncol = 1L)))
    }

    ctor <- quickr_dtype_to_r_ctor(as.character(dtype(out_aval)))
    zero <- quickr_zero_literal_for(out_aval)

    if (identical(dims, 2L)) {
      ii <- as.name(paste0("i_", as.character(out_sym)))
      alloc <- if (isTRUE(drop)) {
        quickr_emit_assign(out_sym, rlang::call2(ctor, m))
      } else {
        quickr_emit_assign(out_sym, rlang::call2("matrix", zero, nrow = m, ncol = 1L))
      }

      row <- rlang::call2("[", operand_expr, ii, rlang::call2("seq_len", n))
      out_at <- if (isTRUE(drop)) rlang::call2("[", out_sym, ii) else rlang::call2("[", out_sym, ii, 1L)
      inner <- rlang::call2("<-", out_at, reduce_call(row))

      return(c(alloc, list(as.call(list(as.name("for"), ii, rlang::call2("seq_len", m), inner)))))
    }

    if (identical(dims, 1L)) {
      jj <- as.name(paste0("j_", as.character(out_sym)))
      alloc <- if (isTRUE(drop)) {
        quickr_emit_assign(out_sym, rlang::call2(ctor, n))
      } else {
        quickr_emit_assign(out_sym, rlang::call2("matrix", zero, nrow = 1L, ncol = n))
      }

      col <- rlang::call2("[", operand_expr, rlang::call2("seq_len", m), jj)
      out_at <- if (isTRUE(drop)) rlang::call2("[", out_sym, jj) else rlang::call2("[", out_sym, 1L, jj)
      inner <- rlang::call2("<-", out_at, reduce_call(col))

      return(c(alloc, list(as.call(list(as.name("for"), jj, rlang::call2("seq_len", n), inner)))))
    }

    cli_abort("{kind}: unsupported reduction dims for rank-2 array")
  }

  if (!identical(dims, seq_len(rank))) {
    cli_abort("{kind}: for rank > 2, only full reductions (dims = seq_len(rank)) are supported")
  }

  reduced <- reduce_call(operand_expr)
  if (isTRUE(drop)) {
    return(quickr_emit_assign(out_sym, reduced))
  }
  quickr_emit_assign(out_sym, rlang::call2("array", reduced, dim = rep(1L, rank)))
}

quickr_emit_reshape <- function(out_sym, operand_expr, shape_in, shape_out, out_aval) {
  shape_in <- as.integer(shape_in)
  shape_out <- as.integer(shape_out)

  if (length(shape_in) > 5L || length(shape_out) > 5L) {
    cli_abort("reshape: only arrays up to rank 5 are supported")
  }

  nflat <- Reduce(`*`, shape_in, init = 1L)
  if (identical(as.integer(nflat), 1L)) {
    scalar_expr <- if (!length(shape_in)) {
      operand_expr
    } else {
      quickr_subscript(operand_expr, rep.int(list(1L), length(shape_in)))
    }
    return(quickr_emit_full_like(out_sym, scalar_expr, shape_out, out_aval))
  }

  ctor <- quickr_dtype_to_r_ctor(as.character(dtype(out_aval)))
  flat_sym <- as.name(paste0("flat_", as.character(out_sym)))
  idx_sym <- as.name(paste0("idx_", as.character(out_sym)))
  rank_in <- length(shape_in)
  rank_out <- length(shape_out)

  stmts <- list(
    rlang::call2("<-", flat_sym, rlang::call2(ctor, as.integer(nflat))),
    rlang::call2("<-", idx_sym, 0L)
  )

  in_idxs <- lapply(seq_len(rank_in), function(d) as.name(paste0("i_", as.character(out_sym), "_", d)))
  elem_in <- quickr_subscript(operand_expr, in_idxs)
  inner_in <- as.call(c(
    list(as.name("{")),
    list(rlang::call2("<-", idx_sym, rlang::call2("+", idx_sym, 1L))),
    list(rlang::call2("<-", rlang::call2("[", flat_sym, idx_sym), elem_in))
  ))
  stmts <- c(stmts, list(quickr_row_major_loop(in_idxs, shape_in, inner_in)))

  stmts <- c(stmts, list(rlang::call2("<-", idx_sym, 0L)))

  alloc_out <- quickr_alloc_zero(shape_out, out_aval)

  stmts <- c(stmts, quickr_emit_assign(out_sym, alloc_out))

  out_idxs <- lapply(seq_len(rank_out), function(d) as.name(paste0("o_", as.character(out_sym), "_", d)))
  assign_out <- if (rank_out == 1L) {
    rlang::call2("<-", rlang::call2("[", out_sym, out_idxs[[1L]]), rlang::call2("[", flat_sym, idx_sym))
  } else {
    rlang::call2("<-", quickr_subscript(out_sym, out_idxs), rlang::call2("[", flat_sym, idx_sym))
  }
  inner_out <- as.call(c(
    list(as.name("{")),
    list(rlang::call2("<-", idx_sym, rlang::call2("+", idx_sym, 1L))),
    list(assign_out)
  ))
  stmts <- c(stmts, list(quickr_row_major_loop(out_idxs, shape_out, inner_out)))
  stmts
}


# Primitive lowering registry ---------------------------------------------------

quickr_register_prim_lowerer <- function(primitive, fun) {
  primitives <- if (inherits(primitive, "AnvlPrimitive") || inherits(primitive, "JitPrimitive")) {
    list(primitive)
  } else {
    primitive
  }
  for (primitive in primitives) {
    primitive[["quickr"]] <- fun
  }
  invisible(fun)
}

quickr_supported_prims <- function() {
  sort(unlist(
    eapply(primitive_env, function(primitive) {
      if (inherits(primitive, "JitPrimitive")) {
        primitive <- attr(primitive, "primitive")
      }
      if (is.null(primitive$rules[["quickr"]])) {
        NULL
      } else {
        primitive$name
      }
    }),
    use.names = FALSE
  ))
}

quickr_lower_graph_calls <- function(graph, ctx) {
  node_expr <- ctx$node_expr
  new_tmp_sym <- ctx$new_tmp_sym

  stmts <- list()
  for (call in graph$calls) {
    input_exprs_call <- lapply(call$inputs, quickr_expr_of_node, node_expr = node_expr)
    out_syms_call <- vector("list", length(call$outputs))
    out_avals_call <- vector("list", length(call$outputs))
    for (i in seq_along(call$outputs)) {
      out_node <- call$outputs[[i]]
      if (!is_graph_value(out_node)) {
        cli_abort("Unsupported: non-GraphValue primitive outputs")
      }
      sym <- new_tmp_sym()
      node_expr[[out_node]] <- sym
      out_syms_call[[i]] <- sym
      out_avals_call[[i]] <- out_node$aval
    }

    lower <- call$primitive[["quickr"]]
    if (is.null(lower)) {
      quickr_abort_unsupported_prims(call$primitive$name, quickr_supported_prims())
    }
    stmts <- c(
      stmts,
      lower(
        call$primitive$name,
        input_exprs_call,
        call$params,
        out_syms_call,
        call$inputs,
        out_avals_call,
        ctx = ctx
      )
    )
  }

  out_exprs <- lapply(graph$outputs, quickr_expr_of_node, node_expr = node_expr)
  list(stmts = stmts, out_exprs = out_exprs)
}

quickr_lower_inline_graph <- function(graph, input_exprs, ctx) {
  if (!is_graph(graph)) {
    cli_abort("{.arg graph} must be a {.cls AnvlGraph}")
  }
  node_expr <- ctx$node_expr

  if (length(input_exprs) != length(graph$inputs)) {
    cli_abort("Internal error: subgraph input arity mismatch")
  }

  for (i in seq_along(graph$inputs)) {
    node_expr[[graph$inputs[[i]]]] <- input_exprs[[i]]
  }

  for (const_node in graph$constants) {
    if (!is_graph_value(const_node)) {
      cli_abort("Internal error: subgraph constants must be GraphValue nodes")
    }
    if (is.null(node_expr[[const_node]])) {
      cli_abort("quickr lowering: subgraph constant is not available in parent graph constants")
    }
  }

  quickr_lower_graph_calls(graph, ctx)
}

local({
  quickr_register_prim_lowerer(
    prim_fill,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      dt_chr <- as.character(params$dtype)
      value_expr <- quickr_scalar_cast(params$value, dt_chr)
      quickr_emit_full_like(out_sym, value_expr, params$shape, out_aval)
    }
  )

  quickr_register_prim_lowerer(
    prim_iota,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      quickr_emit_iota(out_sym, params$dim, params$start, shape(out_aval), out_aval)
    }
  )

  quickr_register_prim_lowerer(
    prim_convert,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      quickr_emit_convert(out_sym, inputs[[1L]], shape(operand_node$aval), operand_node$aval, out_aval, ctx = ctx)
    }
  )

  quickr_register_prim_lowerer(
    prim_reverse,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      quickr_emit_reverse(out_sym, inputs[[1L]], shape(operand_node$aval), params$dims, out_aval)
    }
  )

  quickr_register_prim_lowerer(
    prim_concatenate,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      shapes <- lapply(input_nodes, function(n) shape(n$aval))
      quickr_emit_concatenate(out_sym, inputs, shapes, params$dimension, out_aval)
    }
  )

  quickr_register_prim_lowerer(
    prim_static_slice,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      quickr_emit_static_slice(
        out_sym,
        inputs[[1L]],
        params$start_indices,
        params$limit_indices,
        params$strides,
        shape(out_aval),
        out_aval
      )
    }
  )

  quickr_register_prim_lowerer(
    prim_dynamic_slice,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      start_exprs <- inputs[-1L]
      quickr_emit_dynamic_slice(
        out_sym,
        inputs[[1L]],
        start_exprs,
        shape(operand_node$aval),
        params$slice_sizes,
        out_aval
      )
    }
  )

  quickr_register_prim_lowerer(
    prim_dynamic_update_slice,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      update_node <- input_nodes[[2L]]
      start_exprs <- inputs[-c(1L, 2L)]
      quickr_emit_dyn_update_slice(
        out_sym,
        inputs[[1L]],
        inputs[[2L]],
        start_exprs,
        shape(operand_node$aval),
        shape(update_node$aval),
        out_aval
      )
    }
  )

  quickr_register_prim_lowerer(
    prim_pad,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      quickr_emit_pad(
        out_sym,
        inputs[[1L]],
        inputs[[2L]],
        params$edge_padding_low,
        params$edge_padding_high,
        params$interior_padding,
        shape(operand_node$aval),
        out_aval
      )
    }
  )

  quickr_register_prim_lowerer(
    prim_gather,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      start_indices_node <- input_nodes[[2L]]

      dt_idx <- as.character(dtype(start_indices_node$aval))
      if (dt_idx != "i32") {
        cli_abort("gather: only {.val i32} start_indices are supported by quickr lowering")
      }

      quickr_emit_gather(
        out_sym,
        inputs[[1L]],
        inputs[[2L]],
        shape(operand_node$aval),
        shape(start_indices_node$aval),
        params$slice_sizes,
        params$offset_dims,
        params$collapsed_slice_dims,
        params$operand_batching_dims,
        params$start_indices_batching_dims,
        params$start_index_map,
        params$index_vector_dim,
        out_aval
      )
    }
  )

  quickr_register_prim_lowerer(
    prim_if,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      if (is.null(ctx)) {
        cli_abort("Internal error: missing quickr lowering context for primitive {.val if}")
      }
      true_graph <- params$true_graph
      false_graph <- params$false_graph

      lowered_true <- quickr_lower_inline_graph(true_graph, list(), ctx)
      lowered_false <- quickr_lower_inline_graph(false_graph, list(), ctx)

      if (length(out_syms) != length(lowered_true$out_exprs) || length(out_syms) != length(lowered_false$out_exprs)) {
        cli_abort("if: branch arity mismatch")
      }

      assign_out <- function(exprs) {
        Map(rlang::call2, rep("<-", length(out_syms)), out_syms, exprs)
      }

      true_block <- as.call(c(list(as.name("{")), c(lowered_true$stmts, assign_out(lowered_true$out_exprs))))
      false_block <- as.call(c(list(as.name("{")), c(lowered_false$stmts, assign_out(lowered_false$out_exprs))))

      list(as.call(list(as.name("if"), inputs[[1L]], true_block, false_block)))
    }
  )

  quickr_register_prim_lowerer(
    prim_while,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      if (is.null(ctx)) {
        cli_abort("Internal error: missing quickr lowering context for primitive {.val while}")
      }
      cond_graph <- params$cond_graph
      body_graph <- params$body_graph

      if (length(out_syms) != length(inputs)) {
        cli_abort("while: state arity mismatch between inputs and outputs")
      }
      if (length(cond_graph$outputs) != 1L) {
        cli_abort("while: condition graph must have exactly one output")
      }

      state_init <- Map(rlang::call2, rep("<-", length(out_syms)), out_syms, inputs)
      cond_sym <- as.name(paste0("cond_", as.character(out_syms[[1L]])))

      lower_cond <- function() {
        lowered <- quickr_lower_inline_graph(cond_graph, out_syms, ctx)
        c(lowered$stmts, list(rlang::call2("<-", cond_sym, lowered$out_exprs[[1L]])))
      }

      lower_body <- function() {
        lowered <- quickr_lower_inline_graph(body_graph, out_syms, ctx)
        assigns <- Map(rlang::call2, rep("<-", length(out_syms)), out_syms, lowered$out_exprs)
        c(lowered$stmts, assigns)
      }

      loop_body <- as.call(c(
        list(as.name("{")),
        c(
          lower_body(),
          lower_cond()
        )
      ))

      c(
        state_init,
        lower_cond(),
        list(as.call(list(as.name("while"), cond_sym, loop_body)))
      )
    }
  )

  quickr_register_prim_lowerer(
    prim_scatter,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      if (is.null(ctx)) {
        cli_abort("Internal error: missing quickr lowering context for primitive {.val scatter}")
      }

      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      input_node <- input_nodes[[1L]]
      idx_node <- input_nodes[[2L]]
      update_node <- input_nodes[[3L]]

      dt_idx <- as.character(dtype(idx_node$aval))
      if (dt_idx != "i32") {
        cli_abort("scatter: only {.val i32} scatter_indices are supported by quickr lowering")
      }

      shape_in <- as.integer(shape(input_node$aval))
      rank_in <- length(shape_in)
      if (rank_in != 1L) {
        cli_abort("scatter: only rank-1 inputs are supported by quickr lowering")
      }

      update_window_dims <- sort(unique(as.integer(params$update_window_dims)))
      inserted_window_dims <- sort(unique(as.integer(params$inserted_window_dims)))
      input_batching_dims <- as.integer(params$input_batching_dims)
      scatter_indices_batching_dims <- as.integer(params$scatter_indices_batching_dims)
      scatter_dims_to_operand_dims <- as.integer(params$scatter_dims_to_operand_dims)
      index_vector_dim <- as.integer(params$index_vector_dim)

      if (length(input_batching_dims) || length(scatter_indices_batching_dims)) {
        cli_abort("scatter: batching dims are not supported by quickr lowering")
      }
      if (length(update_window_dims)) {
        cli_abort("scatter: only scalar updates (empty update_window_dims) are supported by quickr lowering")
      }
      if (!identical(inserted_window_dims, 1L)) {
        cli_abort("scatter: only scalar updates into rank-1 inputs are supported by quickr lowering")
      }
      if (!identical(scatter_dims_to_operand_dims, 1L)) {
        cli_abort("scatter: only scatter_dims_to_operand_dims = 1L is supported by quickr lowering")
      }

      shape_idx <- as.integer(shape(idx_node$aval))
      if (length(shape_idx) != 2L || !identical(index_vector_dim, 2L) || !identical(shape_idx[[2L]], 1L)) {
        cli_abort("scatter: scatter_indices must have shape (n, 1) with index_vector_dim = 2")
      }

      n_updates <- as.integer(shape_idx[[1L]])
      shape_upd <- as.integer(shape(update_node$aval))
      if (length(shape_upd) != 1L || !identical(as.integer(shape_upd[[1L]]), n_updates)) {
        cli_abort("scatter: update must be a length-n vector matching scatter_indices")
      }

      update_comp <- params$update_computation_graph
      if (!is_graph(update_comp)) {
        cli_abort("scatter: missing update computation graph")
      }
      if (length(update_comp$inputs) != 2L || length(update_comp$outputs) != 1L) {
        cli_abort("scatter: update computation graph must be a scalar binary function")
      }

      ii <- as.name(paste0("i_", as.character(out_sym), "_scatter"))
      idx_sym <- as.name(paste0("idx_", as.character(out_sym), "_scatter"))
      old_sym <- as.name(paste0("old_", as.character(out_sym), "_scatter"))
      upd_sym <- as.name(paste0("upd_", as.character(out_sym), "_scatter"))
      new_sym <- as.name(paste0("new_", as.character(out_sym), "_scatter"))

      n_in <- as.integer(shape_in[[1L]])

      idx_expr <- rlang::call2("[", inputs[[2L]], ii, 1L)
      upd_expr <- rlang::call2("[", inputs[[3L]], ii)

      lower_update_comp <- function() {
        lowered <- quickr_lower_inline_graph(update_comp, list(old_sym, upd_sym), ctx)
        c(lowered$stmts, list(rlang::call2("<-", new_sym, lowered$out_exprs[[1L]])))
      }

      in_bounds <- rlang::call2(
        "&",
        rlang::call2(">=", idx_sym, 1L),
        rlang::call2("<=", idx_sym, n_in)
      )

      inner <- as.call(c(
        list(as.name("{")),
        list(rlang::call2("<-", idx_sym, idx_expr)),
        list(rlang::call2("<-", upd_sym, upd_expr)),
        list(as.call(list(
          as.name("if"),
          in_bounds,
          as.call(c(
            list(as.name("{")),
            c(
              list(rlang::call2("<-", old_sym, rlang::call2("[", out_sym, idx_sym))),
              lower_update_comp(),
              list(rlang::call2("<-", rlang::call2("[", out_sym, idx_sym), new_sym))
            )
          ))
        )))
      ))

      list(
        rlang::call2("<-", out_sym, inputs[[1L]]),
        as.call(list(as.name("for"), ii, rlang::call2("seq_len", n_updates), inner))
      )
    }
  )

  quickr_register_prim_lowerer(
    list(prim_add, prim_sub, prim_mul),
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      op <- switch(
        prim_name,
        add = "+",
        sub = "-",
        mul = "*",
        cli_abort("Internal error: unknown binary primitive: {.val {prim_name}}")
      )
      quickr_emit_assign(out_syms[[1L]], rlang::call2(op, inputs[[1L]], inputs[[2L]]))
    }
  )

  quickr_register_prim_lowerer(
    prim_div,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_aval <- out_avals[[1L]]
      quotient <- rlang::call2("/", inputs[[1L]], inputs[[2L]])

      if (as.character(dtype(out_aval)) == "i32") {
        return(quickr_emit_truncating_i32(out_syms[[1L]], quotient, shape(out_aval)))
      }

      quickr_emit_assign(out_syms[[1L]], quotient)
    }
  )

  quickr_register_prim_lowerer(
    prim_negate,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      quickr_emit_assign(out_syms[[1L]], rlang::call2("-", inputs[[1L]]))
    }
  )

  quickr_register_prim_lowerer(
    list(prim_eq, prim_ne, prim_gt, prim_ge, prim_lt, prim_le),
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      dt_lhs <- as.character(dtype(input_nodes[[1L]]$aval))
      dt_rhs <- as.character(dtype(input_nodes[[2L]]$aval))

      if (dt_lhs %in% "bool" || dt_rhs %in% "bool") {
        if (!prim_name %in% c("equal", "not_equal")) {
          cli_abort("{prim_name}: comparisons on {.val pred} values are not supported by quickr lowering")
        }

        a <- inputs[[1L]]
        b <- inputs[[2L]]
        not_a <- rlang::call2("!", a)
        not_b <- rlang::call2("!", b)

        eqv <- rlang::call2(
          "|",
          rlang::call2("&", a, b),
          rlang::call2("&", not_a, not_b)
        )
        xor_expr <- rlang::call2(
          "|",
          rlang::call2("&", a, not_b),
          rlang::call2("&", not_a, b)
        )

        out_expr <- if (prim_name == "equal") eqv else xor_expr
        return(quickr_emit_assign(out_syms[[1L]], out_expr))
      }

      op <- switch(
        prim_name,
        equal = "==",
        not_equal = "!=",
        greater = ">",
        greater_equal = ">=",
        less = "<",
        less_equal = "<=",
        cli_abort("Internal error: unknown comparison primitive: {.val {prim_name}}")
      )
      quickr_emit_assign(out_syms[[1L]], rlang::call2(op, inputs[[1L]], inputs[[2L]]))
    }
  )

  quickr_register_prim_lowerer(
    list(prim_and, prim_or, prim_xor),
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      dt <- as.character(dtype(input_nodes[[1L]]$aval))
      if (!dt %in% "bool") {
        cli_abort("{prim_name}: only {.val pred} dtype is supported by quickr lowering")
      }

      a <- inputs[[1L]]
      b <- inputs[[2L]]
      not_a <- rlang::call2("!", a)
      not_b <- rlang::call2("!", b)

      out_expr <- switch(
        prim_name,
        and = rlang::call2("&", a, b),
        or = rlang::call2("|", a, b),
        xor = rlang::call2(
          "|",
          rlang::call2("&", a, not_b),
          rlang::call2("&", not_a, b)
        ),
        cli_abort("Internal error: unknown boolean primitive: {.val {prim_name}}")
      )
      quickr_emit_assign(out_syms[[1L]], out_expr)
    }
  )

  quickr_register_prim_lowerer(
    prim_not,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      dt <- as.character(dtype(input_nodes[[1L]]$aval))
      if (!dt %in% "bool") {
        cli_abort("not: only {.val pred} dtype is supported by quickr lowering")
      }
      quickr_emit_assign(out_syms[[1L]], rlang::call2("!", inputs[[1L]]))
    }
  )

  quickr_register_prim_lowerer(
    prim_ifelse,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      quickr_emit_select(
        out_syms[[1L]],
        inputs[[1L]],
        inputs[[2L]],
        inputs[[3L]],
        shape(input_nodes[[1L]]$aval),
        out_avals[[1L]]
      )
    }
  )

  quickr_register_prim_lowerer(
    list(prim_abs, prim_sqrt, prim_log, prim_floor, prim_ceil, prim_exp, prim_sin, prim_cos, prim_tan),
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      fun <- switch(
        prim_name,
        sine = "sin",
        cosine = "cos",
        ceil = "ceiling",
        prim_name
      )
      quickr_emit_assign(out_syms[[1L]], rlang::call2(fun, inputs[[1L]]))
    }
  )

  quickr_register_prim_lowerer(
    prim_tanh,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      quickr_emit_assign(out_syms[[1L]], rlang::call2("tanh", inputs[[1L]]))
    }
  )

  quickr_register_prim_lowerer(
    prim_expm1,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      empty <- quickr_emit_known_empty(out_syms[[1L]], out_avals[[1L]])
      if (!is.null(empty)) {
        return(empty)
      }
      quickr_emit_assign(out_syms[[1L]], quickr_stable_expm1_expr(inputs[[1L]]))
    }
  )

  quickr_register_prim_lowerer(
    prim_log1p,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      empty <- quickr_emit_known_empty(out_syms[[1L]], out_avals[[1L]])
      if (!is.null(empty)) {
        return(empty)
      }
      quickr_emit_assign(out_syms[[1L]], quickr_stable_log1p_expr(inputs[[1L]]))
    }
  )

  quickr_register_prim_lowerer(
    prim_logistic,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      x <- inputs[[1L]]
      denom <- rlang::call2("+", 1, rlang::call2("exp", rlang::call2("-", x)))
      quickr_emit_assign(out_syms[[1L]], rlang::call2("/", 1, denom))
    }
  )

  quickr_register_prim_lowerer(
    list(prim_max, prim_min),
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      empty <- quickr_emit_known_empty(out_syms[[1L]], out_avals[[1L]])
      if (!is.null(empty)) {
        return(empty)
      }
      cmp <- if (prim_name == "maximum") ">=" else "<="
      quickr_emit_assign(
        out_syms[[1L]],
        rlang::call2("ifelse", rlang::call2(cmp, inputs[[1L]], inputs[[2L]]), inputs[[1L]], inputs[[2L]])
      )
    }
  )

  quickr_register_prim_lowerer(
    prim_pow,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_aval <- out_avals[[1L]]
      expr <- rlang::call2("^", inputs[[1L]], inputs[[2L]])

      if (identical(as.character(dtype(out_aval)), "i32")) {
        return(quickr_emit_truncating_i32(out_syms[[1L]], expr, shape(out_aval)))
      }

      quickr_emit_assign(out_syms[[1L]], expr)
    }
  )

  quickr_register_prim_lowerer(
    prim_broadcast_in_dim,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      quickr_emit_broadcast_in_dim(
        out_sym,
        inputs[[1L]],
        shape(operand_node$aval),
        params$shape,
        params$broadcast_dimensions,
        out_aval
      )
    }
  )

  quickr_register_prim_lowerer(
    prim_dot_general,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      lhs_node <- input_nodes[[1L]]
      rhs_node <- input_nodes[[2L]]
      quickr_emit_dot_general(
        out_sym,
        inputs[[1L]],
        inputs[[2L]],
        shape(lhs_node$aval),
        shape(rhs_node$aval),
        shape(out_aval),
        out_aval,
        params$contracting_dims,
        params$batching_dims
      )
    }
  )

  quickr_register_prim_lowerer(
    prim_transpose,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      quickr_emit_transpose(out_sym, inputs[[1L]], params$permutation, shape(out_aval), out_aval)
    }
  )

  quickr_register_prim_lowerer(
    prim_reshape,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      quickr_emit_reshape(out_sym, inputs[[1L]], shape(operand_node$aval), params$shape, out_aval)
    }
  )

  quickr_register_prim_lowerer(
    prim_reduce_sum,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      quickr_emit_reduce("sum", out_sym, inputs[[1L]], shape(operand_node$aval), params$dims, params$drop, out_aval)
    }
  )

  quickr_register_prim_lowerer(
    prim_reduce_prod,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      quickr_emit_reduce("prod", out_sym, inputs[[1L]], shape(operand_node$aval), params$dims, params$drop, out_aval)
    }
  )

  quickr_register_prim_lowerer(
    prim_reduce_max,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      quickr_emit_reduce("max", out_sym, inputs[[1L]], shape(operand_node$aval), params$dims, params$drop, out_aval)
    }
  )

  quickr_register_prim_lowerer(
    prim_reduce_min,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      quickr_emit_reduce("min", out_sym, inputs[[1L]], shape(operand_node$aval), params$dims, params$drop, out_aval)
    }
  )

  quickr_register_prim_lowerer(
    prim_reduce_any,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      dt_in <- as.character(dtype(operand_node$aval))
      if (!dt_in %in% "bool") {
        cli_abort("reduce_any: only {.val pred} inputs are supported by quickr lowering")
      }
      quickr_emit_reduce_boolean(
        "any",
        out_sym,
        inputs[[1L]],
        shape(operand_node$aval),
        params$dims,
        params$drop,
        out_aval
      )
    }
  )

  quickr_register_prim_lowerer(
    prim_reduce_all,
    function(prim_name, inputs, params, out_syms, input_nodes, out_avals, ctx = NULL) {
      out_sym <- out_syms[[1L]]
      out_aval <- out_avals[[1L]]
      operand_node <- input_nodes[[1L]]
      dt_in <- as.character(dtype(operand_node$aval))
      if (!dt_in %in% "bool") {
        cli_abort("reduce_all: only {.val pred} inputs are supported by quickr lowering")
      }
      quickr_emit_reduce_boolean(
        "all",
        out_sym,
        inputs[[1L]],
        shape(operand_node$aval),
        params$dims,
        params$drop,
        out_aval
      )
    }
  )
})


# Graph -> quickr-compatible R function ----------------------------------------

quickr_abort_unsupported_prims <- function(unsupported_prims, supported_prims) {
  unsupported_prims <- unique(as.character(unsupported_prims))
  if (!length(unsupported_prims)) {
    return(invisible(NULL))
  }

  cli_abort(c(
    "{.fn graph_to_quickr_function} does not support these primitives: {toString(unsupported_prims)}",
    i = "Supported primitives: {toString(sort(supported_prims))}"
  ))
}

quickr_find_unsupported_prims <- function(graph, supported_prims) {
  unsupported_prims <- character()

  for (call in graph$calls) {
    prim_name <- call$primitive$name
    if (!prim_name %in% supported_prims) {
      unsupported_prims <- c(unsupported_prims, prim_name)
    }
    if (is_higher_order_primitive(call$primitive)) {
      for (subgraph in subgraphs(call)) {
        unsupported_prims <- c(
          unsupported_prims,
          quickr_find_unsupported_prims(subgraph, supported_prims)
        )
      }
    }
  }

  unique(unsupported_prims)
}

graph_to_quickr_r_fun_impl <- function(graph, include_declare = TRUE) {
  include_declare <- as.logical(include_declare)

  if (!is_graph(graph)) {
    cli_abort("{.arg graph} must be a {.cls AnvlGraph}")
  }
  if (is.null(graph$in_tree) || is.null(graph$out_tree)) {
    cli_abort("{.arg graph} must have non-NULL {.field in_tree} and {.field out_tree}")
  }

  user_arg_names <- quickr_user_arg_names(length(graph$inputs))
  n_const <- length(graph$constants)
  const_arg_names <- if (n_const) {
    paste0("anvl_const", seq_len(n_const))
  } else {
    character()
  }
  all_arg_names <- c(user_arg_names, const_arg_names)

  prefix <- "anvl_quickr_"
  prefix_i <- 0L
  while (any(grepl(prefix, all_arg_names, fixed = TRUE))) {
    prefix_i <- prefix_i + 1L
    prefix <- paste0("anvl_quickr", prefix_i, "_")
  }

  supported_prims <- quickr_supported_prims()
  unsupported_prims <- quickr_find_unsupported_prims(graph, supported_prims)
  quickr_abort_unsupported_prims(unsupported_prims, supported_prims)

  make_formals <- function(nms) {
    as.pairlist(stats::setNames(rep(list(quote(expr = )), length(nms)), nms))
  }

  node_expr <- hashtab()
  for (i in seq_along(graph$inputs)) {
    node_expr[[graph$inputs[[i]]]] <- as.name(user_arg_names[[i]])
  }

  stmts <- list()

  if (isTRUE(include_declare)) {
    arg_avals <- c(
      lapply(graph$inputs, \(x) x$aval),
      lapply(graph$constants, \(x) x$aval)
    )
    stmts <- c(stmts, list(quickr_declare_stmt(all_arg_names, arg_avals)))
  }

  for (i in seq_along(graph$constants)) {
    const_node <- graph$constants[[i]]
    if (!is_graph_value(const_node)) {
      cli_abort("quickr lowering: graph constants must be GraphValue nodes") # nocov
    }
    if (!is_concrete_tensor(const_node$aval)) {
      cli_abort("quickr lowering: graph constants must be concrete arrays")
    }
    node_expr[[const_node]] <- as.name(const_arg_names[[i]])
  }

  tmp_i <- 0L
  new_tmp_sym <- function() {
    tmp_i <<- tmp_i + 1L
    as.name(paste0(prefix, "v", tmp_i))
  }
  ctx <- list(node_expr = node_expr, new_tmp_sym = new_tmp_sym)

  lowered <- quickr_lower_graph_calls(graph, ctx)
  stmts <- c(stmts, lowered$stmts)
  out_exprs <- lowered$out_exprs
  result_sym <- as.name(paste0(prefix, "out"))

  if (pjrt::tree_kind(graph$out_tree) == "leaf" && length(out_exprs) == 1L) {
    out_node <- graph$outputs[[1L]]
    out_expr <- out_exprs[[1L]]
    if (is_graph_value(out_node) && length(shape(out_node$aval)) == 1L) {
      out_expr <- rlang::call2("array", out_expr, dim = as.integer(shape(out_node$aval)))
    }
    stmts <- c(stmts, list(rlang::call2("<-", result_sym, out_expr), result_sym))
  } else {
    out_syms <- vector("list", length(out_exprs))
    for (i in seq_along(out_exprs)) {
      out_sym <- new_tmp_sym()
      out_syms[[i]] <- out_sym
      stmts <- c(stmts, list(rlang::call2("<-", out_sym, out_exprs[[i]])))
    }
    stmts <- c(stmts, list(as.call(c(list(as.name("list")), out_syms))))
  }

  f <- function() {}
  formals(f) <- make_formals(all_arg_names)
  body(f) <- as.call(c(list(as.name("{")), stmts))

  if (isTRUE(include_declare) && !quickr_base_has_declare()) {
    environment(f)$declare <- function(...) invisible(NULL)
  }
  f
}
