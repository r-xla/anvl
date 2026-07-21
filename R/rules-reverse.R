# length(grads) == length(outputs)
prim_add[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  grad <- grads[[1L]]
  list(
    if (required[[1L]]) grad,
    if (required[[2L]]) grad
  )
})

prim_mul[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  grad <- grads[[1L]]
  list(
    if (required[[1L]]) prim_mul(grad, rhs),
    if (required[[2L]]) prim_mul(grad, lhs)
  )
})

prim_sub[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  grad <- grads[[1L]]
  list(
    if (required[[1L]]) grad,
    if (required[[2L]]) prim_negate(grad)
  )
})

prim_negate[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  grad <- grads[[1L]]
  list(
    if (required[[1L]]) prim_negate(grad)
  )
})

prim_div[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  rhs <- inputs[[2L]]
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (required[[1L]]) prim_div(grad, rhs),
    if (required[[2L]]) prim_div(prim_mul(grad, prim_negate(y)), rhs)
  )
})

prim_remainder[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  # prim_remainder lowers to StableHLO `remainder`, which uses IEEE-754 truncating
  # semantics: y = lhs - trunc(lhs / rhs) * rhs (sign of result follows lhs). Match
  # that here — using floor() would silently disagree with the forward op for inputs
  # of opposite sign. Non-differentiable points (rhs == 0, lhs an integer multiple
  # of rhs) are ignored.
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  grad <- grads[[1L]]
  list(
    if (required[[1L]]) grad,
    if (required[[2L]]) {
      q <- prim_div(lhs, rhs)
      # trunc(q) = sign(q) * floor(|q|); valid at q == 0 too (sign(0) = 0).
      k <- prim_mul(prim_sign(q), prim_floor(prim_abs(q)))
      prim_mul(grad, prim_negate(k))
    }
  )
})

prim_pow[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (required[[1L]]) {
      one <- ones_like(lhs)
      prim_mul(prim_mul(grad, rhs), prim_pow(lhs, prim_sub(rhs, one)))
    },
    if (required[[2L]]) {
      prim_mul(grad, prim_mul(prim_log(lhs), y))
    }
  )
})

prim_log[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (required[[1L]]) prim_div(grad, x)
  )
})

prim_exp[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (required[[1L]]) prim_mul(grad, y)
  )
})

prim_sqrt[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx sqrt(x) = 1 / (2 * sqrt(x))
    if (required[[1L]]) {
      half <- prim_fill(0.5, dtype = dtype(y), shape = shape(y))
      prim_div(prim_mul(grad, half), y)
    }
  )
})

prim_rsqrt[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx 1/sqrt(x) = -0.5 * x^(-3/2) = -0.5 * rsqrt(x)^3
    if (required[[1L]]) {
      neg_half <- prim_fill(-0.5, dtype = dtype(y), shape = shape(y))
      prim_mul(prim_mul(grad, neg_half), prim_mul(y, prim_mul(y, y)))
    }
  )
})

prim_tanh[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx tanh(x) = 1 - tanh(x)^2
    if (required[[1L]]) {
      one <- prim_fill(1, dtype = dtype(y), shape = shape(y))
      prim_mul(grad, prim_sub(one, prim_mul(y, y)))
    }
  )
})

prim_tan[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx tan(x) = 1 + tan(x)^2
    if (required[[1L]]) {
      one <- prim_fill(1, dtype = dtype(y), shape = shape(y))
      prim_mul(grad, prim_add(one, prim_mul(y, y)))
    }
  )
})

prim_sin[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx sin(x) = cos(x)
    if (required[[1L]]) prim_mul(grad, prim_cos(x))
  )
})

prim_cos[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx cos(x) = -sin(x)
    if (required[[1L]]) prim_mul(grad, prim_negate(prim_sin(x)))
  )
})

prim_acos[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx acos(x) = -1 / sqrt(1 - x^2)
    if (required[[1L]]) {
      one <- prim_fill(1, dtype = dtype(x), shape = shape(x))
      denom <- prim_sqrt(prim_sub(one, prim_mul(x, x)))
      prim_negate(prim_div(grad, denom))
    }
  )
})

prim_acosh[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx acosh(x) = 1 / sqrt(x^2 - 1)
    if (required[[1L]]) {
      one <- prim_fill(1, dtype = dtype(x), shape = shape(x))
      denom <- prim_sqrt(prim_sub(prim_mul(x, x), one))
      prim_div(grad, denom)
    }
  )
})

prim_asin[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx asin(x) = 1 / sqrt(1 - x^2)
    if (required[[1L]]) {
      one <- prim_fill(1, dtype = dtype(x), shape = shape(x))
      denom <- prim_sqrt(prim_sub(one, prim_mul(x, x)))
      prim_div(grad, denom)
    }
  )
})

prim_asinh[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx asinh(x) = 1 / sqrt(1 + x^2)
    if (required[[1L]]) {
      one <- prim_fill(1, dtype = dtype(x), shape = shape(x))
      denom <- prim_sqrt(prim_add(one, prim_mul(x, x)))
      prim_div(grad, denom)
    }
  )
})

prim_atan[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx atan(x) = 1 / (1 + x^2)
    if (required[[1L]]) {
      one <- prim_fill(1, dtype = dtype(x), shape = shape(x))
      prim_div(grad, prim_add(one, prim_mul(x, x)))
    }
  )
})

prim_atanh[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx atanh(x) = 1 / (1 - x^2)
    if (required[[1L]]) {
      one <- prim_fill(1, dtype = dtype(x), shape = shape(x))
      prim_div(grad, prim_sub(one, prim_mul(x, x)))
    }
  )
})

prim_cosh[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx cosh(x) = sinh(x)
    if (required[[1L]]) prim_mul(grad, prim_sinh(x))
  )
})

prim_sinh[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx sinh(x) = cosh(x)
    if (required[[1L]]) prim_mul(grad, prim_cosh(x))
  )
})

prim_digamma[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx digamma(x) = trigamma(x) = polygamma(1, x)
    if (required[[1L]]) {
      n_one <- prim_fill(1, dtype = dtype(x), shape = shape(x))
      prim_mul(grad, prim_polygamma(n_one, x))
    }
  )
})

prim_lgamma[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx lgamma(x) = digamma(x)
    if (required[[1L]]) prim_mul(grad, prim_digamma(x))
  )
})

prim_polygamma[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  n <- inputs[[1L]]
  x <- inputs[[2L]]
  grad <- grads[[1L]]
  list(
    # gradient w.r.t. n: polygamma is typically called with integer n,
    # which is not differentiable; follow torch and refuse.
    if (required[[1L]]) cli_abort("Gradient for {.arg n} of {.fn prim_polygamma} not implemented"),
    # d/dx polygamma(n, x) = polygamma(n + 1, x)
    if (required[[2L]]) {
      one <- prim_fill(1, dtype = dtype(n), shape = shape(n))
      prim_mul(grad, prim_polygamma(prim_add(n, one), x))
    }
  )
})

prim_erf[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx erf(x) = (2 / sqrt(pi)) * exp(-x^2)
    if (required[[1L]]) {
      coef <- prim_fill(2 / sqrt(pi), dtype = dtype(x), shape = shape(x))
      prim_mul(grad, prim_mul(coef, prim_exp(prim_negate(prim_mul(x, x)))))
    }
  )
})

prim_erfc[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx erfc(x) = -(2 / sqrt(pi)) * exp(-x^2)
    if (required[[1L]]) {
      coef <- prim_fill(-2 / sqrt(pi), dtype = dtype(x), shape = shape(x))
      prim_mul(grad, prim_mul(coef, prim_exp(prim_negate(prim_mul(x, x)))))
    }
  )
})

prim_erf_inv[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx erf_inv(x) = sqrt(pi)/2 * exp(erf_inv(x)^2)
    if (required[[1L]]) {
      coef <- prim_fill(sqrt(pi) / 2, dtype = dtype(y), shape = shape(y))
      prim_mul(grad, prim_mul(coef, prim_exp(prim_mul(y, y))))
    }
  )
})

prim_abs[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx |x| = sign(x)
    if (required[[1L]]) prim_mul(grad, prim_sign(x))
  )
})

prim_max[["reverse"]] <- prim_min[["reverse"]] <- rule_reverse(function(
  inputs,
  outputs,
  grads,
  params,
  required
) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  grad <- grads[[1L]]

  y <- outputs[[1L]]
  mask_lhs <- prim_convert(prim_eq(lhs, y), dtype = dtype(grad))
  mask_rhs <- prim_convert(prim_eq(rhs, y), dtype = dtype(grad))
  count <- prim_add(mask_lhs, mask_rhs)

  list(
    if (required[[1L]]) prim_div(prim_mul(grad, mask_lhs), count),
    if (required[[2L]]) prim_div(prim_mul(grad, mask_rhs), count)
  )
})

prim_dot_general[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  grad <- grads[[1L]]

  contracting_dims <- params$contracting_dims
  batching_dims <- params$batching_dims
  precision <- params$precision

  # batching dimensions
  bd_lhs <- batching_dims[[1L]]
  bd_rhs <- batching_dims[[2L]]
  # contracting dimensions
  cd_lhs <- contracting_dims[[1L]]
  cd_rhs <- contracting_dims[[2L]]
  # remaining dimensions
  rem_dims <- function(x, b_dims, c_dims) {
    ii <- c(b_dims, c_dims)
    seq_len(ndims(x))[if (length(ii)) -ii else TRUE]
  }
  rd_lhs <- rem_dims(lhs, bd_lhs, cd_lhs)
  rd_rhs <- rem_dims(rhs, bd_rhs, cd_rhs)

  # output dimensions
  bd_out <- seq_along(bd_lhs)
  d_lhs_out <- seq_along(rd_lhs) +
    if (length(bd_out)) bd_out[length(bd_out)] else 0L
  d_rhs_out <- seq_along(rd_rhs) +
    if (length(d_lhs_out)) d_lhs_out[length(d_lhs_out)] else 0L

  conv_perm <- function(x) {
    ids_new <- integer(length(x))
    for (i in seq_along(x)) {
      ids_new[x[i]] <- i
    }
    ids_new
  }

  cd_lhs2 <- cd_lhs[order(cd_rhs)]
  perm_lhs <- conv_perm(c(bd_lhs, rd_lhs, cd_lhs2))

  cd_rhs2 <- cd_rhs[order(cd_lhs)]
  perm_rhs <- conv_perm(c(bd_rhs, rd_rhs, cd_rhs2))

  list(
    if (required[[1L]]) {
      grad_lhs <- prim_dot_general(
        grad,
        rhs,
        contracting_dims = list(d_rhs_out, rd_rhs),
        batching_dims = list(bd_out, bd_rhs),
        precision = precision
      )
      prim_transpose(grad_lhs, perm_lhs)
    },
    if (required[[2L]]) {
      grad_rhs <- prim_dot_general(
        grad,
        lhs,
        contracting_dims = list(d_lhs_out, rd_lhs),
        batching_dims = list(bd_out, bd_lhs),
        precision = precision
      )
      prim_transpose(grad_rhs, perm_rhs)
    }
  )
})

prim_transpose[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  permutation <- params$permutation
  grad <- grads[[1L]]
  inv <- integer(length(permutation))
  for (i in seq_along(permutation)) {
    inv[permutation[[i]]] <- i
  }
  list(
    if (required[[1L]]) prim_transpose(grad, inv)
  )
})

prim_reshape[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (required[[1L]]) prim_reshape(grad, shape(x))
  )
})

prim_reduce_sum[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  dims <- params$dims
  drop <- params$drop
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (required[[1L]]) {
      bdims <- if (drop) {
        without(seq_along(shape(x)), dims)
      } else {
        seq_along(shape(grad))
      }
      prim_broadcast_in_dim(grad, shape(x), bdims)
    }
  )
})

prim_reduce_max[["reverse"]] <- prim_reduce_min[["reverse"]] <- rule_reverse(function(
  inputs,
  outputs,
  grads,
  params,
  required
) {
  dims <- params$dims
  drop <- params$drop
  x <- inputs[[1L]]
  grad <- grads[[1L]]

  list(
    if (required[[1L]]) {
      bdims <- if (drop) {
        without(seq_along(shape(x)), dims)
      } else {
        seq_along(shape(grad))
      }

      y <- outputs[[1L]]
      y_bc <- prim_broadcast_in_dim(y, shape(x), bdims)

      grad_bc <- prim_broadcast_in_dim(grad, shape(x), bdims)
      mask <- prim_eq(x, y_bc)
      mask_f <- prim_convert(mask, dtype = dtype(grad_bc))

      count <- prim_reduce_sum(mask_f, dims = dims, drop = drop)
      count_bc <- prim_broadcast_in_dim(count, shape(x), bdims)

      prim_div(prim_mul(grad_bc, mask_f), count_bc)
    }
  )
})

prim_broadcast_in_dim[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  shape <- params$shape
  broadcast_dimensions <- params$broadcast_dimensions
  x <- inputs[[1L]]
  y <- outputs[[1L]]
  grad <- grads[[1L]]

  list(
    if (required[[1L]]) {
      # Sum grad over the axes that were introduced by broadcasting
      new_dims <- setdiff(seq_len(ndims(y)), broadcast_dimensions)
      expand_dims <- broadcast_dimensions[(shape(y)[broadcast_dimensions] != 1L) & (shape(x) == 1L)]
      reduce_dims <- c(new_dims, expand_dims)

      g <- if (length(reduce_dims)) prim_reduce_sum(grad, dims = reduce_dims, drop = FALSE) else grad

      # Drop the singular added dimensions
      if (length(new_dims)) {
        reshape_dims <- shape(g)
        reshape_dims <- reshape_dims[-new_dims]
        g <- prim_reshape(g, reshape_dims)
      }

      # If broadcast_dimensions are not in increasing order, reorder the
      # remaining axes back to the original input axis order.
      if (is.unsorted(broadcast_dimensions)) {
        g <- prim_transpose(g, order(broadcast_dimensions))
      }
      g
    }
  )
})

# control flow reverse ---------------------------------------------------------

prim_ifelse[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  pred <- inputs[[1L]]
  true_value <- inputs[[2L]]
  grad <- grads[[1L]]
  zero <- if (required[[2L]] || required[[3L]]) {
    zeros_like(true_value, ambiguous = TRUE)
  }

  # Predicate is boolean -- its cotangent is zero a.e. Matches the JAX
  # `select` transpose rule, which returns no contribution for `which`.
  list(
    if (required[[1L]]) zeros_like(pred, ambiguous = TRUE),
    if (required[[2L]]) prim_ifelse(pred, grad, zero),
    if (required[[3L]]) prim_ifelse(prim_not(pred), grad, zero)
  )
})

prim_if[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  cli_abort("Not yet implemented")
})

# convert reverse -----------------

prim_convert[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  # the ambiguity is determined by the input, not the `ambiguous` parameter
  list(
    if (required[[1L]]) prim_convert(grad, dtype(x), inputs[[1L]]$gnode$aval$ambiguous)
  )
})

# for comparison primitives --------------------------
# There are cases, where one wants to propagate through them, because a float was converted
# to an int/bool that eventually influenced the output
# But, we never read the final gradients of such inputs (because we can only differentiate
# with respect to floats)
# so it's okay if the dtype of the gradient does not match the input type
# Instead, we just return ambiguous zeros, that will be promoted to any dtype required

reverse_zero_bin <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad_in <- nv_fill(0L, dtype = dtype(x), shape = shape(x))

  list(
    if (required[[1L]]) grad_in,
    if (required[[2L]]) grad_in
  )
})

prim_eq[["reverse"]] <- reverse_zero_bin
prim_ne[["reverse"]] <- reverse_zero_bin
prim_gt[["reverse"]] <- reverse_zero_bin
prim_ge[["reverse"]] <- reverse_zero_bin
prim_lt[["reverse"]] <- reverse_zero_bin
prim_le[["reverse"]] <- reverse_zero_bin

# zero-grads (ignores the non-differentiable points)

reverse_zero_uni <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  list(
    if (required[[1L]]) prim_fill(0L, dtype = dtype(x), shape = shape(x))
  )
})


prim_floor[["reverse"]] <- reverse_zero_uni
prim_ceil[["reverse"]] <- reverse_zero_uni
prim_sign[["reverse"]] <- reverse_zero_uni
prim_round[["reverse"]] <- reverse_zero_uni
prim_argmax[["reverse"]] <- reverse_zero_uni
prim_argmin[["reverse"]] <- reverse_zero_uni

prim_cbrt[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx cbrt(x) = 1 / (3 * cbrt(x)^2)
    if (required[[1L]]) {
      three <- prim_fill(3, dtype = dtype(y), shape = shape(y))
      prim_div(grad, nv_mul(prim_mul(y, y), three))
    }
  )
})

prim_expm1[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx (exp(x) - 1) = exp(x)
    if (required[[1L]]) prim_mul(grad, prim_exp(x))
  )
})

prim_log1p[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx log(1 + x) = 1 / (1 + x)
    if (required[[1L]]) {
      one <- prim_fill(1, dtype = dtype(x), shape = shape(x))
      prim_div(grad, prim_add(one, x))
    }
  )
})

prim_logistic[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  y <- outputs[[1L]]
  grad <- grads[[1L]]
  list(
    # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    if (required[[1L]]) {
      one <- prim_fill(1, dtype = dtype(y), shape = shape(y))
      prim_mul(grad, prim_mul(y, prim_sub(one, y)))
    }
  )
})

prim_clamp[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  min_val <- inputs[[1L]]
  x <- inputs[[2L]]
  max_val <- inputs[[3L]]
  y <- outputs[[1L]]
  grad <- grads[[1L]]

  # because stablehlo.clamp broadcasts scalars, we need to handle this here before the eq call
  # this is an inconsistency in stablehlo, as it broadcasts scalars in clamp, but not in eq
  # (and most other functions)
  if (ndims(min_val) == 0L) {
    min_val <- prim_broadcast_in_dim(min_val, shape(x), integer())
  }
  if (ndims(max_val) == 0L) {
    max_val <- prim_broadcast_in_dim(max_val, shape(x), integer())
  }

  # the points where `x` is equal to min_val or max_val are non differentiable,
  # so we just implement it like torch, which uses 1 for the gradient there.
  mask_x <- prim_convert(prim_eq(x, y), dtype = dtype(grad))

  list(
    if (required[[1L]]) cli_abort("Gradient for min_val not implemented"),
    if (required[[2L]]) prim_mul(grad, mask_x),
    if (required[[3L]]) cli_abort("Gradient for max_val not implemented")
  )
})

prim_reverse[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  dims <- params$dims
  grad <- grads[[1L]]
  list(
    # Reverse the gradient along the same dimensions
    if (required[[1L]]) prim_reverse(grad, dims)
  )
})

prim_pad[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  edge_padding_low <- params$edge_padding_low
  edge_padding_high <- params$edge_padding_high
  interior_padding <- params$interior_padding
  grad <- grads[[1L]]
  list(
    if (required[[1L]]) {
      # select the non-padded elements
      out_shape <- shape(outputs[[1L]])
      strides <- interior_padding + 1L
      start_indices <- edge_padding_low + 1L
      limit_indices <- out_shape - edge_padding_high
      prim_static_slice(grad, start_indices, limit_indices, strides)
    },
    if (required[[2L]]) {
      cli_abort("Gradient for padding_value not implemented")
    }
  )
})

# Non-differentiable operations (logical/bitwise) return zero gradients
# (Floats might be converted to integers that are then fed into these operations, but we then
# propagate back to the floats, which requires the existence of these rules)

prim_and[["reverse"]] <- reverse_zero_bin
prim_or[["reverse"]] <- reverse_zero_bin
prim_xor[["reverse"]] <- reverse_zero_bin
prim_not[["reverse"]] <- reverse_zero_uni

prim_shift_left[["reverse"]] <- reverse_zero_bin
prim_shift_right_arithmetic[["reverse"]] <- reverse_zero_bin
prim_shift_right_logical[["reverse"]] <- reverse_zero_bin

prim_is_finite[["reverse"]] <- reverse_zero_uni
prim_popcnt[["reverse"]] <- reverse_zero_uni

prim_reduce_all[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  list(
    if (required[[1L]]) prim_fill(FALSE, dtype = dtype(x), shape = shape(x))
  )
})

prim_reduce_any[["reverse"]] <- prim_reduce_all[["reverse"]]

prim_bitcast_convert[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  list(
    if (required[[1L]]) prim_fill(0L, dtype = dtype(x), shape = shape(x))
  )
})

prim_atan2[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  grad <- grads[[1L]]
  denom <- prim_add(prim_mul(lhs, lhs), prim_mul(rhs, rhs))
  list(
    if (required[[1L]]) prim_div(prim_mul(grad, rhs), denom),
    if (required[[2L]]) prim_div(prim_mul(grad, prim_negate(lhs)), denom)
  )
})

# sort reverse: route each upstream gradient back to original positions via
# the inverse permutation.
#
# For y = sort(key)[forward perm π], we have y[i] = key[π[i]], so the gradient
# satisfies dkey[k] = dy[σ[k]] where σ = π^{-1} (= argsort(π)).
#
# Implementation trick: instead of computing σ explicitly and gathering, sort
# (π, grad) ascending — the second output is grad permuted by argsort(π) = σ,
# which is exactly the input gradient.
prim_sort[["reverse"]] <- rule_reverse(forward = function(inputs, params) {
  dim <- params$dim
  descending <- params$descending
  is_stable <- params$is_stable

  key <- inputs[[1L]]
  iota <- prim_iota(dim = dim, dtype = "i64", shape = shape(key), start = 1L)
  sorted <- prim_sort(
    c(inputs, list(iota)),
    dim = dim,
    descending = descending,
    is_stable = is_stable
  )
  perm <- sorted[[length(sorted)]]
  outputs <- sorted[seq_along(inputs)]

  list(
    outputs = outputs,
    backward = function(inputs, outputs, grads, params, required) {
      # Route all required gradients through a single argsort-by-perm: sort
      # (perm, grad_1, ..., grad_K) ascending. Each carried output is then
      # `grad_j` permuted by argsort(perm) = sigma, which is exactly the
      # gradient for the j-th input.
      required_idx <- which(required)
      inv <- prim_sort(
        c(list(perm), grads[required_idx]),
        dim = dim,
        descending = FALSE,
        is_stable = FALSE
      )

      out <- vector("list", length(inputs))
      for (k in seq_along(required_idx)) {
        out[[required_idx[[k]]]] <- inv[[k + 1L]]
      }
      out
    }
  )
})

# top_k reverse: scatter the values gradient at the forward indices into a
# zero buffer of the input shape. Using the actual forward indices (rather
# than recomputing a permutation) is correct by construction even when
# `x` has duplicate values — top-k indices are pairwise unique along
# the last dim.
prim_top_k[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  if (!required[[1L]]) {
    return(list(NULL))
  }
  x <- inputs[[1L]]
  grad_values <- grads[[1L]]
  indices <- outputs[[2L]]
  full_shape <- shape(x)
  rank <- length(full_shape)
  batching <- seq_len(rank - 1L)

  zero_input <- prim_fill(
    0,
    dtype = dtype(grad_values),
    shape = full_shape,
    ambiguous = FALSE
  )

  # All dimensions are iteration dimensions and we have a single index vector dimension that
  # indicates the position to write to (in the last dimension of the input)
  list(prim_scatter(
    input = zero_input,
    # index vectors selecting positions along the last dim of `input`
    scatter_indices = indices,
    update = grad_values,
    update_window_dims = integer(0),
    inserted_window_dims = rank,
    input_batching_dims = batching,
    scatter_indices_batching_dims = batching,
    scatter_dims_to_operand_dims = rank,
    # we sort along a single dimension -> all dims are iteration dims
    index_vector_dim = rank + 1L,
    unique_indices = TRUE
  ))
})

# concatenate reverse: split the gradient back along the concatenation dimension
prim_concatenate[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  dimension <- params$dimension
  grad <- grads[[1L]]
  n_inputs <- length(inputs)
  input_grads <- vector("list", n_inputs)

  offset <- 1L
  limit_indices <- shape(grad)
  start_indices <- rep(1L, length(shape(inputs[[1]])))
  for (i in seq_len(n_inputs)) {
    input_shape <- shape(inputs[[i]])
    dim_size <- input_shape[dimension]
    if (required[[i]]) {
      strides <- rep(1L, length(input_shape))
      start_indices[dimension] <- offset
      limit_indices[dimension] <- offset + dim_size - 1L
      input_grads[[i]] <- prim_static_slice(grad, start_indices, limit_indices, strides)
    }
    offset <- offset + dim_size
  }
  input_grads
})


prim_reduce_prod[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  if (!required[[1L]]) {
    return(list(NULL))
  }
  dims <- params$dims
  drop <- params$drop
  x <- inputs[[1L]]
  grad <- grads[[1L]]

  # Move reduced axes to the end and collapse them, so the safe formula
  # below operates along a single trailing axis.
  s <- shape(x)
  non_reduced <- without(seq_along(s), dims)
  perm <- c(non_reduced, dims)
  reduced_size <- as.integer(prod(s[dims]))
  collapsed_shape <- c(s[non_reduced], reduced_size)
  rank <- length(collapsed_shape)

  operand_c <- prim_reshape(prim_transpose(x, perm), collapsed_shape)
  grad_squeezed <- if (drop) grad else prim_reshape(grad, s[non_reduced])
  grad_bc <- prim_broadcast_in_dim(grad_squeezed, collapsed_shape, seq_len(rank - 1L))

  out_c <- if (reduced_size <= 1L) {
    grad_bc
  } else {
    # PyTorch's prod_safe_zeros_backward: the gradient at position k along the
    # reduced axis is the product of all *other* elements along that axis,
    # decomposed as (exclusive cumprod left) * (exclusive cumprod right). No
    # division, smooth in the inputs, hence safe at zeros and twice-differentiable.
    n <- reduced_size
    ones_shape <- collapsed_shape
    ones_shape[rank] <- 1L
    ones <- prim_fill(1, dtype = dtype(x), shape = ones_shape)
    full_strides <- rep(1L, rank)

    starts_n <- rep(1L, rank)
    limits_n <- collapsed_shape
    limits_n[rank] <- n - 1L
    narrow_n <- prim_static_slice(operand_c, starts_n, limits_n, full_strides)
    exclusive_normal <- prim_cumprod(
      prim_concatenate(ones, narrow_n, dimension = rank),
      dim = rank
    )

    starts_r <- rep(1L, rank)
    starts_r[rank] <- 2L
    narrow_r <- prim_reverse(
      prim_static_slice(operand_c, starts_r, collapsed_shape, full_strides),
      dims = rank
    )
    exclusive_reverse <- prim_reverse(
      prim_cumprod(prim_concatenate(ones, narrow_r, dimension = rank), dim = rank),
      dims = rank
    )

    prim_mul(grad_bc, prim_mul(exclusive_normal, exclusive_reverse))
  }

  out <- prim_transpose(prim_reshape(out_c, s[perm]), order(perm))
  list(out)
})

# cumulative (scan) reverse rules ----------------------------------------------

prim_cumsum[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  dim <- params$dim
  grad <- grads[[1L]]
  list(
    # d/dx_i sum_{k<=j} x_k = 1[i<=j], so grad_i = sum_{j>=i} grad_out_j
    # which is reverse-cumsum of grad_out along dim.
    if (required[[1L]]) prim_reverse(prim_cumsum(prim_reverse(grad, dim), dim), dim)
  )
})

# Because we compute indices in the forward pass, we simply scatter
# into the zeros
# This is also how PyTorch does it
.cum_extreme_reverse <- function(inputs, outputs, grads, params, required) {
  if (!required[[1L]]) {
    return(list(NULL))
  }
  dim <- params$dim
  x <- inputs[[1L]]
  indices <- outputs[[2L]]
  grad <- grads[[1L]]
  rank <- length(shape(x))

  batching <- seq_len(rank)[-dim]
  list(prim_scatter(
    input = zeros_like(x),
    scatter_indices = indices,
    update = grad,
    update_window_dims = integer(0),
    inserted_window_dims = dim,
    input_batching_dims = batching,
    scatter_indices_batching_dims = batching,
    scatter_dims_to_operand_dims = dim,
    index_vector_dim = rank + 1L,
    unique_indices = FALSE,
    update_computation = prim_add
  ))
}

prim_cummax[["reverse"]] <- rule_reverse(.cum_extreme_reverse)
prim_cummin[["reverse"]] <- rule_reverse(.cum_extreme_reverse)

# slice reverse: pad with zeros
prim_static_slice[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  start_indices <- params$start_indices
  limit_indices <- params$limit_indices
  strides <- params$strides
  x <- inputs[[1L]]
  grad <- grads[[1L]]
  list(
    if (required[[1L]]) {
      input_shape <- shape(x)
      grad_shape <- shape(grad)
      edge_padding_low <- start_indices - 1L
      edge_padding_high <- input_shape - start_indices - (grad_shape - 1L) * strides
      interior_padding <- strides - 1L
      prim_pad(
        grad,
        zeros(dtype(grad), integer(), FALSE),
        edge_padding_low,
        edge_padding_high,
        interior_padding
      )
    }
  )
})

prim_dynamic_slice[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  x <- inputs[[1L]]
  start_indices <- inputs[-1L]
  grad <- grads[[1L]]

  result <- vector("list", length(inputs))

  # dynamic_update_slice does the same clamping as dynamic_slice, so it naturally handles
  # out of bound indices (in the same weird way)
  if (required[[1L]]) {
    zeros <- zeros_like(x)
    result[[1L]] <- rlang::exec(prim_dynamic_update_slice, zeros, grad, !!!start_indices)
  }

  result
})

prim_dynamic_update_slice[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  update <- inputs[[2L]]
  start_indices <- inputs[-(1:2)]
  grad <- grads[[1L]]

  result <- vector("list", length(inputs))

  # dynamic_update_slice does the same clamping as dynamic_slice, so it naturally handles
  # out of bound indices (in the same weird way)
  if (required[[1L]]) {
    zeros <- zeros_like(update)
    result[[1L]] <- rlang::exec(prim_dynamic_update_slice, grad, zeros, !!!start_indices)
  }

  if (required[[2L]]) {
    result[[2L]] <- rlang::exec(prim_dynamic_slice, grad, !!!start_indices, slice_sizes = shape(update))
  }

  result
})

prim_gather[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  slice_sizes <- params$slice_sizes
  offset_dims <- params$offset_dims
  collapsed_slice_dims <- params$collapsed_slice_dims
  operand_batching_dims <- params$operand_batching_dims
  start_indices_batching_dims <- params$start_indices_batching_dims
  start_index_map <- params$start_index_map
  index_vector_dim <- params$index_vector_dim
  indices_are_sorted <- params$indices_are_sorted
  unique_indices <- params$unique_indices

  x <- inputs[[1L]]
  start_indices <- inputs[[2L]]
  grad <- grads[[1L]]

  # Scatter is basically the "reverse" of gather
  # We have to take care of two things here:
  # 1. Out-of-bounds indices:
  #    Let's say we have an `x` of shape (5, ) and in the forward pass do y = x[6]
  #    which is clamped to x[5]
  #    If dout/y = g; then dout/dx = c(0, 0, 0, 0, g)
  #    But if we just do the reverse x[6] <- g, this gets lost
  #    (scatter can ignore out-of-bounds indices because the output shape is determined by the input shape)
  #    So we need to clamp the start_indices to what they actually were.
  # 2. Multiple reads (x[array(c(1L, 1L)), 2])
  #    --> accumulate the gradients using update prim_add

  list(
    if (required[[1L]]) {
      # Clamp indices to valid range (same as forward pass clamping behavior)
      scatter_indices <- gather_clamp_indices(
        start_indices = start_indices,
        operand_shape = shape(x),
        slice_sizes = slice_sizes,
        start_index_map = start_index_map,
        index_vector_dim = index_vector_dim
      )

      prim_scatter(
        input = zeros_like(x),
        scatter_indices = scatter_indices,
        update = grad,
        update_window_dims = offset_dims,
        inserted_window_dims = collapsed_slice_dims,
        input_batching_dims = operand_batching_dims,
        scatter_indices_batching_dims = start_indices_batching_dims,
        scatter_dims_to_operand_dims = start_index_map,
        index_vector_dim = index_vector_dim,
        indices_are_sorted = indices_are_sorted,
        unique_indices = unique_indices,
        # Use addition to accumulate gradients when multiple gather positions
        # read from the same source location
        update_computation = prim_add
      )
    },
    if (required[[2L]]) zeros_like(start_indices)
  )
})


prim_scatter[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  update_window_dims <- params$update_window_dims
  inserted_window_dims <- params$inserted_window_dims
  input_batching_dims <- params$input_batching_dims
  scatter_indices_batching_dims <- params$scatter_indices_batching_dims
  scatter_dims_to_operand_dims <- params$scatter_dims_to_operand_dims
  index_vector_dim <- params$index_vector_dim
  indices_are_sorted <- params$indices_are_sorted
  unique_indices <- params$unique_indices
  update_computation_graph <- params$update_computation_graph

  input <- inputs[[1L]]
  scatter_indices <- inputs[[2L]]
  update <- inputs[[3L]]
  grad <- grads[[1L]]

  if (!identical(update_computation_graph$outputs[[1L]], update_computation_graph$inputs[[2L]])) {
    cli_abort("Scatter reverse only supports simple replacement (update_computation = function(old, new) new)")
  }

  # Generally, the reverse of scatter is:
  # - for the update: gather from the gradient
  # - for the input: zero out the positions that were overwritten

  # The only problem is when scatter writes multiple times to the same position (unique_indices = FALSE),
  # XLA doesn't guarantee which update "wins" at each position. We use the ID trick from JAX's scatter JVP:
  # https://github.com/jax-ml/jax/blob/ecd3795959e91c3c28cec8696f4c82f2a28bc086/jax/_src/lax/slicing.py

  update_shape <- shape(update)
  input_shape <- shape(input)

  slice_sizes <- scatter_to_gather_slice_sizes(
    update_shape = update_shape,
    input_shape = input_shape,
    update_window_dims = update_window_dims,
    inserted_window_dims = inserted_window_dims,
    input_batching_dims = input_batching_dims
  )

  list(
    # Gradient for input: zero out overwritten positions.
    # Works for both unique and non-unique: scattering zero is idempotent.
    if (required[[1L]]) {
      prim_scatter(
        input = grad,
        scatter_indices = scatter_indices,
        update = zeros_like(update),
        update_window_dims = update_window_dims,
        inserted_window_dims = inserted_window_dims,
        input_batching_dims = input_batching_dims,
        scatter_indices_batching_dims = scatter_indices_batching_dims,
        scatter_dims_to_operand_dims = scatter_dims_to_operand_dims,
        index_vector_dim = index_vector_dim,
        indices_are_sorted = indices_are_sorted,
        unique_indices = unique_indices,
        update_computation = function(old, new) new
      )
    },
    # Gradient for scatter_indices: not differentiable
    if (required[[2L]]) zeros_like(scatter_indices),
    # Gradient for update: gather from gradient at update positions
    if (required[[3L]]) {
      if (unique_indices) {
        prim_gather(
          x = grad,
          start_indices = scatter_indices,
          slice_sizes = slice_sizes,
          offset_dims = update_window_dims,
          collapsed_slice_dims = inserted_window_dims,
          operand_batching_dims = input_batching_dims,
          start_indices_batching_dims = scatter_indices_batching_dims,
          start_index_map = scatter_dims_to_operand_dims,
          index_vector_dim = index_vector_dim,
          indices_are_sorted = indices_are_sorted,
          unique_indices = TRUE
        )
      } else {
        # Non-unique indices: use the ID trick to determine which updates won.
        # https://github.com/jax-ml/jax/blob/ecd3795959e91c3c28cec8696f4c82f2a28bc086/jax/_src/lax/slicing.py
        # Example: (x has shape (2,)) x[c(1, 1)] <- c(2, 3) --> we don't know whether 2 or 2 was assigned to position 1.
        # -->
        # 1. x <- c(0, 0)
        # 2. Assign unique indices x[c(1, 1)] <- c(1, 2)
        # 3. gather the winners: x[c(1, 1)], which e.g. gives c(2, 2)
        # 4. Compare c(1, 2) == c(2, 2) = c(FALSE, TRUE) -> second one wins
        # 5. Mask out the losers; grad[1] <- 0

        # a) Create unique positive IDs for each update "batch" position
        ids_shape <- update_shape
        ids_shape[update_window_dims] <- 1L
        num_ids <- prod(ids_shape)
        id_dtype <- "i64"
        update_ids <- prim_reshape(prim_iota(1L, id_dtype, num_ids, start = 1L), ids_shape)
        update_ids <- prim_broadcast_in_dim(update_ids, update_shape, seq_along(update_shape))

        # b) Scatter IDs to see which update "wins" at each position
        scattered_ids <- prim_scatter(
          input = prim_fill(0L, dtype = id_dtype, shape = input_shape),
          scatter_indices = scatter_indices,
          update = update_ids,
          update_window_dims = update_window_dims,
          inserted_window_dims = inserted_window_dims,
          input_batching_dims = input_batching_dims,
          scatter_indices_batching_dims = scatter_indices_batching_dims,
          scatter_dims_to_operand_dims = scatter_dims_to_operand_dims,
          index_vector_dim = index_vector_dim,
          indices_are_sorted = indices_are_sorted,
          unique_indices = FALSE,
          update_computation = function(old, new) new
        )

        # c) Gather scattered IDs back to update positions
        gathered_ids <- prim_gather(
          x = scattered_ids,
          start_indices = scatter_indices,
          slice_sizes = slice_sizes,
          offset_dims = update_window_dims,
          collapsed_slice_dims = inserted_window_dims,
          operand_batching_dims = input_batching_dims,
          start_indices_batching_dims = scatter_indices_batching_dims,
          start_index_map = scatter_dims_to_operand_dims,
          index_vector_dim = index_vector_dim,
          indices_are_sorted = indices_are_sorted,
          unique_indices = FALSE
        )

        # d) Gather gradient and mask: only winning updates get gradient
        grad_at_positions <- prim_gather(
          x = grad,
          start_indices = scatter_indices,
          slice_sizes = slice_sizes,
          offset_dims = update_window_dims,
          collapsed_slice_dims = inserted_window_dims,
          operand_batching_dims = input_batching_dims,
          start_indices_batching_dims = scatter_indices_batching_dims,
          start_index_map = scatter_dims_to_operand_dims,
          index_vector_dim = index_vector_dim,
          indices_are_sorted = indices_are_sorted,
          unique_indices = FALSE
        )
        mask <- prim_eq(update_ids, gathered_ids)
        prim_ifelse(mask, grad_at_positions, zeros_like(grad_at_positions))
      }
    }
  )
})

diag_mask <- function(n) {
  prim_iota(dim = 1L, dtype = "i32", shape = c(n, n), start = 0L) ==
    prim_iota(dim = 2L, dtype = "i32", shape = c(n, n), start = 0L)
}

triangular_mask <- function(n, lower, unit_diagonal) {
  mask <- tri_mask(c(n, n), 0L, lower = lower)
  if (unit_diagonal) {
    prim_ifelse(diag_mask(n), prim_fill(FALSE, dtype = "bool", shape = c(n, n)), mask)
  } else {
    mask
  }
}

#' @name prim_chol
#' @rdname prim_chol
#' @references
#' `r xlamisc::format_bib("murray2016differentiation", "walter2012structured")`
prim_chol[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  lower <- params$lower
  if (length(shape(outputs[[1L]])) > 2L) {
    cli_abort("Batched cholesky gradient is not yet supported.")
  }

  # The Jacobian for cholesky is actually not unique.
  # This is, because the derivative property only needs to hold for valid input perturbations.
  # i.e., we want cholesky(x + dx) \approx cholesky(x) + dcholesky(x) %*% dx
  # but here x + dx needs to be positive semi-definite.
  # this is satisfied if dx is symmetric (dx is small)
  # Because the property only needs to hold for symmetric inputs dx, their are an infinite number
  # of solutions.
  # We use the solution that is identical to the one used by {torch} (which we compare against
  # in the test)

  L <- outputs[[1L]]
  grad <- grads[[1L]]
  n <- shape(L)[length(shape(L))]

  # Phi: keep lower triangle, halve diagonal, zero upper triangle
  phi <- function(x) {
    eye <- prim_convert(diag_mask(n), dtype = dtype(x))
    one <- prim_fill(1, dtype = dtype(x), shape = shape(x))
    x <- prim_div(x, prim_add(one, eye))
    prim_ifelse(tri_mask(c(n, n), 0L, lower = TRUE), x, zeros_like(x))
  }

  # The stablehlo lowering already zeros out the non-triangular part of L,
  # so grad is clean and no masking is needed here.

  # For upper triangular, transpose to work with the lower-triangular algorithm.
  # No transpose back needed at the end: grad_A is symmetric.
  if (!lower) {
    L <- t(L)
    grad <- t(grad)
  }
  # We almost use Murray (2016) equation 10, but use M / 2 intead of phi(M) which are both
  # equivalent for symmetric perturbations
  P <- phi(nv_matmul(t(L), grad))
  half <- prim_fill(0.5, dtype = dtype(P), shape = shape(P))
  S <- prim_mul(prim_add(P, t(P)), half)
  S <- prim_triangular_solve(L, S, left_side = TRUE, lower = TRUE, unit_diagonal = FALSE, transpose_a = TRUE)
  grad_A <- prim_triangular_solve(
    L,
    S,
    left_side = FALSE,
    lower = TRUE,
    unit_diagonal = FALSE,
    transpose_a = FALSE
  )

  list(grad_A)
})

#' @name prim_triangular_solve
#' @rdname prim_triangular_solve
#' @references
#' `r xlamisc::format_bib("giles2008extended")`
prim_triangular_solve[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
  left_side <- params$left_side
  lower <- params$lower
  unit_diagonal <- params$unit_diagonal
  transpose_a <- params$transpose_a

  a <- inputs[[1L]]
  x <- outputs[[1L]]
  grad <- grads[[1L]]

  if (length(shape(a)) > 2L) {
    cli_abort("Batched triangular_solve gradient is not yet supported.")
  }

  # op(A) is A or A^T depending on transpose_a
  adj_transpose <- !transpose_a

  if (left_side) {
    # From the paper: grad_b = op(A)^{-1} * grad
    grad_b <- prim_triangular_solve(
      a,
      grad,
      left_side = TRUE,
      lower = lower,
      unit_diagonal = unit_diagonal,
      transpose_a = adj_transpose
    )
    # From the paper: dx/da = -A^{-1} * dA * A^{-1} * B
    # Similar to cholesky, where the perturbation needs to be symmetric, here the perturbation
    # of A can be assumed to be a lower triangular matrix.
    # Out derivative therefore only works when the input is a lower triangular matrix.
    # I.e. dy/dx_{i, j} = 0 for the upper/lower triangular matrix
    # But because the contangent can be anything, we need to handle this here.
    # (we could either modify the contangent or mask the results; we choose the latter)
    # Also if unit_diagonal = TRUE the diagonal elements are not read and hence also have no effect
    # i.e. we also zero out those.
    grad_a <- if (required[[1L]]) {
      raw <- prim_negate(nv_matmul(grad_b, t(x)))
      n <- shape(a)[length(shape(a))]
      mask <- triangular_mask(n, lower, unit_diagonal)
      if (transpose_a) {
        raw <- t(raw)
      }
      prim_ifelse(mask, raw, zeros_like(raw))
    }
    if (!required[[2L]]) grad_b <- NULL
  } else {
    # Right side: x @ op(a) = b
    grad_b <- prim_triangular_solve(
      a,
      grad,
      left_side = FALSE,
      lower = lower,
      unit_diagonal = unit_diagonal,
      transpose_a = adj_transpose
    )
    # Same masking logic as the left_side case above.
    grad_a <- if (required[[1L]]) {
      raw <- prim_negate(nv_matmul(t(x), grad_b))
      n <- shape(a)[length(shape(a))]
      mask <- triangular_mask(n, lower, unit_diagonal)
      if (transpose_a) {
        raw <- t(raw)
      }
      prim_ifelse(mask, raw, zeros_like(raw))
    }
    if (!required[[2L]]) grad_b <- NULL
  }

  list(grad_a, grad_b)
})
