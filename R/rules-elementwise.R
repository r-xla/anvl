# Derivatives of the elementwise primitives -----------------------------------
#
# An output element of an elementwise primitive depends only on the elements
# at the same position of its inputs, so its derivative with respect to each
# input is diagonal: multiplying by it is the same operation in both
# directions. One definition therefore gives a primitive both its reverse rule,
# which multiplies the cotangent of the output by it, and its pushforward
# rule, which multiplies the tangent of each input by it and adds them up.

# Sets the reverse and pushforward rule of the elementwise primitive `prim` from
# `partials`, a `function(inputs, outputs, params)` returning one entry per
# input: the function that multiplies an array of the output's shape by the
# partial derivative of the output with respect to that input, or `NULL` where
# that derivative is zero. The functions receive a cotangent of the output in
# the reverse rule and a tangent of their input in the pushforward rule; anything
# they share is computed in the body of `partials`.
set_elementwise_derivative <- function(prim, partials) {
  def <- primitive_def(prim)
  def[["reverse"]] <- rule_reverse(function(inputs, outputs, grads, params, required) {
    maps <- partials(inputs, outputs, params)
    Map(
      function(x, map, required) {
        if (!required) {
          NULL
        } else if (is.null(map)) {
          zeros_like(x)
        } else {
          map(grads[[1L]])
        }
      },
      inputs,
      maps,
      required
    )
  })
  def[["pushforward"]] <- function(inputs, outputs, tangents, params) {
    maps <- partials(inputs, outputs, params)
    terms <- Map(function(map, tangent) if (!is.null(map) && !is.null(tangent)) map(tangent), maps, tangents)
    terms <- Filter(Negate(is.null), terms)
    list(if (length(terms)) Reduce(prim_add, terms))
  }
  invisible(prim)
}

# The derivative of a step function -- or of one whose output has none, such as
# a comparison -- is zero wherever it exists.
no_derivative <- function(inputs, outputs, params) {
  vector("list", length(inputs))
}

# arithmetic -------------------------------------------------------------------

set_elementwise_derivative(prim_add, function(inputs, outputs, params) {
  list(identity, identity)
})

set_elementwise_derivative(prim_sub, function(inputs, outputs, params) {
  list(identity, prim_negate)
})

set_elementwise_derivative(prim_negate, function(inputs, outputs, params) {
  list(prim_negate)
})

set_elementwise_derivative(prim_mul, function(inputs, outputs, params) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  list(
    \(v) prim_mul(v, rhs),
    \(v) prim_mul(v, lhs)
  )
})

set_elementwise_derivative(prim_div, function(inputs, outputs, params) {
  rhs <- inputs[[2L]]
  y <- outputs[[1L]]
  list(
    \(v) prim_div(v, rhs),
    \(v) prim_div(prim_mul(v, prim_negate(y)), rhs)
  )
})

set_elementwise_derivative(prim_remainder, function(inputs, outputs, params) {
  # prim_remainder lowers to StableHLO `remainder`, which truncates the quotient:
  # the result is x - trunc(x / y) * y, so it follows the sign of x.
  # Match that here — using floor() would silently disagree with the forward op
  # for inputs of opposite sign. Non-differentiable points (y == 0, x an
  # integer multiple of y) are ignored.
  x <- inputs[[1L]]
  y <- inputs[[2L]]
  list(
    identity,
    \(v) {
      q <- prim_div(x, y)
      # trunc(q) = sign(q) * floor(|q|); valid at q == 0 too (sign(0) = 0).
      k <- prim_mul(prim_sign(q), prim_floor(prim_abs(q)))
      prim_mul(v, prim_negate(k))
    }
  )
})

set_elementwise_derivative(prim_pow, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  y <- inputs[[2L]]
  out <- outputs[[1L]]
  # As in JAX, the derivatives are 0 where they would otherwise read 0^-1 or
  # log(0) at base 0: x^0 is constant in x, and 0^y is constant in y.
  list(
    \(v) {
      dx <- prim_mul(y, prim_pow(x, prim_sub(y, ones_like(y))))
      prim_mul(v, prim_ifelse(prim_eq(y, zeros_like(y)), zeros_like(dx), dx))
    },
    \(v) {
      safe_x <- prim_ifelse(prim_eq(x, zeros_like(x)), ones_like(x), x)
      prim_mul(v, prim_mul(prim_log(safe_x), out))
    }
  )
})

set_elementwise_derivative(prim_abs, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx |x| = sign(x)
  list(\(v) prim_mul(v, prim_sign(x)))
})

pmax_partials <- function(inputs, outputs, params) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  y <- outputs[[1L]]
  # At a tie, each input gets half the derivative.
  mask_lhs <- prim_convert(prim_eq(lhs, y), dtype = dtype(y))
  mask_rhs <- prim_convert(prim_eq(rhs, y), dtype = dtype(y))
  count <- prim_add(mask_lhs, mask_rhs)
  list(
    \(v) prim_div(prim_mul(v, mask_lhs), count),
    \(v) prim_div(prim_mul(v, mask_rhs), count)
  )
}

set_elementwise_derivative(prim_pmax, pmax_partials)
set_elementwise_derivative(prim_pmin, pmax_partials)

set_elementwise_derivative(prim_clamp, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  y <- outputs[[1L]]
  # the points where `x` is equal to `min` or `max` are non differentiable,
  # so we just implement it like torch, which uses 1 for the derivative there.
  mask_x <- prim_convert(prim_eq(x, y), dtype = dtype(x))
  list(
    \(v) prim_mul(v, mask_x),
    \(v) cli_abort("The derivative with respect to {.arg min} is not implemented."),
    \(v) cli_abort("The derivative with respect to {.arg max} is not implemented.")
  )
})

set_elementwise_derivative(prim_ifelse, function(inputs, outputs, params) {
  test <- inputs[[1L]]
  # The predicate is boolean and has no derivative, as in JAX's `select`.
  list(
    NULL,
    \(v) prim_ifelse(test, v, zeros_like(v)),
    \(v) prim_ifelse(test, zeros_like(v), v)
  )
})

# exponential and logarithm ----------------------------------------------------

set_elementwise_derivative(prim_exp, function(inputs, outputs, params) {
  y <- outputs[[1L]]
  list(\(v) prim_mul(v, y))
})

set_elementwise_derivative(prim_expm1, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx (exp(x) - 1) = exp(x)
  list(\(v) prim_mul(v, prim_exp(x)))
})

set_elementwise_derivative(prim_log, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  list(\(v) prim_div(v, x))
})

set_elementwise_derivative(prim_log1p, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx log(1 + x) = 1 / (1 + x)
  list(\(v) prim_div(v, prim_add(ones_like(x), x)))
})

set_elementwise_derivative(prim_plogis, function(inputs, outputs, params) {
  y <- outputs[[1L]]
  # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
  list(\(v) prim_mul(v, prim_mul(y, prim_sub(ones_like(y), y))))
})

# roots ------------------------------------------------------------------------

set_elementwise_derivative(prim_sqrt, function(inputs, outputs, params) {
  y <- outputs[[1L]]
  # d/dx sqrt(x) = 1 / (2 * sqrt(x))
  list(\(v) {
    half <- prim_fill(0.5, dtype = dtype(y), shape = shape(y))
    prim_div(prim_mul(v, half), y)
  })
})

set_elementwise_derivative(prim_rsqrt, function(inputs, outputs, params) {
  y <- outputs[[1L]]
  # d/dx 1/sqrt(x) = -0.5 * x^(-3/2) = -0.5 * rsqrt(x)^3
  list(\(v) {
    neg_half <- prim_fill(-0.5, dtype = dtype(y), shape = shape(y))
    prim_mul(prim_mul(v, neg_half), prim_mul(y, prim_mul(y, y)))
  })
})

set_elementwise_derivative(prim_cbrt, function(inputs, outputs, params) {
  y <- outputs[[1L]]
  # d/dx cbrt(x) = 1 / (3 * cbrt(x)^2)
  list(\(v) {
    three <- prim_fill(3L, dtype = dtype(y), shape = shape(y))
    prim_div(v, prim_mul(prim_mul(y, y), three))
  })
})

# trigonometric and hyperbolic -------------------------------------------------

set_elementwise_derivative(prim_sin, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx sin(x) = cos(x)
  list(\(v) prim_mul(v, prim_cos(x)))
})

set_elementwise_derivative(prim_cos, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx cos(x) = -sin(x)
  list(\(v) prim_mul(v, prim_negate(prim_sin(x))))
})

set_elementwise_derivative(prim_tan, function(inputs, outputs, params) {
  y <- outputs[[1L]]
  # d/dx tan(x) = 1 + tan(x)^2
  list(\(v) prim_mul(v, prim_add(ones_like(y), prim_mul(y, y))))
})

set_elementwise_derivative(prim_acos, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx acos(x) = -1 / sqrt(1 - x^2)
  list(\(v) prim_negate(prim_div(v, prim_sqrt(prim_sub(ones_like(x), prim_mul(x, x))))))
})

set_elementwise_derivative(prim_asin, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx asin(x) = 1 / sqrt(1 - x^2)
  list(\(v) prim_div(v, prim_sqrt(prim_sub(ones_like(x), prim_mul(x, x)))))
})

set_elementwise_derivative(prim_atan, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx atan(x) = 1 / (1 + x^2)
  list(\(v) prim_div(v, prim_add(ones_like(x), prim_mul(x, x))))
})

set_elementwise_derivative(prim_atan2, function(inputs, outputs, params) {
  y <- inputs[[1L]]
  x <- inputs[[2L]]
  denom <- prim_add(prim_mul(y, y), prim_mul(x, x))
  list(
    \(v) prim_div(prim_mul(v, x), denom),
    \(v) prim_div(prim_mul(v, prim_negate(y)), denom)
  )
})

set_elementwise_derivative(prim_sinh, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx sinh(x) = cosh(x)
  list(\(v) prim_mul(v, prim_cosh(x)))
})

set_elementwise_derivative(prim_cosh, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx cosh(x) = sinh(x)
  list(\(v) prim_mul(v, prim_sinh(x)))
})

set_elementwise_derivative(prim_tanh, function(inputs, outputs, params) {
  y <- outputs[[1L]]
  # d/dx tanh(x) = 1 - tanh(x)^2
  list(\(v) prim_mul(v, prim_sub(ones_like(y), prim_mul(y, y))))
})

set_elementwise_derivative(prim_asinh, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx asinh(x) = 1 / sqrt(1 + x^2)
  list(\(v) prim_div(v, prim_sqrt(prim_add(ones_like(x), prim_mul(x, x)))))
})

set_elementwise_derivative(prim_acosh, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx acosh(x) = 1 / sqrt(x^2 - 1)
  list(\(v) prim_div(v, prim_sqrt(prim_sub(prim_mul(x, x), ones_like(x)))))
})

set_elementwise_derivative(prim_atanh, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx atanh(x) = 1 / (1 - x^2)
  list(\(v) prim_div(v, prim_sub(ones_like(x), prim_mul(x, x))))
})

# special functions ------------------------------------------------------------

set_elementwise_derivative(prim_lgamma, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx lgamma(x) = digamma(x)
  list(\(v) prim_mul(v, prim_digamma(x)))
})

set_elementwise_derivative(prim_digamma, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx digamma(x) = trigamma(x) = polygamma(1, x)
  list(\(v) prim_mul(v, prim_psigamma(x, ones_like(x))))
})

set_elementwise_derivative(prim_psigamma, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  deriv <- inputs[[2L]]
  list(
    # d/dx psigamma(x, deriv) = psigamma(x, deriv + 1)
    \(v) prim_mul(v, prim_psigamma(x, prim_add(deriv, ones_like(deriv)))),
    # psigamma is typically called with an integer order, which is not
    # differentiable; follow torch and refuse.
    \(v) cli_abort("The derivative with respect to {.arg deriv} of {.fn prim_psigamma} is not implemented.")
  )
})

set_elementwise_derivative(prim_erf, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx erf(x) = (2 / sqrt(pi)) * exp(-x^2)
  list(\(v) {
    coef <- prim_fill(2 / sqrt(pi), dtype = dtype(x), shape = shape(x))
    prim_mul(v, prim_mul(coef, prim_exp(prim_negate(prim_mul(x, x)))))
  })
})

set_elementwise_derivative(prim_erfc, function(inputs, outputs, params) {
  x <- inputs[[1L]]
  # d/dx erfc(x) = -(2 / sqrt(pi)) * exp(-x^2)
  list(\(v) {
    coef <- prim_fill(-2 / sqrt(pi), dtype = dtype(x), shape = shape(x))
    prim_mul(v, prim_mul(coef, prim_exp(prim_negate(prim_mul(x, x)))))
  })
})

set_elementwise_derivative(prim_erf_inv, function(inputs, outputs, params) {
  y <- outputs[[1L]]
  # d/dx erf_inv(x) = sqrt(pi)/2 * exp(erf_inv(x)^2)
  list(\(v) {
    coef <- prim_fill(sqrt(pi) / 2, dtype = dtype(y), shape = shape(y))
    prim_mul(v, prim_mul(coef, prim_exp(prim_mul(y, y))))
  })
})

# step functions and predicates ------------------------------------------------
# Their derivative is zero wherever it exists, the non-differentiable points
# are ignored.

set_elementwise_derivative(prim_floor, no_derivative)
set_elementwise_derivative(prim_ceiling, no_derivative)
set_elementwise_derivative(prim_round, no_derivative)
set_elementwise_derivative(prim_sign, no_derivative)

set_elementwise_derivative(prim_eq, no_derivative)
set_elementwise_derivative(prim_ne, no_derivative)
set_elementwise_derivative(prim_gt, no_derivative)
set_elementwise_derivative(prim_ge, no_derivative)
set_elementwise_derivative(prim_lt, no_derivative)
set_elementwise_derivative(prim_le, no_derivative)
set_elementwise_derivative(prim_is_finite, no_derivative)
