# prim_custom_gradient / refuses a rule that leaves out a required gradient

    Code
      jit(gradient(f))(x, y)
    Condition
      Error in `prim_custom_gradient()`:
      ! `rule` must return the gradient of each entry of `args` that is required.
      x Got none for "b".
      i `rule` receives which entries are required as `required`.

# prim_custom_gradient / refuses a rule that closes over a traced value

    Code
      jit(gradient(f, wrt = "x"))(x, nv_scalar(2, dtype = "f64"))
    Condition
      Error in `prim_custom_gradient()`:
      ! `rule` must not close over a traced value.
      i Pass it to `rule` through `args` instead.

# prim_custom_gradient / refuses to differentiate a value fn closes over

    Code
      jit(gradient(f))(x)
    Condition
      Error:
      ! `gradient()` cannot differentiate with respect to a value that `fn` of `prim_custom_gradient()` closes over.
      i Pass the value to `fn` through `args` instead.

# prim_custom_gradient / refuses a rule that does not return a list of gradients by name

    Code
      differentiate(function(x, out, grad, required) grad)
    Condition
      Error in `prim_custom_gradient()`:
      ! `rule` must return a list of gradients, named by the entries of `args` they belong to.
      x Got f64[3].
      i `args` has names "x".

---

    Code
      differentiate(function(x, out, grad, required) list(x = grad, y = grad))
    Condition
      Error in `prim_custom_gradient()`:
      ! `rule` must return a list of gradients, named by the entries of `args` they belong to.
      x Got a list with names c("x", "y").
      i `args` has names "x".

# prim_custom_gradient / refuses a gradient of another structure or type than its argument

    Code
      differentiate(list(x = s), function(x) x, function(x, out, grad, required) list(
        x = list(grad, grad)))
    Condition
      Error in `prim_custom_gradient()`:
      ! `rule` must return each gradient with the structure of the entry of `args` it belongs to.
      x The gradient of "x" has another structure.

---

    Code
      differentiate(list(p = list(w = s, b = s)), function(p) p$w, function(p, out,
        grad, required) list(p = list(w = grad, b = nv_convert(grad, "f32"))))
    Condition
      Error in `prim_custom_gradient()`:
      ! `rule` must return a gradient of the same type as the argument it belongs to.
      x `p$b` is f64[], its gradient f32[].

---

    Code
      differentiate(list(x = x), function(x) nv_sum(x), function(x, out, grad,
        required) list(x = grad))
    Condition
      Error in `prim_custom_gradient()`:
      ! `rule` must return a gradient of the same type as the argument it belongs to.
      x `x` is f64[3], its gradient f64[].

# prim_custom_gradient / with a forward pass / refuses a forward pass that closes over a traced value

    Code
      differentiate(function(s) function(x, required) list(out = x * s, residuals = list()))
    Condition
      Error in `prim_custom_gradient()`:
      ! `forward` must not close over a traced value.
      i Pass it through `args` instead.

---

    Code
      differentiate(function(s) function(x, required) list(out = x * 4, residuals = list(
        s = s)))
    Condition
      Error in `prim_custom_gradient()`:
      ! `forward` must not close over a traced value.
      i Pass it through `args` instead.

# prim_custom_gradient / with a forward pass / refuses a forward pass that does not return the result and the residuals

    Code
      differentiate(function(x, required) x)
    Condition
      Error in `prim_custom_gradient()`:
      ! `forward` must return a list with the result, out, and the residuals.
      x Got f64[3].

---

    Code
      differentiate(function(x, required) list(out = x))
    Condition
      Error in `prim_custom_gradient()`:
      ! `forward` must return a list with the result, out, and the residuals.
      x Got a list with names "out".

---

    Code
      differentiate(function(x, required) list(out = x, residuals = "r"))
    Condition
      Error in `prim_custom_gradient()`:
      ! `forward` must return residuals that are a list of arrays, or trees of them.
      x Got "r".

---

    Code
      differentiate(function(x, required) list(out = x, residuals = list(a = "r")))
    Condition
      Error in `prim_custom_gradient()`:
      ! `forward` must return residuals that are a list of arrays, or trees of them.
      x Got "r" at `a`.

# prim_custom_gradient / with a forward pass / refuses a result of another structure or type than the result of fn

    Code
      differentiate(function(x) list(x, x))
    Condition
      Error in `prim_custom_gradient()`:
      ! `forward` must return out with the structure of the result of `fn`.
      x Got <list> of length 2.

---

    Code
      differentiate(function(x) nv_convert(x, "f32"))
    Condition
      Error in `prim_custom_gradient()`:
      ! `forward` must return out of the same types as the result of `fn`.
      x `fn` returns f64[3], `forward` f32[3].

# prim_custom_gradient / with a forward pass / refuses a rule that leaves out a required gradient

    Code
      jit(gradient(f))(x)
    Condition
      Error in `prim_custom_gradient()`:
      ! `rule` must return the gradient of each entry of `args` that is required.
      x Got none for "x".
      i `rule` receives which entries are required as `required`.

# prim_eq, prim_ne, prim_gt, prim_ge, prim_lt, prim_le

    x gradient can only be computed for functions that return float scalar
    i Got dtype=bool

---

    x gradient can only be computed for functions that return float scalar
    i Got dtype=bool

---

    x gradient can only be computed for functions that return float scalar
    i Got dtype=bool

---

    x gradient can only be computed for functions that return float scalar
    i Got dtype=bool

---

    x gradient can only be computed for functions that return float scalar
    i Got dtype=bool

---

    x gradient can only be computed for functions that return float scalar
    i Got dtype=bool

