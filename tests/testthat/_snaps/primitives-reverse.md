# prim_custom_gradient / refuses to differentiate a value fn or rule close over

    Code
      jit(gradient(f))(x)
    Condition
      Error:
      ! `gradient()` cannot differentiate with respect to a value that `fn` or `rule` of `prim_custom_gradient()` close over.
      i Pass the value to `fn` through `args` instead.

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

