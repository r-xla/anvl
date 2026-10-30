"""The Binomial functions that `jax.scipy.stats.binom` lacks: `cdf` and `ppf`.

Both are written for 0 <= p <= 1 and whole `n`, without the full parameter
validation of scipy.

- `cdf` uses P(X <= k) = I_{1-p}(n - k, k + 1) through JAX's `betainc`, with the
  ends of the support resolved directly.
- `ppf` mirrors `nv_qbinom()`'s search, so that the comparison measures the CDF
  and the XLA build rather than two search strategies: base R's fuzz and
  Cornish-Fisher starting point, then galloping from the start until the
  quantile is bracketed in (lo, hi], then bisection. Each step evaluates the CDF
  over all elements, and the loop runs while any element is still searching.
"""

import jax.numpy as jnp
from jax import lax
from jax.scipy.special import betainc, ndtri


def cdf(k, n, p):
    k = jnp.floor(k)
    inside = (k >= 0) & (k < n)
    k_safe = jnp.where(inside, k, 0)
    inner = betainc(n - k_safe, k_safe + 1, 1 - p)
    return jnp.where(k < 0, 0, jnp.where(k >= n, 1, inner))


def ppf(q, n, p):
    eps = jnp.finfo(q.dtype).eps
    fuzzed = q * (1 - 8 * eps)
    target = jnp.where(fuzzed > 0, fuzzed, q)
    q_safe = jnp.where((q > 0) & (q < 1), q, 0.5)
    sigma = jnp.sqrt(n * p * (1 - p))
    gamma = ((1 - p) - p) / sigma
    z = jnp.clip(ndtri(q_safe), -40, 40)
    start = jnp.round(n * p + sigma * (z + gamma * (z * z - 1) / 6))

    def probe_at(lo, hi, step, started, galloping, down):
        probe = jnp.where(
            ~started,
            start,
            jnp.where(galloping, jnp.where(down, hi - step, lo + step), jnp.floor((lo + hi) / 2)),
        )
        return jnp.minimum(jnp.maximum(probe, lo + 1), hi - 1)

    def is_active(lo, hi, probe):
        return (hi - lo > 1) & (probe > lo) & (probe < hi)

    def cond(s):
        i, lo, hi, step, started, galloping, down = s
        probe = probe_at(lo, hi, step, started, galloping, down)
        return jnp.any(is_active(lo, hi, probe)) & (i < 2200)

    def body(s):
        i, lo, hi, step, started, galloping, down = s
        probe = probe_at(lo, hi, step, started, galloping, down)
        active = is_active(lo, hi, probe)
        above = cdf(jnp.where(active, probe, 0), n, p) >= target
        # Galloping goes on while each probe lands on the same side as the last
        onward = jnp.where(down, above, ~above)
        return (
            i + 1,
            jnp.where(active & ~above, probe, lo),
            jnp.where(active & above, probe, hi),
            jnp.where(started & galloping & onward, 2 * step, step),
            started | active,
            galloping & (~started | onward),
            jnp.where(started, down, above),
        )

    # Elements resolved without a search start with an empty bracket
    no = jnp.zeros_like(q, dtype=bool)
    resolved = ~((q > 0) & (q < 1))
    hi = jnp.full_like(q, n)
    init = (0, jnp.where(resolved, hi - 1, -1), hi, jnp.ones_like(q), no, ~no, no)
    hi = lax.while_loop(cond, body, init)[2]
    out = jnp.where(q <= 0, 0, jnp.where(q >= 1, n, hi))
    return jnp.where(jnp.isnan(q) | (q < 0) | (q > 1), jnp.nan, out)

