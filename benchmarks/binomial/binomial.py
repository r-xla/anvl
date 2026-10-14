"""JAX and SciPy twin of binomial.R.

Times the Binomial pmf, cdf and ppf at f32 and f64 in JAX next to SciPy, with
`-x` as a baseline, on the inputs that binomial.R writes to `inputs/`. Run
binomial.R first, then this script, and never both at once: both XLA runtimes
use every core.

    python benchmarks/binomial/binomial.py

JAX lacks the Binomial cdf and ppf, so `jbinom` below takes them from
jax_binom.py (see there) and everything else from jax.scipy.stats.binom.
"""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)  # must precede any array creation
import jax.numpy as jnp
import jax.scipy.stats as jstats
import scipy.stats as sstats

import jax_binom

jbinom = SimpleNamespace(pmf=jstats.binom.pmf, cdf=jax_binom.cdf, ppf=jax_binom.ppf)

inputs = Path(__file__).resolve().parent / "inputs"
size = 100
prob = 0.3

# R writes doubles; x_d and x_p are R integers there, which base R's functions
# receive, so scipy gets them as int32 too
x_d = np.fromfile(inputs / "x_d.bin", dtype="<f8").astype(np.int32)
x_p = np.fromfile(inputs / "x_p.bin", dtype="<f8").astype(np.int32)
p_q = np.fromfile(inputs / "p_q.bin", dtype="<f8")

base = jax.jit(lambda x: -x)
baseS = lambda x: -x

# as.vector() always hands back a fresh R double vector (a copy, plus an f32 -> f64
# widening for f32 arrays). np.asarray() on a CPU jax.Array can be a zero-copy view,
# so copy and widen explicitly to do the same work.
to_host = lambda a: np.array(a, dtype=np.float64)


def mark(fn, min_iterations=50, min_time=0.5):
    ts = []
    t_start = time.perf_counter()
    while len(ts) < min_iterations or time.perf_counter() - t_start < min_time:
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return dict(min=ts.min(), median=np.median(ts), n_itr=len(ts))


# f is the JAX function, fS its scipy counterpart
def bench_one(f, fS, x):
    x_f32 = jax.device_put(jnp.asarray(x, dtype=jnp.float32))
    x_f64 = jax.device_put(jnp.asarray(x, dtype=jnp.float64))
    for xx in (x_f32, x_f64):  # compile + warm
        to_host(f(xx))
        to_host(base(xx))
    assert f(x_f32).dtype == jnp.float32 and f(x_f64).dtype == jnp.float64

    cases = {
        "f_32": lambda: to_host(f(x_f32)),
        "base_32": lambda: to_host(base(x_f32)),
        "f_64": lambda: to_host(f(x_f64)),
        "base_64": lambda: to_host(base(x_f64)),
        "fS": lambda: fS(x),  # already a host array; as.vector() on an R vector
        "baseS": lambda: baseS(x),  # is a no-op, so no to_host() here either
    }
    return {name: mark(fn) for name, fn in cases.items()}


parts = {
    "pmf": (jax.jit(lambda x: jbinom.pmf(x, size, prob)), lambda x: sstats.binom.pmf(x, size, prob), x_d),
    "cdf": (jax.jit(lambda q: jbinom.cdf(q, size, prob)), lambda q: sstats.binom.cdf(q, size, prob), x_p),
    "ppf": (jax.jit(lambda p: jbinom.ppf(p, size, prob)), lambda p: sstats.binom.ppf(p, size, prob), p_q),
}

for name, (f, fS, x) in parts.items():
    start = time.perf_counter()
    res = bench_one(f, fS, x)
    elapsed = time.perf_counter() - start
    print(f"\n== {name}: n = {len(x):g}, size = {size:g}, prob = {prob:g} ({elapsed:.1f} s)")
    print(f"{'expression':<10}{'min':>12}{'median':>12}{'n_itr':>8}")
    for case, r in res.items():
        print(f"{case:<10}{r['min'] * 1e3:>10.2f}ms{r['median'] * 1e3:>10.2f}ms{r['n_itr']:>8}")
