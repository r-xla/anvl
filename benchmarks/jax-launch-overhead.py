"""Measure JAX's eager/jitted launch overhead on CPU, mirroring the anvl
benchmark (benchmarks/prim-overhead.R) so the numbers are comparable.

Async handling: JAX dispatch is async; `block_until_ready()` is the analogue of
anvl's `await()`. We block inputs before timing and block outputs before
stopping the clock. We warm up once so the executable cache is hot (we do NOT
measure tracing/compilation).

Reports, per input size and median over many iterations:
  launch_us    : jnp.add(x, y) returned but NOT blocked  (dispatch only)
  roundtrip_us : jnp.add(x, y) then block_until_ready     (full latency)
  await_us     : roundtrip - launch                       (device/sync wait)
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import time
import statistics
import jax
import jax.numpy as jnp

print("jax:", jax.__version__, "device:", jax.devices()[0].platform)


def bench(fn, n_iter, min_total=0.4):
    # time.perf_counter-based; run at least n_iter and at least min_total seconds.
    times = []
    start = time.perf_counter()
    i = 0
    while i < n_iter or (time.perf_counter() - start) < min_total:
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        i += 1
        if i > 500_000:
            break
    return statistics.median(times)


sizes = [1, 100, 10_000, 1_000_000]
N = 5000

print("\n==== jnp.add eager launch overhead vs compute (median per call) ====\n")
print(f"{'n':>9} {'launch_us':>10} {'await_us':>9} {'roundtrip_us':>13}")

rows = []
for n in sizes:
    x = jax.device_put(jnp.arange(n, dtype=jnp.float32))
    y = jax.device_put(jnp.arange(n, dtype=jnp.float32))
    x.block_until_ready()
    y.block_until_ready()

    # warm up the executable cache for this (shape, dtype)
    jnp.add(x, y).block_until_ready()

    launch = bench(lambda: jnp.add(x, y), N)
    roundtrip = bench(lambda: jnp.add(x, y).block_until_ready(), N)

    launch_us = launch * 1e6
    roundtrip_us = roundtrip * 1e6
    await_us = roundtrip_us - launch_us
    rows.append((n, launch_us, await_us, roundtrip_us))
    print(f"{n:>9} {launch_us:>10.2f} {await_us:>9.2f} {roundtrip_us:>13.2f}")

# Also time a pre-jitted binary fn (closest analogue to a cached anvl primitive).
print("\n==== jax.jit(add) (pre-compiled) ====\n")
add = jax.jit(lambda a, b: a + b)
print(f"{'n':>9} {'launch_us':>10} {'roundtrip_us':>13}")
for n in sizes:
    x = jax.device_put(jnp.arange(n, dtype=jnp.float32))
    y = jax.device_put(jnp.arange(n, dtype=jnp.float32))
    x.block_until_ready(); y.block_until_ready()
    add(x, y).block_until_ready()  # warm
    launch = bench(lambda: add(x, y), N)
    roundtrip = bench(lambda: add(x, y).block_until_ready(), N)
    print(f"{n:>9} {launch*1e6:>10.2f} {roundtrip*1e6:>13.2f}")
