# Reducing eager-dispatch (launch) overhead — analysis & roadmap

Goal: cut the per-call overhead of launching an anvl primitive (e.g. `prim_add`)
so it approaches JAX's launch overhead.

## Where we are

Measured on CPU, cache-hit (compiled executable already cached), inputs and
outputs `await()`ed so the async launch is measured honestly:

| framework | per-launch overhead (n=1) |
|---|---|
| JAX `jnp.add` (eager) | **~5.8 µs** |
| JAX `jax.jit(add)` (precompiled) | ~4 µs |
| anvl `prim_add` (before this work) | ~620 µs roundtrip / ~557 µs dispatch |
| anvl `prim_add` (after pjrt device-caching) | ~480 µs roundtrip / ~378 µs dispatch |

So anvl is ~70× JAX. The pjrt device-resolution caching already landed (branch
`perf/device-resolution-caching` in the pjrt repo: memoized platform parsing,
O(1) `cached_device`, identity `==.PJRTDevice`) plus an anvl change to skip the
no-op `copy_buffer` on the eager path. That took dispatch ~557 → ~378 µs. This
doc is the plan for the remaining ~373 µs.

## Benchmark / profiling harness (in this directory)

- `prim-overhead.R` — top-level: launch vs await vs roundtrip across input sizes,
  with an A/B switch for an optimized pjrt (see header).
- `prim-dispatch-profile.R` — `Rprof` line-level profile of the dispatch path.
- `prim-dispatch-segments.R` — `bench::mark` of each dispatch phase in isolation
  (the most useful for attributing gains).
- `pjrt-execute-breakdown.R` — splits `pjrt_execute()` into the real C++ launch
  vs R-side wrapper overhead, plus the baseline R→C++ call cost.
- `jax-launch-overhead.py` — the JAX reference numbers.

**Always re-run `prim-overhead.R` + `prim-dispatch-segments.R` after each change**
to attribute the gain, and keep the anvl + pjrt test suites green.

## Where the remaining ~380 µs goes (per-phase, n=1)

From `prim-dispatch-segments.R`:

| phase | µs | note |
|---|---|---|
| `jit_auto` wrapper layer | ~35 | redundant 2nd dispatch (capture/eval/flatten/backend-detect) |
| `jit_prepare_call` | ~120 | `build_tree`+`mark_some` 46, `check_jit_input` 40, `flatten` 8, rest 26 |
| `to_avals` | ~68 | `nv_aval(dtype(x), shape(x), ambiguous(x))` per input |
| cache key `list()` + `cache$get` | ~30 | hash of nested `list(in_tree, avals, device)` |
| `jit_call_xla` (R parts) | ~35 | phantom `pjrt_empty` 15, `jit_wrap_outputs` 17, unwrap 3 |
| `pjrt_execute` internals | ~49 | real launch ~21, `pjrt_execution_options()` ~8, R checks ~12 |

Key facts established by measurement:
- **The R→C++ boundary is ~0.4 µs (free).** Moving work to C++ helps by deleting
  R *interpreter* work, not by being a faster callee.
- **The real PJRT device launch is only ~21 µs.** The rest of `pjrt_execute` is
  R-side (options rebuilt per call, redundant validation, list repacking).
- **~114 µs of `build_tree` + `to_avals` is built only to be hash material for
  the cache key, then discarded — and it runs BEFORE the cache lookup.**

## How JAX does it (reference)

- Two cache layers: a static per-`jit` key built once, and a per-call key in C++.
- Per-call key = input **PyTreeDef** + one compact **`(dtype enum, shape ints,
  weak_type)`** tuple per leaf, hashed directly with absl. **No digest string** —
  shapes hashed as raw ints.
- The **aval (dtype/shape) is cached on the array object** (`self.aval`, set once
  at construction in C++); dtype/shape are never recomputed from the buffer.
- The whole hot path is in C++; on a cache hit almost nothing happens in Python.
- Eager primitives (`jnp.add`) funnel through the same jit hot path, cached on
  `(prim, params)`.

The lesson: a compact fingerprint key read from metadata **cached on the array**,
computed before any heavy object construction — not a string digest.

## Roadmap (prioritized)

### Tier 1 — pure-R restructuring (target ~380 → ~150 µs)

1. **Cache metadata on the `AnvlArray` at construction** — store
   `(dtype, shape, device, ambiguous)` once so `dtype()/shape()/device()` are
   field reads, not S3-dispatch → C++ calls. (JAX's "aval on the array".)
   Headroom: most of `to_avals` (68) + part of `check_jit_input` (40).
   Risk: medium — touches all array constructors (`new_data`, `new_empty`,
   `jit_wrap_outputs`/`nv_array`).

2. **Cheap-fingerprint cache key, computed before the heavy construction**
   — replace `list(in_tree, avals, device)` with a compact hashable key built
   directly from the cached metadata (per-leaf dtype+shape ints + device id +
   a small structural tag). **Defer `build_tree` + full `to_avals` (the `Node`
   and `nv_aval` objects) to the cache-MISS path only.** Today ~114 µs of that
   runs on every hit purely as hash material, before the cache is even probed.
   Headroom: ~110–140 µs. **Largest single lever.** Couples with #1 (the key
   reads the cached metadata).

3. **Flat-args fast path** — primitives are always a flat list of arrays/scalars;
   skip the general tree-walking `build_tree`/`flatten` for that case.
   Headroom: ~46 µs; feeds #2.

4. **Collapse the double dispatch** — `prim_add()` goes through the `jit_auto`
   wrapper (its own `match.call`+`eval`+`flatten`+backend-detect) then the inner
   closure which redoes `match.call`+`eval`+`flatten` inside `jit_prepare_call`.
   Memoize the resolved backend for fixed-signature primitives and skip the
   wrapper. Headroom: ~35 µs.

5. **Skip redundant input validation on the trusted hot path** — fold
   `check_jit_input` into the single metadata read instead of re-proving
   known-good inputs each call. Headroom: part of the 40 µs.

### Tier 2 — pjrt execute-path cleanups (independent, ~28 µs, low risk)

6. **Cache the default `pjrt_execution_options()`** instead of rebuilding it per
   call. Headroom: ~8 µs. (pjrt-side.)

7. **A "trusted" internal execute** that skips `check_buffer` /
   `check_loaded_executable` / `...names()` / variadic `list(...)` repacking on
   the hot path. Headroom: ~12 µs. (pjrt-side.)

8. **Pool/reuse the donation phantom buffers** instead of `pjrt_empty`-allocating
   one per output per call. Headroom: ~15 µs.

### Tier 3 — output wrapping

9. **Cache the output wrapping structure** — `jit_wrap_outputs` rebuilds
   `AnvlArray`s via `nv_array`/S3 each call; the out tree/dtype/shape is fixed
   per signature. Headroom: ~17 µs.

### Tier 4 — C++ hot path (the only path to JAX's ~5 µs; ~150 → ~10–20 µs)

10. **Move the eager-dispatch fast path into C++** — given a cached executable +
    fastpath data: fingerprint inputs → LRU lookup → splice buffer xptrs into the
    execute call → wrap outputs, natively. This is JAX's `CallSignature`/
    `_cpp_pjit`. The boundary is free, so the win is deleting `match.call`,
    `lapply(eval)`, `.mapply`, hashtab, and S3 dispatch overhead.

11. **Cache the fingerprint/aval on the buffer in C++** (like JAX's
    `PyArray::aval`) so even metadata reads stay native.

## Realistic targets

- **Tiers 1–2 (pure R):** ~380 → **~120–150 µs** (~2.5–3×). The R floor sits here
  because `match.call` + `lapply(eval)` + one hashtab probe + S3 dispatch each
  cost microseconds and recur per call.
- **Tier 4 (C++ hot path):** ~150 → **~10–20 µs**, approaching JAX's ~5 µs.

## Recommended order

Start with **#1 + #2 together** (coupled), measure, then the quick independent
pjrt wins **#6–#8**, then reassess whether the Tier-4 C++ path is worth it based
on how close Tier 1–2 lands.
