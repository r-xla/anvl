# Eager-dispatch optimization — remaining work & insights

Living TODO for closing the gap to JAX's ~5–6 µs eager-launch overhead. Companion
to `dispatch-overhead-roadmap.md` (strategy) and `cpp-hot-path-design.md` (the
native dispatcher design). Keep the measured numbers here so we don't re-derive
them.

## Where we are (measured, CPU, n=1, `prim_add`)

Journey so far:

| stage | launch µs |
|---|---|
| original (start of all this work) | ~557 |
| + pjrt fixes (device cache, options, num_outputs, check flag) | ~480 |
| + anvl Tier-1 (metadata-on-array, cheap key, flat-args fast path) | ~250 |
| + native C++ dispatcher wired in (slices 1–5) | ~98 |
| + cheap backend detection (DONE, **uncommitted**) | **~69** |
| JAX `jnp.add` reference | ~6 |

Branches: pjrt `feat/native-dispatch`, anvl `perf/dispatch-overhead`. Nothing
pushed. All pjrt (1362) + anvl (1935) tests pass as of the static-dispatch +
pytree-move landing (2026-07-03); the backend-detection change is committed.

**Update 2026-07-03 (Parts A+B of the static-dispatch spec landed):** static-arg
jits now dispatch natively (keyed by value via identical(), excluded from
execute) instead of falling back to the R cache, and the pytree module moved to
pjrt (opaque PJRTNode; anvl's R/flatten.R deleted). Measured on the linux/ARM
dev sandbox (absolute numbers NOT comparable to the Mac numbers above):
prim_add launch n=1 ~81 us of which raw pjrt_execute enqueue ~48 us (anvl-side
~32 us); a static jit call ~106 us natively vs ~203 us on the forced R path,
i.e. static calls now cost about the same as dynamic native calls (~80 us)
instead of ~2x. TODO 0 below is DONE (committed as anvl e6cf475f).

## Profiler breakdown of the ~98 µs path (Rprof, real `prim_add` loop)

This is the map that tells us what's left. `jit_auto_detect_backend` was the
single biggest item (~34%) — now fixed but not committed.

- `jit_auto_detect_backend` ~34% (~46 µs) — **FIXED (uncommitted)**: was
  `flatten()` + `vapply(backend())` + `unique` + `%in%`; rewritten as a direct
  short-circuiting scan reading `$backend` as a field. Brought launch ~98→~69 µs.
- `jit_wrap_outputs` / `structure` ~17% (~37 µs) — output wrapping, still in R.
- `impl_dispatch_run` ~10% — the native dispatch itself (this is the part that's
  already fast; little left here).
- closure machinery: `match.call`, `as.list`, `assert_flag`, `tryCatch`, the
  double arg-eval (wrapper + inner) — the remaining ~20–30 µs.

---

## TODO 0 — commit the backend-detection fix (immediate)

`R/jit.R`: `jit_auto_detect_backend(args, static)` is rewritten (direct scan) and
the call site updated. **Still need to**: run full anvl suite (esp.
`test-jit.R` backend/device handling, and the multi-backend error tests),
`air format` + `jarl check`, then commit. Verify the multi-backend conflict
error still fires (mixed xla/quickr inputs) and that nested-list args still
detect correctly.

## TODO 1 — output wrapping `jit_wrap_outputs` (~37 µs, biggest remaining R cost)

`jit_wrap_outputs(out_flat, out_tree, ambiguous_out, "xla")` calls
`nv_array(buf, backend="xla")` per output. Measured `nv_array(buf)` ≈ 27 µs, of
which ~15 µs is **re-reading dtype/shape/device off the output buffer via S3 →
C++** (`tengen::dtype` ~7, `device` ~5, `shape` ~3) — data the dispatcher
already has natively. The rest (~12 µs) is `nv_array` dispatch + `structure()`.

Approach (respecting "don't rebuild AnvlArray inside pjrt"): have the native
dispatcher **return per-output metadata it reads natively** — dtype (enum or
string) + shape — plus the shared device R object (it already holds
`entry->device_xptr`, and all outputs of one executable share a device). anvl
then builds each output `AnvlArray` directly from that metadata, skipping the
per-output S3 dtype/shape/device reads:
`structure(list(data=buf, dtype=<from meta>, shape=<from meta>, device=<entry device>, ambiguous=amb, backend="xla"), class="AnvlArray")`.

Open sub-questions:
- dtype: return the pjrt enum int and have anvl map enum→tengen dtype (needs a
  fast lookup table), OR return the dtype string and use `as_dtype()` (measure
  its cost). The AnvlArray `$dtype` field must end up a tengen dtype object.
- Confirm how `nv_array()` currently turns a **raw `PJRTBuffer`** into an
  AnvlArray without copying (trace the buffer path through `new_data`) — the new
  fast path must preserve that (wrap, never copy).
- Single-output is the common case; keep the general multi-output path correct.

Expected: ~37 → ~10 µs.

## TODO 2 — collapse the wrapper/inner double arg-eval (~8–16 µs)

`prim_add` = `jit_auto` wrapper → `do.call(jit_fns[["xla"]], args)` → inner
`jit_xla_impl` closure. BOTH do `match.call()` + `lapply(eval)`:
- wrapper: `args <- lapply(as.list(match.call())[-1L], eval, parent.frame())`
- inner: same again at the top of its closure.

The wrapper has already evaluated the args; the inner re-parses them. Make the
inner reuse the wrapper's evaluated args instead of re-`match.call`-ing.
Complication: the inner is ALSO user-callable directly (`f <- jit(g); f(a,b)`),
so it can't unconditionally assume pre-evaluated args. Options:
- give the inner a fast entry that takes an already-evaluated args list, and
  have the wrapper call that entry (keep the match.call path for direct calls);
- or lift the dispatcher call into the wrapper for the xla case (couples
  `jit_auto` to xla internals — weigh against keeping backends generic).

Also: `list(...)` capture (~1.5 µs) is ~3.5 µs cheaper than
`match.call()`+`as.list`+`lapply(eval)` (~5 µs) but loses named-arg matching /
the by-name static handling — probably not worth the API change on its own;
folds into the above restructure if done.

## TODO 3 — reduce R↔C++ boundary crossings (~small)

Currently: closure → `pjrt::pjrt_dispatch()` (thin R wrapper over `.Call`) →
then `jit_wrap_outputs()` (separate R work). Minor wins:
- call `impl_dispatch_run` (the `.Call`) directly instead of through the
  `pjrt_dispatch()` R wrapper (saves one R function call);
- once TODO 1 lands, the wrap metadata comes back in the same dispatch result,
  so there's a single boundary crossing for execute + wrap material.

## TODO 4 — pjrt test cleanup: use real anvl arrays (user request)

The pjrt dispatch test currently **hand-fakes** the xla AnvlArray leaf shape:
`structure(list(data=buf, ambiguous=FALSE, backend="xla"), class="AnvlArray")`.
This can silently drift from anvl's real `AnvlArray` structure (the contract
`extract_leaf` depends on). Instead:
- add `anvl` to `Suggests` in pjrt `DESCRIPTION` (a `Suggests` cycle is fine —
  anvl `Imports` pjrt; `stablehlo` is already a precedent),
- in `tests/testthat/test-dispatch.R`, build dispatchable leaves with real
  `anvl::nv_array()` guarded by `skip_if_not_installed("anvl")`, so the test
  exercises the actual AnvlArray contract and skips when anvl isn't installed.

(NOTE: do NOT rebuild the AnvlArray structure inside pjrt C++ — the dispatcher
returns raw buffers + wrap material; anvl owns the AnvlArray layout.)

## TODO 5 — docs / housekeeping

- pjrt's mission expanded to "runtime + jit dispatch" (jaxlib-like). Update
  pjrt's `AGENTS.md`/CLAUDE.md design section and the `pjrt_dispatch` rd so this
  is documented, not implicit.
- `man/` for the new `pjrt_dispatch*` exports has a roxygen cross-ref warning
  about `pjrt_buffer` linking — tidy the `@return`/`\link` so `devtools::check()`
  is clean.
- Update `dispatch-overhead-roadmap.md` "where we are" numbers once the above
  land.

---

## TODO 6 — Phase 2: move arg capture + output construction into C++ (reach ~5–6 µs)

The R floor after TODO 0–3 is likely ~30–50 µs, because `match.call`/arg-eval,
the closure frame, `structure()`, and `unflatten` all remain interpreted R.
JAX's ~6 µs needs the *whole* call shell native:
- make `prim_*` a thin C++ entry (capture args natively, no R `match.call`),
- build output `AnvlArray`s natively (or a leaner native representation),
- cache the aval **on the buffer** in C++ (design doc §4.2) so even input
  metadata never touches R.

This is the largest, most invasive step (the design doc's "v2 / full pjit-style
port"); only pursue once TODO 0–3 are exhausted and we know the R floor.

## Verified-correct invariants to preserve (don't regress these)

- SENTINEL fallback: any call the native path can't handle returns the sentinel
  and runs the unchanged R path — correctness is never worse than R-only.
- Cache key distinguishes dtype/shape/ambiguity/arity/device/tree structure
  (incl. NULL = NullNode); equal signatures from distinct buffers hit.
- GC: phantom donation buffers allocated straight into the rooted `inputs` list;
  cache-entry SEXPs preserved-then-released exactly once. (Stress-tested 5000×
  with `gc()`.)
- `cache_size()` sums the native dispatcher + the R-side cache (a signature lands
  in exactly one).
- Args evaluated once on the fallback (no double side effects).
