# C++ eager-dispatch hot path — design

Target: bring `prim_*` eager launch from ~250 µs (current, after the Tier-1 R
optimizations) toward JAX's ~5 µs by moving the **cache-hit dispatch** into C++,
keeping tracing/compilation in R. Full coverage (nested trees, static args,
device moves, native avals) is the end goal; we build it in verifiable phases.

This mirrors JAX's structure (a native dispatch fast path + an R `cache_miss`
callback that compiles) but uses **anvl's own vocabulary** throughout — the
`Node` tree from `R/flatten.R`, `avals`, `cache_key`, etc. — not JAX's PyTree
class names.

---

## 1. Decisions (locked)

- **Location: pjrt.** The dispatch C++ lives in pjrt (which already has the
  toolchain, the buffer/executable classes, and the execute logic). `pjrt ≈
  jaxlib`. **anvl stays pure R** — no C++ build chain, no second `.so`, no
  cross-package linking. (pjrt's mission expands to "runtime + jit dispatch";
  its docs should say so.)
- **Tracing/compilation stays in R**, in anvl. Only dispatch goes native.
- Per-leaf anvl signature material is just a **`bool ambiguous`** for now (not a
  general opaque field — add more only if ever needed).

---

## 2. Ownership split

- **pjrt owns (C++):** the `Node` tree (flatten/build_tree/unflatten), the
  `cache_key`, the executable cache (an `LRUCache`), and the fast-path
  orchestration (build key → probe cache → marshal buffers → execute → wrap).
- **anvl owns (R):** the `nv_*` API, tracing, **stablehlo lowering /
  compilation = the `cache_miss` callback** handed to pjrt, the quickr/plain
  backends, and the generic `AnvlArray` facade.

The miss path calls back into R (anvl compiles), exactly like jaxlib → Python.

---

## 3. The one coupling contract

pjrt's dispatch must read each xla leaf and emit output leaves. Split the
signature material by who owns it:

- **dtype / shape / device** are genuine `PJRT_Buffer` properties → pjrt reads
  them natively off the buffer (fast in C++; they were only slow in R because of
  S3 dispatch + marshaling). Nothing cached for these on the native path.
- **`ambiguous`** is an anvl type-system concept, *not* a buffer property →
  anvl supplies it per leaf. pjrt folds it into the key but never interprets it.

So the entire interface pjrt depends on is:

> An xla leaf yields a `PJRT_Buffer*` and a `bool ambiguous`.

pjrt learns nothing about anvl's `AnvlArray` layout or type system beyond that.

---

## 4. Native data structures (pjrt C++) — anvl names

### 4.1 `Node` — the input/output tree (port of `R/flatten.R`)

```cpp
struct Node {
  enum Kind { LeafNode, ListNode, NullNode } kind;
  int          i;       // LeafNode: index into the flat leaf list
  std::vector<Node>        nodes;  // ListNode: children (R field name: `nodes`)
  std::vector<std::string> names;  // ListNode: child names (R field name: `names`)
  // hash + operator== are native; a flat ListNode of LeafNodes is the cheap case.
};
```

`flatten(args) -> (leaves, in_tree)` and `unflatten(in_tree, leaves)` match the R
semantics exactly:

- a bare list recurses (`ListNode`),
- a classed object / atomic is a leaf (`LeafNode`),
- **`NULL` is a `NullNode`**: it carries no leaf and consumes no flat index, but
  it **is** part of `in_tree`, so it is in the `cache_key`. Thus `f(x, NULL)`
  and `f(x, y)` compile separately (the `is.null()` branch differs). Verified
  against the R path: `list(x, NULL, list(2, NULL, 3))` → 3 leaves, round-trips
  identically.

The native **flat fast path** mirrors the R one in `jit_prepare_call()`: if no
arg is a `NULL` or a bare list, build `in_tree = ListNode(LeafNodes, names)`
directly; otherwise fall back to the general `build_tree` walk.

### 4.2 `aval` — per-leaf abstract value (== `nv_aval`)

```cpp
struct aval {            // mirrors nv_aval(dtype, shape, ambiguous)
  int dtype;             // PJRT_Buffer_Type    (read from the buffer)
  std::vector<int64_t> shape;  //                (read from the buffer)
  bool ambiguous;        // anvl-supplied
};
```

Note `device` is **not** per-aval (matching anvl): it is a single per-call value
in the `cache_key`.

### 4.3 `cache_key` (== anvl's `list(in_tree, key_leaves, device)`)

```cpp
struct cache_key {
  Node                 in_tree;        // structure (incl. NullNodes + static positions)
  std::vector<KeyLeaf> key_leaves;     // per leaf: an `aval` (dynamic) or a static value
  const void*          device;         // resolved/target device
  // AbslHashValue + operator==; static values compared via R identical().
};
```

`KeyLeaf` is a dynamic `aval` or a static-arg `SEXP` — exactly anvl's
`jit_key_leaves()`, where static positions keep the value and dynamic positions
hold `(dtype, shape, ambiguous)`. Static values are hashed cheaply + compared
with R `identical()` (only touched when a primitive actually has static args;
`prim_add`-style ops don't).

### 4.4 `CacheEntry` (== anvl's stored cache value)

Mirrors `list(exec, out_tree, const_arrays, ambiguous_out, device, phantom_specs)`:

```cpp
struct CacheEntry {
  SEXP                      exec;           // PJRTLoadedExecutable xptr
  Node                      out_tree;
  std::vector<SEXP>         const_arrays;
  std::vector<PhantomSpec>  phantom_specs;  // donation outputs
  std::vector<char>         ambiguous_out;  // per output, or empty
  const void*               device;
};

// per jitted function; see §4.5 for the concrete structure.
```

### 4.5 The cache structure

pjrt ships only the PJRT **C API** headers — no `xla::LRUCache`, no abseil, no
XLA C++ utilities, and no cache of its own. So the cache is built on the C++
**standard library**, mirroring the data structure `xlamisc::LRUCache` already
uses (a hashmap + a doubly-linked MRU↔LRU list):

```cpp
struct Entry { cache_key key; CacheEntry value; };
std::list<Entry> order;                         // front = MRU, back = LRU
std::unordered_map<cache_key, std::list<Entry>::iterator,
                   KeyHash, KeyEq> index;
size_t capacity;                                // = the jit cache_size (default 100)
```

`get` = `index.find` → splice the node to the front; `set` = `push_front` +
insert into `index`, evicting `order.back()` when over `capacity`. Keeping the
same hashmap-plus-list semantics matters because the R fallback shares this cache
(§6), so recency/eviction must agree.

Two things make this more than a plain map:

- **Custom `KeyHash` / `KeyEq` over `cache_key`.** `in_tree` (recursive `Node`),
  `avals` (`dtype` / `shape` / `ambiguous`), and `device` hash and compare
  natively. **Static-arg `SEXP`s** in `key_leaves` cannot be hashed natively, so
  we hash a cheap discriminator (type + length) and fall back to R `identical()`
  for equality — only exercised when a primitive actually has static args (the
  hot `prim_add` path has none).
- **GC for cached `SEXP`s.** `exec`, `const_arrays`, and static-arg keys are R
  objects; if the cache is their only reference, R collects them. So
  `R_PreserveObject` on insert and `R_ReleaseObject` on eviction (and on handle
  finalization). This reuses pjrt's existing lifetime patterns (the
  `impl_loaded_executable_execute` keepalives + deferred-release queue) rather
  than inventing a new scheme.

One cache **per jitted function**, owned by the dispatch `handle` (external
pointer) — matching anvl's "one `LRUCache` per `jit()`" today.

---

## 5. Dispatch flow

The R side of a primitive becomes a thin shim that calls into pjrt with its
arguments and a `cache_miss` callback:

```
pjrt_dispatch(handle, args):
  1. tracing? (anvl-set flag)        -> SENTINEL (R passes through to f)
  2. flatten args -> (leaves, in_tree)         [flat fast path or build_tree]
  3. classify leaves: xla AnvlArray | R literal | R array | static
        non-xla (quickr) or device move?  -> SENTINEL (R slow path)
  4. build cache_key: in_tree + per-leaf aval/static + device
        (dtype/shape/device read from the buffer; ambiguous from the leaf)
  5. entry = cache.get(cache_key)
  6. miss: entry = R cache_miss(args)  -> {exec, out_tree, consts,
                                           phantom_specs, ambiguous_out, device}
           cache.set(cache_key, entry)
  7. marshal: input buffer ptrs + freshly-allocated phantom buffers
  8. out_bufs = execute(entry.exec, inputs, opts)     [reuse existing C++]
  9. wrap out_bufs as xla AnvlArray leaves + unflatten(entry.out_tree)
```

Steps 2–9 are native. The only R calls are the rare `cache_miss` and the
`SENTINEL` fallback. `handle` is an external pointer to a pjrt-side object
holding the per-function `cache`, static argnames, donate set, device policy,
and the R `cache_miss` closure.

### Fallback boundary (SENTINEL → existing R `jit_xla_impl` closure runs)

Tracing context; quickr backend; explicit `device` / `device_arg` move; (early
phases) nested trees / static args; any classification not yet handled. The R
fallback is today's closure, unchanged — so correctness is never worse than now,
only slower for unhandled cases.

---

## 6. Cache coherence

One source of truth: the **native** cache. The R fallback's `cache$get/$set`
become thin wrappers over the pjrt-side `cache`, so both paths share one map
keyed by the same `cache_key` (and one LRU recency/eviction).

---

## 7. Buffer lifetime / donation

Unchanged and reused: `execute` = `impl_loaded_executable_execute`, which already
pins zero-copy CPU input keepalives before Execute, migrates the RAWSXP keepalive
on confirmed donation, and defers releases. The dispatch only assembles the input
list (incl. freshly-allocated phantom buffers) and reads the output xptrs.

---

## 8. Phased implementation

1. **Dispatch skeleton in pjrt**: `pjrt_dispatch(handle, args, cache_miss)` that
   flattens (flat fast path), builds the `cache_key`, probes a native cache,
   calls back to R on miss, marshals + executes + wraps. xla / flat / device=NULL
   / no static; SENTINEL otherwise. anvl wires `prim_*` to call it (tiny R diff).
   Target ~250 → ~30–50 µs.
2. **Read avals natively**: dtype/shape/device straight off the buffer; only
   `ambiguous` comes from the leaf.
3. **General trees + static args + device moves**: native `build_tree` for
   nested structures, static-arg keys, device inference / `device_arg`. Shrinks
   the SENTINEL set toward zero. Approaches ~5–10 µs.

Re-run `benchmarks/prim-overhead.R` + `prim-dispatch-segments.R` and the full
test suite after each slice; the SENTINEL fallback guarantees correctness while
coverage grows.

---

## 9. Risks & open questions

- **Fallback correctness**: a misclassified case must SENTINEL, never silently
  mis-dispatch. Needs exhaustive classification tests (every input kind × device
  policy × backend).
- **GC / lifetime across R↔C++**: output xptrs, phantom buffers, and the
  keepalive migration must stay correct; reuse the existing pjrt logic, do not
  duplicate it.
- **Static-arg hashing**: equality must fall back to R `identical()` to stay
  collision-free.
- **Shared cache eviction**: with one native cache, LRU recency must be driven
  from a single place across the fast path and the R fallback.
- **pjrt scope**: this formally expands pjrt to own jit dispatch; update its
  docs/mission, and keep the `Node`/`aval`/`cache_key` names aligned with
  `anvl/R/flatten.R` so the two layers stay legible together.
- **Maintenance**: dispatch lives in two places (native fast path + R fallback)
  until the R path can be retired.
