# Moving the Dispatcher out of pjrt and into anvl

## Problem

`pjrt` currently owns the **Dispatcher** (`dispatcher()`, `dispatch()`,
`dispatcher_size()`, `src/dispatch*.{h,cpp}`, `src/dispatch_key.h`,
`src/lru_cache.h`) and the **Rtree** module (`R/tree.R`, `src/tree.{h,cpp}`,
`src/hash.{h,cpp}`). Neither is used by pjrt itself:

* nothing in `pjrt`'s runtime (`buffer`, `client`, `program`,
  `loaded_executable`, `safetensors`, `ffi`, linalg) includes `tree.h`,
  `hash.h`, `lru_cache.h` or `dispatch*.h`. `pjrt_types.h` pulls in `dispatch.h`
  only to make the type visible to Rcpp attributes;
* the only R-level consumers of `build_tree()`/`flatten()`/`unflatten()`/
  `tree_*()`/`map_tree()` outside `pjrt` are in `anvl` (33 call sites);
* `pjrt` `Suggests: anvl` purely so `tests/testthat/test-dispatch.R` (1013
  lines) can drive the dispatcher through `anvl::jit()`. That is an inverted
  dependency: the tests of a pjrt feature cannot run without its own downstream.

Worse, anvl's data model has leaked into pjrt's C++. `dispatch.cpp` /
`dispatch_engine.cpp` hard-code the `"AnvlArray"` class, the `$data` /
`$backend` field contract, the `"plain"` backend sentinel, and anvl's dtype
vocabulary (`AnvlDtype`, spelled in tengen's names). pjrt cannot change any of
it without an anvl release, and anvl cannot change its array layout without a
pjrt release.

The dispatcher lives in pjrt only because pjrt already had a `src/` directory
and anvl did not. That is the whole reason, and it is not a good one.

**Goal:** the dispatcher, its cache key, and (see the decision below) the Rtree
module become part of `anvl`. `pjrt` keeps the runtime and gains a narrow,
documented **C interface** so anvl's own native code can read and drive PJRT
objects without linking against pjrt's C++ symbols.

## Decision to make first: does the Rtree move too?

The dispatcher's cache key *contains* an `RTree` by value, and hashes/compares
it (`tree_hash`, `tree_eq`), flattens into it (`flatten_rec`) and unflattens
outputs from it (`unflatten_rec`). So the two modules cannot be cleanly
separated by a C ABI; the choice is which side of the package boundary they
both end up on.

**Recommended: move the Rtree to anvl along with the dispatcher.**

* It has zero dependency on PJRT — it is pure structural code over `SEXP`.
* anvl is its only consumer, and already re-exports `flatten()`,
  `build_tree()`, `unflatten()`, `tree_size()`, `tree_path()`, `map_tree()`,
  `pmap_tree()` under its own name, so **the user-visible API does not change**;
  the re-exports simply become anvl's own exports.
* It removes all cross-package `RTree*` traffic. `in_tree` and `out_tree` are
  created and consumed entirely inside anvl, so no C++ struct
  (`std::vector<std::string>` and friends) ever crosses a DSO boundary.
* It was originally *moved out of anvl into pjrt* (see the header comment in
  `pjrt/tests/testthat/test-tree.R`) precisely because anvl had no `src/`. That
  constraint disappears in this refactor.

Alternative, if you want to keep pjrt's tree API: make `tree.h`/`hash.h`
header-only (inline `tree_hash`, `tree_path`; `tree_xptr` stays pjrt-side) and
ship them from `pjrt/inst/include`. Smaller diff, but it couples anvl's `src/`
to pjrt's C++ struct layout — an ABI dependency that a compiler or libstdc++
mismatch between the two installed packages breaks silently. The rest of this
plan assumes the recommended option and notes where the alternative differs.

## The pjrt C interface

pjrt must export, at C linkage, everything `PjrtEngine` currently reaches for by
calling C++ methods directly. From `dispatch_engine.cpp` that is exactly:

| current C++ call | why |
| --- | --- |
| `PJRTBuffer::element_type()` | leaf dtype for the cache key |
| `PJRTBuffer::dimensions()` | leaf shape for the cache key |
| `PJRTBuffer::device_ptr()` | the device token; also the `move_inputs` comparison |
| `PJRTBuffer::get_api()` | cross-client detection before a copy |
| `PJRTClient::api` | the other half of that comparison |
| `PJRTDevice::device` | the entry's device pointer |
| `new PJRTDevice(ptr, api)` + xptr | interning a device per `PJRT_Device*` |
| `impl_client_buffer_from_{double,integer,logical}` | uploading bare R data |
| `client_buffer_empty` | donation phantom buffers |
| `impl_buffer_copy_to_device` | `move_inputs` placement |
| `impl_loaded_executable_execute` | execution |
| `impl_execution_options_create` | the engine's reusable options object |
| `string_to_pjrt_buffer_type` | parsing `phantom_specs$dtype` |

### Shape of the API

New file `pjrt/inst/include/pjrt/api.h`, plus `pjrt/src/capi.cpp` holding the
implementations and their registration. Everything crossing the boundary is a
`SEXP`, an `int`, an `int64_t*`, or a `const char*` — **no C++ types, no Rcpp
types, no PJRT C++ classes**. That is what makes the interface survive
independent recompilation of the two packages.

```c
/* --- version handshake ------------------------------------------------ */
#define PJRT_C_API_VERSION 1
int pjrt_c_api_version(void);

/* --- object predicates ------------------------------------------------ */
int pjrt_c_is_buffer(SEXP x);
int pjrt_c_is_client(SEXP x);
int pjrt_c_is_device(SEXP x);
int pjrt_c_is_executable(SEXP x);

/* --- hot path: metadata, non-allocating, never fails ------------------ */
int            pjrt_c_buffer_dtype(SEXP buf);            /* PJRT_Buffer_Type; -1 on a non-buffer */
const int64_t *pjrt_c_buffer_dims(SEXP buf, int *rank);  /* borrowed, valid for the buffer's life */
const void    *pjrt_c_buffer_device_ptr(SEXP buf);       /* PJRT_Device*, opaque */
const void    *pjrt_c_buffer_api(SEXP buf);              /* PJRT_Api*, opaque */
const void    *pjrt_c_device_ptr(SEXP dev);
const void    *pjrt_c_client_api(SEXP client);

/* Interned PJRTDevice for one PJRT_Device*, created on first sight and kept
   alive by pjrt for the session. Replaces PjrtEngine::device_for_ptr(). */
SEXP pjrt_c_device_for_buffer(SEXP buf);

/* --- dtype vocabulary ------------------------------------------------- */
int         pjrt_c_dtype_from_name(const char *name);    /* -1 if unknown */
const char *pjrt_c_dtype_name(int dtype);                /* NULL if unknown */

/* --- allocation and execution ----------------------------------------- */
SEXP pjrt_c_buffer_from_r(SEXP client, SEXP device, SEXP data,
                          const int64_t *dims, int rank, int dtype);
SEXP pjrt_c_buffer_empty(SEXP client, SEXP device,
                         const int64_t *dims, int rank, int dtype);
SEXP pjrt_c_buffer_copy_to_device(SEXP buf, SEXP device, SEXP dst_client,
                                  int cross_client);
SEXP pjrt_c_execute(SEXP exec, SEXP inputs, SEXP options);
SEXP pjrt_c_execution_options(SEXP non_donatable_indices, int launch_id);

/* --- error channel ---------------------------------------------------- */
const char *pjrt_c_last_error(void);  /* NULL when the last call succeeded */
```

Three points that are easy to get wrong and should be settled up front:

1. **Errors must not longjmp across the boundary.** Today `Rcpp::stop` /
   `Rf_error` unwinds through pjrt's own frames. After the split it would
   unwind through *anvl's* C++ frames, skipping destructors of the `CacheKey`,
   `std::vector<ExecInput>` and `Rcpp::List` locals live at the call site. So
   every registered function wraps its body in `try { ... } catch (...)`,
   returns `R_NilValue` / a sentinel on failure, and stores the message where
   `pjrt_c_last_error()` reads it. The inline wrapper in `pjrt/api.h` checks the
   sentinel and raises the error *in anvl's own translation unit*, where the
   throw unwinds correctly.

2. **The hot path must stay allocation-free.** `pjrt_c_buffer_dims` returns a
   borrowed pointer into `PJRTBuffer::cached_dims_` rather than a copied vector;
   `pjrt_c_buffer_dtype` / `_device_ptr` read the same memoized `cache_meta()`
   fields they do now. Cost versus today is one indirect call per accessor.

3. **Device interning moves into pjrt.** `PjrtEngine::device_for_ptr()` today
   keeps a per-engine `unordered_map<PJRT_Device*, RObject>`. Making that a
   pjrt-side, session-lifetime intern table (`pjrt_c_device_for_buffer`) means
   a device object is canonical *by construction*, so anvl's
   `Engine::canonical_device()` reduces to identity for the pjrt engine. It also
   makes `as_pjrt_device()`'s "interning is recommended" note in `?dispatcher`
   a guarantee rather than advice. Worth doing as part of this move.

### Registration

`R_RegisterCCallable` calls go in a function tagged `// [[Rcpp::init]]`, which
Rcpp attributes emits a call to from the generated `R_init_pjrt` in
`RcppExports.cpp` — so no hand-written init file and no fighting the generator.

`pjrt/api.h` provides `static inline` stubs that resolve each symbol lazily via
`R_GetCCallable("pjrt", ...)` into a file-static function pointer, plus a
`pjrt_c_api_init()` that checks `pjrt_c_api_version() == PJRT_C_API_VERSION` and
errors with a "reinstall anvl against this pjrt" message. anvl calls it from its
own `R_init_anvl`. The macro guard is what turns a mismatched pair into a loud
failure instead of a crash.

## Work items

### pjrt

1. **Add** `src/capi.cpp`, `inst/include/pjrt/api.h`, the `[[Rcpp::init]]`
   registration, and `PJRT_C_API_VERSION`.
2. **Add** a pjrt-side device intern table backing `pjrt_c_device_for_buffer`.
3. **Add** tests for the new surface — `src/test-capi.cpp` (Catch) plus an R
   smoke test. Without them the exported interface is untested inside pjrt once
   the dispatcher leaves.
4. **Delete** `R/dispatch.R`, `man/dispatch.Rd`, `man/dispatcher.Rd`,
   `src/dispatch.{h,cpp}`, `src/dispatch_engine.{h,cpp}`, `src/dispatch_key.h`,
   `src/lru_cache.h`, `src/test-dispatch.cpp`, `src/test-lru_cache.cpp`,
   `tests/testthat/test-dispatch.R`, `benchmarks/jit-launch-overhead.R`.
5. **Delete** (recommended option) `R/tree.R`, `src/tree.{h,cpp}`,
   `src/hash.{h,cpp}`, `tests/testthat/test-tree.R`, and the 20 tree exports in
   `NAMESPACE`.
6. **Edit** `src/pjrt_types.h` — drop the `dispatch.h` include.
7. **Edit** `DESCRIPTION` — drop `Suggests: anvl`; bump version. Drop the
   `tengen` dependency if nothing else uses it (the dispatcher's
   `build_templates()` is the only `tengen::as_dtype` caller in `src/`; check
   `R/reexports.R` before removing).
8. **Edit** `_pkgdown.yml` (reference index), `NEWS.md`, `CLAUDE.md` /
   `AGENTS.md` (remove the Dispatcher and Rtree paragraphs and the
   `dispatch.R` / `tree.R` entries in *Key Source Files*; the pointer to
   `specs/design/dispatch/dispatch.md` is already dangling — pjrt has no
   `specs/`). Add a "pjrt's C interface" section documenting `pjrt/api.h`.
9. **Edit** `Config/build/compilation-database`, `compile_commands.json`
   regeneration — mechanical.

### anvl

1. **New `src/`**: `Makevars` (`CXX_STD=CXX17`; `PKG_CPPFLAGS` needs nothing
   beyond what `LinkingTo` supplies), `RcppExports.cpp`, `test-runner.cpp`,
   and a `Makevars.win` mirroring pjrt's if Windows is in scope. No `configure`
   — anvl needs neither protobuf nor LAPACK.
2. **Port** `dispatch.{h,cpp}`, `dispatch_engine.{h,cpp}`, `dispatch_key.h`,
   `lru_cache.h`, `tree.{h,cpp}`, `hash.{h,cpp}`, `test-dispatch.cpp`,
   `test-lru_cache.cpp`. Rename the `rpjrt` namespace to `anvl`. Rewrite
   `PjrtEngine`'s body against `pjrt/api.h`.
3. **New R files**: `R/dispatch.R` (internal `new_dispatcher()`, `dispatch()`,
   `dispatcher_size()` — **not exported**, they are an implementation detail of
   `jit()`), `R/tree.R` (moved verbatim), `R/catch-routine-registration.R`,
   `R/cpp-tests.R`.
4. **Edit** `DESCRIPTION`: `Imports: Rcpp`, `LinkingTo: Rcpp, pjrt, testthat`,
   `pjrt (>= <new version>)`; add `tree.R` and `dispatch.R` to `Collate`.
5. **Edit** `NAMESPACE`: `useDynLib(anvl, .registration = TRUE)`,
   `importFrom(Rcpp, sourceCpp)`; the seven tree re-exports in
   `R/reexports.R` become plain exports of anvl's own functions.
6. **Edit** `R/backend-pjrt.R` and `R/backend-quickr.R`: `pjrt::dispatcher` →
   the internal constructor, `pjrt::dispatch` → the internal `dispatch`. The
   `dispatch <- pjrt::dispatch` hoist (added to dodge `::` lookup cost) can go
   away — an internal binding is already a direct lookup.
7. **Edit** `R/utils.R:203-208` (`dispatcher_size`), and every `pjrt::tree_*` /
   `pjrt::flatten` / `pjrt::build_tree` / `pjrt::unflatten` / `pjrt::map_tree` /
   `pjrt::pmap_tree` / `pjrt::flatten_fun` call site in `R/graph.R`,
   `R/graph-to-quickr.R`, `R/primitives.R`, `R/reverse.R`, `R/rules-quickr.R`,
   `R/stablehlo.R`, `R/reexports.R` and `tests/testthat/test-reverse.R`.
8. **Move in** `tests/testthat/test-dispatch.R` (drop `skip_if_not_installed("anvl")`
   and the `anvl::` prefixes), `test-tree.R`, `test-cpp.R`, and pjrt's
   `benchmarks/jit-launch-overhead.R`.
9. **Edit** `CLAUDE.md` / `AGENTS.md`: anvl now has a `src/`; document the
   Dispatcher, the Rtree, and the `pjrt/api.h` boundary.

### Cross-cutting

* **CI / r-universe.** anvl becomes a compiled package. The three r-universe
  feeds (`r-xla.r-universe.dev`, `-cpu`, `-cuda`) and anvl's `R-CMD-check`
  matrix must build it with a C++17 toolchain and must build pjrt *first*
  (`LinkingTo` needs pjrt's installed headers). Check the build order is
  declared, not incidental.
* **The dependency cycle disappears.** Today `anvl Imports pjrt` and
  `pjrt Suggests anvl`. After this, the edge is one-directional.
* **Coordinated release.** anvl's `src/` cannot compile against a pjrt without
  `inst/include/pjrt/api.h`, so `Imports: pjrt (>= X)` and `LinkingTo: pjrt`
  must both name the new version, and pjrt must be released first.

## Staging

Land the two in-flight branches first — `pjrt@feat/rdata-input-dtypes` and
`anvl@feat/rdata-type` both change the dispatcher's `input_dtypes` protocol, and
rebasing this refactor over them afterwards is far more painful than the
reverse. Then branch off `main` in both repos.

* **Phase 1 — pjrt PR: add the C interface.** Items pjrt 1–3. Purely additive;
  the dispatcher still lives in pjrt and still calls the C++ directly. Release
  as pjrt `0.6.0`. Nothing downstream breaks.
* **Phase 2 — anvl PR: grow a `src/` and port the dispatcher.** All anvl items.
  anvl now uses its own dispatcher; pjrt's is dead code but still present, so
  `main` is green on both sides throughout. Gate the merge on the parity checks
  below.
* **Phase 3 — pjrt PR: delete.** Items pjrt 4–9. Release as pjrt `0.7.0`; anvl
  bumps its floor to it.

The cost of staging is that the dispatcher exists in two places between phases 2
and 3 — a few weeks of "fix it in both" if a bug surfaces. The alternative is a
single synchronized pair of merges, which is one atomic change but leaves no
window in which either `main` can be built and tested against a released
counterpart. Given these are separate repos feeding r-universe, staged is
safer; keep the window short.

## Verification

* **Behavioural parity.** `test-dispatch.R` is the contract, and it must move
  across unchanged apart from the `anvl::` prefixes and the skip helper. Every
  one of its cases — cache hits/misses, static keying, device agreement,
  `move_inputs`, `input_dtypes`, the `"plain"` backend rejection, the quickr
  closure engine, the GC loop — exercises a path that this refactor touches.
* **The Catch tests** (`test-dispatch.cpp`, `test-lru_cache.cpp`) move as-is;
  they only touch `dispatch_key.h` and `lru_cache.h`, neither of which changes.
* **Launch overhead.** Run `benchmarks/jit-launch-overhead.R` on
  `pjrt@main` + `anvl@main` before phase 2 and against the ported dispatcher
  after, on the same machine. The per-accessor indirect call is the only
  expected regression and should be in the noise; a visible one means something
  on the hot path started allocating (most likely a `dims` copy).
* **Memory.** The GC loop already in `test-dispatch.R` ("many dispatches with
  periodic `gc()`") plus a `valgrind` / ASan run of the dispatch tests, because
  ownership of the interned device objects changes hands in this refactor.
* **`R CMD check`** both packages, and check anvl on a machine with only the
  released pjrt installed, to prove the `LinkingTo` + version-guard story.

## Simplifications this unlocks (not in scope, worth noting)

Once the dispatcher is inside anvl, several indirections that exist only because
of the package boundary can go:

* the `extractor` callback — `ClosureEngine` can call anvl's `dtype()` /
  `shape()` / `device()` / `backend()` generics directly;
* the `backend` string as a constructor argument, and the `engine` selector
  derived from it;
* the `AnvlDtype` ↔ tengen-name round-trip in `build_templates()`, which exists
  because pjrt cannot see anvl's dtype objects.

Do these as a follow-up, after parity is established — not inside the move,
where a behavioural change would be indistinguishable from a porting bug.

## Interaction with an Rcpp → cpp11 migration

Short answer: no part of this plan gets harder, and the refactor makes a cpp11
migration meaningfully easier.

**The C interface is framework-agnostic by construction.** Everything crossing
the boundary is `SEXP` / `int` / `int64_t*` / `const char*`. Once it exists, pjrt
and anvl can be on different frameworks and migrate on independent schedules.
Today the opposite holds: the dispatcher's ~240 Rcpp references sit inside
pjrt's build, so migrating pjrt means migrating the dispatcher in the same
change.

Verified against cpp11 0.5.5:

* **The init hook exists.** `// [[cpp11::init]]` is picked up by
  `cpp_register()` (`generate_init_functions()`) and emitted as a call inside
  the generated `R_init_<pkg>`, taking `DllInfo*` — exactly like
  `// [[Rcpp::init]]`. The `R_RegisterCCallable` registration in Phase 1 works
  identically either way.
* **Direct analogues exist for every type the moving code uses:**
  `Rcpp::XPtr<T>` → `cpp11::external_pointer<T>` (same `(SEXP)` and
  `(pointer, use_deleter, finalize_on_exit)` constructors), `Rcpp::Function` →
  `cpp11::function`, `Rcpp::Environment::namespace_env("tengen")` →
  `cpp11::package("tengen")`, `Rcpp::Shield` → `cpp11::sexp`, `Rcpp::stop` →
  `cpp11::stop` (also printf-style, so call sites are unchanged).
* **The error-across-the-boundary rule gets easier, not harder.** cpp11's
  `unwind_protect` model is built around never letting an R longjmp cross C++
  frames, which is precisely the catch-and-return-sentinel shape the ABI
  requires of pjrt's registered functions.

Where the Rcpp weight actually is, in the files that move:

| file | `Rcpp::` refs |
| --- | --- |
| `dispatch_engine.cpp` | 76 |
| `test-dispatch.cpp` | 56 |
| `tree.cpp` | 48 |
| `dispatch.cpp` | 43 |
| `dispatch_key.h` | 2 (`Shield`) |
| `tree.h` | 1 (`stop`) |
| `lru_cache.h`, `hash.{h,cpp}` | 0 |

The correctness-critical code — the cache key, the tree encoding, the LRU — is
essentially framework-free. The Rcpp weight is all boundary glue, and
`dispatch_engine.cpp`'s share of it is the part this refactor rewrites against
the C interface anyway.

### Real friction points

1. **Catch tests.** cpp11's generated init adds `R_forceSymbols(dll, TRUE)`,
   and `cpp_register()` only registers `[[cpp11::register]]` functions — so
   `run_testthat_tests` is not in the call table. Rcpp gets it for free from the
   `catch-routine-registration.R` dummy (see `RcppExports.cpp:1045`). The ~750
   lines of Catch tests moving to anvl need a workaround; the `[[cpp11::init]]`
   hook runs *before* `R_forceSymbols`, which is where to put it.
2. **Hot-path list building.** `PjrtEngine::run()` builds an `Rcpp::List` of
   inputs per call and assigns by index. The `cpp11::writable::list` equivalent
   has proxy-assignment and copy-on-write semantics; index assignment within a
   pre-sized list should be equivalent, but this is the one spot to re-run
   `jit-launch-overhead.R` against.
3. **`Rcpp::as<std::vector<int64_t>>`** has no cpp11 counterpart (R has no
   int64 type); hand-roll it. Small, and already half-needed.
4. **`containsElementNamed`** has no counterpart — generalize the existing
   `anvl_field()` helper.

### Sequencing

If a cpp11 migration is already decided, **write anvl's new `src/` on cpp11
directly** and leave pjrt on Rcpp for now. anvl has no compiled code today, so
this avoids adding an Rcpp dependency it would only shed later, and it converts
the migration from one large coupled change into two small independent ones.
The risk that argues the other way — that a simultaneous port and API rewrite
makes a porting bug indistinguishable from a behavioural change — is limited
here, because the files with real Rcpp weight are the boundary glue being
rewritten regardless, and the files being relocated verbatim are nearly
framework-free.

If the migration is *not* decided, move on Rcpp; nothing in the C interface
forecloses converting either side later.
