# Source review: anvl code vs. paper.Rmd

Scope: `R/`, `tests/`, `inst/`, `NAMESPACE`, `NEWS.md` of anvl 0.5.1.9000 (this worktree), plus what anvl
uses from `../pjrt` and `../stablehlo`. Builds on `comparison-tables.md`, `notes/website-review.md`,
`notes/function-coverage.md` and `TODO.md`. Findings already listed there are not repeated (Threefry, cache key and
LRU size, static reference-semantics check, `nv_print()`, clamping and NaN instead of errors, value semantics and
`inplace`, NA handling, subnormals, `saveRDS()`, device inference, platform matrix, eager `gradient()` error,
`prim_reduce`/`sort`/`scatter` taking R functions, convolutions).
All checks ran on CPU with `pkgload::load_all()`. Scripts are in the session scratchpad (`v1.R` to `v5.R`).

## (a) Missing features, ranked by importance

1. **Which functions can be differentiated.** 89 of the 101 primitives have a reverse rule. The 12 without one are
   `lu`, `qr`, `svd`, `eigh`, `cumprod`, `convolution`, `scan`, `while`, `reduce`, plus `fill`, `iota` and
   `rng_bit_generator`, which need none. Because `nv_solve()`, `nv_det()`, `nv_determinant()` and `nv_inv()` are
   built on `prim_lu` (`R/api.R:2393-2421`), **`solve()`, `det()` and the inverse cannot be differentiated**.
   Verified: `jit(gradient(function(a) sum(nv_solve(a, b))))` fails with "No reverse rule for primitive lu". The same
   happens for `det`, `determinant`, `inv`, `eigh`, `svd`, `qr`, `cumprod` and `conv1d`. `chol()`,
   `triangular_solve()`, `sort()`, `quantile()`, `cumsum()`, `qnorm()`/`pnorm()`, `lgamma()`/`digamma()` and
   `dbinom()` can be differentiated. This matters for statisticians: GP and mixed-model likelihoods need
   log-determinants. Suggested sentence (Linear algebra row or the AD section): "Gradients are available through the
   Cholesky factorization and triangular solves, so log-determinants and solves of covariance matrices can be
   differentiated via `chol()`; LU-, QR-, SVD- and eigendecomposition-based functions, including `solve()` and
   `det()`, do not have reverse rules yet."
2. **A native jit dispatch path.** The cache lookup, the flattening of argument trees, input validation, the upload
   of bare R values and the wrapping of outputs all run in C++ in pjrt's `dispatcher()`/`dispatch()`
   (`R/backend-pjrt.R:47-79`, `../pjrt/R/dispatch.R`, `../pjrt/src/dispatch*.cpp`, `lru_cache.h`). R is called back
   only on a cache miss. A cache hit costs about 30 microseconds per call (measured: `x + 1L` on a 3-vector, CPU).
   For comparison, an eager `nv_array()` plus `+` takes about 0.2 ms. The quickr backend uses the same dispatcher
   through a "closure" engine (`R/backend-quickr.R:61-81`). Suggested: "Calls to a compiled function are dispatched
   natively: pjrt hashes the input signature, looks up the executable and wraps the outputs in C++, so a cached call
   costs tens of microseconds, and R code runs only when a program has to be traced."
3. **RTrees are implemented in pjrt, not anvl.** `flatten`, `unflatten`, `build_tree`, `map_tree` and `tree_path`
   come from pjrt (`R/reexports.R:314-340`, `../pjrt/src/tree.cpp`). This fits the "pure R" claim only if pjrt counts
   as the runtime (see (b)).
4. **Matrix products default to `precision = "highest"`** (`R/primitives.R:292`, `R/api.R:2267`, convolution
   `R/primitives.R:3693`). JAX's default lets XLA use TF32/bf16 passes on GPUs; anvl asks for full f32. This
   strengthens the precision narrative, so add one sentence to the "Choosing the precision" section: "Matrix products
   and convolutions request XLA's highest precision by default, so `f32` products on GPUs are not silently computed
   in TF32."
5. **Random variate algorithms.** Normals use Box-Muller on mantissa-filled uniforms (`R/api-rng.R:147`, `:183-215`).
   `nv_rbinom()` sums `size` Bernoulli draws, costs O(n x size), and takes `size` and `prob` as **static scalars**:
   each new `prob` recompiles, and an array `prob` errors (`R/api-rng.R:245-300`; verified). `nv_rng_state()` takes a
   32-bit seed (`R/api-utilities.R:32-38`). There is **no split/fold-in**, so independent streams per chain or device
   come from different seeds or from threading one state through. Table 3's "binomial variates" is overstated for
   statistical use. Suggested: "Normal variates use the Box-Muller transform; binomial variates are currently limited
   to scalar, static parameters."
6. **Interop is at the pjrt level only.** pjrt runs a JAX-exported program (`../pjrt/tests/testthat/test-program.R:12`,
   `inst/programs/jax-stablehlo.mlir`). anvl has no API to wrap a foreign StableHLO program as a primitive, so the
   program cannot be called inside `jit()` or differentiated. Export works: `repr(stablehlo(trace_fn(f, avals))[[1]])`
   gives MLIR text. `pjrt::inspect_hlo()` (`../pjrt/R/hlo.R:67`) shows the HLO after XLA's optimization, given
   `XLA_FLAGS`. Suggested: "`stablehlo()` exports the StableHLO text of a traced graph, and `pjrt::inspect_hlo()` shows
   the program after XLA's optimizations, which helps to check, e.g., fusion."
7. **Multiple CPU devices need `PJRT_CPU_DEVICE_COUNT`** (`../pjrt/R/client.R:51-67`). The benchmark's 16 CPU devices
   depend on it, but the paper never says how they are created. Suggested: "The number of CPU devices is set via
   `PJRT_CPU_DEVICE_COUNT`; each device has its own stream, so independent programs, e.g., groups of chains, run
   concurrently." Arrays move between devices only via `jit(device = )` or `nv_array(x, device = )`. There is no
   eager transfer function and no collective operations.
8. **Batched linear algebra.** `nv_matmul()`, `chol()`, `triangular_solve`, `lu`, `qr`, `svd` and `eigh` operate on
   leading batch axes (e.g. `R/rules-stablehlo.R:1004-1006`, `R/api.R:2294-2300`). This is directly relevant to the
   hand-batched multi-chain code the paper recommends instead of `vmap()`.
9. **The pjrt GC-on-OOM fallback.** When an allocation returns `RESOURCE_EXHAUSTED`, pjrt runs R's `gc()` and retries
   once (`../pjrt/src/utils.h:50-79`, `src/gc.cpp`). This is the same remedy the paper attributes to torch (see (b)).
   Also, a thread-safe deferred-release queue lets PJRT callback threads release R objects
   (`../pjrt/src/deferred_release.h`).
10. **The quickr backend's actual reach.** It has no rules for `rng_bit_generator`, `scan`, `sort`/`top_k`, the
    cumulative ops, any decomposition (`chol`, `lu`, `qr`, `svd`, `eigh`, `triangular_solve`), the special functions
    (`lgamma`, `digamma`, `erf*`) or the inverse trigonometric and hyperbolic functions. So **neither the HMC sampler
    nor any RNG code runs on it**. Its default float is `f64`, not `f32` (`R/backend-quickr.R:253`). One sentence
    would set expectations better than "55 of 101".
11. **Extensible promotion rules.** Besides the four rules named in the paper, `promotion_grouped()` and the
    constructor `promotion_rule()` let developers define their own (`R/promotion.R:173`, `:264`). Most rules also
    take `on =`, which selects the arguments a rule applies to. Suggested: append "and `promotion_rule()` builds new
    rules" to the Promotion rules section.
12. **Graph passes are minimal.** anvl runs only two passes before lowering: inlining one-element constants and
    removing unused ones (`R/graph-passes.R:49-112`). There is no CSE, DCE or constant folding; everything is left to
    XLA. That is fine, but the paper's "graph transformations" framing could say that optimization is XLA's job.
13. **Platform-dependent lowering.** Lowering rules read the target platform, e.g. SVD on CUDA mirrors JAX's
    transpose for m < n (`R/rules-stablehlo.R:1233-1268`), and CPU and CUDA handlers differ per custom call. A traced
    `AnvlGraph` is backend-agnostic, but the StableHLO it lowers to is not platform-agnostic once custom calls are
    involved.
14. **Testing infrastructure** (one sentence at most): 56 test files, about 2,550 `it()`/`test_that()` blocks, about
    23k lines. They include torch comparisons for forward and reverse rules (`inst/extra-tests/*-torch.R`),
    eager-vs-jit equivalence over dtype x R-value grids (`tests/testthat/test-eager-jit-equivalence.R`),
    quickr-vs-PJRT agreement (`test-primitives-quickr-pjrt.R`), a meta test that every primitive is named and
    documented consistently (`test-primitives-meta.R`), CI runs under `ANVL_DEFAULT_DTYPES` and `ANVL_DEFAULT_DEVICE`
    overrides (AGENTS.md), and expensive RNG QQ checks (`inst/random/`).
15. **No on-disk executable cache.** Every R session retraces and recompiles. Worth one clause next to the
    Stan/TMB compile-time comparison (Stan keeps its binary on disk).

Not present at all, so the paper is correct to omit them or call them future work: vmap, forward mode, sharding or
collectives, remat/checkpointing, profiling hooks, a side-effect registry or ordering tokens (`nv_print()` is a
`has_side_effect` custom call with no tokens, `R/rules-stablehlo.R:676-697`), FFT, Metal and TPU.

## (b) Inaccurate or misleading claims

- **paper.Rmd:427** says every primitive "stores a `"reverse"` rule". In fact 12 of 101 do not (list in (a)1), and
  `graph_gradient()` errors on them. Fix: "every differentiable primitive".
- **paper.Rmd:281** (Table 3, Linear algebra) lists solve/QR/LU/SVD/eigen/det without saying they cannot be
  differentiated, while line 277 says "all of them compose". See (a)1.
- **paper.Rmd:433** says "R literals are embedded into the program as constants, while arrays that the traced
  function closes over become additional inputs". The actual rule is by size, not origin: one-element values are
  inlined, including a closed-over one-element `AnvlArray` (`inline_scalarish_constants`,
  `R/graph-passes.R:49-83`). Non-scalar constants become program inputs whether they are `AnvlArray`s or **R
  matrices** (`R/array.R:946-949`, `R/stablehlo.R:74-75`). Verified: in `function(x) x + B` with an R matrix `B`, `B`
  shows up as constant input `%c1`. Fix: "One-element constants are inlined into the program, while larger constants,
  whether R arrays or `AnvlArray`s, become additional inputs."
- **paper.Rmd:435** says "a compiled quickr function can be reused for inputs of different shapes". Under `jit()`,
  the quickr backend goes through the same shape-keyed dispatcher (`R/backend-quickr.R:61-81`). Verified: two shapes
  give two traces and `jit_cache_size() == 2`. The reuse is a property of quickr itself, not of anvl's quickr backend
  today. Reword as potential, consistent with the Conclusion's "routes towards dynamic shapes".
- **paper.Rmd:211-213, 284, 573** say "further backends can be added without changing user code" and "new backends
  via `AnvlBackend`", and that the backend comes "together with a lowering rule for each supported primitive". But
  `register_backend()` is not exported (`R/backend.R:97`), `AnvlBackend` is `@keywords internal`
  (`R/backend.R:43-44`), and the rule slots are a fixed set `c("stablehlo", "quickr", "reverse")` (`R/aaa.R:72`) that
  `[[<-` enforces (`R/primitive.R:43-46`). Lowering rules live in the primitives, not in the backend. So a new
  backend, or a new transformation such as forward mode (line 572: "guided by a rule stored in each primitive"),
  requires changing anvl itself. Either export a registration API or soften the wording to "the design allows adding
  backends; currently this requires changes to anvl".
- **paper.Rmd:213** says "except for the PJRT runtime, anvl and stablehlo are pure R packages". This is true for the
  packages, but the jit cache, tree flattening and dispatch are C++ in pjrt (see (a)2-3). Say "pjrt, which provides
  the runtime and the native dispatch of compiled calls".
- **paper.Rmd:460** contrasts torch triggering the GC with anvl. pjrt also runs `gc()` and retries on
  `RESOURCE_EXHAUSTED` (`../pjrt/src/utils.h:50-79`). The sentence is not false, but it implies anvl needs no such
  mechanism. Suggest "... which \pkg{torch} and \pkg{pjrt} mitigate by triggering the collector when device memory
  runs out".
- **paper.Rmd:517** says "The rules refuse conversions that lose precision unless `coerce = TRUE`". Only
  `promotion_like()` and `promotion_dtype()` have `coerce`. `promotion_common()` turns `i64 + f32` into `f32`
  (verified), which loses integers above 2^24, as in JAX. Line 496 ("integers never become floats on their own") is
  technically right but invites the reading that anvl never loses integer precision. Add "an integer that meets a
  float takes the float's data type, as in JAX".
- **paper.Rmd:129, 454** say the gradient "can be called from within a jitted function". It must be (already in
  website-review). It is restated here because it contradicts line 337 (operations "executed immediately") for AD.
- **paper.Rmd:209** says "JAX can compile the StableHLO programs that \pkg{anvl} produces". No test or example backs
  this, whereas the reverse direction is tested. Either add a check or drop that half.
- **paper.Rmd:232, 258** (tables) write "u8--u64". anvl's dtype names are `ui8`-`ui64` (`R/backend-pjrt.R:270-271`).
  Use anvl's spelling or define the abbreviation.
- **paper.Rmd:282** (Table 3, Random numbers) says "binomial variates": only for scalar, static `size` and `prob`
  (see (a)5).
- **paper.Rmd:362, 280** say "about 100 R generics". This is OK only if anvl's own generics are counted (see (c)).

## (c) Recounted numbers

| Claim | Paper | Count | How |
|---|---|---|---|
| `nv_*` functions | "about 190" (280) | **194** exported (11 typed constructors `nv_bool`/`nv_int*`/`nv_uint*`/`nv_float*`; 10 are primitives themselves, e.g. `nv_if`, `nv_while`, `nv_print`) | `getNamespaceExports()` |
| R generics with methods | "about 100" (280, 362) | **101** S3 methods for `AnvlArray`; **90** are base/stats generics (excluding `as_array`, `as_raw`, `await`, `backend`, `device`, `dtype`, `platform`, `shape`, `hlo_scalar`, `hlo_tensor`, `as.integer64`) | `S3methods` of the namespace |
| Primitives | 101 (278) | **101** unique (113 `AnvlPrimitive` objects, including aliases such as `nv_abs`) | `inherits(, "AnvlPrimitive")`, deduplicated by name |
| quickr rules | 55 (278) | **55** | correct |
| reverse rules | (none) | **89** of 101 | -- |
| stablehlo rules | (none) | 101 of 101 | -- |
| Data types | bool, i8-i64, u8-u64, f32, f64 | 11 dtypes on pjrt, as stated; **quickr: f64, i32, bool only**, default float f64 | `R/backend-pjrt.R:270`, `R/backend-quickr.R:152, 253` |
| Distributions with d/p/q | four (282) | **4** (norm, unif, exp, binom) plus `nv_plogis()` | `R/api-distributions.R` |
| StableHLO operations | "around 100" (307) | stablehlo exports **97** `hlo_*` builders, including CHLO ops (`digamma`, `lgamma`, `erf*`, `top_k`, `is_inf`, ...) and structural ones (`hlo_func`, `hlo_return`, ...) | `../stablehlo/NAMESPACE` |
| Custom calls used | -- | 7 targets: `print_tensor`, `potrf`, `geqrf`, `orgqr`, `lu` (getrf), `svd`, `eigh`, each with CPU (LAPACK from R) and CUDA (cuSOLVER) handlers | `R/rules-stablehlo.R`, `../pjrt/src/*{,_cuda}.cpp` |
| Graph passes | -- | 2 | `R/graph-passes.R:85` |

## (d) Design aspects worth one sentence each (ranked)

1. **Native dispatch with R-side tracing**: tracing and lowering are in R, but steady-state calls never enter the R
   interpreter beyond one `.Call` (see (a)2). This answers the reviewer question "isn't R too slow for this?".
2. **The R argument's dtype is decided by the program, not the caller.** The dispatcher keys R values by R type and
   shape, and the compiled entry declares the dtype at which each R argument is uploaded (`input_dtypes`,
   `R/backend-pjrt.R:236-240`, `R/rdata.R:418-440`). That is how one cached program serves `f64` uploads of R
   doubles. It is the mechanism behind Section "R values have no data type".
3. **Default dtypes and device are part of the cache key** (`context = default_dtypes_context()`,
   `default_device = function()`, `R/backend-pjrt.R:58-65`), so scoped `with_default_dtypes()` never serves a stale
   program.
4. **One active backend, read at call time**: one `JitFunction` holds a lazily built implementation per backend
   (`R/jit.R:138-206`); arrays of another backend are an error rather than being converted.
5. **Two-layer API with jit-wrapped library functions**: every multi-op `nv_*` is itself `jit()`-wrapped, so an
   eager call of a composite function is one XLA program, and under tracing it is inlined.
6. **Reuse of R's LAPACK on the CPU**: CPU decompositions call the LAPACK that R is linked against via XLA FFI
   handlers, so an optimized BLAS/LAPACK build (OpenBLAS, MKL, Accelerate) speeds them up. Row-major layouts are
   passed to `potrf` to avoid transposes (`R/rules-stablehlo.R:1008-1029`).
7. **ALTREP carrier for buffers** (`../pjrt/src/buffer_sexp.cpp:93-97`): serialization works without anvl knowing
   about it.
8. **A "plain" pseudo-backend** holds constants created during tracing without allocating on a device
   (`R/backend.R:160-216`, `R/array.R:223-228`), which keeps graphs device-agnostic until lowering.
9. **Phantom donated inputs** (`donate_unaliased_outputs = TRUE`, `R/backend-pjrt.R:194`): outputs without a donated
   input to alias still get preallocated buffers, a detail relevant to the `donate` claim.
10. **CHLO ops in stablehlo**: special functions lower to CHLO decompositions that XLA expands, so `lgamma` and
    `digamma` cost no anvl-side implementation, while the distribution functions are implemented in anvl with
    per-precision coefficients (Cephes and others, `R/api-distributions.R:245`).
