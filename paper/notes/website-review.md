# Website review: what the anvl site documents that the paper does not (yet) use

Sources read: `vignettes/hmc.Rmd`, all of `vignettes/articles/*.Rmd`, `README.Rmd`, `NEWS.md`,
`R/aaa.R` (options / env vars; there is no `R/package.R`), `NAMESPACE`, `R/jit.R`, `R/api-rng.R`,
`R/serialize.R`, `R/rules-reverse.R`, `../pjrt/README.Rmd` + `ffi.Rmd` / `pjrt.Rmd`,
`../stablehlo/README.Rmd`. Compared against `paper/paper.Rmd`, `paper/comparison-tables.md`,
`paper/notes/function-coverage.md`, `paper/notes/automatic_differentiation.R`.
Spot-checked locally (CPU): `gradient()` outside `jit()` errors; `saveRDS()`/`readRDS()` of an
`AnvlArray` works; `nv_print()` prints on every call of a jitted function.

## 1. Features on the website that are missing from the paper (or only in passing)

Format: what it is -- source -- why it matters -- plausibly unique among native R packages?

### RNG
- **Functional counter-based RNG.** `nv_rng_state(seed)` gives a `ui64[2]` Threefry state (`prim_rng_bit_generator(..., "THREE_FRY")`, as JAX does); each sampler returns `list(state, values)`, and floats come from mantissa bits. -- `random-numbers.Rmd`, `R/api-rng.R` -- Statisticians care about reproducibility: a draw depends only on the state passed in, not on global `.Random.seed`, call order, or (presumably) the thread pool. The paper says "explicit state" three times but never names the algorithm or explains *why* it matters (purity under `jit()`, and the R-level `rnorm()` inside a jitted function is evaluated once at trace time and frozen, see `next_steps.Rmd#random-numbers`). -- Unique as a *functional* RNG among the native packages: torch uses a global stateful generator per device; quickr/nimble/TMB/RTMB use R's RNG; dust2 has per-particle xoshiro streams; Stan seeds per run. TF (via Python) has stateless counter-based ops too. Whether the same state gives bit-identical draws on CPU and GPU is **not documented or tested** ("?"): Threefry bits should match, but `nv_rnorm`'s transform may differ in the last ulp.
- **Pitfall worth a sentence:** base `rnorm()` inside a jitted function runs once at trace time, so its values are baked into the program (`next_steps.Rmd`). Good for the "Tracing" section.
- `nv_sample(n, state, population)` never treats a scalar population as `1:n`, unlike `sample()` (`random-numbers.Rmd`). Another "resolves base R inconsistencies" example.

### Compilation, caching, static arguments
- **The cache is an LRU of `cache_size = 100` executables per backend**, keyed on dtype+shape of dynamic args, `identical()` value of static args, input tree structure, device, and default dtypes; `jit_cache_size()` inspects it. -- `internals.Rmd`, `jit.Rmd`, `R/jit.R` -- Reviewers will ask about memory growth under recompilation. The paper says "caches the executable" but does not give the key or the bound. -- XLA-specific; not in the other packages (quickr/Stan/TMB compile once, no shape key).
- **Static arguments must not have reference semantics** (environments, R6, external pointers are rejected; closures allowed if their environment does not change). -- `R/jit.R` docs, NEWS 0.4.0 -- Contradicts "static arguments of any type" in the paper and Table 1 (see §3).
- **Closed-over values are frozen at trace time; closed-over R `if` conditions silently select a branch that the cache key does not see** (`jit.Rmd`, "R `if` statements pick one branch", the `threshold` example). This is the main correctness pitfall of tracing and the paper only mentions purity and side effects. Same pitfall exists in JAX/`tf.function`/RTMB taping.
- **R loops unroll**, so long loops blow up compile time; `nv_while()` records one statement (`jit.Rmd`). Paper says "a loop is unrolled" but not the compile-time cost.
- **Padding to buckets (e.g. powers of 2) to avoid recompiles, and a table of neutral values for masked reductions** (`efficiency.Rmd`, `static_shapes.Rmd`). Paper mentions "padding inputs to a few fixed sizes" in one clause; the masked-mean pattern (sum of mask as divisor) is a nice concrete example for statisticians.
- **Eager mode = one cached tiny XLA program per primitive and input type**, so `browser()` and stepping work when `jit()` is removed (`efficiency.Rmd#eager-vs-jit`, `next_steps.Rmd#debugging`). Paper covers eager but not the debugging angle.
- **API functions are themselves `jit()`-wrapped** at definition, so a multi-op `nv_*` call is one program even eagerly; jitted functions called inside jitted functions are inlined ("dichotomy", `internals.Rmd`). Paper mentions inlining but not that the library itself is built this way.

### Debugging and errors
- **`nv_print()` prints actual values at run time on every call of a compiled program (CPU and GPU)**; `print()` inside `jit()` prints a placeholder once. Implemented as a PJRT custom call (`print_tensor`). -- `next_steps.Rmd#debugging`, `pjrt/ffi.Rmd`, NEWS 0.2.0 -- Missing from the paper entirely; reviewers always ask "how do I debug compiled code?". -- Comparable: `tf$print`, Stan `print()`, nimble `print()`; torch is eager anyway; quickr/RTMB/TMB: not in compiled code (?).
- **Compiled code cannot throw errors; all checks run at trace time against shapes/dtypes/static values.** Consequences documented on the site but not in the paper: a dynamic out-of-range index is **clamped** to the nearest valid index on read and **dropped** on write; `chol()` of a non-PD matrix returns `NaN`s; duplicate scatter indices give an unspecified winner that may **differ between CPU and GPU**. -- `next_steps.Rmd#compiled-code-cannot-throw-errors`, `subsetting.Rmd` -- Silent clamping is a correctness concern statisticians should be told about; the paper has one sentence ("performs all checks during tracing"). -- Shared with JAX/XLA; unlike torch, Stan (`reject()`), nimble, which raise.
- **Data-dependent input validation needs `static`** (e.g. checking `p` in `[0, 1]`), at the cost of one compile per value (`extending_api.Rmd`). Nice illustration of the static/dynamic split.
- Inference rules must check every StableHLO constraint because MLIR errors are not translated back (0-based axes in messages) (`extending_primitive.Rmd`). Honest limitation for "Extending".

### Memory
- **Value semantics** for `AnvlArray` (assignment never aliases), with opt-out via `jit(donate = )` and **eager `x[i, inplace = TRUE] <- v`**, which donates the buffer and invalidates other references (error on use). -- `next_steps.Rmd#value-semantics`, `subsetting.Rmd#in-place-updates`, `efficiency.Rmd#donation` -- Paper mentions `donate` once; value semantics vs torch's reference semantics (`x$add_()` mutates every alias) is a concrete, R-idiomatic difference. -- Unique among GPU packages in R (torch has reference semantics; TF tensors are immutable but `tf$Variable` mutates).

### Subsetting semantics (R-like but cleaned up)
- **Logical masks are supported** (`m[m > 6]`, `m[mask] <- 0L`) eagerly and in `jit()` when the mask is known at trace time; only masks computed from inputs are rejected. -- `subsetting.Rmd` -- The paper says masks are "not supported" (§3).
- `x[i]` always means `x[i, ]` (no flattened single-subscript indexing, no recycling of a short mask), `x[2]` drops, `x[arr(2)]` keeps; plain `c(2, 4)` is rejected in favor of `arr()`; negative indices unsupported; dynamic ranges `a:b` unsupported. Assignment never changes the array's dtype (`y[1] <- 0.5` on an int array errors, base R silently promotes). -- `subsetting.Rmd` -- The dtype-preserving assignment fits the paper's precision narrative.
- **Broadcasting aligns from the first axis** (size-1 axes appended), the column-major counterpart of NumPy's last-axis alignment, so a length-`nrow` vector meets a matrix as a column, matching base R's recycling in the common case. No recycling otherwise. -- `gotchas.Rmd` -- The paper says NumPy aligns from the last axis but never states what anvl does. **Plausibly unique**: torch and TF follow NumPy.

### Data types, NA, precision
- **NA handling:** float `NA` becomes `NaN`; `NA_integer_` survives as `INT_MIN` in `i32` with a warning both ways; other dtypes (incl. `bool`) error on `NA`; `as_array(check = "warn"|"err"|FALSE)`. -- `gotchas.Rmd#no-nas` -- The paper criticizes reticulate's NA conversion (§2) but never says what anvl does; a reviewer will ask. anvl's behavior (warn/error instead of silent conversion) is a fair contrast.
- **Unsigned / 64-bit integers round-trip to `bit64::integer64`**; `ui64` beyond the `integer64` range warns or errors (`gotchas.Rmd#unsigned-integers`, `anvl.Rmd`). Supports the "Data types" row.
- **Subnormals are flushed to zero on the XLA CPU backend** (storage preserved, arithmetic and comparisons flush; `-1e-310 >= 0` is `TRUE`); the quickr backend preserves them. -- `gotchas.Rmd#subnormal-floating-point-values` -- The paper makes precision a selling point and should acknowledge this caveat (one sentence in "Data types and precision"). -- Shared with JAX/TF-XLA; not an issue for Stan/TMB/nimble/quickr.
- `prim_convert()` / `nv_convert()` build an R double directly at the target dtype instead of rounding through the default float (`internals.Rmd`, R Values in Compiled Programs). Detail; fits §"R values have no data type".
- **Typed constructors** `nv_float32()`, `nv_int8()`, ... `nv_uint64()`, `nv_bool()` (NEWS dev). Minor.

### Devices, platforms, installation
- **Device inference:** a jitted function runs on the device named by its inputs, its captured arrays, or constructors with an explicit `device`; conflicts are errors; `jit(device = "cuda:0")` pins and copies inputs; the device is part of the cache key, so one jitted function holds executables for several devices. Options `anvl.default_device` / `ANVL_DEFAULT_DEVICE`, `with_default_device()`. -- `jit.Rmd#device-placement`, `internals.Rmd`, `R/aaa.R` -- Paper mentions "several devices per platform" only.
- **Platform matrix** (`pjrt::platform_support()`): Linux x86_64 and arm64 CPU+CUDA; macOS x86_64/arm64 CPU only; Windows x86_64 CPU native, CUDA only via WSL2; Windows arm64 unsupported. -- `README.Rmd` -- Not in the paper; relevant for a Hardware/OS row.
- **No CUDA toolkit needed**: a companion R package (`pjrt.cuda*`) ships the CUDA runtime libs; only a driver is needed. `install_anvl()`, `PJRT_INSTALL=0/1/unset`; prebuilt **Docker images** `anvl-cpu`/`anvl-cuda` (`:latest` daily, `:release`). Coexistence with torch's CUDA may need separate processes (mirai). -- `installation.Rmd` -- Paper covers the download; Docker images and the torch/CUDA clash are not mentioned.
- **XLA threads cannot be set from R**; use `taskset` (`faq.Rmd`). Limitation relevant on HPC.
- **BLAS/LAPACK**: CPU `svd`/`eigh`/`qr`/`lu`/`chol` call R's LAPACK via custom calls, so performance depends on R's BLAS build (`efficiency.Rmd`; `chol` via `potrf` "about ten times faster", NEWS dev). Interesting design point: anvl reuses R's own LAPACK.

### Serialization
- **`saveRDS()` works for `AnvlArray`s** (loads onto the default device; NEWS dev, tested locally), plus `nv_save()` / `nv_read()` / `nv_serialize()` / `nv_unserialize()` in safetensors format, interoperable with the `safetensors` package (and hence PyTorch/HF files). -- NEWS, `R/serialize.R`, `pjrt.Rmd#serialization` -- Paper mentions safetensors only in Table 3. torch arrays cannot be `saveRDS()`d directly (need `torch_save`/`torch_serialize`) (believed, "?"). Compiled executables are not persisted across sessions (not documented).

### Operations and extensibility
- **Higher-order primitives taking R functions:** `prim_reduce(x, init, axes, reducer = function(a, b) ...)` (e.g. a log-sum-exp reducer), `prim_sort` with a comparator, `prim_scatter` with an update function. Arbitrary reductions without a new primitive. -- `next_steps.Rmd` (primitives section) -- Not in the paper; strengthens "Extending" (the cheapest extension route).
- **Convolutions** `nv_conv1d/2d/3d` (`prim_convolution`) -- NAMESPACE, NEWS 0.4.0 -- Not in Table 3 or function coverage; relevant for spatial/time-series smoothing and CNNs.
- `map_tree()` / `pmap_tree()` for parameter lists, `tree_path()` etc. (`next_steps.Rmd`, `autodiff.Rmd`) -- Paper describes RTrees but not these helpers.
- **Alternative-forward reverse rules** (`rule_reverse(forward = ...)`) to share intermediates between forward and backward pass (`extending_primitive.Rmd`). Detail for Extending.
- **`gradient()` differentiates through `nv_if()`** including captured values; `nv_while()`/`nv_scan()` have no reverse rule (`autodiff.Rmd`, `R/rules-reverse.R`). Paper is consistent.
- **Whole optimizer loops in one program:** the GP article fits hyperparameters by gradient descent inside `nv_while()` + `gradient()` (`gaussian-process.Rmd`), and argues a user-written kernel is as fast as a built-in one. Good one-liner for the intro / user groups; complements HMC.
- Custom calls: pjrt ships a documented FFI tutorial (`pjrt/ffi.Rmd`) with built-in `print_tensor`, `geqrf/orgqr`, `lu`, `svd`, `eigh`, `potrf`, separate CPU and CUDA handlers. Paper already says this.
- quickr backend: `jit(..., unwrap = TRUE)` returns plain R arrays (`R/jit.R`). Minor.

### Efficiency guidance (useful for a "practical advice" paragraph or the HMC section)
- Always `await()` before timing; on CPU XLA runs *small* ops synchronously and only large ones asynchronously, so naive timings can look realistic for some ops and wildly optimistic for others (`faq.Rmd`). The paper's Table 1 cell "calls return once scheduled; CPU and GPU" is slightly stronger than this.
- Avoid `print()`/`as_array()` inside a training loop (forces sync); diagram in `efficiency.Rmd#asynchronous-execution`.
- `efficiency.Rmd` claims GPU gives "typically 10x to 100x over CPU"; this is unsupported on the site and should not be quoted unless the benchmarks back it.

## 2. Candidate new table rows (only discriminating ones)

Columns: anvl | torch | tensorflow/greta | Stan | quickr | ast2ast | odin2/dust2 | TMB/RTMB | nimble.
"?" = not verified.

- **Random numbers in compiled code**
  anvl: explicit functional state (Threefry), usable anywhere in the program |
  torch: global stateful generator per device (`torch_manual_seed()`) |
  TF: stateful and stateless (counter-based) ops; greta: via `calculate()`/`simulate()` |
  Stan: `_rng` only in transformed data / generated quantities, seed per run |
  quickr: `runif()` via R's RNG |
  ast2ast: – |
  odin2/dust2: per-particle parallel streams (xoshiro) |
  TMB/RTMB: R's RNG in `SIMULATE`/`simulate()`, not taped |
  nimble: R's RNG (`set.seed()`).
  (Partly duplicates the function-coverage RNG row; this row is about the *state model*, which the paper argues is distinctive.)

- **Array semantics / mutation**
  anvl: value semantics; opt-in donation (`donate`, `inplace = TRUE`) |
  torch: reference semantics, in-place `_` methods mutate aliases |
  TF: immutable tensors, mutable `Variable`s |
  Stan: local variables in model blocks (value) |
  quickr, ast2ast, nimble: R value semantics in R; in-place in compiled code (?) |
  odin2: state arrays updated by the system |
  TMB/RTMB: value (R side).
  Discriminates mainly anvl vs torch; drop the other columns if space is tight.

- **When compilation happens / cost**
  anvl: at first call per input signature (ms to s), in memory, no toolchain |
  torch: – (eager); TorchScript trace on demand |
  TF: at first call per signature (`tf.function`) |
  Stan: explicit C++ build (tens of seconds), executable cached on disk |
  quickr: at `quick()` call, or at package install time when used in a package |
  ast2ast: C++ build per `translate()` (seconds to tens of seconds, ?) |
  odin2: C++ build per system, cached (?) |
  TMB: C++ build per template, DLL on disk; RTMB: taping at `MakeADFun`, no C++ build |
  nimble: C++ build per `compileNimble()` call, per session.
  This is likely the most informative new row for statisticians.

- **Runtime errors in compiled code**
  anvl: none; checks at trace time; out-of-range dynamic index clamped, failed decompositions give `NaN` |
  torch: R errors (eager) |
  TF: errors in eager / graph mode; clamped under XLA |
  Stan: exceptions, `reject()` |
  quickr: ? | ast2ast: ? | odin2: ? |
  TMB/RTMB: ? (C++ exceptions / `NaN` objective) |
  nimble: errors in run code (?).
  Only worth adding if the "?" cells can be filled; otherwise mention in text.

- **Debug printing inside compiled code**
  anvl: `nv_print()` (every call); eager mode + `browser()` |
  torch: eager, `browser()` |
  TF: `tf$print()` |
  Stan: `print()` |
  quickr: ? | ast2ast: `print()` (?) | odin2: `print()` statements (?) |
  TMB: `std::cout`/`REPORT`; RTMB: R code before taping |
  nimble: `print()`/`cat()` in run code.
  Weakly discriminating; better as text.

- **Operating systems with GPU**
  anvl: Linux (x86_64, arm64); Windows via WSL2; CPU also native on Windows and macOS |
  torch: Linux, Windows (CUDA); macOS (MPS) |
  TF: Linux (CUDA); Windows only via WSL2 since TF 2.11 |
  Stan: OpenCL on all OSs (?).
  Could be folded into the existing Hardware row as a note.

- **Broadcasting** (could be merged into the "Conventions" row)
  anvl: scalars only; explicit `nv_broadcast_arrays()` aligns from the first axis |
  torch, TF: NumPy rules, from the last axis |
  Stan: explicit (scalar-vector only) (?) |
  quickr/ast2ast/nimble: R recycling (?) |
  odin2: index expressions.

- **NA handling** (only if space allows; the paper criticizes reticulate here)
  anvl: float `NA` becomes `NaN`, `NA_integer_` kept with warning, error otherwise |
  torch: ? | tensorflow: via reticulate (`NA` becomes `nan`) | Stan: data must be complete |
  quickr: ? | ast2ast: ? | odin2: ? | TMB/RTMB: `NA` in data allowed for `keep`/OSA (?) |
  nimble: `NA` data become latent nodes (imputed).

- Rows **not** worth adding: serialization (most packages can save their objects; not discriminating), threads, Docker.

## 3. Statements on the website that contradict the paper or tables (or each other)

- **Logical masks.** Paper §rapi: "logical masks are not supported in subsetting". Website (`subsetting.Rmd`): masks work eagerly and inside `jit()` when known at trace time, incl. `m[m > 6]` and `m[m > 6] <- 0L`; only masks computed from inputs inside `jit()` are rejected. **The paper is wrong**; reword to "a logical mask computed from the inputs of a jitted function".
- **"Static arguments of any type"** (paper §distinguishing and Table 1 "Inputs and outputs"). `jit()` rejects static arguments with reference semantics (environments, R6, external pointers) and compares with `identical()`. Suggest "static arguments of (almost) any R type".
- **Higher-order AD.** Paper and both tables say "HO (nested rev)". `autodiff.Rmd` (Limitations) says "Higher-order derivatives, such as Hessians, are currently not really supported". `notes/automatic_differentiation.R` verifies a Hessian via nested `gradient()` but records that second derivatives through `[` (scatter) fail. Either qualify the cell ("HO by nesting, not through all primitives") or fix the website; the two must agree.
- **AD requires `jit()`.** `autodiff.Rmd`: "the result of `gradient()` must be jitted" (verified: eager call errors). The tables' Execution row ("eager or compiled whole program") together with the AD row suggests eager AD as in torch. Add a note, e.g. AD cell "rev (scalar functions, inside `jit()`)".
- **`saveRDS()`.** `next_steps.Rmd` ("Saving `AnvlArray`s") says `saveRDS()` does not work; NEWS (dev) says it now does, and it does (tested). The website article is stale. Paper Table 3 lists only safetensors; it can add `saveRDS()`.
- **Asynchronous execution on CPU.** Tables say "✓ (calls return once scheduled; CPU and GPU)". `faq.Rmd`: on CPU, XLA may execute small operations synchronously and only dispatch larger ones to a background thread. Not wrong, but the cell overstates it; e.g. "✓ (GPU; CPU for larger programs)".
- **README says "Currently the only available transformation is reverse-mode AD"** and the internals article says the same; consistent with the paper. OK.
- **Precision narrative vs subnormal flushing.** Paper: "it should not lose precision without being asked to". `gotchas.Rmd`: XLA CPU flushes subnormals to zero in arithmetic and comparisons. Not a contradiction of a specific claim, but the paper should acknowledge it in §precision.
- `random-numbers.Rmd` says the explicit-state design "might change in the future to provide a more R-like experience", while the paper presents explicit state as a deliberate distinguishing feature (and the REVIEW note in the intro wants to emphasize it). Decide which message to send.
- Checked: paper Table 3's "55 of the 101 primitives" for quickr matches the installed package (101 `AnvlPrimitive`s, 55 with a quickr rule).
- Minor typos on the site (not the paper): `subsetting.Rmd` table header "Runtimew Value"; `anvl.Rmd` "unpleasent"; `next_steps.Rmd` "We also refer do".
