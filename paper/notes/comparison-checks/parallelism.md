# Parallelism: how the compared packages use several cores

Script: `parallelism.R` in this directory (`Rscript parallelism.R`, about 5 minutes). Run on
2026-10-22 on macOS arm64 with 8 cores (6 performance cores), CPU only, CRAN R 4.5. Each timing
runs in a fresh R process. `cores` = CPU time / wall time, i.e. how many cores were busy on
average. Other jobs were running on the machine at the same time, so read the timings as ratios,
not as absolute numbers.

Versions: anvl dev (source, `load_all`), torch 0.17.0, tensorflow 2.20.0 (R) / TF 2.15 (Python),
greta 0.5.1, quickr 0.3.0.9000, ast2ast 1.0.2, odin2 0.3.46 + dust2 0.3.28 + monty, cmdstanr
0.9.0 / CmdStan 2.39.0, TMB 1.9.19, RTMB 1.8, nimble 1.3.0.

## Table cells ("Parallelism")

| Package | Parallelism |
|---|---|
| anvl | Multithreaded operations (XLA, automatic, no thread setting); several CPU devices (`PJRT_CPU_DEVICE_COUNT`) or GPUs driven from one R thread via asynchronous dispatch |
| torch | Multithreaded operations (`torch_set_num_threads()`); several devices only as GPUs (CUDA / MPS) |
| tensorflow | Multithreaded operations (`tf$config$threading`); several devices (GPUs, logical CPU devices) |
| greta | Through TensorFlow; chains vectorized in one graph; samplers on other processes via future |
| quickr | OpenMP loops (`declare(parallel())`) |
| ast2ast | None (separate R processes only) |
| odin2 + dust2 | OpenMP over particles / parameter sets (`n_threads`); chains in worker processes (monty) |
| Stan | Threads within a chain (`reduce_sum`, `map_rect`); chains in parallel processes; MPI, OpenCL |
| TMB | OpenMP (`parallel_accumulator`, `openmp()`) |
| RTMB | None (separate R processes only) |
| nimble | None (separate R processes only) |

Short form if space is tight: anvl "threads; multi-device", torch "threads", tensorflow "threads;
multi-device", greta "threads (TF)", quickr "OpenMP", ast2ast "–", odin2 "OpenMP", Stan
"threads, MPI, OpenCL", TMB "OpenMP", RTMB "–", nimble "–".

"Separate R processes" (parallel / future / callr) works for every package. The only difference
is what has to be redone in each worker (see "Shipping" below).

## Evidence per package

### anvl

- **Multithreaded operations, automatic.** One 2000x2000 f32 matmul uses 4.4 cores and `lgamma()`
  over 1e7 elements uses 3.4 cores, with nothing configured. XLA's CPU runtime spreads one
  operation over its own thread pool. There is no thread setting: pjrt only passes
  `cpu_device_count` to the CPU client, and `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` changed
  nothing (4.4 cores).
- **Several devices from one R thread.** With `PJRT_CPU_DEVICE_COUNT=4`, `pjrt::devices()` lists 4
  CPU devices, and a jitted function runs on the device its inputs live on (the output of the call
  on `cpu:2` lives on `CpuDevice(id=2)`). A call returns before the program has finished (the 4
  calls dispatch in under 0.01 s), so programs on different devices run at the same time:
  - sequential loop (`nv_scan`, 1e5 steps, runs on one core): 4 calls on 1 device 0.66 s → 1
    call on each of 4 devices 0.19 s (**3.4x**, 3.6 cores);
  - matmul: 0.29 s → 0.18 s (1.6x; one matmul alone already uses ~4.5 cores).
  The same mechanism drives several GPUs (not tested; no GPU on this machine).
- **Shipping.** An `AnvlArray` survives `serialize()`/`saveRDS()` (new since #634). A `jit()`
  function also works in another process as long as it has not been called yet. After a call,
  its cache holds compiled executables (external pointers), and calling it in another process
  fails with "external pointer is not valid". So each worker compiles for itself.

### torch

- **Multithreaded operations**, set by `torch_set_num_threads()` (default 6 here, 8 inter-op
  threads). This has to be set before the first parallel operation, which is why each setting
  runs in its own process. `lgamma` on 1e7: 1 thread 0.64 s → 6 threads 0.17 s (4.8 cores). On
  macOS the matmul goes to Apple Accelerate (AMX), so the thread setting makes no difference to it
  (0.09 s either way).
- **Devices**: CUDA, or MPS on Apple silicon (`mps = TRUE` here). Without a GPU there is only one
  CPU device. Spreading work over several devices is up to the user.
- **Shipping**: a tensor sent with base `serialize()` fails ("external pointer is not valid");
  `torch_serialize()` / `torch_load()` works.

### tensorflow

- **Multithreaded operations**, set by
  `tf$config$threading$set_intra_op_parallelism_threads()` / `..._inter_op_...` before TF
  initializes. matmul (`tf_function`): 1.39 s → 0.35 s (3.3 cores); `lgamma` with
  `jit_compile = TRUE` (XLA): 0.88 s → 0.40 s.
- **Devices**: several GPUs (`tf$device()`). The physical CPU can also be split into logical
  devices (`set_logical_device_configuration` → 2 logical CPU devices, verified).
- **Shipping**: a tensor sent to another process fails ("object is from previous session and is
  now invalid"); the objects are Python objects.

### greta

- Runs on TensorFlow, so operations are multithreaded. `mcmc()` runs all chains of a sampler
  *simultaneously in one TF graph* by vectorizing over chains (`?mcmc`, Details).
- `n_cores` ("maximum number of CPU cores used by each sampler") had **no measurable effect**
  here: 4 chains, n = 20000, p = 20: n_cores = 1 5.68 s / 3.2 cores, n_cores = 4 5.69 s / 3.2
  cores.
- Several samplers can run in parallel processes with `future::plan(multisession / cluster)`, not
  fork (`?mcmc`). The docs advise against doing this on one machine.

### quickr

- **OpenMP**: `declare(parallel())` (or `declare(omp())`) before a `for` loop or `sapply()`
  (`?declare`, NEWS 0.3.0). Thread count from `OMP_NUM_THREADS` / `OMP_THREAD_LIMIT`. 2e5
  iterations of an inner loop: 1 thread 0.80 s → 6 threads 0.15 s (5.7 cores). This works out
  of the box on macOS because quickr compiles Fortran with gfortran `-fopenmp`. Nothing else runs
  in parallel unless it is marked.
- No devices / GPU.
- **Shipping**: a `quick()` function in another process fails ("NULL value passed as symbol
  address"; the DLL is not loaded there). Compile in each worker, or put the function in a package.

### ast2ast

- **None.** `translate()` has no parallel option, and none of its docs or vignettes mention
  threads (the only thread-related code is a `thread_local` error-line buffer in the headers).
  `output = "XPtr"` returns a C++ function pointer that the user's own C++ code could call,
  e.g. from threads. The package itself provides nothing for this.
- **Shipping**: the translated function fails in another process ("NULL value passed as symbol
  address"). Translate in each worker.

### odin2 + dust2 (+ monty)

- **OpenMP over particles / parameter groups**: `n_threads` in `dust_system_create()`,
  `dust_filter_create()`, ... The odin2 "fitting" vignette says the likelihood "will scale fairly
  efficiently through to at least 10 cores".
  - With the default CRAN-R toolchain on macOS, C++ has no OpenMP (`SHLIB_OPENMP_CXXFLAGS` is
    empty, `dust_openmp_support(TRUE)$has_openmp_compiler = FALSE`), so `n_threads = 4` is
    silently ignored: 0.65 s → 0.66 s.
  - With a Makevars adding Homebrew libomp: 4000 particles of a stochastic SIR, n_threads 1 →
    4: 0.65 s → 0.17 s (4.0 cores).
- **Chains**: monty has `monty_runner_parallel()` (parallel package) and `monty_runner_callr()`
  (callr workers), i.e. chains in separate processes, plus `monty_runner_simultaneous()`
  (several chains in one model call).
- **Shipping**: a generator from `odin()` worked in another process (it points at the generated
  package on disk; the script uses a persistent `workdir`). odin2 recommends `odin_package()`
  for parallel use.

### Stan (cmdstanr)

- **Threads within a chain**: `reduce_sum()` / `map_rect()` with `cpp_options = list(stan_threads
  = TRUE)` and `threads_per_chain` (TBB). Logistic regression with n = 20000, p = 20, 1 chain:
  threads_per_chain 1 → 4: 2.62 s → 1.50 s (2.2 cores). The model has to be rewritten with
  `reduce_sum`.
- **Chains in parallel processes**: `parallel_chains`. 4 chains: 9.30 s → 2.96 s (3.1 cores), one
  CmdStan process per chain.
- Not tested: MPI for `map_rect` (`STAN_MPI`) and OpenCL GPU support (`STAN_OPENCL`, e.g. for
  `bernoulli_logit_glm`), both documented in the CmdStan / Stan user guides.
- **Shipping**: a `CmdStanModel` sent to another process works there, because the executable
  is a file on disk (sampled 50 draws).

### TMB

- **OpenMP**: `parallel_accumulator<Type>` in the template plus `openmp(n)` after loading the
  DLL. The thread count is 1 by default (`?openmp`). There is also experimental
  `openmp(autopar = TRUE)` with TMBad.
  - With the default macOS toolchain (no OpenMP): openmp(1) 3.74 s → openmp(4) 3.82 s, no
    effect.
  - With libomp Makevars: 30 gradients, n = 50000, p = 20: 3.73 s → 0.99 s (3.9 cores).
- **Shipping**: a `MakeADFun()` object works in another process once that process has loaded
  the DLL (`fn`/`gr` recreate the tape).

### RTMB

- **None.** RTMB is built from TMB, but its `src/Makevars` (GitHub, kaskr/RTMB) has no OpenMP
  flags, and the installed `RTMB.so` references no OpenMP symbols. `TMB::openmp(4, DLL = "RTMB")`
  returns 4 but cannot have any effect. The docs say that calling R functions from the tape
  (`ADjoint`) is "not thread safe ... cannot be done in parallel using OpenMP". Nothing else in
  the docs mentions parallelism.
- **Shipping**: a `MakeADFun()` object works in another process (the tape is rebuilt from the R
  function).

### nimble

- **None** inside compiled code. The nimble docs ("Parallelizing NIMBLE",
  https://r-nimble.org/examples/parallelizing_NIMBLE.html) run chains with `parallel::parLapply`
  on a PSOCK cluster, building and compiling the model inside each worker: handing one built
  model to the parallel packages "is likely to fail" or to give wrong results without an error.
  Forked workers need separate `dirName`s.
- **Shipping**: a compiled nimbleFunction fails in another process ("NULL value passed as symbol
  address").

## Shipping summary (compiled object → another R process)

| Package | Result |
|---|---|
| anvl | arrays: yes; `jit()` function: yes before first call, no after (recompiles in each worker) |
| torch | tensors: only via `torch_serialize()` |
| tensorflow | no (Python objects) |
| quickr, ast2ast, nimble | no (DLL symbol not loaded) |
| odin2/dust2 | yes (generated package on disk) |
| Stan | yes (executable on disk) |
| TMB | yes, once the worker has loaded the DLL |
| RTMB | yes (re-tapes) |

## Uncertain points

- anvl: I found no supported way to limit XLA's CPU threads. Whether XLA spreads *every* kind of
  operation over threads was not checked; matmul, elementwise `lgamma` and fused elementwise ops
  with a reduction (~3.4 cores) were. Not tested: GPUs, several GPUs.
- Concurrent jobs on the machine add noise; the speed-ups (≈3–6x) are well above that noise.
- greta's `n_cores` made no difference here. greta may set TF's thread pools after TF has
  already started, or the work may be dominated by other TF/Python threads. I did not check
  this further.
- OpenMP for dust2 and TMB needed a non-default setup on macOS (Homebrew libomp + Makevars). On
  Linux with GCC, OpenMP is available by default, so `n_threads` / `openmp()` work without it.
- Stan's MPI and OpenCL, and multiple GPUs for torch / TF, are cited from documentation, not
  run.
- ast2ast, RTMB and nimble: "none" means neither the package docs nor the compiled code offer
  any in-process parallelism (RTMB: no OpenMP symbols; ast2ast: no option in `translate()`).
