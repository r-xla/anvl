# Comparison tables

Working copy of the two comparison tables in `paper.Rmd` (`tab:comparison-gpu`, `tab:comparison-cpu`). We iterate here and port the result back into the LaTeX tables afterwards.

## Table 1: packages with GPU support

|  | anvl | torch | tensorflow, greta | Stan (cmdstanr) |
|---|---|---|---|---|
| **Core focus** | general array programs with compilation and AD on CPUs and GPUs, e.g., for custom samplers and optimizers | deep learning | tensorflow: deep learning; greta: Bayesian modelling (probabilistic programming) | Bayesian inference for models in its own language (also optimization, variational inference) |
| **Abstraction level** | building blocks: array operations, `jit()`, `gradient()`, control flow, random numbers; no built-in samplers or optimizers, users assemble algorithms | building blocks (tensors, autograd) plus high-level modules (`nn_module()`, optimizers, distributions) | tensorflow: building blocks plus Keras; greta: high-level, users define a model and call `mcmc()` or `opt()` | high-level: users write a model, predefined algorithms (NUTS, ADVI, Pathfinder, L-BFGS); custom algorithms only outside Stan via model methods |
| **Program written in** | R (array API) | R (array API) | R, run by Python libraries | Stan language |
| **Program built by** | tracing | – (eager); tracing (TorchScript) | tracing (`tf.function`) | parsing the model language |
| **Inputs and outputs** | (nested) lists of arrays and R values; static arguments (no environments) | any R objects (eager); typed tensors and lists (TorchScript) | nested lists of tensors (`tf_function()`); greta: data and model variables | variables declared in the data and parameters blocks; output: draws |
| **Arrays** | scalars and arrays of any rank | scalars and tensors of any rank | scalars and tensors of any rank | scalars, vectors, matrices, and arrays of any rank (linear algebra only on vectors and matrices) |
| **Execution** | eager or compiled whole program | eager, per operation | eager or compiled graph | compiled model |
| **Asynchronous execution** | ✓ (calls return once scheduled; CPU and GPU) | (✓) (CUDA kernels only; CPU operations block) | (✓) (GPU kernels only) | – |
| **Parallelism** | multithreaded operations (XLA); several CPU devices or GPUs from one R thread via asynchronous dispatch | multithreaded operations; several devices only as GPUs | multithreaded operations; several devices; greta: chains vectorized in one graph | threads within a chain (`reduce_sum()`, `map_rect()`); chains as parallel processes |
| **Compiler** | XLA; Fortran (experimental) | – (TorchScript tracing) | XLA (optional) | C++ |
| **Intermediate representation** | traced into an `AnvlGraph`, lowered to StableHLO (MLIR), compiled by XLA via LLVM; quickr backend: transpiled to R with type declarations, then Fortran | none in eager mode (LibTorch kernels); TorchScript IR via `jit_trace()` / `jit_compile()` | TensorFlow graph, compiled to XLA HLO with `jit_compile = TRUE`; greta: TensorFlow graph | transpiled: Stan language to C++ (stanc3), compiled by a C++ compiler |
| **AD** | rev (scalar functions), HO (nested rev) | rev, HO | fwd, rev, HO | rev (fwd and HO internal only) |
| **Hardware** | CPU, NVIDIA GPU | CPU, NVIDIA GPU, Apple GPU | CPU, NVIDIA GPU | CPU; GPU via OpenCL for log densities of 38 distributions, incl. GLMs |
| **GPU memory** | intermediate results of compiled programs managed by XLA (buffer reuse, donation); R objects freed by R's collector, which `pjrt` runs when an allocation fails | caching allocator; R's collector run when an allocation fails or reserved memory exceeds a threshold | Python reference counting after R's collector releases the reference; TensorFlow reserves most GPU memory by default | OpenCL buffers allocated and freed in C++, not visible to R |
| **Data types** | bool, i8–i64, u8–u64, f32, f64 | bool, i8–i64, u8, f8, f16, bf16, f32, f64, c32, c64, c128 | bool, i8–i64, u8–u64, f16, bf16, f32, f64, c64, c128 (greta: f32, f64) | i32, f64, c128 (no booleans; conditions are i32) |
| **Data-dependent control flow** | `nv_if()`, `nv_while()`, `nv_scan()` | eager R control flow; TorchScript via `jit_compile()` | `tf$cond()`, `tf$while_loop()`, `tf$scan()`, or R syntax via autograph; greta: – | ✓ |
| **Differentiable control flow** | `nv_if()` and loops unrolled during tracing; not `nv_while()`, `nv_scan()` | ✓ | ✓ (`tf$while_loop()` under XLA needs `maximum_iterations`); greta: unrolled loops only | ✓ |
| **Random numbers** | anywhere in the program; explicit functional state (Threefry); CPU and GPU | eager; global generator per device or `Generator` objects | in `tf_function()`; stateless seeds or `Generator`, `split()` for streams; greta: via `calculate()` / `simulate()` | only in generated quantities and transformed data; one stream per chain |
| **Shapes** | static, recompiled per shape | dynamic | static under XLA | dynamic |
| **Conventions** | 1-based, column-major | 1-based, row-major | 0-based, row-major (partly translated) | 1-based, column-major |
| **New operations** | primitives with rules in R; C/C++ custom calls | composition in R; C++ | composition; Python, C++ | C++ functions |
| **Dependencies** | PJRT plugin (downloaded) | LibTorch (downloaded) | Python, TensorFlow | C++ toolchain, CmdStan |

Caption: Comparison of anvl with R packages for compiled or differentiable numerical computing that support GPUs. Stan is used via cmdstanr or rstan. ✓: supported; (✓): possible with limitations; –: not available; fwd: forward mode; rev: reverse mode; HO: higher-order derivatives.

## Table 2: CPU-only packages

|  | anvl | quickr | ast2ast | odin2, dust2 | TMB, RTMB | nimble |
|---|---|---|---|---|---|---|
| **Core focus** | general array programs with compilation and AD on CPUs and GPUs, e.g., for custom samplers and optimizers | speeding up numerical R functions on the CPU | fast R functions called repeatedly, e.g., ODE right-hand sides, likelihoods, objectives | dynamical systems (difference and differential equations), e.g., in epidemiology, fitted by particle filters and MCMC | maximum likelihood for random-effects models via the Laplace approximation | hierarchical (graphical) models and algorithms for them, e.g., MCMC and SMC |
| **Abstraction level** | building blocks: array operations, `jit()`, `gradient()`, control flow, random numbers; no built-in samplers or optimizers, users assemble algorithms | building blocks: compiles a user's R function; no predefined algorithms | building blocks plus a few predefined numerics (`uniroot()`, `lbfgsb()`, `pso()`, `nnls()`, `jacobian()`) | high-level: users write a model, predefined particle filters and samplers (monty) | objective plus predefined Laplace approximation and `sdreport()`; optimization and other algorithms run in R around the tape | both: predefined samplers (MCMC, SMC) that users configure, and `nimbleFunction()` to write custom algorithms |
| **Program written in** | R (array API) | typed subset of R | typed subset of R | R-like language for difference and differential equations | C++ template; R (RTMB) | BUGS-like language, subset of R |
| **Program built by** | tracing | static analysis | static analysis | static analysis | C++ compilation (TMB); tracing (RTMB) | static analysis |
| **Inputs and outputs** | (nested) lists of arrays and R values; static arguments (no environments) | typed atomic arrays; returns one array or a flat named list | typed scalars, vectors, matrices, user-defined structs; returns one value | parameters and data (scalars, arrays); output: trajectories, likelihood | TMB: `DATA_*` / `PARAMETER_*` declarations; RTMB: named list of numeric parameter arrays; output: scalar objective, `REPORT()` | typed scalars and arrays; `nimbleList` for structured values |
| **Arrays** | scalars and arrays of any rank | scalars and arrays of any rank | scalars, vectors, and matrices as arguments; higher ranks only allocated inside | scalars and arrays (up to 8 axes documented) | scalars and arrays of any rank | scalars and arrays up to 6 axes; reductions such as `sum()` up to 2 axes |
| **Execution** | eager or compiled whole program | compiled function | compiled function | compiled model, run in parallel by dust2 | compiled C++, AD tape | compiled C++ |
| **Asynchronous execution** | ✓ (calls return once scheduled; CPU and GPU) | – | – | – | – | – |
| **Parallelism** | multithreaded operations (XLA); several CPU devices or GPUs from one R thread via asynchronous dispatch | OpenMP loops (`declare(parallel())`) | – (separate R processes only) | OpenMP over particles (`n_threads`) | TMB: OpenMP (`parallel_accumulator`); RTMB: – (separate R processes only) | – (separate R processes only) |
| **Compiler** | XLA; Fortran (experimental) | Fortran | C++ | C++ | C++ | C++ |
| **Intermediate representation** | traced into an `AnvlGraph`, lowered to StableHLO (MLIR), compiled by XLA via LLVM; quickr backend: transpiled to R with type declarations, then Fortran | transpiled: R to Fortran | transpiled: R to C++ | transpiled: model language to C++ (dust2) | TMB: C++ template compiled by a C++ compiler; both: AD tape (CppAD / TMBad) evaluated at runtime | transpiled: model language and R subset to C++ (Eigen, CppAD) |
| **AD** | rev (scalar functions), HO (nested rev) | – | fwd, rev (Jacobians) | rev (likelihood of deterministic models) | rev, HO (fwd internal only) | rev, HO (up to second order) |
| **Data types** | bool, i8–i64, u8–u64, f32, f64 | bool, i32, f64, c128 | bool, i32, f64 | f64, i32 | f64 (integers as data) | bool, i32, f64 |
| **Data-dependent control flow** | `nv_if()`, `nv_while()`, `nv_scan()` | ✓ | ✓ | `if` expressions only | TMB: `CppAD::CondExp*`, C++ `if`/`while` frozen at taping; RTMB: taped 0/1 masks (`TapeConfig(comparison = "tape")`) | ✓ in `nimbleFunction` run code |
| **Differentiable control flow** | `nv_if()` and loops unrolled during tracing; not `nv_while()`, `nv_scan()` | – | ✓ (fwd and rev) | (✓) adjoint through `if` and time steps | (✓) `CondExp` / masks and fixed loops; no `while` | (✓) needs `derivs(reset = TRUE)`, otherwise silently wrong |
| **Random numbers** | anywhere in the program; explicit functional state (Threefry); CPU and GPU | `runif()` only; R's generator | – | stochastic models; own generator, one stream per particle | TMB: in `SIMULATE` blocks, R's generator; RTMB: frozen in the tape | in compiled code; R's generator |
| **Shapes** | static, recompiled per shape | dynamic | dynamic | fixed at initialisation | dynamic | dynamic |
| **Conventions** | 1-based, column-major | 1-based, column-major | 1-based, column-major | 1-based | 0-based in C++; 1-based in R | 1-based, column-major |
| **New operations** | primitives with rules in R; C/C++ custom calls | – | typed inner functions in R; C++ via external pointers | C++ (dust2 systems) | atomic functions in C++ | `nimbleFunction`, external C/C++ |
| **Dependencies** | PJRT plugin (downloaded) | Fortran compiler | C++20 toolchain | C++ toolchain; installed from R-universe | C++ toolchain (TMB) | C++ toolchain |

Caption: Comparison of anvl with R packages for compiled or differentiable numerical computing that run on the CPU only. ✓: supported; –: not available; fwd: forward mode; rev: reverse mode; HO: higher-order derivatives.
