# Related Work for r-xla/anvl: A Categorized Map of Relevant R Packages

anvl's closest R relatives are **torch** (GPU tensors plus autograd through LibTorch, with no Python needed), **RTMB/TMB** (AD written in plain R), **quickr** (a JIT/AOT compiler for R by an anvl co-author), and the **reticulate-based TensorFlow/Keras 3/greta stack**, which reaches XLA and JAX only through Python. As far as I could find, no other R package combines tracing of plain R functions, composable function transformations (reverse-mode AD), and whole-program XLA compilation through a native (non-Python) PJRT runtime. That combination is the novelty claim your Related Work section should defend.

## TL;DR
- **What anvl is (verified):** anvl calls itself "Accelerated array computing and code transformations for R" and says it "aspires to be for R what JAX is for Python." It turns R functions on `AnvlArray`s into XLA programs through `jit()`, and its only transformation so far is reverse-mode AD through `gradient()`. The same code runs on CPU or NVIDIA GPU (CUDA). It is written almost entirely in R and sits on two sibling r-xla packages: **stablehlo** (the IR layer) and **pjrt** (the runtime).
- **Must-discuss competitors:** torch for R (the mlverse project, which shares authors with anvl), tensorflow/keras3 (XLA via Python; keras3 can use a JAX backend), greta (TensorFlow-backed probabilistic programming), TMB/RTMB (C++ tape-based AD), nimble (R-to-C++ compilation with CppAD AD), and quickr (R-to-Fortran compilation, by anvl co-author Kalinowski).
- **Deprecated or archived (flag these in the paper):**
  - gpuR: removed from CRAN on 2025-11-28 "as email to the maintainer is undeliverable"
  - rray: archived from CRAN on 2020-02-11
  - ADtools: archived from CRAN
  - mxnet: per the Apache Attic, retired in September 2023, with the move to the Attic completed in February 2024
  - Renjin: unmaintained since 2021
  - FastR: per Oracle's GraalVM docs, "no longer under active development and is in maintenance mode"; last release 22.3.1
  - autodiffr and armacmp: GitHub-only prototypes

## Key Findings

### 1. What anvl actually is (Step 1)
- **Purpose and design.** The README describes three core ideas:
  - **Compilation:** "converts R functions into an optimized program via XLA – the same compiler that powers JAX and TensorFlow." The README also says the resulting programs "can be faster compared to implementing them in {torch}."
  - **Function transformation:** "Currently the only available transformation is reverse-mode automatic differentiation via `gradient()`."
  - **Hardware portability:** the same code runs on CPU or GPU.
- **Execution modes.** Functions run in *eager* mode (each operation executes immediately) or *jit* mode (the whole function is compiled into a single executable). Floats default to f32; f64 is available through `with_default_dtypes()`.
- **Platforms.** Linux x86_64/arm64 support CPU and CUDA. Windows x86_64 supports CPU, with CUDA only through WSL2. macOS supports CPU only.
- **Version, distribution and authors.**
  - Version 0.6.0 is distributed through r-universe (`r-xla.r-universe.dev`), not CRAN. The README carries a CRAN badge, but I found no evidence of a CRAN release.
  - Authors: Sebastian Fischer (maintainer), Daniel Falbel, Tomasz Kalinowski, Nikolai German, Louis Aslett.
  - Funding: supported by MaRDI.
  - Credits: the design is inspired by JAX, especially the "autodidax" tutorial. Compilation builds on OpenXLA, and most compiler binaries are downloaded from zml's pjrt-artifacts.
- **History.** The project was earlier named **anvil** ("Composable code transformation framework for R"). That earlier README listed an experimental **quickr-based Fortran backend (CPU only)** and "asynchronous allocation and execution." The current README no longer mentions the quickr backend, so check the current code before citing it.
- **r-xla ecosystem.**

| Package | Role |
|---|---|
| anvl | User-facing API: JIT compilation plus autodiff |
| stablehlo | IR layer: builds StableHLO programs from R. It is a partial implementation of the spec and also supports selected CHLO ops. |
| pjrt | Runtime: compiles StableHLO/HLO programs with XLA and runs them on CPU/CUDA (Metal is also mentioned). It is inspired by gopjrt and can run StableHLO exported from JAX. |
| pjrt.cuda | Bundled CUDA runtime libraries |
| xlamisc, tengen | Shared utilities; generics and data types for tensors |

  stablehlo and pjrt should be cited as part of the system, not as related work. pjrt is the only R interface to PJRT I found.

### 2. Which packages are closest competitors
Ranked by how closely they overlap with anvl's feature set (tracing, AD, compilation, accelerators, native R):

1. **torch (mlverse)** is the nearest native-R competitor. It has GPU tensors, tape-based reverse-mode autograd, and no Python dependency. Its JIT (`jit_trace`/TorchScript) is mainly for deployment, not for fusing whole programs. anvl's README benchmarks itself against torch, and the two projects share authors (Falbel, Kalinowski; Fischer is a torch contributor).
2. **tensorflow + keras3 (+ tfprobability, greta)** are the only established R route to XLA. They go through Python (reticulate), and keras3 can run on a JAX backend.
3. **RTMB / TMB** offer arbitrary-order AD of plain R code through C++ taping (TMBad). They run on CPU only and use no XLA or GPU.
4. **quickr** is a JIT/AOT compiler for type-annotated R that lowers to Fortran. It runs on CPU and has no AD.
5. **nimble** compiles a DSL to C++ and gets AD through CppAD. It runs on CPU.

## Details (by category)

### A. Deep learning / tensor frameworks in R
| Package | What it is | Relation to anvl | Python? | Repo / status | Citable reference |
|---|---|---|---|---|---|
| **torch** | R port of PyTorch's functionality, built directly on LibTorch: tensors, autograd, neural networks, GPU. | Eager tape-based autograd with no whole-program compilation. `jit_trace()` records operations into TorchScript; the docs warn it only records data-independent paths, and scripting from R is limited to compiling TorchScript source. anvl instead stages R functions to StableHLO/XLA. | No | CRAN; active. v0.17.0 was published 2026-04-11, maintained by Tomasz Kalinowski, and updated to LibTorch 2.8.0. | Falbel D, Luraschi J. *torch: Tensors and Neural Networks with 'GPU' Acceleration*, R package, torch.mlverse.org; PyTorch: Paszke et al. 2019, doi:10.48550/arXiv.1912.01703 |
| **tensorflow** | R interface to TensorFlow. | Gets XLA, AD and GPU/TPU support, but only through Python and reticulate. | Yes | CRAN v2.20.0; maintained | Allaire J, Tang Y et al.; doi:10.32614/CRAN.package.tensorflow |
| **keras3** (and legacy **keras**) | Rebuilt R interface to Keras 3 with multiple backends (`use_backend("jax")`, tensorflow, torch, numpy). For JAX/TensorFlow backends, `jit_compile="auto"` "enables XLA compilation." | The closest *existing* R access to JAX and XLA, but it is a high-level model API built on Python. | Yes | CRAN v1.5.1; active. Legacy keras is still on CRAN but superseded. | Kalinowski T (2024), "Introducing Keras 3 for R," Posit AI Blog |
| **tfprobability** | R interface to TensorFlow Probability. | Probabilistic layer on TensorFlow, so it depends on Python. | Yes | CRAN v0.15.2 (2025-08-21); maintenance only | doi:10.32614/CRAN.package.tfprobability |
| **mlr3torch** | Deep learning within mlr3, built on torch. | Shows the mlr3 team's investment in torch (Fischer is an mlr3torch author). Its preprint's related-work section is a useful template. | No | CRAN; active | arXiv:2604.18152 |
| **mxnet (R)** | R bindings for Apache MXNet. | Historical GPU deep-learning framework for R. | No | **Discontinued**: per the Apache Attic, MXNet "retired in September 2023 and the move to the Attic was completed in February 2024." It was never on CRAN. | Chen T et al. (2015), arXiv:1512.01274 |
| **cuda.ml** | R interface to RAPIDS cuML. | Classical ML on GPU with no AD or compilation. | No | CRAN v0.4.0 (2026-08-21), which breaks compatibility with 0.3.x | doi:10.32614/CRAN.package.cuda.ml |

### B. Automatic differentiation in R
| Package | Approach | Relation to anvl | Repo / status | Reference |
|---|---|---|---|---|
| **TMB** | User writes C++ templates; CppAD/TMBad provide AD and the Laplace approximation. | Mature tape-based AD on CPU for latent-variable models; requires C++. | CRAN; active | Kristensen et al. (2016), *JSS* 70(5), doi:10.18637/jss.v070.i05 |
| **RTMB** | Plain R code is overloaded with an `advector` class and taped by TMBad. It offers AD "to any order" for a rich subset of R, including dense/sparse linear algebra, FFT and distributions, plus a `MakeTape` low-level interface. | The strongest "AD of plain R" competitor. It uses operator-overloading taping in C++ on CPU, not tracing to an ML compiler, and has no GPU support. | CRAN v2.0 (2026-09-17); active | Kristensen K (2026), *Ecology and Evolution*, doi:10.1002/ece3.74572 |
| **RTMBdist** | AD-compatible distributions for RTMB. | Shows how big the RTMB ecosystem has grown. | CRAN | CRAN page |
| **nimble** (+ **nimbleHMC**, **nimbleQuad**) | A BUGS-like DSL plus nimbleFunctions compiled to C++. AD arrived in v1.0.0 (May 2023) through CppAD and enables HMC and Laplace approximation. | Ahead-of-time R-to-C++ compilation with AD; CPU only. | nimble CRAN v1.4.3 (2026-08-22); nimbleHMC v0.2.5; nimbleQuad v1.4.0 (2026-01-14) | de Valpine et al. (2017), *JCGS* 26(2):403–413, doi:10.1080/10618600.2016.1172487; nimbleHMC: Turek et al. (2024), *JOSS*, doi:10.21105/joss.06745 |
| **nCompiler** | The nimble team's next-generation R-to-C++ code generator, with CppAD AD. | Same family as nimble; experimental. | GitHub only | github.com/nimble-dev/nCompiler |
| **rstan / cmdstanr** | Stan language compiled to C++; the Stan Math library provides reverse-mode AD. | A separate DSL rather than R code; CPU (limited OpenCL). | rstan on CRAN; cmdstanr on r-universe (1.0 RC Oct 2026) | Carpenter et al. (2017), *JSS* 76(1), doi:10.18637/jss.v076.i01 |
| **madness** | Forward-mode AD of multivariate (matrix) operations. | Pure R, forward mode. | CRAN v0.2.9 (2026-07-14) | doi:10.32614/CRAN.package.madness |
| **salad** | Forward-mode AD with an S4 `dual` class handling vectors and matrices. | Pure R, forward mode, no compilation. | CRAN v1.2 | CRAN (Perdry) |
| **dual** | Dual-number forward-mode AD. | Pure R, first derivatives only. | CRAN v0.0.6 | CRAN (Sartore) |
| **nabla** | Dual-number AD ("Exact Derivatives via Automatic Differentiation"). | Pure R, forward mode. | CRAN | CRAN |
| **ADtools** | Forward-mode AD using matrix-calculus notation (Magnus & Neudecker). | Pure R. | **Archived from CRAN** (last 0.5.4); GitHub | Kwok, Zhu, Jacobi; related paper in *Stats* (MDPI) 8(2):40 |
| **autodiffr** | Wraps Julia's ForwardDiff.jl/ReverseDiff.jl through JuliaCall. | AD of R functions delegated to Julia; requires Julia. | **GitHub only, unmaintained** (Julia 0.6–1.0 era, 2018) | Li C, Nash J, Borchers HW; non-contradiction.github.io/autodiffr |
| **Deriv**, base `deriv`/`D` | Symbolic differentiation of R expressions. | A contrast class: symbolic, not algorithmic. | Deriv CRAN v4.3.6 (2026-10-21) | doi:10.32614/CRAN.package.Deriv |
| **numDeriv** | Finite differences. | Contrast baseline (numerical, not AD). | CRAN | Gilbert & Varadhan, CRAN |
| **dfdr** | Derivatives of R functions; used by ast2ast for Jacobians. | Small symbolic/AD helper. | CRAN | CRAN (Krämer) |

### C. JIT compilation / compilers for R code
| Project | What it is | Relation to anvl | Status | Reference |
|---|---|---|---|---|
| **quickr** | "Compile R functions annotated with type and shape declarations"; supports JIT and AOT; "lowering R code to Fortran." | Closest compile-R-code analogue: by anvl co-author Kalinowski (Posit), and formerly an experimental anvl backend. It needs declared types and shapes, has no AD, and runs on CPU only. | CRAN v0.3.0 (May 2026); active | Kalinowski T, quickr, CRAN; doi:10.32614/CRAN.package.quickr |
| **compiler** (base R) | Tierney's bytecode compiler. Per the R 3.4.0 NEWS (The R Journal), "The JIT ('Just In Time') byte-code compiler is now enabled by default at its level 3." | The baseline R "JIT"; no array fusion and no accelerators. | Ships with R | Tierney L, "A Byte Code Compiler for R" (technical note) |
| **Ř (rir)** | Research JIT for GNU R with speculative optimization, contextual dispatch and LLVM backend. | General-purpose language-level JIT, unlike anvl's domain-specific array staging. | Research prototype (GitHub) | Flückiger et al. (2020), "Contextual Dispatch for Function Specialization," OOPSLA, doi:10.1145/3428288; Flückiger PhD thesis (2022) |
| **FastR** | GraalVM/Truffle implementation of R. | Alternative-VM approach. | **Maintenance mode**: per Oracle's GraalVM R Runtime docs, "FastR is no longer under active development and is in maintenance mode." The last release was 22.3.1; it is now available only as source. | Oracle GraalVM docs; Kalibera et al., VEE 2014 |
| **Renjin** | JVM-based R interpreter/compiler. | Alternative-VM approach. | **Discontinued** ("developed between 2010 and 2021") | renjin.org |
| **Rcpp / RcppArmadillo / RcppEigen** | Hand-written C++ extensions. | The status-quo "make R fast" route that anvl aims to make unnecessary. | CRAN; active | Eddelbuettel & François (2011), *JSS* 40(8); Eddelbuettel & Sanderson (2014), *CSDA*; Bates & Eddelbuettel (2013), *JSS* 52(5) |
| **armacmp** | DSL compiling linear-algebra R code to Armadillo C++, with ensmallen optimization. | Transpiler to C++ for linear algebra only. | GitHub only, "experimental," inactive | dirkschumacher.github.io/armacmp |
| **ast2ast** | Translates "a tiny subset of R to C++"; can produce Jacobians. | R-to-C++ transpiler aimed at ODE and optimization use. | CRAN (updated 2026-10-05) | CRAN (Krämer) |
| **odin** (→ odin2/dust2) | DSL for ODEs compiled to C/C++. | Domain-specific compilation. | odin CRAN v1.2.7; odin2 on mrc-ide r-universe | doi:10.32614/CRAN.package.odin |

### D. GPU and accelerator computing in R
| Package | Backend | Relation to anvl | Status | Reference |
|---|---|---|---|---|
| **GPUmatrix** | Mimics Matrix-package semantics on top of torch or tensorflow. | Drop-in GPU linear algebra with no compilation or AD. | CRAN v1.0.3 (2026-03-26) | Lobato-Fernández et al., *The R Journal* RJ-2024-041 |
| **gpuR** | OpenCL through ViennaCL (vendor-neutral). | Eager GPU linear algebra. | **Removed from CRAN** on 2025-11-28 "as email to the maintainer is undeliverable" (also archived in 2019 and 2020) | doi:10.32614/CRAN.package.gpuR |
| **gputools** | CUDA statistical routines. | Fixed-function kernels. | Archived 2017; reportedly back on CRAN v2.1.0 (2026-10-24) | doi:10.32614/CRAN.package.gputools |
| **OpenCL** | Low-level OpenCL interface (Urbanek). | User writes kernels by hand. | CRAN v0.2-11 (2026-01-19) | doi:10.32614/CRAN.package.OpenCL |
| torch, tensorflow, keras3, cuda.ml | See sections A–B. | | | |

### E. Array/tensor programming in R
| Package | Notes | Status |
|---|---|---|
| **rray** | Broadcasting arrays (originally built on xtensor). | **Archived from CRAN 2020-02-11**; GitHub rewrite only |
| **tensorA** | Einstein-notation tensor arithmetic. | CRAN v0.36.2.1; minimally maintained |
| **tengen** (r-xla) | Generics and data types for tensors/arrays, used inside r-xla. | GitHub |

### F. Interfaces to JAX / XLA / ML compilers
- **reticulate** (CRAN v1.47.0; maintained by Kalinowski) is the generic bridge. Its docs show `py_require("jax")` followed by `import("jax")`, so calling JAX from R is possible today, but it requires Python and JAX arrays, not R functions. This is the main "why not just use reticulate+JAX?" point anvl must answer.
- **keras3** with the JAX backend uses XLA through Python; see section A.
- **pjrt / stablehlo** (r-xla) are, as far as I found, the only native R bindings to PJRT/StableHLO. They are anvl's own infrastructure.
- I found no maintained native R interface to ONNX Runtime, TVM or MLIR-based compilers. Treat this as a gap I could not verify exhaustively, not a confirmed absence.

### G. Probabilistic programming systems using AD or compilation backends
**greta** builds models "interactively in R code," then "performs statistical inference using TensorFlow." It needs Python, TensorFlow and TensorFlow Probability, and its CRAN manual notes it "does not support TF 2.16+ (Keras 3)." Cite Golding (2019), *JOSS* 4(40):1601, doi:10.21105/joss.01601. It is on CRAN. Extensions such as greta.gp and causact (JOSS, doi:10.21105/joss.04415) build on it. Also relevant: nimble, rstan/cmdstanr, RTMB and tfprobability (see above).

### H. Non-R context systems (keep brief)
- **JAX:** the direct inspiration, including the autodidax design.
- **XLA/OpenXLA, StableHLO, PJRT:** the compiler stack anvl uses.
- **TensorFlow and PyTorch:** torch.compile versus TorchScript is worth contrasting with torch for R's tracing.
- **Julia:**
  - Zygote: source-to-source AD.
  - Enzyme: LLVM-level AD.
  - Reactant.jl (EnzymeAD group, Wicker et al. 2024): the closest analogue in another language. It "JIT-compiles Julia array programs to MLIR and then to XLA" for CPU, CUDA GPU and TPU backends. Its README says it "presently operates as a tracing system" and uses "EnzymeMLIR for automatic differentiation."
- **Go:** gopjrt/gomlx, which inspired the design of the pjrt package.

## Recommendations
1. **Frame the novelty along three axes:**
   - Native R with no Python, unlike tensorflow, keras3, greta and reticulate+JAX.
   - Whole-function compilation through an ML compiler that fuses operations, unlike torch's eager autograd and tracing designed for deployment.
   - Composable transformations with GPU support, unlike RTMB/TMB, nimble and quickr, which are CPU-only compilers or AD systems.
2. **Give torch, RTMB, quickr and tensorflow/keras3/greta their own paragraphs.** List the forward-mode/dual-number packages, the GPU wrappers and the alternative VMs compactly.
3. **Mark statuses explicitly**, especially gpuR, rray, ADtools, mxnet, Renjin, FastR, autodiffr and armacmp. Use the CRAN DOIs (10.32614/CRAN.package.<name>) for packages that have no paper.
4. **Disclose author overlap** with torch, keras3/reticulate and quickr. Readers will notice it.

## Caveats
- Several statuses come from CRAN pages dated 2026 and a secondary research pass. Re-check the following against CRAN immediately before submission:
  - gputools' reported return in Oct 2026
  - cuda.ml 0.4.0
  - the cmdstanr RC
- Some references were not re-verified in this session: the base R compiler technical note, the FastR VEE paper, the MXNet arXiv ID, the Rcpp-family journal citations, numDeriv, and the CRAN status of TMB and dfdr. The CRAN-DOI format is standard but was not opened for every package.
- anvl's CRAN badge suggests a planned CRAN release. Verify it before stating that anvl is "on CRAN."
