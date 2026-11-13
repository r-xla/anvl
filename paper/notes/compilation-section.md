# Draft: transpilation to C++/Fortran vs. compilation with an array compiler

## 1. Heading and placement

**Proposed heading:** `## Transpilers and array compilers {#compilers}`

**Recommended place:** as the last subsection of "Background: the OpenXLA compiler ecosystem" (`# Background ... {#background}`), i.e., after the paragraph on the PJRT plugin downloads and before `# Package design {#design}`.
The Background section currently has no subsection headings, so the existing text either stays as an unlabeled preamble or gets a first heading such as `## StableHLO and PJRT {#stablehlo-pjrt}`.

Reasoning:

- The section explains *what XLA does* with a StableHLO program, which the Background section introduces but describes in a single clause ("fuses operations to reduce memory traffic").
  Readers need StableHLO (static types, SSA, ~100 operations) already defined, so it has to come after that paragraph.
- Related work (Section `in-r`) lists the transpiling packages and Tables 1/2 already state "Compiler: C++ / Fortran / XLA".
  Putting the mechanism in Related work would make that section longer and more technical; Background is where the paper explains the machinery it relies on.
- The runtime section currently gives a one-sentence "plausible reason" for the CPU speedup over Stan.
  With this section in place, that sentence can point back to it: e.g., "A plausible reason is that XLA compiles the whole sampler, including the gradient, into a single optimized program (Section \@ref(compilers)), whereas Stan and \CRANpkg{TMB} ... evaluate the gradient with operator-overloading AD."
- The Conclusion's limitations paragraph (static shapes, recompilation) can then refer to the same section.

Alternative: a subsection at the end of "Related work" (after "Relation to JAX"). This is defensible if the authors want Background to stay a pure description of OpenXLA, but the trade-off discussion reads better once StableHLO has been introduced.

## 2. Section text (R Markdown, ~430 words)

```markdown
## Transpilers and array compilers {#compilers}

Most compiled numerical code in R reaches the machine via a general-purpose compiler.
Stan translates its modeling language to C++ that calls the Stan Math library [@ref-stan; @ref-stanmath], \CRANpkg{nimble} and \pkg{ast2ast} generate C++ from R-like code, \CRANpkg{odin} generates C, and \CRANpkg{quickr} translates a typed subset of R to Fortran [@ref-nimble; @ref-ast2ast; @ref-odin; @ref-quickr].
The C++ or Fortran compiler then optimizes what it sees: scalar loops, inlined functions, and calls into libraries such as Eigen and BLAS.
It does not see array-level semantics of the whole program, so it cannot, e.g., merge a matrix product, an elementwise transformation, and a reduction that live in different library calls into one pass over memory.
Derivatives are obtained at run time: Stan Math builds an expression graph by operator overloading during every evaluation of the log density and traverses it in reverse [@ref-stanmath], and \CRANpkg{TMB} records a CppAD tape once and replays it for every evaluation [@ref-tmb; @ref-cppad].
In both cases, the gradient computation is data that a runtime library interprets, not code that the compiler can optimize.

XLA takes a different route.
Its input is a whole program in an array IR whose values have static shapes and whose operations have no side effects (Section \@ref(background)).
On this representation, XLA performs common subexpression elimination, algebraic simplification, and constant folding, assigns memory layouts, and plans buffers for the whole program so that intermediate results can share memory [@ref-xla-architecture; @ref-xla-gpu].
Its most important optimization is fusion: chains of elementwise operations, broadcasts, and reductions are compiled into a single kernel, so that intermediate arrays are never written to memory, which on GPUs also saves kernel launches [@ref-xla-gpu].
The fused code is generated via LLVM for the CPU and the GPU, while operations such as matrix products are dispatched to vendor libraries such as cuBLAS [@ref-xla-architecture; @ref-xla-gpu].
As \pkg{anvl}, like JAX [@ref-frostig2018compiling], differentiates a program by transforming its graph before compilation, the gradient is part of the program that XLA optimizes, together with the forward pass and, for the sampler in Section \@ref(hmc), with the surrounding leapfrog and acceptance steps.
The same program runs on every device with a PJRT plugin, whereas a transpiler needs a separate GPU code path, such as Stan's OpenCL kernels for selected functions.

These advantages come at a price.
A program is compiled for one set of shapes and has to be recompiled when they change, output shapes cannot depend on data, and data-dependent control flow has to be expressed with dedicated primitives such as `nv_while()`.
Transpilers do not have these restrictions: generated C++ or Fortran handles arbitrary shapes and branches freely, and code that is dominated by scalar loops with data-dependent branching, such as many ODE right-hand sides or recursive likelihoods, maps naturally to it.
Array compilers pay off when a computation can be expressed as a sequence of array operations, which is the case for many statistical models and for the many-chain samplers that motivate \pkg{anvl}.
```

Notes on the text:

- If the authors prefer not to add the two XLA doc entries, `[@ref-openxla]` can replace both `ref-xla-architecture` and `ref-xla-gpu`, at the cost of a less specific citation.
- "Eigen" is mentioned without a citation; Stan Math's use of Eigen and expression templates is stated in the Stan Math paper abstract/body [@ref-stanmath]. An Eigen citation (Guennebaud, Jacob et al., 2010, eigen.tuxfamily.org) can be added if the editor asks.
- \CRANpkg{rxode2} is omitted from the transpiler list for brevity; it can be appended next to \CRANpkg{odin} (both "compile ... to C" in Related work).
- \CRANpkg{nimble} uses CppAD as well; the sentence about tapes could say "\CRANpkg{TMB} and \CRANpkg{nimble} record a CppAD tape" if desired (nimble's AD uses CppAD per the Related work paragraph; the record-once/replay detail is only checked for CppAD itself, see Section 4).

## 3. Proposed BibTeX entries

```bibtex
@Manual{ref-xla-architecture,
  title = {{XLA} Architecture},
  author = {{OpenXLA Team}},
  year = {2026},
  url = {https://openxla.org/xla/architecture},
}

@Manual{ref-xla-gpu,
  title = {{XLA:GPU} Architecture Overview},
  author = {{OpenXLA Team}},
  year = {2026},
  url = {https://openxla.org/xla/gpu_architecture},
}
```

Optional (peer-review status weak -- it is a University of Toronto course report on arXiv; only if a reviewer wants an analysis of fusion beyond the docs):

```bibtex
@article{ref-snider2023fusion,
  title = {Operator Fusion in {XLA}: Analysis and Evaluation},
  author = {Daniel Snider and Ruofan Liang},
  journal = {arXiv preprint arXiv:2301.13062},
  year = {2023},
}
```

Verified against the sources:

- XLA architecture page: target-independent passes incl. "common subexpression elimination, target-independent operation fusion, and buffer analysis"; backends "pattern-match certain operations ... to optimized library calls"; CPU and GPU backends "use LLVM for low-level IR, optimization, and code generation".
- XLA:GPU architecture page: layout assignment pass; "Fusion is XLA's single most important optimization"; buffer assignment using whole-graph shape information; "XLA:GPU uses fast-performance libraries from NVIDIA, such as cuBLAS, cuDNN, and NCCL"; direct LLVM IR emission and Triton for some fusions.
- Stan Math paper (arXiv:1509.07164), p. 4: "The expression graph is constructed in the ordinary evaluation order, with each subexpression being numbered and placed on a stack"; abstract mentions expression templates and custom memory management.
- Snider & Liang metadata checked on arXiv.

## 4. Claims not fully verified

1. **Algebraic simplification and constant folding** in XLA: these passes exist in the XLA source (`AlgebraicSimplifier`, `HloConstantFolding`), but neither of the two docs pages cited above names them explicitly. If a doc citation is required for these two terms, drop them or cite the source tree.
2. **TMB "records a tape once and replays it"**: this is how CppAD `ADFun` objects work (and CppAD's docs note that plain `if` comparisons are frozen into the tape and require retaping), and `MakeADFun()` builds the tape; I did not find a TMB page stating it in those words. The TMB JSS paper [@ref-tmb] should be checked for a quotable sentence. Also note that current TMB can use either CppAD or TMBad (`compile(framework = ...)`, default from `getOption("tmb.ad.framework")`); the paper's Related work says "via CppAD", which may be outdated -- RTMB is built on TMBad. Worth checking which framework the benchmark used.
3. **XLA on the CPU calling Eigen/oneDNN for matrix products**: I could not find a primary doc stating which library the CPU backend uses, so the draft only names cuBLAS (documented) and says "vendor libraries".
4. **"Fast compile for small models" as an advantage of transpilers**: not used in the draft, because the paper's own benchmarks contradict it (about 11 s for Stan and 35 s for TMB vs. under 1 s for tracing + lowering + XLA compilation). The draft instead names dynamic shapes and scalar, branch-heavy code as the transpilers' strengths.
5. **The C++ compiler cannot fuse across library calls**: stated generally; Eigen's expression templates do fuse elementwise expressions *within* a single Eigen expression, which is why the draft says "different library calls". If a reviewer pushes, this distinction is the defensible formulation.
6. **XLA "constant folding" / "no side effects"**: StableHLO does have side-effecting ops (e.g., infeed/outfeed, custom calls with side effects, RNG state ops are functional). The draft's "whose operations have no side effects" is a simplification; "functional" (the word the Background section already uses) is a safe alternative.
