# Does XLA optimize array programs the way the paper claims, and what do transpilers get instead?
#
# The related-work section claims that most compilers for R "translate R or their own language into
# C, C++, or Fortran and leave the optimization to a general-purpose compiler, which only sees
# loops, scalar operations, and library calls", whereas a compiler toolchain for array programs,
# such as StableHLO and XLA, "simplifies algebraic expressions over arrays, chooses memory
# layouts, and fuses chains of elementwise operations and reductions into single kernels that
# avoid materializing intermediate arrays".
#
# This script compiles the same small R functions three ways on the CPU and reads what the
# compilers produce:
#
#   anvl -> StableHLO -> XLA      XLA's dump: the HLO module after its optimization passes, the
#                                 buffers the program allocates, and the LLVM IR of the kernels
#                                 XLA generates itself (`*.ir-no-opt.ll`)
#   quickr -> Fortran -> gfortran the optimized GIMPLE, GCC's intermediate representation after
#                                 its optimization passes (`-fdump-tree-optimized`)
#   ast2ast -> C++ -> clang       the LLVM IR after clang's optimization passes (`-emit-llvm`)
#
# gfortran and clang run with the flags R uses for packages (`R CMD config`), except that clang
# runs without `-g` and with C++20, which ast2ast requires.
#
# For each program, the script prints what each compiler made of it and then asks the same
# question of all three compiled programs, e.g. whether exp, *, and + run in one loop, or whether
# x / 4 became a multiplication. It ends with the paper's claims about XLA, checked on the XLA
# programs, and a table of the answers. The evidence differs by compiler:
# * XLA: the kernels of the optimized HLO and the intermediate arrays of its buffer assignment.
# * gfortran: the loops over arrays of doubles in the GIMPLE, found from its control-flow graph,
#   and the heap allocations whose size depends on the input.
# * clang: the floating-point operations of each function the compiled program runs. Loops are
#   not counted here, because etr, ast2ast's array library, has code paths for vectors and
#   matrices of which only one runs, and evaluates expressions through element getters. The
#   conversions between R and C++ objects (`CheckSEXP`, `Cast`) are left out, as XLA's programs
#   do not include moving data in and out of the program either.
# The questions encode what the output showed when the script was written; their comments say
# where the answer comes from (e.g., quickr's length checks or etr's temporary buffers).
#
# Notes on reading the output:
# * Some simplifications already happen when XLA imports the StableHLO program, so the baseline
#   for XLA is the StableHLO that anvl emits, not XLA's own "before optimizations" dump.
# * A kernel that XLA hands to a library, such as a matrix product or a YNNPACK fusion, has no
#   LLVM IR in the dump; for these, the optimized HLO and the buffer assignment are the evidence.
# * Every program XLA compiles has one more parameter than the function: anvl allocates the
#   output buffer and donates it to the program (`input_output_alias` in the module header).
# * quickr checks at run time that the operands of elementwise operations have equal lengths, by
#   evaluating `size()` of the operand expressions; gfortran materializes these operands in
#   temporary arrays, which shows up as extra passes and allocations.
#
# Run with `Rscript verification/codegen.R [dump_dir]` from the paper directory. Without
# `dump_dir`, all intermediate files go to a temporary directory that is removed at the end.
# Uses the anvl source via pkgload::load_all(), and needs quickr, ast2ast, and the Fortran and C++
# compilers R is configured with. Checked on macOS (arm64) with the PJRT CPU plugin v18.0.0,
# gfortran 14.2, and Apple clang 17.

args <- commandArgs(trailingOnly = TRUE)
dump_dir <- if (length(args)) args[[1L]] else tempfile("codegen-")
unlink(dump_dir, recursive = TRUE)
dir.create(file.path(dump_dir, "xla"), recursive = TRUE)
dir.create(file.path(dump_dir, "fortran"))
dir.create(file.path(dump_dir, "cpp"))
dump_dir <- normalizePath(dump_dir)
xla_dir <- file.path(dump_dir, "xla")

# XLA reads the flags when the PJRT plugin is loaded, so they have to be set before anvl runs
# anything.
Sys.setenv(XLA_FLAGS = paste0("--xla_dump_to=", xla_dir, " --xla_dump_hlo_as_text"))

here <- local({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) dirname(normalizePath(sub("^--file=", "", f))) else getwd()
})
suppressMessages(pkgload::load_all(file.path(here, "..", ".."), quiet = TRUE))

r_config <- function(var) {
  strsplit(system2(file.path(R.home("bin"), "R"), c("CMD", "config", var), stdout = TRUE), " +")[[1L]]
}
fc <- r_config("FC")
fflags <- r_config("FFLAGS")
cxx <- grep("^-std=", r_config("CXX"), value = TRUE, invert = TRUE)
cxxflags <- setdiff(r_config("CXXFLAGS"), "-g")
cxx_includes <- paste0("-I", c(
  R.home("include"),
  system.file("include", package = "Rcpp"),
  system.file("include", package = "ast2ast")
))

# Runs a compiler in `dir` and returns its messages.
run_compiler <- function(dir, cmd, args) {
  old <- setwd(dir)
  on.exit(setwd(old))
  out <- suppressWarnings(system2(cmd[[1L]], c(cmd[-1L], args), stdout = TRUE, stderr = TRUE))
  if (!is.null(attr(out, "status"))) stop(paste(c(cmd, args, out), collapse = "\n"))
  out
}

# ------------------------------------------------------------------------------------------------
# anvl -> XLA

modules <- function() {
  unique(sub("^(module_[0-9]+)\\..*$", "\\1", list.files(xla_dir, pattern = "^module_")))
}

dump_files <- function(module, pattern) {
  list.files(xla_dir, pattern = paste0("^", module, "\\.", pattern), full.names = TRUE)
}

# Traces `f` at the types of the arrays in `args` to get the StableHLO that anvl emits, then
# compiles and runs `jit(f)` on them and records which dumped module XLA compiled for it.
compile_xla <- function(f, args) {
  avals <- lapply(args, function(a) nv_aval(dtype(a), shape = shape(a)))
  names(avals) <- names(formals(f))[seq_along(args)]
  stablehlo_text <- stablehlo::repr(stablehlo(trace_fn(f, avals))[[1L]])

  before <- modules()
  do.call(jit(f), args)
  module <- setdiff(modules(), before)
  if (length(module) != 1L) stop("expected one new module, found ", length(module))

  list(module = module, stablehlo = strsplit(stablehlo_text, "\n", fixed = TRUE)[[1L]])
}

# The operations of anvl's StableHLO program, e.g. "transpose" or "multiply".
stablehlo_ops <- function(p) {
  m <- regmatches(p$stablehlo, regexpr("stablehlo\\.[a-z_]+", p$stablehlo))
  sub("stablehlo.", "", m, fixed = TRUE)
}

# XLA's optimized HLO without source locations, split into computations: a named list of
# character vectors, one per computation, with the entry computation named "ENTRY".
hlo <- function(p) {
  lines <- readLines(dump_files(p$module, ".*cpu_after_optimizations\\.txt$"))
  lines <- gsub(", metadata=\\{[^}]*\\}", "", lines)
  starts <- grep("^(ENTRY )?%\\S+ \\(.*\\{$", lines)
  comps <- lapply(starts, function(s) lines[s:(s + which(lines[(s + 1L):length(lines)] == "}")[1L])])
  names(comps) <- ifelse(startsWith(lines[starts], "ENTRY"), "ENTRY", sub("^%(\\S+) .*$", "\\1", lines[starts]))
  comps
}

# The opcodes of the instructions of HLO computations, e.g. "exponential" or "fusion".
opcodes <- function(lines) {
  instr <- grep(" = ", lines, value = TRUE)
  sub("^.*? = \\S+ ([a-z][a-z0-9-]*)\\(.*$", "\\1", instr, perl = TRUE)
}
hlo_ops <- function(p) opcodes(unlist(hlo(p), use.names = FALSE))

# The number of kernels the program launches: the instructions of the entry computation that
# compute something.
n_kernels <- function(p) {
  sum(!opcodes(hlo(p)$ENTRY) %in% c("parameter", "constant", "bitcast", "tuple", "get-tuple-element"))
}

# Sizes (in bytes) of the intermediate arrays the program stores. Parameters, constants, and
# scalars that live in registers (thread-local) are not intermediate arrays; one allocation can
# hold several intermediate arrays.
intermediates <- function(p) {
  lines <- readLines(dump_files(p$module, ".*after_optimizations-buffer-assignment\\.txt$"))
  alloc <- cumsum(grepl("^allocation ", lines))
  temp_alloc <- unique(alloc[grepl("^allocation .*(preallocated-temp|, temp)", lines)])
  values <- grepl("^ value: ", lines) & alloc %in% temp_alloc
  as.numeric(sub("^.*\\(size=([0-9]+),.*$", "\\1", lines[values]))
}

# The unoptimized LLVM IR of the kernels that XLA generated itself, one character vector each.
xla_llvm_kernels <- function(p) {
  files <- dump_files(p$module, ".*\\.ir-no-opt\\.ll$")
  names(files) <- sub("^.*?\\.main\\.(__compute_module_)?(.*?)(_kernel_module)?\\.ir-no-opt\\.ll$", "\\2", basename(files), perl = TRUE)
  lapply(files, readLines)
}

# The floating-point loads, stores, and arithmetic of LLVM IR, in order.
llvm_fp <- function(ir) {
  ir <- grep("= load (double|<[0-9]+ x double>)|= (tail )?call (fast )?double|= f(mul|add|sub|div) |^\\s*store (double|<[0-9]+ x double>)", ir, value = TRUE)
  trimws(sub(", (align|!).*$", "", ir))
}

# Sizes (in number of doubles) of the arrays a kernel allocates on its stack.
llvm_stack_arrays <- function(ir) {
  m <- regmatches(ir, regexpr("alloca double, i64 [0-9]+", ir))
  as.integer(sub("^.* ", "", m))
}

# ------------------------------------------------------------------------------------------------
# quickr -> Fortran -> gfortran

# quickr needs the types of the arguments declared in the function body.
with_declare <- function(f, declare) {
  b <- as.list(body(f))
  body(f) <- as.call(c(b[1L], list(declare), b[-1L]))
  f
}

fortran <- function(name, f, declare) {
  dir <- file.path(dump_dir, "fortran")
  name <- paste0("quickr_", name) # avoid clashes with Fortran intrinsics such as sum
  src <- as.character(quickr:::new_fortran_subroutine(name, with_declare(f, declare)))
  src <- strsplit(paste(src, collapse = "\n"), "\n", fixed = TRUE)[[1L]]
  writeLines(src, file.path(dir, paste0(name, ".f90")))
  messages <- run_compiler(dir, fc, c(fflags, "-c", "-fdump-tree-optimized", paste0(name, ".f90"), "-o", paste0(name, ".o")))
  gimple <- readLines(list.files(dir, pattern = paste0("^", name, "\\.f90\\.[0-9]+t\\.optimized$"), full.names = TRUE))
  starts <- grep("^;; Function ", gimple)
  s <- grep(paste0("^;; Function ", name, " "), gimple)
  e <- c(starts[starts > s], length(gimple) + 1L)[[1L]] - 1L
  gimple <- gimple[s:e]

  # The statements of the computation, without quickr's checks and error handling.
  stmts <- trimws(grep("^\\s*(out|y|z) = |call dgemm|allocate\\(", src, value = TRUE))
  list(source = src, statements = stmts, gimple = gimple)
}

# The outermost loops of a control-flow graph, given as a list of basic blocks (character vectors
# of their lines, the entry block first) and their successors (indices into the list). A depth-
# first search from the entry finds the back edges; each back edge from a latch to a header
# defines a natural loop, the blocks that reach the latch without passing through the header.
# Loops nested in another loop are dropped, so each returned loop is one pass over the data; it
# is returned as the lines of its blocks, in their original order.
outer_loops <- function(blocks, succ) {
  n <- length(blocks)
  state <- integer(n) # 0: not visited, 1: on the search stack, 2: done
  back <- list()
  visit <- function(v) {
    state[v] <<- 1L
    for (w in succ[[v]]) {
      if (state[w] == 0L) visit(w) else if (state[w] == 1L) back[[length(back) + 1L]] <<- c(v, w)
    }
    state[v] <<- 2L
  }
  visit(1L)
  pred <- lapply(seq_len(n), function(v) which(vapply(succ, function(s) v %in% s, logical(1L))))
  loops <- list()
  for (e in back) {
    latch <- e[[1L]]
    header <- e[[2L]]
    body <- unique(c(header, latch))
    todo <- setdiff(latch, header)
    while (length(todo)) {
      new <- setdiff(pred[[todo[[1L]]]], body)
      body <- c(body, new)
      todo <- c(todo[-1L], new)
    }
    key <- as.character(header)
    loops[[key]] <- union(loops[[key]], body)
  }
  nested <- vapply(loops, function(l) any(vapply(loops, function(m) length(m) > length(l) && all(l %in% m), logical(1L))), logical(1L))
  lapply(loops[!nested], function(l) unlist(blocks[sort(l)], use.names = FALSE))
}

# The basic blocks of a GIMPLE function and their successors: the targets of its `goto`s, or the
# next block if it does not end in a `goto` or `return`.
gimple_loops <- function(fn) {
  starts <- grep("^\\s*<bb [0-9]+>", fn)
  ids <- sub("^\\s*<bb ([0-9]+)>.*$", "\\1", fn[starts])
  ends <- c(starts[-1L] - 1L, length(fn))
  blocks <- Map(function(s, e) fn[s:e], starts, ends)
  succ <- lapply(seq_along(blocks), function(i) {
    b <- blocks[[i]]
    targets <- sub("^.*goto <bb ([0-9]+)>.*$", "\\1", grep("goto <bb [0-9]+>", b, value = TRUE))
    last <- tail(trimws(b[nzchar(trimws(b)) & trimws(b) != "}"]), 1L)
    s <- match(targets, ids)
    if (!length(last) || !grepl("^(goto|return)", last)) s <- c(s, if (i < length(blocks)) i + 1L)
    unique(s[!is.na(s)])
  })
  outer_loops(blocks, succ)
}

# Loops over arrays of doubles: loops that load a double from memory.
fortran_data_loops <- function(p) {
  Filter(function(l) any(grepl("= MEM\\[\\(real\\(kind=8\\) \\*\\)", l)), gimple_loops(p$gimple))
}

# The floating-point loads, stores, and arithmetic of GIMPLE, in order.
gimple_fp <- function(lines) {
  trimws(grep("__builtin_exp|= _[0-9]+ [-+*/] |= MEM\\[\\(real\\(kind=8\\)|MEM\\[\\(real\\(kind=8\\) \\*\\).*\\] = ", lines, value = TRUE))
}

# Heap allocations whose size is not a constant, i.e., depends on the input.
fortran_heap_arrays <- function(p) {
  sum(grepl("__builtin_malloc \\([^0-9]", p$gimple))
}

# Whether one of the loops over the data contains all the given patterns, and how many loops do.
fortran_loops_with <- function(p, patterns) {
  sum(vapply(fortran_data_loops(p), function(l) all(vapply(patterns, function(pt) any(grepl(pt, l)), logical(1L))), logical(1L)))
}

# GIMPLE patterns for the operations in the programs below.
g_exp <- "__builtin_exp"
g_times2 <- "\\* 2\\.0e\\+0"
g_plus1 <- "\\+ 1\\.0e\\+0"
g_accumulate <- "(val|sum)[.0-9_]* = .*\\+ (val|sum)[.0-9_]*;"
g_store_out <- "MEM\\[\\(real\\(kind=8\\) \\*\\)out_"

# ------------------------------------------------------------------------------------------------
# ast2ast -> C++ -> clang

# ast2ast needs the result returned with `return()`.
with_return <- function(f) {
  b <- as.list(body(f))
  b[[length(b)]] <- call("return", b[[length(b)]])
  body(f) <- as.call(b)
  f
}

cpp <- function(name, f) {
  dir <- file.path(dump_dir, "cpp")
  src <- ast2ast::translate(with_return(f), getsource = TRUE)
  src <- strsplit(paste(src, collapse = "\n"), "\n", fixed = TRUE)[[1L]]
  writeLines(src, file.path(dir, paste0(name, ".cpp")))
  run_compiler(dir, cxx, c("-std=gnu++20", cxxflags, cxx_includes, "-S", "-emit-llvm", paste0(name, ".cpp"), "-o", paste0(name, ".ll")))
  ir <- readLines(file.path(dir, paste0(name, ".ll")))

  # Split the module into its function definitions.
  starts <- grep("^define ", ir)
  fns <- lapply(starts, function(s) ir[s:(s + which(ir[(s + 1L):length(ir)] == "}")[1L])])
  names(fns) <- sub("^define [^@]*@([^(]+)\\(.*$", "\\1", ir[starts])

  # The exported function and every function defined in the module that it calls, directly or
  # indirectly.
  entry <- names(fns)[grepl("SEXPREC", names(fns)) & !grepl("linkonce|internal", ir[starts])]
  reached <- entry
  repeat {
    callees <- unlist(lapply(fns[reached], function(fn) {
      m <- regmatches(fn, gregexpr("(call|invoke) [^@]*@([A-Za-z0-9_.$]+)", fn))
      sub("^.*@", "", unlist(m))
    }))
    new <- setdiff(intersect(callees, names(fns)), reached)
    if (!length(new)) break
    reached <- c(reached, new)
  }

  stmts <- trimws(grep("^\\s*(out|y|z) = ", src, value = TRUE))
  list(source = src, statements = stmts, functions = fns[reached])
}

# A readable name for a C++ function: demangled, without template arguments and parameters.
# (`-n` keeps c++filt on macOS from stripping the leading underscore of the mangled name.)
short_name <- function(mangled) {
  x <- system2("c++filt", c("-n", mangled), stdout = TRUE)
  repeat {
    y <- gsub("<[^<>]*>", "", x)
    if (identical(y, x)) break
    x <- y
  }
  sub("\\(.*$", "", x)
}

# The floating-point code of the functions the compiled program runs, as a list of (name,
# floating-point operations), one per function that has any. Counting loops does not work for
# etr: its functions contain code paths for vectors and matrices of which only one runs, and the
# loop that evaluates an expression calls an element getter that does the arithmetic. Left out
# are the conversion between R and C++ objects and etr's buffer management, which copies the old
# contents when a buffer grows (on the first assignment, there are none).
cpp_functions <- function(p) {
  out <- list()
  for (nm in names(p$functions)) {
    fname <- short_name(nm)
    if (grepl("CheckSEXP|Cast|get_dims_from_sexp|initSEXP|realloc", fname)) next
    ops <- llvm_fp(p$functions[[nm]])
    # etr grows a buffer's capacity by a factor of 1.15; this is not arithmetic on the data
    ops <- grep("fmul double %[0-9]+, 1\\.150000e\\+00", ops, value = TRUE, invert = TRUE)
    if (length(ops)) out[[length(out) + 1L]] <- list(name = fname, ops = ops)
  }
  out
}

# Whether one of the functions contains each of the operations `names(ops)` at least `ops` times.
cpp_has <- function(fns, ops) {
  any(vapply(fns, function(f) all(vapply(names(ops), function(op) sum(grepl(op, f$ops, fixed = TRUE)) >= ops[[op]], logical(1L))), logical(1L)))
}
cpp_calls <- function(fns, name) any(vapply(fns, function(f) grepl(name, f$name, fixed = TRUE), logical(1L)))

# ------------------------------------------------------------------------------------------------
# Reporting

show <- function(title, lines, indent = "    ") {
  cat("\n  ", title, ":\n", sep = "")
  cat(paste0(indent, if (length(lines)) lines else "(none)"), sep = "\n")
}

claims <- list()
answers <- list()

# A question asked of all three compiled programs, answered by the functions `xla` (of the XLA
# program), `gfortran` (of quickr's compiled Fortran), and `clang` (of ast2ast's compiled C++).
ask <- function(xla, gfortran, clang) list(xla = xla, gfortran = gfortran, clang = clang)

# Compiles `f` with all three tools and prints what they produce. `declare` gives quickr the
# argument types. `claims_xla` checks the paper's claims about XLA on the compiled program.
compare <- function(name, f, args, declare, claims_xla, questions) {
  stmts <- as.list(body(f))[-1L]
  code <- paste(vapply(stmts[-length(stmts)], deparse, character(1L), width.cutoff = 500L), collapse = "; ")
  cat("\n", strrep("=", 100), "\n", name, ": ", code, "\n", sep = "")

  # anvl -> XLA
  p <- compile_xla(f, args)
  cat("\n-- anvl -> StableHLO -> XLA --\n")
  show("StableHLO emitted by anvl", p$stablehlo)
  after <- hlo(p)
  for (nm in names(after)) show(paste0("HLO after XLA's optimizations (", nm, ")"), after[[nm]])
  temps <- intermediates(p)
  show("intermediate arrays", if (length(temps)) paste(temps, "bytes"))
  kernels <- xla_llvm_kernels(p)
  for (k in names(kernels)) show(paste0("LLVM IR of kernel ", k, " (floating-point operations)"), llvm_fp(kernels[[k]]))
  ok <- vapply(claims_xla(p), isTRUE, logical(1L))
  cat("\n  claims:\n")
  cat(paste0("    [", ifelse(ok, "x", " "), "] ", names(ok)), sep = "\n")
  claims[[length(claims) + 1L]] <<- data.frame(case = name, code = code, claim = names(ok), holds = ok)

  # quickr -> Fortran -> gfortran
  q <- fortran(name, f, declare)
  q_loops <- fortran_data_loops(q)
  cat("\n-- quickr -> Fortran -> gfortran --\n")
  show("Fortran statements of the computation", q$statements)
  for (i in seq_along(q_loops)) show(paste0("GIMPLE loop ", i, " over the data (floating-point operations)"), gimple_fp(q_loops[[i]]))
  show("heap allocations that depend on the input size", fortran_heap_arrays(q))
  library_calls <- unique(unlist(regmatches(q$gimple, gregexpr("\\b(dgemm_?|_gfortran_[a-z0-9_]+|__builtin_mem(cpy|move))\\b", q$gimple))))
  show("calls into libraries", library_calls)

  # ast2ast -> C++ -> clang
  a <- cpp(name, f)
  a_fns <- cpp_functions(a)
  cat("\n-- ast2ast -> C++ -> clang --\n")
  show("C++ statements of the computation", a$statements)
  for (fn in a_fns) show(paste0("LLVM IR of ", fn$name, " (floating-point operations)"), fn$ops)

  for (qn in names(questions)) {
    answers[[length(answers) + 1L]] <<- data.frame(
      case = name, question = qn,
      xla = isTRUE(questions[[qn]]$xla(p)),
      gfortran = isTRUE(questions[[qn]]$gfortran(q)),
      clang = isTRUE(questions[[qn]]$clang(a_fns))
    )
  }
}

# ------------------------------------------------------------------------------------------------
# Inputs

n_small <- 1000L
n_large <- 1000000L
x_small <- nv_array(runif(n_small), dtype = "f64")
x_large <- nv_array(runif(n_large), dtype = "f64")
A <- nv_array(matrix(runif(200 * 300), 200, 300), dtype = "f64")
B <- nv_array(matrix(runif(200 * 100), 200, 100), dtype = "f64")
C <- nv_array(matrix(runif(300 * 200), 300, 200), dtype = "f64")

vector_type <- quote(declare(type(x = double(n))))

no_intermediates <- function(p) length(intermediates(p)) == 0L
fused_reduction <- function(p) n_kernels(p) == 1L && "reduce" %in% hlo_ops(p)
is_copy <- function(p) identical(setdiff(hlo_ops(p), "parameter"), "copy")

# The answers to "the elementwise operations (and the sum) run in one loop".
fused_ew <- ask(
  xla = function(p) n_kernels(p) == 1L,
  gfortran = function(q) fortran_loops_with(q, c(g_exp, g_times2, g_plus1)) >= 1L,
  clang = function(fns) cpp_has(fns, c("llvm.exp" = 1L, "= fmul" = 1L, "= fadd" = 1L))
)
fused_sum <- ask(
  xla = fused_reduction,
  gfortran = function(q) fortran_loops_with(q, c(g_exp, g_times2, g_plus1, g_accumulate)) >= 1L,
  # one fadd for + 1, one for the sum
  clang = function(fns) cpp_has(fns, c("llvm.exp" = 1L, "= fmul" = 1L, "= fadd" = 2L))
)
# The answers to "no array is stored except for the inputs and the output". quickr's length checks
# evaluate operands such as exp(x) into heap arrays; etr's `assign()` evaluates an expression into
# a temporary buffer and copies that into the destination (`etr_bits/Core/ArrayClass.hpp`).
no_temporaries <- ask(
  xla = no_intermediates,
  gfortran = function(q) fortran_heap_arrays(q) == 0L,
  clang = function(fns) !cpp_calls(fns, "copy_with_temp")
)

# ------------------------------------------------------------------------------------------------
# 1. Fusion of elementwise operations
#
# gfortran turns a Fortran array expression into one loop, and etr's expression templates
# evaluate an expression element by element, so both fuse the operations of one R statement.

compare(
  "ew", function(x) { out <- exp(x) * 2 + 1; out }, list(x_small), vector_type,
  claims_xla = function(p) {
    body <- llvm_fp(xla_llvm_kernels(p)[[1L]])
    list(
      "exp, *, and + become a single kernel" = n_kernels(p) == 1L && length(xla_llvm_kernels(p)) == 1L,
      "no intermediate array is stored" = no_intermediates(p),
      "the kernel loads each element once, applies exp, *, and +, and stores the result once" =
        sum(grepl("= load double", body)) == 1L && sum(grepl("^store double", body)) == 1L &&
          any(grepl("xla.exp", body)) && any(grepl("fmul", body)) && any(grepl("fadd", body))
    )
  },
  questions = list(
    "exp(x) * 2 + 1: exp, *, and + run in one loop" = fused_ew,
    "exp(x) * 2 + 1: no array is stored other than the output" = no_temporaries
  )
)

# The same computation, written as three R statements. Tracing records the same operations, so
# the statements make no difference to XLA.
compare(
  "ew_split", function(x) { y <- exp(x); z <- y * 2; out <- z + 1; out }, list(x_small), vector_type,
  claims_xla = function(p) list(
    "exp, *, and + become a single kernel" = n_kernels(p) == 1L && length(xla_llvm_kernels(p)) == 1L,
    "no intermediate array is stored" = no_intermediates(p)
  ),
  questions = list(
    "y <- exp(x); z <- y * 2; z + 1: exp, *, and + run in one loop" = fused_ew,
    "y <- exp(x); z <- y * 2; z + 1: no array is stored other than the output" = no_temporaries
  )
)

# ------------------------------------------------------------------------------------------------
# 2. Fusion of elementwise operations into a reduction

compare(
  "sum", function(x) { out <- sum(exp(x) * 2 + 1); out }, list(x_large), vector_type,
  claims_xla = function(p) list(
    "the elementwise operations and the sum become a single kernel" = fused_reduction(p),
    "no intermediate array is stored" = no_intermediates(p)
  ),
  questions = list("sum(exp(x) * 2 + 1): exp, *, +, and the sum run in one loop" = fused_sum)
)

compare(
  "sum_split", function(x) { y <- exp(x) * 2 + 1; out <- sum(y); out }, list(x_large), vector_type,
  claims_xla = function(p) list(
    "the elementwise operations and the sum become a single kernel" = fused_reduction(p),
    "no intermediate array is stored" = no_intermediates(p)
  ),
  questions = list("y <- exp(x) * 2 + 1; sum(y): exp, *, +, and the sum run in one loop" = fused_sum)
)

# On small inputs, XLA:CPU rewrites the sum into a tree reduction (a reduce-window over blocks of
# 32 elements, then a reduce over the blocks), which it does not fuse with the elementwise
# operations, so their result is stored. Kept to show where the claim does not hold. The
# transpiled programs do not depend on the length of x.
compare(
  "sum_small", function(x) { out <- sum(exp(x) * 2 + 1); out }, list(x_small), vector_type,
  claims_xla = function(p) list(
    "the elementwise operations and the sum become a single kernel" = fused_reduction(p),
    "no intermediate array is stored" = no_intermediates(p)
  ),
  questions = list("sum(exp(x) * 2 + 1), x of length 1000: exp, *, +, and the sum run in one loop" = fused_sum)
)

# ------------------------------------------------------------------------------------------------
# 3. Algebraic simplification
#
# gfortran resolves transpose() by indexing, without a copy, and GCC turns the remaining copy
# into a memcpy. GCC folds `* 1` but keeps `+ 0`, which IEEE arithmetic does not allow to drop
# (-0 + 0 is +0). etr reads constants from objects at run time, so clang cannot fold them.

compare(
  "transposes", function(a) { out <- t(t(a)); out }, list(A), quote(declare(type(a = double(n, m)))),
  claims_xla = function(p) list(
    "anvl emits two transposes, XLA removes both and only copies A to the output" =
      sum(stablehlo_ops(p) == "transpose") == 2L && is_copy(p)
  ),
  questions = list("t(t(A)) is computed as a copy of A" = ask(
    xla = is_copy,
    gfortran = function(q) length(fortran_data_loops(q)) <= 1L && fortran_heap_arrays(q) == 0L &&
      !any(grepl("_gfortran_transpose", q$gimple)),
    clang = function(fns) !cpp_calls(fns, "etr::transpose")
  ))
)

compare(
  "identities", function(x) { out <- x * 1 + 0; out }, list(x_small), vector_type,
  claims_xla = function(p) list(
    "anvl emits a multiplication and an addition, XLA removes both and only copies x to the output" =
      all(c("multiply", "add") %in% stablehlo_ops(p)) && is_copy(p)
  ),
  questions = list("x * 1 + 0 is computed as a copy of x" = ask(
    xla = is_copy,
    gfortran = function(q) fortran_loops_with(q, "\\* 1\\.0e\\+0|\\+ 0\\.0") == 0L,
    clang = function(fns) !cpp_has(fns, c("= fmul" = 1L)) && !cpp_has(fns, c("= fadd" = 1L))
  ))
)

compare(
  "division", function(x) { out <- x / 4; out }, list(x_small), vector_type,
  claims_xla = function(p) list(
    "the division becomes a multiplication by 0.25" =
      "divide" %in% stablehlo_ops(p) && !"divide" %in% hlo_ops(p) &&
        any(grepl("constant(0.25)", unlist(hlo(p)), fixed = TRUE))
  ),
  questions = list("x / 4 becomes a multiplication by 0.25" = ask(
    xla = function(p) !"divide" %in% hlo_ops(p) && any(grepl("constant(0.25)", unlist(hlo(p)), fixed = TRUE)),
    gfortran = function(q) fortran_loops_with(q, "\\* 2\\.5e-1") == 1L,
    clang = function(fns) !cpp_has(fns, c("= fdiv" = 1L)) && cpp_has(fns, c("2.5" = 1L))
  ))
)

compare(
  "cse", function(x) { out <- exp(x) + exp(x); out }, list(x_small), vector_type,
  claims_xla = function(p) list(
    "anvl emits exp twice, XLA computes it once" =
      sum(stablehlo_ops(p) == "exponential") == 2L && sum(hlo_ops(p) == "exponential") == 1L
  ),
  questions = list("exp(x) + exp(x) computes exp once per element" = ask(
    xla = function(p) sum(hlo_ops(p) == "exponential") == 1L,
    # the loop that stores the result; the others are quickr's length checks
    gfortran = function(q) {
      out_loops <- Filter(function(l) any(grepl(g_store_out, l)), fortran_data_loops(q))
      length(out_loops) == 1L && sum(grepl(g_exp, out_loops[[1L]])) == 1L
    },
    clang = function(fns) !cpp_has(fns, c("llvm.exp" = 2L))
  ))
)

# ------------------------------------------------------------------------------------------------
# 4. Memory layouts
#
# quickr's code generator passes the transpose to BLAS (`dgemm('T', ...)`), and gfortran reads a
# transposed array by indexing, so neither copies A; etr's transpose() returns a new array.

compare(
  "t_matmul", function(a, b) { out <- t(a) %*% b; out }, list(A, B),
  quote(declare(type(a = double(n, m)), type(b = double(n, k)))),
  claims_xla = function(p) list(
    "anvl emits a transpose, XLA computes none: the matrix product reads A along its first axis" =
      "transpose" %in% stablehlo_ops(p) && !"transpose" %in% hlo_ops(p) &&
        any(grepl("dot\\(.*lhs_contracting_dims=\\{0\\}", hlo(p)$ENTRY)),
    "no intermediate array is stored" = no_intermediates(p)
  ),
  questions = list("t(A) %*% B: no transposed copy of A is stored" = ask(
    xla = function(p) !"transpose" %in% hlo_ops(p) && no_intermediates(p),
    gfortran = function(q) any(grepl("call dgemm\\('T'", q$statements)) && fortran_heap_arrays(q) == 0L,
    clang = function(fns) !cpp_calls(fns, "etr::transpose")
  ))
)

compare(
  "t_add", function(a, c) { out <- t(a) + c; out }, list(A, C),
  quote(declare(type(a = double(n, m)), type(c = double(m, n)))),
  claims_xla = function(p) {
    kernels <- xla_llvm_kernels(p)
    list(
      "the transpose and the addition become a single kernel" = n_kernels(p) == 1L && length(kernels) == 1L,
      "no intermediate array is stored" = no_intermediates(p),
      "the kernel transposes A in tiles of at most 8 x 8 on its stack" =
        length(kernels) == 1L && length(llvm_stack_arrays(kernels[[1L]])) > 0L &&
          max(llvm_stack_arrays(kernels[[1L]])) <= 64L
    )
  },
  questions = list("t(A) + C runs in one loop, without a transposed copy of A" = ask(
    xla = function(p) n_kernels(p) == 1L && no_intermediates(p),
    gfortran = function(q) length(fortran_data_loops(q)) == 1L && fortran_heap_arrays(q) == 0L,
    clang = function(fns) !cpp_calls(fns, "etr::transpose")
  ))
)

# ------------------------------------------------------------------------------------------------

cat("\n", strrep("=", 100), "\nClaims about XLA\n\n", sep = "")
claims <- do.call(rbind, claims)
cat(sprintf("[%s] %s (%s): %s", ifelse(claims$holds, "x", " "), claims$case, claims$code, claims$claim), sep = "\n")

cat("\nComparison of the compiled programs (anvl/XLA, quickr/gfortran, ast2ast/clang)\n\n")
answers <- do.call(rbind, answers)
yes_no <- function(x) ifelse(x, "yes", "no ")
cat(sprintf("%s  %s  %s  %s", "XLA", "gfortran", "clang", "question"), sep = "\n")
cat(sprintf("%-3s  %-8s  %-5s  %s", yes_no(answers$xla), yes_no(answers$gfortran), yes_no(answers$clang), answers$question), sep = "\n")

if (length(args)) cat("\nIntermediate files in", dump_dir, "\n") else unlink(dump_dir, recursive = TRUE)
