# How do the compared packages handle R's missing values when data enters and leaves a program?
#
# Generates Table \ref{tab:missing-values} in the appendix of the paper: running it writes
# missing_values.csv next to this file, which paper.Rmd reads. For every package, the vectors
#   double:  c(1, NA, NaN, Inf)
#   integer: c(1L, NA_integer_)
#   logical: c(TRUE, NA)
# are passed through a minimal compiled / traced / taped program (multiplication by one, or the
# identity where the package has no arithmetic on that type) and converted back to R. Each element
# of the result is classified as kept, NA -> NaN, NA -> <other value>, or the conversion errors.
#
# R's NA_real_ is a NaN with a special payload; most numeric libraries do not know it and treat it
# as an ordinary NaN, which arithmetic may or may not preserve. NA_integer_ is the integer
# -2147483648, so a library that does not know it computes with that number.
#
# Run with `Rscript verification/missing_values.R` from the paper directory (needs all packages,
# Python with TensorFlow via reticulate, and CmdStan). Uses the anvl source via pkgload::load_all().

x_dbl <- c(1, NA, NaN, Inf)
x_int <- c(1L, NA_integer_)
x_lgl <- c(TRUE, NA)

describe <- function(out, input) {
  out <- as.vector(out)
  if (length(out) != length(input)) {
    return(paste("wrong length:", paste(format(out), collapse = ", ")))
  }
  tags <- vapply(seq_along(input), function(i) {
    o <- out[[i]]
    inp <- input[[i]]
    if (is.na(inp) && !is.nan(inp)) {
      if (is.na(o) && !is.nan(o)) "NA kept" else if (is.nan(o)) "NA -> NaN" else paste("NA ->", format(o))
    } else if (is.nan(inp)) {
      if (is.nan(o)) "NaN kept" else paste("NaN ->", format(o))
    } else {
      if (isTRUE(all.equal(as.numeric(o), as.numeric(inp)))) "ok" else paste(format(inp), "->", format(o))
    }
  }, character(1L))
  paste(unique(tags[tags != "ok" & tags != "NaN kept"]), collapse = "; ") |>
    (\(s) if (nzchar(s)) s else "all values kept")()
}

results <- list()
check <- function(pkg, type, expr, input) {
  out <- tryCatch(
    withCallingHandlers(
      describe(force(expr), input),
      warning = function(w) {
        results[[pkg]][[paste0(type, "_warning")]] <<- conditionMessage(w)
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) paste("error:", strsplit(conditionMessage(e), "\n")[[1L]][1L])
  )
  w <- results[[pkg]][[paste0(type, "_warning")]]
  if (!is.null(w)) out <- paste0(out, " (warning: ", substr(w, 1L, 60L), ")")
  results[[pkg]][[type]] <<- out
  cat(sprintf("  %-10s %-8s %s\n", pkg, type, substr(out, 1L, 120L)))
}
run <- function(pkg, f_dbl, f_int, f_lgl) {
  cat("\n==", pkg, "==\n")
  check(pkg, "double", f_dbl(x_dbl), x_dbl)
  check(pkg, "integer", f_int(x_int), x_int)
  check(pkg, "logical", f_lgl(x_lgl), x_lgl)
}

# ------------------------------------------------------------------------------------------------
here <- local({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) dirname(normalizePath(sub("^--file=", "", f))) else getwd()
})
suppressMessages(pkgload::load_all(file.path(here, "..", ".."), quiet = TRUE))
times_one <- jit(function(x) x * 1L)
identity_jit <- jit(function(x) nv_not(nv_not(x)))
run("anvl",
  function(x) anvl::as_array(times_one(nv_array(x, dtype = "f64"))),
  function(x) anvl::as_array(times_one(nv_array(x, dtype = "i32"))),
  function(x) anvl::as_array(identity_jit(nv_array(x, dtype = "bool")))
)

# ------------------------------------------------------------------------------------------------
suppressMessages(library(torch))
run("torch",
  function(x) as.numeric(torch_tensor(x, dtype = torch_float64()) * 1),
  function(x) as.integer(torch_tensor(x, dtype = torch_int32()) * 1L),
  function(x) as.logical(torch_tensor(x, dtype = torch_bool())$logical_not()$logical_not())
)

# ------------------------------------------------------------------------------------------------
reticulate::py_require(c("tensorflow==2.15.*", "tensorflow_probability==0.23.*"), python_version = "3.10")
suppressMessages(library(tensorflow))
tf_times_one <- tf_function(function(x) x * tf$ones_like(x))
tf_not_not <- tf_function(function(x) tf$logical_not(tf$logical_not(x)))
run("tensorflow",
  function(x) as.array(tf_times_one(tf$constant(x, dtype = tf$float64))),
  function(x) as.array(tf_times_one(tf$constant(x, dtype = tf$int32))),
  function(x) as.array(tf_not_not(tf$constant(x, dtype = tf$bool)))
)

suppressMessages(library(greta))
run("greta",
  function(x) greta::calculate(as_data(x) * 1)[[1L]],
  function(x) greta::calculate(as_data(x) * 1)[[1L]],
  function(x) greta::calculate(as_data(x) * 1)[[1L]]
)

# ------------------------------------------------------------------------------------------------
suppressMessages(library(quickr))
q_dbl <- quick(function(x) {
  declare(type(x = double(n)))
  x * 1
})
q_int <- quick(function(x) {
  declare(type(x = integer(n)))
  x * 1L
})
q_lgl <- quick(function(x) {
  declare(type(x = logical(n)))
  !(!x)
})
run("quickr", q_dbl, q_int, q_lgl)

# ------------------------------------------------------------------------------------------------
suppressMessages(library(ast2ast))
a_dbl <- translate(function(x) {
  argtypes(x |> type(vec(double)))
  return(x * 1)
})
a_int <- translate(function(x) {
  argtypes(x |> type(vec(int)))
  return(x * 1L)
})
a_lgl <- translate(function(x) {
  argtypes(x |> type(vec(logical)))
  return(!(!x))
})
run("ast2ast", a_dbl, a_int, a_lgl)

# ------------------------------------------------------------------------------------------------
suppressMessages(library(cmdstanr))
stan_identity <- function(type_decl, x) {
  code <- sprintf("data { int n; %s x; }\ngenerated quantities { %s y = x; }\n", type_decl, type_decl)
  mod <- cmdstan_model(write_stan_file(code), quiet = TRUE)
  fit <- mod$sample(
    data = list(n = length(x), x = x), fixed_param = TRUE, chains = 1, iter_sampling = 1,
    refresh = 0, show_messages = FALSE
  )
  as.numeric(fit$draws("y"))
}
run("Stan",
  function(x) stan_identity("vector[n]", x),
  function(x) stan_identity("array[n] int", x),
  function(x) stan_identity("array[n] int", as.integer(x)) # Stan has no logical type
)

# ------------------------------------------------------------------------------------------------
suppressMessages(library(TMB))
tmb_dir <- tempfile("tmb")
dir.create(tmb_dir)
writeLines("
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() () {
  DATA_VECTOR(xd);
  DATA_IVECTOR(xi);
  PARAMETER(dummy);
  vector<Type> yd = xd * Type(1);
  vector<int> yi = xi * 1;
  REPORT(yd);
  REPORT(yi);
  return dummy * dummy;
}
", file.path(tmb_dir, "navals.cpp"))
invisible(compile(file.path(tmb_dir, "navals.cpp"), flags = "-w"))
dyn.load(dynlib(file.path(tmb_dir, "navals")))
tmb_report <- function(xd, xi) {
  MakeADFun(list(xd = xd, xi = xi), list(dummy = 0), DLL = "navals", silent = TRUE)$report()
}
run("TMB",
  function(x) tmb_report(x, x_int)$yd,
  function(x) tmb_report(x_dbl, x)$yi,
  function(x) tmb_report(x_dbl, as.integer(x))$yi # TMB has no logical data type
)

suppressMessages(library(RTMB))
run("RTMB",
  function(x) MakeTape(function(p) p * 1, x)(x),
  function(x) MakeTape(function(p) p * 1, as.double(x))(as.double(x)), # tapes are double only
  function(x) MakeTape(function(p) p * 1, as.double(x))(as.double(x))
)

# ------------------------------------------------------------------------------------------------
suppressMessages(library(nimble))
n_dbl <- compileNimble(nimbleFunction(run = function(x = double(1)) {
  returnType(double(1))
  return(x * 1)
}), showCompilerOutput = FALSE)
n_int <- compileNimble(nimbleFunction(run = function(x = integer(1)) {
  returnType(integer(1))
  return(x * 1L)
}), showCompilerOutput = FALSE)
n_lgl <- compileNimble(nimbleFunction(run = function(x = logical(1)) {
  returnType(logical(1))
  return(!(!x))
}), showCompilerOutput = FALSE)
run("nimble", n_dbl, n_int, n_lgl)

# ------------------------------------------------------------------------------------------------
# NA_integer_ is the integer -2147483648, so "kept" above may only mean that multiplying by one
# left that number unchanged. Arithmetic shows whether the package treats it as missing: in R,
# NA_integer_ + 1L is NA; a package that does not know it returns -2147483647.
cat("\n== NA_integer_ + 1L ==\n")
plus_one <- list(
  anvl = function(x) anvl::as_array(jit(function(x) x + 1L)(nv_array(x, dtype = "i32"))),
  torch = function(x) as.integer(torch_tensor(x, dtype = torch_int32()) + 1L),
  tensorflow = function(x) as.array(tf_function(function(x) x + tf$ones_like(x))(tf$constant(x, dtype = tf$int32))),
  quickr = quick(function(x) {
    declare(type(x = integer(n)))
    x + 1L
  }),
  ast2ast = translate(function(x) {
    argtypes(x |> type(vec(int)))
    return(x + 1L)
  }),
  nimble = compileNimble(nimbleFunction(run = function(x = integer(1)) {
    returnType(integer(1))
    return(x + 1L)
  }), showCompilerOutput = FALSE)
)
for (pkg in names(plus_one)) {
  out <- tryCatch(
    suppressWarnings(as.vector(plus_one[[pkg]](x_int))),
    error = function(e) paste("error:", strsplit(conditionMessage(e), "\n")[[1L]][1L])
  )
  res <- if (is.character(out)) out else if (is.na(out[[2L]])) "NA (correct)" else paste("NA ->", format(out[[2L]]), "(wrong)")
  results[[pkg]][["integer_plus_one"]] <- res
  cat(sprintf("  %-10s %s\n", pkg, res))
}

# ------------------------------------------------------------------------------------------------
cat("\n\nSummary\n\n")
tab <- do.call(rbind, lapply(results, function(r) c(double = r$double, integer = r$integer, logical = r$logical, "integer + 1L" = r$integer_plus_one %||% "")))
print(noquote(tab))

# ------------------------------------------------------------------------------------------------
# Short cells for the paper's table.
short <- function(x) {
  if (is.null(x) || !nzchar(x)) {
    return("--")
  }
  if (startsWith(x, "error")) {
    return("error")
  }
  x <- sub(" \\(warning.*", " (warning)", x)
  switch(sub(" \\(warning\\)", "", x),
    "NA kept" = sub("NA kept", "kept", x),
    "NA -> NaN" = "NaN",
    "NA (correct)" = "NA",
    sub("^NA -> ", "", sub(" \\(wrong\\)", "", x))
  )
}
table_out <- data.frame(
  package = names(results),
  double = vapply(results, function(r) short(r$double), character(1L)),
  integer = vapply(results, function(r) short(r$integer), character(1L)),
  integer_plus_one = vapply(results, function(r) short(r$integer_plus_one), character(1L)),
  logical = vapply(results, function(r) short(r$logical), character(1L))
)
write.csv(table_out, file.path(here, "missing_values.csv"), row.names = FALSE)
cat("\nwrote", file.path(here, "missing_values.csv"), "\n")
