# Which array ranks can each package handle inside its compiled / traced / taped programs?
#
# Personal notes, not part of the paper. For every package, a program that sums its input is
# built for inputs of rank 0 (scalar), 1 (vector), 2 (matrix), 3, and a high rank, and the
# result is compared with base R. "ok" means the program compiled and returned the right sum;
# otherwise the (first line of the) error is recorded.
#
# Findings beyond the summary table:
# - ast2ast: arguments can only be scalars, vectors, or matrices; the higher ranks pass because
#   the array is allocated with `array()` inside the function.
# - nimble: `sum()` and other reductions on arrays with more than 2 axes fail ("cannot eigenize a
#   map of dimensions > 2"), hence the loops; arrays with up to 6 axes compile, 7 fails
#   ("implicit instantiation of undefined template 'NimArr<7, double>'").
# - Stan: `sum()` takes arrays with at most one axis, so higher ranks go through `to_array_1d()`.
# - odin2 documents index variables for up to 8 axes; 9 axes compiled in this test.
#
# Run with `Rscript array_ranks.R`. Uses the current anvl source via pkgload::load_all().

ranks <- c(scalar = 0L, vector = 1L, matrix = 2L, "3-d" = 3L, "5-d" = 5L, "9-d" = 9L)
input <- function(rank) {
  if (rank == 0L) {
    return(2)
  }
  shape <- rep(2L, rank)
  array(seq_len(prod(shape)) / 10, dim = shape)
}
expected <- function(rank) sum(input(rank))

results <- list()
record <- function(pkg, rank_name, expr) {
  out <- tryCatch(
    {
      val <- force(expr)
      if (isTRUE(all.equal(as.numeric(val), expected(ranks[[rank_name]]), tolerance = 1e-6))) "ok" else paste("wrong:", format(val))
    },
    error = function(e) paste("error:", strsplit(conditionMessage(e), "\n")[[1L]][1L])
  )
  results[[pkg]][[rank_name]] <<- out
  cat(sprintf("  %-10s %-7s %s\n", pkg, rank_name, substr(out, 1L, 110L)))
}
run_all <- function(pkg, f) {
  cat("\n==", pkg, "==\n")
  for (r in names(ranks)) record(pkg, r, f(ranks[[r]]))
}

# --------------------------------------------------------------------------------------------
suppressMessages(pkgload::load_all("/Users/sebi/r-xla/anvl-paper", quiet = TRUE))
anvl_sum <- jit(function(x) sum(x))
run_all("anvl", function(rank) anvl::as_array(anvl_sum(nv_array(input(rank), dtype = "f64"))))

# --------------------------------------------------------------------------------------------
suppressMessages(library(torch))
run_all("torch", function(rank) {
  x <- torch_tensor(input(rank), dtype = torch_float64())
  as.numeric(torch_sum(x))
})

# --------------------------------------------------------------------------------------------
reticulate::py_require(c("tensorflow==2.15.*", "tensorflow_probability==0.23.*"), python_version = "3.10")
suppressMessages(library(tensorflow))
tf_sum <- tf_function(function(x) tf$reduce_sum(x), jit_compile = TRUE)
run_all("tensorflow", function(rank) as.numeric(tf_sum(tf$constant(input(rank), dtype = tf$float64))))

suppressMessages(library(greta))
run_all("greta", function(rank) {
  x <- as_data(input(rank))
  as.numeric(greta::calculate(sum(x))[[1L]])
})

# --------------------------------------------------------------------------------------------
suppressMessages(library(quickr))
run_all("quickr", function(rank) {
  type_call <- if (rank == 0L) quote(double(1)) else as.call(c(as.name("double"), rep(list(NA), rank)))
  decl <- bquote(declare(type(x = .(type_call))))
  f <- eval(bquote(function(x) {
    .(decl)
    sum(x)
  }))
  quick(f)(input(rank))
})

# --------------------------------------------------------------------------------------------
suppressMessages(library(ast2ast))
run_all("ast2ast", function(rank) {
  # Argument types are scalar, vec or mat; higher ranks are tried via the documented `array()`
  # allocation inside the function, filled from a vector argument.
  if (rank <= 2L) {
    # `x |> type(T)` is parsed as `type(x, T)`, which bquote() can build directly
    type_arg <- switch(rank + 1L, quote(double), quote(vec(double)), quote(mat(double)))
    ret <- if (rank == 0L) quote(x) else quote(sum(x))
    f <- eval(bquote(function(x) {
      argtypes(type(x, .(type_arg)))
      return(.(ret))
    }))
    translate(f)(input(rank))
  } else {
    shape_call <- as.call(c(as.name("c"), as.list(rep(2L, rank))))
    f <- eval(bquote(function(x) {
      argtypes(x |> type(vec(double)))
      a <- array(x, .(shape_call))
      return(sum(a))
    }))
    translate(f)(as.vector(input(rank)))
  }
})

# --------------------------------------------------------------------------------------------
suppressMessages({
  library(odin2)
  library(dust2)
})
run_all("odin2", function(rank) {
  # A parameter array of the given rank, summed into the state.
  if (rank == 0L) {
    gen <- odin(
      {
        initial(s) <- 0
        update(s) <- x
        x <- parameter()
      },
      quiet = TRUE
    )
  } else {
    idx <- paste(rep("", rank), collapse = ",")
    dims <- paste0("c(", paste(rep(2L, rank), collapse = ", "), ")")
    code <- sprintf("initial(s) <- 0\nupdate(s) <- sum(x[%s])\nx <- parameter()\ndim(x) <- %s", idx, dims)
    gen <- odin(code, quiet = TRUE)
  }
  sys <- dust_system_create(gen, list(x = input(rank)))
  dust_system_run_to_time(sys, 1)
  dust_system_state(sys)
})

# --------------------------------------------------------------------------------------------
suppressMessages(library(cmdstanr))
stan_type <- function(rank) {
  switch(as.character(rank),
    "0" = "real x;",
    "1" = "vector[2] x;",
    "2" = "matrix[2, 2] x;",
    sprintf("array[%s] real x;", paste(rep(2L, rank), collapse = ", "))
  )
}
stan_sum <- function(rank) {
  if (rank == 0L) {
    return("x")
  }
  if (rank <= 2L) {
    return("sum(x)")
  }
  # sum() takes arrays of up to one dimension; flatten with to_array_1d()
  "sum(to_array_1d(x))"
}
run_all("Stan", function(rank) {
  code <- sprintf("data { %s }\ngenerated quantities { real s = %s; }\n", stan_type(rank), stan_sum(rank))
  mod <- cmdstan_model(write_stan_file(code), quiet = TRUE)
  fit <- mod$sample(
    data = list(x = input(rank)), fixed_param = TRUE, chains = 1, iter_sampling = 1,
    refresh = 0, show_messages = FALSE
  )
  as.numeric(fit$draws("s"))
})

# --------------------------------------------------------------------------------------------
suppressMessages(library(TMB))
tmb_dir <- tempfile("tmb")
dir.create(tmb_dir)
writeLines("
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() () {
  DATA_ARRAY(x);
  PARAMETER(dummy);
  return x.sum() + Type(0) * dummy;
}
", file.path(tmb_dir, "ranksum.cpp"))
invisible(compile(file.path(tmb_dir, "ranksum.cpp"), flags = "-w"))
dyn.load(dynlib(file.path(tmb_dir, "ranksum")))
writeLines("
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() () {
  DATA_SCALAR(x);
  PARAMETER(dummy);
  return x + Type(0) * dummy;
}
", file.path(tmb_dir, "scalarsum.cpp"))
invisible(compile(file.path(tmb_dir, "scalarsum.cpp"), flags = "-w"))
dyn.load(dynlib(file.path(tmb_dir, "scalarsum")))
run_all("TMB", function(rank) {
  dll <- if (rank == 0L) "scalarsum" else "ranksum" # DATA_SCALAR vs. DATA_ARRAY
  obj <- MakeADFun(list(x = input(rank)), list(dummy = 0), DLL = dll, silent = TRUE)
  obj$fn(0)
})

suppressMessages(library(RTMB))
run_all("RTMB", function(rank) {
  # Taped w.r.t. the array's elements, reshaped to the array inside the tape.
  x <- input(rank)
  shape <- dim(x)
  F <- MakeTape(function(v) {
    a <- if (is.null(shape)) v else array(v, dim = shape)
    sum(a)
  }, as.vector(x))
  F(as.vector(x))
})

# --------------------------------------------------------------------------------------------
suppressMessages(library(nimble))
nimble_body <- function(rank) {
  if (rank == 0L) {
    return(quote(s <- x))
  }
  if (rank <= 2L) {
    return(quote(s <- sum(x)))
  }
  idx <- paste0("i", seq_len(rank))
  inner <- sprintf("s <- s + x[%s]", paste(idx, collapse = ", "))
  loops <- Reduce(function(body, i) sprintf("for (%s in 1:2) { %s }", i, body), rev(idx), inner)
  parse(text = paste0("{ s <- 0; ", loops, " }"))[[1L]]
}
run_all("nimble", function(rank) {
  type_call <- call("double", rank)
  body <- nimble_body(rank)
  f <- eval(bquote(nimbleFunction(run = function(x = .(type_call)) {
    .(body)
    returnType(double(0))
    return(s)
  })))
  cf <- compileNimble(f, showCompilerOutput = FALSE)
  cf(input(rank))
})

# --------------------------------------------------------------------------------------------
cat("\n\nSummary (ok = compiled and correct)\n\n")
tab <- do.call(rbind, lapply(results, function(r) vapply(names(ranks), function(n) if (identical(r[[n]], "ok")) "ok" else "FAIL", character(1L))))
print(noquote(tab))
