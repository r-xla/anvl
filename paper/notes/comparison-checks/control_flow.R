# Control flow, and differentiating through it, in the packages of the comparison tables.
#
# Personal notes, not part of the paper. For every package (and every execution mode a package
# has, e.g. torch eager vs. TorchScript), this script checks, *inside* the compiled / traced /
# taped program:
#
#   (a) if    -- if/else on a run-time (data-dependent) value:   if (x > 0) x^2 else -x
#   (b) while -- a while loop with a data-dependent trip count:   while (x < 10) x <- 2 * x
#   (c) loop  -- a fixed-length loop / scan (3 steps):            for (i in 1:3) x <- x * 1.1
#   (d) grad  -- for AD-capable packages, whether d/dx flows through (a), (b), (c) and through
#                their composition  f(x) = if_(while_(loop_(x))).
#
# The decisive test: every construct is compiled / traced / taped ONCE and then evaluated at TWO
# inputs that take different paths (the other branch, a different trip count). A package that
# freezes the path at tracing/taping time passes on the first input and fails on the second.
#
#   construct   inputs      values            gradients
#   if          2, -2       4, 2              4, -1
#   while       2, 3        16, 12            8, 4        (3 vs. 2 doublings)
#   loop        2, -1       2.662, -1.331     1.331, 1.331
#   combined    2, 0.5      113.36, 113.38    (5.324^2 * 2x, 21.296^2 * 2x; 2 vs. 4 doublings)
#
# Cells: "ok" = correct at both inputs; "ok*" = correct, but with a caveat (see notes); "--" =
# not possible / wrong result (error message or wrong values in the notes); "n/a" = no AD.
#
# Run all:          Rscript control_flow.R
# Run a subset:     CF_ONLY=anvl,torch Rscript control_flow.R
# Section names:    anvl torch tensorflow greta quickr ast2ast odin2 stan tmb rtmb nimble
#
# Needs: anvl (dev source, loaded with pkgload), torch, tensorflow + tfautograph + greta (Python via
# reticulate, TF 2.15), quickr, ast2ast, odin2 + dust2, cmdstanr (CmdStan), TMB, RTMB, nimble.

options(warn = 1)

# ---- reference values ---------------------------------------------------------------------------

ref_if <- function(x) if (x > 0) x^2 else -x
ref_while <- function(x) {
  while (x < 10) x <- 2 * x
  x
}
ref_loop <- function(x) {
  for (i in 1:3) x <- x * 1.1
  x
}
ref_all <- function(x) ref_if(ref_while(ref_loop(x)))
fd <- function(f, x, h = 1e-6) (f(x + h) - f(x - h)) / (2 * h) # paths are locally constant

pts <- list(`if` = c(2, -2), `while` = c(2, 3), loop = c(2, -1), all = c(2, 0.5))
refs <- list(`if` = ref_if, `while` = ref_while, loop = ref_loop, all = ref_all)
val <- lapply(names(pts), function(k) vapply(pts[[k]], refs[[k]], numeric(1L)))
grd <- lapply(names(pts), function(k) vapply(pts[[k]], function(x) fd(refs[[k]], x), numeric(1L)))
names(val) <- names(grd) <- names(pts)

# ---- bookkeeping --------------------------------------------------------------------------------

constructs <- c("if", "while", "loop", "grad if", "grad while", "grad loop", "grad all")
results <- data.frame(pkg = character(), construct = character(), status = character(), note = character())

rec <- function(pkg, construct, status, note = "") {
  results[nrow(results) + 1L, ] <<- list(pkg, construct, status, note)
  cat(sprintf("  %-26s %-11s %-4s %s\n", pkg, construct, status, substr(gsub("\\s+", " ", note), 1L, 160L)))
}

first_lines <- function(msg, n = 2L) paste(head(strsplit(trimws(msg), "\n")[[1L]], n), collapse = " | ")

# Evaluate `expr` (which must return the values at the two test points of `construct`) and record
# ok / wrong result / error. `kind` is "val" or "grad"; `ok_status` lets the caller downgrade a
# correct result to "ok*" when there is a caveat.
chk <- function(pkg, construct, expr, ok_status = "ok", note = "") {
  key <- sub("grad ", "", construct)
  expected <- if (startsWith(construct, "grad")) grd[[key]] else val[[key]]
  got <- tryCatch(as.numeric(force(expr)), error = function(e) e)
  if (inherits(got, "error")) {
    rec(pkg, construct, "--", paste0(if (nzchar(note)) paste0(note, " ERROR: ") else "ERROR: ", first_lines(conditionMessage(got))))
  } else if (length(got) == length(expected) && isTRUE(all.equal(got, expected, tolerance = 1e-5))) {
    rec(pkg, construct, ok_status, note)
  } else {
    rec(
      pkg, construct, "--",
      paste0(if (nzchar(note)) paste0(note, " ") else "", "WRONG RESULT: got ", paste(signif(got, 6L), collapse = ", "),
        ", expected ", paste(signif(expected, 6L), collapse = ", "))
    )
  }
}

# Run `expr` and print the error message (or "no error") -- for documenting expected failures
# that are not table cells.
probe <- function(what, expr) {
  msg <- tryCatch(
    {
      r <- force(expr)
      paste("no error; result:", paste(format(r), collapse = ", "))
    },
    error = function(e) paste("ERROR:", first_lines(conditionMessage(e), 3L))
  )
  cat("  [probe]", what, "\n    ->", msg, "\n")
}
note <- function(...) cat("  [note]", ..., "\n")

only <- strsplit(Sys.getenv("CF_ONLY", ""), ",")[[1L]]
section <- function(name, title, code) {
  if (length(only) && !name %in% only) {
    return(invisible())
  }
  cat("\n", strrep("=", 90), "\n", title, "\n", strrep("=", 90), "\n", sep = "")
  tryCatch(code, error = function(e) cat("  SECTION FAILED:", conditionMessage(e), "\n"))
}

# =================================================================================================
section("anvl", "anvl: nv_if() / nv_while() / nv_scan() are traced into stablehlo case / while ops", {
  pkgload::load_all("/Users/sebi/r-xla/anvl-paper", quiet = TRUE)
  s <- function(v) nv_scalar(v, dtype = "f64")
  num <- function(a) as.numeric(as_array(a))
  at <- function(f, xs) vapply(xs, function(x) num(f(s(x))), numeric(1L))
  P <- "anvl"

  a_if <- function(x) nv_if(x > 0, \() x * x, \() -x)
  a_while <- function(x) nv_while(init = list(x = x), cond = function(x) x < 10, body = function(x) list(x = 2 * x))$x
  a_scan <- function(x) nv_scan(init = x, body = function(carry, xs) list(carry = carry * 1.1, out = NULL), steps = 3L)$carry
  a_for <- function(x) {
    for (i in 1:3) x <- x * 1.1 # plain R loop: unrolled while tracing
    x
  }
  a_all_scan <- function(x) a_if(a_while(a_scan(x)))
  a_all_for <- function(x) a_if(a_while(a_for(x)))
  a_if_for <- function(x) a_if(a_for(x))

  chk(P, "if", at(jit(a_if), pts$`if`), note = "nv_if() -> stablehlo `case`; both branches compiled, one executed.")
  chk(P, "while", at(jit(a_while), pts$`while`), note = "nv_while() -> stablehlo `while`.")
  chk(P, "loop", at(jit(a_scan), pts$loop), note = "nv_scan(steps = 3L) -> stablehlo `while` with static trip count (not unrolled).")
  chk(P, "loop", at(jit(a_for), pts$loop), ok_status = "ok*", note = "plain R for loop: unrolled at trace time.")

  g <- function(f) jit(gradient(f))
  gat <- function(f, xs) vapply(xs, function(x) num(g(f)(s(x))$x), numeric(1L))
  chk(P, "grad if", gat(a_if, pts$`if`), note = "reverse rule of prim_if: differentiates the taken branch.")
  chk(P, "grad while", gat(a_while, pts$`while`))
  chk(P, "grad loop", gat(a_scan, pts$loop), note = "nv_scan:")
  chk(P, "grad loop", gat(a_for, pts$loop), ok_status = "ok*", note = "unrolled R for loop.")
  chk(P, "grad all", gat(a_all_scan, pts$all), note = "with nv_scan + nv_while:")

  probe("plain R `if` on a traced value inside jit()", jit(function(x) if (x > 0) x * x else -x)(s(2)))
  probe("plain R `while` on a traced value inside jit()", jit(function(x) {
    while (x < 10) x <- 2 * x
    x
  })(s(2)))
  probe("eager (no jit) plain R `if` on an AnvlArray", (function(x) if (x > 0) x * x else -x)(s(2)))
  probe("eager (no jit) R `if` on as_array(x > 0)", (function(x) if (as_array(x > 0)) x * x else -x)(s(-2)))
  probe("gradient through nv_if + unrolled R for loop (no while)", gat(a_if_for, c(2, -2)))
})

# =================================================================================================
section("torch", "torch: eager R control flow (dynamic tape) and TorchScript (jit_compile / jit_trace)", {
  library(torch)
  tt <- function(v, rg = FALSE) torch_tensor(v, dtype = torch_float64(), requires_grad = rg)
  at <- function(f, xs) vapply(xs, function(x) as.numeric(f(tt(x))), numeric(1L))
  gat <- function(f, xs) {
    vapply(xs, function(x) {
      xt <- tt(x, rg = TRUE)
      f(xt)$backward()
      as.numeric(xt$grad)
    }, numeric(1L))
  }

  # Eager: R's own `if` / `while` / `for`, the condition pulled to R with $item().
  t_if <- function(x) if ((x > 0)$item()) x * x else -x
  t_while <- function(x) {
    while ((x < 10)$item()) x <- 2 * x
    x
  }
  t_loop <- function(x) {
    for (i in 1:3) x <- x * 1.1
    x
  }
  t_all <- function(x) t_if(t_while(t_loop(x)))
  P <- "torch (eager)"
  chk(P, "if", at(t_if, pts$`if`), note = "R `if`; condition read back to R with $item() (one sync per branch).")
  chk(P, "while", at(t_while, pts$`while`), note = "R `while`, condition via $item().")
  chk(P, "loop", at(t_loop, pts$loop), note = "R `for`.")
  chk(P, "grad if", gat(t_if, pts$`if`), note = "autograd records the path actually taken (define-by-run).")
  chk(P, "grad while", gat(t_while, pts$`while`))
  chk(P, "grad loop", gat(t_loop, pts$loop))
  chk(P, "grad all", gat(t_all, pts$all))
  probe("R `if (x > 0)` directly on a torch tensor (no $item())", (function(x) if (x > 0) x * x else -x)(tt(2)))

  # TorchScript: control flow written in TorchScript (Python subset) source.
  ts <- jit_compile("
def f_if(x):
    if bool(x > 0):
        return x * x
    else:
        return -x

def f_while(x):
    while bool(x < 10):
        x = 2 * x
    return x

def f_loop(x):
    for i in range(3):
        x = x * 1.1
    return x

def f_all(x):
    for i in range(3):
        x = x * 1.1
    while bool(x < 10):
        x = 2 * x
    if bool(x > 0):
        return x * x
    else:
        return -x
")
  P <- "torch (TorchScript)"
  chk(P, "if", at(ts$f_if, pts$`if`), note = "jit_compile(): TorchScript `if` (prim::If node), source is TorchScript, not R.")
  chk(P, "while", at(ts$f_while, pts$`while`), note = "TorchScript `while` (prim::Loop).")
  chk(P, "loop", at(ts$f_loop, pts$loop), note = "TorchScript `for i in range(3)` (prim::Loop).")
  chk(P, "grad if", gat(ts$f_if, pts$`if`), note = "autograd through the scripted function.")
  chk(P, "grad while", gat(ts$f_while, pts$`while`))
  chk(P, "grad loop", gat(ts$f_loop, pts$loop))
  chk(P, "grad all", gat(ts$f_all, pts$all))

  # jit_trace(): records the ops of one run; R control flow is frozen.
  P <- "torch (jit_trace)"
  tr_if <- suppressWarnings(jit_trace(t_if, tt(2)))
  tr_while <- suppressWarnings(jit_trace(t_while, tt(2)))
  chk(P, "if", at(tr_if, pts$`if`), note = "jit_trace() of the eager R function, traced at x = 2:")
  chk(P, "while", at(tr_while, pts$`while`), note = "jit_trace(), traced at x = 2:")
})

# =================================================================================================
section("tensorflow", "tensorflow: tf$cond / tf$while_loop / tf$scan in tf_function(), and R syntax via tfautograph", {
  reticulate::py_require(c("tensorflow==2.15.*", "tensorflow_probability==0.23.*"), python_version = "3.10")
  library(tensorflow)
  tc <- function(v) tf$constant(v, dtype = tf$float64)
  at <- function(f, xs) vapply(xs, function(x) as.numeric(f(tc(x))), numeric(1L))
  # gradient computed *inside* the traced graph
  grad_fn <- function(f, jit = FALSE) {
    tf_function(function(x) {
      with(tf$GradientTape() %as% tape, {
        tape$watch(x)
        y <- f(x)
      })
      tape$gradient(y, x)
    }, autograph = FALSE, jit_compile = jit)
  }

  f_if <- function(x) tf$cond(x > 0, function() x * x, function() -x)
  f_while <- function(x) tf$while_loop(function(x) x < 10, function(x) list(2 * x), list(x))[[1]]
  f_scan <- function(x) {
    out <- tf$scan(function(a, e) a * 1.1, elems = tf$zeros(3L, dtype = tf$float64), initializer = x)
    tf$gather(out, 2L)
  }
  f_all <- function(x) f_if(f_while(f_scan(x)))

  for (jit in c(FALSE, TRUE)) {
    P <- if (jit) "tensorflow (tf ops, XLA)" else "tensorflow (tf ops)"
    tfun <- function(f) tf_function(f, autograph = FALSE, jit_compile = jit)
    chk(P, "if", at(tfun(f_if), pts$`if`), note = "tf$cond() in tf_function(): graph-level Cond/If op.")
    chk(P, "while", at(tfun(f_while), pts$`while`), note = "tf$while_loop().")
    chk(P, "loop", at(tfun(f_scan), pts$loop), note = "tf$scan() (a while loop with static trip count).")
    chk(P, "grad if", at(grad_fn(f_if, jit), pts$`if`), note = "GradientTape inside tf_function().")
    chk(P, "grad while", at(grad_fn(f_while, jit), pts$`while`))
    chk(P, "grad loop", at(grad_fn(f_scan, jit), pts$loop))
    chk(P, "grad all", at(grad_fn(f_all, jit), pts$all))
  }
  # Documented fix for the XLA error above: bound the loop with maximum_iterations, so the
  # per-iteration values kept for the backward pass get a static size.
  P <- "tensorflow (tf ops, XLA)"
  f_while_max <- function(x) tf$while_loop(function(x) x < 10, function(x) list(2 * x), list(x), maximum_iterations = 100L)[[1]]
  chk(P, "grad while", at(grad_fn(f_while_max, TRUE), pts$`while`), ok_status = "ok*", note = "tf$while_loop(maximum_iterations = 100L) under XLA:")
  chk(P, "grad all", at(grad_fn(function(x) f_if(f_while_max(f_scan(x))), TRUE), pts$all), ok_status = "ok*", note = "with maximum_iterations:")

  # tfautograph (R package, used by tf_function(autograph = TRUE), the default): rewrites R's
  # `if` / `while` / `for` on tensors into tf$cond / tf$while_loop.
  library(tfautograph)
  ag_if <- function(x) {
    if (x > 0) y <- x * x else y <- -x
    y
  }
  ag_while <- function(x) {
    while (x < 10) x <- 2 * x
    x
  }
  ag_loop <- function(x) {
    for (i in tf$range(3L)) x <- x * 1.1
    x
  }
  ag_all <- function(x) ag_if(ag_while(ag_loop(x)))
  P <- "tensorflow (autograph)"
  chk(P, "if", at(tf_function(ag_if), pts$`if`), note = "plain R `if` in tf_function(autograph = TRUE) (tfautograph).")
  chk(P, "while", at(tf_function(ag_while), pts$`while`), note = "plain R `while` via tfautograph.")
  chk(P, "loop", at(tf_function(ag_loop), pts$loop), note = "R `for` over tf$range() -> tf while loop (an R vector would be unrolled).")
  # autograph() makes the translation explicit, so it also applies to functions called in the graph
  agrad <- function(f) grad_fn(autograph(f))
  chk(P, "grad if", at(agrad(ag_if), pts$`if`), note = "GradientTape around tfautograph::autograph(f).")
  chk(P, "grad while", at(agrad(ag_while), pts$`while`))
  chk(P, "grad loop", at(agrad(ag_loop), pts$loop))
  chk(P, "grad all", at(grad_fn(function(x) autograph(ag_if)(autograph(ag_while)(autograph(ag_loop)(x)))), pts$all))
  probe("R `if` on a symbolic tensor with autograph = FALSE", tf_function(ag_if, autograph = FALSE)(tc(2)))
})

# =================================================================================================
section("greta", "greta: model is an R-built TF graph; no control-flow constructs on greta arrays", {
  reticulate::py_require(c("tensorflow==2.15.*", "tensorflow_probability==0.23.*"), python_version = "3.10")
  suppressPackageStartupMessages(library(greta))
  P <- "greta"
  x <- variable()
  calc <- function(y, xs) vapply(xs, function(v) as.numeric(calculate(y, values = list(x = v))[[1]]), numeric(1L))

  r_if <- tryCatch(if (x > 0) x^2 else -x, error = function(e) e)
  rec(P, "if", "--", paste("R `if` on a greta array: ERROR:", first_lines(conditionMessage(r_if))))
  probe("comparison x > 0 on a greta array (class of result)", class(x > 0))
  probe("base::ifelse() on a greta array", ifelse(x > 0, x^2, -x))
  # Comparisons return 0/1 greta arrays, so a branch can be written as masked arithmetic
  # (both branches are evaluated; NaN/Inf in the unused branch would poison the result).
  y_mask <- (x > 0) * x^2 + (x <= 0) * (-x)
  chk(P, "if", calc(y_mask, pts$`if`), ok_status = "ok*", note = "no branching: 0/1 mask arithmetic (x > 0) * x^2 + (x <= 0) * (-x), both sides evaluated.")
  r_while <- tryCatch(
    {
      y <- x
      while (y < 10) y <- 2 * y
      y
    },
    error = function(e) e
  )
  rec(P, "while", "--", paste("R `while` on a greta array: ERROR:", first_lines(conditionMessage(r_while))))
  y_loop <- x
  for (i in 1:3) y_loop <- y_loop * 1.1
  chk(P, "loop", calc(y_loop, pts$loop), ok_status = "ok*", note = "R `for` unrolled while building the model graph.")
  rec(P, "grad loop", "ok*", "unrolled loop is ordinary graph ops, so HMC gradients (TF, internal) flow; no user-level gradient API.")
  rec(P, "grad if", "--", "no conditional")
  rec(P, "grad while", "--", "no while loop")
  rec(P, "grad all", "--", "no conditional / while loop")
})

# =================================================================================================
section("quickr", "quickr: R `if` / `while` / `for` compiled to Fortran; no AD", {
  library(quickr)
  P <- "quickr"
  q_if <- quick(function(x) {
    declare(type(x = double(1)))
    if (x > 0) {
      y <- x * x
    } else {
      y <- -x
    }
    y
  })
  q_while <- quick(function(x) {
    declare(type(x = double(1)))
    y <- x
    while (y < 10) y <- 2 * y
    y
  })
  q_loop <- quick(function(x) {
    declare(type(x = double(1)))
    y <- x
    for (i in 1:3) y <- y * 1.1
    y
  })
  at <- function(f, xs) vapply(xs, f, numeric(1L))
  chk(P, "if", at(q_if, pts$`if`), note = "Fortran IF.")
  chk(P, "while", at(q_while, pts$`while`), note = "Fortran DO WHILE.")
  chk(P, "loop", at(q_loop, pts$loop), note = "Fortran DO loop.")
  for (k in c("grad if", "grad while", "grad loop", "grad all")) rec(P, k, "n/a", "quickr has no AD")
})

# =================================================================================================
section("ast2ast", "ast2ast: R `if` / `while` / `for` translated to C++; forward + reverse AD on the run", {
  library(ast2ast)
  P <- "ast2ast"
  a_if <- translate(function(x) {
    argtypes(x |> type(double))
    if (x > 0) {
      y <- x * x
    } else {
      y <- -x
    }
    return(y)
  })
  a_while <- translate(function(x) {
    argtypes(x |> type(double))
    y <- x
    while (y < 10.0) y <- 2.0 * y
    return(y)
  })
  a_loop <- translate(function(x) {
    argtypes(x |> type(double))
    y <- x
    for (i in 1L:3L) y <- y * 1.1
    return(y)
  })
  at <- function(f, xs) vapply(xs, function(x) as.numeric(f(x)), numeric(1L))
  chk(P, "if", at(a_if, pts$`if`), note = "C++ if.")
  chk(P, "while", at(a_while, pts$`while`), note = "C++ while.")
  chk(P, "loop", at(a_loop, pts$loop), note = "C++ for.")

  # Reverse mode: deriv(y, x) with y, x vectors (as in the documented example).
  r_if <- translate(function(y, x) {
    argtypes(y |> type(vec(double)), x |> type(vec(double)))
    if (x[[1L]] > 0.0) {
      y[[1L]] <- x[[1L]] * x[[1L]]
    } else {
      y[[1L]] <- -x[[1L]]
    }
    return(deriv(y, x))
  }, derivative = "reverse")
  r_while <- translate(function(y, x) {
    argtypes(y |> type(vec(double)), x |> type(vec(double)))
    z <- x[[1L]]
    while (z < 10.0) z <- 2.0 * z
    y[[1L]] <- z
    return(deriv(y, x))
  }, derivative = "reverse")
  r_loop <- translate(function(y, x) {
    argtypes(y |> type(vec(double)), x |> type(vec(double)))
    z <- x[[1L]]
    for (i in 1L:3L) z <- z * 1.1
    y[[1L]] <- z
    return(deriv(y, x))
  }, derivative = "reverse")
  r_all <- translate(function(y, x) {
    argtypes(y |> type(vec(double)), x |> type(vec(double)))
    z <- x[[1L]]
    for (i in 1L:3L) z <- z * 1.1
    while (z < 10.0) z <- 2.0 * z
    if (z > 0.0) {
      y[[1L]] <- z * z
    } else {
      y[[1L]] <- -z
    }
    return(deriv(y, x))
  }, derivative = "reverse")
  rat <- function(f, xs) vapply(xs, function(x) as.numeric(f(0, x)), numeric(1L))
  chk(P, "grad if", rat(r_if, pts$`if`), note = "reverse mode, deriv(y, x):")
  chk(P, "grad while", rat(r_while, pts$`while`), note = "reverse mode:")
  chk(P, "grad loop", rat(r_loop, pts$loop), note = "reverse mode:")
  chk(P, "grad all", rat(r_all, pts$all), note = "reverse mode:")

  # Forward mode: seed / get_dot through the same control flow.
  f_all <- translate(function(x) {
    argtypes(x |> type(vec(double)))
    seed(x, 1L)
    z <- x[[1L]]
    for (i in 1L:3L) z <- z * 1.1
    while (z < 10.0) z <- 2.0 * z
    if (z > 0.0) {
      z <- z * z
    } else {
      z <- -z
    }
    d <- get_dot(z)
    unseed(x, 1L)
    return(d)
  }, derivative = "forward")
  chk(P, "grad all", vapply(pts$all, function(x) as.numeric(f_all(x)), numeric(1L)), note = "forward mode, seed()/get_dot():")
})

# =================================================================================================
section("odin2", "odin2 / dust2: DSL with `if` expressions; loops only as array equations / time steps", {
  library(odin2)
  library(dust2)
  P <- "odin2/dust2"

  # (a) if-expression on a (differentiable) parameter; the likelihood obs ~ Normal(x, 1) with
  # obs = 0 has gradient d/da [-x^2/2] = -x * dx/da, so dx/da = -gradient / x.
  m_if <- odin(
    {
      a <- parameter(differentiate = TRUE)
      initial(x) <- 0
      update(x) <- if (a > 0) a * a else -a
      obs <- data()
      sd_obs <- parameter(1)
      obs ~ Normal(x, sd_obs)
    },
    quiet = TRUE
  )
  state_at <- function(gen, a, t = 1) {
    sys <- dust_system_create(gen, list(a = a))
    dust_system_set_state_initial(sys)
    dust_system_run_to_time(sys, t)
    as.numeric(dust_system_state(sys))
  }
  chk(P, "if", vapply(pts$`if`, function(a) state_at(m_if, a), numeric(1L)), note = "`if (a > 0) ... else ...` expression in the DSL (C++ ternary).")
  dx_da <- function(gen, a, t) {
    unf <- dust_unfilter_create(gen, time_start = 0, data = data.frame(time = t, obs = 0))
    dust_likelihood_run(unf, list(a = a), adjoint = TRUE)
    -dust_likelihood_last_gradient(unf) / state_at(gen, a, t)
  }
  chk(P, "grad if", vapply(pts$`if`, function(a) dx_da(m_if, a, 1), numeric(1L)), note = "adjoint gradient through the if-expression.")

  r_while <- tryCatch(
    odin(
      {
        a <- parameter()
        initial(x) <- 0
        y <- a
        while (y < 10) y <- 2 * y
        update(x) <- y
      },
      quiet = TRUE
    ),
    error = function(e) e
  )
  rec(P, "while", "--", paste("no loop statements in the DSL, only assignments and relationships:", if (inherits(r_while, "error")) first_lines(conditionMessage(r_while)) else "compiled?!"))
  r_for <- tryCatch(
    odin(
      {
        a <- parameter()
        initial(x) <- 0
        y <- a
        for (i in 1:3) y <- y * 1.1
        update(x) <- y
      },
      quiet = TRUE
    ),
    error = function(e) e
  )
  probe("explicit R `for` loop in the DSL", if (inherits(r_for, "error")) stop(conditionMessage(r_for)) else "compiled")

  # (c) the time-stepping loop: x_{t+1} = 1.1 * x_t from x_0 = a, read off at t = 3.
  m_loop <- odin(
    {
      a <- parameter(differentiate = TRUE)
      initial(x) <- a
      update(x) <- x * 1.1
      obs <- data()
      sd_obs <- parameter(1)
      obs ~ Normal(x, sd_obs)
    },
    quiet = TRUE
  )
  chk(P, "loop", vapply(pts$loop, function(a) state_at(m_loop, a, 3), numeric(1L)), ok_status = "ok*", note = "only as model time steps (update() over t = 1..3); array equations x[i] <- ... are the other loop form.")
  chk(P, "grad loop", vapply(pts$loop, function(a) dx_da(m_loop, a, 3), numeric(1L)), ok_status = "ok*", note = "adjoint through 3 time steps (likelihood gradient only).")
  rec(P, "grad while", "--", "no while loop")
  rec(P, "grad all", "--", "no while loop")
})

# =================================================================================================
section("stan", "Stan (cmdstanr): if / while / for on parameters; AD records the path taken on every evaluation", {
  library(cmdstanr)
  P <- "Stan"
  code <- "
functions {
  real f_if(real x) { if (x > 0) return x^2; else return -x; }
  real f_while(real x) { real y = x; while (y < 10) y = 2 * y; return y; }
  real f_loop(real x) { real y = x; for (i in 1:3) y = y * 1.1; return y; }
}
data { int which; }
parameters { real x; }
model {
  if (which == 1) target += f_if(x);
  if (which == 2) target += f_while(x);
  if (which == 3) target += f_loop(x);
  if (which == 4) target += f_if(f_while(f_loop(x)));
}
"
  dir <- tempfile("cf_stan")
  dir.create(dir)
  mod <- cmdstan_model(write_stan_file(code, dir = dir), compile_model_methods = TRUE, quiet = TRUE)
  fits <- lapply(1:4, function(w) {
    fit <- mod$sample(
      data = list(which = w), chains = 1, iter_sampling = 1, fixed_param = TRUE, init = list(list(x = 2)),
      refresh = 0, show_messages = FALSE, output_dir = dir
    )
    fit$init_model_methods()
    fit
  })
  lp <- function(w, xs) vapply(xs, function(x) as.numeric(fits[[w]]$log_prob(x)), numeric(1L))
  glp <- function(w, xs) vapply(xs, function(x) as.numeric(fits[[w]]$grad_log_prob(x)), numeric(1L))
  chk(P, "if", lp(1, pts$`if`), note = "Stan `if` on a parameter (log_prob()).")
  chk(P, "while", lp(2, pts$`while`), note = "Stan `while` on a parameter.")
  chk(P, "loop", lp(3, pts$loop), note = "Stan `for`.")
  chk(P, "grad if", glp(1, pts$`if`), note = "grad_log_prob(): reverse mode re-records the expression graph per evaluation.")
  chk(P, "grad while", glp(2, pts$`while`))
  chk(P, "grad loop", glp(3, pts$loop))
  chk(P, "grad all", glp(4, pts$all))
})

# =================================================================================================
section("tmb", "TMB: C++ control flow on AD types is recorded once at MakeADFun(); CondExpGt for branches", {
  library(TMB)
  P <- "TMB"
  cpp <- "
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() () {
  DATA_INTEGER(which);
  PARAMETER(x);
  Type y = x;
  if (which == 1) { if (y > Type(0)) y = y * y; else y = -y; }
  if (which == 2) { while (y < Type(10)) y = Type(2) * y; }
  if (which == 3) { for (int i = 0; i < 3; i++) y = y * Type(1.1); }
  if (which == 4) { y = CppAD::CondExpGt(y, Type(0), y * y, -y); }
  if (which == 5) {
    for (int i = 0; i < 3; i++) y = y * Type(1.1);
    while (y < Type(10)) y = Type(2) * y;
    if (y > Type(0)) y = y * y; else y = -y;
  }
  return y;
}
"
  dir <- tempfile("cf_tmb")
  dir.create(dir)
  file <- file.path(dir, "cftmb.cpp")
  writeLines(cpp, file)
  invisible(capture.output(compile(file, flags = "-w")))
  dyn.load(dynlib(file.path(dir, "cftmb")))
  # tape once at the first test point, then evaluate the tape at both
  obj <- function(w, xs) MakeADFun(list(which = w), list(x = xs[[1L]]), DLL = "cftmb", silent = TRUE)
  fn <- function(w, xs) {
    o <- obj(w, xs)
    vapply(xs, o$fn, numeric(1L))
  }
  gr <- function(w, xs) {
    o <- obj(w, xs)
    vapply(xs, function(x) as.numeric(o$gr(x)), numeric(1L))
  }
  chk(P, "if", fn(1, pts$`if`), note = "C++ `if` on Type, taped at x = 2:")
  chk(P, "if", fn(4, pts$`if`), ok_status = "ok*", note = "CppAD::CondExpGt(): both branches on the tape, selected at run time (expressions only, not statements).")
  chk(P, "while", fn(2, pts$`while`), note = "C++ `while` on Type, taped at x = 2:")
  chk(P, "loop", fn(3, pts$loop), ok_status = "ok*", note = "C++ `for` with fixed count, unrolled onto the tape.")
  chk(P, "grad if", gr(1, pts$`if`), note = "C++ `if`, taped at x = 2:")
  chk(P, "grad if", gr(4, pts$`if`), ok_status = "ok*", note = "CondExpGt:")
  chk(P, "grad while", gr(2, pts$`while`), note = "taped at x = 2:")
  chk(P, "grad loop", gr(3, pts$loop), ok_status = "ok*", note = "unrolled:")
  chk(P, "grad all", gr(5, pts$all), note = "taped at x = 2:")
})

# =================================================================================================
section("rtmb", "RTMB: comparisons of AD values forbidden by default; TapeConfig(comparison = \"tape\") + ifelse()/masks for branches; loops unrolled", {
  library(RTMB)
  P <- "RTMB"
  tape_at <- function(f, xs) MakeTape(f, xs[[1L]])
  fn <- function(f, xs) {
    F <- tape_at(f, xs)
    vapply(xs, function(x) as.numeric(F(x)), numeric(1L))
  }
  gr <- function(f, xs) {
    F <- tape_at(f, xs)
    vapply(xs, function(x) as.numeric(F$jacobian(x)), numeric(1L))
  }
  r_if <- function(x) if (x > 0) x^2 else -x
  r_ifelse <- function(x) ifelse(x > 0, x^2, -x)
  r_while <- function(x) {
    while (x < 10) x <- 2 * x
    x
  }
  r_loop <- function(x) {
    for (i in 1:3) x <- x * 1.1
    x
  }
  r_mask <- function(x) (x > 0) * x^2 + (x <= 0) * (-x)
  r_all_if <- function(x) r_if(r_while(r_loop(x)))

  # Default TapeConfig(comparison = "forbid"): every comparison of AD values errors.
  chk(P, "if", fn(r_if, pts$`if`), note = "R `if` on an advector (default comparison = \"forbid\"):")
  chk(P, "if", fn(r_ifelse, pts$`if`), note = "ifelse() on an advector (default \"forbid\"):")
  chk(P, "while", fn(r_while, pts$`while`), note = "R `while` on an advector (default \"forbid\"):")
  chk(P, "loop", fn(r_loop, pts$loop), ok_status = "ok*", note = "R `for` unrolled while taping.")
  chk(P, "grad loop", gr(r_loop, pts$loop), ok_status = "ok*", note = "unrolled:")

  # Documented idiom (?TapeConfig): comparison = "tape" puts indicator functions on the tape, so
  # mask arithmetic f(x) * (x < 0) + g(x) * (x >= 0) selects at run time. A taped comparison is an
  # advector; R's `if` / `while` coerce it to TRUE regardless of its value (it is stored as a
  # complex number), so `if` silently takes the first branch and `while` never terminates -- the
  # loop below is guarded to show this.
  r_if_neg <- function(x) if (x <= 0) -x else x^2 # same function as r_if, branches swapped
  r_while_guarded <- function(x) {
    n <- 0L
    while (x < 10) {
      n <- n + 1L
      if (n > 60L) stop("while loop did not terminate after 60 iterations (condition coerced to TRUE)")
      x <- 2 * x
    }
    x
  }
  TapeConfig(comparison = "tape")
  tryCatch(
    {
      chk(P, "if", fn(r_mask, pts$`if`), ok_status = "ok*", note = "TapeConfig(comparison = \"tape\") + mask arithmetic (x > 0) * x^2 + (x <= 0) * (-x): indicators on the tape, both branches evaluated.")
      chk(P, "if", fn(r_ifelse, pts$`if`), note = "comparison = \"tape\" + ifelse() (its AD method needs a plain logical test):")
      chk(P, "if", fn(r_if_neg, pts$`if`), note = "comparison = \"tape\" + R `if (x <= 0)` (always takes the first branch):")
      chk(P, "while", fn(r_while_guarded, pts$`while`), note = "comparison = \"tape\" + R `while`:")
      chk(P, "grad if", gr(r_mask, pts$`if`), ok_status = "ok*", note = "comparison = \"tape\" + mask arithmetic:")
      chk(P, "grad while", gr(r_while_guarded, pts$`while`), note = "comparison = \"tape\":")
      chk(P, "grad all", gr(function(x) r_mask(r_while_guarded(r_loop(x))), pts$all), note = "comparison = \"tape\", mask if + R while:")
    },
    finally = TapeConfig(comparison = "forbid")
  )

  # comparison = "allow" ("for testing/illustration purposes. Do not use."): comparisons use the
  # values at taping time, so R `if` / `while` run but their path is frozen into the tape.
  TapeConfig(comparison = "allow")
  tryCatch(
    {
      chk(P, "if", fn(r_if, pts$`if`), note = "comparison = \"allow\" + R `if`, taped at x = 2:")
      chk(P, "while", fn(r_while, pts$`while`), note = "comparison = \"allow\" + R `while`, taped at x = 2:")
      chk(P, "grad while", gr(r_while, pts$`while`), note = "comparison = \"allow\", taped at x = 2:")
      chk(P, "grad all", gr(r_all_if, pts$all), note = "comparison = \"allow\", taped at x = 2:")
    },
    finally = TapeConfig(comparison = "forbid")
  )
})

# =================================================================================================
section("nimble", "nimble: if / while / for in nimbleFunctions; derivs() tapes once unless reset = TRUE", {
  suppressPackageStartupMessages(library(nimble))
  P <- "nimble"
  cf <- nimbleFunction(
    setup = function() {},
    run = function() {},
    methods = list(
      f_if = function(x = double(0)) {
        if (x > 0) {
          y <- x * x
        } else {
          y <- -x
        }
        returnType(double(0))
        return(y)
      },
      f_while = function(x = double(0)) {
        y <- x
        while (y < 10) y <- 2 * y
        returnType(double(0))
        return(y)
      },
      f_loop = function(x = double(0)) {
        y <- x
        declare(i, integer()) # see the probe below: needed when derivatives are built
        for (i in 1:3) y <- y * 1.1
        returnType(double(0))
        return(y)
      },
      f_all = function(x = double(0)) {
        y <- x
        declare(i, integer())
        for (i in 1:3) y <- y * 1.1
        while (y < 10) y <- 2 * y
        if (y > 0) {
          y <- y * y
        } else {
          y <- -y
        }
        returnType(double(0))
        return(y)
      },
      d_if = function(x = double(0), reset = logical(0)) {
        returnType(ADNimbleList())
        return(derivs(f_if(x), wrt = 1, order = 1, reset = reset))
      },
      d_while = function(x = double(0), reset = logical(0)) {
        returnType(ADNimbleList())
        return(derivs(f_while(x), wrt = 1, order = 1, reset = reset))
      },
      d_loop = function(x = double(0), reset = logical(0)) {
        returnType(ADNimbleList())
        return(derivs(f_loop(x), wrt = 1, order = 1, reset = reset))
      },
      d_all = function(x = double(0), reset = logical(0)) {
        returnType(ADNimbleList())
        return(derivs(f_all(x), wrt = 1, order = 1, reset = reset))
      }
    ),
    buildDerivs = c("f_if", "f_while", "f_loop", "f_all")
  )
  cc <- compileNimble(cf(), showCompilerOutput = FALSE)
  chk(P, "if", vapply(pts$`if`, cc$f_if, numeric(1L)), note = "C++ if in a compiled nimbleFunction.")
  chk(P, "while", vapply(pts$`while`, cc$f_while, numeric(1L)), note = "C++ while.")
  chk(P, "loop", vapply(pts$loop, cc$f_loop, numeric(1L)), note = "C++ for.")
  d <- function(m, xs, reset) vapply(xs, function(x) as.numeric(m(x, reset)$jacobian), numeric(1L))
  # each method gets its own fresh tape: call order matters, so test reset = FALSE first
  chk(P, "grad if", d(cc$d_if, pts$`if`, FALSE), note = "derivs(reset = FALSE), tape recorded at first call (x = 2):")
  chk(P, "grad if", d(cc$d_if, pts$`if`, TRUE), ok_status = "ok*", note = "derivs(reset = TRUE) re-records the tape each call:")
  chk(P, "grad while", d(cc$d_while, pts$`while`, FALSE), note = "reset = FALSE:")
  chk(P, "grad while", d(cc$d_while, pts$`while`, TRUE), ok_status = "ok*", note = "reset = TRUE:")
  chk(P, "grad loop", d(cc$d_loop, pts$loop, FALSE), note = "fixed count, reset = FALSE:")
  chk(P, "grad all", d(cc$d_all, pts$all, FALSE), note = "reset = FALSE:")
  chk(P, "grad all", d(cc$d_all, pts$all, TRUE), ok_status = "ok*", note = "reset = TRUE:")

  # Test-design pitfall, not a control-flow limitation: in a method with buildDerivs, an
  # undeclared for-loop index is typed as an AD scalar, and the generated C++ does not compile.
  probe("for loop with undeclared index in a buildDerivs method", {
    nf <- nimbleFunction(
      setup = function() {}, run = function() {},
      methods = list(
        g = function(x = double(0)) {
          y <- x
          for (i in 1:3) y <- y * 1.1
          returnType(double(0))
          return(y)
        }
      ),
      buildDerivs = "g"
    )
    compileNimble(nf(), showCompilerOutput = FALSE)
    "compiled"
  })
  probe("  ... the C++ compiler error (undeclared index)", {
    err <- grep("error:", capture.output(printErrors()), value = TRUE)
    sub(".*error: ", "", err[[1L]])
  })
  # The BUGS model language (nimbleCode) is declarative: `if` there is evaluated when the model
  # is built (on constants), not on node values; step() is the indicator idiom.
  probe("nimbleCode() model with `if` on a stochastic node", {
    m <- nimbleModel(nimbleCode({
      x ~ dnorm(0, 1)
      if (x > 0) {
        y <- x^2
      } else {
        y <- -x
      }
    }), inits = list(x = 2))
    m$calculate()
  })
})

# =================================================================================================
cat("\n", strrep("=", 90), "\nRESULTS (best status per package and construct; all attempts listed below)\n", strrep("=", 90), "\n", sep = "")
if (!nrow(results)) stop("no results recorded")
rank <- c("ok" = 1L, "ok*" = 2L, "--" = 3L, "n/a" = 4L)
best <- aggregate(status ~ pkg + construct, results, function(s) s[which.min(rank[s])])
tab <- reshape(best, idvar = "pkg", timevar = "construct", direction = "wide")
names(tab) <- sub("status.", "", names(tab), fixed = TRUE)
tab <- tab[match(unique(results$pkg), tab$pkg), c("pkg", intersect(constructs, names(tab)))]
tab[is.na(tab)] <- ""
print(tab, row.names = FALSE, right = FALSE)
cat("\nAll attempts:\n")
for (i in seq_len(nrow(results))) {
  with(results[i, ], cat(sprintf("%-26s %-11s %-4s %s\n", pkg, construct, status, note)))
}
