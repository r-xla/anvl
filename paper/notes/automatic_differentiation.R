# AD capabilities and restrictions of the packages in the comparison tables.
#
# Personal notes, not part of the paper. Every section uses the same test function
#
#   f(x) = sum(x^3) + x[1] * x[2],   x = (1, 2, 3)
#
# with analytic gradient and Hessian, and checks what each package offers: reverse mode
# (gradient), forward mode (Jacobian-vector product), and higher order (Hessian). Restrictions
# are triggered on purpose and their error messages printed.
#
# Run with `Rscript automatic_differentiation.R`. Needs: anvl, torch, tensorflow + greta (Python
# via reticulate, see below), quickr, cmdstanr (CmdStan), TMB, RTMB, nimble, ast2ast, odin2 +
# dust2 + monty (from https://mrc-ide.r-universe.dev).

x0 <- c(1, 2, 3)
f_value <- sum(x0^3) + x0[1] * x0[2] # 38
f_grad <- 3 * x0^2 + c(x0[2], x0[1], 0) # 5, 13, 27
f_hess <- diag(6 * x0) + matrix(c(0, 1, 0, 1, 0, 0, 0, 0, 0), 3, 3)
v <- c(1, 0, -1) # direction for forward mode: J v = grad . v
f_jvp <- sum(f_grad * v) # -22

section <- function(name) cat("\n", strrep("=", 70), "\n", name, "\n", strrep("=", 70), "\n", sep = "")
ok <- function(what, cond) {
  stopifnot(isTRUE(cond))
  cat("  [ok]", what, "\n")
}
restriction <- function(what, expr) {
  msg <- tryCatch(
    {
      force(expr)
      "no error"
    },
    error = function(e) conditionMessage(e)
  )
  lines <- strsplit(trimws(msg), "\n")[[1L]]
  cat("  [restriction]", what, "\n    ->", paste(head(lines, 3L), collapse = "\n       "), "\n")
}
note <- function(...) cat("  [note]", ..., "\n")

# ------------------------------------------------------------------------------------------------
section("anvl: reverse mode, higher order by nesting; no forward mode")

local({
  library(anvl)
  f <- function(x) sum(x^3) + x[1] * x[2]
  x <- nv_array(x0, dtype = "f64")

  grad_f <- jit(gradient(f))
  ok("reverse mode: gradient()", isTRUE(all.equal(as.numeric(anvl::as_array(grad_f(x)$x)), f_grad)))

  vg <- jit(value_and_gradient(f))(x)
  ok("value_and_gradient()", isTRUE(all.equal(as.numeric(anvl::as_array(vg$value)), f_value)))

  # Hessian: one reverse pass per row, nested inside another gradient(). Written without `[`,
  # because the reverse rule of the scatter that `[` differentiates into has no reverse rule of
  # its own (see the restriction below).
  e <- lapply(1:3, function(i) nv_array(as.numeric(1:3 == i), dtype = "f64"))
  f_noindex <- function(x, e1, e2) sum(x^3) + sum(x * e1) * sum(x * e2)
  hess_row <- jit(function(x, e1, e2, ei) {
    gradient(function(x) sum(gradient(f_noindex, wrt = "x")(x, e1, e2)$x * ei), wrt = "x")(x)$x
  })
  H <- do.call(rbind, lapply(e, function(ei) as.numeric(anvl::as_array(hess_row(x, e[[1]], e[[2]], ei)))))
  ok("higher order: nested gradient() gives the Hessian", isTRUE(all.equal(H, f_hess)))

  note("no forward mode / JVP:", paste(grep("jvp|forward|jacfwd", getNamespaceExports("anvl"), value = TRUE), collapse = ", "), "(empty = none)")

  restriction(
    "second derivative through subsetting `[`",
    jit(function(x) gradient(function(x) gradient(f)(x)$x[1])(x)$x)(x)
  )
  restriction("gradient() of a non-scalar function", jit(gradient(function(x) x^2))(x))
  restriction("gradient() called eagerly (outside jit())", gradient(f)(x))
  restriction(
    "differentiating through nv_scan()",
    jit(gradient(function(x) {
      nv_scan(init = sum(x), body = function(carry, xs) list(carry = carry * 2, out = NULL), steps = 3L)$carry
    }))(x)
  )
  restriction(
    "differentiating through nv_while()",
    jit(gradient(function(x) {
      nv_while(
        init = list(i = nv_scalar(0L), s = sum(x)),
        cond = function(i, s) i < 3L,
        body = function(i, s) list(i = i + 1L, s = s * 2)
      )$s
    }))(x)
  )
})

# ------------------------------------------------------------------------------------------------
section("torch: reverse mode (autograd), higher order via create_graph; no forward mode in R")

local({
  library(torch)
  f <- function(x) torch_sum(x^3) + x[1] * x[2]
  x <- torch_tensor(x0, dtype = torch_float64(), requires_grad = TRUE)

  y <- f(x)
  g <- autograd_grad(y, x, create_graph = TRUE)[[1]]
  ok("reverse mode: autograd_grad() / $backward()", isTRUE(all.equal(as.numeric(g), f_grad)))

  H <- t(sapply(1:3, function(i) as.numeric(autograd_grad(g[i], x, retain_graph = TRUE)[[1]])))
  ok("higher order: grad of grad (create_graph = TRUE) gives the Hessian", isTRUE(all.equal(H, f_hess)))

  note(
    "no forward-mode API in R torch (PyTorch has torch.func.jvp / forward_ad):",
    paste(grep("jvp|dual|forward_ad", getNamespaceExports("torch"), value = TRUE), collapse = ", "), "(empty = none)"
  )
  restriction(
    "gradient of a non-scalar output without grad_outputs",
    autograd_grad(x^2, x)
  )
})

# ------------------------------------------------------------------------------------------------
section("tensorflow: reverse (GradientTape), forward (ForwardAccumulator), higher order")

# greta pins TensorFlow 2.15 / TFP 0.23 / Python 3.10; request them before Python starts so that
# tensorflow and greta share one environment.
reticulate::py_require(c("tensorflow==2.15.*", "tensorflow_probability==0.23.*"), python_version = "3.10")

local({
  library(tensorflow)
  f <- function(x) tf$reduce_sum(x^3) + x[1] * x[2]
  x <- tf$Variable(x0, dtype = tf$float64)

  with(tf$GradientTape() %as% outer, {
    with(tf$GradientTape() %as% inner, {
      y <- f(x)
    })
    g <- inner$gradient(y, x)
  })
  ok("reverse mode: GradientTape", isTRUE(all.equal(as.numeric(g), f_grad)))

  H <- as.matrix(outer$jacobian(g, x))
  ok("higher order: nested tapes give the Hessian", isTRUE(all.equal(H, f_hess)))

  with(tf$autodiff$ForwardAccumulator(x, tf$constant(v, dtype = tf$float64)) %as% acc, {
    y <- f(x)
  })
  ok("forward mode: tf$autodiff$ForwardAccumulator (JVP)", isTRUE(all.equal(as.numeric(acc$jvp(y)), f_jvp)))
})

section("greta: AD is internal (TensorFlow), not exposed to the user")
local({
  library(greta)
  note("greta has no user-level gradient API; its samplers/optimisers use TF gradients internally.")
  note("exports matching grad|deriv:", paste(grep("grad|deriv", getNamespaceExports("greta"), value = TRUE), collapse = ", "))
})

# ------------------------------------------------------------------------------------------------
section("quickr: no AD")

local({
  library(quickr)
  note("exports:", paste(getNamespaceExports("quickr"), collapse = ", "), "-- no derivative functionality")
  restriction(
    "using a derivative function inside quick()",
    quick(function(x) {
      declare(type(x = double(3)))
      D(x)
    })
  )
})

# ------------------------------------------------------------------------------------------------
section("Stan (cmdstanr): reverse mode on the log density; Hessian via model methods; AD not in the language")

local({
  library(cmdstanr)
  code <- "
parameters {
  vector[3] x;
}
model {
  target += sum(x .^ 3) + x[1] * x[2];
}
"
  mod <- cmdstan_model(write_stan_file(code), compile_model_methods = TRUE, quiet = TRUE)
  # Model methods need a fit object; one fixed-parameter draw suffices.
  fit <- mod$sample(
    chains = 1, iter_sampling = 1, fixed_param = TRUE, init = list(list(x = x0)),
    refresh = 0, show_messages = FALSE
  )
  fit$init_model_methods()

  ok("log_prob()", isTRUE(all.equal(as.numeric(fit$log_prob(x0)), f_value)))
  ok("reverse mode: grad_log_prob()", isTRUE(all.equal(as.numeric(fit$grad_log_prob(x0)), f_grad)))
  ok("higher order: hessian()", isTRUE(all.equal(unname(fit$hessian(x0)$hessian), f_hess)))

  note("gradients exist only for the model's log density, w.r.t. the parameters;")
  note("the Stan language has no gradient() function, so users cannot differentiate their own functions")
  note("or use gradients in their own algorithms. Forward mode (fvar) is internal to Stan Math.")
})

# ------------------------------------------------------------------------------------------------
section("TMB: reverse mode, Hessian; objective must be a C++ template")

local({
  library(TMB)
  cpp <- "
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() () {
  PARAMETER_VECTOR(x);
  return (x * x * x).sum() + x(0) * x(1);
}
"
  dir <- tempfile("tmb")
  dir.create(dir)
  file <- file.path(dir, "adtest.cpp")
  writeLines(cpp, file)
  invisible(compile(file, flags = "-w"))
  dyn.load(dynlib(file.path(dir, "adtest")))
  obj <- MakeADFun(list(), list(x = x0), DLL = "adtest", silent = TRUE)

  ok("obj$fn()", isTRUE(all.equal(obj$fn(x0), f_value)))
  ok("reverse mode: obj$gr()", isTRUE(all.equal(as.numeric(obj$gr(x0)), f_grad)))
  ok("higher order: obj$he()", isTRUE(all.equal(unname(obj$he(x0)), f_hess)))
  note("only the objective (scalar) is differentiated; forward mode is used internally (e.g. Laplace).")
})

section("RTMB: reverse mode, Jacobians, higher order via tapes of tapes; no control flow on AD values")

local({
  library(RTMB)
  f <- function(x) sum(x^3) + x[1] * x[2]
  F <- MakeTape(f, x0)
  ok("tape evaluation", isTRUE(all.equal(F(x0), f_value)))
  ok("reverse mode: F$jacobian()", isTRUE(all.equal(as.numeric(F$jacobian(x0)), f_grad)))
  G <- F$jacfun()
  ok("higher order: F$jacfun()$jacobian() gives the Hessian", isTRUE(all.equal(G$jacobian(x0), f_hess)))

  obj <- RTMB::MakeADFun(function(p) f(p$x), list(x = x0), silent = TRUE)
  ok("MakeADFun(): $gr() and $he()", isTRUE(all.equal(as.numeric(obj$gr(x0)), f_grad)) && isTRUE(all.equal(unname(obj$he(x0)), f_hess)))

  restriction("if () on an AD value", MakeTape(function(x) if (x[1] > 0) sum(x) else -sum(x), x0))
  F_rng <- MakeTape(function(x) sum(x) + rnorm(1), x0)
  note("random numbers are frozen on the tape: F(x0) twice =", F_rng(x0), F_rng(x0))
})

# ------------------------------------------------------------------------------------------------
section("nimble: derivatives of order 0-2 via nimDerivs/derivs (CppAD)")

local({
  library(nimble)
  adtest <- nimbleFunction(
    setup = function() {},
    run = function(x = double(1)) {
      returnType(double(0))
      return(sum(x^3) + x[1] * x[2])
    },
    methods = list(
      derivs_run = function(x = double(1)) {
        d <- derivs(run(x), wrt = 1:3, order = 0:2)
        returnType(ADNimbleList())
        return(d)
      }
    ),
    buildDerivs = "run"
  )
  c_adtest <- compileNimble(adtest(), showCompilerOutput = FALSE)
  d <- c_adtest$derivs_run(x0)

  ok("value (order 0)", isTRUE(all.equal(d$value, f_value)))
  ok("reverse mode: gradient (order 1)", isTRUE(all.equal(as.numeric(d$jacobian), f_grad)))
  ok("higher order: Hessian (order 2)", isTRUE(all.equal(matrix(d$hessian, 3, 3), f_hess)))
  note("maximum order is 2; nimble chooses forward/reverse internally, the user only picks the order.")
})

# ------------------------------------------------------------------------------------------------
section("ast2ast: forward (seed/get_dot) and reverse (deriv) mode inside translated functions")

local({
  library(ast2ast)
  fj <- function(y, x) {
    argtypes(y |> type(vec(double)), x |> type(vec(double)))
    y[[1L]] <- x[[1L]]^3 + x[[2L]]^3 + x[[3L]]^3 + x[[1L]] * x[[2L]]
    return(deriv(y, x))
  }
  jac_rev <- translate(fj, derivative = "reverse")
  ok("reverse mode: deriv(y, x) with derivative = \"reverse\"", isTRUE(all.equal(as.numeric(jac_rev(0, x0)), f_grad)))

  # Forward mode is explicit: seed one input component, evaluate, read the tangent with get_dot().
  ff <- function(x) {
    argtypes(x |> type(vec(double)))
    g <- numeric(3L)
    for (i in 1L:3L) {
      seed(x, i)
      y <- x[[1L]]^3 + x[[2L]]^3 + x[[3L]]^3 + x[[1L]] * x[[2L]]
      g[[i]] <- get_dot(y)
      unseed(x, i)
    }
    return(g)
  }
  grad_fwd <- translate(ff, derivative = "forward")
  ok("forward mode: seed() / get_dot() with derivative = \"forward\"", isTRUE(all.equal(grad_fwd(x0), f_grad)))
  restriction("deriv() in forward mode", translate(fj, derivative = "forward"))

  restriction(
    "higher order: deriv() of a deriv()",
    {
      h <- function(y, x) {
        argtypes(y |> type(vec(double)), x |> type(vec(double)))
        y[[1L]] <- x[[1L]]^3 + x[[2L]]^3 + x[[3L]]^3 + x[[1L]] * x[[2L]]
        g <- deriv(y, x)
        return(deriv(g, x))
      }
      H <- translate(h, derivative = "reverse")(0, x0)
      if (!isTRUE(all.equal(unname(as.matrix(H)), f_hess))) stop("result is not the Hessian: ", paste(H, collapse = ", "))
    }
  )
})

# ------------------------------------------------------------------------------------------------
section("odin2 / dust2: reverse mode (adjoint) of the likelihood of deterministic models")

local({
  library(odin2)
  library(dust2)
  sir <- odin({
    initial(S) <- N - I0
    initial(I) <- I0
    initial(R) <- 0
    initial(incidence, zero_every = 1) <- 0
    update(S) <- S - n_SI
    update(I) <- I + n_SI - n_IR
    update(R) <- R + n_IR
    update(incidence) <- incidence + n_SI
    n_SI <- S * p_SI
    n_IR <- I * p_IR
    p_SI <- 1 - exp(-beta * I / N * dt)
    p_IR <- 1 - exp(-gamma * dt)
    beta <- parameter(differentiate = TRUE)
    gamma <- parameter(differentiate = TRUE)
    I0 <- parameter()
    N <- 1000
    cases <- data()
    cases ~ Poisson(incidence)
  }, quiet = TRUE)

  data <- data.frame(time = 1:20, cases = c(3, 2, 2, 2, 1, 3, 2, 5, 5, 6, 9, 6, 8, 10, 12, 15, 17, 14, 20, 25))
  unfilter <- dust_unfilter_create(sir, time_start = 0, data = data)
  pars <- list(beta = 0.25, gamma = 0.1, I0 = 5)

  ll <- function(beta, gamma) dust_likelihood_run(unfilter, list(beta = beta, gamma = gamma, I0 = 5))
  dust_likelihood_run(unfilter, pars, adjoint = TRUE)
  g <- dust_likelihood_last_gradient(unfilter)
  eps <- 1e-6
  g_fd <- c((ll(0.25 + eps, 0.1) - ll(0.25 - eps, 0.1)) / (2 * eps), (ll(0.25, 0.1 + eps) - ll(0.25, 0.1 - eps)) / (2 * eps))
  ok("reverse mode: adjoint gradient of the log-likelihood (vs. finite differences)", isTRUE(all.equal(unname(g), g_fd, tolerance = 1e-5)))

  note("gradients only of the likelihood w.r.t. parameters marked differentiate = TRUE;")
  note("only for deterministic models (unfilter), no forward mode, no higher order.")
  restriction(
    "adjoint gradient with a stochastic particle filter",
    {
      filter <- dust_filter_create(sir, time_start = 0, data = data, n_particles = 10)
      dust_likelihood_run(filter, pars, adjoint = TRUE) # silently runs without the adjoint
      dust_likelihood_last_gradient(filter)
    }
  )
})

cat("\nAll checks passed.\n")
