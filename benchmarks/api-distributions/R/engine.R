## ---------------------------------------------------------------------------
## The sweep engine: enumerate -> score -> reduce.
##
## These three stages are deliberately separate. Enumeration decides *which*
## bit patterns are visited; scoring decides *how far apart* two answers are;
## reduction decides *what is remembered* about billions of samples. Adding a
## metric or a summary must not mean touching the enumerator, and the selftest
## needs to drive each stage on its own.
##
## No distribution, no reference mathematics and no I/O appear here.
## ---------------------------------------------------------------------------

## ---- 1. enumeration --------------------------------------------------------
##
## Both precisions are swept through one abstraction: a pattern index space of
## 2^31 values per sign, walked with a stride. What the index means differs.
##
##   f32  index = the 31 non-sign bits, so stride 1 enumerates every one of the
##        2^32 float32 bit patterns exactly once. One caveat, and only one:
##        widening an f32 to a double quiets a signalling NaN, so the
##        2 x (2^22 - 1) sNaN patterns (0.195% of the space) arrive as their
##        quiet counterparts. Both sides of the comparison receive the same
##        value, so nothing is mismeasured -- the sweep simply cannot tell two
##        NaN payloads apart, which is of no consequence for a distribution
##        function. Every finite value, both zeros and both infinities are
##        delivered exactly as their pattern encodes them.
##
##   f64  index = the high word (sign + 11 exponent bits + top 20 mantissa
##        bits = exactly 32 bits); the low 32 mantissa bits are drawn at random
##        from a fixed seed. Stride 1 therefore takes exactly one sample from
##        each of the 2^32 contiguous blocks of 2^32 patterns: every (sign,
##        exponent, top-20-mantissa) combination is visited, which is the most
##        that is reachable when 2^64 is out of the question.
##
## Depth sets the stride, so a smoke run is the same sweep at coarser spacing
## rather than a different, narrower sweep -- it covers the whole number line,
## Inf and NaN included, and can fail anywhere the full run can.

DEPTHS <- list(
  ## samples = 2 signs * (2^31 / stride)
  smoke = list(stride = 2^13), #       524,288 samples
  quick = list(stride = 2^7), #     33,554,432 samples
  full = list(stride = 1) #  4,294,967,296 samples
)

CHUNK_INDEX <- 2^20 # pattern indices per sign per chunk

## The low mantissa bits of an f64 sample are random; the seed makes a run on
## one machine reproduce on another, which is what lets HPC shards be compared.
SWEEP_SEED <- 1142212L # "anvl" as a=1 .. z=26

sweep_plan <- function(dtype, depth) {
  d <- DEPTHS[[depth]]
  if (is.null(d)) {
    stop("unknown depth '", depth, "'; expected one of ", paste(names(DEPTHS), collapse = ", "), call. = FALSE)
  }
  per_sign <- 2^31 / d$stride
  list(
    dtype = dtype,
    depth = depth,
    stride = d$stride,
    per_sign = per_sign,
    n_chunks = ceiling(per_sign / CHUNK_INDEX),
    n_samples = 2 * per_sign
  )
}

## The k-th chunk of pattern indices for one sign, and the values they encode.
## Returns NULL past the end.
sweep_chunk <- function(plan, k, sign) {
  from <- (k - 1) * CHUNK_INDEX
  if (from >= plan$per_sign) {
    return(NULL)
  }
  n <- min(CHUNK_INDEX, plan$per_sign - from)
  idx <- (from + seq_len(n) - 1) * plan$stride # pattern index, 0 .. 2^31-1

  x <- if (plan$dtype == "f32") {
    f32_from_bits(as.integer(idx))
  } else {
    f64_from_words(as.integer(idx), rand_word32(n))
  }
  ## An exact sign flip: negating the value is the same as setting the sign
  ## bit, for every finite value, both zeros, both infinities and every NaN.
  if (sign < 0) {
    x <- -x
  }

  list(idx = idx, x = x, n = n)
}

## ---- 2. scoring ------------------------------------------------------------
##
## Two metrics over the same comparison, and one fact beside them.
##
##   rel      |f - g| / |g|      -- scale-free, comparable across distributions
##   ulp      |f - g| / ulp(g)   -- the unit accuracy is actually argued in
##   rounded  f is g correctly rounded to the result's precision, but not g
##
## The reference g is base R's double and is never rounded before scoring, so
## rel and ulp measure numerical error against it. `rounded` answers a
## different question: could this precision have done better? For an f32
## result the two part company at the edges of the range. Where |g| reaches
## the f32 overflow threshold (2^128 - 2^103), the rounded answer is +-Inf, and its relative
## error is still infinite; where g is below half the smallest subnormal, the
## correctly rounded answer is 0, and its relative error is still exactly 1.
## Both statements are true and both are kept: rel is not adjusted, and
## `rounded` is recorded alongside it. For f64, g is already a double, so a
## rounded result is an identical one and `rounded` is never set.
##
## The degenerate cases are pinned down identically for rel and ulp, so
## neither is ever NaN and a disagreement can never be scored as agreement:
##
##   f and g bit-identical (incl. both +-Inf, incl. both NaN)  -> 0
##   anything else without a finite score                     -> Inf
##
## "Anything else" is deliberately not a list of cases. An earlier version
## enumerated them -- g zero, g non-finite, f NaN -- and every case it missed
## scored Inf without being flagged: f = +-Inf against a finite g, an f64
## difference overflowing (-1e308 against 1e308), a finite difference over a
## tiny g overflowing. Those samples reached neither the top-K list nor the
## range tracker and could never make a result unexplained. A difference that
## overflows only because both operands are huge is not left at Inf either:
## it is recomputed on halved operands, which is exact at that magnitude.
##
## An Inf score means "no finite error, and they disagree". Those are routed to
## the range tracker rather than the top-K list, because they arrive in huge
## contiguous blocks (every NaN pattern, everything off the support) and would
## otherwise bury every real finding. A correctly rounded result is never
## routed there: it is the best the precision allows.

score_pair <- function(fx, gx, dtype) {
  ok <- (fx == gx) | (is.nan(fx) & is.nan(gx))
  ok[is.na(ok)] <- FALSE

  gr <- if (dtype == "f32") as_f32(gx) else gx
  rounded <- !ok & (fx == gr)
  rounded[is.na(rounded)] <- FALSE

  d <- abs(fx - gx)
  over <- is.infinite(d) & is.finite(fx) & is.finite(gx)
  rel <- d / abs(gx)
  ulp <- d / ulp_size(gx, dtype)
  if (any(over)) {
    dh <- abs(0.5 * fx[over] - 0.5 * gx[over])
    rel[over] <- 2 * (dh / abs(gx[over]))
    ulp[over] <- 2 * (dh / ulp_size(gx[over], dtype))
  }

  bad <- !ok & !rounded & !is.finite(rel)
  rel[ok] <- 0
  ulp[ok] <- 0
  rel[bad] <- Inf
  ulp[bad] <- Inf
  list(rel = rel, ulp = ulp, bad = bad, rounded = rounded)
}

## ---- 2b. what each side returned, and why a failure happened ---------------
##
## A relative error says how far apart two values are, not what they were, and
## a failure region says where the two sides disagreed, not why. Both are
## recorded here, and neither is inferred from the other.
##
## *Kinds.* Each side's value is one of seven kinds. "subnormal" is judged at
## the result's precision, so a double reference below the f32 range is a
## subnormal reference for an f32 cell -- which is exactly the case where an
## f32 result cannot follow it. Kinds are tallied per binade, in and out of the
## valid input domain separately, and per failure region.

KINDS <- c("nan", "+inf", "-inf", "+0", "-0", "subnormal", "normal")
N_KINDS <- length(KINDS)

value_kind <- function(v, dtype) {
  k <- rep.int(7L, length(v))
  a <- abs(v)
  ## normal is by far the common case: screen it out in one comparison and
  ## classify only the rest (NaN fails `>=`, so it lands in the rest)
  r <- which(!((a >= SMALLEST_NORMAL[[dtype]] & a < Inf) %in% TRUE))
  if (length(r)) {
    vr <- v[r]
    kr <- rep.int(6L, length(r))
    kr[vr == 0 & 1 / vr > 0] <- 4L
    kr[vr == 0 & 1 / vr < 0] <- 5L
    kr[vr == Inf] <- 2L
    kr[vr == -Inf] <- 3L
    kr[is.na(vr)] <- 1L
    k[r] <- kr
  }
  k
}

## Agreement down to the sign of zero, or both NaN (a NaN's payload and sign
## are not compared). Stricter than `==`, which cannot tell -0 from +0.
same_value <- function(a, b) {
  s <- (a == b & (a != 0 | 1 / a == 1 / b)) | (is.na(a) & is.na(b))
  s[is.na(s)] <- FALSE
  s
}

## *Causes.* Every failing sample is given exactly one cause, tested rather
## than assumed from where the input lies. In order:
##
##   nan_input                  the input is NaN
##   input_flushing             a subnormal input the backend flushed to zero:
##                              the result is bit-identical to the function's
##                              own result at the same-signed zero, AND that
##                              zero result is validated -- identical to, or
##                              the correctly rounded, reference at zero
##   flush_inherits_zero_error  as above, but the zero result is not validated
##                              (any error at all, finite or not); the
##                              subnormals inherit an error at zero
##   domain_boundary            an endpoint of the valid input domain (e.g.
##                              p = 0 or 1), or a subnormal inheriting the
##                              behaviour of a zero that is one
##   outside_domain             wholly outside the valid input domain
##   zero_input                 the input is +-0 and not a domain boundary
##   inf_input                  the input is +-Inf, inside the domain
##   unidentified               none of the above
##
## Zero is never a subnormal here: it is the value a subnormal is flushed TO,
## and it is checked, not exempted. A cause is a description; whether it counts
## against an accuracy verdict is the category, below.

CAUSES <- c(
  "nan_input",
  "input_flushing",
  "flush_inherits_zero_error",
  "domain_boundary",
  "outside_domain",
  "zero_input",
  "inf_input",
  "unidentified"
)
CAUSE_CATEGORY <- c(
  nan_input = "failure",
  input_flushing = "backend_limitation",
  flush_inherits_zero_error = "failure",
  domain_boundary = "boundary",
  outside_domain = "failure",
  zero_input = "failure",
  inf_input = "failure",
  unidentified = "failure"
)
## outside_domain is a failure where it is established: a value that should
## be NaN and is not. For a *gradient* whose forward values are both NaN it is
## an undefined-domain convention instead -- but a gradient cell never computes
## forward values, so that is settled at export from the matching value cell,
## which swept the identical inputs (see resolve_domain_conventions()).

## Per-cell context for the above: the domain, its finite endpoints at the
## cell's precision, and the function's own behaviour at +-0.
sweep_context <- function(fun, ref, dtype, outputs, domain = c(-Inf, Inf)) {
  z <- c(0, NEG_ZERO) # not the literal -0: see NEG_ZERO in util.R
  context_from(fun(z), ref(z), dtype, outputs, domain)
}

## The same context from results at c(+0, -0) already in hand.
##
## A zero result is *validated* only if it is what the precision allows: equal
## to the reference (a signed-zero disagreement is recorded, not failed) or the
## reference correctly rounded. Nothing weaker will do. Merely "has a finite
## error" -- the first version of this test -- accepted f(0) = 2 against
## g(0) = 1, and every subnormal flushed onto that zero was then excused as a
## backend limitation although it inherited a 100% error. A 1-ulp error at
## zero also fails validation: the subnormals then carry that error as well as
## the flush, and are not the flush's alone.
context_from <- function(f0, g0, dtype, outputs, domain) {
  bounds <- domain[is.finite(domain)]
  if (dtype == "f32") {
    bounds <- as_f32(bounds)
  }
  list(
    dtype = dtype,
    domain = domain,
    boundaries = bounds,
    zero_is_boundary = any(bounds == 0),
    zero = lapply(stats::setNames(outputs, outputs), function(o) {
      s <- score_pair(f0[[o]], g0[[o]], dtype)
      list(
        value = f0[[o]],
        reference = g0[[o]],
        rel_err = s$rel,
        validated = s$rel == 0 | s$rounded
      )
    })
  )
}

## The facts recorded for every sample of one output: kinds, domain
## membership, signed-zero disagreement, flush consistency, and, for failures,
## the cause.
sample_facts <- function(x, fx, gx, s, ctx, o) {
  dtype <- ctx$dtype
  z <- ctx$zero[[o]]
  n <- length(x)
  nanx <- is.na(x)
  ## Most of this concerns small subsets -- subnormal inputs, zero results,
  ## failures -- and is computed on those alone: the whole-chunk passes are the
  ## cost that scales with the sweep.
  ax <- abs(x)
  sub <- which(ax < SMALLEST_NORMAL[[dtype]] & ax > 0) # NaN compares FALSE
  ## which signed zero a subnormal flushes to: +0 for a positive one. Always an
  ## integer index -- ifelse() on a NaN input would yield a *logical* NA, and a
  ## logical NA index recycles, returning both zeros' results instead of one.
  zsub <- ifelse(x[sub] > 0, 1L, 2L)
  flush_same <- logical(n)
  if (length(sub)) {
    flush_same[sub] <- same_value(fx[sub], z$value[zsub])
  }
  in_domain <- if (ctx$domain[1L] == -Inf && ctx$domain[2L] == Inf) {
    !nanx
  } else {
    !nanx & x >= ctx$domain[1L] & x <= ctx$domain[2L]
  }
  kf <- value_kind(fx, dtype)
  kg <- value_kind(gx, dtype)

  cause <- integer(n)
  b <- which(s$bad)
  if (length(b)) {
    cb <- rep.int(8L, length(b))
    xb <- x[b]
    nb <- nanx[b]
    cb[nb] <- 1L
    fl <- flush_same[b] # FALSE unless a subnormal input matched its signed zero
    zf <- !z$validated[ifelse(!nb & xb > 0, 1L, 2L)]
    cb[fl & !zf] <- 2L
    cb[fl & zf] <- if (ctx$zero_is_boundary) 4L else 3L
    rest <- cb == 8L & !nb
    out <- rest & !in_domain[b]
    cb[out] <- 5L
    rest <- rest & !out
    bnd <- rest & xb %in% ctx$boundaries
    cb[bnd] <- 4L
    rest <- rest & !bnd
    cb[rest & xb == 0] <- 6L
    cb[rest & is.infinite(xb)] <- 7L
    cause[b] <- cb
  }

  zero_sign <- logical(n)
  zf0 <- which(fx == 0)
  if (length(zf0)) {
    zz <- zf0[gx[zf0] %in% 0]
    zero_sign[zz] <- 1 / fx[zz] != 1 / gx[zz]
  }
  ## A subnormal whose result is its signed zero's and differs from base R,
  ## split by whether that zero result was validated: only against a validated
  ## zero is the difference the flush's alone.
  flushed <- flushed_zero_error <- logical(n)
  if (length(sub)) {
    fs <- flush_same[sub] & s$rel[sub] != 0 & !s$rounded[sub]
    zv <- z$validated[zsub]
    flushed[sub] <- fs & zv
    flushed_zero_error[sub] <- fs & !zv
  }

  list(
    kf = kf,
    kg = kg,
    pair = (kf - 1L) * N_KINDS + kg,
    in_domain = in_domain,
    zero_sign = zero_sign,
    ## differs from base R, and the difference is the backend's flush: the
    ## result is what the function gives at the flushed input, and that is
    ## the correct result there
    flushed = flushed,
    ## the same flush onto a zero whose own result is not validated
    flushed_zero_error = flushed_zero_error,
    cause = cause
  )
}

## ---- 2c. disputes with the reference ---------------------------------------
##
## base R is the reference, and it is sometimes the weaker implementation:
## punif(q, 0, 1, lower.tail = FALSE, log.p = TRUE) forms 1 - q before the log
## and returns 0 at q = 1e-100, where the answer is log1p(-q) = -1e-100. Every
## error figure stays against base R regardless. A spec may additionally
## declare `ref_stable`, an accurate evaluation of the same function, and each
## sample is then tested against it. A sample is a *candidate* base R dispute
## only if all three hold:
##
##   base R is off        |g - s| > T_R    = 4 ulp_f64(s)   + B
##   anvl is accurate     |f - s| <= T_anvl = 2 ulp_dtype(s) + B
##   anvl is no further   |f - s| <= |g - s|
##
## where s is the stable value and B the stable reference's own declared error
## bound, converted from double ulps at s to absolute units, so both thresholds
## carry the same uncertainty. The multipliers are provisional exclusion
## thresholds, not accuracy criteria: failing them only means a sample stays
## counted against base R.
##
## Where no ulp comparison is meaningful, the rule is exact: if s is +-0 or
## +-Inf, or finite but beyond the f32 range in an f32 cell, anvl must equal s
## at the cell's precision (down to the sign of zero) and base R must not. A
## NaN on any side is never a dispute: that is a domain question, not accuracy.
##
## A candidate is not an exclusion. It becomes one only when the stable
## reference has passed validation under a matching identity (stage two);
## until then the candidate-filtered figures are recorded but never shown as
## anything but candidates.
##
## The same test also finds the opposite case, which disagreement-only
## evaluation would miss: anvl and base R agreeing with each other, and both
## beyond anvl's tolerance against s ("shared").

DISPUTE_K <- c(anvl = 2, base = 4)
F32_OVERFLOW <- 2^128 - 2^103

dispute_facts <- function(fx, gx, sx, dtype, bound_ulp64) {
  ## The ordinary rule, over every sample at once: ulp_size() is NaN for a
  ## non-finite s, so those fall out as FALSE and are settled below.
  da <- abs(fx - sx)
  dr <- abs(gx - sx)
  ## One spacing computation serves both precisions: an f32 spacing is the
  ## double one times 2^29 down to the f32 subnormal spacing, below which it
  ## stays there (checked in selftest against ulp_size(, "f32")).
  u64 <- ulp_size(sx, "f64")
  ud <- if (dtype == "f32") pmax(u64 * 2^29, SUBNORMAL_MIN[["f32"]]) else u64
  b <- bound_ulp64 * u64
  ta <- DISPUTE_K[["anvl"]] * ud + b
  tr <- DISPUTE_K[["base"]] * u64 + b
  a_ok <- da <= ta
  a_ok[is.na(a_ok)] <- FALSE
  cand <- a_ok & dr > tr & da <= dr
  cand[is.na(cand)] <- FALSE

  ## The exact rule, where no ulp comparison is meaningful.
  exact <- sx == 0 | is.infinite(sx)
  if (dtype == "f32") {
    exact <- exact | abs(sx) >= F32_OVERFLOW
  }
  exact <- which(exact) # NA (a NaN s) drops out
  if (length(exact)) {
    target <- if (dtype == "f32") as_f32(sx[exact]) else sx[exact]
    a_ok[exact] <- same_value(fx[exact], target)
    cand[exact] <- a_ok[exact] & !same_value(gx[exact], sx[exact]) & !is.na(gx[exact])
    ta[exact] <- tr[exact] <- NA_real_
  }

  ## Shared: anvl beyond its tolerance, and base R returning the same value.
  ## Usually a small subset, so tested there alone.
  shared <- logical(length(fx))
  j <- which(!a_ok & !is.na(sx) & !is.na(fx))
  if (length(j)) {
    shared[j] <- same_value(fx[j], gx[j])
  }
  list(candidate = cand, shared = shared, d_anvl = da, d_base = dr, t_anvl = ta, t_base = tr)
}

## The identity a validation must match before a candidate may become an
## exclusion. Anything that changes which samples are candidates must change
## it, so it is taken from the code that actually runs, not from a list:
##
##   - the stable reference, the classifier (dispute_facts) and the spacing
##     (ulp_size), each with every function it calls and every value it reads
##     that is not base R's -- followed recursively, so a change to
##     same_value(), as_f32() or a constant such as DISPUTE_K is seen, and so
##     is a constant captured in the stable reference's closure;
##   - a package function by package and version, not by code;
##   - the declared bound, the exact reference parameters (hex), the flags,
##     the dtype, the parameter policy and the R build.
##
## Code is deparsed without source references and with exact (hex) numbers,
## so the identity does not depend on how a session was started.
STABLE_PARAM_POLICY <- "reference parameters at the cell precision (as_f32 for f32 cells)"
BASE_ENVS <- c("base", "stats", "utils", "methods", "graphics", "grDevices")
DEPARSE <- c("keepNA", "keepInteger", "niceNames", "showAttributes", "hexNumeric")

## Calls whose target is chosen at run time -- by name, by environment, by
## dispatch -- so no reading of the code can say what runs. A function that
## uses one cannot have an identity: code_identity() refuses it, and a result
## without an identity can never be validated (fail closed). Base R's own
## functions are not traversed (their behaviour is pinned by the R build), so
## this applies to the harness and spec code a stable reference reaches.
DYNAMIC_CALLS <- c(
  "get",
  "get0",
  "mget",
  "exists",
  "match.fun",
  "do.call",
  "eval",
  "evalq",
  "eval.parent",
  "sys.function",
  "sys.call",
  "parent.frame",
  "environment",
  "environment<-",
  "assign",
  "<<-",
  "library",
  "require",
  "requireNamespace",
  "loadNamespace",
  "attachNamespace",
  "getExportedValue",
  "source",
  "sys.source",
  "body<-",
  "formals<-",
  "UseMethod",
  "NextMethod",
  "standardGeneric",
  "Recall",
  "import"
)

## Functions that take a function argument, where a character string would
## name the function to call -- sapply(x, "get") -- which no reading of the code
## can follow either.
FUN_TAKERS <- c(
  "lapply",
  "sapply",
  "vapply",
  "mapply",
  "Map",
  "Reduce",
  "Filter",
  "Find",
  "Position",
  "apply",
  "tapply",
  "outer",
  "Vectorize",
  "ave",
  "aggregate",
  "rapply",
  "eapply",
  "by"
)

## The calls in an expression, including `pkg::fn` and `pkg:::fn`, which
## codetools reports only as a call to `::`.
calls_in <- function(e) {
  out <- list()
  walk <- function(x) {
    if (is.call(x)) {
      h <- x[[1L]]
      fn <- NULL
      if (is.call(h) && (identical(h[[1L]], as.name("::")) || identical(h[[1L]], as.name(":::")))) {
        fn <- as.character(h[[3L]])
        out[[length(out) + 1L]] <<- list(ns = as.character(h[[2L]]), fn = fn, strings = FALSE)
      } else if (is.name(h)) {
        fn <- as.character(h)
        out[[length(out) + 1L]] <<- list(ns = NA_character_, fn = fn, strings = FALSE)
      }
      ## a string among the arguments of a function-taking call
      if (!is.null(fn) && fn %in% FUN_TAKERS && any(vapply(as.list(x)[-1L], is.character, TRUE))) {
        out[[length(out) + 1L]] <<- list(ns = NA_character_, fn = fn, strings = TRUE)
      }
      for (a in as.list(x)) {
        walk(a)
      }
    } else if (is.function(x)) {
      walk(body(x))
    } else if (is.pairlist(x) || is.list(x)) {
      for (a in x) {
        if (!missing(a)) walk(a)
      }
    }
  }
  walk(e)
  out
}

code_identity <- function(roots) {
  seen <- character(0)
  out <- character(0)
  where <- function(name, env) {
    while (!identical(env, emptyenv())) {
      if (exists(name, envir = env, inherits = FALSE)) {
        return(env)
      }
      env <- parent.env(env)
    }
    NULL
  }
  pkg <- function(ns) {
    v <- tryCatch(as.character(utils::packageVersion(ns)), error = function(e) "not installed")
    sprintf("## package %s %s", ns, v)
  }
  visit <- function(f, name) {
    out <<- c(out, paste0("## function ", name), deparse(f, control = DEPARSE))
    env <- environment(f)
    ## explicit namespace calls, by package and version; and the dynamic
    ## forms that make the code unreadable, refused
    for (cl in c(calls_in(formals(f)), calls_in(body(f)))) {
      if (isTRUE(cl$strings)) {
        stop(
          sprintf("`%s` passes a function name as a string to %s(), which the identity cannot follow", name, cl$fn),
          call. = FALSE
        )
      }
      ## however it is qualified: base::get is get
      if (cl$fn %in% DYNAMIC_CALLS) {
        stop(
          sprintf(
            "`%s` calls %s%s(), which the identity cannot follow",
            name,
            if (is.na(cl$ns)) "" else paste0(cl$ns, "::"),
            cl$fn
          ),
          call. = FALSE
        )
      }
      if (!is.na(cl$ns)) out <<- c(out, paste(pkg(cl$ns), "::", cl$fn))
    }
    g <- codetools::findGlobals(f, merge = FALSE)
    for (v in sort(c(g$functions, g$variables))) {
      if (v %in% c("::", ":::")) {
        next
      }
      e <- where(v, env)
      if (is.null(e)) {
        next
      }
      en <- environmentName(e)
      if (en %in% BASE_ENVS || identical(e, baseenv())) {
        next
      }
      key <- paste(en, format(e), v)
      if (key %in% seen) {
        next
      }
      seen <<- c(seen, key)
      val <- get(v, envir = e, inherits = FALSE)
      if (isNamespace(e) || en %in% loadedNamespaces()) {
        out <<- c(out, paste(pkg(en), "::", v))
      } else if (is.primitive(val)) {
        ## a primitive bound to a name outside base -- `helper <- abs` in a
        ## closure -- is recorded by which primitive it is
        out <<- c(out, sprintf("## primitive %s = %s", v, deparse(val)))
      } else if (is.function(val)) {
        visit(val, v)
      } else {
        out <<- c(out, paste0("## value ", v), deparse(val, control = DEPARSE))
      }
    }
  }
  for (nm in names(roots)) {
    visit(roots[[nm]], nm)
  }
  out
}

hex_params <- function(params) {
  num <- vapply(params, is.numeric, TRUE)
  paste(
    sprintf("%s=%s", names(params)[num], vapply(params[num], function(v) paste(sprintf("%a", v), collapse = "|"), "")),
    collapse = ";"
  )
}

## NA, with the reason as attribute "unsupported", when the code uses a form
## the identity cannot follow: such a reference can never be validated.
reference_identity <- function(roots, extra) {
  code <- tryCatch(
    code_identity(roots),
    error = function(e) structure(NA_character_, unsupported = conditionMessage(e))
  )
  if (length(code) == 1L && is.na(code)) {
    return(code)
  }
  f <- tempfile()
  on.exit(unlink(f))
  writeLines(c(code, extra, STABLE_PARAM_POLICY, R.version.string), f)
  unname(tools::md5sum(f))
}

identity_extra <- function(bound_ulp64, ref_params, flags, dtype) {
  c(
    sprintf("bound_ulp64 = %a", bound_ulp64),
    sprintf("ref_params = %s", hex_params(ref_params)),
    sprintf("flags = %s", paste(names(flags), unlist(flags), sep = "=", collapse = ",")),
    sprintf("dtype = %s", dtype)
  )
}

## A stable reference is identified together with the classifier and spacing
## code that decide candidacy.
stable_identity <- function(fun, bound_ulp64, ref_params = list(), flags = list(), dtype = "") {
  reference_identity(
    list(ref_stable = fun, dispute_facts = dispute_facts, ulp_size = ulp_size),
    identity_extra(bound_ulp64, ref_params, flags, dtype)
  )
}

## A gradient reference by its own code and declared bound: validating it
## gates no exclusion, but says whether it is trustworthy.
grad_identity <- function(fun, bound_ulp64, ref_params = list(), flags = list(), dtype = "") {
  reference_identity(list(ref_grad = fun), identity_extra(bound_ulp64 %||% NA_real_, ref_params, flags, dtype))
}

## The inverse of hex_params(): the exact parameter values back as doubles.
parse_hex_params <- function(s) {
  if (is.na(s) || !nzchar(s)) {
    return(list())
  }
  kv <- strsplit(strsplit(s, ";", fixed = TRUE)[[1L]], "=", fixed = TRUE)
  out <- lapply(kv, function(p) as.numeric(strsplit(p[2L], "|", fixed = TRUE)[[1L]]))
  names(out) <- vapply(kv, `[`, "", 1L)
  if (!identical(hex_params(out), s)) {
    stop("stored parameters do not round-trip: ", s, call. = FALSE)
  }
  out
}

## ---- 3. reducers -----------------------------------------------------------
##
## Each is a stateful accumulator fed one chunk at a time and drained once at
## the end. A sweep visits billions of samples and can keep only what these
## remember, so anything not reduced here is gone: that is why the histogram
## exists alongside the top-K. The top-K bounds the worst case; the histogram
## says whether the worst case is one pathological input or a systemic floor,
## and there is no way to recover it after the fact.

## -- 3a. the worst inputs, K per binade --
##
## Per binade rather than a single global list. A global top-K clusters: for
## nv_qunif f32 every one of the worst 1000 sat around x = 1/3, so the worst
## input anywhere in the tail had been computed and thrown away. Keeping K in
## each binade guarantees coverage across the whole number line and makes the
## behaviour bands drillable -- a band says how bad, these say exactly where.
##
## K is small (10) because the cost is multiplied by the number of occupied
## binades, not divided by it: per-binade top-100 would be roughly twelve times
## today's store, per-binade top-10 is about the same size.
##
## The overall worst input is still simply the first row, because the retained
## entries are ranked globally on the way out.
reducer_topk <- function(dtype, k = 10L) {
  span <- if (dtype == "f32") 2^23 else 2^20
  ## one slot per (sign, binade); an environment keyed by name keeps the
  ## occupied ones only, which is most of the saving on a well-behaved cell
  store <- new.env(parent = emptyenv())
  cut <- new.env(parent = emptyenv())

  list(
    add = function(ch, s, fx, gx, sgn) {
      i <- which(is.finite(s$rel) & s$rel > 0)
      if (!length(i)) {
        return(invisible(NULL))
      }
      ## Zero has a shortlist of its own, as it has a band of its own: sharing
      ## binade 0's would let ten subnormals displace it. Its slot is keyed -1
      ## and stored as binade 0; `x` itself says which it is.
      b <- floor(ch$idx[i] / span)
      b[ch$x[i] %in% 0] <- -1
      tag <- if (sgn > 0) "p" else "n"
      for (bb in unique(b)) {
        key <- paste0(tag, bb)
        j <- i[b == bb]
        ## Drop anything that cannot displace the current k-th worst before
        ## sorting: at full depth a chunk holds a million candidates and a full
        ## sort of every one of them, every chunk, is the whole cost.
        c0 <- cut[[key]]
        if (!is.null(c0)) {
          j <- j[s$rel[j] > c0]
        }
        if (!length(j)) {
          next
        }
        if (length(j) > k) {
          j <- j[order(s$rel[j], decreasing = TRUE)[seq_len(k)]]
        }
        cand <- data.frame(
          binade = max(bb, 0),
          sign = sgn,
          x = ch$x[j],
          rel_err = s$rel[j],
          ulp_err = s$ulp[j],
          value = fx[j],
          reference = gx[j],
          rounded = s$rounded[j],
          flushed = s$flushed[j]
        )
        old <- store[[key]]
        all <- if (is.null(old)) cand else rbind(old, cand)
        all <- all[order(all$rel_err, decreasing = TRUE), , drop = FALSE]
        all <- utils::head(all, k)
        store[[key]] <- all
        if (nrow(all) == k) cut[[key]] <- all$rel_err[k]
      }
      invisible(NULL)
    },
    get = function() {
      keys <- ls(store)
      if (!length(keys)) {
        return(data.frame(
          rank = integer(0),
          binade = numeric(0),
          sign = numeric(0),
          bits = character(0),
          x = numeric(0),
          value = numeric(0),
          reference = numeric(0),
          rel_err = numeric(0),
          ulp_err = numeric(0),
          rounded = logical(0),
          flushed = logical(0)
        ))
      }
      d <- do.call(rbind, mget(keys, envir = store))
      d <- d[order(d$rel_err, decreasing = TRUE), , drop = FALSE]
      data.frame(
        rank = seq_len(nrow(d)),
        binade = d$binade,
        sign = d$sign,
        bits = bits_of(d$x, dtype),
        x = d$x,
        value = d$value,
        reference = d$reference,
        rel_err = d$rel_err,
        ulp_err = d$ulp_err,
        rounded = d$rounded,
        flushed = d$flushed
      )
    }
  )
}

## -- 3b. Inf-score patterns, collapsed to contiguous [lo, hi] index runs --
## Indices are walked in order, so a run of failures is a maximal stretch of
## consecutive indices; a run touching the previous chunk's tail extends it
## rather than starting a new one. One tracker per sign, so a run never
## straddles the +/- halves of the number line.
reducer_runs <- function(stride) {
  ## One entry per region: its index bounds, cause, sample count, the first and
  ## last failing inputs actually evaluated, one representative sample, and a
  ## tally of what the two sides returned.
  lo <- hi <- n <- x_first <- x_last <- rep_x <- rep_f <- rep_g <- numeric(0)
  cause <- integer(0)
  pairs <- list()
  list(
    ## `cause_code` may carry a candidate-dispute flag as +100, so a region
    ## never mixes candidate and non-candidate samples.
    add = function(idx, bad, cause_code, x, fx, gx, pair) {
      i <- which(bad)
      if (!length(i)) {
        return(invisible(NULL))
      }
      ## A region is a maximal stretch of consecutive samples with one cause.
      ## Causes follow the input's location and the flush test, which is
      ## deterministic per sign, so this cannot fragment the way splitting on
      ## returned values could.
      cc <- cause_code[i]
      st <- which(c(TRUE, diff(i) != 1 | diff(cc) != 0))
      en <- c(st[-1] - 1L, length(i))
      for (q in seq_along(st)) {
        a <- i[st[q]]
        b <- i[en[q]]
        seg <- i[st[q]:en[q]]
        ## a double: one region can exceed 2^31 samples in a full f32 sweep
        tally <- as.numeric(tabulate(pair[seg], nbins = N_KINDS^2))
        m <- length(lo)
        ## contiguous with the previous chunk's last region, with the same cause
        if (m && idx[a] == hi[m] + stride && cc[st[q]] == cause[m]) {
          hi[m] <<- idx[b]
          n[m] <<- n[m] + length(seg)
          x_last[m] <<- x[b]
          pairs[[m]] <<- pairs[[m]] + tally
          next
        }
        lo <<- c(lo, idx[a])
        hi <<- c(hi, idx[b])
        n <<- c(n, length(seg))
        cause <<- c(cause, cc[st[q]])
        x_first <<- c(x_first, x[a])
        x_last <<- c(x_last, x[b])
        rep_x <<- c(rep_x, x[a])
        rep_f <<- c(rep_f, fx[a])
        rep_g <<- c(rep_g, gx[a])
        pairs[[m + 1L]] <<- tally
      }
    },
    get = function() {
      describe <- function(t) {
        j <- which(t > 0)
        j <- j[order(-t[j])]
        paste(
          sprintf(
            "%s vs %s: %s",
            KINDS[(j - 1L) %/% N_KINDS + 1L],
            KINDS[(j - 1L) %% N_KINDS + 1L],
            format(t[j], big.mark = ",", scientific = FALSE, trim = TRUE)
          ),
          collapse = "; "
        )
      }
      top <- function(t) which.max(t)
      data.frame(
        lo = lo,
        hi = hi,
        cause = CAUSES[cause %% 100L],
        ref_candidate = cause >= 100L,
        n_failing = n,
        x_first = x_first,
        x_last = x_last,
        rep_x = rep_x,
        rep_value = rep_f,
        rep_reference = rep_g,
        value_kind = KINDS[(vapply(pairs, top, 1L) - 1L) %/% N_KINDS + 1L],
        reference_kind = KINDS[(vapply(pairs, top, 1L) - 1L) %% N_KINDS + 1L],
        pairs = vapply(pairs, describe, ""),
        stringsAsFactors = FALSE
      )
    }
  )
}

## -- 3b'. the evidence for each dispute, K per binade and kind --
## Candidate disputes are kept by how far base R is beyond its tolerance
## (d_base / t_base), shared disagreements by how far anvl is beyond its own --
## ratios, because an absolute distance ranks by the output's magnitude and
## would bury a base R 0 against -1e-100. An exact-rule case has no tolerance
## and ranks first. each with all three values, both
## distances and both thresholds, so every exclusion can be inspected sample by
## sample. Zero has its own slot, as everywhere else.
reducer_disputes <- function(dtype, k = 10L) {
  span <- if (dtype == "f32") 2^23 else 2^20
  store <- new.env(parent = emptyenv())
  cut <- new.env(parent = emptyenv())
  keep <- function(kind, i, by, ch, fx, gx, sx, d, sgn) {
    if (!length(i)) {
      return()
    }
    ## indices arrive in order, so each binade's samples are one contiguous
    ## slice; zero, keyed -1, is the first of its chunk
    b <- floor(ch$idx[i] / span)
    b[ch$x[i] %in% 0] <- -1
    g <- group_slices(b)
    for (q in seq_along(g$key)) {
      j <- i[g$from[q]:g$to[q]]
      key <- paste(kind, sgn, g$key[q])
      c0 <- cut[[key]]
      if (!is.null(c0)) {
        j <- j[by[j] > c0]
      }
      if (!length(j)) {
        next
      }
      if (length(j) > k) {
        j <- j[order(by[j], decreasing = TRUE)[seq_len(k)]]
      }
      cand <- data.frame(
        kind = kind,
        sign = sgn,
        binade = max(g$key[q], 0),
        x = ch$x[j],
        value = fx[j],
        reference = gx[j],
        stable = sx[j],
        d_anvl = d$d_anvl[j],
        d_base = d$d_base[j],
        t_anvl = d$t_anvl[j],
        t_base = d$t_base[j],
        by = by[j]
      )
      all <- rbind(store[[key]], cand)
      all <- utils::head(all[order(all$by, decreasing = TRUE), , drop = FALSE], k)
      store[[key]] <- all
      if (nrow(all) == k) cut[[key]] <- all$by[k]
    }
  }
  list(
    add = function(ch, d, fx, gx, sx, sgn) {
      over <- function(dist, tol) {
        r <- dist / tol
        r[is.na(r)] <- Inf
        r
      }
      i <- which(d$candidate)
      if (length(i)) {
        keep("candidate", i, over(d$d_base, d$t_base), ch, fx, gx, sx, d, sgn)
      }
      i <- which(d$shared)
      if (length(i)) keep("shared", i, over(d$d_anvl, d$t_anvl), ch, fx, gx, sx, d, sgn)
    },
    get = function() {
      keys <- ls(store)
      if (!length(keys)) {
        return(NULL)
      }
      out <- do.call(rbind, mget(keys, envir = store))
      names(out)[names(out) == "by"] <- "beyond_tolerance"
      out$bits <- bits_of(out$x, dtype)
      rownames(out) <- NULL
      out[order(out$kind, -out$beyond_tolerance), , drop = FALSE]
    }
  )
}

## -- 3c. behaviour across the number line, by binade --
##
## Where does the function stop reproducing base R exactly, and where does it
## stop producing a finite answer at all? Those transitions happen in ranges,
## not at scattered points, and a histogram of error magnitudes cannot show
## them.
##
## Accumulated per binade rather than per bit pattern. Tracking runs of
## individual patterns would explode: at f32 epsilon, exact and differing
## values interleave constantly, so a well-behaved cell would produce hundreds
## of thousands of alternating one-element runs. A binade is the scale at which
## the behaviour actually changes, and there are only 256 of them in f32 and
## 2048 in f64, per sign.
## Contiguous groups of equal values in `key` (already in order), as slices.
group_slices <- function(key) {
  r <- rle(key)
  to <- cumsum(r$lengths)
  list(key = r$values, from = to - r$lengths + 1L, to = to)
}

## The position of each group's extreme value (which.max / which.min).
group_extreme <- function(g, v, pick) {
  vapply(seq_along(g$key), function(q) g$from[q] - 1L + pick(v[g$from[q]:g$to[q]]), 1L)
}

## Exact zero inputs are not a subnormal binade's samples: zero is the value a
## subnormal is flushed *to*, and its result is checked, not excused. Binade 0
## therefore holds the subnormals alone, and each sign's zero samples are kept
## in one extra slot (`nexp + 1`), emitted as a band row with `zero = TRUE`.
## The bands still partition the samples, so every sum over them is unchanged.
reducer_bands <- function(dtype) {
  span <- if (dtype == "f32") 2^23 else 2^20 # pattern indices per binade
  nexp <- if (dtype == "f32") 256L else 2048L
  nslot <- nexp + 1L
  ## `m` is the decade of the relative error: floor(-log10(e)), so an error e
  ## satisfies 10^-(m+1) <= e < 10^-m. A *larger* error has a *smaller* m, so
  ## m_worst is the minimum over a binade and m_best the maximum. Both stay NA
  ## until the binade sees a finite non-zero error: a wholly exact binade has
  ## no envelope and must not pretend to one.
  mk <- function() {
    list(
      n = matrix(0, nslot, 3L),
      ## correctly rounded but not identical; overlaps columns 2 and 3 of `n`
      rounded = rep(0, nslot),
      ## both sides zero, of opposite sign: counted as identical by `n`, which
      ## compares with ==, and recorded here so it stays visible
      zero_sign = rep(0, nslot),
      ## differs from base R because the backend flushed a subnormal input to a
      ## zero whose result is validated
      flushed = rep(0, nslot),
      ## the same, onto a zero whose result is not validated
      flushed_zero_error = rep(0, nslot),
      ## samples whose reference is a normal float at the result's precision:
      ## the ones where a small relative error is the right expectation
      out_normal = rep(0, nslot),
      out_normal_identical = rep(0, nslot),
      out_normal_worst = rep(0, nslot),
      out_normal_x = rep(NA_real_, nslot),
      out_normal_value = rep(NA_real_, nslot),
      out_normal_reference = rep(NA_real_, nslot),
      ## (value kind, reference kind) pairs, in and out of the domain, by the
      ## input's true binade (zero inputs in binade 0):
      ## nexp x 2 x N_KINDS^2, flattened
      kinds = numeric(nexp * 2L * N_KINDS^2),
      worst = rep(0, nslot),
      ## the worst ulp error, accumulated in its own right: the worst relative
      ## error and the worst ulp error need not be the same sample (|g| varies
      ## two-fold within a binade while its spacing does not), so the ulp
      ## maximum cannot be read off a shortlist ranked by relative error
      worst_ulp = rep(0, nslot),
      m_worst = rep(NA_real_, nslot),
      m_best = rep(NA_real_, nslot),
      ## candidate base R disputes (see dispute_facts()), and the figures with
      ## them left out; recorded only where the spec has a stable reference
      ref_candidate = rep(0, nslot),
      ref_candidate_nonfinite = rep(0, nslot),
      ref_shared = rep(0, nslot),
      worst_excl = rep(0, nslot),
      out_normal_worst_excl = rep(0, nslot),
      out_normal_excl_x = rep(NA_real_, nslot),
      out_normal_excl_value = rep(NA_real_, nslot),
      out_normal_excl_reference = rep(NA_real_, nslot)
    )
  }
  acc <- list(mk(), mk()) # [[1]] positive, [[2]] negative

  list(
    add = function(idx, s, sign, x, fx, gx) {
      k <- if (sign > 0) 1L else 2L
      a <- acc[[k]]
      eb <- floor(idx / span) + 1L # the true binade, for the kinds
      e <- eb
      ## Zero is the lowest pattern index of its sign, so its slot is a
      ## contiguous group of its own at the start of a chunk.
      e[which(x == 0)] <- nslot
      ## A chunk's pattern indices increase, so its samples arrive grouped by
      ## binade and in order: every per-binade reduction below is a pass over
      ## contiguous slices, not a split() of the whole chunk. At quick depth an
      ## f64 chunk spans ~129 binades; the old split()-based version was the
      ## single most expensive step of a sweep.
      tab <- function(i) tabulate(e[i], nbins = nslot)
      rel <- s$rel
      ## 1 = identical, 2 = finite difference, 3 = no finite error
      fin <- is.finite(rel)
      same <- rel == 0
      a$n[, 1L] <- a$n[, 1L] + tab(same)
      a$n[, 2L] <- a$n[, 2L] + tab(fin & !same)
      a$n[, 3L] <- a$n[, 3L] + tab(!fin)
      if (any(s$rounded)) {
        a$rounded <- a$rounded + tab(s$rounded)
      }
      if (any(s$zero_sign)) {
        a$zero_sign <- a$zero_sign + tab(s$zero_sign)
      }
      if (any(s$flushed)) {
        a$flushed <- a$flushed + tab(s$flushed)
      }
      if (any(s$flushed_zero_error)) {
        a$flushed_zero_error <- a$flushed_zero_error + tab(s$flushed_zero_error)
      }

      ## Kinds: the bulk -- normal against normal inside the domain -- is never
      ## reported, so only the rest is tallied.
      rest <- which(s$pair != N_KINDS^2 | !s$in_domain)
      if (length(rest)) {
        code <- (eb[rest] - 1L) * (2L * N_KINDS^2) + (!s$in_domain[rest]) * N_KINDS^2 + s$pair[rest]
        a$kinds <- a$kinds + tabulate(code, nbins = nexp * 2L * N_KINDS^2)
      }

      ## Worst and best error per binade, and the decade envelope, which
      ## follows from them: m = floor(-log10(e)) is monotone in e, so the
      ## smallest m comes from the largest error and the largest m from the
      ## smallest -- no log10 over every sample.
      pos <- which(fin & !same)
      if (length(pos)) {
        g <- group_slices(e[pos])
        rp <- rel[pos]
        hi <- group_extreme(g, rp, which.max)
        lo <- group_extreme(g, rp, which.min)
        j <- g$key
        a$worst[j] <- pmax(a$worst[j], rp[hi])
        up <- s$ulp[pos]
        a$worst_ulp[j] <- pmax(a$worst_ulp[j], up[group_extreme(g, up, which.max)])
        a$m_worst[j] <- pmin(a$m_worst[j], floor(-log10(rp[hi])), na.rm = TRUE)
        a$m_best[j] <- pmax(a$m_best[j], floor(-log10(rp[lo])), na.rm = TRUE)
      }

      ## The candidate-filtered figures: the same maxima over the samples that
      ## are not candidate disputes. Only worth a pass when there are any.
      if (any(s$ref_candidate)) {
        a$ref_candidate <- a$ref_candidate + tab(s$ref_candidate)
        a$ref_candidate_nonfinite <- a$ref_candidate_nonfinite + tab(s$ref_candidate & !fin)
        px <- which(fin & !same & !s$ref_candidate)
        if (length(px)) {
          g <- group_slices(e[px])
          a$worst_excl[g$key] <- pmax(a$worst_excl[g$key], rel[px[group_extreme(g, rel[px], which.max)]])
        }
      } else if (length(pos)) {
        g <- group_slices(e[pos])
        a$worst_excl[g$key] <- pmax(a$worst_excl[g$key], rel[pos][group_extreme(g, rel[pos], which.max)])
      }
      if (any(s$ref_shared)) {
        a$ref_shared <- a$ref_shared + tab(s$ref_shared)
      }

      ## Samples whose reference is a normal float: counts, and the worst
      ## with the sample that produced it.
      on <- s$kg == 7L
      if (any(on)) {
        a$out_normal <- a$out_normal + tab(on)
        a$out_normal_identical <- a$out_normal_identical + tab(on & same)
        fo <- which(on & fin & !same)
        if (length(fo)) {
          g <- group_slices(e[fo])
          w <- fo[group_extreme(g, rel[fo], which.max)]
          j <- g$key
          better <- rel[w] > a$out_normal_worst[j]
          if (any(better)) {
            w <- w[better]
            j <- j[better]
            a$out_normal_worst[j] <- rel[w]
            a$out_normal_x[j] <- x[w]
            a$out_normal_value[j] <- fx[w]
            a$out_normal_reference[j] <- gx[w]
          }
          fe <- if (any(s$ref_candidate)) fo[!s$ref_candidate[fo]] else fo
          if (length(fe)) {
            g <- group_slices(e[fe])
            w <- fe[group_extreme(g, rel[fe], which.max)]
            j <- g$key
            better <- rel[w] > a$out_normal_worst_excl[j]
            if (any(better)) {
              w <- w[better]
              j <- j[better]
              a$out_normal_worst_excl[j] <- rel[w]
              a$out_normal_excl_x[j] <- x[w]
              a$out_normal_excl_value[j] <- fx[w]
              a$out_normal_excl_reference[j] <- gx[w]
            }
          }
        }
      }
      acc[[k]] <<- a
    },
    get = function() acc,
    span = function() span
  )
}

## Merge adjacent binades that behave the same way into one reported range.
## A band's label is what makes two binades "the same": a run of binades that
## are all bit-identical collapses to a single line, and the line where that
## stops is the finding.
band_label <- function(n) {
  tot <- sum(n)
  if (tot == 0) {
    return(NA_character_)
  }
  if (n[3L] == tot) {
    return("no finite error")
  }
  if (n[1L] == tot) {
    return("all identical")
  }
  if (n[2L] == tot) {
    return("all differ")
  }
  if (n[3L] == 0) "identical + differ" else "mixed, some non-finite"
}

## The per-binade profile, unmerged: one row for every binade that was
## sampled, carrying its behaviour, its error envelope and its worst error.
##
## Merging is left to the reader (merge_bands() in query.R). A compact view
## wants a dozen rows and a chart wants all 256 (f32) or 2048 (f64) per sign,
## and deriving the compact view from the full one keeps them consistent;
## storing only the merged form would make the chart impossible without
## re-running the sweep.
binade_profile <- function(bands, dtype) {
  span <- bands$span()
  acc <- bands$get()
  nslot <- nrow(acc[[1L]]$n)
  nexp <- nslot - 1L

  out <- lapply(seq_along(acc), function(k) {
    a <- acc[[k]]
    sgn <- if (k == 1L) 1 else -1
    lab <- vapply(seq_len(nslot), function(i) band_label(a$n[i, ]), "")
    ix <- which(!is.na(lab))
    if (!length(ix)) {
      return(NULL)
    }

    ## The highest exponent field is not an interval: it holds +-Inf (zero
    ## mantissa) and every NaN (non-zero mantissa) side by side, so decoding
    ## its upper edge yields NaN. It is flagged and given no numeric range.
    special <- ix == nexp
    ## The zero slot is exponent field 0 too, and covers the one value +-0.
    zero <- ix == nslot
    bin <- ifelse(zero, 1L, ix)
    lo <- (bin - 1L) * span
    hi <- bin * span - 1
    ends <- if (dtype == "f32") {
      cbind(sgn * f32_from_bits(as.integer(lo)), sgn * f32_from_bits(as.integer(hi)))
    } else {
      cbind(
        sgn * f64_from_words(as.integer(lo), 0L),
        sgn * f64_from_words(as.integer(hi), -1L)
      )
    }
    x_from <- pmin(ends[, 1L], ends[, 2L])
    x_to <- pmax(ends[, 1L], ends[, 2L])
    x_from[special] <- NA_real_
    x_to[special] <- NA_real_
    x_from[zero] <- x_to[zero] <- if (sgn > 0) 0 else NEG_ZERO
    ## binade 0 holds only the subnormals, so its edge nearest zero is the
    ## smallest subnormal, not zero
    sub <- !zero & bin == 1L
    if (sgn > 0) {
      x_from[sub] <- SUBNORMAL_MIN[[dtype]]
    } else {
      x_to[sub] <- -SUBNORMAL_MIN[[dtype]]
    }

    data.frame(
      sign = sgn,
      binade = bin - 1L,
      x_from = x_from,
      x_to = x_to,
      special = special,
      zero = zero,
      behaviour = lab[ix],
      n_identical = a$n[ix, 1L],
      n_differ = a$n[ix, 2L],
      n_nonfinite = a$n[ix, 3L],
      n_rounded = a$rounded[ix],
      n_zero_sign = a$zero_sign[ix],
      n_flushed = a$flushed[ix],
      n_flushed_zero_error = a$flushed_zero_error[ix],
      n_out_normal = a$out_normal[ix],
      n_out_normal_identical = a$out_normal_identical[ix],
      worst_out_normal = a$out_normal_worst[ix],
      worst_out_normal_x = a$out_normal_x[ix],
      worst_out_normal_value = a$out_normal_value[ix],
      worst_out_normal_reference = a$out_normal_reference[ix],
      n_ref_candidate = a$ref_candidate[ix],
      n_ref_candidate_nonfinite = a$ref_candidate_nonfinite[ix],
      n_ref_shared = a$ref_shared[ix],
      worst_rel_err_excl = a$worst_excl[ix],
      worst_out_normal_excl = a$out_normal_worst_excl[ix],
      worst_out_normal_excl_x = a$out_normal_excl_x[ix],
      worst_out_normal_excl_value = a$out_normal_excl_value[ix],
      worst_out_normal_excl_reference = a$out_normal_excl_reference[ix],
      m_worst = a$m_worst[ix],
      m_best = a$m_best[ix],
      worst_rel_err = a$worst[ix],
      worst_ulp_err = a$worst_ulp[ix]
    )
  })

  out <- do.call(rbind, out)
  if (is.null(out)) {
    return(data.frame(
      sign = numeric(0),
      binade = integer(0),
      x_from = numeric(0),
      x_to = numeric(0),
      special = logical(0),
      zero = logical(0),
      behaviour = character(0),
      n_identical = numeric(0),
      n_differ = numeric(0),
      n_nonfinite = numeric(0),
      n_rounded = numeric(0),
      n_zero_sign = numeric(0),
      n_flushed = numeric(0),
      n_flushed_zero_error = numeric(0),
      n_out_normal = numeric(0),
      n_out_normal_identical = numeric(0),
      worst_out_normal = numeric(0),
      worst_out_normal_x = numeric(0),
      worst_out_normal_value = numeric(0),
      worst_out_normal_reference = numeric(0),
      n_ref_candidate = numeric(0),
      n_ref_candidate_nonfinite = numeric(0),
      n_ref_shared = numeric(0),
      worst_rel_err_excl = numeric(0),
      worst_out_normal_excl = numeric(0),
      worst_out_normal_excl_x = numeric(0),
      worst_out_normal_excl_value = numeric(0),
      worst_out_normal_excl_reference = numeric(0),
      m_worst = numeric(0),
      m_best = numeric(0),
      worst_rel_err = numeric(0),
      worst_ulp_err = numeric(0)
    ))
  }
  out[order(out$special, out$sign, out$binade, !out$zero), , drop = FALSE]
}

## The (value kind, reference kind) tallies as a sparse table: one row per
## binade, domain side and pair that occurred. The bulk -- normal against
## normal inside the domain -- is left out; it is the complement of the rest.
kinds_table <- function(bands) {
  acc <- bands$get()
  nexp <- nrow(acc[[1L]]$n) - 1L # less the zero slot
  per <- 2L * N_KINDS^2
  out <- lapply(seq_along(acc), function(k) {
    v <- acc[[k]]$kinds
    i <- which(v > 0)
    if (!length(i)) {
      return(NULL)
    }
    r <- (i - 1L) %% per
    d <- data.frame(
      sign = if (k == 1L) 1 else -1,
      binade = (i - 1L) %/% per,
      in_domain = r < N_KINDS^2,
      value_kind = KINDS[((r %% N_KINDS^2) %/% N_KINDS) + 1L],
      reference_kind = KINDS[(r %% N_KINDS) + 1L],
      n = v[i]
    )
    d[!(d$in_domain & d$value_kind == "normal" & d$reference_kind == "normal"), , drop = FALSE]
  })
  out <- do.call(rbind, out)
  if (is.null(out)) {
    out <- data.frame(
      sign = numeric(0),
      binade = numeric(0),
      in_domain = logical(0),
      value_kind = character(0),
      reference_kind = character(0),
      n = numeric(0)
    )
  }
  out
}

## -- 3d. error distribution: counts per decade of relative error --
## Bin j holds the samples with rel_err in [10^j, 10^(j+1)). Exact agreement
## and Inf are counted separately, so the three totals always reconstruct the
## sample count.
HIST_LO <- -20L
HIST_HI <- 4L

## `count_ref_candidate` is how many of each decade's samples are candidate
## base R disputes, so the candidate-filtered histogram is count minus it.
## Counters are doubles: a full f32 sweep puts up to 2^32 samples in a cell,
## past an integer's 2^31 - 1, where R's integer addition returns NA.
reducer_hist <- function() {
  counts <- numeric(HIST_HI - HIST_LO + 1L)
  cand <- numeric(HIST_HI - HIST_LO + 1L)
  n_exact <- 0
  n_inf <- 0
  n_rounded <- 0
  list(
    add = function(s) {
      n_exact <<- n_exact + sum(s$rel == 0)
      n_inf <<- n_inf + sum(is.infinite(s$rel))
      n_rounded <<- n_rounded + sum(s$rounded)
      keep <- s$rel > 0 & is.finite(s$rel)
      e <- s$rel[keep]
      if (!length(e)) {
        return(invisible(NULL))
      }
      b <- pmin(pmax(floor(log10(e)), HIST_LO), HIST_HI) - HIST_LO + 1L
      counts <<- counts + tabulate(b, nbins = length(counts))
      if (any(s$ref_candidate)) {
        cand <<- cand + tabulate(b[s$ref_candidate[keep]], nbins = length(cand))
      }
    },
    get = function() {
      data.frame(
        decade = HIST_LO:HIST_HI,
        count = counts,
        count_ref_candidate = cand
      )
    },
    totals = function() list(n_exact = n_exact, n_inf = n_inf, n_rounded = n_rounded)
  )
}

## ---- exact points ----------------------------------------------------------
##
## A mandatory check beside every sweep, at the inputs a sweep can miss: the
## f64 sweep draws its low 32 bits at random, so it essentially never lands on
## +-0, +-Inf, or an exact point such as p = 1. Each cell evaluates:
##
##   universal         +-0, +-Inf, NaN, the smallest and largest subnormal,
##                     the smallest normal, the largest finite value, +-0.5, +-1
##   domain_boundary   the finite endpoints of the valid input domain
##   support_edge      the finite edges of the distribution's support
##   branch            where anvl's implementation switches algorithm
##
## Every point is taken at the cell's precision -- a boundary of an f32 cell is
## the boundary after conversion to f32 -- and every finite one comes with its
## two representable neighbours, universal points included: an off-by-one-ulp
## threshold is as likely at 1/2 or 1 as at a declared edge. The results are kept in their own table and are
## never added to the sweep's counts, histograms or bands, so a point the sweep
## also visited is not counted twice.

universal_points <- function(dtype) {
  smin <- SUBNORMAL_MIN[[dtype]]
  nmin <- SMALLEST_NORMAL[[dtype]]
  big <- if (dtype == "f32") (2 - 2^-23) * 2^127 else .Machine$double.xmax
  v <- c(
    zero = 0,
    inf = Inf,
    subnormal_min = smin,
    subnormal_max = nmin - smin,
    normal_min = nmin,
    finite_max = big,
    half = 0.5,
    one = 1
  )
  c(stats::setNames(v, paste0("+", names(v))), stats::setNames(-v, paste0("-", names(v))), nan = NaN)
}

exact_points <- function(dtype, domain = c(-Inf, Inf), support = NULL, branch = NULL) {
  at <- function(v) if (dtype == "f32") as_f32(v) else v
  u <- universal_points(dtype)
  pts <- data.frame(label = names(u), role = "universal", x = unname(u), stringsAsFactors = FALSE)
  decl <- rbind(
    data.frame(label = c("domain_lo", "domain_hi"), role = "domain_boundary", x = domain),
    if (length(support)) data.frame(label = c("support_lo", "support_hi"), role = "support_edge", x = support),
    if (length(branch)) data.frame(label = paste0("branch_", names(branch)), role = "branch", x = unname(branch))
  )
  decl <- decl[is.finite(decl$x), , drop = FALSE]
  decl$x <- at(decl$x)
  pts <- rbind(pts, decl)
  f <- pts[is.finite(pts$x), , drop = FALSE]
  nb <- float_neighbours(f$x, dtype)
  pts <- rbind(
    pts,
    data.frame(label = paste0(f$label, "_below"), role = paste0(f$role, "_neighbour"), x = nb[, 1L]),
    data.frame(label = paste0(f$label, "_above"), role = paste0(f$role, "_neighbour"), x = nb[, 2L])
  )
  pts <- pts[!is.na(pts$x) | pts$label == "nan", , drop = FALSE]
  ## One row per bit pattern, keeping every label it arrived under.
  pts$bits <- bits_of(pts$x, dtype)
  key <- factor(pts$bits, levels = unique(pts$bits))
  data.frame(
    label = vapply(split(pts$label, key), function(v) paste(unique(v), collapse = "+"), ""),
    role = vapply(split(pts$role, key), function(v) paste(unique(v), collapse = "+"), ""),
    x = vapply(split(pts$x, key), `[`, 0, 1L),
    bits = levels(key),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

## A points table with no rows, for results that have none (a failed cell).
NO_POINTS <- data.frame(
  label = character(0),
  x = numeric(0),
  rel_err = numeric(0),
  identical = logical(0),
  zero_sign = logical(0),
  rounded = logical(0),
  failure = logical(0),
  category = character(0),
  ref_candidate = logical(0),
  ref_shared = logical(0)
)

## Returns the points' results with, as attribute "context", the context the
## sweep needs: the points include +-0, so one evaluation serves both, and a
## backend that compiles per input shape compiles one extra shape, not two.
run_points <- function(fun, ref, dtype, outputs, pts, domain = c(-Inf, Inf), stable = NULL) {
  fx <- fun(pts$x)
  gx <- ref(pts$x)
  sx <- if (is.null(stable)) NULL else stable$fun(pts$x)
  pz <- which(pts$x == 0 & 1 / pts$x > 0)[1L]
  nz <- which(pts$x == 0 & 1 / pts$x < 0)[1L]
  ctx <- context_from(
    lapply(fx, `[`, c(pz, nz)),
    lapply(gx, `[`, c(pz, nz)),
    dtype,
    outputs,
    domain
  )
  out <- lapply(outputs, function(o) {
    s <- score_pair(fx[[o]], gx[[o]], dtype)
    f <- sample_facts(pts$x, fx[[o]], gx[[o]], s, ctx, o)
    cause <- ifelse(s$bad, CAUSES[pmax(f$cause, 1L)], NA_character_)
    d <- if (is.null(sx[[o]])) NULL else dispute_facts(fx[[o]], gx[[o]], sx[[o]], dtype, stable$bound_ulp64)
    na <- rep(NA, length(pts$x))
    data.frame(
      output = o,
      label = pts$label,
      role = pts$role,
      x = pts$x,
      bits = pts$bits,
      value = fx[[o]],
      reference = gx[[o]],
      rel_err = s$rel,
      ulp_err = s$ulp,
      identical = s$rel == 0 & !s$bad,
      rounded = s$rounded,
      zero_sign = f$zero_sign,
      flushed = f$flushed,
      flushed_zero_error = f$flushed_zero_error,
      in_domain = f$in_domain,
      value_kind = KINDS[f$kf],
      reference_kind = KINDS[f$kg],
      failure = s$bad,
      cause = cause,
      category = unname(CAUSE_CATEGORY[cause]),
      stable = if (is.null(d)) as.numeric(na) else sx[[o]],
      ref_candidate = if (is.null(d)) na else d$candidate,
      ref_shared = if (is.null(d)) na else d$shared,
      stringsAsFactors = FALSE
    )
  })
  structure(do.call(rbind, out), context = ctx)
}

## ---- the sweep -------------------------------------------------------------
##
## `fun` and `ref` each take the chunk's values and return a *named list* of
## outputs. That plural is the point: a reverse-mode pass already computes
## d/dq, d/dmin and d/dmax together, so scoring them from one sweep rather than
## re-sweeping once per argument is a straight 3x saving on what is by far the
## largest part of the grid. One sweep, one set of reducers per output.

## `stable`, when given, is list(fun, bound_ulp64): a stable reference for
## some outputs (a named list like `ref`'s, possibly covering fewer), tested
## against every sample of those outputs -- see dispute_facts().
run_sweep <- function(
  fun,
  ref,
  dtype,
  depth,
  outputs,
  progress = TRUE,
  topk = 10L,
  domain = c(-Inf, Inf),
  ctx = NULL,
  stable = NULL
) {
  plan <- sweep_plan(dtype, depth)
  ## The behaviour at +-0 is taken before seeding, so it cannot shift the
  ## random stream that the f64 samples are drawn from. A caller that has
  ## already evaluated the exact points passes it in (see run_points()).
  if (is.null(ctx)) {
    ctx <- sweep_context(fun, ref, dtype, outputs, domain)
  }
  set.seed(SWEEP_SEED)

  acc <- lapply(outputs, function(o) {
    list(
      topk = reducer_topk(dtype, topk),
      hist = reducer_hist(),
      bands = reducer_bands(dtype),
      runs = list(reducer_runs(plan$stride), reducer_runs(plan$stride)),
      disputes = reducer_disputes(dtype, topk)
    )
  })
  names(acc) <- outputs

  if (progress) {
    cli::cli_progress_bar(
      format = "{cli::pb_extra$tag} {cli::pb_bar} {cli::pb_percent} | ETA {cli::pb_eta}",
      total = plan$n_chunks * 2L,
      extra = list(tag = sprintf("%s/%s", dtype, depth))
    )
  }

  t0 <- proc.time()[["elapsed"]]
  for (k in seq_len(plan$n_chunks)) {
    for (sgn in c(1, -1)) {
      ch <- sweep_chunk(plan, k, sgn)
      if (is.null(ch)) {
        next
      }
      fx <- fun(ch$x)
      gx <- ref(ch$x)
      sx <- if (is.null(stable)) NULL else stable$fun(ch$x)
      for (o in outputs) {
        s <- score_pair(fx[[o]], gx[[o]], dtype)
        s <- c(s, sample_facts(ch$x, fx[[o]], gx[[o]], s, ctx, o))
        if (!is.null(sx[[o]])) {
          d <- dispute_facts(fx[[o]], gx[[o]], sx[[o]], dtype, stable$bound_ulp64)
          s$ref_candidate <- d$candidate
          s$ref_shared <- d$shared
          acc[[o]]$disputes$add(ch, d, fx[[o]], gx[[o]], sx[[o]], sgn)
        }
        a <- acc[[o]]
        a$topk$add(ch, s, fx[[o]], gx[[o]], sgn)
        a$hist$add(s)
        a$bands$add(ch$idx, s, sgn, ch$x, fx[[o]], gx[[o]])
        code <- if (is.null(s$ref_candidate)) s$cause else s$cause + 100L * s$ref_candidate
        a$runs[[if (sgn > 0) 1L else 2L]]$add(ch$idx, s$bad, code, ch$x, fx[[o]], gx[[o]], s$pair)
      }
      if (progress) cli::cli_progress_update()
    }
  }
  elapsed <- proc.time()[["elapsed"]] - t0
  if (progress) {
    cli::cli_progress_done()
  }

  lapply(stats::setNames(outputs, outputs), function(o) {
    a <- acc[[o]]
    ## candidate-filtered figures exist only where a stable reference covered
    ## this output; elsewhere they are NA, not a copy of the unfiltered ones
    has_stable <- !is.null(stable) && o %in% stable$outputs
    na_unless <- function(v) if (has_stable) v else NA_real_
    tot <- a$hist$totals()
    ranges <- decode_runs(a$runs, dtype)
    detail <- a$topk$get()
    have <- nrow(detail) > 0L
    bands <- binade_profile(a$bands, dtype)

    list(
      detail = detail,
      hist = a$hist$get(),
      bands = bands,
      kinds = kinds_table(a$bands),
      ranges = ranges,
      disputes = a$disputes$get(),
      summary = data.frame(
        n_samples = plan$n_samples,
        n_exact = tot$n_exact,
        ## correctly rounded to the result's precision without being identical;
        ## counted apart from n_exact, since its relative error is not zero
        n_rounded = tot$n_rounded,
        n_inf = tot$n_inf,
        n_zero_sign = sum(bands$n_zero_sign),
        n_flushed = sum(bands$n_flushed),
        n_flushed_zero_error = sum(bands$n_flushed_zero_error),
        ## normal reference outputs, over every input: the population where a
        ## small relative error is the right expectation
        n_out_normal = sum(bands$n_out_normal),
        n_out_normal_identical = sum(bands$n_out_normal_identical),
        worst_out_normal = if (nrow(bands)) max(bands$worst_out_normal) else 0,
        ## Candidate base R disputes and the figures without them. Candidates
        ## until a validation of the stable reference says otherwise: nothing
        ## here is an exclusion yet, and the unfiltered figures stand beside.
        n_ref_candidate = na_unless(sum(bands$n_ref_candidate)),
        n_ref_candidate_nonfinite = na_unless(sum(bands$n_ref_candidate_nonfinite)),
        n_ref_shared = na_unless(sum(bands$n_ref_shared)),
        worst_rel_err_excl = na_unless(if (nrow(bands)) max(bands$worst_rel_err_excl) else 0),
        worst_out_normal_excl = na_unless(if (nrow(bands)) max(bands$worst_out_normal_excl) else 0),
        n_inf_runs = nrow(ranges),
        worst_rel_err = if (have) detail$rel_err[1] else 0,
        worst_ulp_err = if (nrow(bands)) max(bands$worst_ulp_err) else 0,
        ## The input that produced the worst error, carried on the summary row
        ## so that "how bad is it" and "at what input" can be read together.
        worst_x = if (have) detail$x[1] else NA_real_,
        worst_bits = if (have) detail$bits[1] else NA_character_,
        worst_value = if (have) detail$value[1] else NA_real_,
        worst_reference = if (have) detail$reference[1] else NA_real_,
        elapsed_sec = elapsed
      )
    )
  })
}

## Turn index runs back into the interval of the number line they cover.
## For f64 an index is a whole block of 2^32 patterns, so the endpoints are the
## exact bounds of the affected interval: low word 0x00000000 at one end and
## 0xFFFFFFFF at the other.
decode_runs <- function(runs, dtype) {
  span <- if (dtype == "f32") 2^23 else 2^20
  out <- lapply(seq_along(runs), function(k) {
    r <- runs[[k]]$get()
    if (!nrow(r)) {
      return(NULL)
    }
    sgn <- if (k == 1L) 1 else -1
    ## For f32 every pattern in a region's bounds that the sweep visited was
    ## evaluated, so bounds and sampled points coincide. For f64 the bounds
    ## are those of the 2^32-pattern blocks the failing samples fell in -- not
    ## evidence that the endpoints themselves were evaluated. The sampled
    ## points are carried separately, and are the ones to cite.
    if (dtype == "f32") {
      lo_x <- sgn * f32_from_bits(as.integer(r$lo))
      hi_x <- sgn * f32_from_bits(as.integer(r$hi))
      n_pat <- r$hi - r$lo + 1
    } else {
      lo_x <- sgn * f64_from_words(as.integer(r$lo), 0L)
      hi_x <- sgn * f64_from_words(as.integer(r$hi), -1L)
      n_pat <- (r$hi - r$lo + 1) * 2^32
    }
    data.frame(
      bits_from = bits_of(lo_x, dtype),
      bits_to = bits_of(hi_x, dtype),
      x_from = lo_x,
      x_to = hi_x,
      n_patterns = n_pat,
      bounds_are_samples = dtype == "f32",
      sampled_from = r$x_first,
      sampled_to = r$x_last,
      sampled_bits_from = bits_of(r$x_first, dtype),
      sampled_bits_to = bits_of(r$x_last, dtype),
      n_failing = r$n_failing,
      sign = sgn,
      binade_from = floor(r$lo / span),
      binade_to = floor(r$hi / span),
      cause = r$cause,
      category = unname(CAUSE_CATEGORY[r$cause]),
      ## a candidate base R dispute; not an exclusion until validated
      ref_candidate = r$ref_candidate,
      value_kind = r$value_kind,
      reference_kind = r$reference_kind,
      pairs = r$pairs,
      rep_x = r$rep_x,
      rep_bits = bits_of(r$rep_x, dtype),
      rep_value = r$rep_value,
      rep_reference = r$rep_reference,
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, out)
  if (is.null(out)) {
    out <- data.frame(
      bits_from = character(0),
      bits_to = character(0),
      x_from = numeric(0),
      x_to = numeric(0),
      n_patterns = numeric(0),
      bounds_are_samples = logical(0),
      sampled_from = numeric(0),
      sampled_to = numeric(0),
      sampled_bits_from = character(0),
      sampled_bits_to = character(0),
      n_failing = numeric(0),
      sign = numeric(0),
      binade_from = numeric(0),
      binade_to = numeric(0),
      cause = character(0),
      category = character(0),
      ref_candidate = logical(0),
      value_kind = character(0),
      reference_kind = character(0),
      pairs = character(0),
      rep_x = numeric(0),
      rep_bits = character(0),
      rep_value = numeric(0),
      rep_reference = numeric(0),
      stringsAsFactors = FALSE
    )
  }
  out
}

## ---- what the failure regions add up to ------------------------------------
##
## Each region carries one cause (see sample_facts()), and each cause one
## category. The categories are what a reader weighs:
##
##   failure             numerical or behavioural failure on valid inputs --
##                       what "unexplained" counts
##   reference_limitation  a candidate base R dispute whose stable reference
##                       has passed validation (set at resolution, never by
##                       the sweep; see apply_reference_validation())
##   backend_limitation  the platform, not the function: input flushing
##   boundary            behaviour at an endpoint of the valid input domain,
##                       which needs an explicit convention or limiting value
##   undefined_domain    a gradient outside the valid input domain, where the
##                       forward values on both sides are NaN, so no derivative
##                       is defined and the two sides only differ in convention
##
## All four stay visible. Only the last is set aside from an accuracy verdict,
## and setting it aside validates nothing.

SMALLEST_NORMAL <- c(f32 = 2^-126, f64 = 2^-1022)
SUBNORMAL_MIN <- c(f32 = 2^-149, f64 = 2^-1074)

## One (cell, output)'s regions, summarised for its results row: how many in
## each category, and where the first failure is.
region_summary <- function(ranges) {
  cat_n <- function(k) sum(ranges$category == k)
  ## failing samples, as opposed to regions: what lets a reader check that
  ## every non-identical sample is accounted for by one category or another
  cat_s <- function(k) sum(ranges$n_failing[ranges$category == k])
  f <- which(ranges$category == "failure")
  list(
    n_runs_unclassified = cat_n("failure"),
    n_regions_backend = cat_n("backend_limitation"),
    n_regions_boundary = cat_n("boundary"),
    n_regions_domain = cat_n("undefined_domain"),
    n_regions_reference = cat_n("reference_limitation"),
    n_failing_failure = cat_s("failure"),
    n_failing_backend = cat_s("backend_limitation"),
    n_failing_boundary = cat_s("boundary"),
    n_failing_domain = cat_s("undefined_domain"),
    n_failing_reference = cat_s("reference_limitation"),
    ## regions of candidate base R disputes, whatever their category
    n_regions_ref_candidate = sum(ranges$ref_candidate %in% TRUE),
    unexplained_from = if (length(f)) ranges$x_from[f[1L]] else NA_real_,
    unexplained_to = if (length(f)) ranges$x_to[f[1L]] else NA_real_
  )
}

## One (cell, output)'s exact points, summarised for its results row. Kept
## apart from every sweep count: a point the sweep also visited is not counted
## twice, and a point the sweep never visits -- most of them, in f64 -- still
## reaches the summary and everything built on it.
point_summary <- function(points) {
  n <- nrow(points)
  cat_n <- function(k) sum(points$failure & points$category %in% k)
  fin <- is.finite(points$rel_err) & points$rel_err > 0
  w <- if (any(fin)) which(fin)[which.max(points$rel_err[fin])] else NA_integer_
  f <- which(points$failure & points$category %in% "failure")
  list(
    n_points = n,
    ## identical down to the sign of zero
    n_points_identical = sum(points$identical & !points$zero_sign),
    n_points_zero_sign = sum(points$zero_sign),
    n_points_rounded = sum(points$rounded),
    n_points_finite_error = sum(fin & !points$rounded),
    n_points_failure = cat_n("failure"),
    n_points_backend = cat_n("backend_limitation"),
    n_points_boundary = cat_n("boundary"),
    n_points_domain = cat_n("undefined_domain"),
    n_points_reference = cat_n("reference_limitation"),
    worst_point_rel_err = if (is.na(w)) 0 else points$rel_err[w],
    worst_point_label = if (is.na(w)) NA_character_ else points$label[w],
    worst_point_x = if (is.na(w)) NA_real_ else points$x[w],
    n_points_ref_candidate = sum(points$ref_candidate %in% TRUE),
    ## the worst finite point error among points that are not candidate
    ## disputes, for results whose candidates are verified exclusions
    worst_point_rel_err_excl = {
      fe <- fin & !(points$ref_candidate %in% TRUE)
      if (any(fe)) max(points$rel_err[fe]) else 0
    },
    n_points_ref_shared = sum(points$ref_shared %in% TRUE),
    first_point_failure = if (length(f)) points$label[f[1L]] else NA_character_,
    first_point_failure_x = if (length(f)) points$x[f[1L]] else NA_real_
  )
}

## Settle outside-domain gradient regions against their value cell. A gradient
## region outside the valid input domain is an undefined-domain convention only
## where the matching value cell found BOTH forward values NaN for every
## out-of-domain sample in every binade the region touches.
##
## The compatibility rule for that evidence is: **the same run**. The value
## cell must have been swept in the gradient region's own run (same run_id),
## which fixes the platform, the anvl build, the harness, the depth and the
## seed together -- so "the identical inputs" is true by construction rather
## than by assumption. Evidence from any other run, however similar, is not
## used; a gradient cell re-run on its own simply stays a failure, and says why.
##
## `ranges` and `kinds` must be the whole store (or everything for the runs in
## question), never a presentation subset: every reader goes through
## resolved_ranges(), so every reader and the export see the same evidence.
##
## Each candidate gets an `evidence` note; every other region gets NA.
resolve_domain_conventions <- function(ranges, kinds) {
  if (is.null(ranges) || !nrow(ranges)) {
    return(ranges)
  }
  ranges$evidence <- NA_character_
  seg <- strsplit(ranges$cell_id, "/", fixed = TRUE)
  is_grad <- vapply(seg, function(p) p[4L] == "grad", TRUE)
  cand <- which(ranges$cause == "outside_domain" & is_grad)
  if (!length(cand)) {
    return(ranges)
  }
  ranges$evidence[cand] <- "no value cell swept in this run"
  if (is.null(kinds) || !nrow(kinds)) {
    return(ranges)
  }
  out <- kinds[!kinds$in_domain & kinds$output == "value", , drop = FALSE]
  key <- paste(out$run_id, out$cell_id, out$sign, out$binade, sep = "\r")
  total <- tapply(out$n, key, sum)
  both_nan <- tapply(out$n * (out$value_kind == "nan" & out$reference_kind == "nan"), key, sum)
  ## a value cell of the run with any out-of-domain tally at all
  swept <- unique(paste(out$run_id, out$cell_id, sep = "\r"))
  for (i in cand) {
    p <- seg[[i]]
    p[4L] <- "value"
    vid <- paste(p, collapse = "/")
    if (!paste(ranges$run_id[i], vid, sep = "\r") %in% swept) {
      next
    }
    b <- seq(ranges$binade_from[i], ranges$binade_to[i])
    k <- paste(ranges$run_id[i], vid, ranges$sign[i], b, sep = "\r")
    tt <- total[k]
    nn <- both_nan[k]
    if (!anyNA(tt) && all(tt > 0) && all(nn == tt)) {
      ranges$category[i] <- "undefined_domain"
      ranges$evidence[i] <- "value cell, same run: both forward values NaN throughout"
    } else {
      ranges$evidence[i] <- "value cell, same run: forward values not NaN/NaN throughout"
    }
  }
  ranges
}

## ---- input categories ------------------------------------------------------
##
## Relative error is only a meaningful measure for ordinary finite inputs inside
## the valid input domain. Everywhere else the right question is whether the
## result matches base R exactly, and folding both into one "worst relative
## error" let an output flushed to zero, or a NaN, stand in for how accurate a
## function is. So every band is assigned one input class, by its exponent
## field:
##
##   normal          binades 1 .. top-1, inside the domain
##   zero            the sweep's +-0 samples, from their own band rows. Zero is
##                   the value a subnormal is flushed to, and is checked like
##                   any other input; it is also in the `points` table.
##   subnormal       binade 0 without its zeros: flushed to zero on entry by
##                   this backend
##   outside_domain  a finite band lying wholly outside the valid input domain,
##                   where the value is NaN by specification
##   inf_nan         the top field: both infinities and every NaN
##
## The distribution's support plays no part: a CDF below its support is an
## ordinary input with an ordinary answer. Within each class the figures for
## samples whose reference is a *normal* float are kept apart (n_out_normal,
## worst_out_normal): normal input and normal output is where a small relative
## error is the right expectation.
##
## Classification is per binade, so a binade straddling a domain edge counts as
## inside it. `x_from`/`x_to` are the smallest and largest values in the band on
## either sign, so "wholly outside" is exact.

input_class <- function(bands, lo, hi) {
  zero <- bands$zero %||% rep(FALSE, nrow(bands))
  cls <- rep("normal", nrow(bands))
  cls[bands$binade == 0L] <- "subnormal"
  cls[zero] <- "zero"
  outside <- !bands$special & bands$binade != 0L & (bands$x_to < lo | bands$x_from > hi)
  cls[outside %in% TRUE] <- "outside_domain"
  cls[bands$special] <- "inf_nan"
  cls
}

## One row per (run, cell, output, input class): sample counts and the worst
## error, both from the bands, whose accumulators saw every sample; and the
## input that produced that worst from `detail`. The detail table keeps the
## top-K per binade and one for each sign's zero, so it holds every band's
## worst -- and where it somehow does not, the sample is left NA rather than
## replaced by a lesser one.
category_table <- function(bands, detail, bounds) {
  lo <- bounds$domain_lo[match(bands$cell_id, bounds$cell_id)]
  hi <- bounds$domain_hi[match(bands$cell_id, bounds$cell_id)]
  bands$input_class <- input_class(bands, lo, hi)
  ## NA for a band from a store written before a column existed: unknown, and
  ## summed as unknown rather than as zero
  col <- function(nm) bands[[nm]] %||% NA_real_

  grp <- paste(bands$run_id, bands$cell_id, bands$output, bands$input_class, sep = "\r")
  counts <- rowsum(
    cbind(
      n = bands$n_identical + bands$n_differ + bands$n_nonfinite,
      n_identical = bands$n_identical,
      n_rounded = col("n_rounded"),
      n_differ = bands$n_differ,
      n_nonfinite = bands$n_nonfinite,
      n_zero_sign = col("n_zero_sign"),
      n_flushed = col("n_flushed"),
      n_flushed_zero_error = col("n_flushed_zero_error"),
      n_out_normal = col("n_out_normal"),
      n_out_normal_identical = col("n_out_normal_identical"),
      n_ref_candidate = col("n_ref_candidate"),
      n_ref_candidate_nonfinite = col("n_ref_candidate_nonfinite"),
      n_ref_shared = col("n_ref_shared")
    ),
    grp,
    reorder = FALSE
  )
  first <- !duplicated(grp)
  out <- bands[first, c("run_id", "cell_id", "output", "input_class")]
  out <- cbind(out, counts[grp[first], , drop = FALSE])

  ## a detail row belongs to its band, and a zero input to the zero band
  bzero <- bands$zero %||% rep(FALSE, nrow(bands))
  bkey <- paste(bands$run_id, bands$cell_id, bands$output, bands$sign, bands$binade, bzero, sep = "\r")
  dkey <- paste(
    detail$run_id,
    detail$cell_id,
    detail$output,
    detail$sign,
    detail$binade,
    !is.null(bands$zero) & detail$x %in% 0,
    sep = "\r"
  )
  dgrp <- paste(detail$run_id, detail$cell_id, detail$output, bands$input_class[match(dkey, bkey)], sep = "\r")
  o <- order(dgrp, -detail$rel_err)
  top <- o[!duplicated(dgrp[o])]
  m <- match(grp[first], dgrp[top])
  w <- detail[top[m], , drop = FALSE]
  ## The class's worst error is the maximum over its bands' own accumulators,
  ## which saw every sample; `detail` only supplies the input that produced
  ## it. 0 means no finite non-zero error in the class -- every sample was
  ## identical or had no finite error at all; the counts say which.
  out$worst_rel_err <- unname(tapply(bands$worst_rel_err, grp, max)[grp[first]])
  if (!is.null(bands$worst_ulp_err)) {
    out$worst_ulp_err <- unname(tapply(bands$worst_ulp_err, grp, max)[grp[first]])
  }
  ## the same without candidate base R disputes; NA where a result has no
  ## stable reference (its summary row says so), or for an older store
  if (!is.null(bands$worst_rel_err_excl)) {
    out$worst_rel_err_excl <- unname(tapply(bands$worst_rel_err_excl, grp, max)[grp[first]])
  }
  ## a retained sample that is not the class's worst would mislead: drop it
  stale <- is.na(m) | w$rel_err != out$worst_rel_err
  w[stale, c("x", "value", "reference")] <- NA_real_
  w$bits[stale] <- NA_character_
  out$worst_x <- w$x
  out$worst_bits <- w$bits
  out$worst_value <- w$value
  out$worst_reference <- w$reference

  ## The worst among samples with a normal reference, and the sample itself,
  ## from whichever band holds it.
  if (!is.null(bands$worst_out_normal)) {
    ob <- order(grp, -bands$worst_out_normal)
    tb <- ob[!duplicated(grp[ob])]
    mb <- match(grp[first], grp[tb])
    wb <- bands[tb[mb], , drop = FALSE]
    out$worst_out_normal <- wb$worst_out_normal
    out$worst_out_normal_x <- wb$worst_out_normal_x
    out$worst_out_normal_value <- wb$worst_out_normal_value
    out$worst_out_normal_reference <- wb$worst_out_normal_reference
  }
  if (!is.null(bands$worst_out_normal_excl)) {
    ob <- order(grp, -bands$worst_out_normal_excl)
    tb <- ob[!duplicated(grp[ob])]
    wb <- bands[tb[match(grp[first], grp[tb])], , drop = FALSE]
    out$worst_out_normal_excl <- wb$worst_out_normal_excl
    out$worst_out_normal_excl_x <- wb$worst_out_normal_excl_x
    out$worst_out_normal_excl_value <- wb$worst_out_normal_excl_value
    out$worst_out_normal_excl_reference <- wb$worst_out_normal_excl_reference
  }
  rownames(out) <- NULL
  out[order(out$cell_id, out$output, out$input_class), , drop = FALSE]
}

## The exact points' counterpart of resolve_domain_conventions(), under the
## same rule: a gradient point outside the valid input domain is an
## undefined-domain convention only if the value cell *of the same run*
## evaluated the same bit pattern and found both values NaN.
resolve_point_conventions <- function(points) {
  if (is.null(points) || !nrow(points)) {
    return(points)
  }
  points$evidence <- NA_character_
  seg <- strsplit(points$cell_id, "/", fixed = TRUE)
  is_grad <- vapply(seg, function(p) p[4L] == "grad", TRUE)
  cand <- which(points$failure & points$cause %in% "outside_domain" & is_grad)
  if (!length(cand)) {
    return(points)
  }
  vid <- vapply(
    seg[cand],
    function(p) {
      p[4L] <- "value"
      paste(p, collapse = "/")
    },
    ""
  )
  vals <- points[points$output == "value", , drop = FALSE]
  j <- match(
    paste(points$run_id[cand], vid, points$bits[cand], sep = "\r"),
    paste(vals$run_id, vals$cell_id, vals$bits, sep = "\r")
  )
  ok <- !is.na(j) & vals$value_kind[j] %in% "nan" & vals$reference_kind[j] %in% "nan"
  points$category[cand[ok]] <- "undefined_domain"
  points$evidence[cand] <- ifelse(
    is.na(j),
    "no value cell evaluated this point in this run",
    ifelse(ok, "value cell, same run: both values NaN", "value cell, same run: values not NaN/NaN")
  )
  points
}
