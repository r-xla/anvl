## ---------------------------------------------------------------------------
## Validating the references against high precision.
##
## Two kinds of reference are validated, for different reasons:
##
##   stable    a spec's `ref_stable`, the evidence against base R in a
##             candidate dispute. A candidate becomes an exclusion only when
##             this reference has passed validation under the identity the
##             sweep recorded (see reference_status()).
##   gradient  a spec's `ref_grad`, the reference every gradient is scored
##             against. Validating it gates nothing, but says whether that
##             scoring can be trusted.
##
## Validation re-derives the reference's identity from the current code and
## the exact parameters the sweep stored; a mismatch means the code is not the
## code that ran, and validation fails without evaluating anything. Otherwise
## the reference is compared, sample by sample, with its spec's MPFR truth at
## `prec` bits, and passes only if no sample is further than the declared bound
## (double ulps at the truth; non-finite and zero truths must match exactly, a
## zero's sign aside). A sampled validation is evidence, not a proof of a
## global bound, and is recorded as such: the precision, the seed, how samples
## were chosen, how many, and the worst ones.
##
## Rmpfr is needed here and nowhere else.
## ---------------------------------------------------------------------------

mp_prec <- function(v) Rmpfr::getPrec(v)[1L]
mp_num <- function(v) Rmpfr::asNumeric(v)
## A constant as an mpfr vector the length of `like` -- never `v + 0 * x`,
## which is NaN wherever x is infinite.
mp_fill <- function(like, v) Rmpfr::mpfr(rep_len(v, length(like)), mp_prec(like))

VALIDATION_DEFAULTS <- list(prec = 256L, per_binade = 1L, focus_per_binade = 8L, random = 512L)

## A deterministic seed per (cell, output), so a validation can be repeated
## sample for sample.
seed_of <- function(s) {
  v <- utf8ToInt(s)
  as.integer(sum(v * seq_along(v) * 131) %% 2147483647)
}

## Random bit patterns: `per_binade` in every exponent field of both signs
## (Inf/NaN field included), `focus` more in each binade listed, and `n_random`
## uniform over all patterns -- which lands anywhere, agreement included.
random_patterns <- function(dtype, per_binade, focus_binades, focus_per_binade, n_random) {
  nexp <- if (dtype == "f32") 256L else 2048L
  one <- function(e, n) {
    if (dtype == "f32") {
      f32_from_bits(as.integer(e * 2^23 + sample.int(2^23, n, replace = TRUE) - 1))
    } else {
      f64_from_words(as.integer(e * 2^20 + sample.int(2^20, n, replace = TRUE) - 1), rand_word32(n))
    }
  }
  x <- unlist(lapply(0:(nexp - 1L), function(e) one(e, per_binade)))
  if (length(focus_binades)) {
    x <- c(x, unlist(lapply(focus_binades, function(e) one(e, focus_per_binade))))
  }
  x <- c(x, one(sample.int(nexp, n_random, replace = TRUE) - 1L, 1L))
  c(x, -x)
}

## Compare a double reference with an MPFR truth, in double ulps at the truth,
## against `bound`. The difference, the division by the spacing and the
## comparison with the bound all happen in MPFR; only the reported error is
## converted to a double afterwards. (Converting the difference first loses
## every fraction of a subnormal ulp: 4.49 ulp became 4 and passed a bound of
## 4.) Exact rules where an ulp is not a meaningful unit:
##
##   truth NaN or +-Inf         the reference must be the same value
##   truth exactly 0            the reference must be 0 (either sign)
##   truth beyond the double    the reference must be what it rounds to, +-Inf
##     range once rounded
##
## Returns the per-sample verdict `pass`, the reported error (Inf where an
## exact rule failed) and the truth rounded to double.
compare_to_truth <- function(ref, truth, bound) {
  t <- mp_num(truth)
  n <- length(ref)
  err <- rep(Inf, n)
  pass <- logical(n)
  tfin <- is.finite(t) # after rounding: an overflowing truth is not finite
  tzero <- as.logical(truth == 0)
  tzero[is.na(tzero)] <- FALSE
  exact_nf <- which(!tfin)
  if (length(exact_nf)) {
    pass[exact_nf] <- same_value(ref[exact_nf], t[exact_nf])
  }
  z <- which(tfin & tzero)
  if (length(z)) {
    pass[z] <- ref[z] %in% 0
  }
  o <- which(tfin & !tzero & is.finite(ref))
  if (length(o)) {
    e <- abs(Rmpfr::mpfr(ref[o], mp_prec(truth)) - truth[o]) / Rmpfr::mpfr(ulp_size(t[o], "f64"), mp_prec(truth))
    pass[o] <- as.logical(e <= bound)
    err[o] <- mp_num(e)
  }
  err[pass & is.infinite(err)] <- 0 # an exact rule met
  list(pass = pass, err = err, truth = t)
}

## The identity of a truth: the spec's own MPFR function and everything it
## calls, and the output selected -- nothing about the run or the result row.
truth_identity <- function(fun, output) {
  reference_identity(list(truth = fun), sprintf("output = %s", output))
}

## The validation method's own identity: the sampler, the comparator and the
## procedure, with the helpers they use. A declared list rather than the
## recursive traversal, because the procedure legitimately manages the random
## state (exists/get/assign on .Random.seed), which the traversal refuses. A
## validation made by any other method is ignored -- in particular every one
## made before the comparator kept its arithmetic in MPFR.
VALIDATION_METHOD_FUNS <- c(
  "compare_to_truth",
  "validate_reference",
  "random_patterns",
  "seed_of",
  "reference_under_test",
  "truth_identity",
  "mp_num",
  "mp_prec",
  "mp_fill",
  "ulp_size",
  "same_value",
  "bits_of",
  "f32_from_bits",
  "f64_from_words",
  "rand_word32",
  "parse_hex_params",
  "hex_params",
  "cmd_validate_refs"
)
validation_method_id <- local({
  cached <- NULL
  function() {
    if (is.null(cached)) {
      src <- unlist(lapply(VALIDATION_METHOD_FUNS, function(f) {
        c(paste0("## ", f), deparse(get(f, envir = globalenv(), mode = "function"), control = DEPARSE))
      }))
      tf <- tempfile()
      on.exit(unlink(tf))
      writeLines(
        c(src, sprintf("defaults: %s", paste(names(VALIDATION_DEFAULTS), VALIDATION_DEFAULTS, collapse = ","))),
        tf
      )
      cached <<- unname(tools::md5sum(tf))
    }
    cached
  }
})

## What is being validated for one result, or NULL if nothing.
reference_under_test <- function(spec, row, r) {
  flags <- cell_functions(spec, row)$flags
  params <- parse_hex_params(r$ref_params)
  if (row$kind == "value") {
    if (is.null(spec$ref_stable) || is.na(r$ref_stable_id %||% NA) && is.na(r$ref_stable_unsupported %||% NA)) {
      return(NULL)
    }
    list(
      reference = "stable",
      stored = r$ref_stable_id,
      unsupported = r$ref_stable_unsupported %||% NA_character_,
      now = if (length(params)) {
        stable_identity(spec$ref_stable, spec$ref_stable_bound_ulp64, params, flags, row$dtype)
      } else {
        NA
      },
      fun = function(x) spec$ref_stable(x, params, flags),
      truth = spec$ref_stable_mpfr,
      output = NULL,
      bound = spec$ref_stable_bound_ulp64,
      params = params,
      flags = flags
    )
  } else {
    list(
      reference = "gradient",
      stored = r$ref_grad_id %||% NA_character_,
      unsupported = r$ref_grad_unsupported %||% NA_character_,
      now = if (length(params)) {
        grad_identity(spec$ref_grad, spec$ref_grad_bound_ulp64, params, flags, row$dtype)
      } else {
        NA
      },
      fun = function(x) spec$ref_grad(x, params, flags)[[r$output]],
      ## the spec's own function; the output is selected on evaluation, and
      ## recorded in the truth's identity separately
      truth = spec$ref_grad_mpfr,
      output = r$output,
      bound = spec$ref_grad_bound_ulp64,
      params = params,
      flags = flags
    )
  }
}

## Validate one result's reference. `fixed` is a data frame (x, source) of the
## sweep's own inputs worth checking: its exact points, retained disputes and
## retained worst inputs.
validate_reference <- function(ru, r, fixed, focus_binades, opt) {
  base <- data.frame(
    reference = ru$reference,
    run_id = r$run_id,
    cell_id = r$cell_id,
    output = r$output,
    ref_id = ru$stored %||% NA_character_,
    identity_now = as.character(ru$now),
    identity_matches = isTRUE(!is.na(ru$stored) && identical(as.character(ru$now), ru$stored)),
    truth_id = if (is.null(ru$truth)) NA_character_ else as.character(truth_identity(ru$truth, ru$output %||% "value")),
    method_id = validation_method_id(),
    bound_ulp64 = ru$bound %||% NA_real_,
    precision = opt$prec,
    seed = NA_integer_,
    selection = NA_character_,
    n_samples = 0L,
    max_err_ulp64 = NA_real_,
    max_err_x = NA_real_,
    max_err_bits = NA_character_,
    pass = FALSE,
    reason = NA_character_,
    stringsAsFactors = FALSE
  )
  fail <- function(why) {
    base$reason <- why
    list(record = base, samples = NULL)
  }
  if (is.na(ru$stored)) {
    return(fail(sprintf("no identity: %s", ru$unsupported %||% "not recorded by the sweep")))
  }
  if (!base$identity_matches) {
    return(fail("identity mismatch: the reference, its dependencies or its parameters are not what the sweep ran"))
  }
  if (is.null(ru$truth)) {
    return(fail("no MPFR truth declared for this reference"))
  }
  if (is.na(base$truth_id)) {
    return(fail(sprintf(
      "the MPFR truth has no identity: %s",
      attr(truth_identity(ru$truth, ru$output %||% "value"), "unsupported") %||% "unknown"
    )))
  }
  if (!is.numeric(ru$bound)) {
    return(fail("no error bound declared for this reference"))
  }

  seed <- seed_of(paste(r$cell_id, r$output, sep = "\r"))
  old <- if (exists(".Random.seed", globalenv())) get(".Random.seed", globalenv()) else NULL
  on.exit(if (!is.null(old)) assign(".Random.seed", old, globalenv()))
  set.seed(seed)
  dtype <- if (grepl("/f32/", r$cell_id, fixed = TRUE)) "f32" else "f64"
  rnd <- random_patterns(dtype, opt$per_binade, focus_binades, opt$focus_per_binade, opt$random)
  x <- c(fixed$x, rnd)
  src <- c(fixed$source, rep("random", length(rnd)))
  bits <- bits_of(x, dtype)
  keep <- !duplicated(bits)
  x <- x[keep]
  src <- src[keep]
  bits <- bits[keep]

  ref <- ru$fun(x)
  mp_params <- lapply(ru$params, function(v) Rmpfr::mpfr(v, opt$prec))
  truth <- ru$truth(Rmpfr::mpfr(x, opt$prec), mp_params, ru$flags)
  if (!is.null(ru$output)) {
    truth <- truth[[ru$output]]
  }
  if (!methods::is(truth, "mpfr")) {
    truth <- Rmpfr::mpfr(truth, opt$prec)
  }
  cmp <- compare_to_truth(ref, truth, ru$bound)
  err <- cmp$err
  w <- which.max(err)
  base$seed <- seed
  base$selection <- sprintf(
    "exact points, retained disputes, the two worst retained inputs per binade and every earlier counterexample for this cell (%d); %d random per binade over all %d exponent fields and both signs; %d more in each of %d binade(s) of interest; %d uniform over all patterns; both signs; deduplicated by bit pattern",
    nrow(fixed),
    opt$per_binade,
    if (dtype == "f32") 256L else 2048L,
    opt$focus_per_binade,
    length(focus_binades),
    opt$random
  )
  base$n_samples <- length(x)
  base$max_err_ulp64 <- err[w]
  base$max_err_x <- x[w]
  base$max_err_bits <- bits[w]
  base$pass <- all(cmp$pass)
  base$reason <- if (base$pass) {
    ""
  } else {
    sprintf(
      "bound exceeded: %s ulp at x = %s (%s) against a bound of %s",
      fmt_num(err[w]),
      format(x[w], digits = 17),
      bits[w],
      ru$bound
    )
  }
  o <- order(-err)
  top <- o[seq_len(min(20L, length(o)))]
  top <- union(top, utils::head(which(!cmp$pass), 200L))
  samples <- data.frame(
    reference = ru$reference,
    cell_id = r$cell_id,
    output = r$output,
    ref_id = ru$stored,
    source = src[top],
    x = x[top],
    bits = bits[top],
    ref = ref[top],
    truth = cmp$truth[top],
    err_ulp64 = err[top],
    pass = cmp$pass[top],
    stringsAsFactors = FALSE
  )
  list(record = base, samples = samples)
}

## ---- what the validations say ---------------------------------------------
##
## The status of a reference identity, failing closed:
##
##   validated      at least one passing validation of exactly this identity,
##                  and no failing one
##   failed         any failing validation of this identity -- a later pass
##                  does not undo it, since the same code failed on some input
##   not validated  no validation of this identity
##   no identity    the sweep could not identify the reference (an
##                  unsupported code form); it can never be validated
##
## "Of this identity" is judged against the most recent MPFR truth used for
## it (by truth_id, the truth's own code identity): a validation is evidence
## about the pair (reference, truth), and once the truth's code has changed an
## older failure or pass says nothing about the reference any more. Older ones
## are kept, and counted as superseded where they are shown. Validations made by
## any other validation method (validation_method_id()) do not count at all.
status_of <- function(v) {
  ## only validations made by the current method count
  v <- v[(v$method_id %||% rep(NA_character_, nrow(v))) %in% validation_method_id(), , drop = FALSE]
  if (!nrow(v)) {
    return("not validated")
  }
  latest <- v$truth_id[which.max(as.POSIXct(v$validated_at, format = "%Y-%m-%dT%H:%M:%S%z"))]
  v <- v[v$truth_id %in% latest, , drop = FALSE]
  if (!all(v$pass)) {
    "failed"
  } else if (any(v$pass & v$identity_matches)) {
    "validated"
  } else {
    "not validated"
  }
}

reference_status <- function(ids, unsupported, reference, validations, outputs = NULL) {
  out <- ifelse(is.na(ids), ifelse(is.na(unsupported), NA_character_, "no identity"), "not validated")
  if (is.null(validations) || !nrow(validations)) {
    return(out)
  }
  v <- validations[validations$reference == reference, , drop = FALSE]
  for (i in which(!is.na(ids))) {
    sel <- v$ref_id %in% ids[i]
    if (!is.null(outputs)) {
      sel <- sel & v$output == outputs[i]
    }
    out[i] <- status_of(v[sel, , drop = FALSE])
  }
  out
}

## A gradient reference is validated per output: the record for a result is
## the one for its own output.
grad_reference_status <- function(res, validations) {
  reference_status(
    res$ref_grad_id %||% rep(NA_character_, nrow(res)),
    res$ref_grad_unsupported %||% rep(NA_character_, nrow(res)),
    "gradient",
    validations,
    outputs = res$output
  )
}

## Candidate disputes become verified reference limitations only where the
## result's stable reference is validated; every other candidate stays exactly
## what it was. `res` must carry run_id, cell_id, output and
## ref_stable_status.
apply_reference_validation <- function(tbl, res) {
  if (is.null(tbl) || !nrow(tbl) || is.null(tbl$ref_candidate) || is.null(res$ref_stable_status)) {
    return(tbl)
  }
  ok <- paste(res$run_id, res$cell_id, res$output, sep = "\r")[res$ref_stable_status %in% "validated"]
  k <- paste(tbl$run_id, tbl$cell_id, tbl$output, sep = "\r")
  hit <- which(tbl$ref_candidate %in% TRUE & k %in% ok & !is.na(tbl$category))
  tbl$category[hit] <- "reference_limitation"
  tbl
}

## The earlier samples a validation must replay: every one not known to have
## passed. A sample recorded before `pass` existed reads back as NA -- or the
## column is missing altogether -- and is kept: unknown is not a pass.
replay_counterexamples <- function(prev) {
  known_pass <- if (is.null(prev$pass)) rep(FALSE, nrow(prev)) else prev$pass %in% TRUE
  prev[!known_pass | is.infinite(prev$err_ulp64), , drop = FALSE]
}

## ---- the command ----------------------------------------------------------

## Which results a shard validates. Results are first grouped into *units*:
## one per distinct (reference kind, reference identity, output). An anvl cell
## and its JAX twin are scored against the same reference with the same
## identity, and one validation of it serves both -- its samples drawn from
## both cells' points, disputes and worst inputs. A result with no identity is
## a unit of its own (and fails). Units are ordered canonically, so shard i of
## n means the same units on every machine, as `run --shard` does for cells.
validation_units <- function(keys, shard = NA_integer_, shards = NA_integer_) {
  u <- sort(unique(keys))
  if (!is.na(shards)) {
    if (is.na(shard)) {
      stop("--shards needs --shard", call. = FALSE)
    }
    u <- u[(seq_along(u) - 1L) %% shards == (shard - 1L)]
  }
  u
}

cmd_validate_refs <- function(opt) {
  if (!requireNamespace("Rmpfr", quietly = TRUE)) {
    stop("validate-refs needs the Rmpfr package; run it where Rmpfr is installed", call. = FALSE)
  }
  for (nm in names(VALIDATION_DEFAULTS)) {
    opt[[nm]] <- opt[[nm]] %||% VALIDATION_DEFAULTS[[nm]]
  }
  dir <- store_dir(opt$store)
  res <- latest_results(dir)
  if (is.null(res)) {
    stop("store is empty; run a sweep first", call. = FALSE)
  }
  ## Validate what the store holds: every backend in it unless --backends
  ## says otherwise (the anvl-only default of `run` would skip every JAX
  ## result, as it once did for export).
  backends <- if (isTRUE(opt$backends_given)) opt$backends else sort(unique(res$backend))
  specs <- load_specs(include_selftest = grepl("selftest", opt$filter))
  g <- apply_filter(build_grid(specs, backends), opt$filter, extra = "output")
  res <- current_results(res[res$cell_id %in% g$cell_id, , drop = FALSE], opt$filter)$results
  if (!is.null(opt$run)) {
    res <- res[res$run_id == opt$run, , drop = FALSE]
  }
  if (!nrow(res)) {
    stop("no results to validate for that filter", call. = FALSE)
  }
  res <- res[order(res$cell_id, res$output), , drop = FALSE]

  ## what each result would validate, and the unit it belongs to
  rows <- lapply(seq_len(nrow(res)), function(i) g[g$cell_id == res$cell_id[i], , drop = FALSE][1L, , drop = FALSE])
  rus <- lapply(seq_len(nrow(res)), function(i) {
    reference_under_test(specs[[rows[[i]]$spec]], rows[[i]], res[i, , drop = FALSE])
  })
  has <- !vapply(rus, is.null, TRUE)
  key <- vapply(
    seq_len(nrow(res)),
    function(i) {
      ru <- rus[[i]]
      if (is.null(ru)) {
        return(NA_character_)
      }
      if (is.na(ru$stored)) {
        paste("unidentified", res$cell_id[i], res$output[i])
      } else {
        paste(ru$reference, ru$stored, res$output[i])
      }
    },
    ""
  )
  units <- validation_units(key[has], opt$shard, opt$shards)
  if (!is.na(opt$shards)) {
    cat(sprintf(
      "shard %d/%d: %d of %d reference unit(s)\n",
      opt$shard,
      opt$shards,
      length(units),
      length(unique(key[has]))
    ))
  }

  ## every earlier failing sample, for any reference or truth version of the
  ## same cell and output: a change must explicitly revisit them
  prev <- store_read(dir, "validation_samples")
  if (!is.null(prev) && nrow(prev)) {
    prev <- replay_counterexamples(prev)
  }
  pts <- store_read(dir, "points")
  dsp <- store_read(dir, "disputes")
  dtl <- store_read(dir, "detail")
  bnd <- store_read(dir, "bands")
  pick <- function(tbl, r) {
    if (is.null(tbl)) {
      return(NULL)
    }
    tbl[tbl$run_id == r$run_id & tbl$cell_id == r$cell_id & tbl$output == r$output, , drop = FALSE]
  }
  inputs_of <- function(r, ru) {
    p <- pick(pts, r)
    ds <- if (ru$reference == "stable") pick(dsp, r)
    c0 <- if (!is.null(prev)) prev[prev$cell_id == r$cell_id & prev$output == r$output, , drop = FALSE]
    d <- pick(dtl, r)
    fixed <- rbind(
      if (!is.null(p) && nrow(p)) data.frame(x = p$x, source = "exact point"),
      if (!is.null(ds) && nrow(ds)) {
        data.frame(x = ds$x, source = paste("dispute", ds$kind))
      },
      if (!is.null(c0) && nrow(c0)) {
        data.frame(x = c0$x, source = "earlier counterexample")
      },
      if (!is.null(d) && nrow(d)) {
        ## the two worst retained inputs of each binade
        d <- d[order(d$sign, d$binade, -d$rel_err), , drop = FALSE]
        d <- d[stats::ave(seq_along(d$x), d$sign, d$binade, FUN = seq_along) <= 2L, , drop = FALSE]
        data.frame(x = d$x, source = "worst input")
      }
    )
    b <- pick(bnd, r)
    focus <- if (is.null(b) || !nrow(b)) {
      integer(0)
    } else {
      ## Stable: every binade with a dispute. Gradient: binades whose worst
      ## finite error is within 1e3 of the result's worst -- where the result
      ## and the reference part company most. (Not every binade with a
      ## non-finite error: off a quantile's domain that is nearly all of them,
      ## and the reference is a constant there.)
      sel <- if (ru$reference == "stable") {
        (b$n_ref_candidate %||% 0) > 0 | (b$n_ref_shared %||% 0) > 0
      } else {
        b$worst_rel_err > 0 & b$worst_rel_err >= max(b$worst_rel_err) / 1e3
      }
      unique(b$binade[sel %in% TRUE])
    }
    list(fixed = fixed, focus = focus)
  }

  vid <- sprintf(
    "%s-%s-%04d%s",
    format(Sys.time(), "%Y%m%dT%H%M%S"),
    Sys.info()[["nodename"]],
    sample.int(9999L, 1L),
    if (is.na(opt$shards)) "" else sprintf("-s%dof%d", opt$shard, opt$shards)
  )
  recs <- list()
  smps <- list()
  for (k in seq_along(units)) {
    members <- which(key %in% units[k])
    i <- members[1L]
    r <- res[i, , drop = FALSE]
    ru <- rus[[i]]
    ins <- lapply(members, function(j) inputs_of(res[j, , drop = FALSE], rus[[j]]))
    fixed <- do.call(rbind, lapply(ins, `[[`, "fixed"))
    if (is.null(fixed)) {
      fixed <- data.frame(x = numeric(0), source = character(0))
    }
    focus <- sort(unique(unlist(lapply(ins, `[[`, "focus"))))
    v <- validate_reference(ru, r, fixed, focus, opt)
    v$record$validation_id <- vid
    v$record$covers <- paste(unique(res$cell_id[members]), collapse = " ")
    recs[[length(recs) + 1L]] <- v$record
    if (!is.null(v$samples)) {
      smps[[length(smps) + 1L]] <- cbind(validation_id = vid, v$samples)
    }
    cat(sprintf(
      "[%3d/%3d] %-4s %-8s %-62s %s\n",
      k,
      length(units),
      if (v$record$pass) "PASS" else "FAIL",
      ru$reference,
      paste0(
        r$cell_id,
        if (r$output != "value") paste0(" d/d", r$output) else "",
        if (length(members) > 1L) sprintf(" (+%d)", length(members) - 1L) else ""
      ),
      if (v$record$pass) {
        sprintf(
          "max %s ulp of %s (%d samples)",
          fmt_num(v$record$max_err_ulp64),
          v$record$bound_ulp64,
          v$record$n_samples
        )
      } else {
        v$record$reason
      }
    ))
  }
  if (!length(recs)) {
    cat("nothing to validate: no result in that selection has a stable or gradient reference\n")
    return(invisible(NULL))
  }
  rec <- do.call(rbind, recs)
  rec$validated_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
  rec$host <- Sys.info()[["nodename"]]
  rec$r_version <- R.version.string
  rec$rmpfr_version <- as.character(utils::packageVersion("Rmpfr"))
  rec$mpfr_version <- as.character(Rmpfr::mpfrVersion())
  store_write(dir, "validations", vid, "all", rec)
  if (length(smps)) {
    store_write(dir, "validation_samples", vid, "all", do.call(rbind, smps))
  }
  cat(sprintf(
    "\nvalidation %s: %d passed, %d failed (%d stable, %d gradient references)\n",
    vid,
    sum(rec$pass),
    sum(!rec$pass),
    sum(rec$reference == "stable"),
    sum(rec$reference == "gradient")
  ))
  invisible(rec)
}
