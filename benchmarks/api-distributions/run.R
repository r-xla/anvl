#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## The single entry point. Run it from anywhere.
##
##   Rscript run.R list                       what cells exist
##   Rscript run.R run --depth smoke          run them all, coarsely
##   Rscript run.R run --filter spec=nv_punif,dtype=f64 --depth full
##   Rscript run.R status                     coverage, errored cells, reference statuses
##   Rscript run.R selftest                   prove the engine still detects errors
##   Rscript run.R validate-refs              check the references against MPFR
##   Rscript run.R diff                       what changed since the previous run
##   Rscript run.R export --out <dir>         publish a snapshot for the site
##   Rscript run.R merge --from <dir>         fold another machine's store in
##
## Everything it writes goes to the store (NV_SWEEP_STORE), never into the
## package tree. See README.md.
## ---------------------------------------------------------------------------

suppressWarnings(suppressMessages({
  ## `here()` is this script's directory, so the harness is runnable from any
  ## working directory -- the old drivers depended on being cd'd into their own
  ## folder to resolve source("../f64-sweep.R").
  HERE <- local({
    a <- commandArgs(trailingOnly = FALSE)
    f <- sub("^--file=", "", grep("^--file=", a, value = TRUE))
    if (length(f)) dirname(normalizePath(f[1L])) else normalizePath(".")
  })
}))
here <- function() HERE

for (f in c("util.R", "engine.R", "cells.R", "provenance.R", "store.R", "validate.R")) {
  source(file.path(HERE, "R", f))
}

## ---- argument parsing ------------------------------------------------------

parse_args <- function(argv) {
  cmd <- if (length(argv) && !startsWith(argv[1L], "--")) argv[1L] else "run"
  argv <- argv[argv != cmd]
  o <- list(
    depth = "smoke",
    depth_given = NULL,
    filter = "",
    jobs = 1L,
    store = NULL,
    from = NULL,
    to = NULL,
    out = NULL,
    dry_run = FALSE,
    shard = NA_integer_,
    shards = NA_integer_,
    backends = "anvl",
    ## Whether --backends was given. `run` defaults to anvl alone because the
    ## JAX side needs Python; `export` must not inherit that default, or it
    ## silently drops every JAX result already in the store.
    backends_given = FALSE,
    ## worst inputs kept *per binade*, not globally -- see reducer_topk()
    topk = 10L,
    quiet = FALSE,
    ## validate-refs: MPFR precision, random samples per binade, extra samples
    ## per binade of interest, uniform random samples, and one run only
    prec = NULL,
    per_binade = NULL,
    focus_per_binade = NULL,
    random = NULL,
    run = NULL
  )
  i <- 1L
  while (i <= length(argv)) {
    a <- argv[i]
    val <- function() {
      if (i + 1L > length(argv)) {
        stop("option ", a, " needs a value", call. = FALSE)
      }
      argv[i + 1L]
    }
    switch(
      sub("^--", "", a),
      depth = {
        o$depth <- val()
        o$depth_given <- o$depth
        i <- i + 1L
      },
      filter = {
        o$filter <- val()
        i <- i + 1L
      },
      jobs = {
        o$jobs <- as.integer(val())
        i <- i + 1L
      },
      store = {
        o$store <- val()
        i <- i + 1L
      },
      to = {
        o$to <- val()
        i <- i + 1L
      },
      out = {
        o$out <- val()
        i <- i + 1L
      },
      from = {
        o$from <- val()
        i <- i + 1L
      },
      shard = {
        o$shard <- as.integer(val())
        i <- i + 1L
      },
      shards = {
        o$shards <- as.integer(val())
        i <- i + 1L
      },
      prec = {
        o$prec <- as.integer(val())
        i <- i + 1L
      },
      `per-binade` = {
        o$per_binade <- as.integer(val())
        i <- i + 1L
      },
      `focus-per-binade` = {
        o$focus_per_binade <- as.integer(val())
        i <- i + 1L
      },
      random = {
        o$random <- as.integer(val())
        i <- i + 1L
      },
      run = {
        o$run <- val()
        i <- i + 1L
      },
      backends = {
        o$backends <- trimws(strsplit(val(), ",")[[1L]])
        o$backends_given <- TRUE
        i <- i + 1L
      },
      topk = {
        o$topk <- as.integer(val())
        i <- i + 1L
      },
      `dry-run` = {
        o$dry_run <- TRUE
      },
      quiet = {
        o$quiet <- TRUE
      },
      stop("unknown option: ", a, call. = FALSE)
    )
    i <- i + 1L
  }
  list(cmd = cmd, opt = o)
}

## ---- running one cell ------------------------------------------------------

## Build the swept function and its reference for one grid row. Both return a
## named list of outputs: one entry for a value cell, one per differentiated
## argument for a gradient cell. Scoring every gradient from a single reverse
## pass is a straight 3x saving on the bulk of the grid -- the old drivers
## re-swept the whole space once per argument and discarded two thirds of each
## pass.
cell_functions <- function(spec, row) {
  params <- spec$params[[row$param_set]]
  flags <- if (nzchar(row$flags) && row$flags != "-") {
    kv <- strsplit(strsplit(row$flags, ",", fixed = TRUE)[[1L]], "=", fixed = TRUE)
    setNames(lapply(kv, function(p) as.logical(p[2L])), vapply(kv, `[`, "", 1L))
  } else {
    list()
  }
  ## The implementation receives its parameters at the cell's precision: anvl
  ## converts a bare `min = -pi` to f32 for an f32 argument. The reference is
  ## evaluated with the parameters exactly as received, so a comparison
  ## measures the distribution function, not parameter conversion. Without
  ## this, f32(-pi) lies below the double -pi and every f32 uniform cell
  ## "fails" at its own boundary by construction. Domain, support and branch
  ## points are taken from the same parameters.
  rparams <- if (row$dtype == "f32") {
    lapply(params, function(v) if (is.numeric(v)) as_f32(v) else v)
  } else {
    params
  }

  if (row$kind == "value") {
    fn <- if (row$backend == "jax") spec$jax_value else spec$value
    ## A stable reference, where the spec declares one for these flags: it
    ## sees exactly the parameters base R sees, and never replaces it.
    stable <- if (
      !is.null(spec$ref_stable) &&
        (is.null(spec$ref_stable_covers) || isTRUE(spec$ref_stable_covers(flags)))
    ) {
      list(
        fun = function(x) list(value = spec$ref_stable(x, rparams, flags)),
        outputs = "value",
        bound_ulp64 = spec$ref_stable_bound_ulp64,
        id = stable_identity(spec$ref_stable, spec$ref_stable_bound_ulp64, rparams, flags, row$dtype),
        note = spec$ref_stable_note %||% NA_character_
      )
    }
    list(
      fun = function(x) list(value = fn(x, row$dtype, params, flags)),
      ref = function(x) list(value = spec$ref_value(x, rparams, flags)),
      stable = stable,
      outputs = "value",
      params = params,
      ref_params = rparams,
      flags = flags
    )
  } else {
    fn <- if (row$backend == "jax") spec$jax_grad else spec$grad
    list(
      fun = function(x) fn(x, row$dtype, params, flags),
      ref = function(x) spec$ref_grad(x, rparams, flags),
      ref_id = grad_identity(spec$ref_grad, spec$ref_grad_bound_ulp64, rparams, flags, row$dtype),
      outputs = spec$grad_wrt,
      params = params,
      ref_params = rparams,
      flags = flags
    )
  }
}

run_cell <- function(spec, row, opt, pv, dir) {
  cf <- cell_functions(spec, row)
  key <- cell_key(row$cell_id)

  ## Where the function is defined, where its distribution lives, and where
  ## anvl switches algorithm -- all at the parameters the implementation
  ## receives. Branch points are anvl's own, so a JAX cell does not get them.
  domain <- if (is.null(spec$domain)) c(-Inf, Inf) else spec$domain(cf$ref_params, cf$flags)
  support <- if (is.null(spec$support)) NULL else spec$support(cf$ref_params, cf$flags)
  branch <- if (row$backend == "anvl" && !is.null(spec$branch_points)) {
    spec$branch_points(cf$ref_params, cf$flags, row$dtype)
  }
  pts <- exact_points(row$dtype, domain, support, branch)

  t0 <- Sys.time()
  out <- tryCatch(
    {
      pr <- run_points(cf$fun, cf$ref, row$dtype, cf$outputs, pts, domain, stable = cf$stable)
      sw <- run_sweep(
        cf$fun,
        cf$ref,
        row$dtype,
        opt$depth,
        cf$outputs,
        progress = !opt$quiet,
        topk = opt$topk,
        domain = domain,
        ctx = attr(pr, "context"),
        stable = cf$stable
      )
      attr(pr, "context") <- NULL
      attr(sw, "points") <- pr
      sw
    },
    error = function(e) {
      structure(conditionMessage(e), class = "sweep_error")
    }
  )
  if (inherits(out, "sweep_error")) {
    res <- cbind(
      row[rep(1L, 1L), ],
      data.frame(
        run_id = pv$run_id,
        output = "-",
        platform_key = pv$platform_key,
        device = pv$device,
        depth = opt$depth,
        n_samples = NA_real_,
        n_exact = NA_real_,
        n_rounded = NA_real_,
        n_inf = NA_real_,
        n_zero_sign = NA_real_,
        n_flushed = NA_real_,
        n_flushed_zero_error = NA_real_,
        n_out_normal = NA_real_,
        n_out_normal_identical = NA_real_,
        worst_out_normal = NA_real_,
        n_ref_candidate = NA_real_,
        n_ref_candidate_nonfinite = NA_real_,
        n_ref_shared = NA_real_,
        worst_rel_err_excl = NA_real_,
        worst_out_normal_excl = NA_real_,
        ref_stable_id = NA_character_,
        ref_stable_bound_ulp64 = NA_real_,
        ref_params = NA_character_,
        ref_stable_unsupported = NA_character_,
        ref_grad_id = NA_character_,
        ref_grad_unsupported = NA_character_,
        ref_stable_note = NA_character_,
        n_inf_runs = NA_integer_,
        n_runs_unclassified = 0L,
        n_regions_backend = 0L,
        n_regions_boundary = 0L,
        n_regions_domain = 0L,
        n_regions_ref_candidate = 0L,
        n_failing_failure = 0,
        n_failing_backend = 0,
        n_failing_boundary = 0,
        n_failing_domain = 0,
        worst_rel_err = NA_real_,
        worst_ulp_err = NA_real_,
        worst_x = NA_real_,
        worst_bits = NA_character_,
        worst_value = NA_real_,
        worst_reference = NA_real_,
        unexplained_from = NA_real_,
        unexplained_to = NA_real_,
        as.data.frame(point_summary(NO_POINTS))[rep(1L, 1L), ],
        elapsed_sec = NA_real_,
        error = as.character(out)
      )
    )
    store_write(dir, "results", pv$run_id, key, res)
    return(list(status = "ERROR", msg = as.character(out)))
  }

  store_write(
    dir,
    "points",
    pv$run_id,
    key,
    cbind(data.frame(run_id = pv$run_id, cell_id = row$cell_id), attr(out, "points"))
  )

  res_rows <- list()
  pts <- attr(out, "points")
  for (o in cf$outputs) {
    r <- out[[o]]
    rs <- region_summary(r$ranges)
    ps <- point_summary(pts[pts$output == o, , drop = FALSE])
    if (nrow(r$kinds)) {
      store_write(
        dir,
        "kinds",
        pv$run_id,
        paste0(key, "-", o),
        cbind(data.frame(run_id = pv$run_id, cell_id = row$cell_id, output = o), r$kinds)
      )
    }
    if (!is.null(r$disputes) && nrow(r$disputes)) {
      store_write(
        dir,
        "disputes",
        pv$run_id,
        paste0(key, "-", o),
        cbind(data.frame(run_id = pv$run_id, cell_id = row$cell_id, output = o), r$disputes)
      )
    }
    if (nrow(r$ranges)) {
      r$ranges <- cbind(
        data.frame(run_id = pv$run_id, cell_id = row$cell_id, output = o),
        r$ranges
      )
      store_write(dir, "ranges", pv$run_id, paste0(key, "-", o), r$ranges)
    }
    if (nrow(r$detail)) {
      d <- r$detail
      store_write(
        dir,
        "detail",
        pv$run_id,
        paste0(key, "-", o),
        cbind(
          data.frame(run_id = pv$run_id, cell_id = row$cell_id, output = o),
          d
        )
      )
    }
    if (nrow(r$bands)) {
      store_write(
        dir,
        "bands",
        pv$run_id,
        paste0(key, "-", o),
        cbind(
          data.frame(run_id = pv$run_id, cell_id = row$cell_id, output = o),
          r$bands
        )
      )
    }
    store_write(
      dir,
      "hist",
      pv$run_id,
      paste0(key, "-", o),
      cbind(
        data.frame(run_id = pv$run_id, cell_id = row$cell_id, output = o),
        r$hist
      )
    )

    res_rows[[o]] <- cbind(
      row,
      data.frame(
        run_id = pv$run_id,
        output = o,
        platform_key = pv$platform_key,
        device = pv$device,
        depth = opt$depth,
        n_samples = r$summary$n_samples,
        n_exact = r$summary$n_exact,
        n_rounded = r$summary$n_rounded,
        n_inf = r$summary$n_inf,
        n_zero_sign = r$summary$n_zero_sign,
        n_flushed = r$summary$n_flushed,
        n_flushed_zero_error = r$summary$n_flushed_zero_error,
        n_out_normal = r$summary$n_out_normal,
        n_out_normal_identical = r$summary$n_out_normal_identical,
        worst_out_normal = r$summary$worst_out_normal,
        n_ref_candidate = r$summary$n_ref_candidate,
        n_ref_candidate_nonfinite = r$summary$n_ref_candidate_nonfinite,
        n_ref_shared = r$summary$n_ref_shared,
        worst_rel_err_excl = r$summary$worst_rel_err_excl,
        worst_out_normal_excl = r$summary$worst_out_normal_excl,
        ## what a validation must match before any candidate is excluded
        ref_stable_id = if (!is.null(cf$stable) && o %in% cf$stable$outputs) {
          as.character(cf$stable$id)
        } else {
          NA_character_
        },
        ref_stable_bound_ulp64 = if (!is.null(cf$stable) && o %in% cf$stable$outputs) {
          cf$stable$bound_ulp64
        } else {
          NA_real_
        },
        ref_stable_note = if (!is.null(cf$stable) && o %in% cf$stable$outputs) cf$stable$note else NA_character_,
        ## the exact parameters the reference received, as hex doubles, so a
        ## later validation (or anyone) can reproduce them without re-running
        ## and without trusting that a parameter-set name still means the same
        ref_params = hex_params(cf$ref_params),
        ## why a stable or gradient reference has no identity, if it has none:
        ## such a reference can never be validated
        ref_stable_unsupported = if (!is.null(cf$stable) && o %in% cf$stable$outputs) {
          attr(cf$stable$id, "unsupported") %||% NA_character_
        } else {
          NA_character_
        },
        ref_grad_id = if (row$kind == "grad") as.character(cf$ref_id) else NA_character_,
        ref_grad_unsupported = if (row$kind == "grad") {
          attr(cf$ref_id, "unsupported") %||% NA_character_
        } else {
          NA_character_
        },
        n_inf_runs = r$summary$n_inf_runs,
        ## regions whose category is "failure" -- the name predates categories
        n_runs_unclassified = rs$n_runs_unclassified,
        n_regions_backend = rs$n_regions_backend,
        n_regions_boundary = rs$n_regions_boundary,
        n_regions_domain = rs$n_regions_domain,
        n_regions_ref_candidate = rs$n_regions_ref_candidate,
        n_failing_failure = rs$n_failing_failure,
        n_failing_backend = rs$n_failing_backend,
        n_failing_boundary = rs$n_failing_boundary,
        n_failing_domain = rs$n_failing_domain,
        worst_rel_err = r$summary$worst_rel_err,
        worst_ulp_err = r$summary$worst_ulp_err,
        ## The input that produced the worst error, and where the first
        ## unexplained disagreement starts, both carried on the summary row so
        ## that `status` can say "how bad, and where" without a second lookup.
        worst_x = r$summary$worst_x,
        worst_bits = r$summary$worst_bits,
        worst_value = r$summary$worst_value,
        worst_reference = r$summary$worst_reference,
        unexplained_from = rs$unexplained_from,
        unexplained_to = rs$unexplained_to,
        ## the exact points, summarised apart from the sweep's counts
        as.data.frame(ps),
        elapsed_sec = as.numeric(difftime(Sys.time(), t0, units = "secs")),
        error = NA_character_
      )
    )
  }
  res <- do.call(rbind, res_rows)
  store_write(dir, "results", pv$run_id, key, res)
  list(status = "OK", res = res)
}

## ---- commands --------------------------------------------------------------

## Every backend the store holds, unless --backends names some: `run`'s
## anvl-only default must not hide the JAX results already in the store.
store_backends <- function(opt, res) {
  if (isTRUE(opt$backends_given) || is.null(res) || !nrow(res)) opt$backends else sort(unique(res$backend))
}

cmd_list <- function(opt) {
  g <- apply_filter(build_grid(load_specs(), opt$backends), opt$filter)
  cat(sprintf("%d cells, %d result rows\n\n", nrow(g), sum(g$n_outputs)))
  print(g[c("cell_id", "n_outputs")], right = FALSE)
  invisible(g)
}

cmd_run <- function(opt) {
  specs <- load_specs(include_selftest = grepl("selftest", opt$filter))
  g <- apply_filter(build_grid(specs, opt$backends), opt$filter)
  if (!nrow(g)) {
    stop("filter matched no cells", call. = FALSE)
  }

  ## HPC sharding: the grid is canonically ordered, so shard i of n means the
  ## same set of cells on every node without any coordination between them.
  if (!is.na(opt$shards)) {
    if (is.na(opt$shard)) {
      stop("--shards needs --shard", call. = FALSE)
    }
    g <- g[(seq_len(nrow(g)) - 1L) %% opt$shards == (opt$shard - 1L), , drop = FALSE]
    cat(sprintf("shard %d/%d: %d cells\n", opt$shard, opt$shards, nrow(g)))
  }

  if (opt$dry_run) {
    cat(sprintf("would run %d cells at depth '%s':\n", nrow(g), opt$depth))
    cat(paste0("  ", g$cell_id, collapse = "\n"), "\n")
    return(invisible(g))
  }

  dir <- store_dir(opt$store)
  store_init(dir)
  pv <- collect_provenance(opt$depth)
  store_write(dir, "runs", pv$run_id, "run", provenance_row(pv))

  cat(sprintf(
    "run %s | depth %s | %d cells | store %s\n",
    pv$run_id,
    opt$depth,
    nrow(g),
    dir
  ))

  one <- function(i) {
    row <- g[i, , drop = FALSE]
    rownames(row) <- NULL
    r <- run_cell(specs[[row$spec]], row, opt, pv, dir)
    cat(sprintf("[%3d/%3d] %-6s %s\n", i, nrow(g), r$status, row$cell_id))
    r$status
  }

  ## Cells are independent and write to distinct files, so forking needs no
  ## coordination at all. Each worker inherits the same run_id, which is what
  ## makes the parts reassemble into one run.
  st <- if (opt$jobs > 1L) {
    unlist(parallel::mclapply(seq_len(nrow(g)), one, mc.cores = opt$jobs, mc.preschedule = FALSE))
  } else {
    vapply(seq_len(nrow(g)), one, "")
  }

  cat(sprintf("\ndone: %d ok, %d error\n", sum(st == "OK"), sum(st != "OK")))
  cat(sprintf("run id: %s\n", pv$run_id))
  invisible(pv$run_id)
}

## Whether the store holds what was meant to be swept: coverage of the
## declared grid per platform and depth, the cells whose newest attempt
## errored, and the validation status of every reference the results use.
cmd_status <- function(opt) {
  dir <- store_dir(opt$store)
  all <- latest_results(dir)
  specs <- load_specs(include_selftest = grepl("selftest", opt$filter))
  g <- apply_filter(build_grid(specs, store_backends(opt, all)), opt$filter, extra = "output")

  if (is.null(all)) {
    cat("The result store is empty.\n  ", dir, "\n\n")
    cat(sprintf("%d cells are declared. To fill them:\n", nrow(g)))
    cat("  Rscript run.R run --depth smoke\n")
    return(invisible(NULL))
  }
  cur <- current_results(all[all$cell_id %in% g$cell_id, , drop = FALSE], opt$filter)
  att <- cur$attempts
  ok <- att[is.na(att$error), , drop = FALSE]
  err <- cur$errors
  runs <- store_read(dir, "runs")

  cat("\nanvl distribution sweeps \u2014 status\n")
  cat("store: ", dir, "\n", sep = "")
  if (!is.null(runs)) {
    cat(sprintf("       %d run(s) recorded, most recent %s\n", nrow(runs), max(runs$started_at)))
  }

  ## ---- coverage ------------------------------------------------------------
  rule("COVERAGE")
  cat("A *cell* is one combination of function, backend, precision, value-or-\n")
  cat("gradient, parameter set and flags. A gradient cell reports one *result* per\n")
  cat("differentiated argument, so cells and results are different counts. A cell\n")
  cat("counts as swept when its newest attempt at that depth succeeded.\n\n")
  for (pk in sort(unique(att$platform_key))) {
    cat(sprintf("  %s\n", pk))
    for (d in names(DEPTHS)) {
      n <- length(unique(ok$cell_id[ok$depth == d & ok$platform_key == pk]))
      ne <- length(unique(err$cell_id[err$depth == d & err$platform_key == pk]))
      cat(sprintf(
        "    %-6s %4d of %4d cells   %s\n",
        d,
        n,
        nrow(g),
        paste0(
          if (n == nrow(g)) {
            "(complete)"
          } else if (n == 0L) {
            "(not run)"
          } else {
            "(partial)"
          },
          if (ne) sprintf(", %d errored", ne) else ""
        )
      ))
    }
  }
  swept <- unique(ok$cell_id)
  cat(sprintf(
    "\n  %d of %d cells swept at some depth, producing %d results.\n",
    length(swept),
    nrow(g),
    nrow(cur$results)
  ))
  miss <- setdiff(g$cell_id, swept)
  if (length(miss)) {
    cat(sprintf("  %d cell(s) not swept successfully at any depth:\n", length(miss)))
    for (id in utils::head(miss, 10L)) {
      cat(sprintf("    %s%s\n", id, if (id %in% err$cell_id) "   (errored)" else ""))
    }
    if (length(miss) > 10L) cat(sprintf("    ... and %d more.\n", length(miss) - 10L))
  }

  ## ---- errors --------------------------------------------------------------
  err <- err[order(err$cell_id, err$platform_key, err$depth), , drop = FALSE]
  rule(sprintf("ERRORS (%d)", nrow(err)))
  if (!nrow(err)) {
    cat("  none\n")
  }
  for (i in seq_len(min(nrow(err), 20L))) {
    cat(sprintf(
      "  %s\n    %s %s, run %s\n    %s\n",
      err$cell_id[i],
      err$platform_key[i],
      err$depth[i],
      err$run_id[i],
      strsplit(err$error[i], "\n", fixed = TRUE)[[1L]][1L]
    ))
  }
  if (nrow(err) > 20L) {
    cat(sprintf("  ... and %d more.\n", nrow(err) - 20L))
  }

  ## ---- references ----------------------------------------------------------
  res <- cur$results
  rule("REFERENCES")
  cat("Each reference against 256-bit MPFR (Rscript run.R validate-refs). A failed or\n")
  cat("unvalidated stable reference excludes nothing; a gradient reference that is\n")
  cat("not validated is still used, and its figures are only as good as it is.\n\n")
  tally <- function(v) {
    v <- v[!is.na(v)]
    if (!length(v)) {
      return("none")
    }
    t <- table(factor(v, levels = c("validated", "failed", "not validated", "no identity")))
    paste(sprintf("%d %s", t[t > 0], names(t)[t > 0]), collapse = ", ")
  }
  cat(sprintf("  stable references    %s\n", tally(res$ref_stable_status)))
  cat(sprintf("  gradient references  %s\n", tally(res$ref_grad_status)))
  broken <- c("failed", "no identity")
  bad <- res[res$ref_stable_status %in% broken | res$ref_grad_status %in% broken, , drop = FALSE]
  for (i in seq_len(min(nrow(bad), 20L))) {
    cat(sprintf(
      "    %-12s %s%s\n",
      if (bad$ref_stable_status[i] %in% broken) bad$ref_stable_status[i] else bad$ref_grad_status[i],
      bad$cell_id[i],
      if (bad$output[i] != "value") paste0(" d/d", bad$output[i]) else ""
    ))
  }
  if (nrow(bad) > 20L) {
    cat(sprintf("    ... and %d more.\n", nrow(bad) - 20L))
  }
  cat("\n")
  invisible(att)
}

cmd_selftest <- function(opt) {
  opt$filter <- "spec=selftest"
  opt$depth <- "smoke"
  opt$quiet <- TRUE
  run_id <- cmd_run(opt)

  res <- store_read(store_dir(opt$store), "results")
  res <- res[res$run_id == run_id, , drop = FALSE]
  get <- function(id, out = "value") {
    r <- res[res$cell_id == id & res$output == out, , drop = FALSE]
    if (nrow(r) != 1L) {
      stop("selftest: expected exactly one row for ", id, "/", out, call. = FALSE)
    }
    r
  }
  check <- function(label, ok) {
    cat(sprintf("  %-4s %s\n", if (isTRUE(ok)) "ok" else "FAIL", label))
    isTRUE(ok)
  }

  ## Scoring, checked directly on the machine that will run the sweep: the f32
  ## rounding edges rest on the platform's double-to-float conversion, which C
  ## leaves implementation-defined out of range, so they are verified, not assumed.
  fmax <- (2 - 2^-23) * 2^127
  smin <- 2^-149
  sc <- function(f, g, dtype) score_pair(f, g, dtype)
  negzero <- function(x) x == 0 & 1 / x < 0
  cat("\nscoring:\n")
  sp <- c(
    check(
      "f32: overflow threshold rounds to +-Inf, the tie included",
      identical(as_f32(c(1e40, -1e40, fmax + 2^103, fmax + 2^103 * 0.99)), c(Inf, -Inf, Inf, fmax))
    ),
    check(
      "f32: below half the smallest subnormal rounds to 0, sign kept, the tie included",
      all(as_f32(c(1e-46, smin / 2)) == 0) && negzero(as_f32(-1e-46)) && as_f32(smin * 0.51) == smin
    ),
    check("f32 overflow that is correctly rounded is a match, its error still infinite", {
      x <- sc(Inf, 1e40, "f32")
      x$rounded && !x$bad && is.infinite(x$rel)
    }),
    check("f32 underflow that is correctly rounded is a match, its error still 1", {
      x <- sc(0, 1e-46, "f32")
      x$rounded && !x$bad && x$rel == 1
    }),
    check("f32 overflow where the value fits is a failure", {
      x <- sc(Inf, 1e30, "f32")
      !x$rounded && x$bad
    }),
    check("f64 flushing a subnormal is not correct rounding: a finite error of 1", {
      x <- sc(0, 1e-310, "f64")
      !x$rounded && !x$bad && x$rel == 1
    }),
    check("f64 -Inf against a finite reference is a failure", sc(-Inf, 2.5, "f64")$bad),
    check("f64 difference overflow is measured, not infinite", {
      x <- sc(-1e308, 1e308, "f64")
      !x$bad && x$rel == 2
    }),
    check("f64 relative error overflowing a tiny reference is a failure", sc(1e-10, 5e-324, "f64")$bad),
    check("an ordinary error is unchanged", sc(1.0000001, 1, "f64")$rel == 1.0000001 - 1),
    check("nothing is left with a non-finite score and no route", {
      f <- c(Inf, -Inf, -1e308, 1e-10, NaN, 1e-300, 0, 2, Inf, NaN)
      g <- c(2.5, 2.5, 1e308, 5e-324, 2.5, 0, 1e-310, NaN, -Inf, NaN)
      x <- sc(f, g, "f64")
      all(is.finite(x$rel) | x$bad | x$rounded)
    })
  )

  ## Causes and categories, on constructed cases: every branch of the cause
  ## test, each for the reason it should fire and not merely by location.
  facts <- function(x, fx, gx, domain, zero_value, zero_fails, dtype = "f64") {
    z <- c(0, NEG_ZERO)
    ctx <- list(
      dtype = dtype,
      domain = domain,
      boundaries = domain[is.finite(domain)],
      zero_is_boundary = any(domain[is.finite(domain)] == 0),
      zero = list(v = list(value = zero_value, reference = z, validated = !zero_fails))
    )
    s <- list(rel = rep(Inf, length(x)), bad = rep(TRUE, length(x)), rounded = rep(FALSE, length(x)))
    CAUSES[sample_facts(x, fx, gx, s, ctx, "v")$cause]
  }
  sub <- 1e-310
  whole <- c(-Inf, Inf)
  unit <- c(0, 1)
  cat("\ncauses:\n")
  ca0 <- check(
    "negative zero survives inside the compiled harness (NEG_ZERO)",
    1 / sweep_context(function(x) list(v = x), function(x) list(v = x), "f64", "v")$zero$v$value[2] == -Inf
  )
  ca <- c(
    check(
      "the seven value kinds, judged at the result's precision",
      identical(
        value_kind(c(NaN, Inf, -Inf, 0, NEG_ZERO, 1e-40, 1, 1e-40), "f32")[1:7],
        1:7
      ) &&
        value_kind(1e-40, "f64") == 7L
    ),
    check(
      "-0 and +0 are not the same value; two NaNs are",
      !same_value(0, NEG_ZERO) && same_value(NEG_ZERO, NEG_ZERO) && same_value(NaN, NaN)
    ),
    check(
      "a NaN input is a nan_input failure, never excused",
      facts(NaN, 0, NaN, whole, c(1, 1), c(FALSE, FALSE)) == "nan_input"
    ),
    check(
      "a subnormal that behaves exactly as +0, where +0 is right, is input flushing",
      facts(sub, 5, Inf, whole, c(5, 5), c(FALSE, FALSE)) == "input_flushing"
    ),
    check(
      "the same subnormal, where +0 itself is wrong, inherits the error at zero",
      facts(sub, 5, Inf, whole, c(5, 5), c(TRUE, TRUE)) == "flush_inherits_zero_error"
    ),
    check(
      "... and is boundary behaviour when zero is a domain endpoint",
      facts(sub, 5, Inf, unit, c(5, 5), c(TRUE, TRUE)) == "domain_boundary"
    ),
    check(
      "a subnormal that does not behave as its signed zero is not excused as flushing",
      facts(sub, 6, Inf, whole, c(5, 5), c(FALSE, FALSE)) == "unidentified"
    ),
    check(
      "-0 is not +0 for the flush test: a negative subnormal must match f(-0)",
      facts(-sub, 5, Inf, whole, c(5, 7), c(FALSE, FALSE)) == "unidentified"
    ),
    check(
      "zero is a failure in its own right, not a subnormal",
      facts(0, 5, Inf, whole, c(5, 5), c(FALSE, FALSE)) == "zero_input"
    ),
    check(
      "an infinite input inside the domain is a failure",
      facts(Inf, 5, NaN, whole, c(5, 5), c(FALSE, FALSE)) == "inf_input"
    ),
    check(
      "outside the valid input domain is recorded as such",
      facts(c(2, -3, Inf), c(1, 1, 1), c(NaN, NaN, NaN), unit, c(5, 5), c(FALSE, FALSE)) %in% "outside_domain" |> all()
    ),
    check(
      "a domain endpoint is boundary behaviour",
      facts(1, 5, Inf, unit, c(5, 5), c(FALSE, FALSE)) == "domain_boundary"
    )
  )

  ## A zero result is validated only when it is identical or correctly rounded:
  ## f(0) = 2 against g(0) = 1 has a finite error, and must not excuse the
  ## subnormals flushed onto it.
  cat("\nzero validation and flushing:\n")
  zctx <- function(f0, g0) context_from(list(v = f0), list(v = g0), "f64", "v", whole)
  zc <- zctx(c(2, 2), c(1, 1))
  zx <- c(sub, 2 * sub)
  zf <- c(2, 2)
  zg <- c(0, 1.5)
  zs <- score_pair(zf, zg, "f64")
  zfa <- sample_facts(zx, zf, zg, zs, zc, "v")
  zok <- zctx(c(1, 1), c(1, 1))
  zfo <- sample_facts(zx, c(1, 1), c(0, 1.5), score_pair(c(1, 1), c(0, 1.5), "f64"), zok, "v")
  zv <- c(
    check("a zero result with a finite 100% error is not validated", !any(zc$zero$v$validated)),
    check(
      "a zero result one ulp off is not validated either",
      !any(zctx(c(1 + 2^-52, 1 + 2^-52), c(1, 1))$zero$v$validated)
    ),
    check(
      "a subnormal flushed onto that zero inherits its error: a failure, not the backend",
      CAUSES[zfa$cause[1]] == "flush_inherits_zero_error" && CAUSE_CATEGORY[["flush_inherits_zero_error"]] == "failure"
    ),
    check(
      "flushing onto an unvalidated zero is counted apart from flushing onto a validated one",
      !any(zfa$flushed) && all(zfa$flushed_zero_error) && all(zfo$flushed) && !any(zfo$flushed_zero_error)
    ),
    check("the same flush onto a validated zero is input flushing", CAUSES[zfo$cause[1]] == "input_flushing")
  )

  cat("\nexact points and conventions:\n")
  ep <- exact_points("f32", domain = c(0, 1), support = c(-pi, 2 * pi), branch = c(mid = 0.3))
  lab <- function(l) ep$x[grepl(paste0("(^|\\+)", l, "($|\\+)"), ep$label)]
  rg <- data.frame(
    run_id = c("A", "A", "B"),
    cell_id = "s/anvl/f64/grad/p/f",
    output = "x",
    cause = "outside_domain",
    category = "failure",
    sign = 1,
    binade_from = c(1030, 1040, 1030),
    binade_to = c(1031, 1040, 1031)
  )
  kd <- data.frame(
    run_id = "A",
    cell_id = "s/anvl/f64/value/p/f",
    output = "value",
    sign = 1,
    binade = c(1030, 1031, 1040, 1040),
    in_domain = FALSE,
    value_kind = c("nan", "nan", "nan", "normal"),
    reference_kind = c("nan", "nan", "nan", "nan"),
    n = c(10, 10, 5, 1)
  )
  rr <- resolve_domain_conventions(rg, kd)
  pt <- data.frame(
    run_id = c("A", "A", "B", "A"),
    cell_id = c("s/anvl/f64/grad/p/f", "s/anvl/f64/value/p/f", "s/anvl/f64/grad/p/f", "s/anvl/f64/grad/p/f"),
    output = c("x", "value", "x", "x"),
    label = c("+inf", "+inf", "+inf", "+one"),
    bits = c("0x7FF0000000000000", "0x7FF0000000000000", "0x7FF0000000000000", "0x7FF0000000000000"),
    failure = c(TRUE, FALSE, TRUE, TRUE),
    cause = c("outside_domain", NA, "outside_domain", "outside_domain"),
    category = c("failure", NA, "failure", "failure"),
    value_kind = c("normal", "nan", "normal", "normal"),
    reference_kind = c("nan", "nan", "nan", "nan")
  )
  pr <- resolve_point_conventions(pt)
  ec <- c(
    check(
      "exact points include +-0, +-Inf and NaN, each once",
      sum(ep$x == 0, na.rm = TRUE) == 2 &&
        sum(is.infinite(ep$x)) == 2 &&
        sum(is.nan(ep$x)) == 1 &&
        !anyDuplicated(ep$bits)
    ),
    check("a domain edge comes with both representable neighbours", all(c(1 - 2^-24, 1, 1 + 2^-23) %in% ep$x)),
    check("an f32 cell's support edge is the edge after conversion to f32", as_f32(-pi) %in% ep$x && !(-pi %in% ep$x)),
    check(
      "a gradient outside the domain is a convention only where both values are NaN",
      rr$category[1] == "undefined_domain"
    ),
    check("... and stays a failure where the value cell found a finite value", rr$category[2] == "failure"),
    check(
      "evidence from another run never settles a convention, and says why",
      rr$category[3] == "failure" && grepl("no value cell", rr$evidence[3])
    ),
    check(
      "a gradient point is a convention only against the same run's value point, by bits",
      identical(pr$category, c("undefined_domain", NA, "failure", "undefined_domain")) &&
        grepl("no value cell", pr$evidence[3])
    ),
    check(
      "the universal points 1/2 and 1 come with both neighbours",
      all(c(0.5 - 2^-25, 0.5, 0.5 + 2^-24, 1 - 2^-24, 1 + 2^-23) %in% exact_points("f32")$x)
    ),
    check("nv_punif declares its log/log1p switch at the midpoint, with neighbours", {
      sp_all <- load_specs()
      if (is.null(sp_all$nv_punif)) {
        TRUE
      } else {
        bp <- sp_all$nv_punif$branch_points(list(min = -1, max = 3), list(log_p = TRUE), "f64")
        e <- exact_points("f64", branch = bp)
        bp == 1 &&
          sum(grepl("branch_", e$label)) == 3 &&
          is.null(sp_all$nv_punif$branch_points(list(min = -1, max = 3), list(log_p = FALSE), "f64"))
      }
    }),
    check("the reference sees f32-rounded parameters in an f32 cell", {
      sp_all <- load_specs()
      if (is.null(sp_all$nv_dunif)) {
        TRUE
      } else {
        cf <- cell_functions(
          sp_all$nv_dunif,
          list(
            spec = "nv_dunif",
            backend = "anvl",
            dtype = "f32",
            kind = "value",
            param_set = names(sp_all$nv_dunif$params)[1],
            flags = "log=FALSE"
          )
        )
        identical(unname(unlist(cf$ref_params)), as_f32(unname(unlist(cf$params))))
      }
    })
  )

  pts <- store_read(store_dir(opt$store), "points")
  pts <- pts[pts$run_id == run_id, , drop = FALSE]
  rng <- store_read(store_dir(opt$store), "ranges")
  rng <- rng[rng$run_id == run_id, , drop = FALSE]
  knd <- store_read(store_dir(opt$store), "kinds")
  knd <- knd[knd$run_id == run_id, , drop = FALSE]
  bnd <- store_read(store_dir(opt$store), "bands")
  bnd <- bnd[bnd$run_id == run_id, , drop = FALSE]
  p <- "selftest/anvl/%s/%s/%s/broken=%s"
  cat("\nwhat the sweep stored:\n")
  sw <- c(
    check(
      "every cell stored its exact points, +-0 among them",
      nrow(pts) > 0 && all(tapply(pts$x %in% 0, paste(pts$cell_id, pts$output), sum) == 2)
    ),
    check(
      "the clean selftest cells pass every exact point",
      !any(pts$failure[grepl("broken=FALSE", pts$cell_id) & !grepl("/(pinhole|weakref)/", pts$cell_id)])
    ),
    check("the f32 break at +Inf is recorded as an infinite-input failure, returning 0 against Inf", {
      r <- rng[rng$cell_id == "selftest/anvl/f32/value/clean/broken=TRUE", , drop = FALSE]
      nrow(r) == 1 && r$cause == "inf_input" && r$category == "failure" && grepl("+0 vs +inf", r$pairs, fixed = TRUE)
    }),
    check("what each side returned is tallied for every cell", nrow(knd) > 0),
    check("zero inputs are a band of their own; binade 0 holds only the subnormals", {
      b <- bnd[grepl("/f32/value/clean/broken=FALSE", bnd$cell_id), , drop = FALSE]
      z <- b[b$zero, , drop = FALSE]
      s0 <- b[!b$zero & b$binade == 0, , drop = FALSE]
      nrow(z) == 2 &&
        all(z$x_from == 0 & z$x_to == 0) &&
        all(z$n_identical == 1) &&
        all(abs(s0$x_from) > 0 & abs(s0$x_to) > 0) &&
        sum(b$n_identical + b$n_differ + b$n_nonfinite) == get(sprintf(p, "f32", "value", "clean", "FALSE"))$n_samples
    }),
    check("a zero error is not displaced from its class by ten worse subnormals", {
      ## +0 with a 0.1 error, then ten subnormals with error 1, in one chunk
      x <- f32_from_bits(0:10)
      fx <- c(1.1, rep(2, 10))
      gx <- c(1, rep(1, 10))
      ctx <- context_from(list(v = 1.1), list(v = 1), "f32", "v", c(-Inf, Inf))
      sc <- score_pair(fx, gx, "f32")
      sc <- c(sc, sample_facts(x, fx, gx, sc, ctx, "v"))
      tk <- reducer_topk("f32", 10L)
      bd <- reducer_bands("f32")
      tk$add(list(idx = 0:10, x = x), sc, fx, gx, 1)
      bd$add(0:10, sc, 1, x, fx, gx)
      tag <- function(d) cbind(data.frame(run_id = "r", cell_id = "c", output = "v"), d)
      ct <- category_table(
        tag(binade_profile(bd, "f32")),
        tag(tk$get()),
        data.frame(cell_id = "c", domain_lo = -Inf, domain_hi = Inf)
      )
      z <- ct[ct$input_class == "zero", ]
      abs(z$worst_rel_err - 0.1) < 1e-6 && z$worst_x == 0 && ct$worst_rel_err[ct$input_class == "subnormal"] == 1
    }),
    check("... and a class of their own in the categories", {
      b <- bnd[grepl("/f32/value/clean/broken=FALSE", bnd$cell_id), , drop = FALSE]
      ct <- category_table(
        b,
        data.frame(
          run_id = character(0),
          cell_id = character(0),
          output = character(0),
          sign = numeric(0),
          binade = numeric(0),
          x = numeric(0),
          rel_err = numeric(0),
          bits = character(0),
          value = numeric(0),
          reference = numeric(0)
        ),
        data.frame(cell_id = b$cell_id[1], domain_lo = -Inf, domain_hi = Inf)
      )
      identical(ct$n[ct$input_class == "zero"], 2) && "subnormal" %in% ct$input_class
    })
  )

  ## Every category counts in a result's state, and exact points with them.
  cat("\nresult state:\n")
  lr <- latest_results(store_dir(opt$store))
  lr <- lr[lr$run_id == run_id, , drop = FALSE]
  one <- function(id, out = "value") lr[lr$cell_id == id & lr$output == out, , drop = FALSE]
  ph <- one(sprintf(p, "f64", "value", "pinhole", "FALSE"))
  row <- function(...) {
    base <- list(
      n_samples = 100,
      n_exact = 100,
      n_zero_sign = 0,
      worst_rel_err = 0,
      n_runs_unclassified = 0,
      n_regions_boundary = 0,
      n_regions_backend = 0,
      n_regions_domain = 0,
      n_failing_domain = 0,
      n_points = 5,
      n_points_identical = 5,
      n_points_failure = 0,
      n_points_boundary = 0,
      n_points_backend = 0,
      n_points_domain = 0,
      worst_point_rel_err = 0
    )
    as.data.frame(utils::modifyList(base, list(...)))
  }
  stt <- c(
    check(
      "a failure only at x = 1 is caught by the exact points while the f64 sweep sees nothing",
      nrow(ph) == 1 &&
        ph$n_inf == 0 &&
        ph$worst_rel_err == 0 &&
        ph$n_points_failure == 1 &&
        ph$first_point_failure %in% c("+one", "+one+domain_hi")
    ),
    check("... and makes the result failing, and not bit-identical", {
      s <- result_state(ph)
      s$failing && !s$identical
    }),
    check("a boundary region alone makes a result not bit-identical, and is shown", {
      s <- result_state(row(n_exact = 99, n_regions_boundary = 1))
      s$boundary && !s$identical && !s$failing
    }),
    check("a failing exact point alone does too", {
      s <- result_state(row(n_points_identical = 4, n_points_boundary = 1))
      s$boundary && !s$identical
    }),
    check("a signed-zero difference is not bit-identical", !result_state(row(n_zero_sign = 1))$identical),
    check("undefined-domain conventions alone are set aside, and said so", {
      s <- result_state(row(n_exact = 90, n_regions_domain = 1, n_failing_domain = 10))
      !s$identical && s$identical_but_conventions && !s$failing
    })
  )

  ## Base R disputes: each of the three conditions, and the exact rules.
  cat("\nbase R disputes:\n")
  df <- function(f, g, st, dtype = "f64", b = 0) dispute_facts(f, g, st, dtype, b)
  u32 <- 2^-23
  dp <- c(
    check(
      "the ulp spacing just below a power of two is that binade's, not the next",
      all(ulp_size(2^(-3:3) * (1 - 2^-53), "f64") == 2^((-3:3) - 53)) && is.nan(ulp_size(Inf, "f64"))
    ),
    check("the f32 spacing derived from the f64 one matches ulp_size(, \"f32\") everywhere", {
      v <- c(10^seq(-50, 38, length.out = 4001), 2^(-150:127), 2^(-150:127) * (1 - 2^-53), 1e-46)
      identical(pmax(ulp_size(v, "f64") * 2^29, SUBNORMAL_MIN[["f32"]]), ulp_size(v, "f32"))
    }),
    check("base R 0 where log1p(-1e-100) = -1e-100 and anvl has it: a candidate", df(-1e-100, 0, -1e-100)$candidate),
    check("anvl and base R equally wrong is never a candidate, and is recorded as shared", {
      d <- df(0, 0, -1e-100)
      !d$candidate && d$shared
    }),
    check(
      "anvl beyond its tolerance is not a candidate, however wrong base R is",
      !df(as_f32(1 + 3 * u32), 5, 1, "f32")$candidate
    ),
    check(
      "anvl within tolerance but further than base R is not a candidate",
      !df(as_f32(1 + u32), 1 + 1e-12, 1, "f32")$candidate && df(as_f32(1 + u32), 1 + 1e-6, 1, "f32")$candidate
    ),
    check("NaN on any side is never a dispute", !any(df(c(NaN, 1, 1), c(1, NaN, 2), c(1, 1, NaN))$candidate)),
    check("a stable +-Inf or +-0 needs anvl to match exactly, down to the sign", {
      d <- df(c(Inf, NEG_ZERO, 0), c(1e300, 1, 1), c(Inf, 0, 0))
      identical(d$candidate, c(TRUE, FALSE, TRUE))
    }),
    check("an f32 cell's stable value beyond the f32 range needs anvl to be its rounding, +-Inf", {
      d <- df(c(Inf, as_f32(3.4e38), Inf), c(0, 0, 1e39), c(1e39, 1e39, 1e39), "f32")
      identical(d$candidate, c(TRUE, FALSE, FALSE))
    }),
    check("the stable reference's identity changes with its bound", {
      f <- function(x, p, fl) abs(x)
      stable_identity(f, 1) != stable_identity(f, 2)
    }),
    {
      wr <- one(sprintf(p, "f64", "value", "weakref", "FALSE"))
      check(
        "a sweep records a wrong reference as candidates, with the filtered figures beside",
        nrow(wr) == 1 &&
          wr$n_ref_candidate > 0 &&
          wr$n_ref_candidate_nonfinite > 0 &&
          wr$worst_rel_err > 0 &&
          wr$worst_rel_err_excl == 0 &&
          !is.na(wr$ref_stable_id) &&
          wr$n_ref_shared == 0
      )
    },
    check("... and flags its no-finite-error region as a candidate, leaving its category alone", {
      r <- rng[grepl("/f64/value/weakref/broken=FALSE", rng$cell_id), , drop = FALSE]
      nrow(r) > 0 && all(r$ref_candidate) && all(r$category == "failure")
    }),
    check("... and keeps the evidence for each candidate: all three values and both distances", {
      d <- store_read(store_dir(opt$store), "disputes")
      d <- d[d$run_id == run_id & grepl("/f64/value/weakref/broken=FALSE", d$cell_id), , drop = FALSE]
      nrow(d) > 0 &&
        all(d$kind == "candidate") &&
        all(d$d_anvl <= d$t_anvl) &&
        all(d$d_base > d$t_base) &&
        all(d$beyond_tolerance > 1)
    }),
    check("a candidate is an exclusion only under a validated stable reference; failed or absent keeps it", {
      r <- rng[grepl("/f64/value/weakref/broken=FALSE", rng$cell_id), , drop = FALSE]
      at <- function(status) {
        res1 <- data.frame(run_id = r$run_id[1], cell_id = r$cell_id[1], output = "value", ref_stable_status = status)
        apply_reference_validation(r, res1)$category
      }
      all(at("not validated") == "failure") &&
        all(at("failed") == "failure") &&
        all(at(NA_character_) == "failure") &&
        all(at("validated") == "reference_limitation")
    }),
    check("validation status fails closed: any failure of the latest truth wins, a changed truth supersedes", {
      v <- function(truth, pass, t) {
        data.frame(
          reference = "stable",
          ref_id = "A",
          output = "value",
          truth_id = truth,
          pass = pass,
          identity_matches = TRUE,
          validated_at = sprintf("2026-09-26T10:%02d:00+0000", t),
          method_id = validation_method_id()
        )
      }
      st <- function(...) reference_status("A", NA, "stable", rbind(...))
      st(v("t1", TRUE, 1)) == "validated" &&
        st(v("t1", TRUE, 1), v("t1", FALSE, 2)) == "failed" &&
        st(v("t1", FALSE, 1), v("t1", TRUE, 2)) == "failed" &&
        st(v("t1", FALSE, 1), v("t2", TRUE, 2)) == "validated" &&
        reference_status(NA, "uses get()", "stable", v("t1", TRUE, 1)) == "no identity" &&
        reference_status("B", NA, "stable", v("t1", TRUE, 1)) == "not validated" &&
        st(data.frame(
          reference = "stable",
          ref_id = "A",
          output = "value",
          truth_id = "t1",
          pass = TRUE,
          identity_matches = FALSE,
          validated_at = "2026-09-26T10:00:00+0000",
          method_id = validation_method_id()
        )) ==
          "not validated"
    }),
    check(
      "cells without a stable reference have no candidate figures, not copies",
      all(is.na(lr$worst_rel_err_excl[lr$kind == "grad"]))
    )
  )

  ## What a stage-two validation relies on, and two measurement reductions.
  cat("\nidentity, parameters and reductions:\n")
  pp <- list(min = -pi, max = 2 * pi)
  fl <- list(log_p = TRUE)
  cl <- local({
    c0 <- 1
    function(x, p, f) x * c0
  })
  id0 <- stable_identity(cl, 8, pp, fl, "f64")
  sv <- same_value
  assign("same_value", function(a, b) a == b, envir = globalenv())
  id_sv <- stable_identity(cl, 8, pp, fl, "f64")
  assign("same_value", sv, envir = globalenv())
  environment(cl)$c0 <- 2
  id_c0 <- stable_identity(cl, 8, pp, fl, "f64")
  environment(cl)$c0 <- 1
  wr <- one(sprintf(p, "f64", "value", "weakref", "FALSE"))
  hx <- one(sprintf(p, "f32", "value", "nudged", "FALSE"))$ref_params
  ## a hist counter at 2^31 - 1 must count on, not become NA (bin 16 is the
  ## decade [1e-5, 1e-4)). Subnormal expected values below are written as
  ## multiples of 2^-1074: R parses a subnormal hex literal such as
  ## 0x1.5ap-1066 as 0.
  hh <- reducer_hist()
  environment(hh$add)$counts[16] <- 2^31 - 1
  environment(hh$add)$cand[16] <- 2^31 - 1
  hh$add(list(rel = 1e-5, rounded = FALSE, ref_candidate = TRUE))
  ## the worst ulp error is not the worst relative error's sample: in [1, 2)
  ## 3 ulp at g = 1 is the larger relative error, 4 ulp at g = 1.99 the larger
  ## ulp error, and a one-deep shortlist ranked by relative error keeps only
  ## the first
  ug <- c(1, 1.99)
  uf <- ug + c(3, 4) * 2^-52
  us <- score_pair(uf, ug, "f64")
  us <- c(
    us,
    sample_facts(ug, uf, ug, us, context_from(list(v = c(1, 1)), list(v = c(1, 1)), "f64", "v", c(-Inf, Inf)), "v")
  )
  ub <- reducer_bands("f64")
  ut <- reducer_topk("f64", 1L)
  uidx <- floor(ug * 0) + 1023 * 2^20 + c(0, 1)
  ub$add(uidx, us, 1, ug, uf, ug)
  ut$add(list(idx = uidx, x = ug), us, uf, ug, 1)
  idr <- c(
    check("the stable identity sees a change in a helper the classifier calls (same_value)", id_sv != id0),
    check("... and a constant captured in the stable reference's closure", id_c0 != id0),
    check(
      "... and a one-ulp change in a reference parameter",
      stable_identity(cl, 8, list(min = -pi, max = 2 * pi * (1 + 2^-52)), fl, "f64") != id0
    ),
    check("... and a captured primitive changed from abs to sqrt", {
      f1 <- local({
        helper <- abs
        function(x, p, f) helper(x)
      })
      f2 <- local({
        helper <- sqrt
        function(x, p, f) helper(x)
      })
      stable_identity(f1, 8, pp, fl, "f64") != stable_identity(f2, 8, pp, fl, "f64")
    }),
    check("... and records the package and version behind a pkg::fn call", {
      m <- code_identity(list(g = function(x, p, f) tools::md5sum(x)))
      any(grepl(sprintf("## package tools %s :: md5sum", utils::packageVersion("tools")), m, fixed = TRUE))
    }),
    check("a reference whose dependencies cannot be read has no identity, so can never validate", {
      a <- stable_identity(function(x, p, f) do.call("abs", list(x)), 8, pp, fl, "f64")
      b <- stable_identity(function(x, p, f) get("abs")(x), 8, pp, fl, "f64")
      is.na(a) &&
        is.na(b) &&
        grepl("do.call", attr(a, "unsupported")) &&
        reference_status(a, attr(a, "unsupported"), "stable", NULL) == "no identity"
    }),
    check(
      "stored parameters round-trip exactly from hex",
      identical(parse_hex_params(hex_params(list(min = -pi, max = 2 * pi))), list(min = -pi, max = 2 * pi))
    ),
    check(
      "each result carries its reference's exact parameters, as hex doubles",
      identical(hx, "err=0x1p+0") && nrow(wr) == 1 && identical(wr$ref_params, "err=-0x1p+1")
    ),
    check(
      "histogram counters count past 2^31 instead of turning NA",
      identical(hh$get()$count[16], 2^31) && identical(hh$get()$count_ref_candidate[16], 2^31)
    ),
    check("the worst ulp error has its own maximum, not the relative-error shortlist's", {
      identical(max(binade_profile(ub, "f64")$worst_ulp_err), 4) && identical(ut$get()$ulp_err, 3)
    })
  )

  ## The analytic gradient references, at inputs where the obvious evaluation
  ## order under- or overflows early; expected values from 256-bit MPFR.
  cat("\ngradient references:\n")
  sp_all <- load_specs()
  near <- function(a, b, n = 4) isTRUE(abs(a - b) <= n * ulp_size(b, "f64"))
  ## inv_mills() lives in the normal specs' own environment; the standard
  ## lower-tail log pnorm gradient d/dq is exactly it
  inv_mills_ref <- function(z) {
    sp_all$nv_pnorm$ref_grad(z, list(mean = 0, sd = 1), list(lower_tail = TRUE, log_p = TRUE))$q
  }
  gr <- c(
    check(
      "dnorm d/dsd at x = 38.6 is the subnormal 1.709467e-321, not 0",
      near(sp_all$nv_dnorm$ref_grad(38.6, list(mean = 0, sd = 1), list(log = FALSE))$sd, 346 * 2^-1074)
    ),
    check(
      "dnorm d/dsd at x = 1e155 is 0, not NaN",
      identical(sp_all$nv_dnorm$ref_grad(1e155, list(mean = 0, sd = 1), list(log = FALSE))$sd, 0)
    ),
    check(
      "log dnorm d/dsd at x = 1e155, mean = -pi, sd = 2 pi is 4.031442e307, not Inf",
      near(sp_all$nv_dnorm$ref_grad(1e155, list(mean = -pi, sd = 2 * pi), list(log = TRUE))$sd, 0x1.cb46efba3778dp+1021)
    ),
    check(
      "pnorm d/dsd at q = 38.6 is the subnormal -4.446591e-323, not 0",
      near(
        sp_all$nv_pnorm$ref_grad(38.6, list(mean = 0, sd = 1), list(lower_tail = TRUE, log_p = FALSE))$sd,
        -9 * 2^-1074
      )
    ),
    check("qnorm d/dp at p = 1e-100 uses the true quantile, not base R's rounded one", {
      near(
        sp_all$nv_qnorm$ref_grad(1e-100, list(mean = 0, sd = 1), list(lower_tail = TRUE, log_p = FALSE))$p,
        4.6903754148377423e98,
        2
      )
    }),
    check("qnorm at a subnormal p: d/dp at 4.04e-310 and z at the smallest subnormal, from the true quantile", {
      qs <- function(pr) sp_all$nv_qnorm$ref_grad(pr, list(mean = 0, sd = 1), list(lower_tail = TRUE, log_p = FALSE))
      near(qs(4.0392245102948702e-310)$p, 0x1.7688e4ee278cep+1022) &&
        near(qs(4.9406564584124654e-324)$sd, -0x1.33bd3f27fcd03p+5)
    }),
    check("no normal-family gradient reference is NaN at x = +-Inf", {
      pp2 <- list(list(mean = 0, sd = 1), list(mean = -pi, sd = 2 * pi))
      fl2 <- list(
        list(sp_all$nv_dnorm, list(log = FALSE)),
        list(sp_all$nv_dnorm, list(log = TRUE)),
        list(sp_all$nv_pnorm, list(lower_tail = TRUE, log_p = FALSE)),
        list(sp_all$nv_pnorm, list(lower_tail = TRUE, log_p = TRUE)),
        list(sp_all$nv_pnorm, list(lower_tail = FALSE, log_p = FALSE)),
        list(sp_all$nv_pnorm, list(lower_tail = FALSE, log_p = TRUE))
      )
      !any(vapply(
        fl2,
        function(sf) any(vapply(pp2, function(q) anyNA(unlist(sf[[1]]$ref_grad(c(-Inf, Inf), q, sf[[2]]))), TRUE)),
        TRUE
      ))
    }),
    check(
      "the inverse Mills ratio is within a few ulp at z = -50 and -100",
      near(inv_mills_ref(-50), 0x1.9028ed635bd0cp+5) && near(inv_mills_ref(-100), 0x1.900a3cea7d44dp+6)
    ),
    check("no exponential-family gradient reference is NaN at x = +-Inf or +-the largest finite x", {
      xe <- c(-Inf, -.Machine$double.xmax, .Machine$double.xmax, Inf)
      flagged <- c(
        list(list(sp_all$nv_dexp, list(log = FALSE)), list(sp_all$nv_dexp, list(log = TRUE))),
        unlist(
          lapply(c("nv_pexp", "nv_qexp"), function(nm) {
            lapply(1:4, function(k) list(sp_all[[nm]], list(lower_tail = k <= 2, log_p = k %% 2 == 0)))
          }),
          recursive = FALSE
        )
      )
      !any(vapply(
        flagged,
        function(sf) any(vapply(sf[[1L]]$params, function(q) anyNA(unlist(sf[[1L]]$ref_grad(xe, q, sf[[2L]]))), TRUE)),
        TRUE
      ))
    }),
    check(
      "dexp d/dx at t = 740 (large rate) is -1.4e-301, not 0: rate^2 is applied before exp(-t) underflows",
      near(
        sp_all$nv_dexp$ref_grad(0x1.94abaf44d6f1ep-26, list(rate = pi * 1e10), list(log = FALSE))$x,
        -0x1.1b80f45a754bap-998
      )
    ),
    check(
      "log dexp d/drate at the largest finite x (large rate) is 1/rate - x, not -Inf where rate * x overflows",
      near(
        sp_all$nv_dexp$ref_grad(.Machine$double.xmax, list(rate = pi * 1e10), list(log = TRUE))$rate,
        -0x1.fffffffffffffp+1023
      )
    ),
    check(
      "pexp's lower log_p stable reference at q = 1e-300 (small rate) is log(rate) + log(q), not -Inf where rate * q underflows",
      near(
        sp_all$nv_pexp$ref_stable(1e-300, list(rate = pi * 1e-10), list(lower_tail = TRUE, log_p = TRUE)),
        -0x1.64540d1292137p+9
      )
    ),
    check(
      "qexp lower log_p d/dp at p = -720 (small rate) is 1.7e-304, not 0: exp(p) is divided by rate before it underflows",
      near(
        sp_all$nv_qexp$ref_grad(-720, list(rate = pi * 1e-10), list(lower_tail = TRUE, log_p = TRUE))$p,
        0x1.c641058cb3e7ap-1008
      )
    ),
    check(
      "qexp lower log_p d/dp at p = +0 is +Inf, the limit from inside the domain, not -Inf from expm1(-0)",
      identical(sp_all$nv_qexp$ref_grad(0, list(rate = 1), list(lower_tail = TRUE, log_p = TRUE))$p, Inf)
    ),
    check(
      "qexp lower d/drate at the smallest subnormal p (small rate) divides by rate^2 in one step, not twice",
      near(
        sp_all$nv_qexp$ref_grad(2^-1074, list(rate = pi * 1e-10), list(lower_tail = TRUE, log_p = FALSE))$rate,
        -0x1.193907f0a4bb8p-1011
      )
    )
  )

  ## The validator itself: what it identifies, and how it compares.
  cat("\nvalidation:\n")
  sq <- sp_all$nv_qnorm
  truth_now <- truth_identity(sq$ref_grad_mpfr, "p")
  qe <- environment(sq$ref_grad_mpfr)
  mq <- qe$mp_qnorm
  qe$mp_qnorm <- function(pr, lower, log_p, prec) mq(pr, lower, log_p, prec) * 1
  truth_changed <- truth_identity(sq$ref_grad_mpfr, "p")
  qe$mp_qnorm <- mq
  gq <- build_grid(list(nv_qnorm = sq), "anvl")
  rowq <- gq[gq$kind == "grad", , drop = FALSE][1L, , drop = FALSE]
  rq <- data.frame(
    run_id = "A",
    cell_id = rowq$cell_id,
    output = "p",
    ref_params = hex_params(list(mean = 0, sd = 1)),
    ref_grad_id = "X",
    ref_grad_unsupported = NA_character_
  )
  ru_a <- reference_under_test(sq, rowq, rq)
  rq$run_id <- "B"
  ru_b <- reference_under_test(sq, rowq, rq)
  bad_truth <- list(
    reference = "gradient",
    stored = "X",
    now = "X",
    unsupported = NA,
    fun = identity,
    truth = function(x, p, f) base::get("abs")(x),
    output = "p",
    bound = 4,
    params = list(),
    flags = list()
  )
  vb <- validate_reference(
    bad_truth,
    rq,
    data.frame(x = numeric(0), source = character(0)),
    integer(0),
    VALIDATION_DEFAULTS
  )
  vl <- c(
    check("a truth's identity is its own function and dependencies, not the run that asked", {
      identical(truth_identity(ru_a$truth, ru_a$output), truth_identity(ru_b$truth, ru_b$output)) &&
        identical(truth_identity(ru_a$truth, ru_a$output), truth_now)
    }),
    check("... and changes when a helper it calls changes (mp_qnorm)", truth_changed != truth_now),
    check(
      "a truth that cannot be identified fails validation before anything is evaluated",
      !vb$record$pass && is.na(vb$record$truth_id) && grepl("truth has no identity", vb$record$reason)
    ),
    check("namespace qualification does not get past the dynamic-call refusal", {
      i1 <- stable_identity(function(x, p, f) base::get("abs")(x), 8, pp, fl, "f64")
      i2 <- stable_identity(function(x, p, f) sapply(x, "abs"), 8, pp, fl, "f64")
      is.na(i1) && is.na(i2)
    }),
    check("validations by another validation method count for nothing", {
      v <- data.frame(
        reference = "stable",
        ref_id = "A",
        output = "value",
        truth_id = "t1",
        pass = TRUE,
        identity_matches = TRUE,
        validated_at = "2026-09-26T10:00:00+0000",
        method_id = "old"
      )
      reference_status("A", NA, "stable", v) == "not validated" &&
        {
          v$method_id <- validation_method_id()
          reference_status("A", NA, "stable", v) == "validated"
        }
    }),
    check("validation shards partition the reference units: disjoint, complete, and the same everywhere", {
      k <- c(sprintf("gradient id%02d x", 1:10), "gradient id03 x", "stable s1 value")
      parts <- lapply(1:3, function(i) validation_units(k, i, 3L))
      all_u <- unlist(parts)
      !anyDuplicated(all_u) && setequal(all_u, unique(k)) && identical(validation_units(k), sort(unique(k)))
    }),
    check("earlier samples are replayed unless known to have passed: NA or a missing column is not a pass", {
      pv <- data.frame(x = 1:4, err_ulp64 = c(9, 9, 0.5, Inf), pass = c(FALSE, NA, TRUE, TRUE))
      old <- pv[c("x", "err_ulp64")]
      identical(replay_counterexamples(pv)$x, c(1L, 2L, 4L)) && identical(replay_counterexamples(old)$x, 1:4)
    }),
    check("qunif d/dp at log p = -745.2 and -732.1773 (wide, lower) keeps w exp(p) from underflowing early", {
      r <- sp_all$nv_qunif$ref_grad(
        c(-745.2, -732.17730000000006),
        list(min = -pi, max = 2 * pi),
        list(lower_tail = TRUE, log_p = TRUE)
      )$p
      identical(r, c(4, 1994919) * 2^-1074)
    }),
    if (requireNamespace("Rmpfr", quietly = TRUE)) {
      check("the comparator keeps fractions of a subnormal ulp, and applies the exact-zero rule", {
        u <- Rmpfr::mpfr(2, 256)^-1074
        tr <- c(4.4, 4.49, 0.4, 0) * u
        ## a reference of 0 against 4.4, 4.49 and 0.4 subnormal units, and of 4
        ## units against an exact zero
        c1 <- compare_to_truth(c(0, 0, 0, 4) * 2^-1074, tr, 4)
        c2 <- compare_to_truth(0, tr[3], 0.3)
        identical(c1$pass, c(FALSE, FALSE, TRUE, FALSE)) &&
          abs(c1$err[1] - 4.4) < 1e-9 &&
          abs(c1$err[2] - 4.49) < 1e-9 &&
          abs(c1$err[3] - 0.4) < 1e-9 &&
          !c2$pass
      })
    } else {
      cat("  skip the comparator checks: Rmpfr is not installed\n")
      TRUE
    }
  )

  ## What a store's newest attempts make current: A errored after succeeding,
  ## B succeeded after erroring, C succeeded at full and then errored at
  ## smoke, D was never run.
  cat("\ncoverage and selection:\n")
  at <- data.frame(
    cell_id = c("A", "A", "B", "B", "C", "C"),
    output = c("value", "-", "-", "value", "value", "-"),
    platform_key = "p",
    depth = c("smoke", "smoke", "smoke", "smoke", "full", "smoke"),
    run_id = c("r1", "r2", "r1", "r2", "r1", "r2"),
    started_at = c("t1", "t2", "t1", "t2", "t1", "t2"),
    error = c(NA, "boom A", "boom B", NA, NA, "boom C")
  )
  cg <- data.frame(
    cell_id = c("A", "B", "C", "D"),
    spec = "s",
    backend = "anvl",
    dtype = "f64",
    kind = "value",
    param_set = "q",
    flags = "-",
    n_outputs = 1L
  )
  cr <- current_results(at)
  cv <- coverage_table(cg, cr$attempts)
  cs <- c(
    check(
      "a result superseded by a newer error is not current; the error is",
      !"A" %in% cr$results$cell_id && "A" %in% cr$errors$cell_id
    ),
    check("an error superseded by a newer success is gone", "B" %in% cr$results$cell_id && !"B" %in% cr$errors$cell_id),
    check(
      "an error at one depth leaves a success at another current, and both are reported",
      identical(cr$results$depth[cr$results$cell_id == "C"], "full") && "C" %in% cr$errors$cell_id
    ),
    check(
      "coverage lists every declared cell: swept depth, newest error, or neither",
      identical(cv$cell_id, c("A", "B", "C", "D")) &&
        identical(cv$depth, c(NA, "smoke", "full", NA)) &&
        identical(cv$error, c("boom A", NA, "boom C", NA)) &&
        identical(cv$error_run_id, c("r2", NA, "r2", NA))
    ),
    check("an output filter selects among results, after the newest attempt is chosen", {
      cf <- current_results(at, "output=value")
      !"A" %in% cf$results$cell_id && "A" %in% cf$errors$cell_id
    }),
    check("a run with no start time is the oldest attempt, never a row of NAs", {
      an <- at
      an$started_at[an$run_id == "r2"] <- NA
      la <- latest_attempts(an)
      !anyNA(la$cell_id) && nrow(la) == nrow(an) - 2L && all(la$run_id[la$cell_id %in% c("A", "B")] == "r1")
    })
  )

  ## What `diff` pairs each row of a new run with: the cell's previous attempt
  ## at the same platform and depth. E succeeded and now errors, F errored and
  ## now succeeds although an older run of it succeeded too, G keeps erroring,
  ## H errors on its first attempt, and I is compared as usual.
  cat("\ndiff pairing:\n")
  dpast <- data.frame(
    cell_id = c("E", "F", "F", "G", "I"),
    output = c("value", "value", "-", "-", "value"),
    platform_key = "p",
    depth = "smoke",
    run_id = c("r1", "r1", "r2", "r2", "r2"),
    started_at = c("t1", "t1", "t2", "t2", "t2"),
    error = c(NA, NA, "boom F", "boom G1", NA)
  )
  dcur <- data.frame(
    cell_id = c("E", "F", "G", "H", "I"),
    output = c("-", "value", "-", "-", "value"),
    platform_key = "p",
    depth = "smoke",
    run_id = "r3",
    started_at = "t3",
    error = c("boom E", NA, "boom G2", "boom H", NA)
  )
  dpr <- diff_pairs(dcur, dpast)
  chg <- stats::setNames(dpr$errors$change, dpr$errors$cell_id)
  dfs <- c(
    check(
      "a cell that errors where its previous attempt succeeded is newly erroring",
      identical(unname(chg["E"]), "newly") && identical(dpr$errors$before_run[dpr$errors$cell_id == "E"], "r1")
    ),
    check("a result whose previous attempt errored has recovered, not been compared across the error", {
      f <- dpr$now$cell_id == "F"
      isTRUE(dpr$recovered[f]) &&
        !dpr$fresh[f] &&
        is.na(dpr$before$run_id[f]) &&
        identical(dpr$recovered_from$error[f], "boom F")
    }),
    check(
      "an error after an error is still erroring, with the earlier message kept",
      identical(unname(chg["G"]), "still") && identical(dpr$errors$before_error[dpr$errors$cell_id == "G"], "boom G1")
    ),
    check("an error on a first attempt is new", identical(unname(chg["H"]), "new")),
    check(
      "a result after a result is paired with it",
      identical(dpr$before$run_id[dpr$now$cell_id == "I"], "r2") && !dpr$fresh[dpr$now$cell_id == "I"]
    )
  )

  cat("\nassertions:\n")
  ok <- c(
    sp,
    ca0,
    ca,
    zv,
    ec,
    sw,
    stt,
    dp,
    idr,
    gr,
    vl,
    cs,
    dfs,
    check(
      "clean f64 value reproduces the reference exactly",
      get(sprintf(p, "f64", "value", "clean", "FALSE"))$worst_rel_err == 0
    ),
    check(
      "a one-ulp nudge is measured as exactly one ulp",
      get(sprintf(p, "f64", "value", "nudged", "FALSE"))$worst_ulp_err == 1
    ),
    check(
      "a zeroed tail is measured as relative error 1",
      get(sprintf(p, "f64", "value", "clean", "TRUE"))$worst_rel_err == 1
    ),
    check(
      "the same break against an infinite reference leaves an unclassified region",
      get(sprintf(p, "f32", "value", "clean", "TRUE"))$n_runs_unclassified == 1
    ),
    check(
      "a 1e-6 gradient error is ~4.5e9 ulp in f64",
      get(sprintf(p, "f64", "grad", "clean", "TRUE"), "scale")$worst_ulp_err > 4e9
    ),
    check(
      "the same error is ~8 ulp in f32",
      get(sprintf(p, "f32", "grad", "clean", "TRUE"), "scale")$worst_ulp_err > 8
    ),
    check(
      "the untouched gradient output is untouched",
      get(sprintf(p, "f64", "grad", "clean", "TRUE"), "x")$worst_rel_err == 0
    ),
    check("no cell errored", all(is.na(res$error)))
  )
  cat(sprintf("\n%d/%d assertions passed\n", sum(ok), length(ok)))
  if (!all(ok)) {
    quit(status = 1L)
  }
  invisible(TRUE)
}

## ---- comparing two runs -----------------------------------------------------
##
## "Did my fix help?"
##
## The sweep is deterministic -- fixed seed, fixed stride, same inputs every
## time -- so two runs of the same cell at the same depth on the same machine
## are bit-identical unless the code changed. That is a strong enough property
## to drop the usual fuzzy "meaningfully different" threshold entirely: any
## difference at all is a real one, and exact equality is a real "no change".

## Every result with its run's timestamp, so "the previous result for this
## cell" is well defined. Deliberately not deepest_per_cell(): the whole point
## here is the history that collapses away.
results_with_time <- function(dir) {
  res <- store_read(dir, "results")
  if (is.null(res)) {
    return(NULL)
  }
  runs <- store_read(dir, "runs")
  if (is.null(runs)) {
    return(NULL)
  }
  ## region and point counts resolved exactly as status and export see them
  res <- resummarise(res, resolved_ranges(dir), resolved_points(dir))
  merge(res, runs[c("run_id", "started_at", "anvl_sha", "branch")], by = "run_id", all.x = TRUE)
}

## Compare like with like. A smoke result and a full result of the same cell
## sample different inputs, so pairing them across depths would report the
## extra coverage as a regression.
diff_key <- function(d) paste(d$cell_id, d$output, d$platform_key, d$depth, sep = "\r")

## Pair one run's rows (`cur`) with what came before them (`past`). A cell's
## comparison point is its previous attempt at the same platform and depth --
## the most recent earlier run that swept it -- so a result is never compared
## with an older one across an error in between.
##
##   now, before   the successful results, each beside its previous result
##                 (NA where there is none)
##   fresh         results with no previous attempt at all
##   recovered     results whose previous attempt errored, and that error
##   errors        the errored cells, with `change` "newly" (the previous
##                 attempt succeeded), "still" (it errored too) or "new" (no
##                 previous attempt), and `before_run` / `before_error`
diff_pairs <- function(cur, past) {
  past <- latest_attempts(past)
  attempt <- function(d) paste(d$cell_id, d$platform_key, d$depth, sep = "\r")
  p_ok <- past[is.na(past$error), , drop = FALSE]
  p_err <- past[!is.na(past$error), , drop = FALSE]
  now <- cur[is.na(cur$error), , drop = FALSE]
  err <- cur[!is.na(cur$error), , drop = FALSE]

  i <- match(diff_key(now), diff_key(p_ok))
  j <- match(attempt(now), attempt(p_err))
  e_ok <- match(attempt(err), attempt(p_ok))
  e_err <- match(attempt(err), attempt(p_err))
  err$change <- ifelse(!is.na(e_ok), "newly", ifelse(!is.na(e_err), "still", "new"))
  err$before_run <- ifelse(!is.na(e_ok), p_ok$run_id[e_ok], p_err$run_id[e_err])
  err$before_error <- p_err$error[e_err]
  list(
    now = now,
    before = p_ok[i, , drop = FALSE],
    fresh = is.na(i) & is.na(j),
    recovered = is.na(i) & !is.na(j),
    recovered_from = p_err[j, , drop = FALSE],
    errors = err
  )
}

cmd_diff <- function(opt) {
  dir <- store_dir(opt$store)
  all <- results_with_time(dir)
  if (is.null(all) || !nrow(all)) {
    stop("store is empty; run a sweep first", call. = FALSE)
  }
  runs <- store_read(dir, "runs")
  runs <- runs[order(runs$started_at, decreasing = TRUE), , drop = FALSE]

  specs <- load_specs(include_selftest = grepl("selftest", opt$filter))
  g <- apply_filter(build_grid(specs, store_backends(opt, all)), opt$filter, extra = "output")
  ## An errored attempt has no output for an output term to select, so the
  ## filter's output term narrows the successful rows only.
  matching <- function(id) {
    r <- all[all$run_id == id & all$cell_id %in% g$cell_id, , drop = FALSE]
    rbind(filter_results(r[is.na(r$error), , drop = FALSE], opt$filter), r[!is.na(r$error), , drop = FALSE])
  }

  ## The newest run that actually contains matching results, not simply the
  ## newest run: a `selftest` or single-cell sweep in between would otherwise
  ## be chosen and then reported as empty.
  if (!is.null(opt$to)) {
    to_id <- opt$to
    if (!to_id %in% all$run_id) {
      stop("no results for run '", to_id, "'", call. = FALSE)
    }
    cur <- matching(to_id)
    if (!nrow(cur)) {
      stop("run '", to_id, "' has no results matching the filter", call. = FALSE)
    }
  } else {
    cur <- NULL
    for (id in runs$run_id) {
      cand <- matching(id)
      if (nrow(cand)) {
        to_id <- id
        cur <- cand
        break
      }
    }
    if (is.null(cur)) {
      stop("no run contains results matching that filter", call. = FALSE)
    }
  }

  ## The comparison point: an explicit --from, or otherwise each cell's
  ## previous attempt, which handles partial re-runs without needing them to
  ## line up as whole runs.
  t_now <- cur$started_at[1L]
  past <- all[(all$started_at < t_now) %in% TRUE & all$cell_id %in% g$cell_id, , drop = FALSE]
  if (!is.null(opt$from)) {
    past <- past[past$run_id == opt$from, , drop = FALSE]
    if (!nrow(past)) stop("no earlier results for run '", opt$from, "'", call. = FALSE)
  }
  dp <- diff_pairs(cur, past)
  now <- dp$now
  before <- dp$before
  fresh <- dp$fresh
  recovered <- dp$recovered
  err <- dp$errors
  compared <- !fresh & !recovered

  cat("\nanvl distribution sweeps — diff\n")
  earlier <- unique(c(before$run_id[compared], dp$recovered_from$run_id[recovered], err$before_run))
  earlier <- earlier[!is.na(earlier)]
  if (length(earlier)) {
    shas <- all$anvl_sha[match(earlier, all$run_id)]
    cat(sprintf(
      "  from  %s  anvl %s\n",
      if (length(earlier) == 1L) earlier else sprintf("%d earlier runs", length(earlier)),
      paste(unique(substr(shas, 1L, 7L)), collapse = ",")
    ))
  }
  cat(sprintf(
    "  to    %s  anvl %s  (%s)\n",
    to_id,
    substr(cur$anvl_sha[1L], 1L, 7L),
    paste(unique(cur$depth), collapse = ",")
  ))

  label <- function(d, k) {
    row <- g[g$cell_id == d$cell_id[k], , drop = FALSE][1L, , drop = FALSE]
    what <- if (d$kind[k] == "value") {
      "value"
    } else if (d$output[k] == "-") {
      "grad"
    } else {
      sprintf("d/d%s", d$output[k])
    }
    sprintf("\n  %-9s %-7s %-4s %s\n", d$spec[k], what, d$dtype[k], short_cell(row))
  }
  first_line <- function(e) vapply(strsplit(e, "\n", fixed = TRUE), `[`, "", 1L)
  err_lines <- function(k, note) {
    for (j in k) {
      cat(label(err, j))
      cat(sprintf("    error                %s\n", first_line(err$error[j])))
      n <- note(j)
      if (nzchar(n)) cat(sprintf("    %s\n", n))
    }
  }
  newly <- which(err$change == "newly")
  still <- which(err$change == "still")
  new_err <- which(err$change == "new")

  if (!any(compared) && !any(recovered) && !length(newly) && !length(still)) {
    ## Nothing to compare against is the normal state after a store reset or a
    ## first run, and reads as an error if the screen does not say so.
    cat(sprintf(
      paste0(
        "\n  Nothing to compare against: all %d result(s) are the first of their\n",
        "  cell at this depth. Re-run after a code change to see what moved.\n"
      ),
      nrow(now) + nrow(err)
    ))
    if (length(new_err)) {
      rule(sprintf("ERRORED (%d)", length(new_err)))
      err_lines(new_err, function(j) "")
    }
    cat("\n")
    return(invisible(NULL))
  }
  cat("\n  The sweep is deterministic, so on one machine any difference below was\n")
  cat("  caused by the code, not by measurement noise.\n")

  ## direction, per result: the worst finite error over sweep and points, and
  ## every category's region and point counts. More failures, boundary or
  ## backend findings is worse; a change only in conventions or signed zeros
  ## is a change, shown, but neither worse nor better.
  now$worst_rel_err <- result_state(now)$worst_any
  before$worst_rel_err <- result_state(before)$worst_any
  counts <- c(
    "failure regions" = "n_runs_unclassified",
    "failing points" = "n_points_failure",
    "boundary regions" = "n_regions_boundary",
    "boundary points" = "n_points_boundary",
    "backend regions" = "n_regions_backend",
    "backend points" = "n_points_backend",
    "convention regions" = "n_regions_domain",
    "convention points" = "n_points_domain",
    "signed-zero samples" = "n_zero_sign"
  )
  cnt <- function(d, nm) {
    v <- d[[nm]]
    if (is.null(v)) rep(0, nrow(d)) else ifelse(is.na(v), 0, v)
  }
  weighs <- counts[1:6]
  same_counts <- Reduce(`&`, lapply(counts, function(nm) cnt(now, nm) == cnt(before, nm)))
  more <- Reduce(`|`, lapply(weighs, function(nm) cnt(now, nm) > cnt(before, nm)))
  fewer <- Reduce(`|`, lapply(weighs, function(nm) cnt(now, nm) < cnt(before, nm)))
  same <- compared &
    (now$worst_rel_err == before$worst_rel_err | (is.na(now$worst_rel_err) & is.na(before$worst_rel_err))) &
    same_counts
  worse <- compared & !same & (now$worst_rel_err > before$worst_rel_err | more) %in% TRUE
  better <- compared & !same & !worse & (now$worst_rel_err < before$worst_rel_err | fewer) %in% TRUE
  moved <- compared & !same & !worse & !better

  line <- function(k) {
    cat(label(now, k))
    a <- before$worst_rel_err[k]
    z <- now$worst_rel_err[k]
    note <- if (is.na(a) || is.na(z)) {
      ""
    } else if (a == 0 && z > 0) {
      "   (was bit-identical everywhere)"
    } else if (z == 0 && a > 0) {
      "   (now bit-identical everywhere)"
    } else if (a > 0 && z > 0 && is.finite(z / a)) {
      sprintf("   (%.3gx %s)", max(z / a, a / z), if (z > a) "worse" else "better")
    } else {
      ""
    }
    if (!isTRUE(all.equal(a, z))) {
      cat(sprintf("    worst rel err        %s -> %s%s\n", fmt_num(a), fmt_num(z), note))
    }
    if (before$worst_ulp_err[k] != now$worst_ulp_err[k]) {
      cat(sprintf(
        "    worst ulp            %s -> %s\n",
        fmt_num(before$worst_ulp_err[k]),
        fmt_num(now$worst_ulp_err[k])
      ))
    }
    for (lab in names(counts)) {
      a <- cnt(before[k, , drop = FALSE], counts[[lab]])
      z <- cnt(now[k, , drop = FALSE], counts[[lab]])
      if (a != z) cat(sprintf("    %-20s %s -> %s\n", lab, format(a, big.mark = ","), format(z, big.mark = ",")))
    }
  }

  results <- function(which) {
    k <- which(which)
    k <- k[order(
      -abs(
        log10(pmax(now$worst_rel_err[k], 1e-300)) -
          log10(pmax(before$worst_rel_err[k], 1e-300))
      )
    )]
    for (j in utils::head(k, 15L)) {
      line(j)
    }
    if (length(k) > 15L) cat(sprintf("\n  ... and %d more.\n", length(k) - 15L))
  }
  rec <- which(recovered)
  rec_lines <- function() {
    for (k in rec) {
      cat(label(now, k))
      cat(sprintf(
        "    no longer errors     run %s: %s\n",
        dp$recovered_from$run_id[k],
        first_line(dp$recovered_from$error[k])
      ))
    }
  }

  ## A cell that errors where its previous attempt succeeded is the plainest
  ## regression there is, so it leads.
  n_worse <- length(newly) + sum(worse)
  if (n_worse) {
    rule(sprintf("REGRESSED (%d)", n_worse))
    err_lines(newly, function(j) sprintf("succeeded in run %s", err$before_run[j]))
    results(worse)
  }
  n_better <- length(rec) + sum(better)
  if (n_better) {
    rule(sprintf("IMPROVED (%d)", n_better))
    rec_lines()
    results(better)
  }
  if (any(moved)) {
    rule(sprintf("CHANGED (conventions or signed zeros only) (%d)", sum(moved)))
    results(moved)
  }
  if (length(still)) {
    rule(sprintf("STILL ERRORING (%d)", length(still)))
    err_lines(still, function(j) {
      if (identical(err$error[j], err$before_error[j])) "" else sprintf("was: %s", first_line(err$before_error[j]))
    })
  }
  if (length(new_err)) {
    rule(sprintf("ERRORED, NO EARLIER ATTEMPT (%d)", length(new_err)))
    err_lines(new_err, function(j) "")
  }

  rule("SUMMARY")
  cat(sprintf(
    "  %4d regressed%s\n",
    n_worse,
    if (length(newly)) sprintf(" (%d newly erroring)", length(newly)) else ""
  ))
  cat(sprintf(
    "  %4d improved%s\n",
    n_better,
    if (length(rec)) sprintf(" (%d no longer erroring)", length(rec)) else ""
  ))
  if (any(moved)) {
    cat(sprintf("  %4d changed in conventions or signed zeros only\n", sum(moved)))
  }
  cat(sprintf("  %4d unchanged (bit-identical to the earlier run)\n", sum(same)))
  if (length(still)) {
    cat(sprintf("  %4d still erroring\n", length(still)))
  }
  cat(sprintf("  %4d had no earlier result to compare against\n", sum(fresh) + length(new_err)))
  cat("\n")
  invisible(rbind(
    data.frame(
      cell_id = now$cell_id,
      output = now$output,
      change = ifelse(
        fresh,
        "new",
        ifelse(
          recovered,
          "improved",
          ifelse(same, "same", ifelse(worse, "regressed", ifelse(better, "improved", "changed")))
        )
      )
    ),
    data.frame(
      cell_id = err$cell_id,
      output = err$output,
      change = c(newly = "regressed", still = "still erroring", new = "new error")[err$change],
      row.names = NULL
    )
  ))
}

## ---- publishing a snapshot --------------------------------------------------
##
## The store accumulates: every run appends, and queries pick the latest per
## cell. A published artifact must instead be a single coherent snapshot --
## one result per cell, output and platform -- with the provenance that
## produced it, laid out so a browser can fetch only the part it needs.
##
## Layout, under --out -- one artifact per (anvl version, platform), holding
## every backend swept, so anvl and its JAX twin can be compared in one place:
##
##   manifest.json     what is here: schema version, platform, specs, depths,
##                     coverage counts, row counts. Small, fetched first, and
##                     readable without a Parquet reader so it can drive
##                     navigation on its own.
##   runs.parquet      the environment fingerprint of every run included
##   summary.parquet   the successful results of every cell of every function
##                     -- a few hundred rows, enough to drive the whole index
##                     and the cross-function overview
##   coverage.parquet  every declared cell: the depth it was swept at, and its
##                     newest error, so errored and never-run cells are visible
##   detail.parquet    the worst inputs, per binade
##   bands.parquet     the per-binade profile, one row per binade
##   hist.parquet      the error distribution
##   ranges.parquet    the no-finite-error regions
##   categories.parquet  per-result figures split by input class: normal,
##                     zero & subnormal, out of support, Inf & NaN
##
## One file per *table*, covering every function -- not one per function. The
## overview page summarises all functions at once, so splitting by function
## would mean fetching every piece anyway, in more requests, and would make
## switching platform a download of many files rather than one artifact.
##
## detail and bands are sorted by cell and their row groups aligned to cell
## boundaries. Parquet can only skip whole row groups, and nanoparquet defaults
## to one row group of ten million rows -- which would mean reading the entire
## file to drill into any single cell. Aligned, a reader fetches the footer and
## then just the groups it needs.

## Minimal JSON writer, so the manifest costs no extra dependency. Only the
## shapes used below are supported: named lists, atomic vectors, data frames.
## Non-finite numerics become null -- JSON cannot represent them, which is
## exactly why the measurements themselves travel as Parquet and never as JSON.
to_json <- function(x, indent = 0L) {
  pad <- strrep(" ", indent)
  esc <- function(s) {
    s <- gsub("\\", "\\\\", s, fixed = TRUE)
    s <- gsub('"', '\\"', s, fixed = TRUE)
    gsub("[[:cntrl:]]", "", s)
  }
  scalar <- function(v) {
    if (is.na(v)) {
      return("null")
    }
    if (is.logical(v)) {
      return(if (v) "true" else "false")
    }
    if (is.numeric(v)) {
      return(if (is.finite(v)) format(v, scientific = FALSE, trim = TRUE) else "null")
    }
    paste0('"', esc(as.character(v)), '"')
  }
  if (is.data.frame(x)) {
    rows <- vapply(
      seq_len(nrow(x)),
      function(i) to_json(as.list(x[i, , drop = FALSE]), indent + 2L),
      ""
    )
    return(paste0("[\n", paste0(strrep(" ", indent + 2L), rows, collapse = ",\n"), "\n", pad, "]"))
  }
  if (is.list(x)) {
    if (!length(x)) {
      return("{}")
    }
    kv <- vapply(
      names(x),
      function(n) paste0(pad, "  \"", esc(n), "\": ", to_json(x[[n]], indent + 2L)),
      ""
    )
    return(paste0("{\n", paste(kv, collapse = ",\n"), "\n", pad, "}"))
  }
  ## A field that is conceptually a list stays an array even with one element,
  ## so a reader never has to handle both shapes for the same key.
  if (length(x) == 1L && !inherits(x, "json_array")) {
    return(scalar(x))
  }
  paste0("[", paste(vapply(x, scalar, ""), collapse = ", "), "]")
}

json_array <- function(x) structure(x, class = c("json_array", class(x)))

cmd_export <- function(opt) {
  if (is.null(opt$out)) {
    stop("export needs --out <dir>", call. = FALSE)
  }
  dir <- store_dir(opt$store)
  all <- latest_results(dir)
  if (is.null(all)) {
    stop("store is empty; run a sweep first", call. = FALSE)
  }
  specs <- load_specs()
  ## Export what the store holds. Falling back to run's anvl-only default here
  ## once published a full anvl+JAX sweep with every JAX result missing.
  backends <- if (isTRUE(opt$backends_given)) opt$backends else sort(unique(all$backend))
  g <- apply_filter(build_grid(specs, backends), opt$filter, extra = "output")
  cur <- current_results(all[all$cell_id %in% g$cell_id, , drop = FALSE], opt$filter)
  res <- cur$results
  if (!nrow(res)) {
    stop("no successful results to export for that filter; see `run.R status`", call. = FALSE)
  }
  ## Every declared cell, so a reader can tell a cell that errored or was
  ## never run from one that does not exist.
  coverage <- coverage_table(g, cur$attempts)

  ## Each cell's valid input domain and distribution support, resolved from its
  ## params and flags exactly as the sweep resolved them -- at the precision
  ## the implementation receives. Carried on the summary, so a reader can shade
  ## either part of the axis without asking the harness.
  cells <- g[match(unique(res$cell_id), g$cell_id), , drop = FALSE]
  bounds <- lapply(seq_len(nrow(cells)), function(i) {
    row <- cells[i, ]
    spec <- specs[[row$spec]]
    cf <- cell_functions(spec, row)
    d <- if (is.null(spec$domain)) c(-Inf, Inf) else spec$domain(cf$ref_params, cf$flags)
    u <- if (is.null(spec$support)) c(NA_real_, NA_real_) else spec$support(cf$ref_params, cf$flags)
    c(d, u)
  })
  support <- data.frame(
    cell_id = cells$cell_id,
    domain_lo = vapply(bounds, `[`, 0, 1L),
    domain_hi = vapply(bounds, `[`, 0, 2L),
    support_lo = vapply(bounds, `[`, 0, 3L),
    support_hi = vapply(bounds, `[`, 0, 4L)
  )
  m <- match(res$cell_id, support$cell_id)
  res$domain_lo <- support$domain_lo[m]
  res$domain_hi <- support$domain_hi[m]
  res$support_lo <- support$support_lo[m]
  res$support_hi <- support$support_hi[m]

  out <- normalizePath(opt$out, mustWork = FALSE)
  dir.create(out, recursive = TRUE, showWarnings = FALSE)

  runs <- store_read(dir, "runs")
  runs <- runs[runs$run_id %in% unique(res$run_id), , drop = FALSE]

  ## Keyed on exactly the rows kept above, so a detail row from a superseded
  ## run can never leak in beside a newer summary row.
  ##
  ## Regions and points are taken already resolved against the WHOLE store,
  ## and only then filtered. Resolving after filtering would let a
  ## gradient-only export drop the value cells whose evidence settles a
  ## convention, turning that convention into an exported failure. `res`
  ## came from latest_results(), which summarised the same resolved tables,
  ## so its region and point counts agree with these.
  keep <- paste(res$run_id, res$cell_id, res$output)
  resolved <- list(ranges = resolved_ranges(dir), points = resolved_points(dir))
  pick <- function(tbl) {
    x <- if (tbl %in% names(resolved)) resolved[[tbl]] else store_read(dir, tbl)
    if (is.null(x) || !nrow(x)) {
      return(NULL)
    }
    x[paste(x$run_id, x$cell_id, x$output) %in% keep, , drop = FALSE]
  }

  ## Row groups aligned to cell boundaries: sort, then start a new group
  ## wherever the cell changes, so a reader can fetch one cell's rows alone.
  write_by_cell <- function(x, path) {
    x <- x[order(x$cell_id, x$output), , drop = FALSE]
    rownames(x) <- NULL
    starts <- which(!duplicated(x$cell_id))
    nanoparquet::write_parquet(x, path, row_groups = as.integer(starts))
    nrow(x)
  }

  kept <- list()
  for (tbl in c("detail", "bands", "hist", "ranges", "kinds", "points", "disputes")) {
    x <- pick(tbl)
    if (!is.null(x) && nrow(x)) kept[[tbl]] <- x
  }
  ## Every validation of a reference identity these results carry, and its
  ## recorded samples: what justifies each exclusion travels with it.
  ids <- unique(stats::na.omit(c(res$ref_stable_id, res$ref_grad_id)))
  vtabs <- list()
  for (tbl in c("validations", "validation_samples")) {
    x <- store_read(dir, tbl)
    if (!is.null(x) && nrow(x)) {
      x <- x[x$ref_id %in% ids, , drop = FALSE]
      if (nrow(x)) vtabs[[tbl]] <- x
    }
  }

  nanoparquet::write_parquet(runs, file.path(out, "runs.parquet"))
  nanoparquet::write_parquet(res[order(res$cell_id, res$output), , drop = FALSE], file.path(out, "summary.parquet"))
  nanoparquet::write_parquet(coverage, file.path(out, "coverage.parquet"))
  counts <- c(runs = nrow(runs), summary = nrow(res), coverage = nrow(coverage))
  for (tbl in names(vtabs)) {
    nanoparquet::write_parquet(vtabs[[tbl]], file.path(out, paste0(tbl, ".parquet")))
    counts[tbl] <- nrow(vtabs[[tbl]])
  }
  for (tbl in names(kept)) {
    counts[tbl] <- write_by_cell(kept[[tbl]], file.path(out, paste0(tbl, ".parquet")))
  }

  ## Per-result figures split by input class (see input_class()). A few rows
  ## per result, so the overview can show normal-input accuracy without ever
  ## fetching `bands`.
  if (!is.null(kept$bands) && !is.null(kept$detail)) {
    cats <- category_table(kept$bands, kept$detail, support)
    nanoparquet::write_parquet(cats, file.path(out, "categories.parquet"))
    counts["categories"] <- nrow(cats)
  }

  manifest <- list(
    schema_version = SCHEMA_VERSION,
    exported_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    platforms = json_array(sort(unique(res$platform_key))),
    backends = json_array(sort(unique(res$backend))),
    devices = json_array(sort(unique(res$device))),
    depths = json_array(sort(unique(res$depth))),
    specs = json_array(sort(unique(res$spec))),
    n_cells = length(unique(res$cell_id)),
    n_results = nrow(res),
    n_cells_declared = nrow(coverage),
    n_cells_errored = sum(!is.na(coverage$error)),
    n_cells_not_run = sum(is.na(coverage$depth) & is.na(coverage$error)),
    anvl_version = json_array(sort(unique(runs$anvl_version))),
    anvl_sha = json_array(sort(unique(runs$anvl_sha))),
    files = data.frame(table = names(counts), rows = as.integer(counts))
  )
  writeLines(to_json(manifest), file.path(out, "manifest.json"))

  files <- list.files(out, recursive = TRUE, full.names = TRUE)
  cat(sprintf(
    "exported %d results across %d cell(s) to %s\n",
    nrow(res),
    length(unique(res$cell_id)),
    out
  ))
  cat(sprintf(
    "  platforms: %s | backends: %s | depth: %s | %.1f MB in %d file(s)\n",
    paste(manifest$platforms, collapse = ", "),
    paste(manifest$backends, collapse = ", "),
    paste(manifest$depths, collapse = ", "),
    sum(file.size(files)) / 1024^2,
    length(files)
  ))
  if (manifest$n_cells_errored || manifest$n_cells_not_run) {
    cat(sprintf(
      "  WARNING: of %d declared cells, %d errored and %d were never run; the site shows them as such\n",
      manifest$n_cells_declared,
      manifest$n_cells_errored,
      manifest$n_cells_not_run
    ))
  }
  invisible(out)
}

cmd_merge <- function(opt) {
  if (is.null(opt$from)) {
    stop("merge needs --from <dir>", call. = FALSE)
  }
  into <- store_dir(opt$store)
  store_init(into)
  n <- store_merge(normalizePath(opt$from, mustWork = TRUE), into)
  cat(sprintf("merged %d new part file(s) into %s\n", n, into))
}

## ---- dispatch --------------------------------------------------------------

main <- function() {
  a <- parse_args(commandArgs(trailingOnly = TRUE))
  switch(
    a$cmd,
    list = cmd_list(a$opt),
    run = cmd_run(a$opt),
    status = cmd_status(a$opt),
    selftest = cmd_selftest(a$opt),
    diff = cmd_diff(a$opt),
    export = cmd_export(a$opt),
    merge = cmd_merge(a$opt),
    `validate-refs` = cmd_validate_refs(a$opt),
    stop(
      "unknown command '",
      a$cmd,
      "'; expected list, run, status, diff, export, merge, selftest or validate-refs",
      call. = FALSE
    )
  )
}

if (!interactive()) {
  main()
}
