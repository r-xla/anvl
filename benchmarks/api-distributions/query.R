#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## Reading the store.
##
## Source it for an R session:
##
##   source("query.R")
##   sw_results()                       # the latest result for every cell
##   sw_worst(spec = "nv_qnorm", n = 20)
##   sw_detail("nv_qnorm/anvl/f64/value/standard/lower_tail=TRUE,log_p=TRUE")
##   sw_ranges(category = "failure")
##   sw_points(failing = TRUE)          # the exact points that disagree
##   sw_bands("nv_qunif/anvl/f32/value/wide/lower_tail=TRUE,log_p=FALSE")
##   sw_hist("nv_pnorm/anvl/f64/value/standard/lower_tail=TRUE,log_p=FALSE")
##   sw_compare("darwin-arm64-cpu", "linux-x86_64-cuda")
##   sw_sql("select spec, max(worst_rel_err) from results group by spec")
##
## or from the shell:
##
##   Rscript query.R worst --spec nv_qnorm --n 20
##   Rscript query.R ranges --category failure
##   Rscript query.R points --category failure
##   Rscript query.R runs
##
## Everything reads Parquet through nanoparquet, which is the only hard
## dependency. sw_sql() additionally needs duckdb; it is worth installing on
## the machine you analyse on, and worth *not* installing on a compute node.
## ---------------------------------------------------------------------------

local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", grep("^--file=", a, value = TRUE))
  HERE <<- if (length(f)) dirname(normalizePath(f[1L])) else normalizePath(".")
})
here <- function() HERE
for (f in c("util.R", "engine.R", "cells.R", "provenance.R", "store.R", "validate.R")) {
  source(file.path(HERE, "R", f))
}

## ---- the tables ------------------------------------------------------------

sw_store <- function() store_dir()

## Latest result per (cell, output, platform, depth).
sw_results <- function(spec = NULL, dtype = NULL, kind = NULL, platform = NULL) {
  r <- latest_results(sw_store())
  if (is.null(r)) {
    return(NULL)
  }
  if (!is.null(spec)) {
    r <- r[r$spec %in% spec, , drop = FALSE]
  }
  if (!is.null(dtype)) {
    r <- r[r$dtype %in% dtype, , drop = FALSE]
  }
  if (!is.null(kind)) {
    r <- r[r$kind %in% kind, , drop = FALSE]
  }
  if (!is.null(platform)) {
    r <- r[r$platform_key %in% platform, , drop = FALSE]
  }
  rownames(r) <- NULL
  r
}

## The cells with the largest errors, worst first -- the usual first question.
sw_worst <- function(n = 20, metric = c("rel", "ulp"), ...) {
  metric <- match.arg(metric)
  r <- sw_results(...)
  if (is.null(r)) {
    return(NULL)
  }
  r$worst_any <- result_state(r)$worst_any
  col <- if (metric == "rel") "worst_any" else "worst_ulp_err"
  r <- r[order(-r[[col]]), , drop = FALSE]
  cols <- c(
    "cell_id",
    "output",
    "dtype",
    "worst_rel_err",
    "worst_point_rel_err",
    "worst_ulp_err",
    "n_runs_unclassified",
    "n_points_failure"
  )
  utils::head(r[intersect(cols, names(r))], n)
}

## The top-K worst individual inputs for one cell, with their bit patterns --
## copy a `bits` value straight into a regression test.
sw_detail <- function(cell_id, output = NULL, n = 20) {
  d <- store_read(sw_store(), "detail")
  if (is.null(d)) {
    return(NULL)
  }
  d <- d[d$cell_id == cell_id, , drop = FALSE]
  if (!is.null(output)) {
    d <- d[d$output == output, , drop = FALSE]
  }
  d <- d[order(-d$rel_err), , drop = FALSE]
  utils::head(d[c("output", "rank", "bits", "x", "value", "reference", "rel_err", "ulp_err")], n)
}

## The intervals of the number line where the two sides disagreed with no
## finite error, with the tested cause of each and its category.
## `category = "failure"` is the list of things nothing accounts for.
## `class` is kept for stores written before causes were recorded.
sw_ranges <- function(cell_id = NULL, category = NULL, cause = NULL, class = NULL) {
  r <- resolved_ranges(sw_store())
  if (is.null(r)) {
    return(NULL)
  }
  if (!is.null(cell_id)) {
    r <- r[r$cell_id %in% cell_id, , drop = FALSE]
  }
  if (!is.null(category)) {
    r <- r[r$category %in% category, , drop = FALSE]
  }
  if (!is.null(cause)) {
    r <- r[r$cause %in% cause, , drop = FALSE]
  }
  if (!is.null(class)) {
    r <- r[r$class %in% class, , drop = FALSE]
  }
  rownames(r) <- NULL
  r
}

## The mandatory exact points (+-0, +-Inf, NaN, extremes, 1/2, 1, domain and
## support edges, branch points, and their neighbours), resolved like regions.
## `failing = TRUE` keeps those with no finite error; `category` narrows those.
sw_points <- function(cell_id = NULL, output = NULL, failing = FALSE, category = NULL) {
  p <- resolved_points(sw_store())
  if (is.null(p)) {
    return(NULL)
  }
  if (!is.null(cell_id)) {
    p <- p[p$cell_id %in% cell_id, , drop = FALSE]
  }
  if (!is.null(output)) {
    p <- p[p$output %in% output, , drop = FALSE]
  }
  if (isTRUE(failing) || !is.null(category)) {
    p <- p[p$failure, , drop = FALSE]
  }
  if (!is.null(category)) {
    p <- p[p$category %in% category, , drop = FALSE]
  }
  rownames(p) <- NULL
  p[c(
    "cell_id",
    "output",
    "label",
    "x",
    "bits",
    "value",
    "reference",
    "rel_err",
    "zero_sign",
    "failure",
    "cause",
    "category",
    intersect("evidence", names(p))
  )]
}

## The error distribution for one cell: is the worst case one pathological
## input, or a floor under everything?
sw_hist <- function(cell_id, output = "value") {
  h <- store_read(sw_store(), "hist")
  if (is.null(h)) {
    return(NULL)
  }
  h <- h[h$cell_id == cell_id & h$output == output & h$count > 0, , drop = FALSE]
  h$rel_err <- 10^h$decade
  h[c("decade", "rel_err", "count")]
}

## Collapse the per-binade profile into runs that behave alike.
##
## Adjacent binades merge when they share a sign, a behaviour and an *upper*
## error bound. The upper bound alone is the key because within one binade the
## finite errors routinely span a dozen decades -- a few inputs round almost
## exactly while most sit on the precision floor -- so a lower bound carries no
## information and keying on it fragments the table threefold. m_best is still
## reported for each merged run.
##
## The store keeps every binade, so this compact view is derived from the same
## numbers a chart of the full series uses.
merge_bands <- function(b) {
  if (is.null(b) || !nrow(b)) {
    return(b)
  }
  if (is.null(b$zero)) {
    b$zero <- FALSE
  }
  b <- b[order(b$special, b$sign, b$binade, !b$zero), , drop = FALSE]
  key <- paste(b$sign, b$special, b$behaviour, b$m_worst)
  ## the Inf/NaN field and each sign's zero are rows of their own, never
  ## merged into a neighbouring range
  alone <- b$special | b$zero
  contiguous <- c(TRUE, diff(b$binade) != 1L | key[-1L] != key[-length(key)] | alone[-1L] | alone[-nrow(b)])
  ## the zero row and binade 0's subnormals share a binade number
  contiguous[c(FALSE, diff(b$binade) == 0L)] <- TRUE
  grp <- cumsum(contiguous)

  agg <- function(f, col) vapply(split(b[[col]], grp), f, numeric(1))
  na_min <- function(v) if (all(is.na(v))) NA_real_ else min(v, na.rm = TRUE)
  na_max <- function(v) if (all(is.na(v))) NA_real_ else max(v, na.rm = TRUE)

  out <- data.frame(
    sign = agg(function(v) v[1L], "sign"),
    x_from = agg(na_min, "x_from"),
    x_to = agg(na_max, "x_to"),
    special = vapply(split(b$special, grp), function(v) v[1L], logical(1)),
    zero = vapply(split(b$zero, grp), function(v) v[1L], logical(1)),
    behaviour = vapply(split(b$behaviour, grp), function(v) v[1L], ""),
    n_identical = agg(sum, "n_identical"),
    n_differ = agg(sum, "n_differ"),
    n_nonfinite = agg(sum, "n_nonfinite"),
    m_worst = agg(na_min, "m_worst"),
    m_best = agg(na_max, "m_best"),
    worst_rel_err = agg(max, "worst_rel_err"),
    n_binades = as.integer(table(grp))
  )
  rownames(out) <- NULL
  ## x_from is NA on the Inf/NaN rows, so sign breaks the tie and their order
  ## does not drift between runs.
  out[order(out$special, out$x_from, ifelse(out$zero, out$sign, -out$sign)), , drop = FALSE]
}

## Where along the number line the behaviour changes: runs of binades that
## agree, differ, or produce no finite answer. `merged = TRUE` gives the
## compact view: runs of binades sharing a behaviour and an upper error bound.
## `merged = FALSE` gives what is actually stored -- one row per binade, 256
## per sign in f32 and 2048 in f64 -- which is the series a chart wants.
sw_bands <- function(cell_id = NULL, output = NULL, merged = TRUE) {
  b <- store_read(sw_store(), "bands")
  if (is.null(b)) {
    return(NULL)
  }
  if (!is.null(cell_id)) {
    b <- b[b$cell_id %in% cell_id, , drop = FALSE]
  }
  if (!is.null(output)) {
    b <- b[b$output %in% output, , drop = FALSE]
  }
  if (!nrow(b)) {
    return(b)
  }
  if (merged) {
    ## merging is only meaningful within a single result
    b <- do.call(
      rbind,
      lapply(
        split(b, paste(b$cell_id, b$output)),
        function(g) {
          cbind(cell_id = g$cell_id[1L], output = g$output[1L], merge_bands(g))
        }
      )
    )
    rownames(b) <- NULL
    return(b[c(
      "cell_id",
      "output",
      "x_from",
      "x_to",
      "behaviour",
      "m_worst",
      "m_best",
      "worst_rel_err",
      "n_binades"
    )])
  }
  b <- b[order(b$cell_id, b$output, b$special, b$sign, b$binade), , drop = FALSE]
  rownames(b) <- NULL
  b[intersect(
    c(
      "cell_id",
      "output",
      "sign",
      "binade",
      "zero",
      "x_from",
      "x_to",
      "behaviour",
      "m_worst",
      "m_best",
      "worst_rel_err"
    ),
    names(b)
  )]
}

sw_runs <- function() {
  r <- store_read(sw_store(), "runs")
  if (is.null(r)) {
    return(NULL)
  }
  r <- r[order(r$started_at, decreasing = TRUE), , drop = FALSE]
  r[c("run_id", "started_at", "host", "platform_key", "device", "depth", "branch", "anvl_sha")]
}

## Same cell, two platforms, side by side -- the reason platform_key exists.
sw_compare <- function(a, b, metric = "worst_rel_err") {
  r <- sw_results()
  if (is.null(r)) {
    return(NULL)
  }
  ra <- r[r$platform_key == a, c("cell_id", "output", metric)]
  rb <- r[r$platform_key == b, c("cell_id", "output", metric)]
  m <- merge(ra, rb, by = c("cell_id", "output"), suffixes = c(paste0(".", a), paste0(".", b)))
  m$ratio <- m[[3L]] / m[[4L]]
  m[order(-m$ratio), ]
}

## Arbitrary SQL over the whole store, for when the helpers run out. DuckDB
## reads the Parquet tree directly; nothing is loaded or copied.
sw_sql <- function(query) {
  if (!requireNamespace("duckdb", quietly = TRUE)) {
    stop("sw_sql() needs the duckdb package: install.packages(\"duckdb\")", call. = FALSE)
  }
  con <- DBI::dbConnect(duckdb::duckdb())
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE))
  for (t in TABLES) {
    d <- file.path(sw_store(), t)
    if (!dir.exists(d) || !length(list.files(d, "\\.parquet$", recursive = TRUE))) {
      next
    }
    DBI::dbExecute(
      con,
      sprintf(
        "create view %s as select * from read_parquet('%s/**/*.parquet', union_by_name = true)",
        t,
        d
      )
    )
  }
  DBI::dbGetQuery(con, query)
}

## ---- shell interface -------------------------------------------------------

if (!interactive() && length(commandArgs(trailingOnly = TRUE))) {
  argv <- commandArgs(trailingOnly = TRUE)
  cmd <- argv[1L]
  kv <- list()
  i <- 2L
  while (i < length(argv)) {
    kv[[sub("^--", "", argv[i])]] <- argv[i + 1L]
    i <- i + 2L
  }
  num <- function(x, d) if (is.null(x)) d else as.integer(x)
  out <- switch(
    cmd,
    worst = sw_worst(n = num(kv$n, 20), spec = kv$spec, dtype = kv$dtype, kind = kv$kind),
    results = sw_results(spec = kv$spec, dtype = kv$dtype),
    detail = sw_detail(kv$cell, kv$output, n = num(kv$n, 20)),
    ranges = sw_ranges(cell_id = kv$cell, category = kv$category, cause = kv$cause, class = kv$class),
    points = sw_points(
      cell_id = kv$cell,
      output = kv$output,
      failing = identical(kv$failing, "1"),
      category = kv$category
    ),
    hist = sw_hist(kv$cell, kv$output %||% "value"),
    bands = sw_bands(cell_id = kv$cell, output = kv$output, merged = !identical(kv$raw, "1")),
    runs = sw_runs(),
    sql = sw_sql(kv$q),
    stop(
      "unknown query '",
      cmd,
      "'; try worst, results, detail, ranges, points, bands, hist, runs, sql",
      call. = FALSE
    )
  )
  if (is.null(out) || !nrow(out)) cat("no rows\n") else print(out, right = FALSE, digits = 6)
}
