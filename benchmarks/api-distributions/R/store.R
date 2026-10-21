## ---------------------------------------------------------------------------
## The result store.
##
## Parquet, in a partitioned directory, written once and never mutated.
##
## That last clause is the whole design. Every cell writes its own file, so two
## processes never touch the same bytes and there is no lock to take, no
## contention to tune and nothing that behaves differently on a network
## filesystem. Parallel jobs, a second machine and an HPC array are all the
## same case, and merging results from different machines is `cp -r`.
##
## This is why SQLite and DuckDB are not used as the write target: SQLite's
## locking is documented-unreliable on NFS/Lustre, and DuckDB permits only one
## writer process per database file. DuckDB is excellent on the *read* side and
## query.R will use it when it is installed -- but nothing depends on it.
##
## Parquet earns its place on type fidelity rather than size: the store is
## small, but it holds NaN, +-Inf, signed zero and subnormals, all of which a
## CSV round-trip would quietly destroy. Verified against nanoparquet, not
## assumed. The one thing that does *not* survive is a 64-bit integer, which
## comes back widened to a double -- so bit patterns are stored as hex strings.
## ---------------------------------------------------------------------------

## Every table a run writes. merge() copies exactly these, so a table missing
## here would be silently left behind when shards come back from a cluster.
TABLES <- c(
  "runs",
  "results",
  "detail",
  "ranges",
  "hist",
  "bands",
  "kinds",
  "points",
  "disputes",
  "validations",
  "validation_samples"
)

## Bumped whenever the shape of an exported artifact changes in a way a reader
## must know about. It travels in the manifest so a website can refuse, or
## adapt to, an artifact it does not understand rather than mis-rendering it.
## 2: regions carry a tested cause and a category rather than a positional
##    class; new tables `kinds` (what each side returned, per binade) and
##    `points` (the exact-point checks); references use precision-rounded
##    parameters.
## 3: bands have a `zero` row per sign, apart from binade 0's subnormals, and
##    categories a `zero` input class; ranges and points carry `evidence`;
##    summary carries exact-point figures and failing-sample counts per
##    category; n_flushed_zero_error beside n_flushed.
## 4: candidate base R disputes against a spec's stable reference: a
##    `disputes` table, `ref_candidate` on ranges and points, candidate counts
##    and candidate-filtered worst errors on bands, hist, categories and
##    summary, and the stable reference's identity on summary. Candidates are
##    not exclusions until validated.
## 5: summary carries `ref_params` (the exact reference parameters, hex);
##    bands and categories carry `worst_ulp_err` from its own accumulator, and
##    summary's worst_ulp_err is its maximum rather than the relative-error
##    shortlist's; hist counts are doubles.
## 6: `reference_limitation` category (candidates whose stable reference passed
##    validation); summary carries ref_stable_status / ref_grad_status and the
##    gradient reference identity; `validations` and `validation_samples`
##    tables travel with the results they justify.
## 7: validations carry `method_id` (records by any other validation method
##    count for nothing -- every record before the MPFR-only comparator) and a
##    truth identity of the spec's own truth function; validation_samples
##    carry each sample's `pass`.
## 8: a `coverage` table: every declared cell with the depth of its deepest
##    successful sweep and its newest error. Summary holds successful results
##    only, selected from the newest attempt per cell, platform and depth.
SCHEMA_VERSION <- 8L

## Where the store lives. Out of the package tree by default, so a result file
## can never be committed by accident and the package stays what upstream
## tracks: scripts, config and docs. NV_SWEEP_STORE overrides it, which is also
## how a compute node is pointed at cluster scratch.
store_dir <- function(override = NULL) {
  d <- override %||% Sys.getenv("NV_SWEEP_STORE", "")
  if (!nzchar(d)) {
    d <- file.path(tools::R_user_dir("anvl-sweeps", "cache"), "store")
  }
  d
}

store_init <- function(dir) {
  for (t in TABLES) {
    dir.create(file.path(dir, t), recursive = TRUE, showWarnings = FALSE)
  }
  invisible(dir)
}

## One cell's contribution to one table. The filename carries the run and the
## cell, so a collision is impossible by construction rather than by locking.
store_write <- function(dir, table, run_id, key, df) {
  if (is.null(df) || !nrow(df)) {
    return(invisible(NULL))
  }
  d <- file.path(dir, table, paste0("run=", run_id))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  f <- file.path(d, paste0(key, ".parquet"))
  ## Write to a temporary name in the same directory and rename into place:
  ## rename is atomic, so a reader never sees a half-written file and an
  ## interrupted job leaves no torn part behind.
  tmp <- paste0(f, ".tmp", Sys.getpid())
  nanoparquet::write_parquet(df, tmp)
  file.rename(tmp, f)
  invisible(f)
}

## Read one table whole. The store is small enough that this is always fine;
## query.R offers DuckDB for anyone who would rather push predicates down.
store_read <- function(dir, table, runs = NULL) {
  d <- file.path(dir, table)
  if (!dir.exists(d)) {
    return(NULL)
  }
  files <- list.files(d, pattern = "\\.parquet$", recursive = TRUE, full.names = TRUE)
  ## Only these runs' partitions (run=<id>/), for a large table read for a few
  ## results rather than for the whole history.
  if (!is.null(runs)) {
    files <- files[basename(dirname(files)) %in% paste0("run=", runs)]
  }
  if (!length(files)) {
    return(NULL)
  }
  parts <- lapply(files, function(f) as.data.frame(nanoparquet::read_parquet(f)))
  ## Shards written by different versions of a spec can differ in columns;
  ## union them rather than failing, so an older run stays readable.
  cols <- unique(unlist(lapply(parts, names)))
  parts <- lapply(parts, function(p) {
    for (m in setdiff(cols, names(p))) {
      p[[m]] <- NA
    }
    p[cols]
  })
  do.call(rbind, parts)
}

## Merge one store into another -- the cross-machine case. Parts are immutable
## and uniquely named, so this is a copy that can be repeated safely; an
## existing destination file is left alone rather than rewritten.
store_merge <- function(from, into) {
  n <- 0L
  for (t in TABLES) {
    src <- file.path(from, t)
    if (!dir.exists(src)) {
      next
    }
    for (f in list.files(src, pattern = "\\.parquet$", recursive = TRUE)) {
      dst <- file.path(into, t, f)
      if (file.exists(dst)) {
        next
      }
      dir.create(dirname(dst), recursive = TRUE, showWarnings = FALSE)
      file.copy(file.path(src, f), dst)
      n <- n + 1L
    }
  }
  n
}

## Collapse to one row per (cell, output): the deepest sweep available.
##
## `latest_results()` keeps a row per depth, which is what coverage wants, but
## every other consumer wants one current result per measurement.
## A shallower sweep visits a subset of a deeper one's inputs -- same index
## space, larger stride -- so the deeper row is strictly better evidence.
## Called through current_results(), so `status`, `export` and `validate-refs`
## can never disagree about what "the current result" is.
deepest_per_cell <- function(res, depths) {
  if (is.null(res) || !nrow(res)) {
    return(res)
  }
  rank <- match(res$depth, depths)
  res <- res[order(res$cell_id, res$output, -rank), , drop = FALSE]
  res <- res[!duplicated(paste(res$cell_id, res$output, sep = "\r")), , drop = FALSE]
  rownames(res) <- NULL
  res
}

## The newest attempt at each cell, per platform and depth: every row of the
## most recent run that swept it. A cell whose newest attempt errored has
## errored, whatever an earlier run of it recorded.
##
## Ordered as latest_results() orders: a run with no start time (no row in
## `runs`) sorts oldest, and run_id breaks ties, so exactly one run is newest.
latest_attempts <- function(res) {
  t <- res$started_at %||% res$run_id
  k <- paste(res$cell_id, res$platform_key, res$depth, sep = "\r")
  o <- order(t, res$run_id, decreasing = TRUE)
  first <- o[!duplicated(k[o])]
  newest <- stats::setNames(res$run_id[first], k[first])
  res[res$run_id == newest[k], , drop = FALSE]
}

## What the store currently holds, as `status`, `export` and `validate-refs`
## all read it: the newest attempts, the successful ones collapsed to the
## deepest result per cell and output, and the ones that errored.
##
## `filter`'s output term selects among the results only, after the newest
## attempt is chosen: an errored attempt has no output, so filtering first
## would drop it and let an older success stand in for it.
current_results <- function(res, filter = "") {
  att <- latest_attempts(res)
  failed <- !is.na(att$error)
  list(
    attempts = att,
    results = filter_results(deepest_per_cell(att[!failed, , drop = FALSE], names(DEPTHS)), filter),
    errors = att[failed, , drop = FALSE]
  )
}

## Every declared cell on every platform the attempts cover, with the depth of
## its deepest successful sweep and the newest error at its deepest errored
## depth -- each NA where there is none. A cell with neither was never run.
coverage_table <- function(g, att) {
  platforms <- sort(unique(att$platform_key))
  cols <- c("cell_id", "spec", "backend", "dtype", "kind", "param_set", "flags", "n_outputs")
  cov <- g[rep(seq_len(nrow(g)), length(platforms)), cols, drop = FALSE]
  cov$platform_key <- rep(platforms, each = nrow(g))
  key <- paste(cov$cell_id, cov$platform_key, sep = "\r")
  rank <- match(att$depth, names(DEPTHS))
  deepest <- function(rows) {
    rows <- rows[order(-rank[rows])]
    rows[!duplicated(paste(att$cell_id[rows], att$platform_key[rows], sep = "\r"))]
  }
  ok <- deepest(which(is.na(att$error)))
  er <- deepest(which(!is.na(att$error)))
  m_ok <- ok[match(key, paste(att$cell_id[ok], att$platform_key[ok], sep = "\r"))]
  m_er <- er[match(key, paste(att$cell_id[er], att$platform_key[er], sep = "\r"))]
  cov$depth <- att$depth[m_ok]
  cov$error_depth <- att$depth[m_er]
  cov$error_run_id <- att$run_id[m_er]
  cov$error <- att$error[m_er]
  rownames(cov) <- NULL
  cov
}

## The most recent result for each (cell, platform), which is what "current
## state" means when history is retained rather than overwritten.
latest_results <- function(dir) {
  res <- store_read(dir, "results")
  if (is.null(res)) {
    return(NULL)
  }
  runs <- store_read(dir, "runs")
  if (!is.null(runs)) {
    res <- merge(
      res,
      runs[c("run_id", "started_at", "host", "cpu", "branch", "anvl_sha")],
      by = "run_id",
      all.x = TRUE,
      suffixes = c("", ".run")
    )
  }
  o <- order(res$started_at %||% res$run_id, decreasing = TRUE)
  res <- res[o, , drop = FALSE]
  k <- paste(res$cell_id, res$output, res$platform_key, res$depth, sep = "\r")
  res <- res[!duplicated(k), , drop = FALSE]
  rownames(res) <- NULL

  st <- with_reference_status(res, store_read(dir, "validations"))
  resummarise(st, resolved_ranges(dir), resolved_points(dir))
}

## Each result's stable- and gradient-reference status (see
## reference_status()), from every validation in the store.
with_reference_status <- function(res, validations) {
  if (is.null(res)) {
    return(res)
  }
  res$ref_stable_status <- reference_status(
    res$ref_stable_id %||% rep(NA_character_, nrow(res)),
    res$ref_stable_unsupported %||% rep(NA_character_, nrow(res)),
    "stable",
    validations
  )
  res$ref_grad_status <- grad_reference_status(res, validations)
  res
}

## Region and point counts with outside-domain gradients settled against
## their value cells over the whole store, so every reader and an export --
## which calls this with the same resolved tables, before any filter -- cannot
## disagree. A result with no regions gets zeros,
## not whatever the sweep wrote before resolution.
resummarise <- function(res, ranges, points) {
  key <- paste(res$run_id, res$cell_id, res$output, sep = "\r")
  if (!is.null(ranges) && !is.null(ranges$cause)) {
    by <- split(ranges, paste(ranges$run_id, ranges$cell_id, ranges$output, sep = "\r"))
    empty <- ranges[0L, , drop = FALSE]
    for (f in setdiff(names(region_summary(empty)), names(res))) {
      res[[f]] <- NA
    }
    for (i in which(res$output != "-")) {
      rs <- region_summary(by[[key[i]]] %||% empty)
      for (f in names(rs)) {
        res[[f]][i] <- rs[[f]]
      }
    }
  }
  if (!is.null(points) && nrow(points)) {
    by <- split(points, paste(points$run_id, points$cell_id, points$output, sep = "\r"))
    for (f in setdiff(names(point_summary(points[0L, , drop = FALSE])), names(res))) {
      res[[f]] <- NA
    }
    for (i in which(key %in% names(by))) {
      ps <- point_summary(by[[key[i]]])
      for (f in names(ps)) {
        res[[f]][i] <- ps[[f]]
      }
    }
  }
  res
}

## Failure regions with outside-domain gradient regions settled against their
## value cells (see resolve_domain_conventions()). Every reader of regions goes
## through here, and it always resolves against the whole store: the evidence
## is chosen by the run rule, never by what a caller happens to be showing.
## A store written before causes were recorded is returned as is.
##
## Candidate base R disputes then become verified reference limitations where,
## and only where, the result's stable reference has passed validation under
## the identity the sweep recorded (apply_reference_validation()).
resolved_ranges <- function(dir) {
  r <- store_read(dir, "ranges")
  if (is.null(r) || is.null(r$cause)) {
    return(r)
  }
  r <- with_legacy_categories(r)
  r <- resolve_domain_conventions(r, store_read(dir, "kinds"))
  apply_reference_validation(r, stable_statuses(dir))
}

## A store can hold regions from runs made before causes were recorded, which
## read back with a positional `class` and no cause or category. They cannot be
## given a tested cause after the fact; "unclassified" is what failure meant
## then, and every other old class is labelled as such, shown and never
## counted as any current category.
with_legacy_categories <- function(r) {
  old <- is.na(r$category)
  if (!any(old)) {
    return(r)
  }
  cls <- r$class %||% rep(NA_character_, nrow(r))
  r$category[old] <- ifelse(cls[old] %in% "unclassified", "failure", "legacy")
  r$cause[old] <- ifelse(is.na(cls[old]), "unrecorded", paste0("old run: ", cls[old]))
  r
}

## The exact points, resolved the same way.
resolved_points <- function(dir) {
  p <- resolve_point_conventions(store_read(dir, "points"))
  apply_reference_validation(p, stable_statuses(dir))
}

stable_statuses <- function(dir) {
  res <- store_read(dir, "results")
  if (is.null(res)) {
    return(NULL)
  }
  with_reference_status(res, store_read(dir, "validations"))
}

## What a result's figures add up to, without judging any of it. Every flag is
## a statement of fact about the samples and points, and each of the four
## categories keeps its own flag: only undefined-domain conventions are
## set aside, and `identical_but_conventions` says so rather than hiding it.
## The site's js/model.js ports this definition.
##
##   failing      a failure region or a failing exact point
##   boundary     a boundary region or point
##   backend      a backend-limitation region or point
##   identical    every sampled input and every exact point bit-identical,
##                down to the sign of zero
##   identical_but_conventions
##                not identical, but every sample and point that differs is an
##                undefined-domain convention
##   reference    has verified reference limitations: candidate base R
##                disputes whose stable reference passed validation
##   identical_but_set_aside
##                not identical, but every sample and point that differs is an
##                undefined-domain convention or a verified reference
##                limitation -- the two things set aside, each shown
result_state <- function(res) {
  col <- function(nm, d = 0) {
    v <- res[[nm]]
    if (is.null(v)) rep(d, nrow(res)) else ifelse(is.na(v), d, v)
  }
  pts_ok <- col("n_points_identical") == col("n_points")
  verified <- (res$ref_stable_status %||% rep(NA_character_, nrow(res))) %in% "validated"
  differ <- col("n_samples") - col("n_exact")
  data.frame(
    failing = col("n_runs_unclassified") > 0 | col("n_points_failure") > 0,
    boundary = col("n_regions_boundary") > 0 | col("n_points_boundary") > 0,
    backend = col("n_regions_backend") > 0 | col("n_points_backend") > 0,
    identical = differ == 0 & col("n_zero_sign") == 0 & pts_ok & !is.na(res$n_samples),
    identical_but_conventions = differ > 0 &
      differ == col("n_failing_domain") &
      col("n_zero_sign") == 0 &
      col("n_points_identical") + col("n_points_domain") == col("n_points"),
    reference = verified & (col("n_ref_candidate") > 0 | col("n_points_ref_candidate") > 0),
    identical_but_set_aside = differ > 0 &
      col("n_zero_sign") == 0 &
      differ == col("n_failing_domain") + ifelse(verified, col("n_ref_candidate"), 0) &
      col("n_points_identical") + col("n_points_domain") + ifelse(verified, col("n_points_ref_candidate"), 0) ==
        col("n_points"),
    ## the same worst error with verified reference limitations left out
    worst_any_set_aside = ifelse(
      verified,
      pmax(col("worst_rel_err_excl"), col("worst_point_rel_err_excl")),
      pmax(col("worst_rel_err"), col("worst_point_rel_err"))
    ),
    ## the larger of the sweep's worst finite error and the points'
    worst_any = pmax(col("worst_rel_err"), col("worst_point_rel_err"))
  )
}
