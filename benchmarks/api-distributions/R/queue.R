## ---------------------------------------------------------------------------
## Sweeping in parts, through a work queue.
##
## A cell is the unit of *results*, but it is a poor unit of *work*: across the
## grid one cell takes minutes and another days, so a job per cell finishes
## when its slowest cell does, however many nodes it has. Here a cell is cut
## into *parts* -- contiguous stretches of its chunks, both signs -- and a pool
## of long-lived workers takes parts from a queue until none are left:
##
##   plan      the grid, cut into parts, written once to a queue directory
##   work      claim a part, sweep it, save its reducers' state; repeat. The
##             worker that completes a cell's last part assembles the cell and
##             writes it to the store, exactly as `run` would have
##   finalise  assemble any cell whose parts are all saved but which no worker
##             finished assembling (the worker died, say); idempotent
##   queue     how far the queue has got
##
## Assembly is exact, not approximate: each chunk is seeded on its own (see
## chunk_seed()), and every reducer absorbs a later part's state precisely as
## it would have added those samples (see their absorb()). The selftest holds
## an assembled cell to the one swept whole, table by table.
##
## Every worker of a plan writes under the plan's one run ID, so a gradient
## cell's value twin is always in the same run, wherever it was swept -- the
## evidence that settles undefined-domain conventions (see HPC.md).
##
## The queue is files on a shared filesystem, coordinated by mkdir alone,
## which is atomic on every filesystem a cluster uses for scratch:
##
##   <queue>/plan.rds             the plan: cells, parts, run ID, depth
##   <queue>/claims/<part>.<n>/   attempt n at a part; its mtime is a
##                                heartbeat, touched every minute while the
##                                worker lives (see start_heartbeat())
##   <queue>/done/<part>.rds      the part's state, or its error
##   <queue>/done/<part>~<s>~<e>  an empty marker, written after the state:
##                                the part's cost in seconds and whether it
##                                errored -- in the name, so that one listing
##                                of the directory reads the whole queue
##                                without opening tens of thousands of files
##   <queue>/final/<cell key>/    the claim to assemble a cell; `done` inside
##                                once it is in the store
##
## A claim whose heartbeat is older than --stale-minutes (default 10) belongs
## to a worker that died, and the next worker takes the part over as attempt
## n + 1. A part
## swept twice (a worker that was only slow) is swept identically, so the
## second write changes nothing that matters.
## ---------------------------------------------------------------------------

QUEUE_FORMAT <- 1L

queue_path <- function(q, ...) file.path(q, ...)

queue_plan <- function(q) {
  f <- queue_path(q, "plan.rds")
  if (!file.exists(f)) {
    stop("no plan in ", q, "; make one with `run.R plan --queue ", q, "`", call. = FALSE)
  }
  pl <- readRDS(f)
  if (!identical(pl$format, QUEUE_FORMAT)) {
    stop("the plan in ", q, " was written by another version of the harness", call. = FALSE)
  }
  pl
}

## Write a file so that no reader ever sees half of it.
write_atomic <- function(path, write) {
  tmp <- paste0(path, ".tmp", Sys.getpid())
  write(tmp)
  file.rename(tmp, path)
  invisible(path)
}

## ---- costs -----------------------------------------------------------------

## What each cell cost in earlier campaigns: any of a store, an export
## directory, a summary.parquet or a release ZIP, several separated by commas
## in order of priority -- a later source overrides every earlier one for the
## cells it measured, whatever the depths. List the last release first and a
## fresh calibration after it, and a function that changed since the release
## is sized by its new cost, not its old one. One row per source, cell and
## depth that succeeded, its `elapsed_sec` the median over the runs that swept
## it there, and `priority` the source's position.
##
## A smoke sweep is a cheap way to measure cells: scaled to full depth its time
## overstates the cost, since compiling is a larger share of a short sweep, and
## an overstated cell is merely cut into more parts.
read_costs <- function(paths) {
  if (is.null(paths)) {
    return(NULL)
  }
  src <- strsplit(paths, ",", fixed = TRUE)[[1L]]
  res <- do.call(
    rbind,
    lapply(seq_along(src), function(i) {
      r <- read_cost_source(src[i])
      if (!is.null(r) && nrow(r)) cbind(r, priority = i)
    })
  )
  if (is.null(res) || !nrow(res)) {
    return(NULL)
  }
  ## one elapsed per cell and run, however many outputs repeat it
  res <- unique(res)
  stats::aggregate(elapsed_sec ~ cell_id + depth + priority, res, stats::median)
}

read_cost_source <- function(path) {
  if (!file.exists(path)) {
    stop("no costs at ", path, call. = FALSE)
  }
  res <- if (dir.exists(file.path(path, "results"))) {
    store_read(path, "results")
  } else if (dir.exists(path)) {
    as.data.frame(nanoparquet::read_parquet(file.path(path, "summary.parquet")))
  } else if (grepl("\\.zip$", path)) {
    ex <- tempfile()
    on.exit(unlink(ex, recursive = TRUE))
    utils::unzip(path, files = "summary.parquet", exdir = ex)
    as.data.frame(nanoparquet::read_parquet(file.path(ex, "summary.parquet")))
  } else {
    as.data.frame(nanoparquet::read_parquet(path))
  }
  if (is.null(res) || !nrow(res)) {
    return(NULL)
  }
  ok <- is.na(res$error %||% rep(NA, nrow(res))) & is.finite(res$elapsed_sec) & res$elapsed_sec > 0
  res[ok, c("run_id", "cell_id", "depth", "elapsed_sec"), drop = FALSE]
}

## Seconds each cell is expected to take at `depth`, from the last source
## that measured it (see read_costs()): its measurement at that depth, or at
## another scaled by sample count; NA where no source measured the cell. No
## guess is made for an unmeasured cell -- its parts are made as small as
## allowed instead, and the workers learn its real rate from its first parts.
## Attribute "source": the priority of the source each estimate came from.
cell_estimates <- function(g, depth, costs) {
  if (is.null(costs) || !nrow(costs)) {
    return(structure(rep(NA_real_, nrow(g)), source = rep(NA_real_, nrow(g))))
  }
  stride <- vapply(DEPTHS, `[[`, 0, "stride")
  costs$scaled <- costs$elapsed_sec * stride[costs$depth] / stride[[depth]]
  costs <- costs[!is.na(costs$scaled), , drop = FALSE]
  ## the last source first; within it, this depth, then the deepest other one
  costs <- costs[order(-costs$priority, costs$depth != depth, stride[costs$depth]), , drop = FALSE]
  costs <- costs[!duplicated(costs$cell_id), , drop = FALSE]
  i <- match(g$cell_id, costs$cell_id)
  ## which source each estimate came from, for the plan to report
  structure(costs$scaled[i], source = costs$priority[i])
}

## ---- planning --------------------------------------------------------------

## Cut each cell into parts of about `unit_sec`, at most `max_parts`; an
## unmeasured cell into `max_parts`. A part covers chunks from..to of both
## signs. Parts are queued biggest first (unmeasured first of all), so the
## long ones start while there is still time to finish them and the small
## ones fill in at the end.
plan_parts <- function(g, depth, est, unit_sec, max_parts) {
  n_chunks <- sweep_plan("f64", depth)$n_chunks
  n_parts <- ifelse(is.na(est), max_parts, ceiling(est / unit_sec))
  n_parts <- as.integer(pmax(1L, pmin(n_parts, max_parts, n_chunks)))
  rows <- lapply(seq_len(nrow(g)), function(i) {
    n <- n_parts[i]
    p <- seq_len(n)
    data.frame(
      cell_id = g$cell_id[i],
      part = p,
      parts = n,
      from = as.integer(floor((p - 1) * n_chunks / n) + 1),
      to = as.integer(floor(p * n_chunks / n)),
      est_sec = est[i] / n
    )
  })
  u <- do.call(rbind, rows)
  u <- u[order(!is.na(u$est_sec), -u$est_sec, match(u$cell_id, g$cell_id), u$part), , drop = FALSE]
  u$unit <- sprintf("%06d", seq_len(nrow(u)))
  rownames(u) <- NULL
  u[c("unit", "cell_id", "part", "parts", "from", "to", "est_sec")]
}

cmd_plan <- function(opt) {
  if (is.null(opt$queue) && !opt$dry_run) {
    stop("plan needs --queue <dir>, or --dry-run", call. = FALSE)
  }
  specs <- load_specs(include_selftest = grepl("selftest", opt$filter))
  g <- apply_filter(build_grid(specs, opt$backends), opt$filter)
  if (!nrow(g)) {
    stop("filter matched no cells", call. = FALSE)
  }
  unit_sec <- 60 * (opt$unit_minutes %||% 15)
  max_parts <- opt$max_parts %||% 64L
  costs <- read_costs(opt$costs)
  est <- cell_estimates(g, opt$depth, costs)
  units <- plan_parts(g, opt$depth, est, unit_sec, max_parts)

  pv <- collect_provenance(opt$depth)
  pl <- list(
    format = QUEUE_FORMAT,
    run_id = pv$run_id,
    created_at = pv$started_at,
    depth = opt$depth,
    backends = opt$backends,
    filter = opt$filter,
    topk = opt$topk,
    chunk_index = CHUNK_INDEX,
    sweep_sampling = SWEEP_SAMPLING,
    anvl_sha = pv$anvl_sha,
    anvl_version = pv$anvl_version,
    unit_sec = unit_sec,
    cells = g,
    units = units
  )

  known <- !is.na(est)
  cat(sprintf(
    "plan: %d cells in %d parts at depth '%s' (backends %s)\n",
    nrow(g),
    nrow(units),
    opt$depth,
    paste(opt$backends, collapse = ",")
  ))
  if (!is.null(opt$costs)) {
    src <- strsplit(opt$costs, ",", fixed = TRUE)[[1L]]
    used <- tabulate(attr(est, "source")[known], nbins = length(src))
    cat("  costs, later sources overriding earlier ones:\n")
    cat(sprintf("    %5d cell(s) from %s\n", used, src), sep = "")
  }
  if (any(known)) {
    cat(sprintf(
      "  %d cell(s) measured before, estimated %.1f core-hours in all; largest part %.0f min\n",
      sum(known),
      sum(est[known]) / 3600,
      max(units$est_sec, na.rm = TRUE) / 60
    ))
  }
  if (!all(known)) {
    cat(sprintf(
      "  %d cell(s) never measured, each cut into %d parts:\n",
      sum(!known),
      min(max_parts, sweep_plan("f64", opt$depth)$n_chunks)
    ))
    sp <- table(g$spec[!known])
    cat(paste0("    ", names(sp), " (", as.integer(sp), ")", collapse = "\n"), "\n")
  }
  tab <- table(units$parts[!duplicated(units$cell_id)])
  cat(sprintf("  parts per cell: %s\n", paste(sprintf("%s x%d", names(tab), as.integer(tab)), collapse = ", ")))

  if (opt$dry_run) {
    return(invisible(pl))
  }
  q <- opt$queue
  if (file.exists(queue_path(q, "plan.rds"))) {
    stop("there is already a plan in ", q, "; a queue holds one campaign", call. = FALSE)
  }
  for (d in c("claims", "done", "final")) {
    dir.create(queue_path(q, d), recursive = TRUE, showWarnings = FALSE)
  }
  write_atomic(queue_path(q, "plan.rds"), function(f) saveRDS(pl, f))
  cat(sprintf("run id: %s\nqueue:  %s\n", pl$run_id, normalizePath(q)))
  invisible(pl)
}

## ---- the queue's state -----------------------------------------------------

## Every part with its state: done (ok or error), running, stale, or open.
queue_state <- function(q, pl, stale_sec) {
  u <- pl$units
  m <- strsplit(list.files(queue_path(q, "done"), pattern = "~"), "~", fixed = TRUE)
  meta <- data.frame(
    unit = vapply(m, `[`, "", 1L),
    elapsed = as.numeric(vapply(m, `[`, "", 2L)),
    error = vapply(m, `[`, "", 3L) == "1"
  )
  i <- match(u$unit, meta$unit)
  u$done <- !is.na(i)
  u$error <- u$done & meta$error[i] %in% TRUE
  u$elapsed <- meta$elapsed[i]

  cl <- list.files(queue_path(q, "claims"))
  att <- data.frame(unit = sub("\\.[0-9]+$", "", cl), n = as.integer(sub("^.*\\.", "", cl)), dir = cl)
  att <- att[order(att$unit, -att$n), , drop = FALSE]
  att <- att[!duplicated(att$unit), , drop = FALSE]
  j <- match(u$unit, att$unit)
  u$attempt <- ifelse(is.na(j), 0L, att$n[j])
  ## heartbeats only of the parts claimed and not done: about one per worker
  live <- which(!u$done & !is.na(j))
  age <- rep(NA_real_, nrow(u))
  age[live] <- as.numeric(Sys.time()) - as.numeric(file.mtime(queue_path(q, "claims", att$dir[j[live]])))
  ## a claim whose heartbeat cannot be read is as good as dead
  u$running <- seq_len(nrow(u)) %in% live & (age <= stale_sec) %in% TRUE
  u$stale <- seq_len(nrow(u)) %in% live & !u$running
  u
}

## Seconds a part is expected to take: from what its cell's finished parts
## cost per chunk, else the plan's estimate, else NA.
part_estimate <- function(u, st) {
  ok <- st$done & !st$error & st$cell_id %in% u$cell_id
  if (any(ok)) {
    rate <- stats::median(st$elapsed[ok] / (st$to[ok] - st$from[ok] + 1))
    return(rate * (u$to - u$from + 1))
  }
  u$est_sec
}

## Claim the next part: of the cell this worker last swept if any is left (its
## functions are compiled already), otherwise the first open or stale part in
## plan order -- that will finish before `until`. Returns list(part, reason),
## the part NULL and the reason saying why when there is none.
claim_next <- function(q, pl, last_cell, until, stale_sec, owner) {
  st <- queue_state(q, pl, stale_sec)
  open <- which(!st$done & !st$running)
  if (!length(open)) {
    return(list(reason = if (all(st$done)) "every part is done" else "every remaining part is claimed"))
  }
  open <- open[order(st$cell_id[open] != (last_cell %||% ""), seq_along(open))]
  left <- until - as.numeric(Sys.time())
  for (i in open) {
    u <- st[i, , drop = FALSE]
    est <- part_estimate(u, st)
    ## an unmeasured part is assumed to take twice the planned part size
    if (is.na(est)) {
      est <- 2 * pl$unit_sec
    }
    if (1.25 * est + 120 > left) {
      next
    }
    claim <- queue_path(q, "claims", sprintf("%s.%d", u$unit, u$attempt + 1L))
    if (dir.create(claim, showWarnings = FALSE)) {
      writeLines(owner, file.path(claim, "owner"))
      u$claim <- claim
      u$est <- est
      return(list(part = u))
    }
  }
  list(reason = "no remaining part fits before the deadline")
}

## ---- working ---------------------------------------------------------------

## A claim's heartbeat: a background shell loop that touches the claim every
## `every` seconds for as long as this R process is alive and the claim's
## `beating` file exists. It cannot run in R itself: R is blocked inside one
## chunk's evaluation, and a single chunk of a slow cell can take longer than
## any sensible stale threshold -- a heartbeat between chunks once made live
## workers look dead, and four of them swept one part at once. A worker killed
## outright (out of memory, say) stops beating within `every` seconds. Returns
## the `beating` file; removing it (stop_heartbeat()) ends the loop.
start_heartbeat <- function(claim, every = 60) {
  beating <- file.path(claim, "beating")
  file.create(beating)
  loop <- sprintf(
    "while [ -e %s ] && kill -0 %d 2>/dev/null; do touch -c %s; sleep %d; done",
    shQuote(beating),
    Sys.getpid(),
    shQuote(claim),
    as.integer(every)
  )
  system2("sh", c("-c", shQuote(loop)), wait = FALSE, stdout = FALSE, stderr = FALSE)
  beating
}

stop_heartbeat <- function(beating) unlink(beating)

## This process's memory in MB: resident now and its peak, as Linux reports
## them in /proc/self/status (VmRSS, VmHWM) -- what a scheduler's memory limit
## and sacct's MaxRSS see. Elsewhere, resident from `ps` and no peak.
process_memory <- function() {
  f <- "/proc/self/status"
  if (file.exists(f)) {
    l <- readLines(f, warn = FALSE)
    kb <- function(k) as.numeric(sub("^\\S+:\\s*([0-9]+).*", "\\1", grep(paste0("^", k, ":"), l, value = TRUE)[1L]))
    return(c(now = kb("VmRSS"), peak = kb("VmHWM")) / 1024)
  }
  rss <- suppressWarnings(as.numeric(system2("ps", c("-o", "rss=", "-p", Sys.getpid()), stdout = TRUE)))
  c(now = rss / 1024, peak = NA_real_)
}

## The worker's own description, as it goes into its claims and parts.
worker_owner <- function() {
  ## an array task as Slurm's own tools name it, <array job>_<task>
  job <- Sys.getenv("SLURM_ARRAY_JOB_ID", Sys.getenv("SLURM_JOB_ID", ""))
  task <- Sys.getenv("SLURM_ARRAY_TASK_ID", "")
  sprintf(
    "%s pid %d%s",
    Sys.info()[["nodename"]],
    Sys.getpid(),
    if (nzchar(job)) sprintf(" job %s%s", job, if (nzchar(task)) paste0("_", task) else "") else ""
  )
}

## The options a cell is swept with, from the plan rather than the command
## line: every worker of a campaign must sweep alike.
plan_opt <- function(pl) list(depth = pl$depth, topk = pl$topk, quiet = TRUE)

## Refuse to work a plan this harness would sweep differently: other chunks,
## another sampling scheme, another anvl, or a grid without its cells.
check_plan <- function(pl, specs, pv) {
  why <- c(
    if (!identical(pl$chunk_index, CHUNK_INDEX)) "a different chunk size",
    if (!identical(pl$sweep_sampling, SWEEP_SAMPLING)) "a different sampling scheme",
    if (!is.na(pl$anvl_sha) && !is.na(pv$anvl_sha) && pl$anvl_sha != pv$anvl_sha) {
      sprintf("anvl %s, not %s", substr(pv$anvl_sha, 1L, 7L), substr(pl$anvl_sha, 1L, 7L))
    },
    if (!identical(pl$anvl_version, pv$anvl_version)) sprintf("anvl version %s", pv$anvl_version)
  )
  miss <- setdiff(pl$cells$cell_id, build_grid(specs, pl$backends)$cell_id)
  if (length(miss)) {
    why <- c(why, sprintf("%d of the plan's cells not in its grid, e.g. %s", length(miss), miss[1L]))
  }
  if (length(why)) {
    stop("this harness cannot work the plan: ", paste(why, collapse = "; "), call. = FALSE)
  }
}

## The run's provenance row, once per store. Any worker may write it, and two
## writing at once write the same run under the same name, so the last rename
## wins and one row remains; its host and CPU are that worker's, and each
## part keeps its own (see finalise_cell()).
ensure_run_row <- function(dir, pv) {
  f <- file.path(dir, "runs", paste0("run=", pv$run_id), "run.parquet")
  if (!file.exists(f)) {
    store_write(dir, "runs", pv$run_id, "run", provenance_row(pv))
  }
}

cmd_work <- function(opt) {
  if (is.null(opt$queue)) {
    stop("work needs --queue <dir>", call. = FALSE)
  }
  q <- opt$queue
  pl <- queue_plan(q)
  specs <- load_specs(include_selftest = "selftest" %in% pl$cells$spec)
  pv <- collect_provenance(pl$depth, run_id = pl$run_id)
  check_plan(pl, specs, pv)
  dir <- store_dir(opt$store)
  store_init(dir)
  ensure_run_row(dir, pv)

  until <- opt$until %||% Inf
  stale_sec <- 60 * (opt$stale_minutes %||% 10)
  owner <- worker_owner()
  wopt <- plan_opt(pl)
  cat(sprintf("worker %s | run %s | depth %s | store %s\n", owner, pl$run_id, pl$depth, dir))
  if (is.finite(until)) {
    cat(sprintf("  claiming only what finishes before %s\n", format(as.POSIXct(until, origin = "1970-01-01"))))
  }

  setups <- list()
  last <- NULL
  restart <- FALSE
  n_ok <- n_err <- 0L
  repeat {
    cl <- claim_next(q, pl, last, until, stale_sec, owner)
    if (is.null(cl$part)) {
      reason <- cl$reason
      break
    }
    u <- cl$part
    row <- pl$cells[match(u$cell_id, pl$cells$cell_id), , drop = FALSE]
    rownames(row) <- NULL
    cs <- setups[[u$cell_id]] %||% cell_setup(specs[[row$spec]], row)
    setups <- list(cs)
    names(setups) <- u$cell_id
    last <- u$cell_id

    beat <- start_heartbeat(u$claim)
    t0 <- Sys.time()
    out <- cell_sweep(cs, row, wopt, chunks = u$from:u$to, finish = FALSE)
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    stop_heartbeat(beat)
    failed <- inherits(out, "sweep_error")
    if (!failed && u$part > 1L) {
      attr(out, "points") <- NULL # the first part's are the cell's
    }
    rec <- list(
      unit = u$unit,
      cell_id = u$cell_id,
      part = u$part,
      parts = u$parts,
      attempt = u$attempt + 1L,
      owner = owner,
      host = pv$host,
      cpu = pv$cpu,
      elapsed = elapsed,
      finished_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
      error = if (failed) as.character(out) else NA_character_,
      sweep = if (failed) NULL else out
    )
    write_atomic(queue_path(q, "done", paste0(u$unit, ".rds")), function(f) saveRDS(rec, f))
    file.create(queue_path(q, "done", sprintf("%s~%.3f~%d", u$unit, elapsed, failed)))
    if (failed) {
      n_err <- n_err + 1L
    } else {
      n_ok <- n_ok + 1L
    }
    ## memory after every part, so a worker's growth can be read off its log
    mem <- process_memory()
    cat(sprintf(
      "[%s] %-5s %s  part %d/%d, chunks %d-%d, %.0f s (expected %.0f), mem %.0f MB (peak %.0f)%s\n",
      u$unit,
      if (failed) "ERROR" else "OK",
      u$cell_id,
      u$part,
      u$parts,
      u$from,
      u$to,
      elapsed,
      u$est,
      mem[["now"]],
      mem[["peak"]],
      if (failed) paste0("\n        ", strsplit(as.character(out), "\n", fixed = TRUE)[[1L]][1L]) else ""
    ))
    fin <- finalise_ready(q, pl, specs, dir, pv, cells = u$cell_id)
    for (m in fin) {
      cat(m, "\n", sep = "")
    }
    ## Past --max-memory, stop at this part boundary for a fresh process to
    ## take over: memory creeps up over a worker's life by a few MB a cell,
    ## for no cause found, and only a new process starts clean.
    if (!is.null(opt$max_memory) && process_memory()[["now"]] > opt$max_memory) {
      reason <- "its memory passed --max-memory; restart it"
      restart <- TRUE
      break
    }
  }
  cat(sprintf("\nworker done: %d part(s) ok, %d error; stopped because %s\n", n_ok, n_err, reason))
  mem <- process_memory()
  cat(sprintf("memory %.0f MB, peak %.0f MB\n", mem[["now"]], mem[["peak"]]))
  invisible(structure(n_err, restart = restart))
}

## ---- assembling cells ------------------------------------------------------

## Assemble every cell (of `cells`, or all) whose parts are all saved and that
## nobody has claimed to assemble -- or, with `force`, that is not yet in the
## store, whoever claimed it. Returns a line per cell assembled.
finalise_ready <- function(q, pl, specs, dir, pv, cells = NULL, force = FALSE) {
  u <- queue_state(q, pl, Inf)
  if (!is.null(cells)) {
    u <- u[u$cell_id %in% cells, , drop = FALSE]
  }
  ready <- tapply(u$done, u$cell_id, all)
  out <- character(0)
  for (id in names(ready)[ready]) {
    key <- cell_key(id)
    fdir <- queue_path(q, "final", key)
    if (file.exists(file.path(fdir, "done"))) {
      next
    }
    if (!dir.create(fdir, showWarnings = FALSE) && !force) {
      next
    }
    ## A cell that cannot be assembled (an unreadable part, say) must not take
    ## the worker down with it: its claim is dropped for `finalise` to retry.
    msg <- tryCatch(finalise_cell(q, pl, specs, dir, pv, id), error = function(e) {
      unlink(fdir, recursive = TRUE)
      sprintf("  could not assemble %s: %s", id, conditionMessage(e))
    })
    if (file.exists(fdir)) {
      writeLines(format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"), file.path(fdir, "done"))
    }
    out <- c(out, msg)
  }
  out
}

## One cell from its parts, into the store under the plan's run. A part that
## errored makes the cell an errored cell, as `run` would have recorded it.
## The parts are read one at a time and absorbed as they come (see
## sweep_assembler()): a cell can have hundreds of ~10 MB parts, and reading
## them all first once took the assembling worker past its memory limit.
finalise_cell <- function(q, pl, specs, dir, pv, id) {
  u <- pl$units[pl$units$cell_id == id, , drop = FALSE]
  u <- u[order(u$part), , drop = FALSE]
  row <- pl$cells[match(id, pl$cells$cell_id), , drop = FALSE]
  rownames(row) <- NULL
  cs <- cell_setup(specs[[row$spec]], row)
  opt <- plan_opt(pl)

  asm <- sweep_assembler(row$dtype, pl$depth, cs$cf$outputs, pl$topk)
  elapsed <- 0
  cpus <- character(0)
  points <- NULL
  err <- NULL
  for (i in seq_len(nrow(u))) {
    r <- readRDS(queue_path(q, "done", paste0(u$unit[i], ".rds")))
    elapsed <- elapsed + r$elapsed
    cpus <- union(cpus, r$cpu %||% NA_character_)
    if (!is.na(r$error)) {
      err <- sprintf("part %d/%d (chunks %d-%d): %s", r$part, r$parts, u$from[i], u$to[i], r$error)
      break
    }
    if (i == 1L) {
      points <- attr(r$sweep, "points") # the first part's are the cell's
    }
    asm$absorb(r$sweep)
  }
  out <- if (is.null(err)) {
    sw <- asm$finish(cs$cf$stable)
    attr(sw, "points") <- points
    sw
  } else {
    structure(err, class = "sweep_error")
  }
  st <- record_cell(row, opt, pv, dir, cs$cf, out, elapsed)

  sprintf(
    "  assembled %-5s %s from %d part(s), %.1f core-hours%s",
    st$status,
    id,
    nrow(u),
    elapsed / 3600,
    if (length(cpus) > 1L) {
      sprintf("\n  WARNING: its parts ran on %d CPU models: %s", length(cpus), paste(cpus, collapse = "; "))
    } else {
      ""
    }
  )
}

cmd_finalise <- function(opt) {
  if (is.null(opt$queue)) {
    stop("finalise needs --queue <dir>", call. = FALSE)
  }
  q <- opt$queue
  pl <- queue_plan(q)
  specs <- load_specs(include_selftest = "selftest" %in% pl$cells$spec)
  pv <- collect_provenance(pl$depth, run_id = pl$run_id)
  check_plan(pl, specs, pv)
  dir <- store_dir(opt$store)
  store_init(dir)
  ensure_run_row(dir, pv)
  fin <- finalise_ready(q, pl, specs, dir, pv, force = TRUE)
  cat(sprintf("assembled %d cell(s) into %s\n", length(fin), dir))
  for (m in fin) {
    cat(m, "\n", sep = "")
  }
  cmd_queue(opt)
}

## ---- status ----------------------------------------------------------------

## Drop every claim on a part that is not done, so that the next workers start
## on those parts at once instead of waiting for the claims to go stale. Only
## for when no worker is running: a live worker's part would be swept twice.
queue_release <- function(q, pl) {
  st <- queue_state(q, pl, Inf)
  cl <- list.files(queue_path(q, "claims"))
  drop <- cl[sub("\\.[0-9]+$", "", cl) %in% st$unit[!st$done]]
  unlink(queue_path(q, "claims", drop), recursive = TRUE)
  cat(sprintf("released %d claim(s) on %d unfinished part(s)\n", length(drop), sum(!st$done & st$attempt > 0L)))
}

cmd_queue <- function(opt) {
  if (is.null(opt$queue)) {
    stop("queue needs --queue <dir>", call. = FALSE)
  }
  q <- opt$queue
  pl <- queue_plan(q)
  if (isTRUE(opt$release)) {
    queue_release(q, pl)
  }
  st <- queue_state(q, pl, 60 * (opt$stale_minutes %||% 10))
  fin <- file.exists(queue_path(q, "final", cell_key(pl$cells$cell_id), "done"))
  ready <- tapply(st$done, st$cell_id, all)[pl$cells$cell_id]

  cat(sprintf("\nqueue %s\n  run %s | depth %s | planned %s\n", normalizePath(q), pl$run_id, pl$depth, pl$created_at))
  cat(sprintf(
    "\n  cells  %5d in all: %d assembled, %d ready to assemble, %d still being swept\n",
    nrow(pl$cells),
    sum(fin),
    sum(ready & !fin),
    sum(!ready)
  ))
  cat(sprintf(
    "  parts  %5d in all: %d done (%d errored), %d running, %d stale, %d not started\n",
    nrow(st),
    sum(st$done),
    sum(st$error),
    sum(st$running),
    sum(st$stale),
    sum(!st$done & st$attempt == 0L)
  ))
  ok <- st$done & !st$error
  if (any(ok)) {
    cat(sprintf("  spent  %.1f core-hours on finished parts\n", sum(st$elapsed[ok]) / 3600))
  }
  left <- which(!st$done)
  if (length(left)) {
    est <- vapply(left, function(i) part_estimate(st[i, , drop = FALSE], st), 0)
    cat(sprintf(
      "  left   %d part(s): %s%s\n",
      length(left),
      if (!all(is.na(est))) {
        sprintf("%.1f core-hours expected for %d", sum(est, na.rm = TRUE) / 3600, sum(!is.na(est)))
      } else {
        ""
      },
      if (anyNA(est)) {
        sprintf("%s%d of cells not yet measured", if (!all(is.na(est))) ", " else "", sum(is.na(est)))
      } else {
        ""
      }
    ))
  }
  show <- function(title, i) {
    if (!length(i)) {
      return()
    }
    cat(sprintf("\n  %s (%d):\n", title, length(i)))
    for (j in utils::head(i, 10L)) {
      cat(sprintf("    %s  %s  part %d/%d\n", st$unit[j], st$cell_id[j], st$part[j], st$parts[j]))
    }
    if (length(i) > 10L) cat(sprintf("    ... and %d more\n", length(i) - 10L))
  }
  show("stale claims, taken over by the next worker", which(st$stale))
  show("errored parts", which(st$error))
  cat("\n")
  invisible(st)
}
