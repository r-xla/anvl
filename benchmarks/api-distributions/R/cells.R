## ---------------------------------------------------------------------------
## Sweep specs, the grid they expand into, and the cell IDs that name it.
##
## A *spec* is one file in sweeps/, describing one anvl function: what to
## sweep, against what reference, over which parameter sets and flags. A
## *cell* is one unit of work -- one (spec, backend, dtype, kind, params,
## flags) combination. A cell produces one or more *results*: a value cell
## produces one, a gradient cell produces one per differentiated argument,
## because all of them fall out of a single reverse pass.
##
## The cell ID is the join key for everything: the Parquet filename, every
## table in the store, and HPC sharding. It is deterministic and identical on
## every machine, which is what makes shards from different machines merge
## rather than collide.
## ---------------------------------------------------------------------------

SPEC_FIELDS_REQUIRED <- c("name", "family", "value", "ref_value")
SPEC_FIELDS_OPTIONAL <- list(
  dtypes = c("f32", "f64"),
  params = list(default = list()),
  flags = list(),
  ## The *valid input domain* of the primary argument: where the function is
  ## defined at all, e.g. p in [0, 1] for a quantile. Outside it the value is
  ## NaN by specification. function(params, flags) -> c(lo, hi), closed.
  domain = NULL,
  ## The *distribution's support*: where the density is positive. Reporting
  ## only -- a CDF below the support still has a perfectly good value, so
  ## nothing is excused by it. function(params, flags) -> c(lo, hi), or NULL.
  support = NULL,
  ## Points where the *implementation* switches algorithm, in the primary
  ## argument: function(params, flags, dtype) -> named numeric. They are
  ## anvl's own, read from R/api-distributions.R, and go stale when it
  ## changes; they are checked exactly, with their neighbours, as exact points.
  branch_points = NULL,
  grad_wrt = character(0),
  grad = NULL,
  ref_grad = NULL,
  jax_value = NULL,
  jax_grad = NULL,
  jax_covers = NULL, # function(flags, kind) -> TRUE if JAX has this variant
  ## A *stable reference* for the value, beside base R and never instead of
  ## it: function(x, params, flags) -> numeric, evaluated with the parameters
  ## base R receives. Samples where base R is off, anvl is accurate and no
  ## further from it are recorded as candidate base R disputes (see
  ## dispute_facts()). It needs an error bound, in double ulps at its value,
  ## that validation against high precision checks; a note saying why base R
  ## is weaker here; and optionally which flags it covers.
  ref_stable = NULL,
  ref_stable_bound_ulp64 = NULL,
  ref_stable_note = NULL,
  ref_stable_covers = NULL, # function(flags) -> TRUE where ref_stable applies
  ## High-precision truths, used only by `validate-refs` (they need Rmpfr and
  ## are never called by a sweep): function(x, params, flags) with x and the
  ## parameters as mpfr numbers, returning mpfr -- a vector for the stable
  ## reference, a named list like ref_grad's for the gradients. Each mirrors
  ## its reference's conventions (endpoint values, out-of-domain zeros) and
  ## differs from it only in evaluating the mathematics exactly. The gradient
  ## references' error bound, in double ulps, is declared here and checked,
  ## never raised after the fact.
  ref_stable_mpfr = NULL,
  ref_grad_mpfr = NULL,
  ref_grad_bound_ulp64 = NULL
)

## Build and validate a spec. An unknown field name is an error rather than a
## silent no-op: a typo in a config that takes hours to run must not be
## discovered from a puzzling result.
sweep_spec <- function(...) {
  s <- list(...)
  miss <- setdiff(SPEC_FIELDS_REQUIRED, names(s))
  if (length(miss)) {
    stop("spec is missing required field(s): ", paste(miss, collapse = ", "), call. = FALSE)
  }
  known <- c(SPEC_FIELDS_REQUIRED, names(SPEC_FIELDS_OPTIONAL))
  extra <- setdiff(names(s), known)
  if (length(extra)) {
    stop(
      "spec '",
      s$name,
      "' has unknown field(s): ",
      paste(extra, collapse = ", "),
      "\n  known fields: ",
      paste(sort(known), collapse = ", "),
      call. = FALSE
    )
  }
  ## Name-wise replacement, deliberately not modifyList(): modifyList recurses
  ## into nested lists, which would merge a spec's `params` with the default
  ## placeholder rather than replacing it, inventing a parameter set that no
  ## spec declared. The selftest caught exactly that.
  for (nm in names(SPEC_FIELDS_OPTIONAL)) {
    if (is.null(s[[nm]])) s[nm] <- SPEC_FIELDS_OPTIONAL[nm]
  }
  if (!is.null(s$ref_stable) && (!is.numeric(s$ref_stable_bound_ulp64) || is.null(s$ref_stable_note))) {
    stop("spec '", s$name, "' declares ref_stable without ref_stable_bound_ulp64 and ref_stable_note", call. = FALSE)
  }
  if (!is.null(s$ref_grad_mpfr) && !is.numeric(s$ref_grad_bound_ulp64)) {
    stop("spec '", s$name, "' declares ref_grad_mpfr without ref_grad_bound_ulp64", call. = FALSE)
  }
  if (length(s$grad_wrt) && (is.null(s$grad) || is.null(s$ref_grad))) {
    stop("spec '", s$name, "' declares grad_wrt but no grad/ref_grad", call. = FALSE)
  }
  structure(s, class = "sweep_spec")
}

## Discover every spec in sweeps/. Files starting with "_" are helpers, not
## specs -- except _selftest.R, which is a spec and is included only when asked
## for by name, so it never pads a real run.
load_specs <- function(dir = file.path(here(), "sweeps"), include_selftest = FALSE) {
  files <- sort(list.files(dir, pattern = "\\.R$", full.names = TRUE))
  files <- files[include_selftest | basename(files) != "_selftest.R"]
  files <- files[!grepl("^_", basename(files)) | basename(files) == "_selftest.R"]
  specs <- lapply(files, function(f) {
    s <- source(f, local = new.env(parent = globalenv()))$value
    if (!inherits(s, "sweep_spec")) {
      stop("sweeps/", basename(f), " did not return a sweep_spec()", call. = FALSE)
    }
    s
  })
  setNames(specs, vapply(specs, function(s) s$name, ""))
}

## ---- flag combinations -----------------------------------------------------

flag_grid <- function(flags) {
  if (!length(flags)) {
    return(list(list()))
  }
  g <- expand.grid(flags, KEEP.OUT.ATTRS = FALSE)
  lapply(seq_len(nrow(g)), function(i) as.list(g[i, , drop = FALSE]))
}

## A short, stable, path-safe tag for one flag combination.
flag_tag <- function(f) {
  if (!length(f)) {
    return("-")
  }
  paste0(names(f), "=", vapply(f, function(v) as.character(v), ""), collapse = ",")
}

## ---- the grid --------------------------------------------------------------

## Expand every spec into its cells. `backends` is "anvl" and/or "jax"; a JAX
## cell is emitted only where the spec says JAX actually has that variant, so
## the grid never contains cells that cannot run.
build_grid <- function(specs, backends = c("anvl", "jax")) {
  rows <- list()
  for (s in specs) {
    kinds <- c("value", if (length(s$grad_wrt)) "grad")
    for (backend in backends) {
      for (dtype in s$dtypes) {
        for (kind in kinds) {
          fn <- if (backend == "jax") {
            if (kind == "value") s$jax_value else s$jax_grad
          } else {
            if (kind == "value") s$value else s$grad
          }
          if (is.null(fn)) {
            next
          }
          for (pname in names(s$params)) {
            for (f in flag_grid(s$flags)) {
              if (backend == "jax" && !is.null(s$jax_covers) && !s$jax_covers(f, kind)) {
                next
              }
              rows[[length(rows) + 1L]] <- data.frame(
                spec = s$name,
                family = s$family,
                backend = backend,
                dtype = dtype,
                kind = kind,
                param_set = pname,
                flags = flag_tag(f),
                n_outputs = if (kind == "grad") length(s$grad_wrt) else 1L
              )
            }
          }
        }
      }
    }
  }
  if (!length(rows)) {
    return(empty_grid())
  }
  g <- do.call(rbind, rows)
  g$cell_id <- cell_id(g)
  ## A canonical order, so that an index into this grid means the same thing on
  ## every machine -- which is what an HPC array job relies on.
  g <- g[order(g$cell_id), , drop = FALSE]
  rownames(g) <- NULL
  g
}

empty_grid <- function() {
  data.frame(
    spec = character(0),
    family = character(0),
    backend = character(0),
    dtype = character(0),
    kind = character(0),
    param_set = character(0),
    flags = character(0),
    n_outputs = integer(0),
    cell_id = character(0)
  )
}

cell_id <- function(g) {
  sprintf("%s/%s/%s/%s/%s/%s", g$spec, g$backend, g$dtype, g$kind, g$param_set, g$flags)
}

## Path-safe rendering of a cell ID, used as a Parquet filename. Deterministic
## and reversible enough to read at a glance; the full ID is also a column, so
## nothing depends on parsing it back.
cell_key <- function(id) gsub("-+", "-", gsub("[^A-Za-z0-9]+", "-", id))

## ---- filtering -------------------------------------------------------------

## `--filter spec=nv_punif,dtype=f64` and friends. Values may be comma-free
## lists separated by "|", e.g. dtype=f32|f64. An unknown key is an error.
## `extra` names keys that are valid but live on the results rather than on the
## grid -- `output`, the differentiated argument, is one row per result and
## several per cell, so it cannot be a grid column without breaking the
## cell-as-unit-of-work model that `run` depends on. Such terms are accepted
## and skipped here, then applied to the results by the caller.
apply_filter <- function(grid, filter, extra = character(0)) {
  if (!nzchar(filter)) {
    return(grid)
  }
  ## Terms are comma-separated, but a cell_id *contains* commas (its flag tag
  ## is "lower_tail=TRUE,log_p=FALSE"), so a naive split tears one in half.
  ## Rejoin any fragment whose key is not a grid column: that is deterministic,
  ## needs no escaping, and leaves --filter 'dtype=f64,kind=grad' reading the
  ## obvious way.
  frags <- strsplit(filter, ",", fixed = TRUE)[[1L]]
  terms <- character(0)
  for (fr in frags) {
    key <- trimws(sub("=.*$", "", fr))
    if (length(terms) && !key %in% c(names(grid), extra)) {
      terms[length(terms)] <- paste0(terms[length(terms)], ",", fr)
    } else {
      terms <- c(terms, fr)
    }
  }

  for (term in terms) {
    kv <- strsplit(trimws(term), "=", fixed = TRUE)[[1L]]
    if (length(kv) < 2L) {
      stop("bad filter term: '", term, "'", call. = FALSE)
    }
    key <- trimws(kv[1L])
    ## a cell_id value contains "=" too, so keep everything after the first one
    kv <- c(key, sub("^[^=]*=", "", trimws(term)))
    if (!key %in% c(names(grid), extra)) {
      stop(
        "unknown filter key '",
        key,
        "'; available: ",
        paste(c(setdiff(names(grid), "n_outputs"), extra), collapse = ", "),
        call. = FALSE
      )
    }
    ## a result-level key: valid, but applied by the caller, not here
    if (!key %in% names(grid)) {
      next
    }
    vals <- trimws(strsplit(kv[2L], "|", fixed = TRUE)[[1L]])
    ## an exact match on the column, or a substring match for cell_id/flags
    grid <- grid[
      grid[[key]] %in%
        vals |
        (key %in% c("cell_id", "flags") & grepl(paste(vals, collapse = "|"), grid[[key]], fixed = FALSE)),
      ,
      drop = FALSE
    ]
  }
  rownames(grid) <- NULL
  grid
}

## Apply the result-level terms of a filter (currently just `output`), which
## `apply_filter` accepted and skipped because they are not grid columns.
filter_results <- function(res, filter) {
  if (!nzchar(filter)) {
    return(res)
  }
  for (term in strsplit(filter, ",", fixed = TRUE)[[1L]]) {
    kv <- strsplit(trimws(term), "=", fixed = TRUE)[[1L]]
    if (length(kv) == 2L && kv[1L] == "output") {
      res <- res[res$output %in% trimws(strsplit(kv[2L], "|", fixed = TRUE)[[1L]]), , drop = FALSE]
    }
  }
  res
}
