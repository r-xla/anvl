## ---------------------------------------------------------------------------
## Exhaustive float32 sweep: relative error of f() against g(), all 2^32 values
## ---------------------------------------------------------------------------

f32_sweep <- function(f, g, out) {
  ## ---- config ----------------------------------------------------------------
  CHUNK <- 1e6L     # values per call
  TOPK  <- 1000L    # how many worst cases to retain
  
  ## ---- bit-level helpers -----------------------------------------------------
  ## 32-bit integer bit pattern -> the float32 it encodes, widened to a double
  bits_to_dbl <- function(i)
    readBin(writeBin(i, raw(), size = 4L), "double", size = 4L, n = length(i))
  
  ## double -> nearest float32, widened back to a double (round-to-nearest-even)
  to_f32 <- function(x)
    readBin(writeBin(x, raw(), size = 4L), "double", size = 4L, n = length(x))
  
  ## ---- error metric ----------------------------------------------------------
  ## Relative error |f - g| / |g|, with the degenerate cases pinned down:
  ##   f and g bit-identical (incl. +-Inf, incl. both NaN)  -> 0
  ##   g == 0 but f != 0                                    -> Inf  (unbounded)
  ##   exactly one of them NaN, or g infinite and f isn't    -> Inf
  rel_err <- function(fx, gx) {
    e <- abs(fx - gx) / abs(gx)
    ok <- (fx == gx) | (is.nan(fx) & is.nan(gx))
    ok[is.na(ok)] <- FALSE
    e[ok] <- 0
    e[!ok & (gx == 0 | !is.finite(gx) | is.nan(fx))] <- Inf
    e
  }
  
  ## ---- running top-K, finite errors only -------------------------------------
  top <- local({
    x <- numeric(0); e <- numeric(0); cut <- -Inf
    list(
      add = function(xs, es) {
        i <- which(es > cut & is.finite(es))       # Inf goes to the run tracker
        if (!length(i)) return(invisible(NULL))
        x <<- c(x, xs[i]); e <<- c(e, es[i])
        o <- order(e, decreasing = TRUE)[seq_len(min(TOPK, length(e)))]
        x <<- x[o]; e <<- e[o]
        if (length(e) >= TOPK) cut <<- e[TOPK]
      },
      get = function() data.frame(x = x, rel_err = e)
    )
  })
  
  ## ---- Inf tracker: contiguous bit patterns collapsed to [lo, hi] runs -------
  ## Each sweep block hands over consecutive bit patterns, so a run of Inf is
  ## just a maximal stretch of consecutive indices; runs that touch the previous
  ## block's tail are extended rather than appended. One tracker per sign, so a
  ## run never straddles the +/- halves.
  new_runs <- function() {
    lo <- hi <- numeric(0)                        # uint32 patterns, held as doubles
    list(
      add = function(p, bad) {
        i <- which(bad)
        if (!length(i)) return(invisible(NULL))
        st <- which(c(TRUE, diff(i) != 1))        # run starts, in index space
        s <- p[i[st]]; e <- p[i[c(st[-1] - 1, length(i))]]
        if (length(lo) && s[1] == hi[length(hi)] + 1) {
          hi[length(hi)] <<- e[1]; s <- s[-1]; e <- e[-1]
        }
        lo <<- c(lo, s); hi <<- c(hi, e)
      },
      get = function() data.frame(lo = lo, hi = hi)
    )
  }
  runs <- list(new_runs(), new_runs())            # [[1]] sign bit clear, [[2]] set
  
  ## ---- pattern decoding (for reporting) --------------------------------------
  pat_to_dbl <- function(p) {
    neg <- p >= 2^31
    v <- bits_to_dbl(as.integer(p - neg * 2^31))
    ifelse(neg, -v, v)
  }
  hex32 <- function(p) sprintf("0x%04X%04X", p %/% 65536, p %% 65536)
  
  ## ---- the sweep -------------------------------------------------------------
  ## Patterns 0x00000000..0x7FFFFFFF are enumerated directly (they fit in R's
  ## signed 32-bit integer); the other half is exactly their negation.
  N <- 2^31
  
  ## cli's progress bar derives the ETA itself from the observed chunk rate; the
  ## chunks are uniform in cost, so it settles within the first few. (It stays
  ## hidden for the first 2s by default -- see options(cli.progress_show_after).)
  cli::cli_progress_bar(
    format = paste("sweeping {cli::pb_bar} {cli::pb_percent}",
                   "| {cli::pb_current}/{cli::pb_total} | ETA {cli::pb_eta}"),
    total = ceiling(N / CHUNK))
  
  for (start in seq(0, N - 1, by = CHUNK)) {
    n <- min(CHUNK, N - start)
    p <- start + seq_len(n) - 1                   # bit patterns of the + half
    xp <- bits_to_dbl(as.integer(p))
    for (k in 1:2) {
      x <- if (k == 1) xp else -xp
      e <- rel_err(f(x), to_f32(g(x)))            # finite or Inf, never NaN
      top$add(x, e)
      runs[[k]]$add(p + (k - 1) * 2^31, is.infinite(e))
    }
    cli::cli_progress_update()
  }
  cli::cli_progress_done()
  
  ## ---- results ---------------------------------------------------------------
  worst <- top$get()                              # top TOPK finite relative errors
  
  inf_ranges <- do.call(rbind, lapply(runs, function(r) r$get()))
  inf_ranges <- with(inf_ranges, data.frame(
    bits_from = hex32(lo), bits_to = hex32(hi),
    x_from = pat_to_dbl(lo), x_to = pat_to_dbl(hi),
    n = hi - lo + 1))                             # x_from/x_to descend in the - half
  
  sink(out)
  cat(sprintf("worst finite rel err : %.6g\n", if (nrow(worst)) worst$rel_err[1] else NA))
  cat(sprintf("Inf-error values     : %.0f in %d range(s)\n",
              sum(inf_ranges$n), nrow(inf_ranges)))
  print(worst)
  print(inf_ranges)
  sink()
}
