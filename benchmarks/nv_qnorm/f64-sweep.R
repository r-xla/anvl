## ---------------------------------------------------------------------------
## Stratified float64 sweep: relative error of f() against g(), 2^32 samples
##
## The 2^64 bit patterns split into 2^32 contiguous blocks of 2^32; this draws
## exactly one sample from every block. Equivalently: the high word (sign + 11
## exponent bits + top 20 mantissa bits = exactly 32 bits) is enumerated
## deterministically, and the low 32 mantissa bits are drawn uniformly at
## random. One chunk = one (sign, exponent) pair = one binade = 2^20 samples.
## ---------------------------------------------------------------------------

f64_sweep <- function(f, g) {
  ## ---- config ----------------------------------------------------------------
  EXPS <- 0:2047 # exponent fields to sweep (0 = subnormals, 2047 = Inf/NaN)
  CHUNK <- 2^20 # samples per call = one full binade; do not change
  TOPK <- 1000L # how many worst finite cases to retain
  set.seed(as.integer(paste0(setNames(1:26, letters)[c("a", "n", "v", "l")], collapse = ""))) # low bits are random: seed for reproduciblity

  ## ---- bit-level helpers -----------------------------------------------------
  LE <- .Platform$endian == "little"

  ## (high word, low word) -> the float64 those 64 bits encode
  build <- function(hi, lo) {
    readBin(
      writeBin(as.vector(if (LE) rbind(lo, hi) else rbind(hi, lo)), raw(), size = 4L),
      "double",
      size = 8L,
      n = length(hi)
    )
  }

  ## n uniform 32-bit words, as R integers holding the raw bit patterns.
  ## as.integer(-2^31) overflows to NA_integer_, whose own bit pattern is
  ## 0x80000000 -- exactly the word we wanted -- so the coercion is correct.
  rand32 <- function(n) {
    suppressWarnings(as.integer(sample.int(2^32, n, replace = TRUE) - (2^31 + 1)))
  }

  u32 <- function(i) ifelse(i < 0, i + 2^32, i) # signed int -> unsigned
  hex32 <- function(p) sprintf("%04X%04X", p %/% 65536, p %% 65536)
  bits64 <- function(x) {
    w <- matrix(readBin(writeBin(x, raw(), size = 8L), "integer", size = 4L, n = 2 * length(x)), nrow = 2)
    sprintf("0x%s%s", hex32(u32(w[2, ])), hex32(u32(w[1, ]))) # sprintf, not
  } # paste0: keeps length 0 at length 0

  ## ---- error metric ----------------------------------------------------------
  ## |f - g| / |g|; bit-identical (incl. +-Inf, incl. both NaN) -> 0, and the
  ## cases with no meaningful denominator -> Inf. Never NaN.
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
    x <- numeric(0)
    e <- numeric(0)
    cut <- -Inf
    list(
      add = function(xs, es) {
        i <- which(es > cut & is.finite(es)) # Inf goes to the run tracker
        if (!length(i)) {
          return(invisible(NULL))
        }
        x <<- c(x, xs[i])
        e <<- c(e, es[i])
        o <- order(e, decreasing = TRUE)[seq_len(min(TOPK, length(e)))]
        x <<- x[o]
        e <<- e[o]
        if (length(e) >= TOPK) cut <<- e[TOPK]
      },
      get = function() data.frame(x = x, rel_err = e)
    )
  })

  ## ---- Inf tracker: consecutive high words collapsed to [lo, hi] runs --------
  ## High words are swept in order, so a run of Inf is a maximal stretch of
  ## consecutive indices; a run touching the previous block's tail extends it.
  ## One tracker per sign, so a run never straddles the +/- halves. A run means
  ## every sampled point across that stretch of the number line failed.
  new_runs <- function() {
    lo <- hi <- numeric(0)
    list(
      add = function(p, bad) {
        i <- which(bad)
        if (!length(i)) {
          return(invisible(NULL))
        }
        st <- which(c(TRUE, diff(i) != 1))
        s <- p[i[st]]
        e <- p[i[c(st[-1] - 1, length(i))]]
        if (length(lo) && s[1] == hi[length(hi)] + 1) {
          hi[length(hi)] <<- e[1]
          s <- s[-1]
          e <- e[-1]
        }
        lo <<- c(lo, s)
        hi <<- c(hi, e)
      },
      get = function() data.frame(lo = lo, hi = hi)
    )
  }
  runs <- list(new_runs(), new_runs()) # [[1]] sign bit clear, [[2]] set

  ## ---- the sweep -------------------------------------------------------------
  cli::cli_progress_bar(
    format = paste(
      "binade {cli::pb_extra$mag} {cli::pb_bar} {cli::pb_percent}",
      "| {cli::pb_current}/{cli::pb_total} | ETA {cli::pb_eta}"
    ),
    total = length(EXPS),
    extra = list(mag = "")
  )

  for (e in EXPS) {
    hw <- as.integer(e * 2^20 + seq_len(CHUNK) - 1) # high words, + half
    for (k in 1:2) {
      x <- build(hw, rand32(CHUNK)) # fresh low bits
      if (k == 2) {
        x <- -x
      } # exact sign flip
      err <- rel_err(f(x), g(x))
      top$add(x, err)
      runs[[k]]$add(hw + (k - 1) * 2^31, is.infinite(err))
    }
    cli::cli_progress_update(extra = list(mag = sprintf("%8.1e", 2^(e - 1023))))
  }
  cli::cli_progress_done()

  ## ---- results ---------------------------------------------------------------
  worst <- top$get() # top TOPK finite relative errors
  worst$bits <- bits64(worst$x)

  ## Each run covers whole blocks, so its endpoints are the exact bounds of the
  ## affected interval: low word 0x00000000 at one end, 0xFFFFFFFF at the other.
  inf_ranges <- do.call(rbind, lapply(runs, function(r) r$get()))
  inf_ranges <- with(inf_ranges, {
    side <- lo >= 2^31
    dbl <- function(p, w) {
      v <- build(as.integer(p - side * 2^31), w)
      ifelse(side, -v, v)
    }
    data.frame(
      bits_from = sprintf("0x%s00000000", hex32(lo)),
      bits_to = sprintf("0x%sFFFFFFFF", hex32(hi)),
      x_from = dbl(lo, 0L),
      x_to = dbl(hi, -1L),
      blocks = hi - lo + 1
    )
  })

  cat(sprintf("sampled              : %.0f values\n", 2 * length(EXPS) * CHUNK))
  cat(sprintf("worst finite rel err : %.6g\n", if (nrow(worst)) worst$rel_err[1] else NA))
  cat(sprintf("Inf-error samples    : %.0f in %d range(s)\n", sum(inf_ranges$blocks), nrow(inf_ranges)))
  print(worst, digits = 17)
  print(inf_ranges, digits = 17)
}
