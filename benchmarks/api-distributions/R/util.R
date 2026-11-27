## ---------------------------------------------------------------------------
## Bit-level helpers shared by the sweep engine.
##
## Everything here is about moving between an IEEE-754 value and the integer
## bit pattern that encodes it. The pattern is the ground truth: it survives
## transport exactly, it is what a regression test pins, and it is what makes
## "is this the same failing input as last month" a string comparison rather
## than a floating-point one.
## ---------------------------------------------------------------------------

LITTLE_ENDIAN <- .Platform$endian == "little"

## ---- f32 -------------------------------------------------------------------

## 32-bit pattern (as a signed R integer) -> the float32 it encodes, widened to
## a double.
f32_from_bits <- function(i) {
  readBin(writeBin(i, raw(), size = 4L), "double", size = 4L, n = length(i))
}

## double -> nearest float32, widened back to a double (round-to-nearest-even).
as_f32 <- function(x) {
  readBin(writeBin(x, raw(), size = 4L), "double", size = 4L, n = length(x))
}

## ---- f64 -------------------------------------------------------------------

## (high word, low word) -> the float64 those 64 bits encode. Both words are
## signed R integers holding raw patterns.
f64_from_words <- function(hi, lo) {
  readBin(
    writeBin(as.vector(if (LITTLE_ENDIAN) rbind(lo, hi) else rbind(hi, lo)), raw(), size = 4L),
    "double",
    size = 8L,
    n = length(hi)
  )
}

## Negative zero, built from its bit pattern (0x8000000000000000; the high word
## 0x80000000 is NA_integer_'s pattern). Never write the literal `-0` inside a
## function: R's byte-code compiler (JIT level 3, R 4.6.1, verified) folds it to
## +0 in some call shapes -- `c(0, -0)` in a compiled function returns two
## positive zeros -- so every "evaluate at -0" silently became "at +0".
NEG_ZERO <- f64_from_words(NA_integer_, 0L)
stopifnot(NEG_ZERO == 0, 1 / NEG_ZERO == -Inf)

## n uniform 32-bit words, as R integers holding the raw bit patterns.
## as.integer(-2^31) overflows to NA_integer_, whose own bit pattern is
## 0x80000000 -- exactly the word we wanted -- so the coercion is correct.
rand_word32 <- function(n) {
  suppressWarnings(as.integer(sample.int(2^32, n, replace = TRUE) - (2^31 + 1)))
}

## ---- pattern formatting ----------------------------------------------------

## Signed R integer -> the unsigned value its bits encode, held as a double.
##
## NA_integer_ is not missingness here. R represents NA_integer_ as the bit
## pattern 0x80000000, so readBin() hands back NA for precisely the word whose
## unsigned value is 2^31 -- and without this branch every -0, and every f64
## whose high or low word happens to be 0x80000000, formatted as "0x00NA00NA".
u32 <- function(i) ifelse(is.na(i), 2^31, ifelse(i < 0, i + 2^32, i))

hex32 <- function(p) sprintf("%04X%04X", p %/% 65536, p %% 65536)

## Hex bit patterns, as strings. Strings are deliberate: nanoparquet widens
## 64-bit integers to doubles on the way back out, which would silently round
## any pattern above 2^53; a string round-trips exactly. Verified, not assumed.
bits_of <- function(x, dtype) {
  if (!length(x)) {
    return(character(0))
  }
  if (dtype == "f32") {
    w <- readBin(writeBin(as_f32(x), raw(), size = 4L), "integer", size = 4L, n = length(x))
    sprintf("0x%s", hex32(u32(w)))
  } else {
    w <- matrix(
      readBin(writeBin(x, raw(), size = 8L), "integer", size = 4L, n = 2 * length(x)),
      nrow = 2
    )
    ## sprintf rather than paste0: keeps a length-0 input at length 0
    sprintf("0x%s%s", hex32(u32(w[2, ])), hex32(u32(w[1, ])))
  }
}

## ---- neighbours ------------------------------------------------------------

## The representable neighbours of each x in `dtype`, one step towards -Inf and
## one towards +Inf, as a two-column matrix (down, up). Worked on the bit
## pattern -- sign plus magnitude -- so it is exact across binade and sign
## boundaries: the step down from +0 is the smallest negative subnormal, the
## step up from the largest finite value is Inf. Non-finite x gives NA.
float_neighbours <- function(x, dtype) {
  move <- function(neg, hi, lo, towards_pos, carry) {
    grow <- (towards_pos & !neg) | (!towards_pos & neg)
    zero <- hi == 0 & lo == 0
    cross <- !grow & zero
    lo2 <- ifelse(grow, lo + 1, lo - 1)
    hi2 <- hi
    if (carry) {
      up <- grow & lo2 == 2^32
      hi2[up] <- hi[up] + 1
      lo2[up] <- 0
      dn <- !grow & !zero & lo == 0
      hi2[dn] <- hi[dn] - 1
      lo2[dn] <- 2^32 - 1
    }
    neg2 <- neg
    neg2[cross] <- !neg[cross]
    lo2[cross] <- 1
    hi2[cross] <- 0
    list(neg = neg2, hi = hi2, lo = lo2)
  }
  as_int <- function(u) suppressWarnings(as.integer(ifelse(u >= 2^31, u - 2^32, u)))
  finite <- is.finite(x)
  out <- matrix(NA_real_, length(x), 2L, dimnames = list(NULL, c("down", "up")))
  if (!any(finite)) {
    return(out)
  }
  xf <- x[finite]
  if (dtype == "f32") {
    w <- u32(readBin(writeBin(as_f32(xf), raw(), size = 4L), "integer", size = 4L, n = length(xf)))
    neg <- w >= 2^31
    for (j in 1:2) {
      m <- move(neg, 0, w %% 2^31, towards_pos = j == 2L, carry = FALSE)
      out[finite, j] <- f32_from_bits(as_int(m$lo + m$neg * 2^31))
    }
  } else {
    w <- matrix(readBin(writeBin(xf, raw(), size = 8L), "integer", size = 4L, n = 2 * length(xf)), nrow = 2)
    hi <- u32(w[2, ])
    lo <- u32(w[1, ])
    neg <- hi >= 2^31
    for (j in 1:2) {
      m <- move(neg, hi %% 2^31, lo, towards_pos = j == 2L, carry = TRUE)
      out[finite, j] <- f64_from_words(as_int(m$hi + m$neg * 2^31), as_int(m$lo))
    }
  }
  out
}

## ---- ulp spacing -----------------------------------------------------------

## The spacing of representable values in the binade holding |x|: the unit
## the ulp error metric is measured in, and the unit the reference-dispute
## thresholds are built from. Subnormals and zero share the spacing of the
## smallest normal binade, hence the floor on the exponent. For an f32 spacing
## at a double that is not itself an f32, the binade is the double's.
##
## The exponent is floor(log2|x|) *corrected*: log2 rounds up to the next
## integer for values just below a power of two (the largest double below 2^e
## gives exactly e), which doubled the spacing there. 2^e is exact, so one
## comparison each way fixes it. At a power of two itself this is the spacing
## above x, the larger of the two neighbouring gaps.
##
## Non-finite x has no spacing and gets NaN, so it can never enter a
## threshold silently; score_pair() routes non-finite values before dividing.
ulp_size <- function(x, dtype) {
  p <- if (dtype == "f32") 23L else 52L
  emin <- if (dtype == "f32") -126L else -1022L
  a <- abs(x)
  e <- floor(log2(a))
  ok <- is.finite(e)
  e[ok] <- e[ok] - (2^e[ok] > a[ok]) + (2^(e[ok] + 1) <= a[ok])
  e <- pmax(e, emin)
  e[x %in% 0] <- emin
  out <- 2^(e - p)
  out[!is.finite(x)] <- NaN
  out
}

## ---- formatting ------------------------------------------------------------

## Compact, readable numbers: no 17-digit doubles and no e+00 on small integers.
fmt_num <- function(x) {
  if (length(x) != 1L) {
    return(vapply(x, fmt_num, ""))
  }
  ## is.na() is TRUE for NaN, so NaN must be tested first or it prints as NA --
  ## a distinction that matters a great deal in these results.
  if (is.nan(x)) {
    return("NaN")
  }
  if (is.na(x)) {
    return("NA")
  }
  if (is.infinite(x)) {
    return(if (x > 0) "Inf" else "-Inf")
  }
  if (x == 0) {
    return("0")
  }
  if (abs(x) >= 1e-3 && abs(x) < 1e5) format(signif(x, 4), trim = TRUE) else sprintf("%.2e", x)
}

## A section heading, padded to a fixed width.
rule <- function(title) {
  cat("\n", title, " ", strrep("\u2500", max(4L, 68L - nchar(title))), "\n", sep = "")
}

## A compact one-line identifier for a cell: parameter set plus the flags that
## are actually on, e.g. "unit/upper,log".
short_cell <- function(row) {
  fl <- character(0)
  if (nzchar(row$flags) && row$flags != "-") {
    for (kv in strsplit(strsplit(row$flags, ",", fixed = TRUE)[[1L]], "=", fixed = TRUE)) {
      on <- isTRUE(as.logical(kv[2L]))
      fl <- c(
        fl,
        switch(
          kv[1L],
          lower_tail = if (on) "lower" else "upper",
          log_p = if (on) "log" else NULL,
          log = if (on) "log" else NULL,
          if (on) kv[1L] else NULL
        )
      )
    }
  }
  paste0(row$param_set, if (length(fl)) paste0("/", paste(fl, collapse = ",")) else "")
}
