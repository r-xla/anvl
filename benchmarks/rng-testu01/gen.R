#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## Writes anvl's random stream as raw binary to stdout, for a test battery to
## read from a pipe:
##
##   Rscript gen.R --seed 42 | ./testu01 smallcrush
##   Rscript gen.R --seed 42 | ./RNG_test stdin32
##
## The stream is what a user sees: the values of consecutive sampling calls
## (calls of `prim_sample_bits()`), --per-call values each. Every sampling call
## splits the random state (k, c) and draws from the fresh key ThreeFry_k(c), so
## with few values per call the stream consists of many keys each used for a
## few values.
##
## Drawing hundreds of millions of values one sampling call at a time is far too
## slow, so the stream is computed vectorized, many sampling calls at once, by a
## ThreeFry-2x32 written below in anvl's array operations. Before writing
## anything, gen.R checks that this gives exactly what the same number of real
## sampling calls gives, and stops if it does not.
##
## Options (defaults in brackets):
##   --seed S       the seed passed to nv_set_seed() [42]
##   --dtype D      ui32 | ui64: the data type of prim_sample_bits() [ui32]
##   --mode M       bits:  the raw words of prim_sample_bits(), 32 bits each
##                         (a ui64 word is written low half first)
##                  unif:  nv_runif(dtype = "f64") as 64-bit doubles, for
##                         `testu01 -d` [bits]
##   --per-call P   values drawn per sampling call [1]
##   --streams K    interleave the K streams seeded S, S + 1, ..., S + K - 1:
##                  --chunk numbers of the first, then of the second, ..., then
##                  the next --chunk numbers of the first again [1]
##   --chunk C      numbers taken from one stream before switching to the next
##                  when --streams > 1 [1]
##   --calls M      sampling calls computed at once, per stream [2^20 / K]
##   --n N          stop after N numbers (default: until the reader closes)
##   --out PATH     where to write [/dev/stdout]
##
## A "number" is one element of a draw: a ui32 / ui64 word or a double.
## anvl is loaded with pkgload::load_all() from the checkout this script lives
## in; set ANVL_INSTALLED=1 to use the installed package instead. Everything
## that is not the stream goes to stderr.
## ---------------------------------------------------------------------------

HERE <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", grep("^--file=", a, value = TRUE))
  if (length(f)) dirname(normalizePath(f[1L])) else normalizePath(".")
})

parse_args <- function(argv) {
  o <- list(
    seed = 42L,
    dtype = "ui32",
    mode = "bits",
    `per-call` = 1L,
    streams = 1L,
    chunk = 1L,
    calls = NA,
    n = Inf,
    out = "/dev/stdout"
  )
  i <- 1L
  while (i <= length(argv)) {
    key <- sub("^--", "", argv[[i]])
    if (!key %in% names(o) || i == length(argv)) {
      stop("unknown or incomplete option: ", argv[[i]], call. = FALSE)
    }
    val <- argv[[i + 1L]]
    o[[key]] <- switch(
      key,
      seed = ,
      `per-call` = ,
      streams = ,
      chunk = as.integer(val),
      calls = ,
      n = as.numeric(val),
      val
    )
    i <- i + 2L
  }
  stopifnot(
    o$dtype %in% c("ui32", "ui64"),
    o$mode %in% c("bits", "unif"),
    o$`per-call` >= 1L,
    o$streams >= 1L,
    o$chunk >= 1L
  )
  if (is.na(o$calls)) {
    o$calls <- max(1, 2^20 %/% o$streams)
  }
  o$calls <- as.integer(o$calls)
  o
}

o <- parse_args(commandArgs(trailingOnly = TRUE))

## Nothing but the stream may reach stdout.
sink(stderr())
suppressPackageStartupMessages({
  if (identical(Sys.getenv("ANVL_INSTALLED"), "1")) {
    library(anvl)
  } else {
    pkgload::load_all(file.path(HERE, "..", ".."), quiet = TRUE)
  }
})

per_call <- o$`per-call`

## ThreeFry-2x32 with 20 rounds of the keys (k0, k1) and the counters (x0, x1),
## `ui32` arrays of one shape.
threefry <- function(k0, k1, x0, x1) {
  rotl <- function(v, d) nv_or(nv_shift_left(v, d), nv_shift_right_logical(v, 32L - d))
  ks <- list(k0, k1, nv_xor(nv_xor(k0, k1), 0x1BD11BDAL))
  x0 <- x0 + ks[[1L]]
  x1 <- x1 + ks[[2L]]
  rotations <- list(c(13L, 15L, 26L, 6L), c(17L, 29L, 16L, 24L))
  for (i in 1:5) {
    for (d in rotations[[(i - 1L) %% 2L + 1L]]) {
      x0 <- x0 + x1
      x1 <- nv_xor(x0, rotl(x1, d))
    }
    x0 <- x0 + ks[[i %% 3L + 1L]]
    x1 <- x1 + ks[[(i + 1L) %% 3L + 1L]] + as.integer(i)
  }
  list(x0, x1)
}

## The numbers of the sampling calls with the counters (c_lo, c_hi) under the
## key (k0, k1), in the order the stream writes them, as JAX computes them: the
## key of a call is the hash of its counter (high word first), and its value at
## position t the hash of (0, t) under that key. A `ui32` value is the hash's
## two words xor'd, a 64-bit value its first word above its second. The stream
## writes the `ui32` values bitcast to `i32`, the `ui64` values bitcast to
## `i64`, and the `f64` values of a unif stream, computed as nv_runif()
## computes them. A bitcast keeps the bits, and reads back into R as an integer
## or a bit64 integer64 that writeBin() writes with these bits.
calls_numbers <- jit(function(k0, k1, c_lo, c_hi) {
  m <- shape(c_lo)
  keys <- threefry(nv_broadcast_to(k0, m), nv_broadcast_to(k1, m), c_hi, c_lo)
  zero <- nv_fill(0L, m, dtype = "ui32")
  # one hash per value of a call
  hashes <- lapply(seq_len(per_call) - 1L, function(t) {
    threefry(keys[[1L]], keys[[2L]], zero, nv_fill(t, m, dtype = "ui32"))
  })
  # one row per value of a call, one column per call
  rows <- function(values) do.call(nv_concatenate, c(lapply(values, nv_reshape, c(1L, m)), list(axis = 1L)))
  if (o$mode == "bits" && o$dtype == "ui32") {
    return(nv_bitcast_convert(rows(lapply(hashes, function(h) nv_xor(h[[1L]], h[[2L]]))), dtype = "i32"))
  }
  values <- rows(lapply(hashes, function(h) {
    nv_or(nv_shift_left(nv_convert(h[[1L]], dtype = "ui64"), 32L), nv_convert(h[[2L]], dtype = "ui64"))
  }))
  if (o$mode == "bits") {
    return(nv_bitcast_convert(values, dtype = "i64"))
  }
  one <- nv_bitcast_convert(nv_scalar(1, dtype = "f64"), dtype = "ui64")
  nv_bitcast_convert(nv_or(nv_shift_right_logical(values, 11L), one), dtype = "f64") - 1
})

## The numbers of the sampling calls c, ..., c + m - 1 of the random state with
## the key words `key` (doubles), in the order the stream writes them.
block_numbers <- function(key, c, m) {
  counters <- c + seq_len(m) - 1
  x <- as_array(
    calls_numbers(
      nv_scalar(key[[1L]], dtype = "ui32"),
      nv_scalar(key[[2L]], dtype = "ui32"),
      nv_array(counters %% 2^32, dtype = "ui32"),
      nv_array(counters %/% 2^32, dtype = "ui32")
    ),
    check = FALSE
  )
  attributes(x) <- NULL
  x
}

## The same numbers from real sampling calls, eagerly.
real_numbers <- function(seed, m) {
  nv_set_seed(seed)
  draw <- if (o$mode == "unif") {
    function() nv_runif(per_call, dtype = "f64")
  } else if (o$dtype == "ui32") {
    function() nv_bitcast_convert(prim_sample_bits(per_call, "ui32"), dtype = "i32")
  } else {
    function() nv_bitcast_convert(prim_sample_bits(per_call, "ui64"), dtype = "i64")
  }
  out <- lapply(seq_len(m), function(i) {
    x <- as_array(draw(), check = FALSE)
    attributes(x) <- NULL
    x
  })
  unlist(out)
}

## The check: the first 64 sampling calls of seed S.
local({
  key <- c(0, o$seed %% 2^32)
  vectorized <- block_numbers(key, 0, 64L)
  real <- real_numbers(o$seed, 64L)
  if (!identical(vectorized, real)) {
    stop("the vectorized stream differs from the one real sampling calls give", call. = FALSE)
  }
  message("gen.R: the vectorized stream matches 64 real sampling calls")
})

## Each stream is its key and the counter of its next sampling call.
seeds <- o$seed + seq_len(o$streams) - 1L
counters <- rep(0, o$streams)

next_block <- function() {
  xs <- lapply(seq_len(o$streams), function(k) {
    x <- block_numbers(c(0, seeds[[k]] %% 2^32), counters[[k]], o$calls)
    counters[[k]] <<- counters[[k]] + o$calls
    x
  })
  if (o$streams == 1L) {
    return(xs[[1L]])
  }
  # [chunk, stream, round] in column-major order is the interleaved stream
  per_stream <- length(xs[[1L]])
  x <- array(unlist(xs), c(o$chunk, per_stream %/% o$chunk, o$streams))
  as.vector(aperm(x, c(1L, 3L, 2L)))
}

width <- if (o$mode == "bits" && o$dtype == "ui32") 4L else 8L
con <- file(o$out, open = "wb", raw = TRUE)
message(sprintf(
  "gen.R: mode=%s dtype=%s %s per-call=%d calls=%d chunk=%d",
  o$mode,
  o$dtype,
  if (o$streams == 1L) sprintf("seed=%d", o$seed) else sprintf("seeds=%d..%d", seeds[[1L]], seeds[[o$streams]]),
  per_call,
  o$calls,
  o$chunk
))
written <- 0
start <- Sys.time()
done <- tryCatch(
  {
    while (written < o$n) {
      x <- next_block()
      if (written + length(x) > o$n) {
        x <- x[seq_len(o$n - written)]
      }
      writeBin(x, con, size = width)
      written <- written + length(x)
    }
    flush(con)
    TRUE
  },
  # usually the reader closing the pipe once it has read all it needed
  error = function(e) {
    message("gen.R: stopped: ", conditionMessage(e))
    FALSE
  }
)
secs <- as.numeric(difftime(Sys.time(), start, units = "secs"))
message(sprintf(
  "gen.R: wrote %.0f numbers (%.1f MB) in %.1f s, %.1f MB/s%s",
  written,
  written * width / 1e6,
  secs,
  written * width / 1e6 / secs,
  if (done) "" else " (reader closed the stream)"
))
