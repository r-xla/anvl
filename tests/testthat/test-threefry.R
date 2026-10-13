# Runs a program on the RNG state given as its four 32-bit words (key low/high,
# counter low/high): `draw(state)` builds the draw from the `ui64[2]` state, and
# each result comes back as `ui32` words, a `ui64` as its low and high word.
run_on_state <- function(words, draw) {
  func <- stablehlo::local_func()
  input <- hlo_input("words", "ui32", shape = 4L)
  state <- hlo_bitcast_convert(hlo_reshape(input, c(2L, 2L)), "ui64")
  as_words <- function(x) {
    if (identical(as.character(dtype(x$value_type)), "ui64")) hlo_bitcast_convert(x, "ui32") else x
  }
  func <- do.call(hlo_return, lapply(draw(state), as_words))
  exec <- pjrt::pjrt_compile(pjrt::pjrt_program(stablehlo::repr(func)))
  out <- pjrt::pjrt_execute(exec, pjrt::pjrt_buffer(words, dtype = "ui32"))
  if (inherits(out, "PJRTBuffer")) {
    out <- list(out)
  }
  # a `ui64` comes back as a matrix of its words, one row per value
  lapply(out, \(x) {
    a <- pjrt::as_array(x)
    unname(as.numeric(if (is.matrix(a)) t(a) else a))
  })
}

# A 32-bit word given in hex, as a number.
h <- function(s) strtoi(substr(s, 1L, 4L), 16L) * 65536 + strtoi(substr(s, 5L, 8L), 16L)

describe("hlo_threefry_bits", {
  it("matches the known-answer vectors of the ThreeFry-2x32 reference", {
    kat <- list(
      list(ctr = c("00000000", "00000000"), key = c("00000000", "00000000"), out = c("6b200159", "99ba4efe")),
      list(ctr = c("ffffffff", "ffffffff"), key = c("ffffffff", "ffffffff"), out = c("1cb996fc", "bb002be7")),
      list(ctr = c("243f6a88", "85a308d3"), key = c("13198a2e", "03707344"), out = c("c4923a9c", "483df7a0"))
    )
    for (k in kat) {
      out <- run_on_state(h(c(k$key, k$ctr)), \(state) hlo_threefry_bits(state, 1L, "ui64")[1L])
      expect_identical(out[[1L]], h(k$out))
    }
  })

  it("draws what XLA's THREE_FRY draws, and advances the state alike", {
    # the counter carries into its high word, and then wraps, during the draw
    for (words in list(c(123456789, 0, 2^32 - 3, 0), c(7, 1, 2^32 - 2, 2^32 - 1))) {
      ours <- run_on_state(words, \(state) hlo_threefry_bits(state, 9L, "ui64"))
      xla <- run_on_state(words, \(state) rev(hlo_rng_bit_generator(state, "THREE_FRY", "ui64", 9L)))
      expect_identical(ours, xla)
    }
  })

  it("continues a draw where the previous one stopped", {
    words <- c(99, 0, 2^32 - 5, 0)
    for (dtype in c("ui8", "ui16", "ui32", "ui64")) {
      parts <- run_on_state(words, \(state) {
        first <- hlo_threefry_bits(state, 5L, dtype)
        list(first[[1L]], hlo_threefry_bits(first[[2L]], 3L, dtype)[[1L]])
      })
      whole <- run_on_state(words, \(state) hlo_threefry_bits(state, 8L, dtype)[1L])
      expect_identical(unlist(parts), whole[[1L]])
    }
  })

  it("advances the counter by the number of values whatever the data type", {
    for (dtype in c("ui8", "ui32", "ui64")) {
      out <- run_on_state(c(5, 0, 10, 0), \(state) hlo_threefry_bits(state, c(2L, 3L), dtype)[2L])
      # key (5, 0), counter (16, 0)
      expect_identical(out[[1L]], c(5, 0, 16, 0))
    }
  })
})

describe("hlo_threefry_split", {
  # The keys a split of the key with words `key` (low, high) into `shape` gives,
  # as their words in column-major order of `shape`.
  split_words <- function(key, shape) {
    run_on_state(c(key, 0, 0), \(state) {
      keys <- hlo_threefry_split(hlo_reshape(hlo_slice(state, 0L, 1L, 1L), integer()), shape)
      if (length(shape) > 1L) {
        keys <- hlo_transpose(keys, rev(seq_along(shape)) - 1L)
      }
      list(hlo_reshape(keys, prod(shape)))
    })[[1L]]
  }

  it("derives the keys that jax.random.split() derives", {
    # jax 0.7.2: jax.random.key_data(jax.random.split(key, 5)) for the key data
    # `key`, whose first word is the low word of our key
    ref <- list(
      list(
        key = c(0, 0),
        out = c(
          1797259609,
          2579123966,
          928981903,
          3453687069,
          4146024105,
          2718843009,
          2467461003,
          3840466878,
          2285895361,
          433833334
        )
      ),
      list(
        key = c(0, 42),
        out = c(
          1832780943,
          270669613,
          64467757,
          2916123636,
          2465931498,
          255383827,
          3134548294,
          894150801,
          2954079971,
          3276725750
        )
      ),
      list(
        key = c(h("13198a2e"), h("03707344")),
        out = c(
          1095259177,
          3512222810,
          3520574224,
          438507793,
          4239014620,
          2355263051,
          2319929243,
          4059551135,
          893101043,
          618869991
        )
      )
    )
    for (r in ref) {
      expect_identical(split_words(r$key, 5L), r$out)
    }
  })

  it("numbers the keys of a shape in column-major order", {
    # jax.random.split(key, (2, 3)) for the key data (1, 2), whose keys JAX
    # numbers in row-major order
    jax <- c(
      629071667,
      2343584484,
      629003988,
      1317161160,
      4128513521,
      2345921600,
      2052595727,
      1880214352,
      1226884191,
      508220933,
      2593290630,
      1369596596
    )
    expect_identical(split_words(c(1, 2), c(2L, 3L)), jax)
  })
})
