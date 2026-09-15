# Helper: build a minimal `@jit` roxy_tag and run the parser, returning the
# parsed `static` value (or the tag itself, to inspect warnings).
parse_jit_static <- function(raw) {
  tag <- structure(list(raw = raw), class = c("roxy_tag_jit", "roxy_tag"))
  roxy_tag_parse.roxy_tag_jit(tag)$val$static
}

test_that("@jit parses an empty tag as no static args", {
  expect_equal(parse_jit_static(""), character())
})

test_that("@jit parses `static = <expr>` forms", {
  expect_equal(parse_jit_static("static = c(2L, 3L, 4L, 5L)"), 2:5)
  expect_equal(parse_jit_static("static = 2:5"), 2:5)
  expect_equal(parse_jit_static('static = c("a", "b")'), c("a", "b"))
  expect_equal(parse_jit_static("static = c(1L, 3L, 4L)"), c(1L, 3L, 4L))
})

test_that("@jit supports the terse form without `=`", {
  expect_equal(parse_jit_static("static 2:5"), 2:5)
  expect_equal(parse_jit_static("static  2:5"), 2:5)
  expect_equal(parse_jit_static("static c(1L, 3L, 4L)"), c(1L, 3L, 4L))
  expect_equal(parse_jit_static("static 2L"), 2L)
  expect_equal(parse_jit_static('static "axis"'), "axis")
})

test_that("@jit coerces numeric static positions to integer", {
  expect_identical(parse_jit_static("static 2:5"), 2:5)
  expect_identical(parse_jit_static("static c(2, 3)"), c(2L, 3L))
})
