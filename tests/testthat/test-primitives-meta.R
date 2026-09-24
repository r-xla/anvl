test_that("stablehlo rule is tested", {
  nms <- names(asNamespace("anvl"))
  primitive_names <- nms[grepl("^prim_", nms)]

  tests_dir <- testthat::test_path()
  candidate_files <- c(
    system.file("extra-tests", "test-primitives-stablehlo-torch.R", package = "anvl"),
    file.path(testthat::test_path(), "test-primitives-stablehlo.R")
  )

  content <- paste(
    vapply(candidate_files, function(file) paste(readLines(file, warn = FALSE), collapse = "\n"), character(1L)),
    collapse = "\n"
  )
  missing <- Filter(
    function(nm) {
      !grepl(paste0('(test_that|describe|it)\\("', nm), content)
    },
    primitive_names
  )

  expect_true(length(missing) == 0L, info = paste(missing, collapse = ", "), label = "stablehlo rule is tested")
})


test_that("reverse rule is tested", {
  nms <- names(asNamespace("anvl"))
  primitive_names <- nms[grepl("^prim_", nms)]

  primitive_names <- Filter(
    function(nm) {
      !is.null(getFromNamespace(nm, "anvl")[["reverse"]])
    },
    primitive_names
  )

  candidate_files <- c(
    system.file("extra-tests", "test-primitives-reverse-torch.R", package = "anvl"),
    file.path(testthat::test_path(), "test-primitives-reverse.R")
  )

  content <- do.call(c, lapply(candidate_files, readLines))
  content <- content[grepl("(test_that|describe|it)\\(", content)]
  missing <- Filter(function(nm) !any(grepl(nm, content, fixed = TRUE)), primitive_names)

  expect_true(length(missing) == 0L, info = paste(missing, collapse = ", "), label = "Reverse rule is tested")
})

test_that("the call an inference error reports is a function that exists", {
  # A primitive is named for the `prim_*()` that exports it, not for the
  # StableHLO op it lowers to, so an error can name the call by putting the
  # prefix back on.
  nms <- names(asNamespace("anvl"))
  exported <- nms[grepl("^prim_", nms)]
  for (nm in exported) {
    obj <- getFromNamespace(nm, "anvl")
    if (!inherits(obj, "JitPrimitive")) {
      next
    }
    reported <- deparse(print_call_repr(attr(obj, "primitive")))
    expect_identical(reported, paste0(nm, "()"), info = nm)
  }
})
