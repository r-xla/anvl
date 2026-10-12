# The `author` field of every `bibentry()` below MUST be given as `person()`
# objects (a single `person()`, or several combined with `c()`)
# nolint start
bibentries <- c(
  murray2016differentiation = bibentry(
    bibtype = "article",
    title = "Differentiation of the Cholesky decomposition",
    author = person("Iain", "Murray"),
    journal = "arXiv preprint arXiv:1602.07527",
    year = "2016"
  ),
  giles2008extended = bibentry(
    bibtype = "techreport",
    title = "An extended collection of matrix derivative results for forward and reverse mode automatic differentiation",
    author = person("Mike", "Giles"),
    year = "2008",
    institution = "Oxford University Computing Laboratory"
  ),
  walter2012structured = bibentry(
    bibtype = "phdthesis",
    title = "Structured higher-order algorithmic differentiation in the forward and reverse mode with application in optimum experimental design",
    author = person("Sebastian", "Walter"),
    year = "2012",
    school = "Mathematisch-Naturwissenschaftliche Fakult{\"a}t II"
  ),
  abramowitz1964handbook = bibentry(
    bibtype = "book",
    title = "Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables",
    author = c(person("Milton", "Abramowitz"), person("Irene A.", "Stegun")),
    year = "1964",
    publisher = "Dover Publications",
    address = "New York",
    series = "Applied Mathematics Series",
    number = "55",
    isbn = "0-486-61272-4"
  ),
  moshier1989methods = bibentry(
    bibtype = "book",
    title = "Methods and Programs for Mathematical Functions",
    author = person("Stephen L.", "Moshier"),
    year = "1989",
    publisher = "Ellis Horwood",
    isbn = "0-7458-0289-3"
  )
)
# nolint end

# Formats the complete entries for the given keys for an `@references` section.
format_bib <- function(...) {
  str <- vapply(
    list(...),
    function(key) tools::toRd(bibentries[[key]]),
    character(1)
  )
  paste0(str, collapse = "\n\n")
}

# A short inline citation listing all authors, e.g. "A (Year)", "A & B (Year)",
# "A, B & C (Year)", or "A et al. (Year)" for four or more.
cite_bib <- function(...) {
  str <- vapply(
    list(...),
    function(key) {
      x <- bibentries[[key]]
      sprintf("%s (%s)", format_authors(x$author), x$year)
    },
    character(1)
  )
  if (length(str) >= 3L) {
    str <- c(toString(head(str, -1L)), tail(str, 1L))
  }
  paste0(str, collapse = " and ")
}

format_authors <- function(authors) {
  families <- vapply(
    authors,
    # Organizational/mononymous authors (e.g. person(given = "R Core Team"))
    # have no family name.
    function(p) p$family %||% as.character(p),
    character(1)
  )
  n <- length(families)
  if (n <= 2L) {
    paste(families, collapse = " & ")
  } else if (n == 3L) {
    paste0(toString(families[1:2]), " & ", families[3L])
  } else {
    paste(families[1L], "et al.")
  }
}
