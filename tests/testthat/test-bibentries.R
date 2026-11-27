describe("cite_bib", {
  it("cites every entry by its authors and year", {
    for (key in names(bibentries)) {
      expect_match(cite_bib(key), sprintf("\\(%s\\)$", bibentries[[key]]$year))
    }
  })

  it("joins several citations with 'and'", {
    expect_identical(
      cite_bib("murray2016differentiation", "giles2008extended"),
      paste(
        cite_bib("murray2016differentiation"),
        "and",
        cite_bib("giles2008extended")
      )
    )
  })
})

describe("format_authors", {
  it("lists up to three authors and abbreviates four or more", {
    p <- function(family) person(given = "X", family = family)
    expect_identical(format_authors(p("A")), "A")
    expect_identical(format_authors(c(p("A"), p("B"))), "A & B")
    expect_identical(format_authors(c(p("A"), p("B"), p("C"))), "A, B & C")
    expect_identical(format_authors(c(p("A"), p("B"), p("C"), p("D"))), "A et al.")
  })

  it("uses the given name of authors without a family name", {
    expect_identical(format_authors(person(given = "R Core Team")), "R Core Team")
  })
})

describe("format_bib", {
  it("formats each entry as Rd, separated by blank lines", {
    out <- format_bib("murray2016differentiation", "giles2008extended")
    expect_identical(
      out,
      paste(
        format_bib("murray2016differentiation"),
        format_bib("giles2008extended"),
        sep = "\n\n"
      )
    )
    expect_match(format_bib("giles2008extended"), "Giles", fixed = TRUE)
  })
})
