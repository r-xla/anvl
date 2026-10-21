describe("article_titles", {
  it("lists every article with its own title", {
    dir <- test_path("..", "..", "vignettes", "articles")
    skip_if_not(dir.exists(dir), "the articles are not part of the built package")
    files <- list.files(dir, pattern = "\\.Rmd$")
    titles <- vapply(
      file.path(dir, files),
      function(f) sub('^title: "(.*)"$', "\\1", grep("^title:", readLines(f), value = TRUE)[[1L]]),
      character(1L)
    )
    expect_equal(article_titles[sort(names(article_titles))], setNames(titles, sub("\\.Rmd$", "", files)))
  })
})

describe("article_url", {
  it("links to the article on the website", {
    expect_equal(article_url("subsetting"), "https://r-xla.github.io/anvl/articles/subsetting.html")
  })

  it("errors on an unknown article", {
    expect_error(article_url("does-not-exist"), "Unknown article")
  })
})

describe("roxy_article", {
  it("reads as a markdown link to the article", {
    expect_equal(
      roxy_article("jit"),
      "the [JIT Deep Dive](https://r-xla.github.io/anvl/articles/jit.html) article"
    )
  })
})
