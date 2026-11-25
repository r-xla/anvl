# Extract the R code shown in main.tex into the standalone replication script paper_code.R.
#
# Run from the paper/ directory: Rscript extract.R (or: make code)
#
# Takes the lines of all CodeInput environments, which use the JSS prompts "R> " and "+  ",
# strips the prompts, and precedes the code of each (sub)section with a comment
# "# <n>. <section title>" / "# <n>.<m> <subsection title>". The code of one CodeChunk is written
# as one block. A CodeInput whose output starts with "Error" is wrapped in a spin chunk with
# error = TRUE, so that knitr::spin() shows the error and continues.
#
# Render the result with knitr::spin("paper_code.R"); a plain Rscript run stops at that error.

tex_file <- readLines("main.tex")

code_lines <- character(0)

# Section numbers, used to precede the code with comments that link it to the sections of the paper.
# The comment of a (sub)section is only written before its first code block.
section <- 0L
subsection <- 0L
in_appendix <- FALSE
pending_headers <- character(0)

# Title of a (sub)section, preferring the short title in [...], which has no LaTeX markup
section_title <- function(line) {
  title <- if (grepl("^\\\\(sub)?section\\[", line)) {
    sub("^\\\\(sub)?section\\[([^]]*)\\].*", "\\2", line)
  } else {
    sub("^\\\\(sub)?section\\{([^}]*)\\}.*", "\\2", line)
  }
  gsub("\\\\(code|pkg|proglang)\\{([^}]*)\\}", "\\2", title)
}

section_number <- function() if (in_appendix) LETTERS[section] else as.character(section)

in_code_block <- FALSE
block <- character(0) # code of the current CodeInput
chunk <- character(0) # code of the current CodeChunk
last_block <- character(0) # code of the last CodeInput, until its output is known
await_output <- FALSE

flush_block <- function(errors) {
  if (errors) {
    chunk <<- c(chunk, "#+ error = TRUE", last_block, "#+")
  } else {
    chunk <<- c(chunk, last_block)
  }
  last_block <<- character(0)
  await_output <<- FALSE
}

for (i in seq_along(tex_file)) {
  line <- tex_file[[i]]

  if (grepl("^\\\\begin\\{appendix\\}", line)) {
    in_appendix <- TRUE
    section <- 0L
  }

  # Track the (unstarred) sections and subsections
  if (grepl("^\\\\section[[{]", line)) {
    section <- section + 1L
    subsection <- 0L
    pending_headers <- c(sprintf("# %s. %s", section_number(), section_title(line)), "")
  } else if (grepl("^\\\\subsection[[{]", line)) {
    subsection <- subsection + 1L
    # keep the header of the section if none of its code was written yet
    pending_headers <- c(
      pending_headers[startsWith(pending_headers, sprintf("# %s. ", section_number())) | pending_headers == ""],
      sprintf("# %s.%i %s", section_number(), subsection, section_title(line))
    )
  }

  if (grepl("^\\\\begin\\{CodeInput\\}", line)) {
    if (await_output) flush_block(errors = FALSE)
    in_code_block <- TRUE
    block <- character(0)
    next
  }

  if (grepl("^\\\\end\\{CodeInput\\}", line)) {
    in_code_block <- FALSE
    last_block <- block
    await_output <- TRUE
    next
  }

  if (grepl("^\\\\begin\\{CodeOutput\\}", line)) {
    if (await_output) flush_block(errors = startsWith(tex_file[[i + 1L]], "Error"))
    next
  }

  if (grepl("^\\\\end\\{CodeChunk\\}", line)) {
    if (await_output) flush_block(errors = FALSE)
    if (length(chunk)) {
      code_lines <- c(code_lines, pending_headers, chunk, "")
      pending_headers <- character(0)
    }
    chunk <- character(0)
    next
  }

  # Inside a code block, every line starts with a prompt
  if (in_code_block) {
    if (startsWith(line, "R> ")) {
      block <- c(block, substring(line, 4L))
    } else if (startsWith(line, "+  ")) {
      block <- c(block, substring(line, 4L))
    } else {
      stop(sprintf("Line %i of main.tex is in a CodeInput but has no prompt: %s", i, line))
    }
  }
}

code_lines <- c(
  "#' ```{r setup, include = FALSE}",
  "#' knitr::opts_chunk$set(cache = FALSE)",
  "#' ```",
  "# Replication script for the code shown in main.tex, extracted by extract.R.",
  "# Render it with knitr::spin(\"paper_code.R\"): the call in the chunk with error = TRUE",
  "# fails on purpose, so a plain Rscript run stops there.",
  "Sys.time()",
  "options(width = 80)",
  "",
  code_lines,
  "Sys.time()",
  "sessionInfo()"
)

writeLines(code_lines, "paper_code.R")
cat(sprintf("Wrote %i lines to paper_code.R\n", length(code_lines)))
