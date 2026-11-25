# JSS Guidelines

Everything the Journal of Statistical Software asks of authors, from three sources:

1. [Style Guide](#style-guide): the [Style Guide page](https://www.jstatsoft.org/style) of the website, verbatim.
2. [Author Requirements](#author-requirements): the other author-facing pages of the website, summarized.
3. [Style Manual](#style-manual): the manual of the document class, `jss-template/jss.pdf`, verbatim.

The three overlap. For example, the style manual repeats three FAQ items: the style checklist and the two hints on compile errors.

## Style Guide

In addition to the general instructions for manuscripts, we require all submissions to be formatted using LaTeX and the JSS style files (linked below along with the templates). It is recommended to keep the LaTeX code as simple as possible, i.e., avoid inclusion of packages/commands that are not necessary. We ask you to pay special attention to the [frequently asked questions (FAQ)](#frequently-asked-questions).

To facilitate authoring manuscripts in JSS style, we provide several templates on this page. Specifically, a short demo article is available using either plain LaTeX or in combined R/LaTeX. For outlining the typical structure of a JSS article some brief text snippets are employed that have been inspired by [doi:10.18637/jss.v027.i08](https://doi.org/10.18637/jss.v027.i08) (Zeileis et al. 2008), discussing count data regression in R.

For authors of R software we recommend to write an R/LaTeX document (`.Rnw`) using Sweave or knitr. From this the LaTeX document (`.tex`) can be obtained by weaving/knitting along with the R replication code (`.R`) by tangling.

### LaTeX

When using plain LaTeX, you may start from the following article template: `jss-article-tex.zip` (including all necessary style files). The source file `article.tex` can be edited in any text editor or dedicated LaTeX editor (including RStudio). To produce the manuscript pdfLaTeX should be used for which dedicated editors usually have a "Compile PDF" button or something similar. On the command line `texi2pdf` is convenient:

```sh
texi2pdf article.tex
```

The corresponding references are in the BibTeX bibliography `ref.bib`. The replication code `article.R` has to be prepared separately and should ideally include comments that make it easy to match it to the manuscript.

#### Code Snippets

Code snippet contributions are produced in the same way as articles. You only have to switch the from `\documentclass[article]{jss}` to `\documentclass[codesnippet]{jss}` in the first line of the LaTeX document.

#### Software and Book Reviews

For software and book reviews, LaTeX templates are provided in `jss-review-tex.zip` (including all necessary style files). The procedure to generate PDF output does not change from the above LaTeX section.

### R/LaTeX via Sweave or knitr

For producing a JSS article with R/LaTeX, please start from the following template: `jss-article-rnw.zip` (including all necessary style files). The source files for the JSS contribution are the `article.Rnw` manuscript and the `ref.bib` bibliography in BibTeX format. To produce the PDF manuscript the article must first be weaved:

```r
Sweave("article.Rnw")
```

yielding `article.tex` along with the graphics file (here `article-visualization.pdf`). This LaTeX file can be compiled in the "usual" way to the PDF using pdfLaTeX, i.e., using the shell, some LaTeX editor (see also above), or simply with R:

```r
library("tools")
texi2pdf("article.tex")
```

An alternative route for RStudio users is to use the "Compile PDF" button directly for the `article.Rnw`. However, make sure not to use concordance mode: *Tools > Global Options > Sweave*, then uncheck *Always enable Rnw concordance*.

The replication code can be obtained by tangling:

```r
Stangle("article.Rnw")
```

yielding `article.R`. However, editing the comments of this `.R` file typically makes it more accessible to the readers/reviewers.

If knitr is used to prepare the `article.Rnw` file, the `render_sweave()` hook should be used to ensure JSS style code formatting. The weaving and tangling can then by done by `knit()` and `purl()`, respectively.

### Frequently Asked Questions

- [What are the most important style guidelines in JSS?](#what-are-the-most-important-style-guidelines-in-jss)
- [What are the capitalization rules in JSS? What is title style and sentence style?](#what-are-the-capitalization-rules-in-jss-what-is-title-style-and-sentence-style)
- [How to cite software?](#how-to-cite-software)
- [How to cite R packages?](#how-to-cite-r-packages)
- [What are the different `\cite`, `\citet`, `\citep` commands about?](#what-are-the-different-cite-citet-citep-commands-about)
- [How should abbrevations be formatted?](#how-should-abbrevations-be-formatted)
- [How to format figure/table captions?](#how-to-format-figuretable-captions)
- [How should code be formatted in the manuscript?](#how-should-code-be-formatted-in-the-manuscript)
- [What is a good pen-to-paper ratio for my graphics?](#what-is-a-good-pen-to-paper-ratio-for-my-graphics)
- [Some of my graphics files are very large, what should I do?](#some-of-my-graphics-files-are-very-large-what-should-i-do)
- [How should my R package reflect that a manuscript about it was published in JSS?](#how-should-my-r-package-reflect-that-a-manuscript-about-it-was-published-in-jss)
- [How can I turn my JSS paper into an R package vignette?](#how-can-i-turn-my-jss-paper-into-an-r-package-vignette)
- [Which naming conventions are used for software, journal, and publisher names in JSS?](#which-naming-conventions-are-used-for-software-journal-and-publisher-names-in-jss)
- [My LaTeX paper does not compile when there is JSS markup in section titles, what should I do?](#my-latex-paper-does-not-compile-when-there-is-jss-markup-in-section-titles-what-should-i-do)
- [Compiling my LaTeX paper fails with an error at `\begin{document}`, what went wrong?](#compiling-my-latex-paper-fails-with-an-error-at-begindocument-what-went-wrong)
- [Miscellaneous](#miscellaneous)

#### What are the most important style guidelines in JSS?

The items below provide a quick style checklist for the most important style guidelines in JSS. More details can be found in the JSS style manual (`jss.pdf`) and in the other FAQ items.

- The manuscript can be compiled by pdfLaTeX.
- `\proglang`, `\pkg` and `\code` have been used for highlighting throughout the paper (including titles and references), except where explicitly escaped.
- References are provided in a `.bib` BibTeX database and included in the text by `\cite`, `\citep`, `\citet`, etc.
- Titles and headers are formatted as described in the JSS manual:
  - `\title` in title style,
  - all titles in the BibTeX file in title style,
  - `\section`, `\subsection`, etc. in sentence style,
  - annotations of figures/tables (including captions) in sentence style.
- Figures, tables and equations are marked with a `\label` and referred to by `\ref`, e.g., `Figure~\ref{...}`.
- Software packages are `\cite{}`d properly.

#### What are the capitalization rules in JSS? What is title style and sentence style?

In English there are basically two styles of capitalization in titles, typically referred to as "sentence style" (or "sentence case") and "title style" (or "title case"). Although there are few strict rules, the subsequent set of guidelines should be helpful.

**Sentence style:** Only the first word in a title is capitalized, as is the first word after a colon or a hyphen. Of course, proper names remain in upper case. A simple example would be

> A fancy topic: Implementation in Java

**Title style:** All principal words should be capitalized. This includes the first and last words of a headline, and all nouns, pronouns, adjectives, verbs, adverbs, and subordinating conjunctions (if, because, as, that, etc.). Do capitalize the first word after a colon. Articles (a, an, the), coordinating conjunctions (and, but, or, nor, for), and prepositions of any length, are to remain lowercased. However, if any of these are the first or last word of the headline, they should be capitalized. If you have an abbreviation in your headline that is normally lowercase, it should be left lowercase, particularly abbreviations for units of measure. Two part words separated by a hyphen should have both words capitalized. Examples:

- Come Join Us for a Celebration
- Caring for Your Houseplants
- We're Getting Ready for an Early Spring
- The Forecast for Summer: Hot!
- As the Wind Blows
- Spraying Schedule Posted on Office Memo Board
- Remember to Observe All Parking Rules
- What Are They Fighting For?
- A New Record: 37-in. Snow Fall Accumulation
- Tick-Tock: It's Daylight Savings Time Again!

#### How to cite software?

Many software packages tell their users how they want to be cited, i.e., there may be a pointer on the webpage or in the manual to a suitable publication (book, journal article, technical reports, etc.). If there is no recommended citation, please cite the corresponding manual or webpage. An example is given below for SAS/STAT 9.1.

```bibtex
@Manual{SAS-STAT,
  author  = {{\proglang{SAS} Institute Inc.}},
  title   = {\proglang{SAS/STAT} Software, Version~9.1},
  year    = {2003},
  address = {Cary, NC},
  url     = {https://www.sas.com/}
}
```

#### How to cite R packages?

Please check if there is an official citation for the package. If so, this can be seen on the associated CRAN webpage, i.e., `https://CRAN.R-project.org/package=foo` or queried in R (if the package is installed) via `citation("foo")`. If there is no citation, please use a CRAN style reference as exemplified for the rJava package below.

In any case, please make sure that the BibTeX is valid, that the title is in title style and that `\proglang`/`\pkg`/`\code` are used where appropriate.

```bibtex
@Manual{rJava,
  title  = {\pkg{rJava}: Low-Level \proglang{R} to \proglang{Java} Interface},
  author = {Simon Urbanek},
  year   = {2009},
  note   = {\proglang{R}~package version~0.8-1},
  url    = {https://CRAN.R-project.org/package=rJava},
}
```

#### What are the different `\cite`, `\citet`, `\citep` commands about?

BibTeX can process citations differently and provides different commands for that. In particular, these are necessary if comments such as page references etc. should be added in brackets.

`\cite{...}` and `\citet{...}` yield the usual Author (Year). `\citep{...}` yields (Author Year).

These can be modified to incorporate further text, e.g., `\citet[p.~123]{...}` yields Author (Year, p. 123). `\citep[see][for further details]{...}` yields (see Author Year for further details).

`\citealp` might be useful for further flexibility, see standard LaTeX/BibTeX manuals for details.

Never use brackets-within-brackets constructs like `(\cite{...})`.

#### How should abbrevations be formatted?

Abbreviations should be spelled in upper-case letters without additional formatting (i.e., without periods, without small caps, italics, etc.). All abbreviations should be introduced with their expansion where the expansion should not be capitalized. (Exceptions are, of course, when the expansion contains proper names or the first word is the first word of a sentence.) Examples would include:

- support vector machine (SVM)
- MCMC (Markov chain Monte Carlo)

#### How to format figure/table captions?

All captions should appear below the corresponding figure/table. The captions should be in sentence style and end with a period. No additional formatting (such as `\emph`, `\bf` or `\it`) should be used for the caption.

All table row/column headers should also be in sentence style. There should not be further footnote-style annotations in tables; these should all be placed in the caption.

#### How should code be formatted in the manuscript?

Code should preferably be presented in the usual text flow. It should have enough spaces to facilitate reading. Please include spaces before and after operators and after commas (unless spaces have syntactical meaning). An example would be to use

```r
y = a + b * x
```

and not

```r
y=a+b*x
```

The spacing should be used both for code that appears inline in the text and for code in verbatim code chunks.

The code presented in the manuscript should not contain comments within the verbatim code. Instead the comments should be made in the normal LaTeX text.

`\code{...}` can be used for code chunks that should appear inline. Various options are available for verbatim code chunks:

- The `{Code}` environment.
- The `{CodeInput}`/`{CodeOutput}` environments.
- For R packages the usual Sweave environments can be used. See Section 2.4 in the style manual (`jss.pdf`) for more details and examples.

In all cases, code input/output must fit within the normal textwidth of the manuscript. Thus, code input should have appropriate line breaks and code output should preferably be generated with a suitable width (or otherwise edited).

#### What is a good pen-to-paper ratio for my graphics?

JSS papers should be created in such a way that they are not only easy to read when printed on paper but also easy to read on screen. Hence, graphics should have a pen-to-paper ratio that makes them easy to read in both media. In particular, this means that annotations should not be too large or too small. As a rough guidance, graphics annotation should be a about the size of the figure caption or a little bit smaller.

A simple way of emulating increased/decreased font size in vector graphics (such as `.pdf`) is to just plot on a smaller/larger plotting device. When font size is kept fixed and the size of the figure as included in LaTeX is kept fixed, this will render the fonts relatively larger/smaller. (Note that a larger device will lead to relatively smaller annotation.)

#### Some of my graphics files are very large, what should I do?

This typically occurs when vector graphics (typically `.pdf`) are used for graphics with a very large amount of points or lines, e.g., in maps, heatmaps, or scatterplots with a very large amount of points. If a vector graphic is too large (i.e., larger than a few hundred kilobytes and takes a moment to load in PDF viewers), a raster graphic version of it should also be provided (`.jpg` or `.png`). It should be assured that all annotation in the raster graphic is still sufficiently easy to read (and not "pixelated").

In R, this can typically be achieved by using the `png()` device instead of the `pdf()` device. Suppose the original graphic was created by

```r
pdf(..., height = H, width = W)
```

Then a useful starting point for a `png()` version is

```r
png(..., height = H, width = W, units = "in", res = 144)
```

although higher resolutions may be needed for some graphics.

#### How should my R package reflect that a manuscript about it was published in JSS?

You should enhance the R package in three ways:

1. Include a reference for the JSS paper in the `\references{}` section of the relevant `.Rd` pages.
2. Include a `CITATION` file in `inst/CITATION` within the package. A suitable file is provided by the JSS editors when the final manuscript is ready for publication.
3. Turn the JSS manuscript into a package vignette. For details see the [next FAQ item](#how-can-i-turn-my-jss-paper-into-an-r-package-vignette).

#### How can I turn my JSS paper into an R package vignette?

For authors that publish their R packages in JSS we recommend:

- Use the JSS paper as the basis for a package vignette. If you have not done so already, use Sweave-style code chunks for all input/output and graphics. See the Sweave manual for more details.
- Turn off the JSS header and footer by using the `nojss` option: `\documentclass[nojss]{jss}`.
- Cite the JSS paper in the vignette (in the abstract or intro).

By doing so, the vignette is also visibly distinct from the JSS paper and you can easily extend/improve/correct the vignette in the future, keeping it up to date. For an example, see

```r
vignette("sandwich", package = "sandwich")
```

Note that you do not need to include the JSS LaTeX style files in your package. They are shipped with R and installed on all relevant R servers (CRAN, R-Forge, …).

The preferred layout can be obtained by setting

```r
options(prompt = "R> ", continue = "+  ", width = 70, useFancyQuotes = FALSE)
```

invisibly in a code chunk at the beginning. (Note, however, that even though `width` was set the input/output might still be modified because `width` is not respected by all R functions.)

#### Which naming conventions are used for software, journal, and publisher names in JSS?

**Software:**

- Fortran (not: FORTRAN)
- Java (not: JAVA, java)
- MATLAB (not: Matlab, matlab)
- S-PLUS (not: Splus, S-Plus)

**Journals:**

- The American Statistician (not: American Statistician)
- The Annals of Statistics (not: Annals of Statistics)
- Journal of the Royal Statistical Society B (not: Journal of the Royal Statistical Society, Series B)

**Publishers:**

- Springer-Verlag (not: Springer)
- John Wiley & Sons (not: Wiley, John Wiley & Sons Inc.)

#### My LaTeX paper does not compile when there is JSS markup in section titles, what should I do?

If you want to use markup in section headers you will usually have to escape it for the PDF bookmarks by giving the text for the bookmark explicitly without markup, e.g.,

```latex
\section[Calling C++ from R]{Calling \proglang{C++} from \proglang{R}}
```

#### Compiling my LaTeX paper fails with an error at `\begin{document}`, what went wrong?

The reason is almost surely that some of the declarations in the header have not been made properly. For example, `\Plainauthor`, `\Plaintitle` or `\Plainkeywords` might be missing or still containing markup.

#### Miscellaneous

Below is a list of miscellaneous style conventions used in JSS:

- For referring to subsections, do not use Subsection x.y, just Section x.y.
- When using e.g. and i.e. add a comma after the period to keep LaTeX from interpreting them as the end of a sentence, i.e.: `e.g.,` and `i.e.,`.
- House style is to use `$p$~value`, `$t$~statistic`, etc. (without hyphen).
- `\top` should be used as the transpose symbol, e.g., `X^\top`.
- For R-related manuscripts: The first argument of `data()` and `library()` should always be quoted, e.g., `library("foo")`.
- For books with an edition, this should be indicated as 2nd, 3rd, etc.
- To refer to equations, one can use either `Equation~\ref{...}` (with capitalization) or `(\ref{...})` with the former being preferred if the number of equation references is not too large.

## Author Requirements

Summarized (not verbatim) from the other author-facing pages of the JSS website, as of 2026-11-13:

- [Information for Authors](https://www.jstatsoft.org/authors) (the most detailed page)
- [Submission Guide](https://www.jstatsoft.org/guides/submission)
- [Information on Mission](https://www.jstatsoft.org/mission)
- [Submissions](https://www.jstatsoft.org/about/submissions) (checklist and copyright)

The more detailed manual the Submission Guide mentions is `jss.pdf`, reproduced in the [Style Manual](#style-manual) part below.

### Scope

- JSS publishes open-source software together with a reproducible article. The software and the article are reviewed together.
- JSS does **not** publish methodological innovations in computational statistics. New models or estimators should be published elsewhere first.
- Software without an article that documents and illustrates it cannot be published, and neither can supplements to methodological papers.
- The article must explain the scope of the software and its underlying models so that "statistically literate readers can understand what the software does". Design principles and implementation are "at the heart of" the article.
- "The general analysis workflow must be illustrated by an enlightening non-trivial case study."
- Narrower software can be submitted as a code snippet.

### Comparison with existing software

- The article must cover implementations of similar scope, including those in other languages and environments. It should discuss the advantages and disadvantages of the new software compared with them, using empirical illustrations where possible.
- The Mission page asks for "a careful comparison with other open-source implementations of similar models or procedures".
- "Extensive simulation studies are discouraged."

### Manuscript

- English, LaTeX only, uploaded as PDF. Manuscripts that do not meet the formal requirements are returned immediately. JSS does not help convert from other typesetting systems.
- There is no page limit and no limit on figures or tables, but manuscripts over 30 pages "typically require much longer review times".
- A readable manuscript shortens the review.

### Software

- Source code in ASCII files, readable and commented.
- For package systems (R, Python, Stata, ...): formatted help files, and packaged for easy installation.
- "If at all possible", the software should be on a standard repository such as CRAN, Bioconductor or PyPI, "and not just GitHub, GitLab, etc".
- For R packages that return compound objects, the minimum expectation is to use R's class and method systems, e.g., S3 classes with standard methods such as `print()`, `plot()` and `summary()`.
- The JSS paper should be included as a package vignette.
- Licence: GPL-2, GPL-3, or a GPL-compatible licence. "JSS does NOT consider software distributed under other licenses."
- Code and articles cannot be updated after publication, since JSS is not a software repository.
- Following the [rOpenSci Statistical Software Peer Review guide](https://stats-devguide.ropensci.org/) is suggested but not required.

### Replication materials

- Required for all results in the manuscript, preferably as "a single, commented standalone replication script".
- All figures, tables and output must be "fully and exactly reproducible on at least one platform". Any platform dependence must be stated at submission. Ideally the results are reproducible across platforms. If they are not, the manuscript should discuss why.
- Set the random seed wherever results depend on simulation.
- Runtime: anybody with sufficient technical background should be able to install the software and run the replication material "within reasonable time (one hour on a regular PC)".
- If the full replication takes much longer or needs special hardware, provide a second script that reproduces similar results within reasonable time on a regular PC, in addition to the full script.
- Authors are strongly encouraged to provide an output file to compare against. For R, this is a `code.html` produced by `knitr::spin("code.R")`, with `sessionInfo()` called at the end of the script.

### Submission

- Required attachments (or links to them): the PDF manuscript in JSS style, the source code of the software, and the replication materials.
- Uploads are limited to 50 MB. For larger supplements, include a file with download instructions and an external link, and explain this in the submission. Contact editor@jstatsoft.org if you cannot host them.
- Checklist ticked at submission: privacy policy accepted, PDF in JSS style, source code, replication materials, and the licence clearly indicated. Submissions that do not follow the checklist may be returned.
- In the submission system: register and switch your role to "Author". Upload the manuscript as "PDF Manuscript" and everything else (software, replication script, data) as "Other". Title, abstract and the list of contributors are required metadata. Co-authors do not need an account.
- Revisions: give the files disambiguating names that include version numbers, upload them in the Revision section, and add a discussion comment saying the revision is ready.

### Review and publication

- The editor-in-chief assigns a section editor, who selects two reviewers. One of the reviewers can be the section editor.
- Reviewers check that the software works as described, is clearly documented and serves a useful purpose. They also verify reproducibility, and only recommend conditional acceptance once all reproducibility problems are resolved.
- Possible decisions: decline (editorial reject), reject, resubmit for review, revision required (goes back to the editor only), and (conditional) accept.
- Every revision needs a point-by-point reply to all comments from the reviewers and the editor.
- Rejected manuscripts cannot be resubmitted.
- A revision not returned within six months counts as withdrawn, and so does a conditional acceptance not finalized within a year.
- Acceptance is conditional on meeting the style and reproducibility requirements. A manuscript that does not meet them cannot be published.
- Articles are published under CC BY 3.0, and the authors keep the copyright.

## Style Manual

From `jss-template/jss.pdf`: "jss: A Document Class for Publications in the Journal of Statistical Software" by Achim Zeileis, version of April 28, 2026. Sections 1–3 are reproduced verbatim, except for a few obvious typos that are fixed and section references that are turned into links. Section 4 only lists the source code of `jss.cls`, so the last subsection, [What the class sets up](#what-the-class-sets-up), summarizes what that code does instead.

### Introduction

The LaTeX2ε document class `jss` is an extension of the standard LaTeX2ε `article` class for publications in the Journal of Statistical Software (JSS, https://www.jstatsoft.org/). Additionally, the JSS-specific header/footer can be easily switched off so that the document class can easily be used for other publications, e.g., R package vignettes.

The document class provides infrastructure for all four kinds of publications in JSS: regular articles, code snippets, book reviews and software reviews. Each document requires several declarations to be made in the header (before `\begin{document}`) which are described in [Instructions for authors](#instructions-for-authors) separately for articles/code snippets and book/software reviews along with some general commands which can be used in all documents.

The final version of JSS papers should be prepared using this JSS style file; the submission of the final version needs to include the full sources (`.tex`, `.bib`, and all graphics). A quick check for the most important aspects of the JSS style is given in the [Style checklist](#style-checklist); authors should make sure that all of them are addressed in the final version. A list of frequently asked questions (FAQ) is available online https://www.jstatsoft.org/style that provides additional details and tries to address typical problems (the [Style Guide](#style-guide) part of this file).

All documents need to be processed by pdfLaTeX, some useful information on this is provided in [Using pdfLaTeX and BibTeX](#using-pdflatex-and-bibtex), which also contains some information on using BibTeX. BibTeX together with the style file `jss.bst` produces references and citations in the required format.

The actual code for the batch file (`jss.ins`), the driver (`jss.drv`) and the class (`jss.cls`) are briefly described in Section 4. Note, that usually you do not have to read that section when you want to prepare a submission for JSS.

### Instructions for authors

To use the JSS styles, you have to include the class file `jss.cls`, the logo `jsslogo.jpg` and the BibTeX style `jss.bst` in your search path. This can either be your local working directory or in your texmf or localtexmf tree.

The LaTeX documents have to include the `jss.cls` first by

```latex
\documentclass[type]{jss}
```

where `type` can be `article` (which is the default), `codesnippet`, `bookreview` or `softwarereview`. Templates with brief instructions are provided in `article.tex`, `codesnippet.tex`, `bookreview.tex` and `softwarereview.tex` respectively. The corresponding commands used for the header declarations are described in more detail in the following.

By using `jss.cls`, the packages graphicx, xcolor, hyperref, fancyvrb and natbib are loaded automatically. Additionally, lmodern and upquote are loaded if available. Authors may, of course, include further packages if necessary (e.g., amsmath, amsfonts, bm, tikz, ...) but should not do so excessively. In particular, no packages must be loaded that change the page layout, the font, or font encoding. If the package thumbpdf is available, its inclusion is encouraged.

The titles of JSS publications are capitalized, i.e., in title style, but the section headers are not and should be in sentence-style.

Acknowledgments should be included at the end of the paper before the references in a separate section set up via `\section*{Acknowledgments}`.

*Hint.* If you want to use markup in section headers you will usually have to escape it for the PDF bookmarks by giving the text for the bookmark explicitly without markup, e.g.,

```latex
\section[Calling C++ from R]{Calling \proglang{C++} from \proglang{R}}
```

*Hint.* If compilation with pdfLaTeX fails with an error at `\begin{document}` the reason is almost surely that some of the declarations in the header have not been made properly. For example, `\Plainauthor`, `\Plaintitle` or `\Plainkeywords` might be missing or still containing markup.

*Hint.* If you want to use the JSS style for a non-JSS paper (or a modification of a JSS paper, e.g., in a vignette), you can set the option `nojss` in the `\documentclass` statement to suppress JSS-specific layout.

#### Style checklist

A quick check for the most important aspects of the JSS style is given below. Authors should make sure that all of them are addressed in the final version. More details can be found in the remainder of this manual.

- The manuscript can be compiled by pdfLaTeX.
- `\proglang`, `\pkg` and `\code` have been used for highlighting throughout the paper (including titles and references), except where explicitly escaped.
- References are provided in a `.bib` BibTeX database and included in the text by `\cite`, `\citep`, `\citet`, etc.
- Titles and headers are formatted properly:
  - `\title` in title style,
  - `\section` etc. in sentence style,
  - all titles in the BibTeX file in title style.
- Figures, tables and equations are marked with a `\label` and referred to by `\ref`, e.g., `Figure~\ref{...}`.
- Software packages are `\cite{}`d properly.

#### Articles and code snippets

For JSS articles and code snippets respectively, the following declarations have to be made in the header of the LaTeX sources (before `\begin{document}`). See also the template `article.tex` or `codesnippet.tex` respectively.

**`\author`**: The command `\author` specifies the list of authors. The name of each author should be followed by a linebreak and his affiliation (only the university, in a single line). The authors should be separated by `\And` (instead of `\and`), e.g.,

```latex
\author{Achim Zeileis\\Universit\"at Innsbruck \And
        Second Author\\Plus Affiliation}
```

If not all authors fit into a single line, `\AND` (instead of `\And`) should be used in front of authors that should go into the next line. If authors have an ORCID it is recommended to include `\usepackage{orcidlink}` in the header and extend the author names in the `\author` command in the following format: `Achim Zeileis~\orcidlink{0000-0003-0918-3766}`.

**`\Plainauthor`**: The list of authors without affiliations. It needs to be comma-separated and must not contain any markup (bold fonts etc.), e.g.,

```latex
\Plainauthor{Achim Zeileis, Second Author}
```

**`\title`**: The title of the paper. It should be capitalized and may contain further markup (in particular markup such as `\pkg` and `\proglang`), e.g.,

```latex
\title{A Capitalized Title for a Package \pkg{foo}}
```

**`\Plaintitle`**: The full title without any markup. The default is to use `\title`, therefore it needs to be specified only if it is different from `\title`, e.g.,

```latex
\Plaintitle{A Capitalized Title for a Package foo}
```

**`\Shorttitle`**: A shorter version of the title to be used for page headings. The default is to use `\title`, therefore it needs to be specified only if it is different from `\title`, e.g.,

```latex
\Shorttitle{foo: A Capitalized Title}
```

**`\Abstract`**: Enter the abstract for your article here, e.g.,

```latex
\Abstract{
  The abstract of the article.
}
```

**`\Keywords`**: A comma-separated list of (at least one) keyword(s) which should not be capitalized, e.g., `\Keywords{keywords, comma-separated, not capitalized}`.

**`\Plainkeywords`**: The list of keywords without any markup. The default is to use `\Keywords`, therefore it needs to be specified only if it is different from `\Keywords`.

For each of the following six commands the manual adds: *Note: This information will be provided upon acceptance or added by the technical editor. Prior to acceptance, do not use this command.*

| Command | Description | Example |
|---|---|---|
| `\Volume` | The JSS volume number in which the article is published | `\Volume{11}` |
| `\Issue` | The JSS issue number in which the article is published | `\Issue{9}` |
| `\Month` | The month in which the article is published | `\Month{September}` |
| `\Year` | The year in which the article is published | `\Year{2004}` |
| `\Submitdate` | The date of submission for the article | `\Submitdate{2004-09-29}` |
| `\Acceptdate` | The date of acceptance for the article | `\Acceptdate{2004-09-29}` |

**`\Address`**: The address of (at least) one author should be given in the following format

```latex
\Address{
  Achim Zeileis\\
  Universit\"at Innsbruck\\
  Department of Statistics and Mathematics\\
  Faculty of Economics and Statistics\\
  6020 Innsbruck, Austria\\
  E-mail: \email{Achim.Zeileis@uibk.ac.at}\\
  URL: \url{https://www.zeileis.org/}
}
```

It is also possible to include your telephone number, by adding it in the format

```latex
  Telephone: +43/512/507-7103
```

before the e-mail address.

Furthermore, if the document is prepared using the Sweave functions in R, something like the following line

```latex
%% need no \usepackage{Sweave.sty}
```

(with `%%`) needs to be included in the header.

#### Book and software reviews

For JSS book and software respectively, the following declarations have to be made in the header of the LaTeX sources (before `\begin{document}`). See also the template `bookreview.tex` or `softwarereview.tex` respectively. Note that some commands might differ between book and software reviews, this is always stated explicitly below.

**`\Reviewer`**: The command `\Reviewer` specifies the name of the reviewer followed by a linebreak and his affiliation (only the university, in a single line), e.g.,

```latex
\Reviewer{Achim Zeileis\\Universit\"at Innsbruck}
```

**`\Plainreviewer`**: The name of the reviewer without affiliation. It must not contain any markup (bold fonts etc.), e.g.,

```latex
\Plainreviewer{Achim Zeileis}
```

The following five commands are just required for book reviews.

**`\Booktitle`**: The title of the book. It should be capitalized and may contain further markup (in particular markup such as `\pkg` and `\proglang`), e.g.,

```latex
\Booktitle{Visualizing Categorical Data}
```

**`\Bookauthor`**: Author(s) of the book, e.g.,

```latex
\Bookauthor{Michael Friendly}
```

If there are several authors they should be comma-separated, and the last author separated by and, e.g., `\Bookauthor{A and B}` or `\Bookauthor{A, B and C}`.

**`\Pubyear`**: Year of publication, e.g., `\Pubyear{2000}`.

**`\ISBN`**: ISBN number, e.g., `\ISBN{1-58025-660-0}`.

**`\Pages`**: Number of pages, both arabic and roman (if available), e.g., `\Pages{456}` or `\Pages{xvi + 145}`.

The following command is just required for software reviews.

**`\Softwaretitle`**: The title of the software. It should be capitalized and may contain further markup (in particular markup such as `\pkg` and `\proglang`), e.g.,

```latex
\Softwaretitle{\pkg{Aabel} 1.5.7}
```

The remaining commands are again required for both book and software reviews.

**`\Publisher`**: Publisher of the book/software, e.g., `\Publisher{SAS Institute Inc.}` or `\Publisher{Gigawiz Ltd. Co.}`.

**`\Pubaddress`**: Address of the publisher of the book/software, e.g., `\Pubaddress{Carey, NC}`.

**`\Price`**: Price of the book/software. For books this might simply be `\Price{USD 69.95}` or `\Price{USD 69.95 (P)}`, but could also distinguish between hardcover and paperback versions `\Price{USD 69.95 (P), USD 89.95 (H)}`. Analogously, for a software it could be `\Price{USD 349 (standard), USD 249 (academic)}`.

**`\URL`**: A URL for the book or software, e.g.,

```latex
\URL{http://www.math.yorku.ca/SCS/vcd/}
```

If no URL is available, use `\URL{}`.

**`\Plaintitle`**: The full book or software title without any markup (line breaks, bold fonts etc.). The default is to use `\Booktitle` or `\Softwaretitle` respectively, therefore it needs to be specified only if it is different from `\Booktitle` or `\Softwaretitle`, e.g.,

```latex
\Plaintitle{Visualizing Categorical Data}
```

**`\Shorttitle`**: A shorter version of the book or software title to be used for page headings. The default is to use `\Booktitle` or `\Softwaretitle` respectively, therefore it needs to be specified only if it is different from `\Booktitle` or `\Softwaretitle`, e.g.,

```latex
\Shorttitle{Visualizing Categorical Data}
```

For each of the following five commands the manual adds: *Note: This information will be provided upon acceptance or added by the technical editor.*

| Command | Description | Example |
|---|---|---|
| `\Volume` | The JSS volume number in which the review is published | `\Volume{11}` |
| `\Issue` | The JSS issue number in which the review is published | `\Issue{9}` |
| `\Month` | The month in which the review is published | `\Month{September}` |
| `\Year` | The year in which the review is published | `\Year{2004}` |
| `\Submitdate` | The date of publication for the review | `\Submitdate{2004-09-29}` |

**`\Address`**: The address of (at least) one author should be given in the same format as for [articles and code snippets](#articles-and-code-snippets), again optionally with a telephone number before the e-mail address.

#### Further commands

The jss package provides several commands for typesetting names related to software (programming languages, packages, code) and mathematical formulae.

##### Writing about software

**`\proglang`**: This should be used for typesetting the names of programming languages, e.g., `\proglang{Java}`, `\proglang{C++}` or `\proglang{R}`. This applies also to programmable environments which also have a GUI like `\proglang{SAS}`, `\proglang{Stata}` or `\proglang{S-PLUS}`.

**`\pkg`**: This should be used for typesetting the names of packages, e.g., `\pkg{CMregr}`, `\pkg{MATCH}` or `\pkg{strucchange}`.

**`\code`**: This should be used for typesetting code chunks within the text, e.g., `\code{plot(1:10)}`. Currently, this simply uses a typewriter font. Although it escapes most special characters, it might still lead to problems with some special characters. In such cases the code can also be set using `\verb`, e.g., `\verb/print("hello world")/`.

##### Layout of code

`jss.cls` only provides very simple means of including code which are mostly borrowed from Sweave. There are three verbatim environments for code: `Code`, `CodeInput` and `CodeOutput`. Furthermore, there is an environment `CodeChunk` which can be put around sequences of `CodeInput`s and `CodeOutput`s to (hopefully) keep LaTeX from page-breaking in the middle of a code chunk. In short, there are two options: a) if no distinction between input and output is necessary, the code is placed between `\begin{Code}` and `\end{Code}`. b) If input and output should be distinguished, this can be done like in the following example.

```latex
\begin{CodeChunk}
\begin{CodeInput}
first input first line
first input second line
\end{CodeInput}
\begin{CodeOutput}
output of first input
\end{CodeOutput}
\begin{CodeInput}
second input
\end{CodeInput}
\begin{CodeOutput}
second output
\end{CodeOutput}
\end{CodeChunk}
```

An example what this could look like, is the following R code. The first three lines are the input, the rest is output.

```latex
\begin{CodeChunk}
\begin{CodeInput}
R> data(cars)
R> fm <- lm(dist ~ speed, data = log(cars))
R> summary(fm)
\end{CodeInput}
\begin{CodeOutput}
Call:
lm(formula = dist ~ speed, data = log(cars))

Residuals:
     Min       1Q   Median       3Q      Max
-1.00215 -0.24578 -0.02898  0.20717  0.88289

Coefficients:
            Estimate Std. Error t value Pr(>|t|)
(Intercept)  -0.7297     0.3758  -1.941   0.0581 .
speed         1.6024     0.1395  11.484 2.26e-15 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 0.4053 on 48 degrees of freedom
Multiple R-Squared: 0.7331,     Adjusted R-squared: 0.7276
F-statistic: 131.9 on 1 and 48 DF,  p-value: 2.259e-15
\end{CodeOutput}
\end{CodeChunk}
```

If you prepare your paper using Sweave (which is recommended if you describe an R package) do *not* include `Sweave.sty` into your document, the necessary commands are already available within `jss.cls`. To prevent Sweave from including `Sweave.sty` automatically you need to include a line like

```latex
%% need no \usepackage{Sweave.sty}
```

(with `%%`) into the header of your document.

If this basic infrastructure for typesetting your code is not sufficient, you can also use other LaTeX packages like the listings package.

##### Mathematical formulae

Commonly used operators like E, VAR, COV, and P should be set using the commands `\E`, `\VAR`, `\COV` and `\Prob`. Beyond this, jss does not provide (or enforce) a certain mathematical notation. However, using the AMS packages (amsmath, amssymb, etc.) could be useful.

### Using pdfLaTeX and BibTeX

#### Using pdfLaTeX

A LaTeX document (`foo.tex`, say) using `jss.cls` needs to be compiled using pdfLaTeX, typically this will be done using either of the following commands:

```sh
pdflatex foo.tex
texi2dvi --pdf foo.tex
texi2pdf foo.tex
```

If you are not using command line tools but some integrated GUI editor for LaTeX documents you will have to press the 'pdfLaTeX' button (as opposed to the 'LaTeX' button).

All graphics included into the document have to be in a format pdfLaTeX can deal with, i.e., PDF for vector graphics or JPG/PNG/etc. for bitmaps/raster graphics. If you cannot produce PDF graphics directly but only PS/EPS, these can be converted using `ps2pdf` or `epstopdf` (usually preferred).

*Hint.* If you are used to compiling your documents with standard LaTeX and then getting automatic reloads of the resulting DVI document in your DVI viewer, which is not possible with PDF documents in many PDF viewers: you might want to look at xpdf (Linux) or gsview (Windows, see http://www.cs.wisc.edu/~ghost/gsview/) which have a reload function.

*Hint.* (The manual repeats here the hint on markup in section headers from [Instructions for authors](#instructions-for-authors).)

*Hint.* If you know how to produce LaTeX documents that can be processed with both LaTeX and pdfLaTeX, you can do so if you provide an EPS substitute for `jsslogo.jpg` (e.g. an empty or converted `jsslogo.eps`). Note, however, that the final document needs to be processed with pdfLaTeX. Neither this manual nor the JSS encourage or support compilation of JSS documents with standard LaTeX.

#### References with BibTeX

The format for references (e.g., articles, books, software, proceedings) should look like this

> Brown RL, Durbin J, Evans JM (1975). "Techniques for Testing the Constancy of Regression Relationships over Time." *Journal of the Royal Statistical Society B*, 37, 149–163.
>
> Friendly M (2000). *Visualizing Categorical Data*. SAS Institute, Carey, NC.
>
> R Core Team (2020). *R: A Language and Environment for Statistical Computing*. R Foundation for Statistical Computing, Vienna, Austria. URL https://www.R-project.org/.
>
> Plummer M (2003). "JAGS – A Program for Analysis of Bayesian Graphical Models Using Gibbs Sampling." In K Hornik, F Leisch, A Zeileis (eds.), "Proceedings of the 3rd International Workshop on Distributed Statistical Computing, Vienna, Austria," ISSN 1609-395X, URL https://www.R-project.org/conferences/DSC-2003/Proceedings/.

*Important.* Note, that also the titles of papers are in title style (as opposed to sentence style), i.e., they are capitalized. The first word after a colon ':' is always capitalized. Furthermore, commands like `\proglang` and `\pkg` should also be used for the references. The names of journals or proceeding volumes should not be abbreviated.

The easiest way to achieve this is to use BibTeX together with the style file `jss.bst`. To do so, the references just have to be included in a BibTeX file, `foo.bib` say, which has to be included at the end of the LaTeX document by `\bibliography{foo}`. Note, that to obtain references in the format above, the title field in your bib file, needs to be capitalized (contrary to the folklore, there are BibTeX styles that rely on this even for `@Article` entries), i.e. the entry `title = {Visualizing Categorical Data}` is correct, while entries like `title = {Visualizing categorical data}` or (even worse) `title = {{Visualizing categorical data}}` are not.

The default in `jss.cls` is to use the natbib package with options `authoryear`, `round` and `longnamesfirst`. If you cite any article with six or more authors, the citations with all names should be avoided. This can either be done by declaring `\shortcites{...}` for the particular references or by turning the `longnamesfirst` option off completely. The latter can be done by using the option `shortnames` when loading the `jss.cls` class

```latex
\documentclass[article,shortnames]{jss}
```

### What the class sets up

Summarized (not verbatim) from section 4 of the manual, which lists the source code of `jss.cls`:

- **Options:** `article` (the default), `codesnippet`, `bookreview` and `softwarereview` choose the kind of publication. `shortnames` loads natbib without `longnamesfirst`. `nojss` suppresses the JSS header and footer. `notitle` suppresses the automatic `\maketitle` at the beginning of the document, `noheadings` the headings on the pages, and `nofooter` the automatic `\makefooter` at the end of the document.
- **Base class and packages:** the `article` class with `11pt,a4paper,twoside`; graphicx, xcolor, fancyvrb, fontenc with `T1`, upquote and lmodern if available, natbib with `authoryear,round` (plus `longnamesfirst` unless `shortnames` is set) and `\bibliographystyle{jss}`, and hyperref.
- **Layout:** a text width of 6.125in and a text height of 46 lines. Paragraphs are not indented; instead the space between paragraphs is increased (`\parskip` of 0.7ex).
- **Title page and footer:** `\maketitle` runs automatically at the beginning of the document and prints the logo, title, authors, abstract and keywords (for reviews, the reviewer and the book or software details). `\makefooter` runs at the end and prints the address together with the publication information. Without `\Abstract` or `\Keywords` the placeholders "an abstract is required" and "at least one keyword is required" show up instead.
- **Headings:** sections are numbered "1.", centred, bold and large; subsections bold; subsubsections italic. Reviews use unnumbered, simpler headings.
- **Code:** the Sweave environments `Sinput`, `Soutput`, `Scode` and `Schunk`, and the neutral `Code`, `CodeInput`, `CodeOutput` and `CodeChunk`. Input is set in slanted type. Graphics default to a width of `0.8\textwidth` (`\setkeys{Gin}{width=0.8\textwidth}`).
- **Links:** coloured, dark blue for internal links and citations, dark red for URLs.
- **Commands:** `\proglang` is `\textsf`, `\pkg` is bold, and `\code` is typewriter without hyphenation. `\email{}` makes a mailto link, `\doi{}` a link to `https://doi.org/`, and `\DOI{}` sets the DOI in the header and footer. `\E`, `\VAR`, `\COV` and `\Prob` are sans-serif operators.
