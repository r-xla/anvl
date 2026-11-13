# Render the paper: knit with rjtools (writes paper.tex and RJwrapper.tex), then run LaTeX.
pdf: figures/overview.pdf
	-Rscript -e 'rmarkdown::render("paper.Rmd", output_format = rjtools::rjournal_pdf_article(toc = FALSE), clean = FALSE)'
	./build-pdf.sh

# JSS version: plain LaTeX in main.tex (figures in figures/, code outputs pasted in).
JSS_FIGURES = figures/overview.pdf figures/runtime-single-chain.pdf \
	figures/runtime-strong-scaling.pdf figures/runtime-multiple-chains.pdf

jss: main.pdf

main.pdf: main.tex references.bib jss.cls jss.bst $(JSS_FIGURES)
	pdflatex -interaction=nonstopmode main.tex > /dev/null || true
	bibtex main > /dev/null || true
	pdflatex -interaction=nonstopmode main.tex > /dev/null || true
	pdflatex -interaction=nonstopmode main.tex > /dev/null || true
	@grep -E '^!|Warning: (Citation|Reference)|undefined|Overfull' main.log | sort -u || true

# Replication script: the R code shown in main.tex, extracted to paper_code.R
# (render it with knitr::spin("paper_code.R")).
code: paper_code.R

paper_code.R: main.tex extract.R
	Rscript extract.R

.PHONY: jss code

# TikZ figures are standalone LaTeX documents.
figures/%.pdf: figures/%.tex
	cd figures && pdflatex -interaction=nonstopmode $*.tex > /dev/null

# Build the Docker image in which all results were produced.
image:
	docker build -t anvl-paper docker

# Re-run the runtime benchmark locally (CPU part), inside the image.
bench-cpu:
	docker run --rm -v "$(CURDIR):/paper" anvl-paper bash /paper/benchmarks/lrz/run-cpu.sh

bench-gpu:
	docker run --rm --gpus all -v "$(CURDIR):/paper" anvl-paper bash /paper/benchmarks/lrz/run-gpu.sh
