# Render the paper: knit with rjtools (writes paper.tex and RJwrapper.tex), then run LaTeX.
pdf: figures/overview.pdf
	-Rscript -e 'rmarkdown::render("paper.Rmd", output_format = rjtools::rjournal_pdf_article(toc = FALSE))'
	./build-pdf.sh

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
