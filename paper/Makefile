# Builds the JSS article (main.pdf) and its replication script. Run from this directory.

FIGURES = figures/overview.pdf figures/hmc.pdf figures/runtime-single-chain.pdf \
	figures/runtime-strong-scaling.pdf figures/runtime-multiple-chains.pdf

all: main.pdf paper_code.R

# The article.
main.pdf: main.tex references.bib $(FIGURES)
	latexmk -pdf -interaction=nonstopmode main.tex

# The R code shown in main.tex as a standalone script (render it with knitr::spin("paper_code.R")).
paper_code.R: main.tex extract.R
	Rscript extract.R

# The figures: the overview is TikZ, the HMC draws and the runtime results are drawn with ggplot2.
figures/overview.pdf: figures/overview.tex figures/style.tex
	cd figures && pdflatex -interaction=nonstopmode overview.tex > /dev/null

figures/hmc.pdf: figures/hmc.R
	Rscript figures/hmc.R

# One run of figures/runtime.R writes all three runtime figures (`%` stands for the dot).
figures/runtime-single-chain%pdf figures/runtime-strong-scaling%pdf figures/runtime-multiple-chains%pdf: \
		figures/runtime.R benchmarks/plots.R
	Rscript figures/runtime.R

# The environment of the article: R and the packages of renv.lock on NVIDIA's CUDA image.
image:
	docker build -f envs/Dockerfile -t anvl-jss .

# A shell in that image, with this directory mounted (for the GPU, add --gpus all).
shell:
	docker run -it --rm -v "$(CURDIR)":/mnt/data/paper anvl-jss

clean:
	latexmk -c main.tex

.PHONY: all image shell clean
