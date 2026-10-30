# Figures of the articles

The articles include the PNGs here, which are built from the TikZ sources next
to them and committed, so that building the articles needs no LaTeX. After
changing a source, rebuild its PNG from this directory, e.g. `rng-key-tree.png`
with

```sh
pdflatex rng-key-tree.tex && pdftocairo -png -r 300 -singlefile rng-key-tree.pdf rng-key-tree
```

`rng-f32-cost.tex` reads curves that `Rscript rng-f32-data.R` writes; run it
first. The numbers in the `rng-*` figures come from the analysis in
`rng-analysis/` of the [anvl-bench](https://github.com/r-xla/anvl-bench)
repository.
