# Writes the runtime figures of main.tex (figures/runtime-*.pdf) from benchmarks/results, in the
# style of the other figures of the paper: Latin Modern, the font of the text, and the colors of
# benchmarks/plots.R, which follow figures/style.tex. The sizes are those of the figure chunks of
# paper.Rmd.
#
# Run from the paper directory: `Rscript figures/runtime.R` (or `make figures/runtime-single-chain.pdf`).
# Needs the Latin Modern Type 1 fonts of TeX Live, found via kpsewhich, and Ghostscript, which
# embeds them.
library(ggplot2)
source("benchmarks/plots.R")

kpse <- function(file) system2("kpsewhich", file, stdout = TRUE)
afm <- vapply(c("lmr10.afm", "lmbx10.afm", "lmri10.afm", "lmbxi10.afm"), kpse, character(1L))
grDevices::pdfFonts(LM = grDevices::Type1Font("LM", afm))
options(paper.plot_family = "LM")

save_pdf <- function(plot, file, width, height) {
  ggsave(file, plot, device = grDevices::pdf, width = width, height = height, family = "LM")
  grDevices::embedFonts(file, fontpaths = dirname(kpse("lmr10.pfb")))
}

res_files <- list.files("benchmarks/results", pattern = "\\.csv$", full.names = TRUE)

save_pdf(
  plot_single(res_files) + theme(legend.position = "bottom") +
    guides(color = guide_legend(ncol = 2), linetype = guide_legend(ncol = 2)),
  "figures/runtime-single-chain.pdf", width = 3.4, height = 2.75
)
save_pdf(
  plot_scaling(res_files) + theme(legend.position = "bottom") + guides(color = guide_legend(ncol = 2)),
  "figures/runtime-strong-scaling.pdf", width = 3.4, height = 2.75
)
save_pdf(plot_chains(res_files), "figures/runtime-multiple-chains.pdf", width = 6, height = 2.4)
