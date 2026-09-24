# Figures and tables of the runtime evaluation, used by paper.Rmd.
library(ggplot2)

read_results <- function(files) {
  files <- files[basename(files) %in% c("cpu.csv", "gpu.csv")]
  res <- do.call(rbind, lapply(files, utils::read.csv))
  res$config <- factor(
    paste(res$framework, res$label),
    levels = c("anvl cpu1", "anvl cpu16", "anvl gpu", "stan cpu1", "stan cpu16"),
    labels = c("anvl, 1 core", "anvl, 16 cores", "anvl, GPU", "Stan, 1 core", "Stan, 16 cores")
  )
  res
}

summarize_results <- function(res) {
  stats::aggregate(time ~ config + experiment + n + K, data = res, FUN = function(t) {
    c(median = stats::median(t), lo = min(t), hi = max(t))
  })
}

bench_theme <- function() {
  theme_bw(base_size = 9) + theme(legend.position = "right", legend.title = element_blank())
}

bench_colors <- c(
  "anvl, 1 core" = "#1b9e77", "anvl, 16 cores" = "#66a61e", "anvl, GPU" = "#7570b3",
  "Stan, 1 core" = "#d95f02", "Stan, 16 cores" = "#e6ab02"
)

plot_bench <- function(files, exp, x) {
  s <- summarize_results(read_results(files))
  s <- s[s$experiment == exp, ]
  s <- data.frame(s[c("config", "n", "K")], as.data.frame(s$time))
  ggplot(s, aes(.data[[x]], median, color = config)) +
    geom_line() +
    geom_point(size = 1) +
    scale_x_log10() +
    scale_y_log10() +
    scale_color_manual(values = bench_colors) +
    labs(x = if (x == "n") "Number of observations n" else "Number of chains K",
      y = "Time (seconds)") +
    bench_theme()
}

plot_single <- function(files) plot_bench(files, "single", "n")
plot_chains <- function(files) plot_bench(files, "chains", "K")

overhead_table <- function(files) {
  o <- utils::read.csv(files[basename(files) == "overhead.csv"])
  o$time <- format(signif(o$time, 2L), scientific = FALSE, drop0trailing = TRUE)
  names(o) <- c("Step", "Time (s)")
  knitr::kable(o, format = "latex", booktabs = TRUE, label = "overhead",
    caption = "One-off and per-call costs for $n = 10^4$ on 16 CPU cores, measured once. The sampler draws 1000 samples from one chain.")
}
