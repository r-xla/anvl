# Figures and tables of the runtime evaluation, used by paper.Rmd.
library(ggplot2)

read_results <- function(files) {
  files <- files[basename(files) %in% c("cpu.csv", "gpu.csv")]
  res <- do.call(rbind, lapply(files, utils::read.csv))
  res$config <- factor(
    paste(res$framework, res$label),
    levels = c("anvl cpu1", "anvl cpu16", "anvl cpu16prog", "anvl gpu", "stan cpu1", "stan cpu16"),
    labels = c("anvl, 1 core", "anvl, 16 cores", "anvl, 16 cores (one program)", "anvl, GPU",
      "Stan, 1 core", "Stan, 16 cores")
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
  "anvl, 1 core" = "#1b9e77", "anvl, 16 cores" = "#66a61e", "anvl, 16 cores (one program)" = "#66a61e",
  "anvl, GPU" = "#7570b3",
  "Stan, 1 core" = "#d95f02", "Stan, 16 cores" = "#e6ab02"
)

bench_linetypes <- c(
  "anvl, 1 core" = "solid", "anvl, 16 cores" = "solid", "anvl, 16 cores (one program)" = "dashed",
  "anvl, GPU" = "solid", "Stan, 1 core" = "solid", "Stan, 16 cores" = "solid"
)

plot_bench <- function(files, exp, x) {
  s <- summarize_results(read_results(files))
  s <- s[s$experiment == exp, ]
  s <- droplevels(data.frame(s[c("config", "n", "K")], as.data.frame(s$time)))
  ggplot(s, aes(.data[[x]], median, color = config, linetype = config)) +
    geom_line() +
    geom_point(size = 1) +
    scale_x_log10() +
    scale_y_log10() +
    scale_color_manual(values = bench_colors) +
    scale_linetype_manual(values = bench_linetypes) +
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

plot_scaling <- function(files) {
  res <- utils::read.csv(files[basename(files) == "scaling.csv"])
  res <- res[res$experiment == "scaling", ]
  res$cores <- as.integer(sub(".*-", "", res$label))
  res$method <- sub("-.*", "", res$label)
  s <- stats::aggregate(time ~ method + cores, data = res, FUN = stats::median)
  s$method <- factor(s$method, levels = c("anvl", "stan_threads", "stan"),
    labels = c("anvl", "Stan (reduce_sum)", "Stan"))
  p <- ggplot(s, aes(cores, time, color = method)) +
    geom_line(data = s[s$method != "Stan", ]) +
    geom_point(size = 1.5) +
    scale_x_continuous(trans = "log2", breaks = c(1, 2, 4, 8, 16)) +
    scale_y_log10() +
    scale_color_manual(values = c("anvl" = "#1b9e77", "Stan (reduce_sum)" = "#e6ab02", "Stan" = "#d95f02",
      "anvl, GPU" = "#7570b3")) +
    labs(x = "Number of CPU cores", y = "Time (seconds)") +
    bench_theme()
  gpu_file <- files[basename(files) == "scaling-gpu.csv"]
  if (length(gpu_file)) {
    g <- stats::median(utils::read.csv(gpu_file)$time)
    p <- p + geom_hline(aes(yintercept = g, color = "anvl, GPU"), data = data.frame(g = g), linetype = "dashed")
  }
  p
}
