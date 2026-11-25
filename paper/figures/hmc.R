# Writes figures/hmc.pdf: the draws of chain 1 of the HMC sampler of the use case in main.tex
# (Section "Hamiltonian Monte Carlo"): its draws and an example leapfrog trajectory in the plane of
# (beta_1, beta_2) over contours of the posterior density, and the trace plots and posterior densities
# of both coefficients, in the style of the other figures of the paper: Latin Modern and shades of teal
# (figures/style.tex). The sampler and the data are those of main.tex, so the posterior means match
# the ones printed there.
#
# Run from the paper directory: `Rscript figures/hmc.R` (or `make figures/hmc.pdf`).
# Needs anvl 0.6.0, the Latin Modern Type 1 fonts of TeX Live, found via kpsewhich, and Ghostscript.
library(anvl)
library(ggplot2)
library(patchwork)

# The sampler of main.tex, unchanged.
potential <- function(beta, X, y) {
  eta <- X %*% t(beta)
  y <- nv_broadcast_to(y, shape(eta))
  loglik <- nv_sum(y * eta - nv_log1p(exp(eta)), axes = 1L)
  logprior <- -nv_sum(beta^2, axes = 2L) / 2
  -(loglik + logprior)
}
grad_potential <- gradient(
  function(beta, X, y) sum(potential(beta, X, y)),
  wrt = "beta"
)
leapfrog <- function(beta, p, X, y, eps, L) {
  init <- list(beta = beta, p = p, grad = grad_potential(beta, X, y)$beta)
  nv_scan(init, steps = L, body = function(s, x) {
    p <- s$p - eps / 2 * s$grad
    beta <- s$beta + eps * p
    grad <- grad_potential(beta, X, y)$beta
    p <- p - eps / 2 * grad
    list(carry = list(beta = beta, p = p, grad = grad), out = NULL)
  })$carry
}
hmc_step <- function(beta, X, y, eps, L) {
  p <- nv_rnorm(shape(beta))
  prop <- leapfrog(beta, p, X, y, eps, L)
  H0 <- potential(beta, X, y) + nv_sum(p^2, axes = 2L) / 2
  H1 <- potential(prop$beta, X, y) + nv_sum(prop$p^2, axes = 2L) / 2
  u <- nv_runif(shape(beta)[1L])
  accept <- log(u) < H0 - H1
  accept <- nv_broadcast_to(accept, shape(beta))
  nv_ifelse(accept, prop$beta, beta)
}
hmc_step_jit <- jit(hmc_step, static = "L")
hmc <- function(beta, X, y, eps, L, n_draws) {
  draws <- vector("list", n_draws)
  for (i in seq_len(n_draws)) {
    beta <- hmc_step_jit(beta, X, y, eps, L)
    draws[[i]] <- beta
  }
  simplify2array(lapply(draws, as_array))
}

# The random draws of main.tex up to the use case: the seed right after loading anvl and the draw of
# three normal variates in the section on arrays, at the default float of that point (f32).
nv_set_seed(1L)
invisible(nv_rnorm(3L))
options(anvl.default_dtypes = c(float = "f64", int = "i32"))
n <- 1000; p <- 2
X <- nv_rnorm(c(n, p))
eta <- X %*% nv_rnorm(c(p, 1L), sd = 0.5)
y <- nv_ifelse(nv_runif(c(n, 1L)) < nv_plogis(eta), 1, 0)
Xr <- as_array(X); yr <- as.vector(as_array(y))
K <- 4L; L <- 20L; eps <- 0.5 / sqrt(n); n_draws <- 2000L
beta0 <- nv_array(matrix(0, K, p))
draws <- hmc(beta0, X, y, eps = eps, L = L, n_draws = n_draws) # chain x coefficient x draw
mle <- unname(coef(glm(yr ~ Xr - 1, family = binomial)))

# The posterior means printed in main.tex.
stopifnot(isTRUE(all.equal(
  unname(apply(draws[, , 1001:2000], 2, mean)), c(0.2453013, 0.1593219), tolerance = 1e-6
)))

# An example leapfrog trajectory: the same integrator, returning the position after every step,
# started at a draw of chain 1 from the second half with a random momentum.
leapfrog_path <- jit(function(beta, p, X, y, eps, L) {
  init <- list(beta = beta, p = p, grad = grad_potential(beta, X, y)$beta)
  nv_scan(init, steps = L, body = function(s, x) {
    p <- s$p - eps / 2 * s$grad
    beta <- s$beta + eps * p
    grad <- grad_potential(beta, X, y)$beta
    p <- p - eps / 2 * grad
    list(carry = list(beta = beta, p = p, grad = grad), out = beta)
  })$out
}, static = "L")
start <- draws[1L, , 1500L]
nv_set_seed(3L)
momentum <- nv_rnorm(c(K, p))
path <- as_array(leapfrog_path(nv_array(matrix(start, K, p, byrow = TRUE)), momentum, X, y, eps, L))
traj <- data.frame(b1 = c(start[1L], path[, 1L, 1L]), b2 = c(start[2L], path[, 1L, 2L]))

warmup <- n_draws / 2
chain_df <- function(idx) {
  do.call(rbind, lapply(seq_len(K), function(k) data.frame(
    chain = factor(k, levels = seq_len(K)), draw = idx,
    b1 = draws[k, 1L, idx], b2 = draws[k, 2L, idx]
  )))
}
early <- do.call(rbind, lapply(seq_len(K), function(k) data.frame(
  chain = factor(k, levels = seq_len(K)), b1 = c(0, draws[k, 1L, 1:8]), b2 = c(0, draws[k, 2L, 1:8])
)))
late <- chain_df((warmup + 1):n_draws)
long <- rbind(
  transform(chain_df(seq_len(n_draws))[, c("chain", "draw")], coef = "beta[1]", value = as.vector(t(draws[, 1L, ]))),
  transform(chain_df(seq_len(n_draws))[, c("chain", "draw")], coef = "beta[2]", value = as.vector(t(draws[, 2L, ])))
)
mle_df <- data.frame(coef = c("beta[1]", "beta[2]"), value = mle)
# The figure shows chain 1; the other chains look alike.
late <- late[late$chain == "1", ]
long <- long[long$chain == "1", ]

# Contours of the posterior density at the levels of 1, 2, and 3 standard deviations of a normal
# approximation, on a grid around the posterior mode.
log_post <- function(b1, b2) {
  eta <- Xr %*% rbind(b1, b2)
  colSums(yr * eta - log1p(exp(eta))) - (b1^2 + b2^2) / 2
}
mode <- optim(mle, function(b) -log_post(b[1L], b[2L]), method = "BFGS")$par
grid <- expand.grid(b1 = mode[1L] + seq(-0.45, 0.45, length.out = 151),
  b2 = mode[2L] + seq(-0.45, 0.45, length.out = 151))
grid$lp <- log_post(grid$b1, grid$b2)
grid$lp <- grid$lp - max(grid$lp)

# Fonts and colors of the other figures (figures/runtime.R, benchmarks/plots.R).
kpse <- function(file) system2("kpsewhich", file, stdout = TRUE)
afm <- vapply(c("lmr10.afm", "lmbx10.afm", "lmri10.afm", "lmbxi10.afm"), kpse, character(1L))
grDevices::pdfFonts(LM = grDevices::Type1Font("LM", afm))
accent <- "#21908c"

hmc_theme <- theme_bw(base_size = 9, base_family = "LM") +
  theme(strip.background = element_blank(), strip.text = element_text(size = 9),
    panel.grid.minor = element_blank(), legend.position = "bottom",
    legend.key.width = unit(14, "pt"), legend.margin = margin(0, 0, 0, 0))

plane <- ggplot() +
  geom_point(aes(b1, b2), data = late, size = 0.2, alpha = 0.4, color = accent) +
  geom_contour(aes(b1, b2, z = lp), data = grid, breaks = c(-4.5, -2, -0.5), color = "black",
    linewidth = 0.35) +
  geom_path(aes(b1, b2), data = traj, linetype = "dashed", linewidth = 0.45) +
  geom_point(aes(b1, b2), data = traj, size = 0.6) +
  annotate("point", x = start[1L], y = start[2L], shape = 21, size = 2, fill = "white", stroke = 0.6) +
  annotate("point", x = mle[1L], y = mle[2L], shape = 4, size = 1.8, stroke = 0.6) +
  coord_fixed(xlim = mode[1L] + c(-0.38, 0.38), ylim = mode[2L] + c(-0.38, 0.38)) +
  labs(x = expression(beta[1]), y = expression(beta[2])) +
  hmc_theme

# The trace plots show the range of the draws after the first 20 transitions; the plane shows the start.
zoom <- long[long$draw > 20L, ]
limits <- lapply(split(zoom$value, zoom$coef), function(v) range(v) + c(-0.15, 0.15) * diff(range(v)))
long$value_zoom <- pmin(pmax(long$value, unlist(lapply(long$coef, function(k) limits[[k]][1L]))),
  unlist(lapply(long$coef, function(k) limits[[k]][2L])))
trace <- ggplot(long, aes(draw, value_zoom)) +
  annotate("rect", xmin = 0, xmax = warmup, ymin = -Inf, ymax = Inf, fill = "grey92") +
  geom_line(linewidth = 0.15, color = accent) +
  geom_hline(aes(yintercept = value), data = mle_df, linetype = "dashed", linewidth = 0.3) +
  facet_wrap(~coef, ncol = 1L, scales = "free_y", labeller = label_parsed, strip.position = "left") +
  scale_x_continuous(breaks = c(0, 1000, 2000)) +
  labs(x = "Draw", y = NULL) +
  hmc_theme +
  theme(strip.placement = "outside")

density <- ggplot(long[long$draw > warmup, ], aes(value)) +
  geom_density(linewidth = 0.45, color = accent) +
  geom_vline(aes(xintercept = value), data = mle_df, linetype = "dashed", linewidth = 0.3) +
  facet_wrap(~coef, ncol = 1L, scales = "free", labeller = label_parsed) +
  scale_x_continuous(breaks = scales::breaks_pretty(n = 3)) +
  labs(x = "Value", y = "Posterior density") +
  hmc_theme +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), strip.text = element_blank())

plot <- (plane | trace | density) + plot_layout(widths = c(1.25, 1.35, 0.8))

file <- "figures/hmc.pdf"
ggsave(file, plot, device = grDevices::pdf, width = 6.1, height = 2.6, family = "LM")
grDevices::embedFonts(file, fontpaths = dirname(kpse("lmr10.pfb")))
