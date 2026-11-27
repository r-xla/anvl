f <- function() {
  nv_rnorm(dtype = "f64", shape = c(200L, 300L, 400L))
}
g <- jit(f)
out <- g()
Z <- as_array(out)

# Open PNG device first
png("inst/random/qq_rnorm.png", width = 800, height = 800)

# Create plots
set.seed(1)
par(mfrow = c(2, 2))
for (i in sort(sample(200, 4))) {
  qqnorm(Z[i, , ], col = "grey", main = sprintf("Q-Q Plot of Z[%d, , ]", i))
  qqline(Z[i, , ], lwd = 2)
}

# Close device to save
dev.off()
