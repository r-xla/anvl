# Shared setup for the usage examples: simulated logistic-regression data and reference values.
#
# U(beta) = -sum_i (y_i x_i' beta - log(1 + exp(x_i' beta))) + ||beta||^2 / 2
# grad U(beta) = -X' (y - plogis(X beta)) + beta

set.seed(1)
n <- 200L
p <- 3L
X <- matrix(rnorm(n * p), n, p)
y <- rbinom(n, 1L, plogis(X %*% c(1, -1, 0.5)))
beta0 <- c(0.1, -0.2, 0.3)

# Reference values, computed by hand
U_ref <- function(beta) {
  eta <- drop(X %*% beta)
  -sum(y * eta - log1p(exp(eta))) + 0.5 * sum(beta^2)
}
grad_ref <- function(beta) drop(-crossprod(X, y - plogis(X %*% beta))) + beta

cat("reference: U(beta0) =", format(U_ref(beta0)), "; grad =", format(grad_ref(beta0)), "\n")
