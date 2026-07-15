# Exact workload of the faq.Rmd vignette chunk, which segfaults
# stochastically on CI during "checking re-building of vignette outputs".
library(anvl)

mul_n <- function(x, n) {
  for (i in 1:n) {
    x <- x * x
  }
  return(x)
}

x <- nv_array(rnorm(1e8))
await(x)

print(system.time(mul_n(x, 20)))
print(system.time(await(mul_n(x, 20))))
cat("iteration survived\n")
