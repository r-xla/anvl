# Conversions between R and Python via reticulate, and memory held by Python
# objects, as cited in the introduction. Run with: Rscript conversions.R
# (reticulate provisions numpy via uv; uses about 7 GB of memory at peak).
library(reticulate)
py_require("numpy")
cat("reticulate", format(packageVersion("reticulate")), "\n")
np <- import("numpy", convert = FALSE)

show <- function(label, x) {
  p <- r_to_py(x)
  cat(sprintf("%-24s -> %s\n", label, py_to_r(py_repr(p))))
}
cat("--- missing values ---\n")
show("NA_real_", NA_real_)
show("c(1, NA, NaN)", c(1, NA, NaN))
show("NA_integer_", NA_integer_)
show("c(1L, NA)", c(1L, NA))
show("c(TRUE, NA)", c(TRUE, NA))
show("array(c(TRUE, NA))", array(c(TRUE, NA)))
show("NA_character_", NA_character_)
back <- py_to_r(np_array(c(1, NA, NaN)))
cat("round trip c(1, NA, NaN): is.na =", is.na(back), "| is.nan =", is.nan(back), "\n")

cat("--- integers returned to R ---\n")
for (e in c("2**31", "2**40", "2**70")) cat(e, "->", format(py_to_r(py_eval(e, convert = FALSE))), "\n")

cat("--- reshaping ---\n")
print(py_to_r(np$reshape(r_to_py(1:6), tuple(2L, 3L))))
x <- 1:6; dim(x) <- c(2, 3); print(x)

cat("--- memory held by Python objects ---\n")
rss <- function() as.numeric(gsub("\\D", "", grep("^VmRSS", readLines("/proc/self/status"), value = TRUE))) / 1024
invisible(gc())
cat(sprintf("start: RSS %.0f MB\n", rss()))
for (i in 1:8) {
  a <- np$ones(100000000L) # 800 MB
  cat(sprintf("iteration %d: RSS %.0f MB, object.size(a) = %s bytes\n", i, rss(), object.size(a)))
}
invisible(gc())
cat(sprintf("after gc(): RSS %.0f MB\n", rss()))
