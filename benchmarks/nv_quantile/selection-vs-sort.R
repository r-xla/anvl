## ---------------------------------------------------------------------------
## nv_quantile(): top_k selection window vs a full sort
##
## When every requested prob lies in one half of the axis, nv_quantile() gathers
## from a top_k window instead of sorting the whole axis. This compares the two
## on the same build: a scalar prob takes the selection path, and adding the
## sentinel probs 0.1 and 0.9 (one at each end) forces the sort path at the cost
## of two extra gathers, which is negligible next to the sort.
##
## Shapes mimic a stack of t rasters reduced over time, once with time leading
## and once with time trailing.
## ---------------------------------------------------------------------------

library(anvl)

# Rscript selection-vs-sort.R [cpu|cuda]
device <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1] else "cpu"

time_one <- function(f, x, reps = 5L) {
  f(x)
  f(x)
  t <- vapply(
    seq_len(reps),
    function(i) {
      s <- proc.time()[["elapsed"]]
      r <- f(x)
      pjrt::await(r$data)
      proc.time()[["elapsed"]] - s
    },
    numeric(1L)
  )
  median(t)
}

rows <- list()
for (t in c(31L, 55L, 90L)) {
  for (axis in c(1L, 3L)) {
    shp <- if (axis == 1L) c(t, 512L, 512L) else c(512L, 512L, t)
    a <- array(rnorm(prod(shp)), shp)
    a[sample(length(a), length(a) %/% 5L)] <- NaN
    x <- nv_array(a, dtype = "f32", device = device)
    for (q in c(0.25, 0.5, 0.9)) {
      selection <- jit(function(x) nv_quantile(x, q, axis = axis, nan_rm = TRUE))
      sort <- jit(function(x) nv_quantile(x, array(c(0.1, 0.9, q)), axis = axis, nan_rm = TRUE))
      t_sel <- time_one(selection, x)
      t_sort <- time_one(sort, x)
      stopifnot(identical(as_array(selection(x)), as_array(sort(x))[3L, , ]))
      rows[[length(rows) + 1L]] <- data.frame(
        t = t,
        axis = axis,
        prob = q,
        sort_s = round(t_sort, 3),
        selection_s = round(t_sel, 3),
        speedup = round(t_sort / t_sel, 2)
      )
    }
    rm(x, a)
    invisible(gc())
  }
}
print(do.call(rbind, rows), row.names = FALSE)
