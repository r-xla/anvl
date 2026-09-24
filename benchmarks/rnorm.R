devtools::load_all("~/r-xla/anvl")
library(pjrt)

f <- function() {
  nv_rnorm(nv_array(c(1, 2), dtype = "ui64"), dtype = "f32", shape = c(2, 3))
}


trace_f <- function() {
  trace_fn(f, list())
}
graph <- trace_f()

hlo_g <- function() {
  stablehlo(graph)[[1L]]
}

hlo <- hlo_g()

cmp_g <- function() {
  pjrt::pjrt_compile(pjrt::pjrt_program(stablehlo::repr(hlo)))
}

bench::mark(
  hlo_g(),
  trace_f(),
  cmp_g(),
  check = FALSE,
  memory = FALSE
)
