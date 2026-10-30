devtools::load_all()

f <- function(x) {
  print(globals$default_traced_random_state)
  nv_exp(x)
  nv_runif(100L)
  print(globals$default_traced_random_state)
  nv_runif(100L)
}

jit(f)(1)
